import { EnvironmentVectorDBProvider } from '../../index.js';
import type {
	EmbeddingModelProvider,
	FileSystem,
	RetrievalProvider,
	RetrievalRegistry,
} from '../../index.js';

export class Document {
	pageContent: string;
	metadata: Record<string, unknown>;
	id?: string;

	constructor(options: {
		pageContent: string;
		metadata?: Record<string, unknown>;
		id?: string;
	}) {
		this.pageContent = options.pageContent;
		this.metadata = { ...(options.metadata ?? {}) };
		this.id = options.id;
	}
}

export class TextLoader {
	constructor(
		readonly path: string,
		readonly fileSystem: FileSystem,
	) {}

	async load(): Promise<Document[]> {
		return [
			new Document({
				pageContent: new TextDecoder().decode(
					this.fileSystem.read(this.path),
				),
				metadata: { source: this.path },
				id: this.path,
			}),
		];
	}
}

export class JSONLoader {
	constructor(
		readonly path: string,
		readonly fileSystem: FileSystem,
		readonly options: { contentKey?: string } = {},
	) {}

	async load(): Promise<Document[]> {
		const value = JSON.parse(
			new TextDecoder().decode(this.fileSystem.read(this.path)),
		) as unknown;
		const items = Array.isArray(value) ? value : [value];
		return items.map((item, index) => {
			const content =
				this.options.contentKey &&
				item &&
				typeof item === 'object' &&
				!Array.isArray(item)
					? (item as Record<string, unknown>)[this.options.contentKey]
					: item;
			return new Document({
				pageContent:
					typeof content === 'string'
						? content
						: JSON.stringify(content),
				metadata: { source: this.path, index },
				id: `${this.path}:${index}`,
			});
		});
	}
}

function parseCsvLine(line: string): string[] {
	const values: string[] = [];
	let current = '';
	let quoted = false;
	for (let index = 0; index < line.length; index += 1) {
		const character = line[index] ?? '';
		if (character === '"') {
			if (quoted && line[index + 1] === '"') {
				current += '"';
				index += 1;
			} else {
				quoted = !quoted;
			}
		} else if (character === ',' && !quoted) {
			values.push(current);
			current = '';
		} else {
			current += character;
		}
	}
	values.push(current);
	return values;
}

export class CSVLoader {
	constructor(
		readonly path: string,
		readonly fileSystem: FileSystem,
	) {}

	async load(): Promise<Document[]> {
		const lines = new TextDecoder()
			.decode(this.fileSystem.read(this.path))
			.split(/\r?\n/)
			.filter((line) => line.length > 0);
		if (!lines.length) return [];
		const headers = parseCsvLine(lines[0] ?? '');
		return lines.slice(1).map((line, row) => {
			const values = parseCsvLine(line);
			return new Document({
				pageContent: headers
					.map((header, index) => `${header}: ${values[index] ?? ''}`)
					.join('\n'),
				metadata: { source: this.path, row },
				id: `${this.path}:${row}`,
			});
		});
	}
}

export class DirectoryLoader {
	constructor(
		readonly path: string,
		readonly fileSystem: FileSystem,
		readonly options: { recursive?: boolean; extensions?: string[] } = {},
	) {}

	async load(): Promise<Document[]> {
		const paths = this.options.recursive
			? this.fileSystem.glob(this.path ? `${this.path}/**/*` : '**/*')
			: this.fileSystem
					.list(this.path)
					.filter((entry) => !entry.isDirectory)
					.map((entry) => entry.path);
		const extensions = this.options.extensions?.map((value) =>
			value.startsWith('.')
				? value.toLowerCase()
				: `.${value.toLowerCase()}`,
		);
		const docs: Document[] = [];
		for (const path of paths) {
			if (
				extensions?.length &&
				!extensions.some((extension) =>
					path.toLowerCase().endsWith(extension),
				)
			) {
				continue;
			}
			docs.push(...(await new TextLoader(path, this.fileSystem).load()));
		}
		return docs;
	}
}

export class RecursiveCharacterTextSplitter {
	readonly chunkSize: number;
	readonly chunkOverlap: number;

	constructor(options: { chunkSize?: number; chunkOverlap?: number } = {}) {
		this.chunkSize = options.chunkSize ?? 1000;
		this.chunkOverlap = options.chunkOverlap ?? 200;
		if (this.chunkSize < 1) throw new Error('chunkSize must be at least 1');
		if (this.chunkOverlap < 0 || this.chunkOverlap >= this.chunkSize) {
			throw new Error(
				'chunkOverlap must be non-negative and smaller than chunkSize',
			);
		}
	}

	async splitDocuments(documents: Document[]): Promise<Document[]> {
		const output: Document[] = [];
		const step = this.chunkSize - this.chunkOverlap;
		for (const document of documents) {
			const text = document.pageContent;
			for (
				let start = 0, index = 0;
				start < text.length;
				start += step, index += 1
			) {
				const chunk = text.slice(start, start + this.chunkSize);
				if (!chunk) break;
				output.push(
					new Document({
						pageContent: chunk,
						metadata: { ...document.metadata, chunk: index },
						id: document.id ? `${document.id}:${index}` : undefined,
					}),
				);
				if (start + this.chunkSize >= text.length) break;
			}
		}
		return output;
	}

	async createDocuments(texts: string[]): Promise<Document[]> {
		return await this.splitDocuments(
			texts.map((pageContent) => new Document({ pageContent })),
		);
	}
}

export interface Embeddings {
	embedDocuments(texts: string[]): Promise<number[][]>;
	embedQuery(text: string): Promise<number[]>;
}

export class AgentRTEmbeddings implements Embeddings {
	constructor(
		readonly provider: EmbeddingModelProvider,
		readonly model?: string,
	) {}

	async embedDocuments(texts: string[]): Promise<number[][]> {
		const response = await this.provider.embed({
			input: texts,
			model: this.model,
		});
		return [...response.data]
			.sort((left, right) => left.index - right.index)
			.map((item) => {
				if (!Array.isArray(item.embedding)) {
					throw new Error(
						'LangChain compatibility requires numeric embeddings',
					);
				}
				return [...item.embedding];
			});
	}

	async embedQuery(text: string): Promise<number[]> {
		const [vector] = await this.embedDocuments([text]);
		if (!vector) throw new Error('embedding provider returned no vector');
		return vector;
	}
}

function cosine(left: number[], right: number[]): number {
	if (!left.length || left.length !== right.length) return 0;
	let dot = 0;
	let l = 0;
	let r = 0;
	for (const [index, value] of left.entries()) {
		const [other = 0] = right.slice(index, index + 1);
		dot += value * other;
		l += value * value;
		r += other * other;
	}
	return l && r ? dot / (Math.sqrt(l) * Math.sqrt(r)) : 0;
}

type VectorRecord = {
	document: Document;
	embedding: number[];
};

export class MemoryVectorStore {
	private readonly records: VectorRecord[] = [];

	constructor(readonly embeddings: Embeddings) {}

	static async fromDocuments(
		documents: Document[],
		embeddings: Embeddings,
	): Promise<MemoryVectorStore> {
		const store = new MemoryVectorStore(embeddings);
		await store.addDocuments(documents);
		return store;
	}

	async addDocuments(documents: Document[]): Promise<string[]> {
		const vectors = await this.embeddings.embedDocuments(
			documents.map((document) => document.pageContent),
		);
		const ids: string[] = [];
		for (const [index, document] of documents.entries()) {
			const [vector] = vectors.slice(index, index + 1);
			if (!vector) continue;
			this.records.push({ document, embedding: vector });
			ids.push(document.id ?? `doc-${this.records.length}`);
		}
		return ids;
	}

	async similaritySearch(query: string, k = 4): Promise<Document[]> {
		return (await this.similaritySearchWithScore(query, k)).map(
			([document]) => document,
		);
	}

	async similaritySearchWithScore(
		query: string,
		k = 4,
	): Promise<Array<[Document, number]>> {
		const vector = await this.embeddings.embedQuery(query);
		return this.records
			.map(
				(record) =>
					[record.document, cosine(vector, record.embedding)] as [
						Document,
						number,
					],
			)
			.sort((left, right) => right[1] - left[1])
			.slice(0, k);
	}

	asRetriever(options: { k?: number } = {}): {
		invoke(query: string): Promise<Document[]>;
		getRelevantDocuments(query: string): Promise<Document[]>;
	} {
		const k = options.k ?? 4;
		return {
			invoke: async (query) => await this.similaritySearch(query, k),
			getRelevantDocuments: async (query) =>
				await this.similaritySearch(query, k),
		};
	}
}

export type VectorDBCompatibilityOptions = {
	collectionName?: string;
	indexName?: string;
	url?: string;
	apiKey?: string;
	namespace?: string;
	tenant?: string;
	database?: string;
	contentField?: string;
	titleField?: string;
	uriField?: string;
	filter?: string;
	environment?: Readonly<Record<string, string | undefined>>;
	fetchImpl?: (
		input: string | URL | Request,
		init?: RequestInit,
	) => Promise<Response>;
};

function currentVectorEnvironment(): Readonly<
	Record<string, string | undefined>
> {
	const runtime = globalThis as typeof globalThis & {
		process?: { env?: Record<string, string | undefined> };
	};
	return runtime.process?.env ?? {};
}

export class AgentRTVectorStore {
	readonly provider: EnvironmentVectorDBProvider;
	readonly embeddings?: Embeddings;

	constructor(
		backend: string,
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		this.embeddings = embeddings;
		const environment: Record<string, string | undefined> = {
			...(options.environment ?? currentVectorEnvironment()),
			AGENT_RT_VECTOR_DB: backend,
		};
		const collection = options.collectionName ?? options.indexName;
		if (collection) environment.AGENT_RT_VECTOR_DB_COLLECTION = collection;
		if (options.url) environment.AGENT_RT_VECTOR_DB_URL = options.url;
		if (options.apiKey)
			environment.AGENT_RT_VECTOR_DB_API_KEY = options.apiKey;
		const backendOptions: Array<[string, string | undefined]> = [
			['NAMESPACE', options.namespace],
			['TENANT', options.tenant],
			['DATABASE', options.database],
			['CONTENT_FIELD', options.contentField],
			['TITLE_FIELD', options.titleField],
			['URI_FIELD', options.uriField],
			['FILTER', options.filter],
		];
		for (const [key, value] of backendOptions) {
			if (value !== undefined) {
				environment[`AGENT_RT_VECTOR_DB_OPTION_${key}`] = value;
			}
		}
		this.provider = EnvironmentVectorDBProvider.fromEnvironment(
			environment,
			{
				fetchImpl: options.fetchImpl,
			},
		);
	}

	async similaritySearchWithScore(
		query: string,
		k = 4,
		filter?: Record<string, unknown>,
	): Promise<Array<[Document, number | undefined]>> {
		const filters = { ...(filter ?? {}) };
		if (this.embeddings)
			filters.vector = await this.embeddings.embedQuery(query);
		const results = await this.provider.search({
			text: query,
			limit: k,
			filters,
		});
		return results.map((result) => [
			new Document({
				id: result.id,
				pageContent:
					typeof result.content === 'string'
						? result.content
						: JSON.stringify(result.content),
				metadata: {
					...(result.metadata ?? {}),
					title: result.title,
					...(result.uri ? { source: result.uri } : {}),
					...(result.score !== undefined
						? { score: result.score }
						: {}),
				},
			}),
			result.score,
		]);
	}

	async similaritySearch(
		query: string,
		k = 4,
		filter?: Record<string, unknown>,
	): Promise<Document[]> {
		return (await this.similaritySearchWithScore(query, k, filter)).map(
			([document]) => document,
		);
	}

	asRetriever(
		options: { k?: number; filter?: Record<string, unknown> } = {},
	): {
		invoke(query: string): Promise<Document[]>;
		getRelevantDocuments(query: string): Promise<Document[]>;
	} {
		const k = options.k ?? 4;
		return {
			invoke: async (query) =>
				await this.similaritySearch(query, k, options.filter),
			getRelevantDocuments: async (query) =>
				await this.similaritySearch(query, k, options.filter),
		};
	}
}

export class Chroma extends AgentRTVectorStore {
	constructor(
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		super('chroma', embeddings, options);
	}
}

export class PineconeStore extends AgentRTVectorStore {
	constructor(
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		super('pinecone', embeddings, options);
	}
}

export const PineconeVectorStore = PineconeStore;

export class QdrantVectorStore extends AgentRTVectorStore {
	constructor(
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		super('qdrant', embeddings, options);
	}
}

export class Milvus extends AgentRTVectorStore {
	constructor(
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		super('milvus', embeddings, options);
	}
}

export class WeaviateStore extends AgentRTVectorStore {
	constructor(
		embeddings?: Embeddings,
		options: VectorDBCompatibilityOptions = {},
	) {
		super('weaviate', embeddings, options);
	}
}

export const WeaviateVectorStore = WeaviateStore;

export class AgentRTRetriever {
	constructor(
		readonly source:
			| RetrievalProvider
			| { registry: RetrievalRegistry; providerName: string },
		readonly k = 4,
		readonly filters?: Record<string, unknown>,
	) {}

	async invoke(query: string): Promise<Document[]> {
		const results =
			'search' in this.source
				? await this.source.search({
						text: query,
						limit: this.k,
						filters: this.filters,
					})
				: await this.source.registry.search(this.source.providerName, {
						text: query,
						limit: this.k,
						filters: this.filters,
					});
		return results.map(
			(result) =>
				new Document({
					id: result.id,
					pageContent:
						typeof result.content === 'string'
							? result.content
							: JSON.stringify(result.content),
					metadata: {
						...(result.metadata ?? {}),
						title: result.title,
						...(result.uri ? { source: result.uri } : {}),
						...(result.score !== undefined
							? { score: result.score }
							: {}),
					},
				}),
		);
	}

	async getRelevantDocuments(query: string): Promise<Document[]> {
		return await this.invoke(query);
	}
}

export class ExternalVectorStoreRetriever {
	constructor(
		readonly vectorStore: {
			similaritySearch?: (
				query: string,
				k?: number,
				filter?: Record<string, unknown>,
			) => Promise<Document[]> | Document[];
		},
		readonly options: { k?: number; filter?: Record<string, unknown> } = {},
	) {}

	async invoke(query: string): Promise<Document[]> {
		if (typeof this.vectorStore.similaritySearch !== 'function') {
			throw new Error('vectorStore must expose similaritySearch()');
		}
		return await this.vectorStore.similaritySearch(
			query,
			this.options.k ?? 4,
			this.options.filter,
		);
	}

	async getRelevantDocuments(query: string): Promise<Document[]> {
		return await this.invoke(query);
	}
}

type RetrieverLike = {
	invoke(query: string): Promise<Document[]>;
};

function documentKey(document: Document): string {
	if (document.id) return `id:${document.id}`;
	return JSON.stringify([document.pageContent, document.metadata]);
}

export class EnsembleRetriever {
	readonly retrievers: RetrieverLike[];
	readonly weights: number[];
	readonly c: number;
	readonly k?: number;

	constructor(options: {
		retrievers: RetrieverLike[];
		weights?: number[];
		c?: number;
		k?: number;
	}) {
		if (!options.retrievers.length)
			throw new Error('retrievers must not be empty');
		this.retrievers = options.retrievers;
		this.weights = options.weights ?? options.retrievers.map(() => 1);
		this.c = options.c ?? 60;
		this.k = options.k;
		if (this.weights.length !== this.retrievers.length)
			throw new Error('weights must match retrievers');
		if (this.c < 1) throw new Error('c must be at least 1');
		if (
			this.weights.some((weight) => weight < 0) ||
			!this.weights.some(Boolean)
		)
			throw new Error('weights must contain at least one positive value');
	}

	async invoke(query: string): Promise<Document[]> {
		const scores = new Map<string, number>();
		const documents = new Map<string, Document>();
		for (const [index, retriever] of this.retrievers.entries()) {
			const ranked = await retriever.invoke(query);
			const weight = this.weights[index] ?? 0;
			for (const [rankIndex, document] of ranked.entries()) {
				const key = documentKey(document);
				documents.set(key, documents.get(key) ?? document);
				scores.set(
					key,
					(scores.get(key) ?? 0) + weight / (this.c + rankIndex + 1),
				);
			}
		}
		const keys = [...scores.keys()].sort(
			(left, right) => (scores.get(right) ?? 0) - (scores.get(left) ?? 0),
		);
		return (this.k === undefined ? keys : keys.slice(0, this.k)).map(
			(key) => documents.get(key) as Document,
		);
	}

	async getRelevantDocuments(query: string): Promise<Document[]> {
		return await this.invoke(query);
	}
}

export class ContextualCompressionRetriever {
	constructor(
		readonly baseRetriever: RetrieverLike,
		readonly baseCompressor: {
			compressDocuments(
				documents: Document[],
				query: string,
			): Document[] | Promise<Document[]>;
		},
	) {}

	async invoke(query: string): Promise<Document[]> {
		return await this.baseCompressor.compressDocuments(
			await this.baseRetriever.invoke(query),
			query,
		);
	}

	async getRelevantDocuments(query: string): Promise<Document[]> {
		return await this.invoke(query);
	}
}

function outputText(value: unknown): string {
	if (typeof value === 'string') return value;
	if (value && typeof value === 'object') {
		const record = value as Record<string, unknown>;
		if (typeof record.text === 'string') return record.text;
		if (typeof record.content === 'string') return record.content;
	}
	return String(value ?? '');
}

export class StringOutputParser {
	parse(text: string): string {
		return text;
	}

	async invoke(value: unknown): Promise<string> {
		return this.parse(outputText(value));
	}
}

export const StrOutputParser = StringOutputParser;

export class JsonOutputParser<T = unknown> {
	parse(text: string): T {
		return JSON.parse(text.trim()) as T;
	}

	async invoke(value: unknown): Promise<T> {
		return this.parse(outputText(value));
	}
}

export class PromptTemplate {
	constructor(readonly template: string) {}

	static fromTemplate(template: string): PromptTemplate {
		return new PromptTemplate(template);
	}

	format(values: Record<string, unknown>): string {
		// Single pass: substituted values are never re-scanned for placeholders.
		return this.template.replace(/\{([^{}]+)\}/g, (match, key: string) =>
			Object.prototype.hasOwnProperty.call(values, key)
				? String(values[key])
				: match,
		);
	}
}

type CompletionModel = {
	invoke(input: unknown): Promise<{ text?: string; content?: unknown }>;
};

export function createStuffDocumentsChain(options: {
	llm: CompletionModel;
	prompt: PromptTemplate;
	documentSeparator?: string;
}): {
	invoke(input: {
		context: Document[];
		[key: string]: unknown;
	}): Promise<string>;
} {
	return {
		invoke: async (input) => {
			const context = input.context
				.map((document) => document.pageContent)
				.join(options.documentSeparator ?? '\n\n');
			const prompt = options.prompt.format({ ...input, context });
			const response = await options.llm.invoke(prompt);
			return response.text ?? String(response.content ?? '');
		},
	};
}

export function createMapReduceDocumentsChain(options: {
	llm: CompletionModel;
	mapPrompt: PromptTemplate;
	reducePrompt: PromptTemplate;
}): {
	invoke(input: {
		context: Document[];
		[key: string]: unknown;
	}): Promise<string>;
} {
	return {
		invoke: async (input) => {
			const mapped: string[] = [];
			for (const document of input.context) {
				const response = await options.llm.invoke(
					options.mapPrompt.format({
						...input,
						context: document.pageContent,
					}),
				);
				mapped.push(response.text ?? String(response.content ?? ''));
			}
			const reduced = options.reducePrompt.format({
				...input,
				context: mapped.join('\n\n'),
			});
			const response = await options.llm.invoke(reduced);
			return response.text ?? String(response.content ?? '');
		},
	};
}

export function createRefineDocumentsChain(options: {
	llm: CompletionModel;
	initialPrompt: PromptTemplate;
	refinePrompt: PromptTemplate;
}): {
	invoke(input: {
		context: Document[];
		[key: string]: unknown;
	}): Promise<string>;
} {
	return {
		invoke: async (input) => {
			const [first, ...rest] = input.context;
			if (!first) return '';
			let response = await options.llm.invoke(
				options.initialPrompt.format({
					...input,
					context: first.pageContent,
				}),
			);
			let answer = response.text ?? String(response.content ?? '');
			for (const document of rest) {
				response = await options.llm.invoke(
					options.refinePrompt.format({
						...input,
						context: document.pageContent,
						existing_answer: answer,
					}),
				);
				answer = response.text ?? String(response.content ?? '');
			}
			return answer;
		},
	};
}

export function createRetrievalChain(options: {
	retriever: { invoke(query: string): Promise<Document[]> };
	combineDocsChain: {
		invoke(input: {
			context: Document[];
			[key: string]: unknown;
		}): Promise<string>;
	};
}): {
	invoke(input: string | { input: string; [key: string]: unknown }): Promise<{
		context: Document[];
		answer: string;
	}>;
} {
	return {
		invoke: async (input) => {
			const values = typeof input === 'string' ? { input } : input;
			const context = await options.retriever.invoke(values.input);
			const answer = await options.combineDocsChain.invoke({
				...values,
				context,
			});
			return { context, answer };
		},
	};
}

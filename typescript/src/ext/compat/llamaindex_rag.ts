import { Settings } from './llamaindex_prompts.js';
import { EnvironmentVectorDBProvider } from '../../index.js';
import {
	type EmbeddingModelProvider,
	type FileSystem,
	type RetrievalProvider,
	type RetrievalRegistry,
	type RetrievalResult,
} from '../../index.js';

type JSONObject = Record<string, unknown>;

let generatedId = 0;

function nextId(prefix: string): string {
	generatedId += 1;
	return `${prefix}-${generatedId}`;
}

function contentText(value: unknown): string {
	if (typeof value === 'string') return value;
	if (value === null || value === undefined) return '';
	if (typeof value === 'object') {
		try {
			return JSON.stringify(value);
		} catch {
			return String(value);
		}
	}
	return String(value);
}

export class Document {
	readonly id_: string;
	readonly text: string;
	readonly metadata: JSONObject;

	constructor(options: {
		text?: string;
		content?: string;
		id_?: string;
		id?: string;
		metadata?: JSONObject;
	}) {
		this.id_ = options.id_ ?? options.id ?? nextId('doc');
		this.text = options.text ?? options.content ?? '';
		this.metadata = { ...(options.metadata ?? {}) };
	}

	getText(): string {
		return this.text;
	}

	toString(): string {
		return this.text;
	}
}

export class SimpleDirectoryReader {
	constructor(
		readonly options: {
			fileSystem: FileSystem;
			inputDir?: string;
			requiredExts?: string[];
			recursive?: boolean;
		},
	) {}

	loadData(): Document[] {
		const root = this.options.inputDir ?? '';
		const paths = this.options.recursive
			? this.options.fileSystem.glob(root ? `${root}/**/*` : '**/*')
			: this.options.fileSystem
					.list(root)
					.filter((entry) => !entry.isDirectory)
					.map((entry) => entry.path);
		const requiredExts = this.options.requiredExts?.map((value) =>
			value.startsWith('.')
				? value.toLowerCase()
				: `.${value.toLowerCase()}`,
		);
		const decoder = new TextDecoder();
		return paths
			.filter((path) => {
				if (!requiredExts?.length) return true;
				const lower = path.toLowerCase();
				return requiredExts.some((extension) =>
					lower.endsWith(extension),
				);
			})
			.map(
				(path) =>
					new Document({
						id_: path,
						text: decoder.decode(
							this.options.fileSystem.read(path),
						),
						metadata: { filePath: path },
					}),
			);
	}
}

export class TextNode {
	readonly id_: string;
	readonly text: string;
	readonly metadata: JSONObject;
	readonly embedding?: number[];
	readonly refDocId?: string;

	constructor(options: {
		text: string;
		id_?: string;
		id?: string;
		metadata?: JSONObject;
		embedding?: number[];
		refDocId?: string;
	}) {
		this.id_ = options.id_ ?? options.id ?? nextId('node');
		this.text = options.text;
		this.metadata = { ...(options.metadata ?? {}) };
		this.embedding = options.embedding ? [...options.embedding] : undefined;
		this.refDocId = options.refDocId;
	}

	getText(): string {
		return this.text;
	}

	toString(): string {
		return this.text;
	}
}

export class NodeWithScore {
	constructor(
		readonly node: TextNode,
		readonly score?: number,
	) {}
}

export type TransformComponent = {
	transform(nodes: TextNode[]): Promise<TextNode[]> | TextNode[];
};

export class SentenceSplitter implements TransformComponent {
	readonly chunkSize: number;
	readonly chunkOverlap: number;

	constructor(options: { chunkSize?: number; chunkOverlap?: number } = {}) {
		this.chunkSize = options.chunkSize ?? 512;
		this.chunkOverlap = options.chunkOverlap ?? 20;
		if (!Number.isInteger(this.chunkSize) || this.chunkSize < 1) {
			throw new Error('SentenceSplitter chunkSize must be at least 1');
		}
		if (
			!Number.isInteger(this.chunkOverlap) ||
			this.chunkOverlap < 0 ||
			this.chunkOverlap >= this.chunkSize
		) {
			throw new Error(
				'SentenceSplitter chunkOverlap must be non-negative and smaller than chunkSize',
			);
		}
	}

	splitText(text: string): string[] {
		const words = text.trim().split(/\s+/).filter(Boolean);
		if (!words.length) return [];
		const step = this.chunkSize - this.chunkOverlap;
		const chunks: string[] = [];
		for (let start = 0; start < words.length; start += step) {
			const chunk = words.slice(start, start + this.chunkSize).join(' ');
			if (chunk) chunks.push(chunk);
			if (start + this.chunkSize >= words.length) break;
		}
		return chunks;
	}

	getNodesFromDocuments(documents: Document[]): TextNode[] {
		const nodes: TextNode[] = [];
		for (const document of documents) {
			const chunks = this.splitText(document.text);
			for (let index = 0; index < chunks.length; index += 1) {
				const [text] = chunks.slice(index, index + 1);
				if (text === undefined) continue;
				nodes.push(
					new TextNode({
						id_: `${document.id_}-chunk-${index}`,
						text,
						metadata: { ...document.metadata },
						refDocId: document.id_,
					}),
				);
			}
		}
		return nodes;
	}

	transform(nodes: TextNode[]): TextNode[] {
		const result: TextNode[] = [];
		for (const node of nodes) {
			const chunks = this.splitText(node.text);
			for (let index = 0; index < chunks.length; index += 1) {
				const [text] = chunks.slice(index, index + 1);
				if (text === undefined) continue;
				result.push(
					new TextNode({
						id_: `${node.id_}-chunk-${index}`,
						text,
						metadata: node.metadata,
						refDocId: node.refDocId,
					}),
				);
			}
		}
		return result;
	}
}

export interface BaseEmbedding {
	getTextEmbedding(text: string): Promise<number[]>;
	getTextEmbeddings(texts: string[]): Promise<number[][]>;
	getQueryEmbedding(query: string): Promise<number[]>;
}

export class AgentRTEmbedding implements BaseEmbedding {
	constructor(
		readonly provider: EmbeddingModelProvider,
		readonly model?: string,
	) {}

	async getTextEmbedding(text: string): Promise<number[]> {
		const [embedding] = await this.getTextEmbeddings([text]);
		if (!embedding)
			throw new Error('embedding provider returned no embedding');
		return embedding;
	}

	async getTextEmbeddings(texts: string[]): Promise<number[][]> {
		const response = await this.provider.embed({
			input: texts,
			model: this.model,
		});
		return [...response.data]
			.sort((left, right) => left.index - right.index)
			.map((item) => {
				if (!Array.isArray(item.embedding)) {
					throw new Error(
						'LlamaIndex compatibility requires numeric embedding vectors',
					);
				}
				return [...item.embedding];
			});
	}

	async getQueryEmbedding(query: string): Promise<number[]> {
		return await this.getTextEmbedding(query);
	}
}

type VectorRecord = {
	node: TextNode;
	embedding: number[];
};

function cosineSimilarity(left: number[], right: number[]): number {
	if (left.length !== right.length || left.length === 0) return 0;
	let dot = 0;
	let leftNorm = 0;
	let rightNorm = 0;
	for (const [index, leftValue] of left.entries()) {
		const [rightValue = 0] = right.slice(index, index + 1);
		dot += leftValue * rightValue;
		leftNorm += leftValue * leftValue;
		rightNorm += rightValue * rightValue;
	}
	if (!leftNorm || !rightNorm) return 0;
	return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function metadataMatches(
	metadata: JSONObject,
	filters: Record<string, unknown> | undefined,
): boolean {
	if (!filters) return true;
	const entries = Object.entries(metadata);
	return Object.entries(filters).every(([key, value]) =>
		entries.some(
			([metadataKey, metadataValue]) =>
				metadataKey === key && metadataValue === value,
		),
	);
}

export class SimpleVectorStore {
	private readonly records = new Map<string, VectorRecord>();

	add(nodes: TextNode[]): string[] {
		const ids: string[] = [];
		for (const node of nodes) {
			if (!node.embedding) {
				throw new Error(
					`node ${node.id_} has no embedding; embed nodes before adding them`,
				);
			}
			this.records.set(node.id_, {
				node,
				embedding: [...node.embedding],
			});
			ids.push(node.id_);
		}
		return ids;
	}

	delete(refDocId: string): void {
		for (const [id, record] of this.records.entries()) {
			if (record.node.refDocId === refDocId || id === refDocId) {
				this.records.delete(id);
			}
		}
	}

	query(options: {
		queryEmbedding: number[];
		similarityTopK?: number;
		filters?: Record<string, unknown>;
	}): NodeWithScore[] {
		const topK = options.similarityTopK ?? 2;
		if (!Number.isInteger(topK) || topK < 1) {
			throw new Error('similarityTopK must be at least 1');
		}
		return [...this.records.values()]
			.filter((record) =>
				metadataMatches(record.node.metadata, options.filters),
			)
			.map(
				(record) =>
					new NodeWithScore(
						record.node,
						cosineSimilarity(
							options.queryEmbedding,
							record.embedding,
						),
					),
			)
			.sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
			.slice(0, topK);
	}

	values(): TextNode[] {
		return [...this.records.values()].map((record) => record.node);
	}
}

export type VectorStoreCompatibilityOptions = {
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
	) => Promise<globalThis.Response>;
};

export interface VectorStoreLike {
	add(nodes: TextNode[]): string[];
	delete(refDocId: string): void;
	query(options: {
		queryEmbedding: number[];
		queryStr?: string;
		similarityTopK?: number;
		filters?: Record<string, unknown>;
	}): NodeWithScore[] | Promise<NodeWithScore[]>;
}

function currentVectorStoreEnvironment(): Readonly<
	Record<string, string | undefined>
> {
	const runtime = globalThis as typeof globalThis & {
		process?: { env?: Record<string, string | undefined> };
	};
	return runtime.process?.env ?? {};
}

export class AgentRTVectorStore implements VectorStoreLike {
	readonly provider: EnvironmentVectorDBProvider;

	constructor(
		backend: string,
		options: VectorStoreCompatibilityOptions = {},
	) {
		const environment: Record<string, string | undefined> = {
			...(options.environment ?? currentVectorStoreEnvironment()),
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

	add(nodes: TextNode[]): string[] {
		if (nodes.length === 0) return [];
		throw new Error(
			'Agent RT vector DB compatibility currently supports retrieval from existing collections; write/upsert remains backend-native',
		);
	}

	delete(_refDocId: string): void {
		throw new Error(
			'Agent RT vector DB compatibility currently supports retrieval from existing collections; delete remains backend-native',
		);
	}

	async query(options: {
		queryEmbedding: number[];
		queryStr?: string;
		similarityTopK?: number;
		filters?: Record<string, unknown>;
	}): Promise<NodeWithScore[]> {
		const filters = {
			...(options.filters ?? {}),
			vector: [...options.queryEmbedding],
		};
		const results = await this.provider.search({
			text: options.queryStr ?? 'vector query',
			limit: options.similarityTopK ?? 2,
			filters,
		});
		return results.map(
			(result) =>
				new NodeWithScore(
					new TextNode({
						id_: result.id,
						text: contentText(result.content),
						metadata: {
							...(result.metadata ?? {}),
							title: result.title,
							...(result.uri ? { uri: result.uri } : {}),
						},
					}),
					result.score,
				),
		);
	}
}

export class ChromaVectorStore extends AgentRTVectorStore {
	constructor(options: VectorStoreCompatibilityOptions = {}) {
		super('chroma', options);
	}
}

export class PineconeVectorStore extends AgentRTVectorStore {
	constructor(options: VectorStoreCompatibilityOptions = {}) {
		super('pinecone', options);
	}
}

export class QdrantVectorStore extends AgentRTVectorStore {
	constructor(options: VectorStoreCompatibilityOptions = {}) {
		super('qdrant', options);
	}
}

export class MilvusVectorStore extends AgentRTVectorStore {
	constructor(options: VectorStoreCompatibilityOptions = {}) {
		super('milvus', options);
	}
}

export class WeaviateVectorStore extends AgentRTVectorStore {
	constructor(options: VectorStoreCompatibilityOptions = {}) {
		super('weaviate', options);
	}
}

export class StorageContext {
	constructor(
		readonly vectorStore: VectorStoreLike = new SimpleVectorStore(),
	) {}

	static fromDefaults(
		options: {
			vectorStore?: VectorStoreLike;
		} = {},
	): StorageContext {
		return new StorageContext(options.vectorStore);
	}
}

async function embedNodes(
	nodes: TextNode[],
	embedModel: BaseEmbedding,
): Promise<TextNode[]> {
	const missing = nodes.filter((node) => !node.embedding);
	if (!missing.length) return nodes;
	const embeddings = await embedModel.getTextEmbeddings(
		missing.map((node) => node.text),
	);
	const byId = new Map<string, number[]>();
	for (let index = 0; index < missing.length; index += 1) {
		const [node] = missing.slice(index, index + 1);
		const [embedding] = embeddings.slice(index, index + 1);
		if (node && embedding) byId.set(node.id_, embedding);
	}
	return nodes.map((node) => {
		const embedding = node.embedding ?? byId.get(node.id_);
		return new TextNode({
			id_: node.id_,
			text: node.text,
			metadata: node.metadata,
			refDocId: node.refDocId,
			...(embedding ? { embedding } : {}),
		});
	});
}

export class IngestionPipeline {
	readonly transformations: TransformComponent[];
	readonly embedModel?: BaseEmbedding;

	constructor(
		options: {
			transformations?: TransformComponent[];
			embedModel?: BaseEmbedding;
		} = {},
	) {
		this.transformations = [...(options.transformations ?? [])];
		this.embedModel = options.embedModel;
	}

	async run(options: {
		documents?: Document[];
		nodes?: TextNode[];
	}): Promise<TextNode[]> {
		let nodes = options.nodes?.map(
			(node) =>
				new TextNode({
					id_: node.id_,
					text: node.text,
					metadata: node.metadata,
					embedding: node.embedding,
					refDocId: node.refDocId,
				}),
		);
		if (!nodes && options.documents) {
			nodes = this.transformations.length
				? options.documents.map(
						(document) =>
							new TextNode({
								id_: document.id_,
								text: document.text,
								metadata: document.metadata,
								refDocId: document.id_,
							}),
					)
				: new SentenceSplitter().getNodesFromDocuments(
						options.documents,
					);
		}
		nodes ??= [];
		for (const transformation of this.transformations) {
			nodes = await transformation.transform(nodes);
		}
		if (this.embedModel) nodes = await embedNodes(nodes, this.embedModel);
		return nodes;
	}
}

export interface BaseRetriever {
	retrieve(query: string): Promise<NodeWithScore[]>;
}

export class VectorIndexRetriever implements BaseRetriever {
	constructor(
		readonly index: VectorStoreIndex,
		readonly similarityTopK = 2,
		readonly filters?: Record<string, unknown>,
	) {}

	async retrieve(query: string): Promise<NodeWithScore[]> {
		const embedding = await this.index.embedModel.getQueryEmbedding(query);
		return this.index.storageContext.vectorStore.query({
			queryEmbedding: embedding,
			queryStr: query,
			similarityTopK: this.similarityTopK,
			filters: this.filters,
		});
	}
}

function retrievalResultNode(result: RetrievalResult): TextNode {
	return new TextNode({
		id_: result.id,
		text: contentText(result.content),
		metadata: {
			...(result.metadata ?? {}),
			title: result.title,
			...(result.uri ? { uri: result.uri } : {}),
		},
	});
}

export class AgentRTRetriever implements BaseRetriever {
	constructor(
		readonly source:
			| RetrievalProvider
			| { registry: RetrievalRegistry; providerName: string },
		readonly similarityTopK = 2,
		readonly filters?: Record<string, unknown>,
	) {}

	async retrieve(query: string): Promise<NodeWithScore[]> {
		const results =
			'search' in this.source
				? await this.source.search({
						text: query,
						limit: this.similarityTopK,
						filters: this.filters,
					})
				: await this.source.registry.search(this.source.providerName, {
						text: query,
						limit: this.similarityTopK,
						filters: this.filters,
					});
		return results.map(
			(result) =>
				new NodeWithScore(retrievalResultNode(result), result.score),
		);
	}
}

export class VectorStoreIndex {
	readonly storageContext: StorageContext;
	readonly embedModel: BaseEmbedding;

	constructor(
		nodes: TextNode[],
		options: {
			embedModel: BaseEmbedding;
			storageContext?: StorageContext;
		},
	) {
		this.embedModel = options.embedModel;
		this.storageContext =
			options.storageContext ?? StorageContext.fromDefaults();
		if (nodes.some((node) => !node.embedding)) {
			throw new Error(
				'VectorStoreIndex constructor requires embedded nodes; use VectorStoreIndex.fromDocuments() or create()',
			);
		}
		this.storageContext.vectorStore.add(nodes);
	}

	static async create(
		nodes: TextNode[],
		options: {
			embedModel: BaseEmbedding;
			storageContext?: StorageContext;
		},
	): Promise<VectorStoreIndex> {
		const embedded = await embedNodes(nodes, options.embedModel);
		return new VectorStoreIndex(embedded, options);
	}

	static fromVectorStore(
		vectorStore: VectorStoreLike,
		options: { embedModel: BaseEmbedding },
	): VectorStoreIndex {
		return new VectorStoreIndex([], {
			embedModel: options.embedModel,
			storageContext: new StorageContext(vectorStore),
		});
	}

	async insertNodes(nodes: TextNode[]): Promise<string[]> {
		const embedded = await embedNodes(nodes, this.embedModel);
		return this.storageContext.vectorStore.add(embedded);
	}

	deleteRefDoc(refDocId: string): void {
		this.storageContext.vectorStore.delete(refDocId);
	}

	static async fromDocuments(
		documents: Document[],
		options: {
			embedModel?: BaseEmbedding;
			storageContext?: StorageContext;
			transformations?: TransformComponent[];
		},
	): Promise<VectorStoreIndex> {
		const embedModel =
			options.embedModel ??
			(Settings.embedModel as BaseEmbedding | undefined);
		if (!embedModel || typeof embedModel.getTextEmbeddings !== 'function') {
			throw new Error(
				'VectorStoreIndex.fromDocuments requires embedModel or Settings.embedModel',
			);
		}
		const pipeline = new IngestionPipeline({
			transformations: options.transformations,
			embedModel,
		});
		const nodes = await pipeline.run({ documents });
		return new VectorStoreIndex(nodes, { ...options, embedModel });
	}

	asRetriever(
		options: {
			similarityTopK?: number;
			filters?: Record<string, unknown>;
		} = {},
	): VectorIndexRetriever {
		return new VectorIndexRetriever(
			this,
			options.similarityTopK ?? 2,
			options.filters,
		);
	}

	asQueryEngine(
		options: {
			llm?: LLMLike;
			similarityTopK?: number;
			filters?: Record<string, unknown>;
			responseSynthesizer?: ResponseSynthesizer;
		} = {},
	): RetrieverQueryEngine {
		const llm = options.llm ?? (Settings.llm as LLMLike | undefined);
		return new RetrieverQueryEngine({
			retriever: this.asRetriever(options),
			llm,
			responseSynthesizer: options.responseSynthesizer,
		});
	}
}

export type LLMLike = {
	complete(params: {
		prompt: string;
		stream?: false;
	}): Promise<{ text: string; raw?: unknown }>;
};

export class Response {
	constructor(
		readonly response: string,
		readonly sourceNodes: NodeWithScore[] = [],
		readonly metadata: JSONObject = {},
	) {}

	toString(): string {
		return this.response;
	}
}

export class ResponseSynthesizer {
	constructor(
		readonly llm: LLMLike,
		readonly promptTemplate = [
			'Use only the context below to answer the query.',
			'Context:',
			'{context}',
			'Query: {query}',
			'Answer:',
		].join('\n'),
	) {}

	async synthesize(query: string, nodes: NodeWithScore[]): Promise<Response> {
		const context = nodes
			.map((item, index) => `[${index + 1}] ${item.node.text}`)
			.join('\n\n');
		const prompt = this.promptTemplate
			.split('{context}')
			.join(context)
			.split('{query}')
			.join(query);
		const completion = await this.llm.complete({ prompt, stream: false });
		return new Response(completion.text, nodes, { raw: completion.raw });
	}
}

export function getResponseSynthesizer(
	options: {
		llm?: LLMLike;
		promptTemplate?: string;
	} = {},
): ResponseSynthesizer {
	const llm = options.llm ?? (Settings.llm as LLMLike | undefined);
	if (!llm || typeof llm.complete !== 'function') {
		throw new Error('getResponseSynthesizer requires llm or Settings.llm');
	}
	return new ResponseSynthesizer(llm, options.promptTemplate);
}

export class RetrieverQueryEngine {
	readonly retriever: BaseRetriever;
	readonly responseSynthesizer?: ResponseSynthesizer;

	constructor(options: {
		retriever: BaseRetriever;
		llm?: LLMLike;
		responseSynthesizer?: ResponseSynthesizer;
	}) {
		this.retriever = options.retriever;
		this.responseSynthesizer =
			options.responseSynthesizer ??
			(options.llm ? new ResponseSynthesizer(options.llm) : undefined);
	}

	async query(input: string | { query: string }): Promise<Response> {
		const query = typeof input === 'string' ? input : input.query;
		const nodes = await this.retriever.retrieve(query);
		if (!this.responseSynthesizer) {
			return new Response(
				nodes.map((item) => item.node.text).join('\n\n'),
				nodes,
			);
		}
		return await this.responseSynthesizer.synthesize(query, nodes);
	}
}

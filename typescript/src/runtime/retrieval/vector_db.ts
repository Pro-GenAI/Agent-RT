/** Vector database configuration and the environment-configured vector DB provider. */
import { type EmbeddingModelProvider } from '../models/interfaces.js';
import { currentProviderEnvironment } from '../models/sdk.js';
import {
	type RetrievalProvider,
	type RetrievalQuery,
	type RetrievalResult,
} from './registry.js';

export const AGENT_RT_VECTOR_DB_ENV = 'AGENT_RT_VECTOR_DB';

export const AGENT_RT_VECTOR_DB_COLLECTION_ENV =
	'AGENT_RT_VECTOR_DB_COLLECTION';

export const AGENT_RT_VECTOR_DB_URL_ENV = 'AGENT_RT_VECTOR_DB_URL';

export const AGENT_RT_VECTOR_DB_OPTION_PREFIX = 'AGENT_RT_VECTOR_DB_OPTION_';

export interface VectorDBConfig {
	backend: string;
	collection?: string;
	url?: string;
	options: Readonly<Record<string, string>>;
}

export type VectorDBProviderFactory = (
	config: VectorDBConfig,
) => RetrievalProvider;

export class VectorDBProviderRegistry {
	private readonly factories = new Map<string, VectorDBProviderFactory>();

	register(
		name: string,
		factory: VectorDBProviderFactory,
		options: { replace?: boolean } = {},
	): void {
		const normalized = name.trim().toLowerCase();
		if (!normalized)
			throw new Error('vector DB backend name must not be empty');
		if (typeof factory !== 'function')
			throw new TypeError('vector DB provider factory must be callable');
		if (this.factories.has(normalized) && !options.replace) {
			throw new Error(
				`vector DB backend already registered: ${normalized}`,
			);
		}
		this.factories.set(normalized, factory);
	}

	create(config: VectorDBConfig): RetrievalProvider {
		const backend = config.backend.trim().toLowerCase();
		const factory = this.factories.get(backend);
		if (!factory) {
			const supported =
				[...this.factories.keys()].sort().join(', ') || '<none>';
			throw new Error(
				`unsupported vector DB backend ${JSON.stringify(config.backend)}; registered backends: ${supported}`,
			);
		}
		const provider = factory(config);
		if (!provider || typeof provider.search !== 'function') {
			throw new TypeError(
				`vector DB backend ${JSON.stringify(backend)} factory must return a RetrievalProvider`,
			);
		}
		return provider;
	}
}

export function vectorDBConfigFromEnvironment(
	environment?: Readonly<Record<string, string | undefined>>,
): VectorDBConfig {
	const env = environment ?? currentProviderEnvironment();
	const backend = (env[AGENT_RT_VECTOR_DB_ENV] ?? '').trim().toLowerCase();
	if (!backend)
		throw new Error(
			`vector DB selection requires ${AGENT_RT_VECTOR_DB_ENV}`,
		);
	const collection =
		(env[AGENT_RT_VECTOR_DB_COLLECTION_ENV] ?? '').trim() || undefined;
	const url = (env[AGENT_RT_VECTOR_DB_URL_ENV] ?? '').trim() || undefined;
	const options: Record<string, string> = {};
	for (const [key, value] of Object.entries(env)) {
		if (
			!key.startsWith(AGENT_RT_VECTOR_DB_OPTION_PREFIX) ||
			value === undefined
		)
			continue;
		const optionName = key
			.slice(AGENT_RT_VECTOR_DB_OPTION_PREFIX.length)
			.trim()
			.toLowerCase();
		const optionValue = value.trim();
		if (optionName && optionValue) options[optionName] = optionValue;
	}
	return { backend, collection, url, options };
}

export function vectorDBProviderFromEnvironment(
	registry: VectorDBProviderRegistry,
	environment?: Readonly<Record<string, string | undefined>>,
): RetrievalProvider {
	return registry.create(vectorDBConfigFromEnvironment(environment));
}

export const AGENT_RT_VECTOR_DB_API_KEY_ENV = 'AGENT_RT_VECTOR_DB_API_KEY';

export const AGENT_RT_VECTOR_DB_TIMEOUT_ENV =
	'AGENT_RT_VECTOR_DB_TIMEOUT_SECONDS';

export const POPULAR_VECTOR_DB_BACKENDS = [
	'chroma',
	'milvus',
	'pinecone',
	'qdrant',
	'weaviate',
] as const;

const VECTOR_DB_CREDENTIAL_ENV: Readonly<Record<string, string>> = {
	chroma: 'CHROMA_API_KEY',
	milvus: 'MILVUS_TOKEN',
	pinecone: 'PINECONE_API_KEY',
	qdrant: 'QDRANT_API_KEY',
	weaviate: 'WEAVIATE_API_KEY',
};

export type VectorDBFetch = (
	input: string | URL | Request,
	init?: RequestInit,
) => Promise<Response>;

function vectorDBHttpUrl(value: string, backend: string): string {
	let parsed: URL;
	try {
		parsed = new URL(value.trim());
	} catch {
		throw new Error(
			`${backend} vector DB URL must be an absolute HTTP(S) URL`,
		);
	}
	if (!['http:', 'https:'].includes(parsed.protocol)) {
		throw new Error(
			`${backend} vector DB URL must be an absolute HTTP(S) URL`,
		);
	}
	if (parsed.username || parsed.password) {
		throw new Error(
			`${backend} vector DB URL must not contain embedded credentials`,
		);
	}
	return parsed.toString().replace(/\/$/, '');
}

function vectorDBNumber(value: unknown): number | undefined {
	const number = typeof value === 'number' ? value : Number(value);
	return Number.isFinite(number) ? number : undefined;
}

function vectorDBRecord(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

export class EnvironmentVectorDBProvider implements RetrievalProvider {
	readonly kind = 'knowledge' as const;
	readonly backend: (typeof POPULAR_VECTOR_DB_BACKENDS)[number];
	readonly url: string;
	readonly collection: string;
	readonly timeoutSeconds: number;
	private readonly embeddingProvider?: EmbeddingModelProvider;
	private readonly apiKey?: string;
	private readonly fetchImpl: VectorDBFetch;

	constructor(
		readonly config: VectorDBConfig,
		options: {
			embeddingProvider?: EmbeddingModelProvider;
			apiKey?: string;
			timeoutSeconds?: number;
			fetchImpl?: VectorDBFetch;
		} = {},
	) {
		const backend = config.backend.trim().toLowerCase();
		if (
			!(POPULAR_VECTOR_DB_BACKENDS as readonly string[]).includes(backend)
		) {
			throw new Error(
				`unsupported built-in vector DB backend ${JSON.stringify(backend)}; expected one of: ${POPULAR_VECTOR_DB_BACKENDS.join(', ')}`,
			);
		}
		if (!config.url)
			throw new Error(
				`${backend} vector DB requires ${AGENT_RT_VECTOR_DB_URL_ENV}`,
			);
		if (!config.collection) {
			throw new Error(
				`${backend} vector DB requires ${AGENT_RT_VECTOR_DB_COLLECTION_ENV}`,
			);
		}
		const timeoutSeconds = options.timeoutSeconds ?? 20;
		if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) {
			throw new Error('vector DB timeout must be positive');
		}
		this.backend = backend as (typeof POPULAR_VECTOR_DB_BACKENDS)[number];
		this.url = vectorDBHttpUrl(config.url, backend);
		this.collection = config.collection;
		this.timeoutSeconds = timeoutSeconds;
		this.embeddingProvider = options.embeddingProvider;
		this.apiKey = options.apiKey?.trim() || undefined;
		this.fetchImpl = options.fetchImpl ?? fetch;
	}

	static fromEnvironment(
		environment?: Readonly<Record<string, string | undefined>>,
		options: {
			embeddingProvider?: EmbeddingModelProvider;
			fetchImpl?: VectorDBFetch;
		} = {},
	): EnvironmentVectorDBProvider {
		const env = environment ?? currentProviderEnvironment();
		const config = vectorDBConfigFromEnvironment(env);
		if (
			!(POPULAR_VECTOR_DB_BACKENDS as readonly string[]).includes(
				config.backend,
			)
		) {
			throw new Error(
				`unsupported built-in vector DB backend ${JSON.stringify(config.backend)}; expected one of: ${POPULAR_VECTOR_DB_BACKENDS.join(', ')}`,
			);
		}
		const credentialEnv = VECTOR_DB_CREDENTIAL_ENV[config.backend];
		const apiKey =
			env[credentialEnv] ?? env[AGENT_RT_VECTOR_DB_API_KEY_ENV];
		const timeoutSeconds = Number(
			env[AGENT_RT_VECTOR_DB_TIMEOUT_ENV] ?? '20',
		);
		if (!Number.isFinite(timeoutSeconds)) {
			throw new Error(
				`${AGENT_RT_VECTOR_DB_TIMEOUT_ENV} must be numeric`,
			);
		}
		return new EnvironmentVectorDBProvider(config, {
			embeddingProvider: options.embeddingProvider,
			fetchImpl: options.fetchImpl,
			apiKey,
			timeoutSeconds,
		});
	}

	async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
		const limit = query.limit ?? 10;
		if (!Number.isInteger(limit) || limit < 1) {
			throw new Error('retrieval limit must be at least 1');
		}
		if (!query.text.trim())
			throw new Error('vector DB query must not be empty');
		const vector = await this.queryVector(query);
		if (this.backend === 'qdrant')
			return await this.searchQdrant(query, vector, limit);
		if (this.backend === 'pinecone')
			return await this.searchPinecone(query, vector, limit);
		if (this.backend === 'milvus')
			return await this.searchMilvus(query, vector, limit);
		if (this.backend === 'weaviate')
			return await this.searchWeaviate(query, vector, limit);
		return await this.searchChroma(query, vector, limit);
	}

	private async queryVector(query: RetrievalQuery): Promise<number[]> {
		const supplied = query.filters?.vector;
		if (Array.isArray(supplied)) {
			const vector = supplied.map(Number);
			if (vector.length > 0 && vector.every(Number.isFinite))
				return vector;
			throw new Error(
				'filters.vector must be a non-empty finite numeric vector',
			);
		}
		if (!this.embeddingProvider) {
			throw new Error(
				'vector DB text search requires an EmbeddingModelProvider or a precomputed filters.vector',
			);
		}
		const response = await this.embeddingProvider.embed({
			input: query.text,
		});
		const embedding = response.data[0]?.embedding;
		if (!Array.isArray(embedding)) {
			throw new Error('vector DB search requires numeric embeddings');
		}
		const vector = embedding.map(Number);
		if (vector.length === 0 || !vector.every(Number.isFinite)) {
			throw new Error('embedding provider returned an invalid vector');
		}
		return vector;
	}

	private filtersWithoutVector(
		query: RetrievalQuery,
	): Record<string, unknown> {
		const filters = { ...(query.filters ?? {}) };
		delete filters.vector;
		return filters;
	}

	private async request(
		url: string,
		init: RequestInit,
	): Promise<Record<string, unknown>> {
		const controller = new AbortController();
		const timer = setTimeout(
			() => controller.abort(new Error('vector DB request timed out')),
			this.timeoutSeconds * 1000,
		);
		try {
			const response = await this.fetchImpl(url, {
				...init,
				signal: controller.signal,
			});
			if (!response.ok) {
				throw new Error(
					`${this.backend} vector DB returned HTTP ${response.status}`,
				);
			}
			const payload: unknown = await response.json();
			const record = vectorDBRecord(payload);
			if (!record)
				throw new Error(
					`${this.backend} vector DB returned a non-object JSON response`,
				);
			return record;
		} finally {
			clearTimeout(timer);
		}
	}

	private result(
		id: unknown,
		content: unknown,
		options: {
			score?: unknown;
			title?: unknown;
			uri?: unknown;
			metadata?: Record<string, unknown>;
		} = {},
	): RetrievalResult {
		const metadata = options.metadata ?? {};
		const score = vectorDBNumber(options.score);
		const uri =
			typeof options.uri === 'string'
				? options.uri
				: typeof metadata.url === 'string'
					? metadata.url
					: undefined;
		return {
			id: String(id),
			title: String(options.title ?? metadata.title ?? id),
			content: content ?? metadata.text ?? metadata.content ?? '',
			...(score !== undefined ? { score } : {}),
			...(uri ? { uri } : {}),
			metadata: { provider: this.backend, ...metadata },
		};
	}

	private async searchQdrant(
		query: RetrievalQuery,
		vector: number[],
		limit: number,
	): Promise<RetrievalResult[]> {
		const filters = this.filtersWithoutVector(query);
		const body: Record<string, unknown> = {
			vector,
			limit,
			with_payload: true,
		};
		if (Object.keys(filters).length) body.filter = filters;
		const payload = await this.request(
			`${this.url}/collections/${encodeURIComponent(this.collection)}/points/search`,
			{
				method: 'POST',
				headers: this.apiKey
					? {
							'api-key': this.apiKey,
							'content-type': 'application/json',
						}
					: { 'content-type': 'application/json' },
				body: JSON.stringify(body),
			},
		);
		return (Array.isArray(payload.result) ? payload.result : []).flatMap(
			(value, index) => {
				const item = vectorDBRecord(value);
				if (!item) return [];
				const metadata = vectorDBRecord(item.payload) ?? {};
				return [
					this.result(item.id ?? index, metadata.text, {
						score: item.score,
						metadata,
					}),
				];
			},
		);
	}

	private async searchPinecone(
		query: RetrievalQuery,
		vector: number[],
		limit: number,
	): Promise<RetrievalResult[]> {
		const filters = this.filtersWithoutVector(query);
		const body: Record<string, unknown> = {
			vector,
			topK: limit,
			includeMetadata: true,
		};
		if (Object.keys(filters).length) body.filter = filters;
		if (this.config.options.namespace)
			body.namespace = this.config.options.namespace;
		const headers: Record<string, string> = {
			'content-type': 'application/json',
		};
		if (this.apiKey) headers['Api-Key'] = this.apiKey;
		const payload = await this.request(`${this.url}/query`, {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
		});
		return (Array.isArray(payload.matches) ? payload.matches : []).flatMap(
			(value, index) => {
				const item = vectorDBRecord(value);
				if (!item) return [];
				const metadata = vectorDBRecord(item.metadata) ?? {};
				return [
					this.result(item.id ?? index, metadata.text, {
						score: item.score,
						metadata,
					}),
				];
			},
		);
	}

	/** Fail closed: never silently drop filters that may enforce isolation. */
	private rejectUnsupportedFilters(query: RetrievalQuery): void {
		if (Object.keys(this.filtersWithoutVector(query)).length > 0) {
			throw new Error(
				`the ${this.backend} adapter does not support per-query filters; ` +
					'refusing to run an unfiltered search (use a collection per ' +
					'scope, or a backend that supports filters)',
			);
		}
	}

	private async searchMilvus(
		query: RetrievalQuery,
		vector: number[],
		limit: number,
	): Promise<RetrievalResult[]> {
		this.rejectUnsupportedFilters(query);
		const body: Record<string, unknown> = {
			collectionName: this.collection,
			data: [vector],
			limit,
			outputFields: ['*'],
		};
		if (this.config.options.filter) {
			body.filter = this.config.options.filter;
		}
		const headers: Record<string, string> = {
			'content-type': 'application/json',
		};
		if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
		const payload = await this.request(
			`${this.url}/v2/vectordb/entities/search`,
			{
				method: 'POST',
				headers,
				body: JSON.stringify(body),
			},
		);
		const raw = Array.isArray(payload.data) ? payload.data : [];
		const items =
			raw.length > 0 && Array.isArray(raw[0])
				? (raw[0] as unknown[])
				: raw;
		return items.flatMap((value, index) => {
			const item = vectorDBRecord(value);
			if (!item) return [];
			return [
				this.result(
					item.id ?? item.pk ?? index,
					item.text ?? item.content,
					{
						score: item.score,
						metadata: item,
					},
				),
			];
		});
	}

	private async searchWeaviate(
		query: RetrievalQuery,
		vector: number[],
		limit: number,
	): Promise<RetrievalResult[]> {
		this.rejectUnsupportedFilters(query);
		const identifier = /^[A-Za-z_][A-Za-z0-9_]*$/;
		const className = this.collection;
		const contentField = this.config.options.content_field ?? 'text';
		const titleField = this.config.options.title_field ?? 'title';
		const uriField = this.config.options.uri_field ?? 'url';
		for (const value of [className, contentField, titleField, uriField]) {
			if (!identifier.test(value)) {
				throw new Error(
					'Weaviate collection and field names must be GraphQL identifiers',
				);
			}
		}
		const vectorLiteral = vector
			.map((value) => Number(value).toString())
			.join(', ');
		const graphQuery =
			`{ Get { ${className}(nearVector: {vector: [${vectorLiteral}]}, limit: ${limit}) ` +
			`{ ${contentField} ${titleField} ${uriField} _additional { id distance certainty } } } }`;
		const headers: Record<string, string> = {
			'content-type': 'application/json',
		};
		if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
		const payload = await this.request(`${this.url}/v1/graphql`, {
			method: 'POST',
			headers,
			body: JSON.stringify({ query: graphQuery }),
		});
		const data = vectorDBRecord(payload.data);
		const getValue = vectorDBRecord(data?.Get);
		const items =
			getValue && Array.isArray(getValue[className])
				? (getValue[className] as unknown[])
				: [];
		return items.flatMap((value, index) => {
			const item = vectorDBRecord(value);
			if (!item) return [];
			const additional = vectorDBRecord(item._additional) ?? {};
			let score = additional.certainty;
			if (score === undefined) {
				const distance = vectorDBNumber(additional.distance);
				if (distance !== undefined) score = 1 - distance;
			}
			const metadata = Object.fromEntries(
				Object.entries(item).filter(([key]) => key !== '_additional'),
			);
			return [
				this.result(additional.id ?? index, item[contentField], {
					score,
					title: item[titleField],
					uri: item[uriField],
					metadata,
				}),
			];
		});
	}

	private async searchChroma(
		query: RetrievalQuery,
		vector: number[],
		limit: number,
	): Promise<RetrievalResult[]> {
		const tenant = encodeURIComponent(
			this.config.options.tenant ?? 'default_tenant',
		);
		const database = encodeURIComponent(
			this.config.options.database ?? 'default_database',
		);
		const collection = encodeURIComponent(this.collection);
		const headers: Record<string, string> = {
			'content-type': 'application/json',
		};
		if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
		const filters = this.filtersWithoutVector(query);
		const body: Record<string, unknown> = {
			query_embeddings: [vector],
			n_results: limit,
			include: ['documents', 'metadatas', 'distances', 'uris'],
		};
		if (Object.keys(filters).length) body.where = filters;
		const payload = await this.request(
			`${this.url}/api/v2/tenants/${tenant}/databases/${database}/collections/${collection}/query`,
			{ method: 'POST', headers, body: JSON.stringify(body) },
		);
		const unwrap = (value: unknown): unknown[] => {
			if (!Array.isArray(value)) return [];
			return value.length > 0 && Array.isArray(value[0])
				? (value[0] as unknown[])
				: value;
		};
		const ids = unwrap(payload.ids);
		const documents = unwrap(payload.documents);
		const metadatas = unwrap(payload.metadatas);
		const distances = unwrap(payload.distances);
		const uris = unwrap(payload.uris);
		return ids.map((id, index) => {
			const metadata = vectorDBRecord(metadatas[index]) ?? {};
			const distance = vectorDBNumber(distances[index]);
			const score =
				distance === undefined
					? undefined
					: 1 / (1 + Math.max(0, distance));
			return this.result(id, documents[index], {
				score,
				uri: uris[index],
				metadata,
			});
		});
	}
}

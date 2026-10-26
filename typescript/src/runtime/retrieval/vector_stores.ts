/** Popular vector DB backends, Bedrock Knowledge Bases, and configured vector stores. */
import {
	type VectorStoreConfig,
	secretValue,
	requireResolved,
	type AgentRTConfig,
	findVectorStore,
} from '../../ext/config.js';
import {
	type EmbeddingModelProvider,
	type EmbeddingRequest,
	type EmbeddingResponse,
} from '../models/interfaces.js';
import { currentProviderEnvironment } from '../models/sdk.js';
import {
	type RetrievalProvider,
	type RetrievalQuery,
	RetrievalRegistry,
	type RetrievalResult,
} from './registry.js';
import {
	AGENT_RT_VECTOR_DB_COLLECTION_ENV,
	AGENT_RT_VECTOR_DB_ENV,
	AGENT_RT_VECTOR_DB_OPTION_PREFIX,
	AGENT_RT_VECTOR_DB_URL_ENV,
	EnvironmentVectorDBProvider,
	POPULAR_VECTOR_DB_BACKENDS,
	type VectorDBFetch,
	type VectorDBProviderFactory,
	type VectorDBProviderRegistry,
} from './vector_db.js';
import { importOptionalPeer } from '../platform.js';

export function registerPopularVectorDBBackends(
	registry: VectorDBProviderRegistry,
	options: {
		embeddingProvider?: EmbeddingModelProvider;
		environment?: Readonly<Record<string, string | undefined>>;
		fetchImpl?: VectorDBFetch;
		replace?: boolean;
	} = {},
): VectorDBProviderRegistry {
	const env = options.environment ?? currentProviderEnvironment();
	const factory: VectorDBProviderFactory = (config) => {
		const backendEnv: Record<string, string | undefined> = {
			...env,
			[AGENT_RT_VECTOR_DB_ENV]: config.backend,
			[AGENT_RT_VECTOR_DB_COLLECTION_ENV]: config.collection,
			[AGENT_RT_VECTOR_DB_URL_ENV]: config.url,
		};
		for (const [key, value] of Object.entries(config.options)) {
			backendEnv[
				`${AGENT_RT_VECTOR_DB_OPTION_PREFIX}${key.toUpperCase()}`
			] = value;
		}
		return EnvironmentVectorDBProvider.fromEnvironment(backendEnv, {
			embeddingProvider: options.embeddingProvider,
			fetchImpl: options.fetchImpl,
		});
	};
	for (const backend of POPULAR_VECTOR_DB_BACKENDS) {
		registry.register(backend, factory, { replace: options.replace });
	}
	return registry;
}

class BoundEmbeddingProvider implements EmbeddingModelProvider {
	constructor(
		private readonly provider: EmbeddingModelProvider,
		readonly modelName: string,
	) {}

	embed(request: EmbeddingRequest): Promise<EmbeddingResponse> {
		return this.provider.embed({ ...request, model: this.modelName });
	}
}

type BedrockRetrieveClient = {
	retrieve(input: Record<string, unknown>): Promise<unknown>;
};

async function defaultBedrockClient(
	store: VectorStoreConfig,
): Promise<BedrockRetrieveClient> {
	let sdk: {
		BedrockAgentRuntimeClient: new (options: Record<string, unknown>) => {
			send(
				command: unknown,
				options?: { abortSignal?: AbortSignal },
			): Promise<unknown>;
		};
		RetrieveCommand: new (input: Record<string, unknown>) => unknown;
	};
	try {
		sdk = await importOptionalPeer<typeof sdk>(
			'@aws-sdk/client-bedrock-agent-runtime',
		);
	} catch {
		throw new Error(
			'Bedrock vector stores require @aws-sdk/client-bedrock-agent-runtime: npm install @aws-sdk/client-bedrock-agent-runtime',
		);
	}
	const client = new sdk.BedrockAgentRuntimeClient({
		...(store.options.region ? { region: store.options.region } : {}),
		...(store.apiBase ? { endpoint: secretValue(store.apiBase) } : {}),
	});
	return {
		retrieve: (input) =>
			client.send(new sdk.RetrieveCommand(input), {
				abortSignal: AbortSignal.timeout(store.timeout * 1000),
			}),
	};
}

class BedrockKnowledgeBaseProvider implements RetrievalProvider {
	readonly kind = 'knowledge';
	private client?: BedrockRetrieveClient;
	constructor(
		private readonly store: VectorStoreConfig,
		private readonly clientFactory: (
			store: VectorStoreConfig,
		) => BedrockRetrieveClient | Promise<BedrockRetrieveClient>,
	) {}

	async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
		if (!query.text.trim())
			throw new Error('file_search query must not be empty');
		if (!Number.isInteger(query.limit ?? 10) || (query.limit ?? 10) < 1) {
			throw new Error('retrieval limit must be at least 1');
		}
		this.client ??= await this.clientFactory(this.store);
		const response = (await this.client.retrieve({
			knowledgeBaseId: this.store.storeId,
			retrievalQuery: { text: query.text },
			retrievalConfiguration: {
				vectorSearchConfiguration: {
					numberOfResults: query.limit ?? 10,
				},
			},
		})) as {
			retrievalResults?: Array<{
				content?: { text?: string };
				location?: Record<string, { uri?: string; url?: string }>;
				score?: number;
				metadata?: Record<string, unknown>;
			}>;
		};
		return (response.retrievalResults ?? []).map((item, index) => {
			const location = item.location ?? {};
			const uri =
				location.webLocation?.url ??
				location.s3Location?.uri ??
				location.confluenceLocation?.url ??
				location.salesforceLocation?.url ??
				location.sharePointLocation?.url;
			return {
				id: `${this.store.storeId}:${index}`,
				title: uri ?? this.store.name,
				content: item.content?.text ?? '',
				...(item.score !== undefined ? { score: item.score } : {}),
				...(uri !== undefined ? { uri } : {}),
				metadata: item.metadata ?? {},
			};
		});
	}
}

class LazyConfiguredVectorStore implements RetrievalProvider {
	readonly kind = 'knowledge';
	private provider?: RetrievalProvider;
	constructor(
		private readonly store: VectorStoreConfig,
		readonly embeddingProvider: EmbeddingModelProvider | undefined,
		private readonly fetchImpl: VectorDBFetch | undefined,
		private readonly bedrockClientFactory: (
			store: VectorStoreConfig,
		) => BedrockRetrieveClient | Promise<BedrockRetrieveClient>,
	) {}

	async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
		if (!this.provider) {
			requireResolved(this.store.apiKey, this.store.apiBase);
			if (this.store.provider === 'bedrock') {
				this.provider = new BedrockKnowledgeBaseProvider(
					this.store,
					this.bedrockClientFactory,
				);
			} else {
				const apiBase = secretValue(this.store.apiBase);
				if (!apiBase) {
					throw new Error(
						`vector store ${JSON.stringify(this.store.name)} requires api_base`,
					);
				}
				if (!this.store.collection) {
					throw new Error(
						`vector store ${JSON.stringify(this.store.name)} requires collection`,
					);
				}
				this.provider = new EnvironmentVectorDBProvider(
					{
						backend: this.store.provider,
						collection: this.store.collection,
						url: apiBase,
						options: { ...this.store.options },
					},
					{
						embeddingProvider: this.embeddingProvider,
						apiKey: secretValue(this.store.apiKey),
						timeoutSeconds: this.store.timeout,
						fetchImpl: this.fetchImpl,
					},
				);
			}
		}
		return this.provider.search(query);
	}
}

/** Register store aliases without initializing unused providers or their secrets. */
export function vectorStoreRegistryFromConfig(
	config: AgentRTConfig,
	options: {
		embeddingProvider?: EmbeddingModelProvider;
		fetchImpl?: VectorDBFetch;
		bedrockClientFactory?: (
			store: VectorStoreConfig,
		) => BedrockRetrieveClient | Promise<BedrockRetrieveClient>;
	} = {},
): RetrievalRegistry {
	const registry = new RetrievalRegistry();
	for (const store of config.vectorStores) {
		const embeddingProvider =
			options.embeddingProvider && store.embeddingModel
				? new BoundEmbeddingProvider(
						options.embeddingProvider,
						store.embeddingModel,
					)
				: options.embeddingProvider;
		const provider = new LazyConfiguredVectorStore(
			store,
			embeddingProvider,
			options.fetchImpl,
			options.bedrockClientFactory ?? defaultBedrockClient,
		);
		registry.register(store.name, provider);
		if (store.storeId !== store.name)
			registry.register(store.storeId, provider);
	}
	return registry;
}

/** Validate requested `vector_store_ids`; unknown or disallowed ids throw. */
export function resolveVectorStores(
	config: AgentRTConfig,
	references: readonly string[],
	options: { modelName?: string } = {},
): VectorStoreConfig[] {
	const resolved: VectorStoreConfig[] = [];
	for (const reference of references) {
		const store =
			typeof reference === 'string'
				? findVectorStore(config, reference)
				: undefined;
		if (!store) {
			throw new Error(
				`unknown vector store: ${JSON.stringify(reference)}`,
			);
		}
		if (
			store.allowedModels.length &&
			(options.modelName === undefined ||
				!store.allowedModels.includes(options.modelName))
		) {
			throw new Error(
				`vector store ${JSON.stringify(reference)} is not available to this model`,
			);
		}
		if (!resolved.includes(store)) resolved.push(store);
	}
	if (!resolved.length) {
		throw new Error('file_search requires at least one vector store id');
	}
	return resolved;
}

/** Config-driven model providers and `loadModel`. */
import {
	type ModelDeployment,
	type AgentRTConfig,
	defaultModel as configDefaultModel,
	modelNames as configModelNames,
	secretValue,
	requireResolved,
	deployments as configDeployments,
} from '../../ext/config.js';
import {
	type EmbeddingModelProvider,
	type EmbeddingRequest,
	type EmbeddingResponse,
	type ModelCatalogEntry,
	type ModelProvider,
	type ModelRequest,
	type ModelResponse,
	type ModelStreamEvent,
	type StreamingModelProvider,
} from './interfaces.js';
import { type ModelListClientFactory } from './sdk.js';
import {
	resolveAnthropicProviderSettings,
	resolveOpenAIProviderSettings,
	validateAnthropicProviderSettings,
	validateOpenAIProviderSettings,
} from './provider_settings.js';
import { OpenAIModelProvider } from './openai.js';
import { AnthropicModelProvider } from './anthropic.js';
import { isStreamingModelProvider } from '../agent/config.js';

/**
 * Routes requests by public `modelName` across configured deployments.
 * Deployments sharing a `modelName` act as ordered fallbacks; provider
 * clients are created lazily, one per deployment.
 */
export class ConfiguredModelProvider
	implements StreamingModelProvider, EmbeddingModelProvider
{
	readonly name = 'agent-rt-config';
	readonly settings: { defaultModel?: string };
	private readonly providers = new Map<ModelDeployment, ModelProvider>();

	constructor(
		readonly config: AgentRTConfig,
		private readonly providerFactory: (
			deployment: ModelDeployment,
		) => ModelProvider = ConfiguredModelProvider.defaultFactory,
		private readonly validationClientFactory?: ModelListClientFactory,
	) {
		if (!config.models.length) {
			throw new Error('model_list must contain at least one model');
		}
		this.settings = { defaultModel: configDefaultModel(config) };
	}

	get modelNames(): string[] {
		return configModelNames(this.config);
	}

	static defaultFactory(deployment: ModelDeployment): ModelProvider {
		const apiKey = secretValue(deployment.apiKey);
		const apiBase = secretValue(deployment.apiBase);
		if (deployment.provider === 'openai') {
			return new OpenAIModelProvider({
				apiKey,
				defaultModel: deployment.model,
				...(apiBase ? { baseUrl: apiBase } : {}),
				...(deployment.websocket !== undefined
					? { websocket: deployment.websocket }
					: apiBase
						? { websocket: false }
						: {}),
			});
		}
		return new AnthropicModelProvider({
			apiKey,
			defaultModel: deployment.model,
			...(apiBase ? { baseUrl: apiBase } : {}),
		});
	}

	private provider(deployment: ModelDeployment): ModelProvider {
		let provider = this.providers.get(deployment);
		if (!provider) {
			requireResolved(deployment.apiKey, deployment.apiBase);
			provider = this.providerFactory(deployment);
			this.providers.set(deployment, provider);
		}
		return provider;
	}

	private targets(requested?: string): ModelDeployment[] {
		const name = requested ?? configDefaultModel(this.config);
		const found = name ? configDeployments(this.config, name) : [];
		if (!found.length) {
			throw new Error(
				`unknown model ${JSON.stringify(name)}; configured models: ${this.modelNames.join(', ')}`,
			);
		}
		return found;
	}

	async validate(): Promise<void> {
		for (const deployment of this.config.models) {
			const provider = this.provider(deployment);
			// An injected provider must not trigger a separate vendor SDK connection.
			if (
				this.providerFactory !==
					ConfiguredModelProvider.defaultFactory &&
				!this.validationClientFactory
			) {
				const validator = (
					provider as { validate?: () => Promise<void> }
				).validate;
				if (typeof validator === 'function')
					await validator.call(provider);
				continue;
			}
			const apiBase = secretValue(deployment.apiBase);
			const settings = {
				apiKey: secretValue(deployment.apiKey),
				defaultModel: deployment.model,
				...(apiBase ? { baseUrl: apiBase } : {}),
			};
			if (deployment.provider === 'openai') {
				await validateOpenAIProviderSettings(
					resolveOpenAIProviderSettings(settings),
					{
						checkDefaultModel: true,
						clientFactory: this.validationClientFactory,
					},
				);
			} else {
				await validateAnthropicProviderSettings(
					resolveAnthropicProviderSettings(settings),
					{
						checkDefaultModel: true,
						clientFactory: this.validationClientFactory,
					},
				);
			}
		}
	}

	async complete(request: ModelRequest): Promise<ModelResponse> {
		let lastError: unknown;
		for (const deployment of this.targets(request.model)) {
			try {
				return await this.provider(deployment).complete({
					...request,
					model: deployment.model,
				});
			} catch (error) {
				lastError = error;
			}
		}
		throw lastError;
	}

	async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
		let lastError: unknown;
		for (const deployment of this.targets(request.model)) {
			const inner = { ...request, model: deployment.model };
			let started = false;
			try {
				const provider = this.provider(deployment);
				if (isStreamingModelProvider(provider)) {
					for await (const event of provider.stream(inner)) {
						started = true;
						yield event;
					}
				} else {
					yield {
						type: 'completed',
						response: await provider.complete(inner),
					};
				}
				return;
			} catch (error) {
				if (started) throw error;
				lastError = error;
			}
		}
		throw lastError;
	}

	async embed(request: EmbeddingRequest): Promise<EmbeddingResponse> {
		let lastError: unknown;
		for (const deployment of this.targets(request.model)) {
			const provider = this.provider(
				deployment,
			) as Partial<EmbeddingModelProvider>;
			try {
				if (typeof provider.embed !== 'function') {
					throw new Error(
						`${deployment.provider} deployment cannot embed`,
					);
				}
				return await provider.embed({
					...request,
					model: deployment.model,
				});
			} catch (error) {
				lastError = error;
			}
		}
		throw lastError;
	}

	async listModels(): Promise<ModelCatalogEntry[]> {
		return this.modelNames.map((id) => ({ id, ownedBy: 'agent-rt' }));
	}
}

/**
 * Build the model provider declared by `model_list` in a config. Use
 * `loadConfig(path?)` to read one (explicit path, else the current folder,
 * then the home folder). The process environment is only used to resolve
 * `os.environ/NAME` secret references.
 */
export async function loadModel(
	config: AgentRTConfig,
	options: {
		validate?: boolean;
		providerFactory?: (deployment: ModelDeployment) => ModelProvider;
		validationClientFactory?: ModelListClientFactory;
	} = {},
): Promise<ConfiguredModelProvider> {
	const provider = new ConfiguredModelProvider(
		config,
		options.providerFactory,
		options.validationClientFactory,
	);
	if (options.validate) await provider.validate();
	return provider;
}

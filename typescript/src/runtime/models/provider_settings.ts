/** OpenAI and Anthropic provider settings. */
import {
	DEFAULT_ANTHROPIC_BASE_URL,
	DEFAULT_OPENAI_BASE_URL,
	type ModelListClient,
	type ModelListClientFactory,
	anthropicListModels,
	modelListIds,
	setOpenAIWebSocketCapability,
	validateProviderBaseUrl,
} from './sdk.js';
import { importOptionalPeer } from '../platform.js';

export interface OpenAIProviderSettings {
	baseUrl?: string;
	apiKey?: string;
	defaultModel?: string;
	transport?: 'compatible' | 'sdk';
	websocket?: boolean;
	/**
	 * HTTP implementation for every request (the SDKs' `fetch` option); test
	 * doubles and offline transports depend on it. Defaults to global fetch.
	 */
	fetch?: typeof fetch;
}

export interface ResolvedOpenAIProviderSettings {
	baseUrl: string;
	apiKey?: string;
	defaultModel?: string;
	transport?: 'compatible' | 'sdk';
	websocket: boolean;
	fetch?: typeof fetch;
}

export interface AnthropicProviderSettings {
	baseUrl?: string;
	apiKey?: string;
	defaultModel?: string;
	/**
	 * HTTP implementation for every request (the SDKs' `fetch` option); test
	 * doubles and offline transports depend on it. Defaults to global fetch.
	 */
	fetch?: typeof fetch;
}

export interface ResolvedAnthropicProviderSettings {
	baseUrl: string;
	apiKey?: string;
	defaultModel?: string;
	fetch?: typeof fetch;
}

export function resolveOpenAIProviderSettings(
	settings: OpenAIProviderSettings = {},
): ResolvedOpenAIProviderSettings {
	const baseUrl = validateProviderBaseUrl(
		settings.baseUrl ?? DEFAULT_OPENAI_BASE_URL,
		'OpenAI',
	);
	const apiKey = settings.apiKey?.trim();
	const defaultModel = settings.defaultModel?.trim();
	if (settings.defaultModel !== undefined && !defaultModel) {
		throw new Error('OpenAI defaultModel must not be empty when provided');
	}
	const transport = settings.transport ?? 'compatible';
	if (transport !== 'compatible' && transport !== 'sdk') {
		throw new Error("OpenAI transport must be 'compatible' or 'sdk'");
	}
	return {
		baseUrl,
		...(apiKey ? { apiKey } : {}),
		...(defaultModel ? { defaultModel } : {}),
		...(settings.transport ? { transport } : {}),
		// An injected fetch must carry every request; the WebSocket transport
		// would bypass it.
		websocket: settings.fetch ? false : (settings.websocket ?? true),
		...(settings.fetch ? { fetch: settings.fetch } : {}),
	};
}

export function resolveAnthropicProviderSettings(
	settings: AnthropicProviderSettings = {},
): ResolvedAnthropicProviderSettings {
	const baseUrl = validateProviderBaseUrl(
		settings.baseUrl ?? DEFAULT_ANTHROPIC_BASE_URL,
		'Anthropic',
	);
	const configuredCredential = settings.apiKey;
	const credential = configuredCredential?.trim();
	if (typeof configuredCredential === 'string' && !credential) {
		throw new Error('Anthropic apiKey must not be empty when provided');
	}
	const defaultModel = settings.defaultModel?.trim();
	if (settings.defaultModel !== undefined && !defaultModel) {
		throw new Error(
			'Anthropic defaultModel must not be empty when provided',
		);
	}
	return {
		baseUrl,
		...(credential ? { apiKey: credential } : {}),
		...(defaultModel ? { defaultModel } : {}),
		...(settings.fetch ? { fetch: settings.fetch } : {}),
	};
}

export async function validateOpenAIProviderSettings(
	settings: OpenAIProviderSettings,
	options: {
		checkDefaultModel?: boolean;
		clientFactory?: ModelListClientFactory;
	} = {},
): Promise<string[]> {
	const resolved = resolveOpenAIProviderSettings(settings);
	const probeWebSocket = options.clientFactory === undefined;
	let factory = options.clientFactory;
	if (!factory) {
		try {
			const { default: OpenAI } =
				await importOptionalPeer<typeof import('openai')>('openai');
			factory = (clientOptions) =>
				new OpenAI(clientOptions) as unknown as ModelListClient;
		} catch (error) {
			if (
				typeof error === 'object' &&
				error !== null &&
				'code' in error &&
				(error as { code?: unknown }).code === 'ERR_MODULE_NOT_FOUND'
			) {
				throw new Error(
					"The OpenAI SDK is optional. Install 'openai' alongside " +
						"'agent-rt' to use OpenAI SDK startup validation, or " +
						'pass clientFactory explicitly.',
				);
			}
			throw error;
		}
	}
	let modelIds: string[];
	try {
		modelIds = modelListIds(
			await factory({
				baseURL: resolved.baseUrl,
				apiKey: resolved.apiKey ?? 'not-provided',
			}).models.list(),
		);
	} catch (error) {
		throw new Error(
			`OpenAI startup validation failed: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
	if (resolved.websocket && probeWebSocket) {
		const { openAIResponsesWebSocketSupported } =
			await import('../../ext/transports/openai_websocket.js');
		setOpenAIWebSocketCapability(
			resolved.baseUrl,
			await openAIResponsesWebSocketSupported(
				resolved.baseUrl,
				resolved.apiKey,
			),
		);
	}
	if (
		options.checkDefaultModel &&
		resolved.defaultModel &&
		!modelIds.includes(resolved.defaultModel)
	) {
		throw new Error(
			`OpenAI default model ${JSON.stringify(resolved.defaultModel)} is not available`,
		);
	}
	return modelIds;
}

export async function validateAnthropicProviderSettings(
	settings: AnthropicProviderSettings,
	options: {
		checkDefaultModel?: boolean;
		clientFactory?: ModelListClientFactory;
	} = {},
): Promise<string[]> {
	const resolved = resolveAnthropicProviderSettings(settings);
	let factory = options.clientFactory;
	if (!factory) {
		try {
			const { default: Anthropic } =
				await importOptionalPeer<typeof import('@anthropic-ai/sdk')>(
					'@anthropic-ai/sdk',
				);
			factory = (clientOptions) =>
				new Anthropic(clientOptions) as unknown as ModelListClient;
		} catch (error) {
			if (
				typeof error === 'object' &&
				error !== null &&
				'code' in error &&
				(error as { code?: unknown }).code === 'ERR_MODULE_NOT_FOUND'
			) {
				throw new Error(
					"The Anthropic SDK is optional. Install '@anthropic-ai/sdk' " +
						"alongside 'agent-rt' to use Anthropic startup " +
						'validation, or pass clientFactory explicitly.',
				);
			}
			throw error;
		}
	}
	const client = factory({
		baseURL: resolved.baseUrl,
		apiKey: resolved.apiKey ?? 'not-provided',
	});
	// The optional peer range starts at 0.23 (the oldest tested release); SDKs before
	// the models API (0.39) reach `/v1/models` through the generic `get()`.
	const listable = client as Partial<ModelListClient> & {
		get?(path: string): Promise<unknown>;
	};
	if (
		typeof listable.models?.list !== 'function' &&
		typeof listable.get !== 'function'
	) {
		throw new Error(
			"Anthropic startup validation needs the models API: upgrade '@anthropic-ai/sdk' " +
				'to 0.39.0 or newer, or pass clientFactory explicitly.',
		);
	}
	let modelIds: string[];
	try {
		modelIds = modelListIds(await anthropicListModels(listable));
	} catch (error) {
		throw new Error(
			`Anthropic startup validation failed: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
	if (
		options.checkDefaultModel &&
		resolved.defaultModel &&
		!modelIds.includes(resolved.defaultModel)
	) {
		throw new Error(
			`Anthropic default model ${JSON.stringify(resolved.defaultModel)} is not available`,
		);
	}
	return modelIds;
}

export function resolveOpenAIModel(
	settings: OpenAIProviderSettings,
	requestedModel?: string,
): string {
	const resolved = resolveOpenAIProviderSettings(settings);
	const model = requestedModel?.trim() || resolved.defaultModel;
	if (!model)
		throw new Error(
			'OpenAI model is required when no defaultModel is configured',
		);
	return model;
}

export function resolveAnthropicModel(
	settings: AnthropicProviderSettings,
	requestedModel?: string,
): string {
	const resolved = resolveAnthropicProviderSettings(settings);
	const model = requestedModel?.trim() || resolved.defaultModel;
	if (!model)
		throw new Error(
			'Anthropic model is required when no defaultModel is configured',
		);
	return model;
}

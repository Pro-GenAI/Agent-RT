/** Provider defaults, environment names, and SDK compatibility helpers. */
import { runtimeSettingsMap } from '../settings.js';

export const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';

export const OPENAI_BASE_URL_ENV = 'OPENAI_BASE_URL';

export const OPENAI_API_KEY_ENV = 'OPENAI_' + 'API_KEY';

export const OPENAI_MODEL_ENV = 'OPENAI_MODEL';

export const DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com';

export const ANTHROPIC_BASE_URL_ENV = 'ANTHROPIC_BASE_URL';

export const ANTHROPIC_API_KEY_ENV = 'ANTHROPIC_' + 'API_KEY';

export const ANTHROPIC_MODEL_ENV = 'ANTHROPIC_MODEL';

export type ModelListClient = {
	models: { list(): Promise<unknown> };
};

export type ModelListClientFactory = (options: {
	baseURL: string;
	apiKey: string;
}) => ModelListClient;

export function currentProviderEnvironment(): Readonly<
	Record<string, string | undefined>
> {
	return runtimeSettingsMap;
}

export function validateProviderBaseUrl(
	value: string,
	provider: string,
): string {
	const baseUrl = value.trim().replace(/\/+$/, '');
	if (!baseUrl) throw new Error(`${provider} baseUrl must not be empty`);
	let parsed: URL;
	try {
		parsed = new URL(baseUrl);
	} catch {
		throw new Error(`${provider} baseUrl must be an absolute http(s) URL`);
	}
	if (
		(parsed.protocol !== 'http:' && parsed.protocol !== 'https:') ||
		!parsed.host
	) {
		throw new Error(`${provider} baseUrl must be an absolute http(s) URL`);
	}
	if (parsed.search || parsed.hash) {
		throw new Error(
			`${provider} baseUrl must not include query parameters or fragments`,
		);
	}
	return baseUrl;
}

const openAIWebSocketCapability = new Map<string, boolean>();

export function cachedOpenAIWebSocketCapability(
	baseUrl: string,
): boolean | undefined {
	return openAIWebSocketCapability.get(baseUrl);
}

export function setOpenAIWebSocketCapability(
	baseUrl: string,
	supported: boolean,
): void {
	openAIWebSocketCapability.set(baseUrl, supported);
}

export function modelListIds(page: unknown): string[] {
	const data =
		typeof page === 'object' && page !== null
			? (page as { data?: unknown }).data
			: undefined;
	if (!Array.isArray(data)) {
		throw new Error('model listing response must contain a data array');
	}
	return data.flatMap((item) => {
		if (typeof item !== 'object' || item === null) return [];
		const id = (item as { id?: unknown }).id;
		return typeof id === 'string' ? [id] : [];
	});
}

export type OpenAICompletionClient = {
	chat: {
		completions: {
			create(
				params: Record<string, unknown>,
				options?: { signal?: AbortSignal },
			): Promise<unknown>;
		};
	};
	embeddings?: {
		create(params: Record<string, unknown>): Promise<unknown>;
	};
	models?: {
		list(): Promise<unknown>;
		retrieve(model: string): Promise<unknown>;
	};
	batches?: {
		create(params: Record<string, unknown>): Promise<unknown>;
		retrieve(id: string): Promise<unknown>;
		list(params?: Record<string, unknown>): Promise<unknown>;
		cancel(id: string): Promise<unknown>;
	};
};

type AnthropicBatchesResource = {
	create(params: Record<string, unknown>): Promise<unknown>;
	retrieve(id: string): Promise<unknown>;
	list(params?: Record<string, unknown>): Promise<unknown>;
	cancel(id: string): Promise<unknown>;
	results(id: string): Promise<unknown>;
};

export type AnthropicCompletionClient = {
	messages: {
		create(params: Record<string, unknown>): Promise<unknown>;
		countTokens?(params: Record<string, unknown>): Promise<unknown>;
		batches?: AnthropicBatchesResource;
	};
	// SDKs 0.28-0.38 expose batches only under the beta namespace.
	beta?: { messages?: { batches?: AnthropicBatchesResource } };
	models?: {
		list(): Promise<unknown>;
		retrieve(model: string): Promise<unknown>;
	};
	// Generic request helpers every SDK release ships; they reach endpoints
	// that older SDKs have no typed resource for.
	get?(path: string): Promise<unknown>;
	post?(path: string, options: { body: unknown }): Promise<unknown>;
};

export function anthropicBatches(
	client: AnthropicCompletionClient,
): AnthropicBatchesResource | undefined {
	return client.messages.batches ?? client.beta?.messages?.batches;
}

/** List models, or fetch `/v1/models` on SDKs that predate the models API (0.39). */
export async function anthropicListModels(client: {
	models?: { list(): Promise<unknown> };
	get?(path: string): Promise<unknown>;
}): Promise<unknown> {
	if (typeof client.models?.list === 'function') return client.models.list();
	if (typeof client.get === 'function') return client.get('/v1/models');
	throw new TypeError(
		'Anthropic provider client does not support models.list()',
	);
}

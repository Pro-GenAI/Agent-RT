/**
 * `@openrouter/sdk` `OpenRouter` over the Agent RT OpenAI shim.
 *
 * `chat.send()` takes the SDK's camelCase request (wrapped in `chatRequest`
 * as in SDK 1.x, or flat as in older releases) and returns its camelCase
 * result (`choices[0].message.toolCalls`, `usage.promptTokens`); streams
 * yield camelCase chunks. Requests go to OpenRouter's OpenAI-compatible
 * endpoint through an Agent RT provider; the key comes from `apiKey` or
 * `OPENROUTER_API_KEY` and never falls back to OpenAI's.
 */
import { type ModelProvider, OpenAIModelProvider } from '../../index.js';
import { sdkEnv } from './migration_tools.js';
import { OpenAI } from './openai.js';

type JSONObject = Record<string, unknown>;

export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

// JSON Schema and free-form values keep the caller's own keys.
const RAW_KEYS = new Set(['parameters', 'schema', 'metadata']);
const CHAT_OPTIONS = new Set([
	'model',
	'messages',
	'temperature',
	'max_tokens',
	'max_completion_tokens',
	'tools',
	'response_format',
	'reasoning_effort',
	'stream',
]);

function convertKeys(value: unknown, key: (name: string) => string): unknown {
	if (Array.isArray(value))
		return value.map((item) => convertKeys(item, key));
	if (typeof value !== 'object' || value === null) return value;
	return Object.fromEntries(
		Object.entries(value).map(([name, item]) => [
			key(name),
			RAW_KEYS.has(name) ? item : convertKeys(item, key),
		]),
	);
}

const snake = (name: string) =>
	name.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
const camel = (name: string) =>
	name.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

async function* camelStream(stream: AsyncIterable<unknown>) {
	for await (const chunk of stream) yield convertKeys(chunk, camel);
}

export type OpenRouterOptions = {
	apiKey?: string;
	serverURL?: string;
	provider?: ModelProvider;
	fetch?: typeof fetch;
	// SDK attribution/transport options are accepted; the Agent RT provider owns the transport.
	httpReferer?: string;
	appTitle?: string;
	timeoutMs?: number;
	[option: string]: unknown;
};

export class OpenRouter {
	readonly provider: ModelProvider;
	readonly chat: {
		send(request: JSONObject, options?: unknown): Promise<unknown>;
	};
	readonly embeddings: {
		generate(request: JSONObject, options?: unknown): Promise<unknown>;
	};

	constructor(options: OpenRouterOptions = {}) {
		if (
			options.apiKey !== undefined &&
			typeof options.apiKey !== 'string'
		) {
			throw new TypeError(
				'agent-rt/@openrouter/sdk accepts a string apiKey',
			);
		}
		this.provider =
			options.provider ??
			new OpenAIModelProvider({
				baseUrl: options.serverURL ?? OPENROUTER_BASE_URL,
				apiKey: options.apiKey ?? sdkEnv('OPENROUTER_API_KEY'),
				websocket: false,
				...(options.fetch ? { fetch: options.fetch } : {}),
			});
		const client = new OpenAI({ provider: this.provider });
		const body = (request: JSONObject, wrapper: string) => {
			if (wrapper in request) {
				const extras = Object.keys(request).filter(
					(name) => name !== wrapper,
				);
				if (extras.length)
					throw new Error(
						`OpenRouter request does not support wrapper options: ${extras.join(', ')}`,
					);
			}
			const payload = request[wrapper] ?? request;
			if (
				!payload ||
				typeof payload !== 'object' ||
				Array.isArray(payload)
			)
				throw new TypeError(
					'OpenRouter request body must be an object',
				);
			return convertKeys(payload, snake) as JSONObject;
		};
		this.chat = {
			send: async (request, options) => {
				if (options !== undefined)
					throw new Error(
						'OpenRouter chat.send does not support per-call SDK options',
					);
				const params = body(
					request,
					'chatRequest' in request
						? 'chatRequest'
						: 'chatGenerationParams',
				);
				const unsupported = Object.keys(params).filter(
					(name) => !CHAT_OPTIONS.has(name),
				);
				if (unsupported.length)
					throw new Error(
						`OpenRouter chat.send cannot honor request options: ${unsupported.sort().join(', ')}`,
					);
				const result = await client.chat.completions.create(
					params as never,
				);
				return params.stream
					? camelStream(result as AsyncIterable<unknown>)
					: convertKeys(result, camel);
			},
		};
		this.embeddings = {
			generate: async (request) =>
				convertKeys(
					await client.embeddings.create(
						body(request, 'requestBody'),
					),
					camel,
				),
		};
	}
}

export default OpenRouter;

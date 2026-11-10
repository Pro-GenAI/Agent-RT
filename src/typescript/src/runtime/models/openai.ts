/** The OpenAI model provider. */
import { type ModelMessage } from '../types.js';
import { type ToolDefinition } from '../tools/definitions.js';
import {
	type BatchJob,
	type EmbeddingItem,
	type EmbeddingRequest,
	type EmbeddingResponse,
	type FinishReason,
	type ModelCatalogEntry,
	type ModelRequest,
	type ModelResponse,
	type ModelStreamEvent,
	type ModelUsage,
	type StreamingModelProvider,
	type StructuredOutputRequirement,
} from './interfaces.js';
import {
	type OpenAICompletionClient,
	cachedOpenAIWebSocketCapability,
	setOpenAIWebSocketCapability,
} from './sdk.js';
import {
	type OpenAIProviderSettings,
	type ResolvedOpenAIProviderSettings,
	resolveOpenAIModel,
	resolveOpenAIProviderSettings,
} from './provider_settings.js';
import {
	normalizeFinishReason,
	numberValue,
	providerToolCall,
	providerValue,
} from './conversion.js';
import {
	openAIMessage,
	openAIParams,
	openAIResponse,
	openAIResponseFormat,
	openAIToolParams,
	openAIUsage,
} from './openai_payloads.js';
import {
	OpenAIRateLimitGate,
	batchData,
	batchJob,
	modelCatalogData,
	modelCatalogEntry,
	openAICompatibleFetchClient,
	openAIResponseHeaders,
} from './openai_http.js';
import { importOptionalPeer } from '../platform.js';

export class OpenAIModelProvider implements StreamingModelProvider {
	readonly name = 'openai';
	readonly settings: ResolvedOpenAIProviderSettings;
	private client?: OpenAICompletionClient;
	private clientPromise?: Promise<OpenAICompletionClient>;
	private ownsClient = false;
	private cachedToolDefinitions?: ToolDefinition[];
	private cachedToolPayload?: Record<string, unknown>[];
	private cachedStructuredOutput?: StructuredOutputRequirement;
	private cachedResponseFormat?: Record<string, unknown>;
	private readonly messageConversionCache = new WeakMap<
		ModelMessage,
		Record<string, unknown>
	>();
	private websocketSession?: import('../../ext/transports/openai_websocket.js').OpenAIResponsesWebSocketSession;
	private websocketEnabled: boolean;
	private readonly rateLimitGate = new OpenAIRateLimitGate();

	constructor(
		settings: OpenAIProviderSettings,
		client?: OpenAICompletionClient,
	) {
		this.settings = resolveOpenAIProviderSettings(settings);
		this.client = client;
		const capability = cachedOpenAIWebSocketCapability(
			this.settings.baseUrl,
		);
		this.websocketEnabled =
			this.settings.websocket &&
			capability !== false &&
			client === undefined;
	}

	private async getClient(): Promise<OpenAICompletionClient> {
		if (this.client) return this.client;
		if (!this.clientPromise) {
			this.clientPromise = (async () => {
				if (
					(this.settings.transport ?? 'compatible') === 'compatible'
				) {
					this.ownsClient = true;
					return openAICompatibleFetchClient(this.settings);
				}
				let OpenAI: new (options: Record<string, unknown>) => unknown;
				try {
					({ default: OpenAI } =
						await importOptionalPeer<typeof import('openai')>(
							'openai',
						));
				} catch (error) {
					if (
						typeof error === 'object' &&
						error !== null &&
						'code' in error &&
						(error as { code?: unknown }).code ===
							'ERR_MODULE_NOT_FOUND'
					) {
						throw new Error(
							"The OpenAI SDK is optional. Install 'openai' alongside " +
								"'agent-rt' to use transport='sdk', or use the " +
								'default compatible transport.',
						);
					}
					throw error;
				}
				const client = new OpenAI({
					...(this.settings.fetch
						? { fetch: this.settings.fetch }
						: {}),
					baseURL: this.settings.baseUrl,
					apiKey: this.settings.apiKey ?? 'not-provided',
				}) as unknown as OpenAICompletionClient;
				this.ownsClient = true;
				return client;
			})();
		}
		try {
			this.client = await this.clientPromise;
			return this.client;
		} finally {
			this.clientPromise = undefined;
		}
	}

	private async disableWebSocket(): Promise<void> {
		this.websocketEnabled = false;
		setOpenAIWebSocketCapability(this.settings.baseUrl, false);
		if (this.websocketSession) {
			await this.websocketSession.close();
			this.websocketSession = undefined;
		}
	}

	async createBatch(params: Record<string, unknown>): Promise<BatchJob> {
		const client = await this.getClient();
		const create = client.batches?.create;
		if (!create)
			throw new TypeError(
				'OpenAI provider client does not support batches.create()',
			);
		return batchJob(await create.call(client.batches, params));
	}

	async retrieveBatch(id: string): Promise<BatchJob> {
		const client = await this.getClient();
		const retrieve = client.batches?.retrieve;
		if (!retrieve)
			throw new TypeError(
				'OpenAI provider client does not support batches.retrieve()',
			);
		return batchJob(await retrieve.call(client.batches, id));
	}

	async listBatches(
		params: Record<string, unknown> = {},
	): Promise<BatchJob[]> {
		const client = await this.getClient();
		const list = client.batches?.list;
		if (!list)
			throw new TypeError(
				'OpenAI provider client does not support batches.list()',
			);
		const raw = await list.call(client.batches, params);
		return batchData(raw).map(batchJob);
	}

	async cancelBatch(id: string): Promise<BatchJob> {
		const client = await this.getClient();
		const cancel = client.batches?.cancel;
		if (!cancel)
			throw new TypeError(
				'OpenAI provider client does not support batches.cancel()',
			);
		return batchJob(await cancel.call(client.batches, id));
	}

	async listModels(): Promise<ModelCatalogEntry[]> {
		const client = await this.getClient();
		const list = client.models?.list;
		if (!list)
			throw new TypeError(
				'OpenAI provider client does not support models.list()',
			);
		const raw = await list.call(client.models);
		return modelCatalogData(raw).map((item) => modelCatalogEntry(item));
	}

	async retrieveModel(model: string): Promise<ModelCatalogEntry> {
		const client = await this.getClient();
		const retrieve = client.models?.retrieve;
		if (!retrieve) {
			throw new TypeError(
				'OpenAI provider client does not support models.retrieve()',
			);
		}
		return modelCatalogEntry(await retrieve.call(client.models, model));
	}

	async embed(request: EmbeddingRequest): Promise<EmbeddingResponse> {
		const client = await this.getClient();
		const create = client.embeddings?.create;
		if (!create) {
			throw new TypeError(
				'OpenAI provider client does not support embeddings.create()',
			);
		}
		const model = resolveOpenAIModel(this.settings, request.model);
		const params: Record<string, unknown> = { model, input: request.input };
		if (request.dimensions !== undefined)
			params.dimensions = request.dimensions;
		if (request.encodingFormat !== undefined)
			params.encoding_format = request.encodingFormat;
		const raw = await create.call(client.embeddings, params);
		const rawData = providerValue(raw, 'data');
		if (!Array.isArray(rawData)) {
			throw new Error('OpenAI embedding response did not contain data');
		}
		const data = rawData.map((item, index): EmbeddingItem => {
			const value = providerValue(item, 'embedding');
			if (typeof value === 'string') {
				return {
					index: numberValue(providerValue(item, 'index')) ?? index,
					embedding: value,
				};
			}
			if (
				!Array.isArray(value) ||
				!value.every((part) => typeof part === 'number')
			) {
				throw new Error(
					'embedding response must contain a numeric vector or base64 string',
				);
			}
			return {
				index: numberValue(providerValue(item, 'index')) ?? index,
				embedding: value as number[],
			};
		});
		const usageValue = providerValue(raw, 'usage');
		const inputTokens = numberValue(
			providerValue(usageValue, 'prompt_tokens'),
		);
		const totalTokens =
			numberValue(providerValue(usageValue, 'total_tokens')) ??
			inputTokens;
		const modelValue = providerValue(raw, 'model');
		return {
			data,
			model: typeof modelValue === 'string' ? modelValue : model,
			usage:
				usageValue === undefined
					? undefined
					: { inputTokens, totalTokens },
			raw,
		};
	}

	async close(): Promise<void> {
		if (this.websocketSession) {
			await this.websocketSession.close();
			this.websocketSession = undefined;
		}
		const client = this.client;
		if (!client || !this.ownsClient) return;
		const close = (
			client as unknown as { close?: () => void | Promise<void> }
		).close;
		if (close) await close.call(client);
		if (this.client === client) {
			this.client = undefined;
			this.ownsClient = false;
		}
	}

	private cachedMessages(
		messages: ModelMessage[],
	): Record<string, unknown>[] {
		return messages.map((message) => {
			const cached = this.messageConversionCache.get(message);
			if (cached) return cached;
			const converted = openAIMessage(message);
			this.messageConversionCache.set(message, converted);
			return converted;
		});
	}

	private cachedParams(request: ModelRequest): Record<string, unknown> {
		const messagesPayload = this.cachedMessages(request.messages);
		let toolsPayload: Record<string, unknown>[] | undefined;
		const tools = request.tools;
		if (tools?.length) {
			const cached = this.cachedToolDefinitions;
			const matches =
				cached !== undefined &&
				cached.length === tools.length &&
				cached.every((tool, index) => tool === tools[index]);
			if (!matches) {
				this.cachedToolDefinitions = [...tools];
				this.cachedToolPayload = openAIToolParams(tools);
			}
			toolsPayload = this.cachedToolPayload;
		} else {
			this.cachedToolDefinitions = undefined;
			this.cachedToolPayload = undefined;
		}

		let responseFormat: Record<string, unknown> | undefined;
		const structured = request.structuredOutput;
		if (structured) {
			if (this.cachedStructuredOutput !== structured) {
				this.cachedStructuredOutput = structured;
				this.cachedResponseFormat = openAIResponseFormat(structured);
			}
			responseFormat = this.cachedResponseFormat;
		} else {
			this.cachedStructuredOutput = undefined;
			this.cachedResponseFormat = undefined;
		}

		return openAIParams(
			this.settings,
			request,
			messagesPayload,
			toolsPayload,
			responseFormat,
		);
	}

	private async getWebSocketSession(): Promise<
		import('../../ext/transports/openai_websocket.js').OpenAIResponsesWebSocketSession
	> {
		if (!this.websocketSession) {
			const { OpenAIResponsesWebSocketSession } =
				await import('../../ext/transports/openai_websocket.js');
			const auth = (this.settings as unknown as Record<string, unknown>)[
				['api', 'Key'].join('')
			];
			this.websocketSession = new OpenAIResponsesWebSocketSession(
				this.settings,
				typeof auth === 'string' ? auth : undefined,
			);
		}
		return this.websocketSession;
	}

	async complete(request: ModelRequest): Promise<ModelResponse> {
		const model = resolveOpenAIModel(this.settings, request.model);
		await this.rateLimitGate.wait(model, request.signal);
		if (this.websocketEnabled) {
			const { OpenAIWebSocketUnavailableError } =
				await import('../../ext/transports/openai_websocket.js');
			try {
				let completed: ModelResponse | undefined;
				for await (const event of (
					await this.getWebSocketSession()
				).stream(request)) {
					if (event.type === 'completed') completed = event.response;
				}
				if (!completed)
					throw new Error(
						'OpenAI WebSocket did not produce a completed response',
					);
				return completed;
			} catch (error) {
				if (!(error instanceof OpenAIWebSocketUnavailableError))
					throw error;
				await this.disableWebSocket();
			}
		}
		const client = await this.getClient();
		try {
			const pending = client.chat.completions.create(
				this.cachedParams(request),
				request.signal ? { signal: request.signal } : undefined,
			) as Promise<unknown> & {
				withResponse?: () => Promise<{
					data: unknown;
					response: Response;
				}>;
			};
			if (typeof pending.withResponse === 'function') {
				const wrapped = await pending.withResponse();
				this.rateLimitGate.update(
					model,
					Object.fromEntries(wrapped.response.headers.entries()),
				);
				return openAIResponse(wrapped.data);
			}
			const raw = await pending;
			this.rateLimitGate.update(model, openAIResponseHeaders(raw));
			return openAIResponse(raw);
		} catch (error) {
			let headers: Record<string, string> = {};
			if (typeof error === 'object' && error !== null) {
				if ('responseHeaders' in error) {
					headers =
						(error as { responseHeaders?: Record<string, string> })
							.responseHeaders ?? {};
				} else if ('response' in error) {
					const response = (error as { response?: Response })
						.response;
					if (response?.headers)
						headers = Object.fromEntries(
							response.headers.entries(),
						);
				}
			}
			this.rateLimitGate.update(model, headers);
			throw error;
		}
	}

	async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
		const modelKey = resolveOpenAIModel(this.settings, request.model);
		await this.rateLimitGate.wait(modelKey, request.signal);
		if (this.websocketEnabled) {
			const { OpenAIWebSocketUnavailableError } =
				await import('../../ext/transports/openai_websocket.js');
			try {
				for await (const event of (
					await this.getWebSocketSession()
				).stream(request)) {
					yield event;
				}
				return;
			} catch (error) {
				if (!(error instanceof OpenAIWebSocketUnavailableError))
					throw error;
				await this.disableWebSocket();
			}
		}
		const params = this.cachedParams(request);
		params.stream = true;
		// Chat Completions only reports token usage on streams when asked;
		// without it token budgets and cost ledgers never see usage.
		params.stream_options = { include_usage: true };
		const client = await this.getClient();
		const rawStream = await client.chat.completions.create(
			params,
			request.signal ? { signal: request.signal } : undefined,
		);
		if (
			typeof rawStream !== 'object' ||
			rawStream === null ||
			!(Symbol.asyncIterator in rawStream)
		) {
			throw new Error('OpenAI streaming response is not async iterable');
		}
		this.rateLimitGate.update(modelKey, openAIResponseHeaders(rawStream));
		const textParts: string[] = [];
		const toolState = new Map<
			number,
			{ id: string; name: string; arguments: string }
		>();
		let model: string | undefined;
		let finishReason: FinishReason | undefined;
		let usage: ModelUsage | undefined;
		let lastRaw: unknown;
		for await (const chunk of rawStream as AsyncIterable<unknown>) {
			lastRaw = chunk;
			const chunkModel = providerValue(chunk, 'model');
			if (typeof chunkModel === 'string') model = chunkModel;
			usage = openAIUsage(providerValue(chunk, 'usage')) ?? usage;
			const choices = providerValue(chunk, 'choices');
			if (!Array.isArray(choices) || choices.length === 0) continue;
			const choice = choices[0];
			finishReason =
				normalizeFinishReason(providerValue(choice, 'finish_reason')) ??
				finishReason;
			const delta = providerValue(choice, 'delta');
			const text = providerValue(delta, 'content');
			if (typeof text === 'string' && text) {
				textParts.push(text);
				yield { type: 'text_delta', text, raw: chunk };
			}
			const rawCalls = providerValue(delta, 'tool_calls');
			if (Array.isArray(rawCalls)) {
				for (const call of rawCalls) {
					const indexValue = providerValue(call, 'index');
					const index =
						typeof indexValue === 'number' ? indexValue : 0;
					const state = toolState.get(index) ?? {
						id: '',
						name: '',
						arguments: '',
					};
					const id = providerValue(call, 'id');
					if (typeof id === 'string') state.id = id;
					const fn = providerValue(call, 'function');
					const name = providerValue(fn, 'name');
					const argumentsDelta = providerValue(fn, 'arguments');
					if (typeof name === 'string') state.name = name;
					if (typeof argumentsDelta === 'string')
						state.arguments += argumentsDelta;
					toolState.set(index, state);
					yield {
						type: 'tool_call_delta',
						...(state.id ? { toolCallId: state.id } : {}),
						...(state.name ? { toolName: state.name } : {}),
						...(typeof argumentsDelta === 'string'
							? { argumentsDelta }
							: {}),
						raw: chunk,
					};
				}
			}
		}
		const toolCalls = [...toolState.entries()]
			.sort(([a], [b]) => a - b)
			.map(([, state]) =>
				providerToolCall(state.id, state.name, state.arguments),
			);
		const response: ModelResponse = {
			message: {
				role: 'assistant',
				content: textParts.length
					? [{ type: 'text', text: textParts.join('') }]
					: [],
				...(toolCalls.length ? { toolCalls } : {}),
			},
			model,
			usage,
			finishReason,
			raw: lastRaw,
		};
		yield { type: 'completed', response, raw: lastRaw };
	}
}

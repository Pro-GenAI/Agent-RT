/** The Anthropic model provider. */
import { type ModelMessage } from '../types.js';
import {
	type BatchJob,
	type FinishReason,
	type ModelCatalogEntry,
	type ModelRequest,
	type ModelResponse,
	type ModelStreamEvent,
	type StreamingModelProvider,
} from './interfaces.js';
import {
	type AnthropicCompletionClient,
	anthropicBatches,
	anthropicListModels,
} from './sdk.js';
import {
	type AnthropicProviderSettings,
	type ResolvedAnthropicProviderSettings,
	resolveAnthropicProviderSettings,
} from './provider_settings.js';
import {
	normalizeFinishReason,
	numberValue,
	providerToolCall,
	providerValue,
	thinkingPart,
} from './conversion.js';
import {
	batchData,
	batchJob,
	modelCatalogData,
	modelCatalogEntry,
} from './openai_http.js';
import {
	type AnthropicConvertedMessage,
	anthropicMessage,
	anthropicParams,
	anthropicResponse,
} from './anthropic_payloads.js';
import { importOptionalPeer } from '../platform.js';

export class AnthropicModelProvider implements StreamingModelProvider {
	readonly name = 'anthropic';
	readonly settings: ResolvedAnthropicProviderSettings;
	private client?: AnthropicCompletionClient;
	private clientPromise?: Promise<AnthropicCompletionClient>;
	private ownsClient = false;
	private readonly messageConversionCache = new WeakMap<
		ModelMessage,
		AnthropicConvertedMessage
	>();

	constructor(
		settings: AnthropicProviderSettings,
		client?: AnthropicCompletionClient,
	) {
		this.settings = resolveAnthropicProviderSettings(settings);
		this.client = client;
	}

	private async getClient(): Promise<AnthropicCompletionClient> {
		if (this.client) return this.client;
		if (!this.clientPromise) {
			this.clientPromise = (async () => {
				let Anthropic: new (
					options: Record<string, unknown>,
				) => unknown;
				try {
					({ default: Anthropic } =
						await importOptionalPeer<
							typeof import('@anthropic-ai/sdk')
						>('@anthropic-ai/sdk'));
				} catch (error) {
					if (
						typeof error === 'object' &&
						error !== null &&
						'code' in error &&
						(error as { code?: unknown }).code ===
							'ERR_MODULE_NOT_FOUND'
					) {
						throw new Error(
							"The Anthropic SDK is optional. Install '@anthropic-ai/sdk' " +
								"alongside 'agent-rt' to use AnthropicModelProvider " +
								'without an injected client.',
						);
					}
					throw error;
				}
				const client = new Anthropic({
					...(this.settings.fetch
						? { fetch: this.settings.fetch }
						: {}),
					baseURL: this.settings.baseUrl,
					apiKey: this.settings.apiKey ?? 'not-provided',
				}) as unknown as AnthropicCompletionClient;
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

	async close(): Promise<void> {
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
	): AnthropicConvertedMessage[] {
		return messages.map((message) => {
			const cached = this.messageConversionCache.get(message);
			if (cached) return cached;
			const converted = anthropicMessage(message);
			this.messageConversionCache.set(message, converted);
			return converted;
		});
	}

	private cachedParams(request: ModelRequest): Record<string, unknown> {
		return anthropicParams(
			this.settings,
			request,
			this.cachedMessages(request.messages),
		);
	}

	async createBatch(params: Record<string, unknown>): Promise<BatchJob> {
		const client = await this.getClient();
		const create = anthropicBatches(client)?.create;
		if (!create)
			throw new TypeError(
				'Anthropic provider client does not support messages.batches.create()',
			);
		return batchJob(await create.call(anthropicBatches(client), params));
	}

	async retrieveBatch(id: string): Promise<BatchJob> {
		const client = await this.getClient();
		const retrieve = anthropicBatches(client)?.retrieve;
		if (!retrieve)
			throw new TypeError(
				'Anthropic provider client does not support messages.batches.retrieve()',
			);
		return batchJob(await retrieve.call(anthropicBatches(client), id));
	}

	async listBatches(
		params: Record<string, unknown> = {},
	): Promise<BatchJob[]> {
		const client = await this.getClient();
		const list = anthropicBatches(client)?.list;
		if (!list)
			throw new TypeError(
				'Anthropic provider client does not support messages.batches.list()',
			);
		const raw = await list.call(anthropicBatches(client), params);
		return batchData(raw).map(batchJob);
	}

	async cancelBatch(id: string): Promise<BatchJob> {
		const client = await this.getClient();
		const cancel = anthropicBatches(client)?.cancel;
		if (!cancel)
			throw new TypeError(
				'Anthropic provider client does not support messages.batches.cancel()',
			);
		return batchJob(await cancel.call(anthropicBatches(client), id));
	}

	async batchResults(id: string): Promise<unknown> {
		const client = await this.getClient();
		const results = anthropicBatches(client)?.results;
		if (!results)
			throw new TypeError(
				'Anthropic provider client does not support messages.batches.results()',
			);
		return results.call(anthropicBatches(client), id);
	}

	async listModels(): Promise<ModelCatalogEntry[]> {
		const client = await this.getClient();
		const raw = await anthropicListModels(client);
		return modelCatalogData(raw).map((item) =>
			modelCatalogEntry(item, true),
		);
	}

	async retrieveModel(model: string): Promise<ModelCatalogEntry> {
		const client = await this.getClient();
		const retrieve = client.models?.retrieve;
		if (!retrieve) {
			throw new TypeError(
				'Anthropic provider client does not support models.retrieve()',
			);
		}
		return modelCatalogEntry(
			await retrieve.call(client.models, model),
			true,
		);
	}

	async countTokens(request: ModelRequest): Promise<number> {
		const client = await this.getClient();
		const countTokens = client.messages.countTokens;
		// SDKs before 0.39 lack messages.countTokens(); the generic request
		// helper reaches the same endpoint.
		if (!countTokens && typeof client.post !== 'function') {
			throw new TypeError(
				'Anthropic provider client does not support messages.countTokens()',
			);
		}
		const params = this.cachedParams(request);
		delete params.max_tokens;
		delete params.temperature;
		delete params.output_config;
		const raw = countTokens
			? await countTokens.call(client.messages, params)
			: await client.post?.('/v1/messages/count_tokens', {
					body: params,
				});
		const inputTokens = numberValue(providerValue(raw, 'input_tokens'));
		if (inputTokens === undefined) {
			throw new Error(
				'Anthropic token count response did not contain input_tokens',
			);
		}
		return inputTokens;
	}

	async complete(request: ModelRequest): Promise<ModelResponse> {
		const client = await this.getClient();
		return anthropicResponse(
			await client.messages.create(this.cachedParams(request)),
		);
	}

	async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
		const params = this.cachedParams(request);
		params.stream = true;
		const client = await this.getClient();
		const rawStream = await client.messages.create(params);
		if (
			typeof rawStream !== 'object' ||
			rawStream === null ||
			!(Symbol.asyncIterator in rawStream)
		) {
			throw new Error(
				'Anthropic streaming response is not async iterable',
			);
		}
		const textParts: string[] = [];
		const thinkingState = new Map<
			number,
			{ type: string; thinking: string; signature: string; data: string }
		>();
		const toolState = new Map<
			number,
			{ id: string; name: string; arguments: string }
		>();
		let model = params.model as string;
		let finishReason: FinishReason | undefined;
		let inputTokens: number | undefined;
		let outputTokens: number | undefined;
		let lastRaw: unknown;
		for await (const event of rawStream as AsyncIterable<unknown>) {
			lastRaw = event;
			const eventType = providerValue(event, 'type');
			if (eventType === 'message_start') {
				const message = providerValue(event, 'message');
				const modelValue = providerValue(message, 'model');
				if (typeof modelValue === 'string') model = modelValue;
				inputTokens =
					numberValue(
						providerValue(
							providerValue(message, 'usage'),
							'input_tokens',
						),
					) ?? inputTokens;
			} else if (eventType === 'content_block_start') {
				const indexValue = providerValue(event, 'index');
				const index = typeof indexValue === 'number' ? indexValue : 0;
				const block = providerValue(event, 'content_block');
				const blockType = providerValue(block, 'type');
				if (blockType === 'text') {
					const text = providerValue(block, 'text');
					if (typeof text === 'string' && text) {
						textParts.push(text);
						yield { type: 'text_delta', text, raw: event };
					}
				} else if (
					blockType === 'thinking' ||
					blockType === 'redacted_thinking'
				) {
					thinkingState.set(index, {
						type: blockType,
						thinking: String(
							providerValue(block, 'thinking') ?? '',
						),
						signature: String(
							providerValue(block, 'signature') ?? '',
						),
						data: String(providerValue(block, 'data') ?? ''),
					});
				} else if (blockType === 'tool_use') {
					const initial = providerValue(block, 'input');
					const initialObject =
						typeof initial === 'object' &&
						initial !== null &&
						!Array.isArray(initial)
							? (initial as Record<string, unknown>)
							: undefined;
					toolState.set(index, {
						id: String(providerValue(block, 'id') ?? ''),
						name: String(providerValue(block, 'name') ?? ''),
						arguments:
							initialObject && Object.keys(initialObject).length
								? JSON.stringify(initialObject)
								: '',
					});
				}
			} else if (eventType === 'content_block_delta') {
				const indexValue = providerValue(event, 'index');
				const index = typeof indexValue === 'number' ? indexValue : 0;
				const delta = providerValue(event, 'delta');
				const deltaType = providerValue(delta, 'type');
				if (deltaType === 'text_delta') {
					const text = providerValue(delta, 'text');
					if (typeof text === 'string' && text) {
						textParts.push(text);
						yield { type: 'text_delta', text, raw: event };
					}
				} else if (deltaType === 'thinking_delta') {
					const chunk = providerValue(delta, 'thinking');
					const state = thinkingState.get(index) ?? {
						type: 'thinking',
						thinking: '',
						signature: '',
						data: '',
					};
					thinkingState.set(index, state);
					if (typeof chunk === 'string' && chunk) {
						state.thinking += chunk;
						yield {
							type: 'reasoning_delta',
							text: chunk,
							raw: event,
						};
					}
				} else if (deltaType === 'signature_delta') {
					const signature = providerValue(delta, 'signature');
					const state = thinkingState.get(index) ?? {
						type: 'thinking',
						thinking: '',
						signature: '',
						data: '',
					};
					thinkingState.set(index, state);
					if (typeof signature === 'string')
						state.signature += signature;
				} else if (deltaType === 'input_json_delta') {
					const partial = providerValue(delta, 'partial_json');
					const state = toolState.get(index) ?? {
						id: '',
						name: '',
						arguments: '',
					};
					if (typeof partial === 'string') state.arguments += partial;
					toolState.set(index, state);
					yield {
						type: 'tool_call_delta',
						...(state.id ? { toolCallId: state.id } : {}),
						...(state.name ? { toolName: state.name } : {}),
						...(typeof partial === 'string'
							? { argumentsDelta: partial }
							: {}),
						raw: event,
					};
				}
			} else if (eventType === 'message_delta') {
				finishReason =
					normalizeFinishReason(
						providerValue(
							providerValue(event, 'delta'),
							'stop_reason',
						),
						true,
					) ?? finishReason;
				outputTokens =
					numberValue(
						providerValue(
							providerValue(event, 'usage'),
							'output_tokens',
						),
					) ?? outputTokens;
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
				content: [
					...[...thinkingState.entries()]
						.sort(([a], [b]) => a - b)
						.map(([, state]) => thinkingPart(state, state.type)),
					...(textParts.length
						? [{ type: 'text' as const, text: textParts.join('') }]
						: []),
				],
				...(toolCalls.length ? { toolCalls } : {}),
			},
			model,
			usage: {
				inputTokens,
				outputTokens,
				totalTokens:
					Number.isFinite(inputTokens) &&
					Number.isFinite(outputTokens)
						? (inputTokens as number) + (outputTokens as number)
						: undefined,
			},
			finishReason,
			raw: lastRaw,
		};
		yield { type: 'completed', response, raw: lastRaw };
	}
}

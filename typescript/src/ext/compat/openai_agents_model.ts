/**
 * OpenAI Agents SDK model-interface bridges for the `@openai/agents` shim.
 *
 * The SDK talks to models through `Model.getResponse()` /
 * `getStreamedResponse()` with its own request and item shapes. These helpers
 * translate both ways, so an SDK model (a test double, `aisdk(...)`, or a
 * custom `ModelProvider.getModel()`) can drive the Agent RT `AgentLoop`, and an
 * Agent RT provider can be handed to code that calls the SDK model interface.
 */

import type {
	ModelMessage,
	ModelProvider,
	ModelRequest,
	ModelResponse,
	ModelStreamEvent,
	StreamingModelProvider,
	ToolCall,
	ToolDefinition,
} from '../../index.js';
import { parseToolCallArguments } from '../../internal/media.js';

type JSONObject = Record<string, unknown>;

/** The SDK `Model` interface: what `getResponse()` / `getStreamedResponse()` take and return. */
export interface SdkModel {
	getResponse(request: SdkModelRequest): Promise<SdkModelResponse>;
	getStreamedResponse?(
		request: SdkModelRequest,
	): AsyncIterable<SdkStreamEvent>;
}

export interface SdkModelRequest {
	systemInstructions?: string;
	input: string | JSONObject[];
	modelSettings: JSONObject;
	tools: JSONObject[];
	outputType: 'text' | JSONObject;
	handoffs: JSONObject[];
	tracing: boolean | string;
	signal?: AbortSignal;
	previousResponseId?: string;
	[key: string]: unknown;
}

export interface SdkUsageShape {
	requests?: number;
	inputTokens?: number;
	outputTokens?: number;
	totalTokens?: number;
}

export interface SdkModelResponse {
	usage?: SdkUsageShape;
	output: JSONObject[];
	responseId?: string;
}

export type SdkStreamEvent = { type: string; [key: string]: unknown };

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isSdkModel(value: unknown): value is SdkModel {
	return isObject(value) && typeof value.getResponse === 'function';
}

function messageText(message: ModelMessage): string {
	return message.content
		.filter((part) => part.type === 'text')
		.map((part) => part.text ?? '')
		.join('');
}

function partsText(content: unknown): string {
	if (typeof content === 'string') return content;
	if (!Array.isArray(content)) return '';
	return content
		.map((part) => {
			if (typeof part === 'string') return part;
			if (!isObject(part)) return '';
			if (typeof part.text === 'string') return part.text;
			if (typeof part.refusal === 'string') return part.refusal;
			return '';
		})
		.join('');
}

function outputText(output: unknown): string {
	if (typeof output === 'string') return output;
	if (isObject(output) && typeof output.text === 'string') return output.text;
	if (Array.isArray(output)) return partsText(output);
	return output === undefined ? '' : JSON.stringify(output);
}

/** True when `value` is an SDK input item list rather than Agent RT messages. */
export function isSdkItemList(value: unknown): value is JSONObject[] {
	if (!Array.isArray(value)) return false;
	return value.some(
		(item) =>
			isObject(item) &&
			(typeof item.type === 'string' ||
				typeof item.content === 'string' ||
				(Array.isArray(item.content) &&
					item.content.some(
						(part) =>
							isObject(part) &&
							typeof part.type === 'string' &&
							part.type !== 'text',
					))),
	);
}

/**
 * The SDK items each Agent RT message came from. SDK models get these back
 * verbatim, so reasoning items, `providerData`, and non-text content
 * (images, files) survive a turn through the Agent RT loop. Kept beside the
 * messages rather than in them, so nothing reaches Agent RT providers or
 * structured-output parsing.
 */
const sourceItems = new WeakMap<ModelMessage, JSONObject[]>();

/** The SDK items a message came from, for saving with `RunState`. */
export function messageSourceItems(
	message: ModelMessage,
): JSONObject[] | undefined {
	return sourceItems.get(message);
}

/** Restore saved SDK items onto a message (see `messageSourceItems`). */
export function restoreSourceItems(
	message: ModelMessage,
	items: JSONObject[],
): void {
	sourceItems.set(message, items);
}

function addSource(message: ModelMessage, item: JSONObject): void {
	const items = sourceItems.get(message);
	if (items) items.push(item);
	else sourceItems.set(message, [item]);
}

function functionCall(item: JSONObject): ToolCall {
	const parsed = parseToolCallArguments(item.arguments);
	return {
		id: String(item.callId ?? item.id ?? ''),
		name: String(item.name ?? ''),
		arguments: parsed.arguments,
		...(parsed.argumentError
			? { argumentError: parsed.argumentError }
			: {}),
	};
}

/** SDK input items (`{ role, content }`, `function_call`, ...) as Agent RT messages. */
export function sdkItemsToMessages(
	items: string | JSONObject[],
): ModelMessage[] {
	if (typeof items === 'string')
		return [{ role: 'user', content: [{ type: 'text', text: items }] }];
	const messages: ModelMessage[] = [];
	// An assistant turn: reasoning, then text, then tool calls, one message.
	const assistantTurn = (): ModelMessage => {
		const previous = messages[messages.length - 1];
		if (previous?.role === 'assistant') return previous;
		const message: ModelMessage = { role: 'assistant', content: [] };
		messages.push(message);
		return message;
	};
	for (const item of items) {
		if (!isObject(item)) continue;
		if (item.type === 'function_call') {
			const message = assistantTurn();
			message.toolCalls = [
				...(message.toolCalls ?? []),
				functionCall(item),
			];
			addSource(message, item);
			continue;
		}
		if (item.type === 'function_call_result') {
			const message: ModelMessage = {
				role: 'tool',
				toolCallId: String(item.callId ?? ''),
				content: [{ type: 'text', text: outputText(item.output) }],
			};
			addSource(message, item);
			messages.push(message);
			continue;
		}
		if (item.type === 'reasoning') {
			const previous = messages[messages.length - 1];
			// Reasoning opens a new assistant turn.
			const message: ModelMessage =
				previous?.role === 'assistant' &&
				!previous.content.length &&
				!previous.toolCalls?.length
					? previous
					: { role: 'assistant', content: [] };
			if (message !== previous) messages.push(message);
			addSource(message, item);
			continue;
		}
		const role = item.role;
		if (role === 'user' || role === 'system' || role === 'assistant') {
			const text = partsText(item.content);
			const previous = messages[messages.length - 1];
			let message: ModelMessage;
			if (
				role === 'assistant' &&
				previous?.role === 'assistant' &&
				!previous.content.length &&
				!previous.toolCalls?.length
			) {
				message = previous;
				if (text) message.content = [{ type: 'text', text }];
			} else {
				message = {
					role,
					content: text ? [{ type: 'text', text }] : [],
				};
				messages.push(message);
			}
			addSource(message, item);
			continue;
		}
		// Other SDK items ride along with the current assistant turn.
		addSource(assistantTurn(), item);
	}
	return messages;
}

function toolMessageText(message: ModelMessage): string {
	return message.content
		.map((part) =>
			part.type === 'text'
				? (part.text ?? '')
				: part.type === 'json'
					? JSON.stringify(part.data)
					: '',
		)
		.join('');
}

/** Agent RT messages as SDK input items, the system prompt split out. */
export function messagesToSdkItems(messages: ModelMessage[]): {
	systemInstructions?: string;
	input: JSONObject[];
} {
	const system: string[] = [];
	const input: JSONObject[] = [];
	const toolNames = new Map<string, string>();
	for (const message of messages) {
		for (const call of message.toolCalls ?? [])
			toolNames.set(call.id, call.name);
		const original = sourceItems.get(message);
		if (original && message.role !== 'system') {
			input.push(...original);
			continue;
		}
		const text = messageText(message);
		if (message.role === 'system') {
			if (text) system.push(text);
		} else if (message.role === 'user') {
			input.push({ role: 'user', content: text });
		} else if (message.role === 'assistant') {
			if (text) {
				input.push({
					type: 'message',
					role: 'assistant',
					status: 'completed',
					content: [{ type: 'output_text', text }],
				});
			}
			for (const call of message.toolCalls ?? []) {
				input.push({
					type: 'function_call',
					callId: call.id,
					name: call.name,
					arguments: JSON.stringify(call.arguments),
					status: 'completed',
				});
			}
		} else if (message.role === 'tool') {
			const callId = message.toolCallId ?? '';
			input.push({
				type: 'function_call_result',
				callId,
				name: toolNames.get(callId) ?? '',
				status: 'completed',
				output: { type: 'text', text: toolMessageText(message) },
			});
		}
	}
	return {
		...(system.length ? { systemInstructions: system.join('\n\n') } : {}),
		input,
	};
}

/** SDK output items as an Agent RT model response. */
export function sdkOutputToResponse(
	response: SdkModelResponse,
	model?: string,
): ModelResponse {
	const text: string[] = [];
	const toolCalls: ToolCall[] = [];
	for (const item of response.output ?? []) {
		if (!isObject(item)) continue;
		if (
			item.type === 'message' ||
			(item.role === 'assistant' && item.type === undefined)
		) {
			text.push(partsText(item.content));
		} else if (item.type === 'function_call') {
			const parsed = parseToolCallArguments(item.arguments);
			toolCalls.push({
				id: String(item.callId ?? item.id ?? ''),
				name: String(item.name ?? ''),
				arguments: parsed.arguments,
				...(parsed.argumentError
					? { argumentError: parsed.argumentError }
					: {}),
			});
		}
	}
	const joined = text.join('');
	const usage = response.usage;
	const message: ModelMessage = {
		role: 'assistant',
		content: joined ? [{ type: 'text', text: joined }] : [],
		...(toolCalls.length ? { toolCalls } : {}),
	};
	for (const item of response.output ?? [])
		if (isObject(item)) addSource(message, item);
	return {
		message,
		...(model !== undefined ? { model } : {}),
		...(usage
			? {
					usage: {
						inputTokens: usage.inputTokens ?? 0,
						outputTokens: usage.outputTokens ?? 0,
						totalTokens:
							usage.totalTokens ??
							(usage.inputTokens ?? 0) +
								(usage.outputTokens ?? 0),
					},
				}
			: {}),
		finishReason: toolCalls.length ? 'tool_calls' : 'stop',
		raw: response,
	};
}

/** An Agent RT model response as SDK output items. */
export function responseToSdkOutput(response: ModelResponse): JSONObject[] {
	const output: JSONObject[] = [];
	const text = messageText(response.message);
	if (text) {
		output.push({
			type: 'message',
			role: 'assistant',
			status: 'completed',
			content: [{ type: 'output_text', text }],
		});
	}
	for (const call of response.message.toolCalls ?? []) {
		output.push({
			type: 'function_call',
			callId: call.id,
			name: call.name,
			arguments: JSON.stringify(call.arguments),
			status: 'completed',
		});
	}
	return output;
}

function sdkUsage(response: ModelResponse): SdkUsageShape {
	const usage = response.usage;
	return {
		requests: 1,
		inputTokens: usage?.inputTokens ?? 0,
		outputTokens: usage?.outputTokens ?? 0,
		totalTokens:
			usage?.totalTokens ??
			(usage?.inputTokens ?? 0) + (usage?.outputTokens ?? 0),
	};
}

export interface SdkModelAdapterOptions {
	/** The agent's `outputType`, passed through when it is already an SDK JSON-schema definition. */
	outputType?: unknown;
	modelSettings?: JSONObject;
	/** Tool name to its `strict` flag, for the serialized tool list. */
	toolStrictness?: ReadonlyMap<string, boolean>;
	/** Receives every raw SDK stream event, for `raw_model_stream_event`. */
	onRawEvent?: (event: SdkStreamEvent) => void;
}

/** An SDK `Model` as an Agent RT streaming provider. */
export class SdkModelProvider implements StreamingModelProvider {
	readonly name = 'openai-agents-model';

	constructor(
		readonly model: SdkModel,
		readonly options: SdkModelAdapterOptions = {},
	) {}

	private sdkRequest(request: ModelRequest): SdkModelRequest {
		const { systemInstructions, input } = messagesToSdkItems(
			request.messages,
		);
		const strictness = this.options.toolStrictness;
		const tools = (request.tools ?? []).map((tool: ToolDefinition) => ({
			type: 'function',
			name: tool.name,
			description: tool.description,
			parameters: tool.inputSchema,
			strict: strictness?.get(tool.name) ?? true,
		}));
		const configured = this.options.outputType;
		let outputType: SdkModelRequest['outputType'] = 'text';
		if (request.structuredOutput) {
			outputType =
				isObject(configured) && configured.type === 'json_schema'
					? configured
					: {
							type: 'json_schema',
							name:
								request.structuredOutput.name ?? 'final_output',
							strict: request.structuredOutput.strict ?? true,
							schema: request.structuredOutput.schema,
						};
		}
		return {
			...(systemInstructions !== undefined ? { systemInstructions } : {}),
			input,
			modelSettings: {
				...(this.options.modelSettings ?? {}),
				...(request.temperature !== undefined
					? { temperature: request.temperature }
					: {}),
				...(request.maxOutputTokens !== undefined
					? { maxTokens: request.maxOutputTokens }
					: {}),
			},
			tools,
			outputType,
			handoffs: [],
			tracing: false,
		};
	}

	async complete(request: ModelRequest): Promise<ModelResponse> {
		const response = await this.model.getResponse(this.sdkRequest(request));
		return sdkOutputToResponse(response, request.model);
	}

	async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
		if (typeof this.model.getStreamedResponse !== 'function') {
			const response = await this.complete(request);
			const text = messageText(response.message);
			if (text) {
				this.options.onRawEvent?.({
					type: 'output_text_delta',
					delta: text,
				});
				yield { type: 'text_delta', text };
			}
			this.options.onRawEvent?.({
				type: 'response_done',
				response: { output: responseToSdkOutput(response) },
			});
			yield { type: 'completed', response };
			return;
		}
		let completed: ModelResponse | undefined;
		for await (const event of this.model.getStreamedResponse(
			this.sdkRequest(request),
		)) {
			this.options.onRawEvent?.(event);
			if (
				event.type === 'output_text_delta' &&
				typeof event.delta === 'string'
			) {
				yield { type: 'text_delta', text: event.delta, raw: event };
			} else if (
				event.type === 'response_done' &&
				isObject(event.response)
			) {
				completed = sdkOutputToResponse(
					event.response as unknown as SdkModelResponse,
					request.model,
				);
			}
		}
		if (!completed)
			throw new Error('model stream ended without response_done');
		yield { type: 'completed', response: completed };
	}
}

/**
 * Exposes an Agent RT provider through the SDK `Model` interface, so code that
 * calls `getModel(...).getResponse()` or `getStreamedResponse()` directly keeps
 * working.
 */
export function sdkRequestToModelRequest(
	request: SdkModelRequest,
	model: string | undefined,
): ModelRequest {
	const messages = sdkItemsToMessages(request.input);
	if (request.systemInstructions) {
		messages.unshift({
			role: 'system',
			content: [{ type: 'text', text: request.systemInstructions }],
		});
	}
	const settings = request.modelSettings ?? {};
	const outputType = request.outputType;
	return {
		...(model !== undefined ? { model } : {}),
		messages,
		...((request.tools ?? []).length
			? {
					tools: request.tools.map((tool) => ({
						name: String(tool.name ?? ''),
						description: String(tool.description ?? ''),
						inputSchema: isObject(tool.parameters)
							? tool.parameters
							: { type: 'object' },
					})),
				}
			: {}),
		...(typeof settings.temperature === 'number'
			? { temperature: settings.temperature }
			: {}),
		...(typeof settings.maxTokens === 'number'
			? { maxOutputTokens: settings.maxTokens }
			: {}),
		...(isObject(outputType) && isObject(outputType.schema)
			? {
					structuredOutput: {
						schema: outputType.schema,
						...(typeof outputType.name === 'string'
							? { name: outputType.name }
							: {}),
						strict: outputType.strict !== false,
					},
				}
			: {}),
	};
}

export async function providerGetResponse(
	provider: ModelProvider,
	model: string | undefined,
	request: SdkModelRequest,
): Promise<SdkModelResponse> {
	const response = await provider.complete(
		sdkRequestToModelRequest(request, model),
	);
	return {
		usage: sdkUsage(response),
		output: responseToSdkOutput(response),
	};
}

export async function* providerGetStreamedResponse(
	provider: ModelProvider,
	model: string | undefined,
	request: SdkModelRequest,
): AsyncIterable<SdkStreamEvent> {
	const modelRequest = sdkRequestToModelRequest(request, model);
	yield { type: 'response_started' };
	let response: ModelResponse | undefined;
	const streaming = provider as Partial<StreamingModelProvider>;
	if (typeof streaming.stream === 'function') {
		for await (const event of streaming.stream(modelRequest)) {
			if (event.type === 'text_delta' && event.text)
				yield { type: 'output_text_delta', delta: event.text };
			else if (event.type === 'completed') response = event.response;
		}
	} else {
		response = await provider.complete(modelRequest);
		const text = messageText(response.message);
		if (text) yield { type: 'output_text_delta', delta: text };
	}
	if (!response) throw new Error('model stream ended without a response');
	yield {
		type: 'response_done',
		response: {
			usage: sdkUsage(response),
			output: responseToSdkOutput(response),
		},
	};
}

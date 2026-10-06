/**
 * `OpenAIChatCompletionsModel` for the `@openai/agents` shim: an SDK `Model`
 * over an OpenAI-shaped client's `chat.completions.create`, building the same
 * Chat Completions requests and SDK output items as the upstream model, so
 * clients that wrap or inspect that wire format keep working.
 */

import type {
	SdkModel,
	SdkModelRequest,
	SdkModelResponse,
	SdkStreamEvent,
} from './openai_agents_model.js';

type JSONObject = Record<string, unknown>;

type ChatClient = {
	baseURL?: string;
	chat: {
		completions: {
			create(
				body: JSONObject,
				options?: {
					signal?: AbortSignal;
					headers?: Record<string, string>;
				},
			): Promise<unknown>;
		};
	};
};

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** `providerData` minus the keys the converter sets itself. */
function extra(providerData: unknown, reserved: string[]): JSONObject {
	if (!isObject(providerData)) return {};
	const out: JSONObject = {};
	for (const [key, value] of Object.entries(providerData))
		if (!reserved.includes(key)) out[key] = value;
	return out;
}

function assistantContent(content: unknown): unknown {
	if (typeof content === 'string') return content;
	if (!Array.isArray(content)) return [];
	const out: JSONObject[] = [];
	for (const part of content) {
		if (!isObject(part)) continue;
		if (part.type === 'output_text') {
			out.push({
				type: 'text',
				text: part.text,
				...extra(part.providerData, ['type', 'text']),
			});
		} else if (part.type === 'refusal') {
			out.push({
				type: 'refusal',
				refusal: part.refusal,
				...extra(part.providerData, ['type', 'refusal']),
			});
		}
	}
	return out;
}

function userContent(content: unknown): unknown {
	if (typeof content === 'string') return content;
	if (!Array.isArray(content)) return [];
	const out: JSONObject[] = [];
	for (const part of content) {
		if (!isObject(part)) continue;
		if (part.type === 'input_text') {
			out.push({
				type: 'text',
				text: part.text,
				...extra(part.providerData, ['type', 'text']),
			});
		} else if (part.type === 'input_image') {
			const url =
				typeof part.image === 'string'
					? part.image
					: typeof part.imageUrl === 'string'
						? part.imageUrl
						: undefined;
			if (!url)
				throw new Error(
					`Only image URLs are supported for input_image: ${JSON.stringify(part)}`,
				);
			const providerData = isObject(part.providerData)
				? part.providerData
				: {};
			out.push({
				type: 'image_url',
				image_url: {
					url,
					...(part.detail !== undefined
						? { detail: part.detail }
						: {}),
					...extra(providerData.image_url, ['url']),
				},
				...extra(providerData, ['type', 'image_url']),
			});
		} else if (part.type === 'input_file') {
			const file: JSONObject = {};
			if (
				typeof part.file === 'string' &&
				part.file.trim().startsWith('data:')
			) {
				file.file_data = part.file.trim();
			} else if (isObject(part.file) && 'id' in part.file) {
				file.file_id = part.file.id;
			} else {
				throw new Error(
					`File input requires a data URL or file ID: ${JSON.stringify(part)}`,
				);
			}
			if (part.filename) file.filename = part.filename;
			out.push({
				type: 'file',
				file,
				...extra(part.providerData, ['type', 'file', 'filename']),
			});
		}
	}
	return out;
}

function toolOutput(output: unknown): string {
	if (typeof output === 'string') return output;
	if (Array.isArray(output)) {
		const texts = output.filter(
			(item) => isObject(item) && item.type === 'input_text',
		);
		return texts.length
			? texts
					.map((item) => String((item as JSONObject).text ?? ''))
					.join('')
			: '[tool output omitted]';
	}
	if (
		isObject(output) &&
		output.type === 'text' &&
		typeof output.text === 'string'
	)
		return output.text;
	return '[tool output omitted]';
}

/** SDK input items as Chat Completions messages. */
export function itemsToChatMessages(items: string | unknown[]): JSONObject[] {
	if (typeof items === 'string') return [{ role: 'user', content: items }];
	const result: JSONObject[] = [];
	let assistant: JSONObject | null = null;
	const flush = () => {
		if (!assistant) return;
		if (
			!Array.isArray(assistant.tool_calls) ||
			!assistant.tool_calls.length
		)
			delete assistant.tool_calls;
		result.push(assistant);
		assistant = null;
	};
	const openAssistant = (): JSONObject => {
		assistant ??= { role: 'assistant', content: null, tool_calls: [] };
		return assistant;
	};
	for (const item of items) {
		if (!isObject(item)) continue;
		const isMessage =
			item.type === 'message' ||
			(item.type === undefined && typeof item.role === 'string');
		if (isMessage) {
			flush();
			const providerData = item.providerData;
			if (item.role === 'assistant') {
				result.push({
					role: 'assistant',
					content: assistantContent(item.content),
					...extra(providerData, [
						'role',
						'content',
						'tool_calls',
						'audio',
						'phase',
					]),
				});
			} else if (item.role === 'user') {
				result.push({
					role: 'user',
					content: userContent(item.content),
					...extra(providerData, ['role', 'content', 'phase']),
				});
			} else if (item.role === 'system') {
				result.push({
					role: 'system',
					content: item.content,
					...extra(providerData, ['role', 'content', 'phase']),
				});
			}
		} else if (item.type === 'reasoning') {
			// Reasoning rides on the assistant message, as some providers accept.
			const rawContent = Array.isArray(item.rawContent)
				? item.rawContent
				: [];
			openAssistant().reasoning = isObject(rawContent[0])
				? rawContent[0].text
				: undefined;
		} else if (item.type === 'function_call') {
			const message = openAssistant();
			const calls = message.tool_calls as JSONObject[];
			const providerData = isObject(item.providerData)
				? item.providerData
				: {};
			calls.push({
				id: item.callId,
				type: 'function',
				function: {
					name: item.name,
					arguments: item.arguments ?? '{}',
					...extra(providerData.function, ['name', 'arguments']),
				},
				...extra(providerData, [
					'id',
					'type',
					'function',
					'role',
					'content',
					'tool_calls',
					'audio',
				]),
			});
			Object.assign(
				message,
				extra(providerData, [
					'role',
					'content',
					'tool_calls',
					'audio',
					'id',
					'type',
					'function',
				]),
			);
		} else if (item.type === 'function_call_result') {
			flush();
			result.push({
				role: 'tool',
				tool_call_id: item.callId,
				content: toolOutput(item.output),
				...extra(item.providerData, [
					'role',
					'tool_call_id',
					'content',
				]),
			});
		} else if (item.type === 'unknown') {
			result.push({
				...(isObject(item.providerData) ? item.providerData : {}),
			});
		} else {
			throw new Error(
				`Unsupported item for chat completions: ${JSON.stringify(item)}`,
			);
		}
	}
	flush();
	return result;
}

function chatTool(tool: JSONObject): JSONObject {
	if (tool.type !== 'function')
		throw new Error(
			`Hosted tools are not supported with the ChatCompletions API: ${String(tool.type)}`,
		);
	return {
		type: 'function',
		function: {
			name: tool.name,
			description: tool.description || '',
			parameters: tool.parameters,
			strict: tool.strict,
		},
	};
}

function toolChoice(choice: unknown): unknown {
	if (choice === undefined || choice === null) return undefined;
	if (choice === 'auto' || choice === 'required' || choice === 'none')
		return choice;
	return { type: 'function', function: { name: choice } };
}

function responseFormat(
	outputType: SdkModelRequest['outputType'],
): JSONObject | undefined {
	if (outputType === 'text') return undefined;
	if (isObject(outputType) && outputType.type === 'json_schema') {
		return {
			type: 'json_schema',
			json_schema: {
				name: outputType.name,
				strict: outputType.strict,
				schema: outputType.schema,
			},
		};
	}
	return { type: 'json_object' };
}

type ChatUsage = {
	prompt_tokens?: number;
	completion_tokens?: number;
	total_tokens?: number;
	prompt_tokens_details?: { cached_tokens?: number };
	completion_tokens_details?: { reasoning_tokens?: number };
};

function usageShape(usage: ChatUsage | undefined) {
	return {
		requests: 1,
		inputTokens: usage?.prompt_tokens ?? 0,
		outputTokens: usage?.completion_tokens ?? 0,
		totalTokens: usage?.total_tokens ?? 0,
		inputTokensDetails: {
			cached_tokens: usage?.prompt_tokens_details?.cached_tokens ?? 0,
		},
		outputTokensDetails: {
			reasoning_tokens:
				usage?.completion_tokens_details?.reasoning_tokens ?? 0,
		},
	};
}

export class OpenAIChatCompletionsModel implements SdkModel {
	readonly #client: ChatClient;
	readonly #model: string;

	constructor(
		client: unknown,
		model: string,
		_options: { strictFeatureValidation?: boolean } = {},
	) {
		this.#client = client as ChatClient;
		this.#model = model;
	}

	get model(): string {
		return this.#model;
	}

	#body(request: SdkModelRequest, stream: boolean): JSONObject {
		const settings = request.modelSettings ?? {};
		const tools = (request.tools ?? []).map(chatTool);
		for (const handoff of request.handoffs ?? []) {
			tools.push({
				type: 'function',
				function: {
					name: handoff.toolName,
					description: handoff.toolDescription || '',
					parameters: handoff.inputJsonSchema,
				},
			});
		}
		const messages = itemsToChatMessages(request.input);
		if (request.systemInstructions)
			messages.unshift({
				content: request.systemInstructions,
				role: 'system',
			});
		const providerData: JSONObject = {
			...(isObject(settings.providerData) ? settings.providerData : {}),
		};
		const reasoning = isObject(settings.reasoning)
			? settings.reasoning
			: undefined;
		if (reasoning?.effort) providerData.reasoning_effort = reasoning.effort;
		const text = isObject(settings.text) ? settings.text : undefined;
		if (text?.verbosity) providerData.verbosity = text.verbosity;
		const body: JSONObject = {
			model: this.#model,
			messages,
			tools: tools.length ? tools : undefined,
			temperature: settings.temperature,
			top_p: settings.topP,
			frequency_penalty: settings.frequencyPenalty,
			presence_penalty: settings.presencePenalty,
			max_tokens: settings.maxTokens,
			tool_choice: toolChoice(settings.toolChoice),
			parallel_tool_calls:
				typeof settings.parallelToolCalls === 'boolean'
					? settings.parallelToolCalls
					: undefined,
			stream,
			stream_options: stream ? { include_usage: true } : undefined,
			store: settings.store,
			...providerData,
		};
		const format = responseFormat(request.outputType);
		if (format) body.response_format = format;
		return body;
	}

	async getResponse(request: SdkModelRequest): Promise<SdkModelResponse> {
		const completion = (await this.#client.chat.completions.create(
			this.#body(request, false),
			{
				...(request.signal ? { signal: request.signal } : {}),
			},
		)) as {
			id?: string;
			choices?: Array<{ message?: JSONObject }>;
			usage?: ChatUsage;
		};
		const output: JSONObject[] = [];
		const message = completion.choices?.[0]?.message;
		if (message) {
			if (typeof message.reasoning === 'string' && message.reasoning) {
				output.push({
					type: 'reasoning',
					content: [],
					rawContent: [
						{ type: 'reasoning_text', text: message.reasoning },
					],
				});
			}
			const {
				content,
				refusal,
				tool_calls: toolCalls,
				...rest
			} = message;
			const hasContent =
				content !== undefined &&
				content !== null &&
				!(toolCalls && content === '');
			if (hasContent) {
				output.push({
					id: completion.id,
					type: 'message',
					role: 'assistant',
					content: [
						{
							type: 'output_text',
							text: content || '',
							providerData: {
								refusal,
								tool_calls: toolCalls,
								...rest,
							},
						},
					],
					status: 'completed',
				});
			} else if (refusal) {
				output.push({
					id: completion.id,
					type: 'message',
					role: 'assistant',
					content: [{ type: 'refusal', refusal, providerData: rest }],
					status: 'completed',
				});
			}
			for (const call of Array.isArray(toolCalls)
				? (toolCalls as JSONObject[])
				: []) {
				if (call.type !== 'function' || !isObject(call.function))
					continue;
				const { id: callId, ...callRest } = call;
				const {
					arguments: args,
					name,
					...functionRest
				} = call.function;
				output.push({
					id: completion.id,
					type: 'function_call',
					arguments: args,
					name,
					callId,
					status: 'completed',
					providerData: { ...callRest, ...functionRest },
				});
			}
		}
		return {
			usage: usageShape(completion.usage),
			output,
			responseId: completion.id,
		};
	}

	async *getStreamedResponse(
		request: SdkModelRequest,
	): AsyncIterable<SdkStreamEvent> {
		const stream = (await this.#client.chat.completions.create(
			this.#body(request, true),
			{
				...(request.signal ? { signal: request.signal } : {}),
			},
		)) as AsyncIterable<JSONObject>;
		let responseId: string | undefined;
		let usage: ChatUsage | undefined;
		let started = false;
		let text:
			| { type: 'output_text'; text: string; providerData: JSONObject }
			| undefined;
		let refusal: { type: 'refusal'; refusal: string } | undefined;
		let reasoning = '';
		const calls = new Map<
			number,
			JSONObject & { arguments: string; name: string; callId: string }
		>();
		for await (const chunk of stream) {
			if (typeof chunk.id === 'string' && !responseId)
				responseId = chunk.id;
			if (!started) {
				started = true;
				yield { type: 'response_started', providerData: { ...chunk } };
			}
			yield { type: 'model', event: chunk };
			if (chunk.usage) usage = chunk.usage as ChatUsage;
			const choices = Array.isArray(chunk.choices)
				? (chunk.choices as JSONObject[])
				: [];
			const choice = choices.find(
				(item) => item.index === 0 || item.index === undefined,
			);
			const delta = isObject(choice?.delta) ? choice.delta : undefined;
			if (!delta) continue;
			if (typeof delta.content === 'string' && delta.content) {
				text ??= {
					type: 'output_text',
					text: '',
					providerData: { annotations: [] },
				};
				text.text += delta.content;
				yield {
					type: 'output_text_delta',
					delta: delta.content,
					providerData: { ...chunk },
				};
			}
			if (typeof delta.reasoning === 'string')
				reasoning += delta.reasoning;
			if (typeof delta.refusal === 'string' && delta.refusal) {
				refusal ??= { type: 'refusal', refusal: '' };
				refusal.refusal += delta.refusal;
			}
			for (const toolDelta of Array.isArray(delta.tool_calls)
				? (delta.tool_calls as JSONObject[])
				: []) {
				const index = Number(toolDelta.index ?? 0);
				let call = calls.get(index);
				if (!call) {
					call = {
						id: responseId ?? 'FAKE_ID',
						arguments: '',
						name: '',
						type: 'function_call',
						callId: '',
					};
					calls.set(index, call);
				}
				const fn = isObject(toolDelta.function)
					? toolDelta.function
					: {};
				call.arguments +=
					typeof fn.arguments === 'string' ? fn.arguments : '';
				call.name += typeof fn.name === 'string' ? fn.name : '';
				if (typeof toolDelta.id === 'string' && !call.callId)
					call.callId = toolDelta.id;
			}
		}
		const id = responseId ?? 'FAKE_ID';
		const output: JSONObject[] = [];
		if (reasoning)
			output.push({
				type: 'reasoning',
				content: [],
				rawContent: [{ type: 'reasoning_text', text: reasoning }],
			});
		if (text || refusal) {
			output.push({
				id,
				content: [
					...(text ? [text] : []),
					...(refusal ? [refusal] : []),
				],
				role: 'assistant',
				type: 'message',
				status: 'completed',
			});
		}
		for (const [, call] of [...calls.entries()].sort(
			([left], [right]) => left - right,
		)) {
			call.id = id;
			// Some providers send `{}` before the real arguments.
			if (call.arguments.startsWith('{}{'))
				call.arguments = call.arguments.slice(2);
			output.push(call);
		}
		yield {
			type: 'response_done',
			response: { id: responseId, usage: usageShape(usage), output },
		};
	}
}

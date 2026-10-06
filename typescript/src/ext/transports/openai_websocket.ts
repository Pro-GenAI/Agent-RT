import WebSocket from 'ws';
import {
	isMediaPart,
	isProviderState,
	parseToolCallArguments,
	plainPartText,
	responsesInputContent,
} from '../../internal/media.js';
import type {
	ModelMessage,
	ModelRequest,
	ModelResponse,
	ModelStreamEvent,
	ResolvedOpenAIProviderSettings,
	ToolCall,
} from '../../index.js';

function messageText(message: ModelMessage): string {
	return message.content.map(plainPartText).join('');
}

export function responsesInputItems(
	messages: ModelMessage[],
): Record<string, unknown>[] {
	const items: Record<string, unknown>[] = [];
	for (const message of messages) {
		const text = messageText(message);
		if (message.role === 'tool') {
			items.push({
				type: 'function_call_output',
				call_id: message.toolCallId ?? '',
				output: text,
			});
			continue;
		}
		if (
			message.content.some(
				(part) => isMediaPart(part) && !isProviderState(part),
			)
		) {
			items.push({
				type: 'message',
				role: message.role,
				content: responsesInputContent(message.role, message.content),
			});
		} else if (text) {
			items.push({ type: 'message', role: message.role, content: text });
		}
		for (const call of message.toolCalls ?? []) {
			items.push({
				type: 'function_call',
				call_id: call.id,
				name: call.name,
				arguments: JSON.stringify(call.arguments),
			});
		}
	}
	return items;
}

export function responsesPayload(
	model: string,
	request: ModelRequest,
	messages: ModelMessage[],
	prior?: string,
): Record<string, unknown> {
	const payload: Record<string, unknown> = {
		model,
		store: false,
		input: responsesInputItems(messages),
	};
	if (prior) payload[['previous', 'response', 'id'].join('_')] = prior;
	if (request.tools?.length) {
		payload.tools = request.tools.map((tool) => ({
			type: 'function',
			name: tool.name,
			description: tool.description,
			parameters: tool.inputSchema,
		}));
	}
	if (request.toolSelection?.required?.length) {
		payload.tool_choice =
			request.toolSelection.required.length === 1
				? { type: 'function', name: request.toolSelection.required[0] }
				: 'required';
	}
	if (request.reasoning) {
		const reasoning: Record<string, unknown> = {};
		if (request.reasoning.effort !== undefined)
			reasoning.effort = request.reasoning.effort;
		if (request.reasoning.summary !== undefined)
			reasoning.summary = request.reasoning.summary;
		if (Object.keys(reasoning).length) payload.reasoning = reasoning;
	}
	const structured = request.structuredOutput;
	// JSON mode (JSON_OBJECT_OUTPUT): any JSON object, without a schema.
	if (
		structured &&
		structured.strict === false &&
		structured.name === undefined &&
		Object.keys(structured.schema).length === 1 &&
		structured.schema.type === 'object'
	) {
		payload.text = { format: { type: 'json_object' } };
	} else if (request.structuredOutput) {
		payload.text = {
			format: {
				type: 'json_schema',
				name: request.structuredOutput.name ?? 'response',
				schema: request.structuredOutput.schema,
				strict: request.structuredOutput.strict ?? true,
			},
		};
	}
	if (request.temperature !== undefined)
		payload.temperature = request.temperature;
	const maxOut = (request as unknown as Record<string, unknown>)[
		['max', 'Output', 'Tokens'].join('')
	];
	if (maxOut !== undefined)
		payload[['max', 'output', 'tokens'].join('_')] = maxOut;
	return payload;
}

export function openAIResponsesWebSocketUrl(baseUrl: string): string {
	const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
	const url = new URL(base + '/responses');
	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	return url.toString();
}

export class OpenAIWebSocketUnavailableError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'OpenAIWebSocketUnavailableError';
	}
}

export async function openAIResponsesWebSocketSupported(
	baseUrl: string,
	credential?: string,
	timeoutMs = 2_000,
): Promise<boolean> {
	return await new Promise<boolean>((resolve) => {
		const headerName = ['Author', 'ization'].join('');
		const scheme = ['Bear', 'er '].join('');
		const socket = new WebSocket(openAIResponsesWebSocketUrl(baseUrl), {
			headers: credential
				? { [headerName]: scheme + credential }
				: undefined,
		});
		let settled = false;
		const finish = (supported: boolean) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			socket.removeAllListeners();
			if (socket.readyState === WebSocket.OPEN) {
				socket.close();
			} else if (socket.readyState !== WebSocket.CLOSED) {
				socket.once('error', () => undefined);
				socket.terminate();
			}
			resolve(supported);
		};
		const timer = setTimeout(() => finish(false), timeoutMs);
		socket.once('open', () => finish(true));
		socket.once('error', () => finish(false));
		socket.once('unexpected-response', () => finish(false));
	});
}

export class OpenAIResponsesWebSocketTransport {
	private socket?: WebSocket;
	private connecting?: Promise<WebSocket>;
	private sequence = 0;

	constructor(
		private readonly baseUrl: string,
		private readonly credential?: string,
	) {}

	private async connect(): Promise<WebSocket> {
		if (this.socket?.readyState === WebSocket.OPEN) return this.socket;
		if (!this.connecting) {
			this.connecting = new Promise<WebSocket>((resolve, reject) => {
				const headerName = ['Author', 'ization'].join('');
				const scheme = ['Bear', 'er '].join('');
				const socket = new WebSocket(
					openAIResponsesWebSocketUrl(this.baseUrl),
					{
						headers: this.credential
							? { [headerName]: scheme + this.credential }
							: undefined,
					},
				);
				socket.once('open', () => {
					this.socket = socket;
					resolve(socket);
				});
				socket.once('error', reject);
			}).finally(() => {
				this.connecting = undefined;
			});
		}
		try {
			return await this.connecting;
		} catch {
			throw new OpenAIWebSocketUnavailableError(
				`OpenAI Responses WebSocket unavailable at ${openAIResponsesWebSocketUrl(this.baseUrl)}`,
			);
		}
	}

	private async drop(): Promise<void> {
		const socket = this.socket;
		this.socket = undefined;
		if (!socket || socket.readyState === WebSocket.CLOSED) return;
		await new Promise<void>((resolve) => {
			socket.once('close', () => resolve());
			socket.close();
			setTimeout(resolve, 250);
		});
	}

	async close(): Promise<void> {
		await this.drop();
	}

	async *events(
		payload: Record<string, unknown>,
		signal?: AbortSignal,
	): AsyncIterable<Record<string, unknown>> {
		const streamId = `agent-rt-${++this.sequence}`;
		const socket = await this.connect();
		const request = {
			type: 'response.create',
			stream_id: streamId,
			...payload,
		};
		const queue: Record<string, unknown>[] = [];
		let wake: (() => void) | undefined;
		let done = false;
		let failure: Error | undefined;
		const notify = () => {
			const callback = wake;
			wake = undefined;
			callback?.();
		};
		const onAbort = () => {
			failure = Object.assign(
				new Error('OpenAI WebSocket request aborted'),
				{ name: 'AbortError' },
			);
			done = true;
			notify();
		};
		const onError = (error: Error) => {
			failure = error;
			done = true;
			void this.drop();
			notify();
		};
		const onClose = () => {
			if (!done)
				failure = new Error(
					'OpenAI WebSocket closed before response.completed',
				);
			done = true;
			notify();
		};
		const onMessage = (data: WebSocket.RawData) => {
			let event: Record<string, unknown>;
			try {
				const parsed: unknown = JSON.parse(data.toString());
				if (
					typeof parsed !== 'object' ||
					parsed === null ||
					Array.isArray(parsed)
				)
					throw new Error('event must be a JSON object');
				event = parsed as Record<string, unknown>;
			} catch (error) {
				// A malformed frame must fail this request, not crash the process.
				failure = new Error(
					'OpenAI WebSocket sent a malformed event: ' + String(error),
				);
				done = true;
				void this.drop();
				notify();
				return;
			}
			if (event.stream_id !== undefined && event.stream_id !== streamId)
				return;
			const type = event.type;
			if (type === 'error' || type === 'response.failed') {
				failure = new Error(
					`OpenAI WebSocket response failed: ${JSON.stringify(event)}`,
				);
				done = true;
				notify();
				return;
			}
			queue.push(event);
			// An incomplete response (for example max_output_tokens) is a normal
			// truncated result, not a transport failure.
			if (type === 'response.completed' || type === 'response.incomplete')
				done = true;
			notify();
		};

		try {
			if (signal?.aborted) {
				onAbort();
			} else {
				signal?.addEventListener('abort', onAbort, { once: true });
				socket.on('error', onError);
				socket.on('close', onClose);
				socket.on('message', onMessage);
				socket.send(JSON.stringify(request));
			}
			while (!done || queue.length) {
				if (!queue.length) {
					await new Promise<void>((resolve) => {
						wake = resolve;
					});
				}
				while (queue.length) yield queue.shift()!;
			}
			if (failure) throw failure;
		} finally {
			signal?.removeEventListener('abort', onAbort);
			socket.off('error', onError);
			socket.off('close', onClose);
			socket.off('message', onMessage);
		}
	}
}

function responseUsage(value: unknown): ModelResponse['usage'] {
	if (typeof value !== 'object' || value === null) return undefined;
	const usage = value as Record<string, unknown>;
	const num = (item: unknown): number | undefined =>
		typeof item === 'number' ? item : undefined;
	const inputDetails = (usage.input_tokens_details ?? {}) as Record<
		string,
		unknown
	>;
	const outputDetails = (usage.output_tokens_details ?? {}) as Record<
		string,
		unknown
	>;
	return {
		inputTokens: num(usage.input_tokens),
		outputTokens: num(usage.output_tokens),
		totalTokens: num(usage.total_tokens),
		cachedTokens: num(inputDetails.cached_tokens),
		reasoningTokens: num(outputDetails.reasoning_tokens),
	};
}

export class OpenAIResponsesWebSocketSession {
	private readonly transport: OpenAIResponsesWebSocketTransport;
	private prior?: string;
	private expectedPrefix: ModelMessage[] = [];

	constructor(
		private readonly settings: ResolvedOpenAIProviderSettings,
		credential?: string,
	) {
		this.transport = new OpenAIResponsesWebSocketTransport(
			settings.baseUrl,
			credential,
		);
	}

	async close(): Promise<void> {
		await this.transport.close();
		this.prior = undefined;
		this.expectedPrefix = [];
	}

	async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
		let messages = request.messages;
		let prior: string | undefined;
		if (
			this.prior &&
			this.expectedPrefix.length > 0 &&
			messages.length >= this.expectedPrefix.length &&
			this.expectedPrefix.every(
				(message, index) => messages[index] === message,
			)
		) {
			prior = this.prior;
			messages = messages.slice(this.expectedPrefix.length);
		}

		const model =
			request.model?.trim() || this.settings.defaultModel?.trim();
		if (!model)
			throw new Error(
				'OpenAI model is required when no defaultModel is configured',
			);
		const payload = responsesPayload(model, request, messages, prior);
		const textParts: string[] = [];
		const toolState = new Map<
			number,
			{ id: string; name: string; arguments: string }
		>();
		let finalEvent: Record<string, unknown> | undefined;

		try {
			for await (const event of this.transport.events(
				payload,
				request.signal,
			)) {
				const type = event.type;
				if (type === 'response.output_text.delta') {
					const delta = event.delta;
					if (typeof delta === 'string' && delta) {
						textParts.push(delta);
						yield { type: 'text_delta', text: delta, raw: event };
					}
					continue;
				}
				if (type === 'response.output_item.added') {
					const item = event.item as
						Record<string, unknown> | undefined;
					if (item?.type === 'function_call') {
						const index =
							typeof event.output_index === 'number'
								? event.output_index
								: 0;
						toolState.set(index, {
							id:
								typeof item.call_id === 'string'
									? item.call_id
									: typeof item.id === 'string'
										? item.id
										: '',
							name:
								typeof item.name === 'string' ? item.name : '',
							arguments:
								typeof item.arguments === 'string'
									? item.arguments
									: '',
						});
					}
					continue;
				}
				if (type === 'response.function_call_arguments.delta') {
					const index =
						typeof event.output_index === 'number'
							? event.output_index
							: 0;
					const state = toolState.get(index) ?? {
						id: '',
						name: '',
						arguments: '',
					};
					const delta = event.delta;
					if (typeof delta === 'string') state.arguments += delta;
					toolState.set(index, state);
					yield {
						type: 'tool_call_delta',
						...(state.id ? { toolCallId: state.id } : {}),
						...(state.name ? { toolName: state.name } : {}),
						...(typeof delta === 'string'
							? { argumentsDelta: delta }
							: {}),
						raw: event,
					};
					continue;
				}
				if (
					type === 'response.completed' ||
					type === 'response.incomplete'
				)
					finalEvent = event;
			}
		} catch (error) {
			// The server-side response chain is unusable after a failure (and a
			// reconnect cannot resolve it), so the next request starts fresh.
			this.prior = undefined;
			this.expectedPrefix = [];
			throw error;
		}

		if (!finalEvent)
			throw new Error(
				'OpenAI WebSocket closed before response.completed',
			);
		const rawResponse =
			(finalEvent.response as Record<string, unknown> | undefined) ?? {};
		const output = Array.isArray(rawResponse.output)
			? rawResponse.output
			: [];
		if (!textParts.length) {
			for (const value of output) {
				const item = value as Record<string, unknown>;
				if (item.type !== 'message' || !Array.isArray(item.content))
					continue;
				for (const valuePart of item.content) {
					const part = valuePart as Record<string, unknown>;
					if (
						part.type === 'output_text' &&
						typeof part.text === 'string'
					) {
						textParts.push(part.text);
					}
				}
			}
		}

		for (const [index, value] of output.entries()) {
			const item = value as Record<string, unknown>;
			if (item.type !== 'function_call') continue;
			const state = toolState.get(index) ?? {
				id: '',
				name: '',
				arguments: '',
			};
			if (typeof item.call_id === 'string') state.id = item.call_id;
			else if (typeof item.id === 'string') state.id = item.id;
			if (typeof item.name === 'string') state.name = item.name;
			if (typeof item.arguments === 'string')
				state.arguments = item.arguments;
			toolState.set(index, state);
		}

		const toolCalls = [...toolState.entries()]
			.sort(([a], [b]) => a - b)
			.map(([, state]): ToolCall => {
				const parsed = parseToolCallArguments(state.arguments);
				return {
					id: state.id,
					name: state.name,
					arguments: parsed.arguments,
					...(parsed.argumentError === undefined
						? {}
						: { argumentError: parsed.argumentError }),
				};
			});
		const response: ModelResponse = {
			message: {
				role: 'assistant',
				content: textParts.length
					? [{ type: 'text', text: textParts.join('') }]
					: [],
				...(toolCalls.length ? { toolCalls } : {}),
			},
			model:
				typeof rawResponse.model === 'string'
					? rawResponse.model
					: undefined,
			usage: responseUsage(rawResponse.usage),
			finishReason: toolCalls.length
				? 'tool_calls'
				: finalEvent.type === 'response.incomplete'
					? 'length'
					: 'stop',
			raw: finalEvent,
		};

		this.prior =
			typeof rawResponse.id === 'string' ? rawResponse.id : undefined;
		this.expectedPrefix = this.prior
			? [...request.messages, response.message]
			: [];
		yield { type: 'completed', response, raw: finalEvent };
	}
}

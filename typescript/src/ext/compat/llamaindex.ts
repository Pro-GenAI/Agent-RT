import {
	AnthropicModelProvider,
	DEFAULT_ANTHROPIC_BASE_URL,
	DEFAULT_OPENAI_BASE_URL,
	OpenAIModelProvider,
	type AgentCheckpoint,
	type CheckpointStore,
	type ContentPart,
	type JSONValue,
	type ModelMessage,
	type ModelProvider,
	type ModelRequest,
	type ModelResponse,
	type StreamingModelProvider,
	type StructuredOutputRequirement,
	type TokenCountingModelProvider,
	type ToolCall as AgentRTToolCall,
	type ToolDefinition,
} from '../../index.js';
import {
	CallbackManager,
	Settings,
} from './llamaindex_prompts.js';

type JSONObject = Record<string, unknown>;
type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function env(): Record<string, string | undefined> {
	const runtime = globalThis as typeof globalThis & {
		process?: { env?: Record<string, string | undefined> };
	};
	return runtime.process?.env ?? {};
}

function textFromContent(value: unknown): string {
	if (typeof value === 'string') return value;
	if (!Array.isArray(value)) return value == null ? '' : String(value);
	return value
		.map((part) => {
			if (typeof part === 'string') return part;
			if (!isObject(part)) return '';
			return typeof part.text === 'string' ? part.text : '';
		})
		.join('');
}

function contentParts(value: unknown): ContentPart[] {
	if (typeof value === 'string') {
		return value ? [{ type: 'text', text: value }] : [];
	}
	if (!Array.isArray(value)) {
		return value == null ? [] : [{ type: 'text', text: String(value) }];
	}
	const parts: ContentPart[] = [];
	for (const item of value) {
		if (typeof item === 'string') {
			parts.push({ type: 'text', text: item });
			continue;
		}
		if (!isObject(item)) continue;
		const type = typeof item.type === 'string' ? item.type : 'text';
		if (type === 'text') {
			if (typeof item.text === 'string')
				parts.push({ type: 'text', text: item.text });
			continue;
		}
		if (
			[
				'image',
				'audio',
				'video',
				'pdf',
				'document',
				'file',
				'json',
			].includes(type)
		) {
			parts.push({
				type: type as ContentPart['type'],
				...(typeof item.text === 'string' ? { text: item.text } : {}),
				...(item.data !== undefined ? { data: item.data } : {}),
				...(typeof item.mimeType === 'string'
					? { mimeType: item.mimeType }
					: {}),
			});
		}
	}
	return parts;
}

export type LlamaIndexToolCall = {
	id: string;
	name: string;
	input: Record<string, unknown>;
	/** Set when the model's arguments were malformed; the call must not run. */
	argumentError?: string;
};

export type ChatMessage = {
	role: MessageRole;
	content: string | ContentPart[];
	toolCallId?: string;
	toolCalls?: LlamaIndexToolCall[];
	options?: JSONObject;
};

export type ChatResponse = {
	message: ChatMessage;
	raw?: unknown;
};

export type ChatResponseChunk = ChatResponse & {
	delta: string;
};

export type CompletionResponse = {
	text: string;
	raw?: unknown;
	delta?: string;
};

function normalizeToolCalls(value: unknown): AgentRTToolCall[] | undefined {
	if (!Array.isArray(value)) return undefined;
	const calls: AgentRTToolCall[] = [];
	for (const item of value) {
		if (!isObject(item)) continue;
		const fn = isObject(item.function) ? item.function : item;
		const name = fn.name ?? item.name;
		if (typeof name !== 'string') continue;
		const rawArgs = fn.arguments ?? item.input ?? item.args ?? {};
		let parsed = rawArgs;
		if (typeof rawArgs === 'string') {
			try {
				parsed = JSON.parse(rawArgs || '{}');
			} catch {
				parsed = {};
			}
		}
		calls.push({
			id: String(item.id ?? ''),
			name,
			arguments: isObject(parsed) ? { ...parsed } : {},
		});
	}
	return calls;
}

function toModelMessage(message: ChatMessage): ModelMessage {
	const optionCalls = isObject(message.options)
		? normalizeToolCalls(
				message.options.toolCalls ?? message.options.toolCall,
			)
		: undefined;
	const calls =
		message.toolCalls?.map((call) => ({
			id: call.id,
			name: call.name,
			arguments: { ...call.input },
		})) ?? optionCalls;
	const optionToolCallId =
		isObject(message.options) &&
		typeof message.options.toolCallId === 'string'
			? message.options.toolCallId
			: undefined;
	return {
		role: message.role,
		content: contentParts(message.content),
		...(calls?.length ? { toolCalls: calls } : {}),
		...((message.toolCallId ?? optionToolCallId)
			? { toolCallId: message.toolCallId ?? optionToolCallId }
			: {}),
	};
}

function responseMessage(response: ModelResponse): ChatMessage {
	const toolCalls = (response.message.toolCalls ?? []).map((call) => ({
		id: call.id,
		name: call.name,
		input: { ...call.arguments },
		...(call.argumentError ? { argumentError: call.argumentError } : {}),
	}));
	const content = response.message.content.some(
		(part) => part.type !== 'text',
	)
		? response.message.content.map((part) => ({ ...part }))
		: response.message.content.map((part) => part.text ?? '').join('');
	return {
		role: 'assistant',
		content,
		...(toolCalls.length ? { toolCalls } : {}),
		options: {
			...(toolCalls.length ? { toolCalls } : {}),
			...(response.model ? { model: response.model } : {}),
			...(response.finishReason
				? { finishReason: response.finishReason }
				: {}),
			...(response.usage ? { usage: response.usage } : {}),
		},
	};
}

function zodSchemaFrom(
	value: Record<string, unknown>,
): Record<string, unknown> | undefined {
	const def = isObject(value._def) ? value._def : undefined;
	if (!def) return undefined;
	const typeName = String(def.typeName ?? def.type ?? '');
	if (typeName.includes('String')) return { type: 'string' };
	if (typeName.includes('Number')) return { type: 'number' };
	if (typeName.includes('Boolean')) return { type: 'boolean' };
	if (typeName.includes('Literal')) return { const: def.value };
	if (typeName.includes('Enum')) {
		const values = Array.isArray(def.values) ? def.values : [];
		return { type: 'string', enum: values };
	}
	if (typeName.includes('Optional')) {
		const inner = isObject(def.innerType)
			? schemaFrom(def.innerType)
			: undefined;
		return inner;
	}
	if (typeName.includes('Nullable')) {
		const inner = isObject(def.innerType)
			? schemaFrom(def.innerType)
			: undefined;
		return inner ? { anyOf: [inner, { type: 'null' }] } : undefined;
	}
	if (typeName.includes('Array')) {
		const inner = isObject(def.type)
			? schemaFrom(def.type)
			: isObject(def.element)
				? schemaFrom(def.element)
				: undefined;
		return inner ? { type: 'array', items: inner } : { type: 'array' };
	}
	if (typeName.includes('Union') && Array.isArray(def.options)) {
		const anyOf = def.options
			.map((item) => schemaFrom(item))
			.filter(
				(item): item is Record<string, unknown> => item !== undefined,
			);
		return anyOf.length ? { anyOf } : undefined;
	}
	if (typeName.includes('Object')) {
		const rawShape =
			typeof def.shape === 'function'
				? (def.shape as () => unknown)()
				: def.shape;
		if (!isObject(rawShape)) return { type: 'object', properties: {} };
		const properties: Record<string, unknown> = {};
		const required: string[] = [];
		for (const [key, child] of Object.entries(rawShape)) {
			const childSchema = schemaFrom(child);
			if (childSchema) properties[key] = childSchema;
			const childDef =
				isObject(child) && isObject(child._def)
					? child._def
					: undefined;
			const childType = String(
				childDef?.typeName ?? childDef?.type ?? '',
			);
			if (!childType.includes('Optional')) required.push(key);
		}
		return {
			type: 'object',
			properties,
			...(required.length ? { required } : {}),
		};
	}
	return undefined;
}
function schemaFrom(value: unknown): Record<string, unknown> | undefined {
	if (!isObject(value)) return undefined;
	if (typeof value.toJSONSchema === 'function') {
		const schema = (value.toJSONSchema as () => unknown)();
		return isObject(schema) ? { ...schema } : undefined;
	}
	if (isObject(value.jsonSchema)) return { ...value.jsonSchema };
	if (isObject(value.schema)) return { ...value.schema };
	const zodSchema = zodSchemaFrom(value);
	if (zodSchema) return zodSchema;
	if ('type' in value || 'properties' in value || '$schema' in value)
		return { ...value };
	return undefined;
}

function structuredRequirement(
	value: unknown,
): StructuredOutputRequirement | undefined {
	if (value === undefined || value === null) return undefined;
	const schema = schemaFrom(value);
	if (!schema) {
		throw new TypeError(
			'LlamaIndex compatibility structured output requires JSON Schema or an object exposing toJSONSchema()',
		);
	}
	return {
		schema,
		...(isObject(value) && typeof value.name === 'string'
			? { name: value.name }
			: {}),
		strict: true,
	};
}

function parseStructured(schema: unknown, text: string): unknown {
	const value: unknown = JSON.parse(text);
	if (isObject(schema) && typeof schema.safeParse === 'function') {
		const result = (schema.safeParse as (input: unknown) => unknown)(value);
		if (isObject(result) && result.success === true) return result.data;
		if (isObject(result) && result.success === false) throw result.error;
	}
	if (isObject(schema) && typeof schema.parse === 'function') {
		return (schema.parse as (input: unknown) => unknown)(value);
	}
	return value;
}

export type ToolMetadata = {
	name: string;
	description: string;
	parameters: unknown;
	returnDirect?: boolean;
	maxRetries?: number;
	errorBehavior?: 'raise' | 'return_error';
};

export type ToolOutput = {
	content: string;
	toolName: string;
	rawInput: Record<string, unknown>;
	rawOutput: unknown;
	isError?: boolean;
};

function rejectedToolOutput(toolName: string, error: string): ToolOutput {
	return {
		content: `Tool call rejected: ${error}`,
		toolName,
		rawInput: {},
		rawOutput: undefined,
		isError: true,
	};
}

export type LlamaIndexTool = {
	name: string;
	description: string;
	parameters: unknown;
	metadata: ToolMetadata;
	execute(
		input: Record<string, unknown>,
		options?: { signal?: AbortSignal },
	): Promise<unknown>;
	call(
		input: Record<string, unknown>,
		options?: { signal?: AbortSignal },
	): Promise<ToolOutput>;
	toAgentRTTool(): ToolDefinition;
};

export type ToolOptions = {
	name: string;
	description?: string;
	parameters?: unknown;
	returnDirect?: boolean;
	maxRetries?: number;
	errorBehavior?: 'raise' | 'return_error';
};

function makeTool(
	execute: (
		input: Record<string, unknown>,
		options?: { signal?: AbortSignal },
	) => unknown | Promise<unknown>,
	options: ToolOptions,
): LlamaIndexTool {
	const metadata: ToolMetadata = {
		name: options.name,
		description: options.description ?? '',
		parameters: options.parameters ?? { type: 'object', properties: {} },
		returnDirect: options.returnDirect,
		maxRetries: options.maxRetries ?? 0,
		errorBehavior: options.errorBehavior ?? 'raise',
	};
	return {
		name: metadata.name,
		description: metadata.description,
		parameters: metadata.parameters,
		metadata,
		async execute(input, callOptions = {}) {
			return await execute(input, callOptions);
		},
		async call(input, callOptions = {}) {
			const maxRetries = metadata.maxRetries ?? 0;
			for (let attempt = 0; ; attempt += 1) {
				if (callOptions.signal?.aborted) {
					throw new Error(
						'LlamaIndex compatibility tool call aborted',
					);
				}
				try {
					const rawOutput = await execute(input, callOptions);
					return {
						content:
							typeof rawOutput === 'string'
								? rawOutput
								: JSON.stringify(rawOutput),
						toolName: metadata.name,
						rawInput: { ...input },
						rawOutput,
					};
				} catch (error) {
					if (attempt < maxRetries) continue;
					if (metadata.errorBehavior === 'return_error') {
						const content =
							error instanceof Error
								? error.message
								: String(error);
						return {
							content,
							toolName: metadata.name,
							rawInput: { ...input },
							rawOutput: error,
							isError: true,
						};
					}
					throw error;
				}
			}
		},
		toAgentRTTool() {
			const inputSchema = schemaFrom(metadata.parameters);
			if (!inputSchema) {
				throw new TypeError(
					`tool ${metadata.name} parameters must be JSON Schema or expose toJSONSchema()`,
				);
			}
			return {
				name: metadata.name,
				description: metadata.description,
				inputSchema,
			};
		},
	};
}

export function tool(
	config:
		| (ToolOptions & {
				execute: (
					input: Record<string, unknown>,
				) => unknown | Promise<unknown>;
		  })
		| ((input: Record<string, unknown>) => unknown | Promise<unknown>),
	options?: ToolOptions,
): LlamaIndexTool {
	if (typeof config === 'function') {
		if (!options?.name)
			throw new TypeError(
				'tool(function, options) requires options.name',
			);
		return makeTool(config, options);
	}
	return makeTool(config.execute, config);
}

export class FunctionTool {
	static fromDefaults(
		execute: (input: Record<string, unknown>) => unknown | Promise<unknown>,
		options: ToolOptions,
	): LlamaIndexTool {
		return makeTool(execute, options);
	}
}

export type LLMOptions = {
	model: string;
	provider?: ModelProvider;
	temperature?: number;
	maxTokens?: number;
	apiKey?: string;
	baseURL?: string;
	callbackManager?: CallbackManager;
};

export type ChatParams = {
	messages: ChatMessage[];
	tools?: LlamaIndexTool[];
	stream?: boolean;
	responseFormat?: unknown;
	temperature?: number;
	maxTokens?: number;
	signal?: AbortSignal;
};

export type CompletionParams = {
	prompt: string;
	stream?: boolean;
	responseFormat?: unknown;
	temperature?: number;
	maxTokens?: number;
	signal?: AbortSignal;
};

export class LLM {
	readonly model: string;
	readonly provider: ModelProvider;
	readonly temperature?: number;
	readonly maxTokens?: number;
	readonly callbackManager: CallbackManager;

	constructor(options: LLMOptions, provider: ModelProvider) {
		this.model = options.model;
		this.provider = provider;
		this.temperature = options.temperature;
		this.maxTokens = options.maxTokens;
		this.callbackManager =
			options.callbackManager ?? Settings.callbackManager;
	}

	protected request(params: ChatParams): ModelRequest {
		return {
			model: this.model,
			messages: params.messages.map(toModelMessage),
			...(params.tools?.length
				? {
						tools: params.tools.map((candidate) =>
							candidate.toAgentRTTool(),
						),
					}
				: {}),
			...((params.temperature ?? this.temperature) !== undefined
				? { temperature: params.temperature ?? this.temperature }
				: {}),
			...((params.maxTokens ?? this.maxTokens) !== undefined
				? { maxOutputTokens: params.maxTokens ?? this.maxTokens }
				: {}),
			...(params.responseFormat !== undefined
				? {
						structuredOutput: structuredRequirement(
							params.responseFormat,
						),
					}
				: {}),
			...(params.signal ? { signal: params.signal } : {}),
		};
	}

	async chat(
		params: ChatParams & { stream: true },
	): Promise<AsyncIterable<ChatResponseChunk>>;
	async chat(
		params: ChatParams & { stream?: false | undefined },
	): Promise<ChatResponse>;
	async chat(
		params: ChatParams,
	): Promise<ChatResponse | AsyncIterable<ChatResponseChunk>> {
		if (params.stream) return this.streamChat(params);
		await this.callbackManager.emit({
			type: 'llm-start',
			model: this.model,
			input: params.messages,
		});
		try {
			const response = await this.provider.complete(this.request(params));
			const output = {
				message: responseMessage(response),
				raw: response.raw,
			};
			await this.callbackManager.emit({
				type: 'llm-end',
				model: this.model,
				output,
			});
			return output;
		} catch (error) {
			await this.callbackManager.emit({
				type: 'llm-error',
				model: this.model,
				error,
			});
			throw error;
		}
	}

	private async *streamChat(
		params: ChatParams,
	): AsyncIterable<ChatResponseChunk> {
		const provider = this.provider as ModelProvider &
			Partial<StreamingModelProvider>;
		if (typeof provider.stream !== 'function') {
			const response = await this.provider.complete(
				this.request({ ...params, stream: false }),
			);
			const message = responseMessage(response);
			yield {
				message,
				raw: response.raw,
				delta: textFromContent(message.content),
			};
			return;
		}

		let text = '';
		let emittedCompleted = false;
		const toolBuffers = new Map<
			string,
			{ name: string; arguments: string }
		>();
		await this.callbackManager.emit({
			type: 'llm-start',
			model: this.model,
			input: params.messages,
			stream: true,
		});
		for await (const event of provider.stream(this.request(params))) {
			await this.callbackManager.emit({
				type: 'llm-stream',
				model: this.model,
				event,
			});
			if (event.type === 'text_delta' && event.text) {
				text += event.text;
				yield {
					message: { role: 'assistant', content: text },
					delta: event.text,
					raw: event.raw,
				};
			} else if (event.type === 'tool_call_delta') {
				const id = event.toolCallId ?? '';
				const current = toolBuffers.get(id) ?? {
					name: '',
					arguments: '',
				};
				if (event.toolName) current.name = event.toolName;
				current.arguments += event.argumentsDelta ?? '';
				toolBuffers.set(id, current);
				let input: Record<string, unknown> = {};
				try {
					const parsed: unknown = JSON.parse(
						current.arguments || '{}',
					);
					if (isObject(parsed)) input = parsed;
				} catch {
					input = {};
				}
				yield {
					message: {
						role: 'assistant',
						content: text,
						toolCalls: [
							{
								id,
								name: current.name,
								input,
							},
						],
					},
					delta: '',
					raw: event.raw,
				};
			} else if (event.type === 'completed' && event.response) {
				emittedCompleted = true;
				await this.callbackManager.emit({
					type: 'llm-end',
					model: this.model,
					output: event.response,
					stream: true,
				});
				yield {
					message: responseMessage(event.response),
					delta: '',
					raw: event.response.raw,
				};
			}
		}
		if (!emittedCompleted && !text) {
			const response = await this.provider.complete(
				this.request({ ...params, stream: false }),
			);
			const message = responseMessage(response);
			yield {
				message,
				delta: textFromContent(message.content),
				raw: response.raw,
			};
		}
	}

	async countTokens(input: string | ChatMessage[]): Promise<number> {
		const messages: ChatMessage[] =
			typeof input === 'string'
				? [{ role: 'user', content: input }]
				: input;
		const provider = this.provider as ModelProvider &
			Partial<TokenCountingModelProvider>;
		if (typeof provider.countTokens === 'function') {
			return await provider.countTokens(this.request({ messages }));
		}
		if (Settings.tokenizer) {
			const text = messages
				.map((message) => textFromContent(message.content))
				.join('\n');
			return Settings.tokenizer(text).length;
		}
		throw new Error(
			'provider does not support token counting and Settings.tokenizer is not configured',
		);
	}

	async complete(
		params: CompletionParams & { stream: true },
	): Promise<AsyncIterable<CompletionResponse>>;
	async complete(
		params: CompletionParams & { stream?: false | undefined },
	): Promise<CompletionResponse>;
	async complete(
		params: CompletionParams,
	): Promise<CompletionResponse | AsyncIterable<CompletionResponse>> {
		if (!params.stream) {
			const response = await this.chat({
				messages: [{ role: 'user', content: params.prompt }],
				responseFormat: params.responseFormat,
				temperature: params.temperature,
				maxTokens: params.maxTokens,
				signal: params.signal,
			});
			return {
				text: textFromContent(response.message.content),
				raw: response.raw,
			};
		}
		const stream = await this.chat({
			messages: [{ role: 'user', content: params.prompt }],
			stream: true,
			responseFormat: params.responseFormat,
			temperature: params.temperature,
			maxTokens: params.maxTokens,
			signal: params.signal,
		});
		return (async function* () {
			let text = '';
			for await (const chunk of stream) {
				text =
					textFromContent(chunk.message.content) ||
					text + chunk.delta;
				yield { text, delta: chunk.delta, raw: chunk.raw };
			}
		})();
	}

	async structuredPredict(
		schema: unknown,
		prompt: string,
		values: Record<string, unknown> = {},
	): Promise<unknown> {
		const rendered = Object.entries(values).reduce(
			(current, [key, value]) =>
				current.split(`{${key}}`).join(String(value)),
			prompt,
		);
		const response = await this.complete({
			prompt: rendered,
			responseFormat: schema,
		});
		return parseStructured(schema, response.text);
	}

	asStructuredLLM(schema: unknown): StructuredLLM {
		return new StructuredLLM(this, schema);
	}

	async predictAndCall(
		tools: LlamaIndexTool[],
		params: { userMsg: string; chatHistory?: ChatMessage[] },
	): Promise<AgentResult> {
		const response = await this.chat({
			messages: [
				...(params.chatHistory ?? []),
				{ role: 'user', content: params.userMsg },
			],
			tools,
		});
		const outputs: ToolOutput[] = [];
		for (const call of response.message.toolCalls ?? []) {
			const candidate = tools.find((item) => item.name === call.name);
			if (!candidate)
				throw new Error(`model requested unknown tool ${call.name}`);
			outputs.push(
				call.argumentError
					? rejectedToolOutput(call.name, call.argumentError)
					: await candidate.call(call.input),
			);
		}
		return new AgentResult({
			response: response.message,
			toolCalls: outputs,
			raw: response.raw,
			currentAgentName: 'Agent',
		});
	}
}

export class OpenAI extends LLM {
	constructor(options: LLMOptions) {
		const settings = {
			baseUrl:
				options.baseURL ??
				env().OPENAI_BASE_URL ??
				DEFAULT_OPENAI_BASE_URL,
			defaultModel: options.model,
			apiKey: options.apiKey ?? env().OPENAI_API_KEY,
		};
		super(options, options.provider ?? new OpenAIModelProvider(settings));
	}
}

export class Anthropic extends LLM {
	constructor(options: LLMOptions) {
		const settings = {
			baseUrl:
				options.baseURL ??
				env().ANTHROPIC_BASE_URL ??
				DEFAULT_ANTHROPIC_BASE_URL,
			defaultModel: options.model,
			apiKey: options.apiKey ?? env().ANTHROPIC_API_KEY,
		};
		super(
			options,
			options.provider ?? new AnthropicModelProvider(settings),
		);
	}
}

export function openai(options: LLMOptions): OpenAI {
	return new OpenAI(options);
}

export function anthropic(options: LLMOptions): Anthropic {
	return new Anthropic(options);
}

function parsePartialStructured(schema: unknown, text: string): unknown {
	try {
		return parseStructured(schema, text);
	} catch {
		let candidate = text.trim();
		if (!candidate) return undefined;
		const quoteCount = (candidate.match(/"/g) ?? []).length;
		if (quoteCount % 2 === 1) candidate += '"';
		const opens = (candidate.match(/{/g) ?? []).length;
		const closes = (candidate.match(/}/g) ?? []).length;
		const arrayOpens = (candidate.match(/\[/g) ?? []).length;
		const arrayCloses = (candidate.match(/\]/g) ?? []).length;
		candidate += ']'.repeat(Math.max(0, arrayOpens - arrayCloses));
		candidate += '}'.repeat(Math.max(0, opens - closes));
		try {
			return parseStructured(schema, candidate);
		} catch {
			try {
				return JSON.parse(candidate);
			} catch {
				return undefined;
			}
		}
	}
}

export class StructuredLLM {
	constructor(
		readonly llm: LLM,
		readonly schema: unknown,
	) {}

	async complete(
		params: Omit<CompletionParams, 'responseFormat'> & { stream: true },
	): Promise<AsyncIterable<CompletionResponse>>;
	async complete(
		params: Omit<CompletionParams, 'responseFormat'> & {
			stream?: false | undefined;
		},
	): Promise<CompletionResponse>;
	async complete(
		params: Omit<CompletionParams, 'responseFormat'>,
	): Promise<CompletionResponse | AsyncIterable<CompletionResponse>> {
		if (params.stream) {
			const stream = await this.llm.complete({
				prompt: params.prompt,
				responseFormat: this.schema,
				temperature: params.temperature,
				maxTokens: params.maxTokens,
				signal: params.signal,
				stream: true,
			});
			const schema = this.schema;
			return (async function* () {
				let text = '';
				for await (const chunk of stream) {
					text = chunk.text || text + (chunk.delta ?? '');
					yield {
						...chunk,
						text,
						raw: parsePartialStructured(schema, text),
					};
				}
			})();
		}
		const response = await this.llm.complete({
			prompt: params.prompt,
			responseFormat: this.schema,
			temperature: params.temperature,
			maxTokens: params.maxTokens,
			signal: params.signal,
			stream: false,
		});
		return {
			...response,
			raw: parseStructured(this.schema, response.text),
		};
	}

	async chat(
		params: Omit<ChatParams, 'responseFormat'> & { stream: true },
	): Promise<AsyncIterable<ChatResponseChunk>>;
	async chat(
		params: Omit<ChatParams, 'responseFormat'> & {
			stream?: false | undefined;
		},
	): Promise<ChatResponse>;
	async chat(
		params: Omit<ChatParams, 'responseFormat'>,
	): Promise<ChatResponse | AsyncIterable<ChatResponseChunk>> {
		if (params.stream) {
			const stream = await this.llm.chat({
				messages: params.messages,
				tools: params.tools,
				responseFormat: this.schema,
				temperature: params.temperature,
				maxTokens: params.maxTokens,
				signal: params.signal,
				stream: true,
			});
			const schema = this.schema;
			return (async function* () {
				let text = '';
				for await (const chunk of stream) {
					text =
						textFromContent(chunk.message.content) ||
						text + chunk.delta;
					yield {
						...chunk,
						raw: parsePartialStructured(schema, text),
					};
				}
			})();
		}
		const response = await this.llm.chat({
			messages: params.messages,
			tools: params.tools,
			responseFormat: this.schema,
			temperature: params.temperature,
			maxTokens: params.maxTokens,
			signal: params.signal,
			stream: false,
		});
		return {
			...response,
			raw: parseStructured(
				this.schema,
				textFromContent(response.message.content),
			),
		};
	}
}

export class Memory {
	readonly sessionId: string;
	readonly tokenLimit: number;
	private messages: ChatMessage[];

	constructor(
		options: {
			sessionId?: string;
			tokenLimit?: number;
			chatHistory?: ChatMessage[];
		} = {},
	) {
		this.sessionId = options.sessionId ?? 'default';
		this.tokenLimit = options.tokenLimit ?? 30000;
		this.messages = [...(options.chatHistory ?? [])];
	}

	static fromDefaults(
		options: {
			sessionId?: string;
			tokenLimit?: number;
			chatHistory?: ChatMessage[];
		} = {},
	): Memory {
		return new Memory(options);
	}

	put(message: ChatMessage): void {
		this.messages.push(message);
	}

	putMessage(message: ChatMessage): void {
		this.put(message);
	}

	putMessages(messages: ChatMessage[]): void {
		this.messages.push(...messages);
	}

	get(): ChatMessage[] {
		return [...this.messages];
	}

	getAll(): ChatMessage[] {
		return [...this.messages];
	}

	set(messages: ChatMessage[]): void {
		this.messages = [...messages];
	}

	reset(): void {
		this.messages = [];
	}

	toJSON(): Record<string, unknown> {
		return {
			type: 'agent-rt.llamaindex.Memory',
			sessionId: this.sessionId,
			tokenLimit: this.tokenLimit,
			messages: this.getAll(),
		};
	}

	static fromJSON(value: Record<string, unknown>): Memory {
		const messages = Array.isArray(value.messages)
			? (value.messages as ChatMessage[])
			: [];
		return new Memory({
			sessionId:
				typeof value.sessionId === 'string'
					? value.sessionId
					: 'default',
			tokenLimit:
				typeof value.tokenLimit === 'number' ? value.tokenLimit : 30000,
			chatHistory: messages,
		});
	}
}

export class ChatMemoryBuffer extends Memory {}

class ContextStore {
	private readonly values = new Map<string, unknown>();

	async get<T = unknown>(
		key: string,
		defaultValue?: T,
	): Promise<T | undefined> {
		return (this.values.has(key) ? this.values.get(key) : defaultValue) as
			T | undefined;
	}

	async set(key: string, value: unknown): Promise<void> {
		this.values.set(key, value);
	}

	snapshot(): Record<string, unknown> {
		return Object.fromEntries(this.values);
	}

	restore(values: Record<string, unknown>): void {
		this.values.clear();
		for (const [key, value] of Object.entries(values))
			this.values.set(key, value);
	}
}

function encodeContextValue(value: unknown): unknown {
	if (value instanceof Memory) return value.toJSON();
	if (Array.isArray(value)) return value.map(encodeContextValue);
	if (isObject(value)) {
		return Object.fromEntries(
			Object.entries(value).map(([key, item]) => [
				key,
				encodeContextValue(item),
			]),
		);
	}
	return value;
}

function decodeContextValue(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(decodeContextValue);
	if (isObject(value)) {
		if (value.type === 'agent-rt.llamaindex.Memory') {
			return Memory.fromJSON(value);
		}
		return Object.fromEntries(
			Object.entries(value).map(([key, item]) => [
				key,
				decodeContextValue(item),
			]),
		);
	}
	return value;
}

export class Context {
	readonly store = new ContextStore();
	state: Record<string, unknown>;
	private workflowEvents: WorkflowEvent[] = [];

	constructor(
		readonly workflow?: unknown,
		state: Record<string, unknown> = {},
	) {
		this.state = { ...state };
	}

	sendEvent(event: WorkflowEvent): void {
		this.workflowEvents.push(event);
	}

	collectEvents(
		event: WorkflowEvent,
		expected: WorkflowEventConstructor[],
	): WorkflowEvent[] | undefined {
		this.workflowEvents.push(event);
		const remaining = [...this.workflowEvents];
		const collected: WorkflowEvent[] = [];
		for (const eventType of expected) {
			const index = remaining.findIndex(
				(candidate) => candidate instanceof eventType,
			);
			if (index < 0) return undefined;
			const [match] = remaining.splice(index, 1);
			if (match) collected.push(match);
		}
		this.workflowEvents = remaining;
		return collected;
	}

	toJSON(): Record<string, unknown> {
		return {
			state: encodeContextValue(this.state),
			store: encodeContextValue(this.store.snapshot()),
		};
	}

	serialize(): string {
		return JSON.stringify(this.toJSON());
	}

	static fromJSON(
		workflow: unknown,
		value: Record<string, unknown>,
	): Context {
		const decodedState = decodeContextValue(value.state);
		const state = isObject(decodedState) ? decodedState : {};
		const context = new Context(workflow, state);
		const decodedStore = decodeContextValue(value.store);
		if (isObject(decodedStore)) context.store.restore(decodedStore);
		return context;
	}

	saveCheckpoint(
		checkpointStore: CheckpointStore,
		checkpointId: string,
		agentName = 'llamaindex-workflow',
	): AgentCheckpoint {
		const checkpoint: AgentCheckpoint = {
			checkpointId,
			agentName,
			messages: [],
			turns: 0,
			toolCalls: 0,
			totalTokens: 0,
			metadata: {
				context: JSON.parse(JSON.stringify(this.toJSON())) as JSONValue,
			},
			createdAtMs: Date.now(),
		};
		checkpointStore.save(checkpoint);
		return checkpoint;
	}

	static loadCheckpoint(
		workflow: unknown,
		checkpointStore: CheckpointStore,
		checkpointId: string,
	): Context {
		const checkpoint = checkpointStore.load(checkpointId);
		if (!checkpoint)
			throw new Error(`checkpoint not found: ${checkpointId}`);
		const contextValue = checkpoint.metadata?.context;
		if (!isObject(contextValue)) {
			throw new TypeError(
				'checkpoint context metadata must be an object',
			);
		}
		return Context.fromJSON(workflow, contextValue);
	}

	static deserialize(workflow: unknown, value: string): Context {
		const parsed: unknown = JSON.parse(value);
		if (!isObject(parsed)) {
			throw new TypeError(
				'serialized LlamaIndex Context must decode to an object',
			);
		}
		return Context.fromJSON(workflow, parsed);
	}
}

export class WorkflowEvent<T = unknown> {
	constructor(readonly data?: T) {}
}

export class StartEvent<T = Record<string, unknown>> extends WorkflowEvent<T> {}

export class StopEvent<T = unknown> extends WorkflowEvent<T> {
	readonly result: T;
	constructor(result: T) {
		super(result);
		this.result = result;
	}
}

export class InputRequiredEvent extends WorkflowEvent<{ prefix?: string }> {
	constructor(readonly prefix = '') {
		super({ prefix });
	}
}

export class HumanResponseEvent extends WorkflowEvent<{ response: string }> {
	constructor(readonly response: string) {
		super({ response });
	}
}

export interface WorkflowMiddleware {
	beforeStep?(
		context: Context,
		event: WorkflowEvent,
		stepName: string,
	): WorkflowEvent | Promise<WorkflowEvent>;
	afterStep?(
		context: Context,
		event: WorkflowEvent,
		result: unknown,
		stepName: string,
	): unknown | Promise<unknown>;
}

export type WorkflowEventConstructor<T extends WorkflowEvent = WorkflowEvent> =
	new (...args: never[]) => T;

export type WorkflowStep<T extends WorkflowEvent = WorkflowEvent> = {
	name: string;
	eventType: WorkflowEventConstructor<T>;
	handler: (context: Context, event: T) => unknown | Promise<unknown>;
	numWorkers: number;
};

export function step<T extends WorkflowEvent>(
	eventType: WorkflowEventConstructor<T>,
	handler: (context: Context, event: T) => unknown | Promise<unknown>,
	options: { name?: string; numWorkers?: number } = {},
): WorkflowStep<T> {
	return {
		name: options.name ?? handler.name ?? 'step',
		eventType,
		handler,
		numWorkers: Math.max(1, options.numWorkers ?? 1),
	};
}

class WorkflowEventChannel {
	private values: WorkflowEvent[] = [];
	private waiters: Array<() => void> = [];
	done = false;
	push(value: WorkflowEvent): void {
		this.values.push(value);
		const waiters = this.waiters;
		this.waiters = [];
		for (const waiter of waiters) waiter();
	}
	close(): void {
		this.done = true;
		const waiters = this.waiters;
		this.waiters = [];
		for (const waiter of waiters) waiter();
	}
	async *iterate(): AsyncIterable<WorkflowEvent> {
		let index = 0;
		while (!this.done || index < this.values.length) {
			const [value] = this.values.slice(index, index + 1);
			if (value !== undefined) {
				index += 1;
				yield value;
				continue;
			}
			await new Promise<void>((resolve) => this.waiters.push(resolve));
		}
	}
}

export class WorkflowRunContext<T = unknown>
	implements PromiseLike<T>, AsyncIterable<WorkflowEvent>
{
	constructor(
		private readonly task: Promise<T>,
		private readonly events: WorkflowEventChannel,
		private readonly respondWith: (event: WorkflowEvent) => void,
	) {}
	then<TResult1 = T, TResult2 = never>(
		onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
		onrejected?:
			((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
	): PromiseLike<TResult1 | TResult2> {
		return this.task.then(onfulfilled, onrejected);
	}
	[Symbol.asyncIterator](): AsyncIterator<WorkflowEvent> {
		return this.events.iterate()[Symbol.asyncIterator]();
	}
	streamEvents(): AsyncIterable<WorkflowEvent> {
		return this.events.iterate();
	}
	sendEvent(event: WorkflowEvent): void {
		this.respondWith(event);
	}
	respond(response: string): void {
		this.respondWith(new HumanResponseEvent(response));
	}
}

export class Workflow {
	readonly steps: WorkflowStep[];
	readonly middleware: WorkflowMiddleware[];
	readonly checkpointStore?: CheckpointStore;
	readonly checkpointId?: string;
	readonly maxSteps: number;

	constructor(
		options: {
			steps?: WorkflowStep[];
			middleware?: WorkflowMiddleware[];
			checkpointStore?: CheckpointStore;
			checkpointId?: string;
			maxSteps?: number;
		} = {},
	) {
		this.steps = [...(options.steps ?? [])];
		this.middleware = [...(options.middleware ?? [])];
		this.checkpointStore = options.checkpointStore;
		this.checkpointId = options.checkpointId;
		this.maxSteps = options.maxSteps ?? 100;
	}

	addStep<T extends WorkflowEvent>(definition: WorkflowStep<T>): this {
		this.steps.push(definition as unknown as WorkflowStep);
		return this;
	}

	run(
		input: Record<string, unknown> = {},
		options: {
			ctx?: Context;
			checkpointStore?: CheckpointStore;
			checkpointId?: string;
		} = {},
	): WorkflowRunContext {
		const checkpointStore = options.checkpointStore ?? this.checkpointStore;
		const checkpointId = options.checkpointId ?? this.checkpointId;
		let context = options.ctx;
		if (
			!context &&
			checkpointStore &&
			checkpointId &&
			checkpointStore.load(checkpointId)
		) {
			context = Context.loadCheckpoint(
				this,
				checkpointStore,
				checkpointId,
			);
		}
		context ??= new Context(this);
		const events = new WorkflowEventChannel();
		const responses: WorkflowEvent[] = [];
		let responseWake: (() => void) | undefined;
		const respondWith = (event: WorkflowEvent): void => {
			responses.push(event);
			responseWake?.();
			responseWake = undefined;
		};
		const waitForResponse = async (): Promise<WorkflowEvent> => {
			while (!responses.length) {
				await new Promise<void>((resolve) => {
					responseWake = resolve;
				});
			}
			const [value] = responses.splice(0, 1);
			if (!value)
				throw new Error('workflow response queue unexpectedly empty');
			return value;
		};

		const task = (async (): Promise<unknown> => {
			const pending: WorkflowEvent[] = [new StartEvent(input)];
			let count = 0;
			while (pending.length) {
				if (count >= this.maxSteps)
					throw new Error(
						`workflow exceeded maxSteps=${this.maxSteps}`,
					);
				const [event] = pending.splice(0, 1);
				if (!event) continue;
				if (event instanceof InputRequiredEvent) {
					const response = await waitForResponse();
					pending.push(
						response instanceof HumanResponseEvent
							? response
							: new HumanResponseEvent(
									String(response.data ?? ''),
								),
					);
					continue;
				}
				const matches = this.steps.filter(
					(definition) => event instanceof definition.eventType,
				);
				if (!matches.length) {
					if (event instanceof StopEvent) return event.result;
					continue;
				}
				for (const definition of matches) {
					let current = event;
					for (const middleware of this.middleware) {
						if (middleware.beforeStep)
							current = await middleware.beforeStep(
								context,
								current,
								definition.name,
							);
					}
					let result = await definition.handler(context, current);
					for (const middleware of [...this.middleware].reverse()) {
						if (middleware.afterStep)
							result = await middleware.afterStep(
								context,
								current,
								result,
								definition.name,
							);
					}
					count += 1;
					if (checkpointStore && checkpointId)
						context.saveCheckpoint(checkpointStore, checkpointId);
					const outputs = Array.isArray(result) ? result : [result];
					for (const output of outputs) {
						if (output === undefined || output === null) continue;
						if (output instanceof StopEvent) {
							events.push(output);
							return output.result;
						}
						const next =
							output instanceof WorkflowEvent
								? output
								: new StopEvent(output);
						events.push(next);
						pending.push(next);
					}
				}
			}
			return undefined;
		})().finally(() => events.close());
		return new WorkflowRunContext(task, events, respondWith);
	}
}
export class AgentStream {
	constructor(readonly data: { delta: string; currentAgentName: string }) {}
}

export class AgentInput {
	constructor(
		readonly data: { input: ChatMessage[]; currentAgentName: string },
	) {}
}

export class ToolCall {
	constructor(
		readonly data: {
			toolName: string;
			toolKwargs: Record<string, unknown>;
			toolId: string;
			currentAgentName: string;
		},
	) {}
}

export class ToolCallResult {
	constructor(
		readonly data: {
			toolName: string;
			toolOutput: ToolOutput;
			toolId: string;
			currentAgentName: string;
		},
	) {}
}

export class AgentResult {
	readonly data: string;
	readonly response: ChatMessage;
	readonly toolCalls: ToolOutput[];
	readonly raw?: unknown;
	readonly currentAgentName: string;

	constructor(options: {
		response: ChatMessage;
		toolCalls?: ToolOutput[];
		raw?: unknown;
		currentAgentName: string;
	}) {
		this.response = options.response;
		this.data = textFromContent(options.response.content);
		this.toolCalls = options.toolCalls ?? [];
		this.raw = options.raw;
		this.currentAgentName = options.currentAgentName;
	}

	toString(): string {
		return this.data;
	}
}

export class AgentOutput {
	constructor(readonly data: AgentResult) {}
}

type AgentEvent =
	AgentStream | AgentInput | AgentOutput | ToolCall | ToolCallResult;

class EventChannel {
	private values: AgentEvent[] = [];
	private waiters: Array<() => void> = [];
	done = false;

	push(value: AgentEvent): void {
		this.values.push(value);
		this.wake();
	}

	close(): void {
		this.done = true;
		this.wake();
	}

	private wake(): void {
		const waiters = this.waiters;
		this.waiters = [];
		for (const waiter of waiters) waiter();
	}

	async *iterate(): AsyncIterable<AgentEvent> {
		let index = 0;
		while (!this.done || index < this.values.length) {
			if (index < this.values.length) {
				const [value] = this.values.slice(index, index + 1);
				index += 1;
				if (value !== undefined) yield value;
				continue;
			}
			await new Promise<void>((resolve) => this.waiters.push(resolve));
		}
	}
}

export class AgentRunContext
	implements PromiseLike<AgentResult>, AsyncIterable<AgentEvent>
{
	constructor(
		private readonly task: Promise<AgentResult>,
		private readonly events: EventChannel,
		private readonly cancelRun: () => void = () => {},
	) {}

	then<TResult1 = AgentResult, TResult2 = never>(
		onfulfilled?:
			((value: AgentResult) => TResult1 | PromiseLike<TResult1>) | null,
		onrejected?:
			((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
	): PromiseLike<TResult1 | TResult2> {
		return this.task.then(onfulfilled, onrejected);
	}

	[Symbol.asyncIterator](): AsyncIterator<AgentEvent> {
		return this.events.iterate()[Symbol.asyncIterator]();
	}

	streamEvents(): AsyncIterable<AgentEvent> {
		return this.events.iterate();
	}

	cancel(): void {
		this.cancelRun();
	}
}

function ensureNotAborted(signal?: AbortSignal): void {
	if (signal?.aborted) {
		throw new Error('LlamaIndex compatibility agent run aborted');
	}
}

export type AgentOptions = {
	tools?: LlamaIndexTool[];
	llm?: LLM;
	systemPrompt?: string;
	name?: string;
	description?: string;
	streaming?: boolean;
	maxIterations?: number;
	memory?: Memory;
	context?: Context;
	canHandoffTo?: string[];
};

export class FunctionAgent {
	readonly tools: LlamaIndexTool[];
	readonly llm: LLM;
	readonly systemPrompt?: string;
	readonly name: string;
	readonly description?: string;
	readonly streaming: boolean;
	readonly maxIterations: number;
	readonly memory?: Memory;
	readonly context?: Context;
	readonly canHandoffTo: string[];

	constructor(options: AgentOptions) {
		this.canHandoffTo = [...(options.canHandoffTo ?? [])];
		this.tools = [...(options.tools ?? [])];
		if (this.canHandoffTo.length) {
			this.tools.push(
				tool(
					async (input) => {
						const agentName = String(
							input.agentName ?? input.agent_name ?? '',
						);
						if (!this.canHandoffTo.includes(agentName)) {
							throw new Error(
								`agent ${agentName} is not an allowed handoff target`,
							);
						}
						return {
							handoffTo: agentName,
							message: String(input.message ?? ''),
						};
					},
					{
						name: 'handoff_to_agent',
						description:
							'Hand off the current task to another allowed agent.',
						parameters: {
							type: 'object',
							properties: {
								agentName: {
									type: 'string',
									enum: this.canHandoffTo,
								},
								message: { type: 'string' },
							},
							required: ['agentName'],
						},
						returnDirect: true,
					},
				),
			);
		}
		const configuredLLM = options.llm ?? (Settings.llm as LLM | undefined);
		if (!configuredLLM || typeof configuredLLM.chat !== 'function') {
			throw new Error('FunctionAgent requires llm or Settings.llm');
		}
		this.llm = configuredLLM;
		this.systemPrompt = options.systemPrompt;
		this.name = options.name ?? 'Agent';
		this.description = options.description;
		this.streaming = options.streaming ?? true;
		this.maxIterations = options.maxIterations ?? 8;
		this.memory = options.memory;
		this.context = options.context;
	}

	run(
		input: string,
		options: {
			memory?: Memory;
			ctx?: Context;
			chatHistory?: ChatMessage[];
			signal?: AbortSignal;
		} = {},
	): AgentRunContext {
		const events = new EventChannel();
		const controller = new AbortController();
		if (options.signal) {
			if (options.signal.aborted) controller.abort();
			else
				options.signal.addEventListener(
					'abort',
					() => controller.abort(),
					{
						once: true,
					},
				);
		}
		const executionOptions = { ...options, signal: controller.signal };
		const task = this.execute(input, executionOptions, events).finally(() =>
			events.close(),
		);
		return new AgentRunContext(task, events, () => controller.abort());
	}

	protected async execute(
		input: string,
		options: {
			memory?: Memory;
			ctx?: Context;
			chatHistory?: ChatMessage[];
			signal?: AbortSignal;
		},
		events: EventChannel,
	): Promise<AgentResult> {
		ensureNotAborted(options.signal);
		const context = options.ctx ?? this.context;
		let memory = options.memory ?? this.memory;
		if (!memory && context) {
			const stored = await context.store.get<Memory>('memory');
			if (stored instanceof Memory) memory = stored;
		}
		memory ??= Memory.fromDefaults();

		const history = options.chatHistory
			? [...options.chatHistory]
			: memory.get();
		if (this.systemPrompt && history[0]?.role !== 'system') {
			history.unshift({ role: 'system', content: this.systemPrompt });
		}
		const userMessage: ChatMessage = { role: 'user', content: input };
		history.push(userMessage);
		memory.put(userMessage);
		events.push(
			new AgentInput({
				input: [...history],
				currentAgentName: this.name,
			}),
		);

		const outputs: ToolOutput[] = [];
		for (
			let iteration = 0;
			iteration < this.maxIterations;
			iteration += 1
		) {
			ensureNotAborted(options.signal);
			let response: ChatResponse;
			if (this.streaming) {
				const stream = await this.llm.chat({
					messages: history,
					tools: this.tools,
					stream: true,
					signal: options.signal,
				});
				let last: ChatResponseChunk | undefined;
				for await (const chunk of stream) {
					last = chunk;
					if (chunk.delta) {
						events.push(
							new AgentStream({
								delta: chunk.delta,
								currentAgentName: this.name,
							}),
						);
					}
				}
				if (!last)
					throw new Error(
						'LlamaIndex compatibility model stream produced no response',
					);
				response = { message: last.message, raw: last.raw };
			} else {
				response = await this.llm.chat({
					messages: history,
					tools: this.tools,
					signal: options.signal,
				});
				const delta = textFromContent(response.message.content);
				if (delta) {
					events.push(
						new AgentStream({ delta, currentAgentName: this.name }),
					);
				}
			}

			history.push(response.message);
			memory.put(response.message);
			const calls = response.message.toolCalls ?? [];
			if (!calls.length) {
				if (context) await context.store.set('memory', memory);
				const result = new AgentResult({
					response: response.message,
					toolCalls: outputs,
					raw: response.raw,
					currentAgentName: this.name,
				});
				events.push(new AgentOutput(result));
				return result;
			}

			for (const call of calls) {
				const candidate = this.tools.find(
					(item) => item.name === call.name,
				);
				if (!candidate)
					throw new Error(
						`model requested unknown tool ${call.name}`,
					);
				events.push(
					new ToolCall({
						toolName: call.name,
						toolKwargs: { ...call.input },
						toolId: call.id,
						currentAgentName: this.name,
					}),
				);
				ensureNotAborted(options.signal);
				const output = call.argumentError
					? rejectedToolOutput(call.name, call.argumentError)
					: await candidate.call(call.input, {
							signal: options.signal,
						});
				outputs.push(output);
				events.push(
					new ToolCallResult({
						toolName: call.name,
						toolOutput: output,
						toolId: call.id,
						currentAgentName: this.name,
					}),
				);
				const toolMessage: ChatMessage = {
					role: 'tool',
					content: output.content,
					toolCallId: call.id,
					options: { toolCallId: call.id, name: call.name },
				};
				history.push(toolMessage);
				memory.put(toolMessage);
				if (candidate.metadata.returnDirect) {
					// Every tool call in the assistant message needs a reply, or the
					// next agent sharing this memory sends an invalid transcript.
					for (const skipped of calls.slice(
						calls.indexOf(call) + 1,
					)) {
						const skippedMessage: ChatMessage = {
							role: 'tool',
							content:
								'Tool call not executed: the task was returned directly.',
							toolCallId: skipped.id,
							options: {
								toolCallId: skipped.id,
								name: skipped.name,
							},
						};
						history.push(skippedMessage);
						memory.put(skippedMessage);
					}
					const directResponse: ChatMessage = {
						role: 'assistant',
						content: output.content,
					};
					history.push(directResponse);
					memory.put(directResponse);
					if (context) await context.store.set('memory', memory);
					const result = new AgentResult({
						response: directResponse,
						toolCalls: outputs,
						raw: output.rawOutput,
						currentAgentName: this.name,
					});
					events.push(new AgentOutput(result));
					return result;
				}
			}
		}
		throw new Error(
			`LlamaIndex compatibility agent exceeded maxIterations=${this.maxIterations}`,
		);
	}
}

export type ReActStep = {
	thought: string;
	action?: string;
	actionInput?: Record<string, unknown>;
	answer?: string;
};

export class ReActOutputParser {
	parse(text: string): ReActStep {
		const value = text.trim();
		if (value.includes('Answer:')) {
			const index = value.lastIndexOf('Answer:');
			const before = value.slice(0, index);
			return {
				thought: before.includes('Thought:')
					? (before.split('Thought:').slice(-1)[0]?.trim() ?? '')
					: '',
				answer: value.slice(index + 'Answer:'.length).trim(),
			};
		}
		if (!value.includes('Action:'))
			throw new Error('ReAct output must contain Action: or Answer:');
		const thought = value.includes('Thought:')
			? (value.split('Thought:')[1]?.split('Action:')[0]?.trim() ?? '')
			: '';
		const afterAction = value.split('Action:')[1] ?? '';
		const action = afterAction.split('\n')[0]?.trim();
		const rawInput = afterAction.includes('Action Input:')
			? (afterAction.split('Action Input:')[1]?.trim().split('\n')[0] ??
				'{}')
			: '{}';
		const parsed: unknown = JSON.parse(rawInput || '{}');
		if (!isObject(parsed))
			throw new TypeError('ReAct Action Input must decode to an object');
		return { thought, action, actionInput: parsed };
	}
}

export class ReActAgent extends FunctionAgent {
	readonly outputParser: ReActOutputParser;

	constructor(options: AgentOptions & { outputParser?: ReActOutputParser }) {
		super(options);
		this.outputParser = options.outputParser ?? new ReActOutputParser();
	}

	protected override async execute(
		input: string,
		options: {
			memory?: Memory;
			ctx?: Context;
			chatHistory?: ChatMessage[];
			signal?: AbortSignal;
		},
		events: EventChannel,
	): Promise<AgentResult> {
		ensureNotAborted(options.signal);
		const context = options.ctx ?? this.context;
		let memory = options.memory ?? this.memory;
		if (!memory && context) {
			const stored = await context.store.get<Memory>('memory');
			if (stored instanceof Memory) memory = stored;
		}
		memory ??= Memory.fromDefaults();
		const history = options.chatHistory
			? [...options.chatHistory]
			: memory.get();
		const availableTools = this.tools.filter(
			(item) => item.name !== 'handoff_to_agent',
		);
		const instructions = [
			this.systemPrompt,
			'Use ReAct format exactly.',
			'For tool use: Thought: <reasoning> then Action: <tool name> then Action Input: <JSON object>.',
			'After an observation continue. When finished: Thought: <reasoning> then Answer: <final answer>.',
			'Available tools:',
			...availableTools.map(
				(item) => '- ' + item.name + ': ' + item.description,
			),
		]
			.filter(Boolean)
			.join('\n');
		if (history[0]?.role !== 'system')
			history.unshift({ role: 'system', content: instructions });
		const user: ChatMessage = { role: 'user', content: input };
		history.push(user);
		memory.put(user);
		const outputs: ToolOutput[] = [];
		let parseFailures = 0;
		for (
			let iteration = 0;
			iteration < this.maxIterations;
			iteration += 1
		) {
			ensureNotAborted(options.signal);
			const response = await this.llm.chat({
				messages: history,
				stream: false,
				signal: options.signal,
			});
			const text = textFromContent(response.message.content);
			history.push(response.message);
			memory.put(response.message);
			if (text)
				events.push(
					new AgentStream({
						delta: text,
						currentAgentName: this.name,
					}),
				);
			let parsed: ReActStep;
			try {
				parsed = this.outputParser.parse(text);
			} catch (error) {
				parseFailures += 1;
				if (parseFailures > 2) throw error;
				history.push({
					role: 'user',
					content:
						'Invalid ReAct format: ' +
						String(error) +
						'. Use Thought/Action/Action Input or Thought/Answer exactly.',
				});
				continue;
			}
			if (parsed.answer !== undefined) {
				if (context) await context.store.set('memory', memory);
				const result = new AgentResult({
					response: { role: 'assistant', content: parsed.answer },
					toolCalls: outputs,
					raw: response.raw,
					currentAgentName: this.name,
				});
				events.push(new AgentOutput(result));
				return result;
			}
			const candidate = availableTools.find(
				(item) => item.name === parsed.action,
			);
			if (!candidate)
				throw new Error(
					'ReAct requested unknown tool ' + String(parsed.action),
				);
			const callId = 'react-' + String(outputs.length + 1);
			events.push(
				new ToolCall({
					toolName: candidate.name,
					toolKwargs: parsed.actionInput ?? {},
					toolId: callId,
					currentAgentName: this.name,
				}),
			);
			const output = await candidate.call(parsed.actionInput ?? {}, {
				signal: options.signal,
			});
			outputs.push(output);
			events.push(
				new ToolCallResult({
					toolName: candidate.name,
					toolOutput: output,
					toolId: callId,
					currentAgentName: this.name,
				}),
			);
			const observation: ChatMessage = {
				role: 'user',
				content: 'Observation: ' + output.content,
			};
			history.push(observation);
			memory.put(observation);
		}
		throw new Error(
			'LlamaIndex ReAct agent exceeded maxIterations=' +
				String(this.maxIterations),
		);
	}
}

export class PlanningAgent extends FunctionAgent {
	private readonly planner: (input: string) => string[] | Promise<string[]>;
	constructor(
		options: AgentOptions & {
			planner: (input: string) => string[] | Promise<string[]>;
		},
	) {
		super(options);
		this.planner = options.planner;
	}
	async plan(input: string): Promise<string[]> {
		const tasks = await this.planner(input);
		return tasks.map((item) => String(item)).filter(Boolean);
	}
	async runPlan(
		input: string,
		options: { memory?: Memory; ctx?: Context; signal?: AbortSignal } = {},
	): Promise<AgentResult[]> {
		const results: AgentResult[] = [];
		for (const task of await this.plan(input))
			results.push(await this.run(task, options));
		return results;
	}
}
export function agent(options: AgentOptions): FunctionAgent {
	return new FunctionAgent(options);
}

export class AgentWorkflow {
	readonly agents: Map<string, FunctionAgent>;
	readonly rootAgent: string;
	readonly maxHandoffs: number;

	constructor(options: {
		agents: FunctionAgent[];
		rootAgent?: string;
		maxHandoffs?: number;
	}) {
		if (!options.agents.length)
			throw new Error('AgentWorkflow requires at least one agent');
		this.agents = new Map(options.agents.map((item) => [item.name, item]));
		if (this.agents.size !== options.agents.length)
			throw new Error('AgentWorkflow agent names must be unique');
		this.rootAgent = options.rootAgent ?? options.agents[0].name;
		if (!this.agents.has(this.rootAgent))
			throw new Error('unknown root agent ' + this.rootAgent);
		this.maxHandoffs = options.maxHandoffs ?? 8;
		for (const current of options.agents) {
			if (current.canHandoffTo.length) continue;
			const allowed = options.agents
				.filter((item) => item.name !== current.name)
				.map((item) => item.name);
			if (!allowed.length) continue;
			current.canHandoffTo.push(...allowed);
			if (
				!current.tools.some((item) => item.name === 'handoff_to_agent')
			) {
				current.tools.push(
					tool(
						async (input) => {
							const agentName = String(
								input.agentName ?? input.agent_name ?? '',
							);
							if (!allowed.includes(agentName))
								throw new Error(
									'agent ' +
										agentName +
										' is not an allowed handoff target',
								);
							return {
								handoffTo: agentName,
								message: String(input.message ?? ''),
							};
						},
						{
							name: 'handoff_to_agent',
							description:
								'Hand off the current task to another named agent.',
							parameters: {
								type: 'object',
								properties: {
									agentName: {
										type: 'string',
										enum: allowed,
									},
									message: { type: 'string' },
								},
								required: ['agentName'],
							},
							returnDirect: true,
						},
					),
				);
			}
		}
	}

	static fromAgents(
		agents: FunctionAgent[],
		options: { rootAgent?: string; maxHandoffs?: number } = {},
	): AgentWorkflow {
		return new AgentWorkflow({ agents, ...options });
	}

	static fromToolsOrFunctions(
		tools: LlamaIndexTool[],
		options: Omit<AgentOptions, 'tools'>,
	): FunctionAgent {
		return new FunctionAgent({ ...options, tools });
	}

	run(
		input: string,
		options: {
			memory?: Memory;
			ctx?: Context;
			startAgent?: string;
			signal?: AbortSignal;
		} = {},
	): AgentRunContext {
		const events = new EventChannel();
		const controller = new AbortController();
		if (options.signal) {
			if (options.signal.aborted) controller.abort();
			else
				options.signal.addEventListener(
					'abort',
					() => controller.abort(),
					{ once: true },
				);
		}
		const memory = options.memory ?? Memory.fromDefaults();
		const context = options.ctx ?? new Context(this);
		let currentName = options.startAgent ?? this.rootAgent;
		let prompt = input;
		const task = (async (): Promise<AgentResult> => {
			for (let handoff = 0; handoff <= this.maxHandoffs; handoff += 1) {
				const current = this.agents.get(currentName);
				if (!current) throw new Error('unknown agent ' + currentName);
				const run = current.run(prompt, {
					memory,
					ctx: context,
					signal: controller.signal,
				});
				for await (const event of run.streamEvents())
					events.push(event);
				const result = await run;
				if (
					!isObject(result.raw) ||
					typeof result.raw.handoffTo !== 'string'
				)
					return result;
				currentName = result.raw.handoffTo;
				prompt =
					typeof result.raw.message === 'string' && result.raw.message
						? result.raw.message
						: result.data || prompt;
			}
			throw new Error(
				'LlamaIndex AgentWorkflow exceeded maxHandoffs=' +
					String(this.maxHandoffs),
			);
		})().finally(() => events.close());
		return new AgentRunContext(task, events, () => controller.abort());
	}
}

export * from './llamaindex_rag.js';
export * from './llamaindex_prompts.js';
export * from './llamaindex_cloud.js';
export * from './llamaindex_eval.js';

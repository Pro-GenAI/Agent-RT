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
	type ModelUsage,
	type StreamingModelProvider,
	type StructuredOutputRequirement,
	type ToolCall,
	type ToolDefinition,
} from '../../index.js';

type JSONObject = Record<string, unknown>;
type LangChainContent = string | Array<string | JSONObject>;

function env(): Record<string, string | undefined> {
	const runtime = globalThis as typeof globalThis & {
		process?: { env?: Record<string, string | undefined> };
	};
	return runtime.process?.env ?? {};
}

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function contentText(value: unknown): string {
	if (typeof value === 'string') return value;
	if (!Array.isArray(value)) return value == null ? '' : String(value);
	return value
		.map((part) => {
			if (typeof part === 'string') return part;
			if (!isObject(part)) return '';
			if (typeof part.text === 'string') return part.text;
			if (typeof part.content === 'string') return part.content;
			return '';
		})
		.join('');
}

function normalizeContent(value: unknown): ContentPart[] {
	if (typeof value === 'string')
		return value ? [{ type: 'text', text: value }] : [];
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
		if (type === 'text' && typeof item.text === 'string') {
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

export type LangChainToolCall = {
	id: string;
	name: string;
	args: Record<string, unknown>;
	type: 'tool_call';
	/** Set when the model's arguments were malformed; the call must not run. */
	argument_error?: string;
};

export type MessageFields = {
	tool_calls?: LangChainToolCall[];
	tool_call_id?: string;
	additional_kwargs?: JSONObject;
	response_metadata?: JSONObject;
	usage_metadata?: JSONObject;
	id?: string;
	name?: string;
};

export class BaseMessage {
	readonly content: LangChainContent;
	readonly role: 'system' | 'user' | 'assistant' | 'tool';
	readonly tool_calls: LangChainToolCall[];
	readonly tool_call_id?: string;
	readonly additional_kwargs: JSONObject;
	readonly response_metadata: JSONObject;
	readonly usage_metadata?: JSONObject;
	readonly id?: string;
	readonly name?: string;

	constructor(
		content: LangChainContent,
		role: BaseMessage['role'] = 'user',
		fields: MessageFields = {},
	) {
		this.content = content;
		this.role = role;
		this.tool_calls = fields.tool_calls ?? [];
		this.tool_call_id = fields.tool_call_id;
		this.additional_kwargs = fields.additional_kwargs ?? {};
		this.response_metadata = fields.response_metadata ?? {};
		this.usage_metadata = fields.usage_metadata;
		this.id = fields.id;
		this.name = fields.name;
	}

	get type(): string {
		return this.role === 'user'
			? 'human'
			: this.role === 'assistant'
				? 'ai'
				: this.role;
	}

	get text(): string {
		return contentText(this.content);
	}

	get content_blocks(): Array<string | JSONObject> {
		if (Array.isArray(this.content)) return [...this.content];
		return this.content ? [{ type: 'text', text: this.content }] : [];
	}
}

export class HumanMessage extends BaseMessage {
	constructor(content: LangChainContent, fields: MessageFields = {}) {
		super(content, 'user', fields);
	}
}

export class SystemMessage extends BaseMessage {
	constructor(content: LangChainContent, fields: MessageFields = {}) {
		super(content, 'system', fields);
	}
}

export class AIMessage extends BaseMessage {
	constructor(content: LangChainContent, fields: MessageFields = {}) {
		super(content, 'assistant', fields);
	}
}

export class AIMessageChunk extends AIMessage {
	concat(other: AIMessageChunk): AIMessageChunk {
		return new AIMessageChunk(this.text + other.text, {
			tool_calls: [...this.tool_calls, ...other.tool_calls],
			additional_kwargs: {
				...this.additional_kwargs,
				...other.additional_kwargs,
			},
			response_metadata: {
				...this.response_metadata,
				...other.response_metadata,
			},
			usage_metadata: other.usage_metadata ?? this.usage_metadata,
		});
	}
}

export class ToolMessage extends BaseMessage {
	constructor(
		content: LangChainContent,
		toolCallId: string,
		fields: MessageFields = {},
	) {
		super(content, 'tool', { ...fields, tool_call_id: toolCallId });
	}
}

export class MessagesPlaceholder {
	constructor(
		readonly variableName: string,
		readonly options: { optional?: boolean } = {},
	) {}
}

export class ChatPromptTemplate {
	constructor(
		readonly messages: Array<
			BaseMessage | MessagesPlaceholder | [string, string]
		>,
	) {}

	static fromMessages(
		messages: Array<BaseMessage | MessagesPlaceholder | [string, string]>,
	): ChatPromptTemplate {
		return new ChatPromptTemplate(messages);
	}

	formatMessages(values: Record<string, unknown>): BaseMessage[] {
		const output: BaseMessage[] = [];
		for (const item of this.messages) {
			if (item instanceof MessagesPlaceholder) {
				if (!(item.variableName in values)) {
					if (item.options.optional) continue;
					throw new Error(
						`missing prompt variable: ${item.variableName}`,
					);
				}
				output.push(
					...inputMessages(values[item.variableName]).map(asMessage),
				);
				continue;
			}
			if (item instanceof BaseMessage) {
				output.push(item);
				continue;
			}
			const [role, template] = item;
			const content = Object.entries(values).reduce(
				(current, [key, value]) =>
					current.split(`{${key}}`).join(String(value)),
				template,
			);
			const normalized = normalizeRole(role);
			if (normalized === 'system')
				output.push(new SystemMessage(content));
			else if (normalized === 'assistant')
				output.push(new AIMessage(content));
			else output.push(new HumanMessage(content));
		}
		return output;
	}

	async invoke(values: unknown): Promise<BaseMessage[]> {
		if (!isObject(values))
			throw new TypeError('ChatPromptTemplate input must be an object');
		return this.formatMessages(values);
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([this, next]);
	}
}

export type MessageLike =
	| string
	| BaseMessage
	| [string, unknown]
	| {
			role?: unknown;
			type?: unknown;
			content?: unknown;
			tool_calls?: unknown;
			tool_call_id?: unknown;
	  };

function normalizeRole(value: unknown): BaseMessage['role'] {
	const raw = String(value ?? 'user');
	if (raw === 'human') return 'user';
	if (raw === 'ai') return 'assistant';
	if (raw === 'system' || raw === 'assistant' || raw === 'tool') return raw;
	return 'user';
}

function normalizeToolCalls(value: unknown): ToolCall[] | undefined {
	if (!Array.isArray(value)) return undefined;
	const calls: ToolCall[] = [];
	for (const item of value) {
		if (!isObject(item)) continue;
		const fn = isObject(item.function) ? item.function : item;
		const args = fn.arguments ?? item.args ?? {};
		let parsed: unknown = args;
		if (typeof args === 'string') {
			try {
				parsed = JSON.parse(args || '{}');
			} catch {
				parsed = {};
			}
		}
		if (typeof (fn.name ?? item.name) !== 'string') continue;
		calls.push({
			id: String(item.id ?? ''),
			name: String(fn.name ?? item.name),
			arguments: isObject(parsed) ? parsed : {},
		});
	}
	return calls;
}

function toModelMessage(value: MessageLike): ModelMessage {
	if (typeof value === 'string') {
		return { role: 'user', content: [{ type: 'text', text: value }] };
	}
	if (value instanceof BaseMessage) {
		return {
			role: value.role,
			content: normalizeContent(value.content),
			...(value.tool_calls.length
				? {
						toolCalls: value.tool_calls.map((call) => ({
							id: call.id,
							name: call.name,
							arguments: { ...call.args },
						})),
					}
				: {}),
			...(value.tool_call_id ? { toolCallId: value.tool_call_id } : {}),
		};
	}
	if (Array.isArray(value) && value.length === 2) {
		return {
			role: normalizeRole(value[0]),
			content: normalizeContent(value[1]),
		};
	}
	return {
		role: normalizeRole(value.role ?? value.type),
		content: normalizeContent(value.content),
		...(normalizeToolCalls(value.tool_calls)
			? { toolCalls: normalizeToolCalls(value.tool_calls) }
			: {}),
		...(typeof value.tool_call_id === 'string'
			? { toolCallId: value.tool_call_id }
			: {}),
	};
}

function asMessage(value: MessageLike): BaseMessage {
	if (value instanceof BaseMessage) return value;
	if (typeof value === 'string') return new HumanMessage(value);
	if (Array.isArray(value) && value.length === 2) {
		const role = normalizeRole(value[0]);
		const content = value[1] as LangChainContent;
		if (role === 'system') return new SystemMessage(content);
		if (role === 'assistant') return new AIMessage(content);
		if (role === 'tool') return new ToolMessage(content, '');
		return new HumanMessage(content);
	}
	const role = normalizeRole(value.role ?? value.type);
	const content = (value.content ?? '') as LangChainContent;
	if (role === 'system') return new SystemMessage(content);
	if (role === 'assistant') {
		return new AIMessage(content, {
			tool_calls: normalizeToolCalls(value.tool_calls)?.map((call) => ({
				id: call.id,
				name: call.name,
				args: call.arguments,
				type: 'tool_call',
			})),
		});
	}
	if (role === 'tool') {
		return new ToolMessage(
			content,
			typeof value.tool_call_id === 'string' ? value.tool_call_id : '',
		);
	}
	return new HumanMessage(content);
}

function fromNativeModelMessage(value: ModelMessage): BaseMessage {
	const content: LangChainContent = value.content.some(
		(part) => part.type !== 'text',
	)
		? value.content.map((part) => ({
				type: part.type,
				...(part.text !== undefined ? { text: part.text } : {}),
				...(part.data !== undefined ? { data: part.data } : {}),
				...(part.mimeType !== undefined
					? { mimeType: part.mimeType }
					: {}),
			}))
		: value.content.map((part) => part.text ?? '').join('');
	if (value.role === 'system') return new SystemMessage(content);
	if (value.role === 'assistant') {
		return new AIMessage(content, {
			tool_calls: (value.toolCalls ?? []).map((call) => ({
				id: call.id,
				name: call.name,
				args: { ...call.arguments },
				type: 'tool_call',
			})),
		});
	}
	if (value.role === 'tool') {
		return new ToolMessage(content, value.toolCallId ?? '');
	}
	return new HumanMessage(content);
}

function inputMessages(value: unknown): MessageLike[] {
	if (typeof value === 'string' || value instanceof BaseMessage)
		return [value];
	if (isObject(value) && 'messages' in value)
		return inputMessages(value.messages);
	if (Array.isArray(value)) return value as MessageLike[];
	throw new TypeError(
		'LangChain model input must be a string, message, message array, or { messages }',
	);
}

function usageMetadata(usage?: ModelUsage): JSONObject | undefined {
	if (!usage) return undefined;
	return {
		input_tokens: usage.inputTokens,
		output_tokens: usage.outputTokens,
		total_tokens: usage.totalTokens,
	};
}

function fromResponse(response: ModelResponse): AIMessage {
	const toolCalls = (response.message.toolCalls ?? []).map((call) => ({
		id: call.id,
		name: call.name,
		args: { ...call.arguments },
		type: 'tool_call' as const,
		...(call.argumentError ? { argument_error: call.argumentError } : {}),
	}));
	const metadata = {
		model: response.model,
		finish_reason: response.finishReason,
	};
	const content = response.message.content.map((part) => ({
		type: part.type,
		...(part.text !== undefined ? { text: part.text } : {}),
		...(part.data !== undefined ? { data: part.data } : {}),
		...(part.mimeType !== undefined ? { mimeType: part.mimeType } : {}),
	}));
	const messageContent: LangChainContent = content.some(
		(part) => part.type !== 'text',
	)
		? (content as Array<JSONObject>)
		: content.map((part) => String(part.text ?? '')).join('');
	return new AIMessage(messageContent, {
		tool_calls: toolCalls,
		additional_kwargs: metadata,
		response_metadata: metadata,
		usage_metadata: usageMetadata(response.usage),
	});
}

function schemaFrom(value: unknown): Record<string, unknown> | undefined {
	if (isObject(value)) {
		if (typeof value.toJSONSchema === 'function') {
			const schema = (value.toJSONSchema as () => unknown)();
			return isObject(schema) ? schema : undefined;
		}
		if ('type' in value || 'properties' in value || '$schema' in value)
			return { ...value };
		if (isObject(value.schema)) return { ...value.schema };
	}
	return undefined;
}

function structuredRequirement(
	value: unknown,
): StructuredOutputRequirement | undefined {
	if (value === undefined || value === null) return undefined;
	if (
		isObject(value) &&
		value.type === 'json_schema' &&
		isObject(value.json_schema)
	) {
		const payload = value.json_schema;
		if (isObject(payload.schema)) {
			return {
				schema: { ...payload.schema },
				...(typeof payload.name === 'string'
					? { name: payload.name }
					: {}),
				strict: payload.strict !== false,
			};
		}
	}
	const schema = schemaFrom(value);
	if (schema) return { schema, strict: true };
	if (
		isObject(value) &&
		(typeof value.parse === 'function' ||
			typeof value.safeParse === 'function')
	) {
		return { schema: {}, strict: false };
	}
	throw new TypeError(
		'structured output requires JSON Schema or an object exposing toJSONSchema(), parse(), or safeParse()',
	);
}

export type ToolOptions = {
	name: string;
	description?: string;
	schema?: Record<string, unknown> | { toJSONSchema(): unknown };
};

export type LangChainTool = {
	name: string;
	description: string;
	schema: Record<string, unknown>;
	invoke(input: unknown): Promise<unknown>;
};

export function tool<TInput = unknown>(
	handler: (input: TInput) => unknown | Promise<unknown>,
	options: ToolOptions,
): LangChainTool {
	const schema = schemaFrom(
		options.schema ?? { type: 'object', properties: {} },
	);
	if (!schema)
		throw new TypeError(
			'tool schema must be JSON Schema or expose toJSONSchema()',
		);
	return {
		name: options.name,
		description: options.description ?? '',
		schema,
		async invoke(input: unknown) {
			return await handler(input as TInput);
		},
	};
}

function toolDefinition(value: unknown): ToolDefinition {
	if (
		isObject(value) &&
		value.type === 'function' &&
		isObject(value.function)
	) {
		const fn = value.function;
		if (typeof fn.name !== 'string' || !fn.name)
			throw new TypeError('tool function name is required');
		return {
			name: fn.name,
			description:
				typeof fn.description === 'string' ? fn.description : '',
			inputSchema: isObject(fn.parameters)
				? { ...fn.parameters }
				: { type: 'object' },
		};
	}
	if (isObject(value) && typeof value.name === 'string') {
		const schema = schemaFrom(value.schema) ??
			(isObject(value.inputSchema) ? value.inputSchema : undefined) ??
			(isObject(value.input_schema) ? value.input_schema : undefined) ?? {
				type: 'object',
			};
		return {
			name: value.name,
			description:
				typeof value.description === 'string' ? value.description : '',
			inputSchema: { ...schema },
		};
	}
	throw new TypeError(
		'tool must be an OpenAI function schema or LangChain-like tool object',
	);
}

function requestFor(
	model: ChatModel,
	input: unknown,
	overrides: ChatInvokeOptions = {},
): ModelRequest {
	const tools = overrides.tools ?? model.boundTools;
	return {
		model: model.model,
		messages: inputMessages(input).map(toModelMessage),
		...(tools.length ? { tools: tools.map(toolDefinition) } : {}),
		...((overrides.temperature ?? model.temperature) !== undefined
			? { temperature: overrides.temperature ?? model.temperature }
			: {}),
		...((overrides.maxTokens ?? model.maxTokens) !== undefined
			? { maxOutputTokens: overrides.maxTokens ?? model.maxTokens }
			: {}),
		...((overrides.responseFormat ?? model.responseFormat) !== undefined
			? {
					structuredOutput: structuredRequirement(
						overrides.responseFormat ?? model.responseFormat,
					),
				}
			: {}),
	};
}

export type ChatInvokeOptions = {
	temperature?: number;
	maxTokens?: number;
	tools?: unknown[];
	responseFormat?: unknown;
	config?: RunnableConfig;
};

export type ChatModelOptions = {
	model: string;
	provider?: ModelProvider;
	baseURL?: string;
	apiKey?: string;
	temperature?: number;
	maxTokens?: number;
	tools?: unknown[];
	responseFormat?: unknown;
};

function isStreamingProvider(
	provider: ModelProvider,
): provider is StreamingModelProvider {
	return typeof (provider as StreamingModelProvider).stream === 'function';
}

export class ChatModel {
	readonly model: string;
	readonly provider: ModelProvider;
	readonly temperature?: number;
	readonly maxTokens?: number;
	readonly boundTools: unknown[];
	readonly responseFormat?: unknown;

	constructor(options: ChatModelOptions) {
		this.model = options.model;
		this.provider =
			options.provider ??
			new OpenAIModelProvider({
				baseUrl:
					options.baseURL ??
					env().OPENAI_BASE_URL ??
					DEFAULT_OPENAI_BASE_URL,
				apiKey: options.apiKey ?? env().OPENAI_API_KEY,
				defaultModel: options.model,
			});
		this.temperature = options.temperature;
		this.maxTokens = options.maxTokens;
		this.boundTools = [...(options.tools ?? [])];
		this.responseFormat = options.responseFormat;
	}

	protected clone(
		changes: Partial<Pick<ChatModel, 'boundTools' | 'responseFormat'>>,
	): this {
		const clone = Object.create(Object.getPrototypeOf(this)) as this;
		Object.assign(clone, this, changes);
		return clone;
	}

	bind(options: ChatInvokeOptions): this {
		return this.clone({
			responseFormat: options.responseFormat ?? this.responseFormat,
			boundTools: options.tools ?? this.boundTools,
		});
	}

	bindTools(tools: unknown[]): this {
		return this.clone({ boundTools: [...tools] });
	}

	withRetry(options: { stopAfterAttempt?: number } = {}): RunnableRetry {
		return new RunnableRetry(this, options.stopAfterAttempt ?? 3);
	}

	withFallbacks(fallbacks: RunnableLike[]): RunnableFallbacks {
		return new RunnableFallbacks(this, fallbacks);
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([this, next]);
	}

	configurableFields(fields: Record<string, string>): ConfigurableRunnable {
		return new ConfigurableRunnable(this, fields);
	}

	async invoke(
		input: unknown,
		options: ChatInvokeOptions = {},
	): Promise<AIMessage> {
		await emitRunnableCallbacks(options.config, { type: 'start', input });
		try {
			const output = fromResponse(
				await this.provider.complete(requestFor(this, input, options)),
			);
			await emitRunnableCallbacks(options.config, {
				type: 'end',
				input,
				output,
			});
			return output;
		} catch (error) {
			await emitRunnableCallbacks(options.config, {
				type: 'error',
				input,
				error,
			});
			throw error;
		}
	}

	async batch(
		inputs: unknown[],
		options: ChatInvokeOptions | ChatInvokeOptions[] = {},
	): Promise<AIMessage[]> {
		const maxConcurrency = Array.isArray(options)
			? inputs.length || 1
			: (options.config?.maxConcurrency ?? inputs.length) || 1;
		const results = new Array<AIMessage>(inputs.length);
		let next = 0;
		async function worker(model: ChatModel): Promise<void> {
			while (next < inputs.length) {
				const index = next;
				next += 1;
				const [input] = inputs.slice(index, index + 1);
				if (input === undefined) continue;
				const [inputOptions] = Array.isArray(options)
					? options.slice(index, index + 1)
					: [options];
				const result = await model.invoke(input, inputOptions ?? {});
				results.splice(index, 1, result);
			}
		}
		await Promise.all(
			Array.from(
				{ length: Math.min(maxConcurrency, inputs.length) },
				async () => await worker(this),
			),
		);
		return results;
	}

	async *stream(
		input: unknown,
		options: ChatInvokeOptions = {},
	): AsyncIterable<AIMessageChunk> {
		const request = requestFor(this, input, options);
		if (!isStreamingProvider(this.provider)) {
			yield new AIMessageChunk((await this.invoke(input, options)).text);
			return;
		}
		const toolBuffers = new Map<
			string,
			{ name: string; arguments: string }
		>();
		for await (const event of this.provider.stream(request)) {
			if (event.type === 'text_delta' && event.text) {
				yield new AIMessageChunk(event.text);
			} else if (event.type === 'tool_call_delta') {
				const id = event.toolCallId ?? '';
				const current = toolBuffers.get(id) ?? {
					name: '',
					arguments: '',
				};
				if (event.toolName) current.name = event.toolName;
				current.arguments += event.argumentsDelta ?? '';
				toolBuffers.set(id, current);
				let args: Record<string, unknown> = {};
				try {
					const parsed = JSON.parse(current.arguments || '{}');
					if (isObject(parsed)) args = parsed;
				} catch {
					args = {};
				}
				yield new AIMessageChunk('', {
					tool_calls: [
						{ id, name: current.name, args, type: 'tool_call' },
					],
					additional_kwargs: {
						tool_call_chunk: {
							id,
							name: current.name,
							args: current.arguments,
						},
					},
				});
			} else if (event.type === 'completed' && event.response) {
				const final = fromResponse(event.response);
				if (final.tool_calls.length || final.usage_metadata) {
					yield new AIMessageChunk('', {
						tool_calls: final.tool_calls,
						response_metadata: final.response_metadata,
						usage_metadata: final.usage_metadata,
					});
				}
			}
		}
	}

	withStructuredOutput(
		schema: unknown,
		options: { includeRaw?: boolean } = {},
	): StructuredRunnable {
		return new StructuredRunnable(
			this.clone({ responseFormat: schema }),
			schema,
			options.includeRaw ?? false,
		);
	}
}

export class ChatOpenAI extends ChatModel {
	constructor(options: ChatModelOptions) {
		super({
			...options,
			provider:
				options.provider ??
				new OpenAIModelProvider({
					baseUrl:
						options.baseURL ??
						env().OPENAI_BASE_URL ??
						DEFAULT_OPENAI_BASE_URL,
					apiKey: options.apiKey ?? env().OPENAI_API_KEY,
					defaultModel: options.model,
				}),
		});
	}
}

export class ChatAnthropic extends ChatModel {
	constructor(options: ChatModelOptions) {
		super({
			...options,
			provider:
				options.provider ??
				new AnthropicModelProvider({
					baseUrl:
						options.baseURL ??
						env().ANTHROPIC_BASE_URL ??
						DEFAULT_ANTHROPIC_BASE_URL,
					apiKey: options.apiKey ?? env().ANTHROPIC_API_KEY,
					defaultModel: options.model,
				}),
		});
	}
}

function validateJSONSchemaValue(
	schema: Record<string, unknown>,
	value: unknown,
	path = '$',
): void {
	const expected = schema.type;
	const matches =
		expected === 'object'
			? isObject(value)
			: expected === 'array'
				? Array.isArray(value)
				: expected === 'string'
					? typeof value === 'string'
					: expected === 'integer'
						? Number.isInteger(value) && typeof value === 'number'
						: expected === 'number'
							? typeof value === 'number' &&
								Number.isFinite(value)
							: expected === 'boolean'
								? typeof value === 'boolean'
								: expected === 'null'
									? value === null
									: true;
	if (!matches) throw new Error(`${path} must be ${String(expected)}`);
	if (
		Array.isArray(schema.enum) &&
		!schema.enum.some((item) => Object.is(item, value))
	) {
		throw new Error(`${path} must be one of the allowed enum values`);
	}
	if (expected === 'object' && isObject(value)) {
		if (Array.isArray(schema.required)) {
			for (const key of schema.required) {
				if (typeof key === 'string' && !(key in value)) {
					throw new Error(`${path}.${key} is required`);
				}
			}
		}
		if (isObject(schema.properties)) {
			for (const [key, child] of Object.entries(schema.properties)) {
				if (key in value && isObject(child)) {
					validateJSONSchemaValue(
						child,
						value[key],
						`${path}.${key}`,
					);
				}
			}
		}
	}
	if (
		expected === 'array' &&
		Array.isArray(value) &&
		isObject(schema.items)
	) {
		for (const [index, item] of value.entries()) {
			validateJSONSchemaValue(schema.items, item, `${path}[${index}]`);
		}
	}
}

function validateStructuredValue(schema: unknown, value: unknown): unknown {
	if (isObject(schema) && typeof schema.safeParse === 'function') {
		const result = (schema.safeParse as (input: unknown) => unknown)(value);
		if (isObject(result) && result.success === true) return result.data;
		if (isObject(result) && result.success === false) throw result.error;
	}
	if (isObject(schema) && typeof schema.parse === 'function') {
		return (schema.parse as (input: unknown) => unknown)(value);
	}
	const jsonSchema = schemaFrom(schema);
	if (jsonSchema) validateJSONSchemaValue(jsonSchema, value);
	return value;
}

export class ProviderStrategy {
	readonly kind = 'provider';
	constructor(
		readonly schema: unknown,
		readonly options: { strict?: boolean; maxRetries?: number } = {},
	) {}
	get strict(): boolean {
		return this.options.strict !== false;
	}
	get maxRetries(): number {
		return this.options.maxRetries ?? 1;
	}
}

export class ToolStrategy {
	readonly kind = 'tool';
	constructor(
		readonly schema: unknown,
		readonly options: {
			toolName?: string;
			handleErrors?: boolean;
			maxRetries?: number;
			toolMessageContent?: string;
		} = {},
	) {}
	get toolName(): string {
		return this.options.toolName ?? 'structured_response';
	}
	get handleErrors(): boolean {
		return this.options.handleErrors !== false;
	}
	get maxRetries(): number {
		return this.options.maxRetries ?? 2;
	}
}

export class AutoStrategy {
	readonly kind = 'auto';
	constructor(
		readonly schema: unknown,
		readonly options: { strict?: boolean; maxRetries?: number } = {},
	) {}
	get strict(): boolean {
		return this.options.strict !== false;
	}
	get maxRetries(): number {
		return this.options.maxRetries ?? 1;
	}
}

type ResponseStrategy = ProviderStrategy | ToolStrategy | AutoStrategy;

function responseStrategy(value: unknown): ResponseStrategy | undefined {
	if (value === undefined || value === null) return undefined;
	if (
		value instanceof ProviderStrategy ||
		value instanceof ToolStrategy ||
		value instanceof AutoStrategy
	)
		return value;
	return new AutoStrategy(value);
}

function strategyTool(
	strategy: ResponseStrategy | undefined,
): LangChainTool | undefined {
	if (!(strategy instanceof ToolStrategy)) return undefined;
	const schema = schemaFrom(strategy.schema);
	if (!schema) {
		throw new TypeError(
			'ToolStrategy requires JSON Schema or an object exposing toJSONSchema()',
		);
	}
	return {
		name: strategy.toolName,
		description: 'Return the final structured response.',
		schema,
		async invoke(input: unknown) {
			return input;
		},
	};
}

function parseStructured(schema: unknown, text: string): unknown {
	return validateStructuredValue(schema, JSON.parse(text));
}

export class StructuredRunnable {
	constructor(
		readonly model: ChatModel,
		readonly schema: unknown,
		readonly includeRaw = false,
	) {}

	withRetry(options: { stopAfterAttempt?: number } = {}): RunnableRetry {
		return new RunnableRetry(this, options.stopAfterAttempt ?? 3);
	}

	withFallbacks(fallbacks: RunnableLike[]): RunnableFallbacks {
		return new RunnableFallbacks(this, fallbacks);
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([this, next]);
	}

	configurableFields(fields: Record<string, string>): ConfigurableRunnable {
		return new ConfigurableRunnable(this, fields);
	}

	async invoke(
		input: unknown,
		options: ChatInvokeOptions = {},
	): Promise<unknown> {
		const raw = await this.model.invoke(input, options);
		try {
			const parsed = parseStructured(this.schema, raw.text);
			return this.includeRaw
				? { raw, parsed, parsing_error: null }
				: parsed;
		} catch (error) {
			if (this.includeRaw)
				return { raw, parsed: null, parsing_error: error };
			throw error;
		}
	}

	async batch(
		inputs: unknown[],
		options: ChatInvokeOptions = {},
	): Promise<unknown[]> {
		return await Promise.all(
			inputs.map((input) => this.invoke(input, options)),
		);
	}
}

export function initChatModel(
	model: string,
	options: Omit<ChatModelOptions, 'model'> & { modelProvider?: string } = {},
): ChatModel {
	const provider =
		options.modelProvider ??
		(model.includes(':') ? model.split(':', 1)[0] : 'openai');
	const modelName = model.includes(':')
		? model.slice(model.indexOf(':') + 1)
		: model;
	if (provider === 'anthropic')
		return new ChatAnthropic({ ...options, model: modelName });
	if (provider === 'openai')
		return new ChatOpenAI({ ...options, model: modelName });
	throw new Error(`unsupported LangChain model provider: ${provider}`);
}

export type AgentRuntimeContext = {
	context: Record<string, unknown>;
	state: Record<string, unknown>;
	store?: InMemoryStore | unknown;
	config?: Record<string, unknown>;
};

export class MemorySaver {
	private readonly threads = new Map<
		string,
		{ messages: MessageLike[]; state: Record<string, unknown> }
	>();

	get(
		threadId: string,
	): { messages: MessageLike[]; state: Record<string, unknown> } | undefined {
		const value = this.threads.get(String(threadId));
		if (!value) return undefined;
		return { messages: [...value.messages], state: { ...value.state } };
	}

	put(
		threadId: string,
		value: { messages: MessageLike[]; state: Record<string, unknown> },
	): void {
		this.threads.set(String(threadId), {
			messages: [...value.messages],
			state: { ...value.state },
		});
	}

	delete(threadId: string): boolean {
		return this.threads.delete(String(threadId));
	}
}

export class InMemoryStore {
	private readonly values = new Map<string, unknown>();

	private key(namespace: string | string[], key: string): string {
		const ns = Array.isArray(namespace) ? namespace : [namespace];
		return JSON.stringify([ns, key]);
	}

	put(namespace: string | string[], key: string, value: unknown): void {
		this.values.set(this.key(namespace, key), value);
	}

	get(namespace: string | string[], key: string): unknown {
		return this.values.get(this.key(namespace, key));
	}

	delete(namespace: string | string[], key: string): boolean {
		return this.values.delete(this.key(namespace, key));
	}

	search(
		namespace: string | string[],
	): Array<{ key: string; value: unknown }> {
		const ns = Array.isArray(namespace) ? namespace : [namespace];
		const prefix = JSON.stringify(ns).slice(0, -1);
		const result: Array<{ key: string; value: unknown }> = [];
		for (const [encoded, value] of this.values) {
			if (!encoded.includes(prefix)) continue;
			const parsed = JSON.parse(encoded) as [string[], string];
			result.push({ key: parsed[1], value });
		}
		return result;
	}
}

export interface AgentMiddleware {
	beforeModel?(
		runtime: AgentRuntimeContext,
		messages: MessageLike[],
	): MessageLike[] | Promise<MessageLike[]>;
	afterModel?(
		runtime: AgentRuntimeContext,
		response: AIMessage,
	): AIMessage | Promise<AIMessage>;
	wrapToolCall?(
		runtime: AgentRuntimeContext,
		call: LangChainToolCall,
		handler: (call: LangChainToolCall) => Promise<unknown>,
	): Promise<unknown>;
	afterAgent?(
		runtime: AgentRuntimeContext,
		result: Record<string, unknown>,
	): Record<string, unknown> | Promise<Record<string, unknown>>;
	selectTools?(runtime: AgentRuntimeContext, tools: unknown[]): unknown[];
}

export class ModelCallLimitMiddleware implements AgentMiddleware {
	private calls = 0;
	constructor(readonly maxCalls: number) {}
	beforeModel(
		_runtime: AgentRuntimeContext,
		messages: MessageLike[],
	): MessageLike[] {
		this.calls += 1;
		if (this.calls > this.maxCalls)
			throw new Error(`model call limit exceeded: ${this.maxCalls}`);
		return messages;
	}
}

export class ToolCallLimitMiddleware implements AgentMiddleware {
	private calls = 0;
	constructor(readonly maxCalls: number) {}
	async wrapToolCall(
		_runtime: AgentRuntimeContext,
		call: LangChainToolCall,
		handler: (call: LangChainToolCall) => Promise<unknown>,
	): Promise<unknown> {
		this.calls += 1;
		if (this.calls > this.maxCalls)
			throw new Error(`tool call limit exceeded: ${this.maxCalls}`);
		return await handler(call);
	}
}

export class HumanInTheLoopMiddleware implements AgentMiddleware {
	constructor(
		readonly approval: (
			call: LangChainToolCall,
			runtime: AgentRuntimeContext,
		) => boolean | Promise<boolean>,
	) {}
	async wrapToolCall(
		runtime: AgentRuntimeContext,
		call: LangChainToolCall,
		handler: (call: LangChainToolCall) => Promise<unknown>,
	): Promise<unknown> {
		if (!(await this.approval(call, runtime)))
			throw new Error(`tool call ${call.name} requires approval`);
		return await handler(call);
	}
}

export class SummarizationMiddleware implements AgentMiddleware {
	constructor(readonly maxMessages = 20) {}
	beforeModel(
		_runtime: AgentRuntimeContext,
		messages: MessageLike[],
	): MessageLike[] {
		if (messages.length <= this.maxMessages) return messages;
		const keep = Math.max(2, this.maxMessages - 1);
		const prefix = messages
			.slice(0, -keep)
			.map((message) => asMessage(message).text);
		return [
			new SystemMessage(`Conversation summary:\n${prefix.join('\n')}`),
			...messages.slice(-keep),
		];
	}
}

export class PIIMiddleware implements AgentMiddleware {
	constructor(readonly patterns: RegExp[] = []) {}
	beforeModel(
		_runtime: AgentRuntimeContext,
		messages: MessageLike[],
	): MessageLike[] {
		return messages.map((value) => {
			const message = asMessage(value);
			if (typeof message.content !== 'string') return value;
			let content = message.content;
			for (const pattern of this.patterns)
				content = content.replace(pattern, '[REDACTED]');
			if (content === message.content) return value;
			if (message.role === 'system') return new SystemMessage(content);
			if (message.role === 'assistant')
				return new AIMessage(content, {
					tool_calls: message.tool_calls,
				});
			if (message.role === 'tool')
				return new ToolMessage(content, message.tool_call_id ?? '');
			return new HumanMessage(content);
		});
	}
}

export class LLMToolSelectorMiddleware implements AgentMiddleware {
	constructor(
		readonly selector: (
			runtime: AgentRuntimeContext,
			tools: unknown[],
		) => unknown[],
	) {}
	selectTools(runtime: AgentRuntimeContext, tools: unknown[]): unknown[] {
		return this.selector(runtime, tools);
	}
}

function validateAgentSchema(
	schema: unknown,
	value: Record<string, unknown>,
	label: string,
): Record<string, unknown> {
	if (schema === undefined || schema === null) return value;
	if (isObject(schema) && typeof schema.parse === 'function') {
		return schema.parse(value) as Record<string, unknown>;
	}
	if (isObject(schema) && typeof schema.safeParse === 'function') {
		const result = schema.safeParse(value) as {
			success: boolean;
			data?: unknown;
			error?: unknown;
		};
		if (!result.success)
			throw result.error ?? new Error(`${label} validation failed`);
		return result.data as Record<string, unknown>;
	}
	if (isObject(schema) && Array.isArray(schema.required)) {
		for (const key of schema.required) {
			if (typeof key === 'string' && !(key in value))
				throw new Error(`${label} missing required field: ${key}`);
		}
	}
	return value;
}

function threadIdFromConfig(
	config?: Record<string, unknown>,
): string | undefined {
	const configurable = isObject(config?.configurable)
		? config?.configurable
		: undefined;
	const value = configurable?.thread_id ?? configurable?.threadId;
	return value === undefined || value === null ? undefined : String(value);
}

type AgentOptions = {
	model?: ChatModel | string;
	llm?: ChatModel | string;
	tools?: unknown[];
	systemPrompt?: string;
	responseFormat?: unknown;
	maxIterations?: number;
	middleware?: AgentMiddleware[];
	checkpointer?:
		| CheckpointStore
		| {
				get?(threadId: string):
					| {
							messages: MessageLike[];
							state: Record<string, unknown>;
					  }
					| undefined;
				put?(
					threadId: string,
					value: {
						messages: MessageLike[];
						state: Record<string, unknown>;
					},
				): unknown;
		  };
	store?: unknown;
	contextSchema?: unknown;
	stateSchema?: unknown;
};

function findTool(tools: unknown[], name: string): unknown {
	return tools.find(
		(candidate) => isObject(candidate) && candidate.name === name,
	);
}

async function invokeTool(
	candidate: unknown,
	call: LangChainToolCall,
	runtime?: AgentRuntimeContext,
): Promise<unknown> {
	if (!isObject(candidate)) throw new Error(`tool not found: ${call.name}`);
	if (typeof candidate.invoke === 'function') {
		return await (candidate.invoke as (input: unknown) => unknown)({
			...call.args,
			...(runtime ? { runtime } : {}),
		});
	}
	if (typeof candidate.func === 'function') {
		return await (
			candidate.func as (
				input: unknown,
				runtime?: AgentRuntimeContext,
			) => unknown
		)(call.args, runtime);
	}
	throw new Error(`tool ${call.name} is not executable`);
}

export class AgentRunnable {
	readonly model: ChatModel;
	readonly tools: unknown[];
	readonly systemPrompt?: string;
	readonly responseFormat?: unknown;
	readonly responseStrategy?: ResponseStrategy;
	readonly maxIterations: number;
	readonly middleware: AgentMiddleware[];
	readonly checkpointer?: AgentOptions['checkpointer'];
	readonly store?: unknown;
	readonly contextSchema?: unknown;
	readonly stateSchema?: unknown;

	constructor(options: AgentOptions) {
		const selectedModel = options.model ?? options.llm;
		if (!selectedModel) {
			throw new Error(
				'LangChain compatibility createAgent requires model or llm',
			);
		}
		this.model =
			typeof selectedModel === 'string'
				? initChatModel(selectedModel)
				: selectedModel;
		this.tools = [...(options.tools ?? [])];
		this.systemPrompt = options.systemPrompt;
		this.responseStrategy = responseStrategy(options.responseFormat);
		this.responseFormat = this.responseStrategy?.schema;
		this.maxIterations = options.maxIterations ?? 8;
		this.middleware = [...(options.middleware ?? [])];
		this.checkpointer = options.checkpointer;
		this.store = options.store;
		this.contextSchema = options.contextSchema;
		this.stateSchema = options.stateSchema;
	}

	private prepare(
		input: unknown,
		config: Record<string, unknown> = {},
	): {
		messages: MessageLike[];
		runtime: AgentRuntimeContext;
		threadId?: string;
	} {
		const threadId = threadIdFromConfig(config);
		let checkpoint:
			| { messages: MessageLike[]; state: Record<string, unknown> }
			| undefined;
		if (threadId && this.checkpointer) {
			if (
				'load' in this.checkpointer &&
				typeof this.checkpointer.load === 'function'
			) {
				const native = this.checkpointer.load(threadId);
				if (native) {
					const nativeState = isObject(native.metadata?.state)
						? (native.metadata?.state as Record<string, unknown>)
						: {};
					checkpoint = {
						messages: native.messages.map(fromNativeModelMessage),
						state: { ...nativeState },
					};
				}
			} else if (
				'get' in this.checkpointer &&
				typeof this.checkpointer.get === 'function'
			) {
				checkpoint = this.checkpointer.get(threadId);
			}
		}
		const incoming = inputMessages(input);
		const messages = checkpoint
			? [...checkpoint.messages, ...incoming]
			: [...incoming];
		if (
			this.systemPrompt &&
			(!messages.length || asMessage(messages[0]).role !== 'system')
		) {
			messages.unshift(new SystemMessage(this.systemPrompt));
		}
		const inputState =
			isObject(input) && isObject(input.state) ? input.state : {};
		const state = validateAgentSchema(
			this.stateSchema,
			{ ...(checkpoint?.state ?? {}), ...inputState },
			'state',
		);
		const configurable = isObject(config.configurable)
			? config.configurable
			: {};
		const contextValue = isObject(configurable.context)
			? configurable.context
			: isObject(configurable.runtimeContext)
				? configurable.runtimeContext
				: {};
		const context = validateAgentSchema(
			this.contextSchema,
			contextValue,
			'context',
		);
		return {
			messages,
			runtime: { context, state, store: this.store, config },
			threadId,
		};
	}

	private save(
		threadId: string | undefined,
		messages: MessageLike[],
		state: Record<string, unknown>,
	): void {
		if (!threadId || !this.checkpointer) return;
		if (
			'save' in this.checkpointer &&
			typeof this.checkpointer.save === 'function'
		) {
			const serializableState = JSON.parse(
				JSON.stringify(state),
			) as Record<string, JSONValue>;
			const checkpoint: AgentCheckpoint = {
				checkpointId: threadId,
				agentName: 'langchain-compat',
				messages: messages.map(toModelMessage),
				turns: 0,
				toolCalls: 0,
				totalTokens: 0,
				metadata: { state: serializableState },
				createdAtMs: Date.now(),
			};
			this.checkpointer.save(checkpoint);
			return;
		}
		if (
			'put' in this.checkpointer &&
			typeof this.checkpointer.put === 'function'
		) {
			this.checkpointer.put(threadId, {
				messages: [...messages],
				state: { ...state },
			});
		}
	}

	private async run(
		input: unknown,
		config: Record<string, unknown> = {},
		emit?: (
			node: 'model' | 'tools',
			messages: BaseMessage[],
		) => Promise<void>,
	): Promise<Record<string, unknown>> {
		const { messages, runtime, threadId } = this.prepare(input, config);
		let structuredRetries = 0;
		for (
			let iteration = 0;
			iteration < this.maxIterations;
			iteration += 1
		) {
			let modelMessages = messages;
			for (const middleware of this.middleware) {
				if (middleware.beforeModel)
					modelMessages = await middleware.beforeModel(
						runtime,
						modelMessages,
					);
			}
			let selectedTools = [...this.tools];
			for (const middleware of this.middleware) {
				if (middleware.selectTools)
					selectedTools = middleware.selectTools(
						runtime,
						selectedTools,
					);
			}
			const outputTool = strategyTool(this.responseStrategy);
			const modelTools = outputTool
				? [...selectedTools, outputTool]
				: selectedTools;
			const model = this.model.bindTools(modelTools);
			const invokeOptions: ChatInvokeOptions = {};
			if (
				this.responseStrategy &&
				!(this.responseStrategy instanceof ToolStrategy)
			) {
				invokeOptions.responseFormat = this.responseFormat;
			}
			let response = await model.invoke(modelMessages, invokeOptions);
			for (const middleware of [...this.middleware].reverse()) {
				if (middleware.afterModel)
					response = await middleware.afterModel(runtime, response);
			}
			messages.push(response);
			if (emit) await emit('model', [response]);
			if (this.responseStrategy instanceof ToolStrategy) {
				const toolStrategy = this.responseStrategy;
				const structuredCall = response.tool_calls.find(
					(call) => call.name === toolStrategy.toolName,
				);
				if (structuredCall) {
					try {
						const parsed = validateStructuredValue(
							this.responseStrategy.schema,
							structuredCall.args,
						);
						let result: Record<string, unknown> = {
							messages,
							state: runtime.state,
							structuredResponse: parsed,
						};
						for (const middleware of [
							...this.middleware,
						].reverse()) {
							if (middleware.afterAgent)
								result = await middleware.afterAgent(
									runtime,
									result,
								);
						}
						this.save(threadId, messages, runtime.state);
						return result;
					} catch (error) {
						if (
							!this.responseStrategy.handleErrors ||
							structuredRetries >=
								this.responseStrategy.maxRetries
						)
							throw error;
						structuredRetries += 1;
						const errorMessage = new ToolMessage(
							`Structured response validation failed: ${String(error)}`,
							structuredCall.id,
							{ name: this.responseStrategy.toolName },
						);
						messages.push(errorMessage);
						if (emit) await emit('tools', [errorMessage]);
						continue;
					}
				}
			}
			if (!response.tool_calls.length) {
				let result: Record<string, unknown> = {
					messages,
					state: runtime.state,
				};
				if (this.responseFormat !== undefined) {
					try {
						result.structuredResponse = parseStructured(
							this.responseFormat,
							response.text,
						);
					} catch (error) {
						const retryLimit =
							this.responseStrategy?.maxRetries ?? 0;
						if (structuredRetries >= retryLimit) throw error;
						structuredRetries += 1;
						messages.push(
							new SystemMessage(
								'The previous structured response failed validation. Return only a valid response matching the required schema.',
							),
						);
						continue;
					}
				}
				for (const middleware of [...this.middleware].reverse()) {
					if (middleware.afterAgent)
						result = await middleware.afterAgent(runtime, result);
				}
				this.save(threadId, messages, runtime.state);
				return result;
			}
			const toolMessages: ToolMessage[] = [];
			for (const call of response.tool_calls) {
				const candidate = findTool(selectedTools, call.name);
				if (call.argument_error) {
					// Malformed model arguments are never executed, not even with
					// defaults; the model gets the error and can retry.
					const rejected = new ToolMessage(
						`Tool call rejected: ${call.argument_error}`,
						call.id,
						{ name: call.name },
					);
					toolMessages.push(rejected);
					messages.push(rejected);
					continue;
				}
				let handler = async (
					currentCall: LangChainToolCall,
				): Promise<unknown> =>
					await invokeTool(candidate, currentCall, runtime);
				for (const middleware of [...this.middleware].reverse()) {
					if (!middleware.wrapToolCall) continue;
					const inner = handler;
					handler = async (currentCall) =>
						await middleware.wrapToolCall?.(
							runtime,
							currentCall,
							inner,
						);
				}
				const output = await handler(call);
				const toolMessage = new ToolMessage(
					typeof output === 'string'
						? output
						: JSON.stringify(output),
					call.id,
					{ name: call.name },
				);
				toolMessages.push(toolMessage);
				messages.push(toolMessage);
			}
			if (emit) await emit('tools', toolMessages);
		}
		throw new Error(
			`LangChain compatibility agent exceeded maxIterations=${this.maxIterations}`,
		);
	}

	async invoke(
		input: unknown,
		config: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		return await this.run(input, config);
	}

	async *stream(
		input: unknown,
		options: {
			streamMode?: 'updates' | 'messages' | 'values';
			config?: Record<string, unknown>;
		} = {},
	): AsyncIterable<unknown> {
		const mode = options.streamMode ?? 'updates';
		const queue: unknown[] = [];
		const emit = async (
			node: 'model' | 'tools',
			emittedMessages: BaseMessage[],
		): Promise<void> => {
			if (mode === 'messages') {
				for (const message of emittedMessages)
					queue.push([message, { langgraph_node: node }]);
				return;
			}
			if (mode === 'values') {
				queue.push({ messages: [...emittedMessages] });
				return;
			}
			queue.push({ [node]: { messages: emittedMessages } });
		};
		const task = this.run(input, options.config ?? {}, emit);
		while (true) {
			while (queue.length) yield queue.shift();
			const settled = await Promise.race([
				task.then(() => true),
				new Promise<boolean>((resolve) =>
					setTimeout(() => resolve(false), 0),
				),
			]);
			if (settled) {
				while (queue.length) yield queue.shift();
				await task;
				return;
			}
		}
	}
}

export function createAgent(options: AgentOptions): AgentRunnable {
	return new AgentRunnable(options);
}

export type RunnableEvent = {
	type: 'start' | 'end' | 'error';
	input: unknown;
	output?: unknown;
	error?: unknown;
};

export type RunnableCallback =
	| ((event: RunnableEvent) => unknown | Promise<unknown>)
	| {
			handleChainStart?(event: RunnableEvent): unknown | Promise<unknown>;
			handleChainEnd?(event: RunnableEvent): unknown | Promise<unknown>;
			handleChainError?(event: RunnableEvent): unknown | Promise<unknown>;
	  };

export type RunnableConfig = {
	callbacks?: RunnableCallback[];
	maxConcurrency?: number;
	configurable?: Record<string, unknown>;
};

type RunnableLike = {
	invoke(input: unknown, options?: Record<string, unknown>): Promise<unknown>;
};

async function emitRunnableCallbacks(
	config: RunnableConfig | undefined,
	event: RunnableEvent,
): Promise<void> {
	for (const callback of config?.callbacks ?? []) {
		if (typeof callback === 'function') {
			await callback(event);
			continue;
		}
		const handler =
			event.type === 'start'
				? callback.handleChainStart
				: event.type === 'end'
					? callback.handleChainEnd
					: callback.handleChainError;
		if (handler) await handler.call(callback, event);
	}
}

export class RunnableLambda implements RunnableLike {
	constructor(
		readonly func: (
			input: unknown,
			options?: Record<string, unknown>,
		) => unknown | Promise<unknown>,
	) {}

	async invoke(
		input: unknown,
		options: Record<string, unknown> = {},
	): Promise<unknown> {
		return await this.func(input, options);
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([this, next]);
	}
}

export class RunnablePassthrough implements RunnableLike {
	async invoke(input: unknown): Promise<unknown> {
		return input;
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([this, next]);
	}
}

export class RunnableSequence implements RunnableLike {
	constructor(readonly steps: RunnableLike[]) {}

	async invoke(
		input: unknown,
		options: Record<string, unknown> = {},
	): Promise<unknown> {
		let value = input;
		for (const step of this.steps)
			value = await step.invoke(value, options);
		return value;
	}

	pipe(next: RunnableLike): RunnableSequence {
		return new RunnableSequence([...this.steps, next]);
	}
}

export class RunnableRetry implements RunnableLike {
	constructor(
		readonly runnable: RunnableLike,
		readonly maxAttempts = 3,
	) {
		if (maxAttempts < 1) throw new Error('maxAttempts must be at least 1');
	}

	async invoke(
		input: unknown,
		options: Record<string, unknown> = {},
	): Promise<unknown> {
		let lastError: unknown;
		for (let attempt = 0; attempt < this.maxAttempts; attempt += 1) {
			try {
				return await this.runnable.invoke(input, options);
			} catch (error) {
				lastError = error;
			}
		}
		throw lastError;
	}
}

export class RunnableFallbacks implements RunnableLike {
	constructor(
		readonly runnable: RunnableLike,
		readonly fallbacks: RunnableLike[],
	) {}

	async invoke(
		input: unknown,
		options: Record<string, unknown> = {},
	): Promise<unknown> {
		let lastError: unknown;
		for (const candidate of [this.runnable, ...this.fallbacks]) {
			try {
				return await candidate.invoke(input, options);
			} catch (error) {
				lastError = error;
			}
		}
		throw lastError;
	}
}

export class ConfigurableRunnable implements RunnableLike {
	constructor(
		readonly runnable: RunnableLike,
		readonly fields: Record<string, string>,
	) {}

	async invoke(
		input: unknown,
		options: Record<string, unknown> = {},
	): Promise<unknown> {
		const config = options.config as RunnableConfig | undefined;
		const configurable = config?.configurable ?? {};
		const configuredEntries = Object.entries(this.fields)
			.filter(([, configKey]) => configKey in configurable)
			.map(([field, configKey]) => [
				field,
				Object.entries(configurable).find(
					([key]) => key === configKey,
				)?.[1],
			]);
		const mapped = Object.fromEntries([
			...Object.entries(options),
			...configuredEntries,
		]);
		return await this.runnable.invoke(input, mapped);
	}
}

export async function runnableBatch(
	runnable: RunnableLike,
	inputs: unknown[],
	options: {
		config?: RunnableConfig | RunnableConfig[];
		maxConcurrency?: number;
	} = {},
): Promise<unknown[]> {
	const limit =
		options.maxConcurrency ??
		(Array.isArray(options.config)
			? undefined
			: options.config?.maxConcurrency) ??
		(inputs.length || 1);
	if (!Number.isInteger(limit) || limit < 1) {
		throw new Error('maxConcurrency must be at least 1');
	}
	const results = new Array<unknown>(inputs.length);
	let next = 0;
	async function worker(): Promise<void> {
		while (next < inputs.length) {
			const index = next;
			next += 1;
			const [config] = Array.isArray(options.config)
				? options.config.slice(index, index + 1)
				: [options.config];
			const [input] = inputs.slice(index, index + 1);
			const result = await runnable.invoke(input, { config });
			results.splice(index, 1, result);
		}
	}
	await Promise.all(
		Array.from(
			{ length: Math.min(limit, inputs.length) },
			async () => await worker(),
		),
	);
	return results;
}

export * from './langchain_retrieval.js';

import {
	AgentLoop,
	OpenAIModelProvider,
	ToolRegistry,
	loadModel,
	type AgentConfig,
	type AgentRunResult,
	type ModelMessage,
	type ModelProvider,
	type ModelRequest as AgentRTModelRequest,
	type ModelResponse,
	type StreamingModelProvider,
	type ToolCall,
	type ToolDefinition,
} from '../../index.js';
import {
	SdkModelProvider,
	isSdkItemList,
	messagesToSdkItems,
	messageSourceItems,
	restoreSourceItems,
	isSdkModel,
	providerGetResponse,
	providerGetStreamedResponse,
	sdkItemsToMessages,
	type SdkModel,
	type SdkModelRequest,
	type SdkModelResponse,
	type SdkStreamEvent,
} from './openai_agents_model.js';
import { sdkEnv } from './migration_tools.js';

export type {
	SdkModel as Model,
	SdkModelRequest as ModelRequest,
	SdkModelResponse as ModelResponseShape,
};

export class Usage {
	requests = 0;
	inputTokens = 0;
	outputTokens = 0;
	totalTokens = 0;

	constructor(
		input: Partial<
			Pick<
				Usage,
				'requests' | 'inputTokens' | 'outputTokens' | 'totalTokens'
			>
		> = {},
	) {
		this.requests = input.requests ?? 0;
		this.inputTokens = input.inputTokens ?? 0;
		this.outputTokens = input.outputTokens ?? 0;
		this.totalTokens =
			input.totalTokens ?? this.inputTokens + this.outputTokens;
	}

	add(other: Usage): void {
		this.requests += other.requests;
		this.inputTokens += other.inputTokens;
		this.outputTokens += other.outputTokens;
		this.totalTokens += other.totalTokens;
	}
}

/** Per-run state handed to tools, matching the OpenAI Agents SDK shape. */
export class RunContext<TContext = unknown> {
	context: TContext;
	usage: Usage;

	constructor(context: TContext = {} as TContext) {
		this.context = context;
		this.usage = new Usage();
	}

	toJSON(): { context: TContext; usage: Usage } {
		return { context: this.context, usage: this.usage };
	}
}

/** Raised when the model sends tool input that cannot be parsed. */
export class ModelBehaviorError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'ModelBehaviorError';
	}
}

/** Tracing is not exported by Agent RT; accepted for migration parity. */
export function setTracingDisabled(_disabled: boolean): void {}

/** Raised when a run reaches `maxTurns` without a final output, as in the SDK. */
export class MaxTurnsExceededError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'MaxTurnsExceededError';
	}
}

export type GuardrailFunctionOutput = {
	outputInfo?: unknown;
	tripwireTriggered: boolean;
};

/** An agent-level input guardrail, in the SDK's `{ name, execute }` shape. */
export interface InputGuardrail {
	name: string;
	runInParallel?: boolean;
	execute: (args: {
		input: RunInput;
		context: RunContext;
		agent: Agent;
	}) => GuardrailFunctionOutput | Promise<GuardrailFunctionOutput>;
}

export interface OutputGuardrail {
	name: string;
	execute: (args: {
		agentOutput: unknown;
		context: RunContext;
		agent: Agent;
	}) => GuardrailFunctionOutput | Promise<GuardrailFunctionOutput>;
}

export type GuardrailResult = {
	guardrail: { type: 'input' | 'output'; name: string };
	output: GuardrailFunctionOutput;
};

export class InputGuardrailTripwireTriggered extends Error {
	constructor(
		message: string,
		readonly result: GuardrailResult,
	) {
		super(message);
		this.name = 'InputGuardrailTripwireTriggered';
	}
}

export class OutputGuardrailTripwireTriggered extends Error {
	constructor(
		message: string,
		readonly result: GuardrailResult & { agentOutput: unknown },
	) {
		super(message);
		this.name = 'OutputGuardrailTripwireTriggered';
	}
}

export interface FunctionTool<
	Args = Record<string, unknown>,
	Result = unknown,
> {
	type: 'function';
	name: string;
	description: string;
	parameters: Record<string, unknown>;
	strict: boolean;
	execute: (
		args: Args,
		context?: unknown,
		details?: unknown,
	) => Result | Promise<Result>;
	invoke: (
		runContext: RunContext,
		input: string,
		details?: unknown,
	) => Promise<Result | string>;
	/** Pause the run for approval before this tool runs (see `RunState`). */
	needsApproval: NeedsApproval;
	inputGuardrails: ToolInputGuardrail[];
}

export type NeedsApproval =
	| boolean
	| ((
			context: RunContext,
			input: Record<string, unknown>,
			callId?: string,
	  ) => boolean | Promise<boolean>);

/** A tool input guardrail: `allow`, return `message` to the model, or throw. */
export interface ToolInputGuardrail {
	name?: string;
	run(args: {
		context: RunContext;
		agent: unknown;
		toolCall: Record<string, unknown>;
	}): Promise<ToolGuardrailResult> | ToolGuardrailResult;
}

export type ToolGuardrailResult = {
	behavior:
		| { type: 'allow' }
		| { type: 'rejectContent'; message: string }
		| { type: 'throwException' };
	outputInfo?: unknown;
};

/** Raised when a tool guardrail asks to stop the run. */
export class ToolGuardrailTripwireTriggered extends Error {
	constructor(readonly guardrail: string) {
		super(`Tool guardrail ${guardrail} triggered`);
		this.name = 'ToolGuardrailTripwireTriggered';
	}
}

type SchemaParser = { parse(value: unknown): unknown };

/**
 * The SDK's tool-name normalization: anything outside `[a-zA-Z0-9]` becomes
 * `_`, so a tool registered as `update-page` is called as `update_page`.
 */
export function toFunctionToolName(name: string): string {
	const normalized = name.replace(/\s/g, '_').replace(/[^a-zA-Z0-9]/g, '_');
	if (normalized.length === 0) throw new Error('Tool name cannot be empty');
	return normalized;
}

function schemaParser(value: unknown): SchemaParser | undefined {
	return value &&
		typeof value === 'object' &&
		typeof (value as { parse?: unknown }).parse === 'function'
		? (value as SchemaParser)
		: undefined;
}

function defaultToolErrorFunction(
	_context: RunContext,
	error: unknown,
): string {
	const detail = error instanceof Error ? error.toString() : String(error);
	return `An error occurred while running the tool. Please try again. Error: ${detail}`;
}

function schemaFrom(value: unknown): Record<string, unknown> {
	if (!value || typeof value !== 'object') {
		return { type: 'object', properties: {} };
	}
	const candidate = value as Record<string, unknown>;
	if (typeof candidate.toJSONSchema === 'function') {
		return (candidate.toJSONSchema as () => Record<string, unknown>)();
	}
	if (
		candidate.jsonSchema &&
		typeof candidate.jsonSchema === 'object' &&
		!Array.isArray(candidate.jsonSchema)
	) {
		return candidate.jsonSchema as Record<string, unknown>;
	}
	if (
		typeof candidate.type === 'string' ||
		candidate.properties !== undefined ||
		candidate.anyOf !== undefined
	) {
		return candidate;
	}
	throw new TypeError(
		'OpenAI Agents compatibility requires raw JSON Schema or a schema exposing toJSONSchema()',
	);
}

export function tool<
	Args = Record<string, unknown>,
	Result = unknown,
>(options: {
	name?: string;
	description: string;
	parameters: unknown;
	execute: (
		args: Args,
		context?: unknown,
		details?: unknown,
	) => Result | Promise<Result>;
	strict?: boolean;
	needsApproval?: NeedsApproval;
	inputGuardrails?: ToolInputGuardrail[];
	errorFunction?:
		| ((context: RunContext, error: unknown) => string | Promise<string>)
		| null;
}): FunctionTool<Args, Result> {
	const name = toFunctionToolName(options.name || options.execute.name || '');
	const parser = schemaParser(options.parameters);
	const errorFunction =
		options.errorFunction === undefined
			? defaultToolErrorFunction
			: options.errorFunction;
	const invoke = async (
		runContext: RunContext,
		input: string,
		details?: unknown,
	): Promise<Result | string> => {
		try {
			let args: unknown;
			try {
				args = input ? JSON.parse(input) : {};
				if (parser) args = parser.parse(args);
			} catch {
				throw new ModelBehaviorError('Invalid JSON input for tool');
			}
			return await options.execute(args as Args, runContext, details);
		} catch (error) {
			if (errorFunction) return await errorFunction(runContext, error);
			throw error;
		}
	};
	return {
		type: 'function',
		name,
		description: options.description,
		parameters: schemaFrom(options.parameters),
		strict: options.strict ?? true,
		execute: options.execute,
		invoke,
		needsApproval: options.needsApproval ?? false,
		inputGuardrails: [...(options.inputGuardrails ?? [])],
	};
}

/** Exposes an Agent RT provider through the SDK `Model` interface. */
function sdkModelMethods(
	provider: ModelProvider,
	model: () => string | undefined,
) {
	return {
		getResponse: (request: SdkModelRequest): Promise<SdkModelResponse> =>
			providerGetResponse(provider, model(), request),
		getStreamedResponse: (
			request: SdkModelRequest,
		): AsyncIterable<SdkStreamEvent> =>
			providerGetStreamedResponse(provider, model(), request),
	};
}

/**
 * OpenAI model provider for `new Runner({ modelProvider })`. An injected
 * `openAIClient` is used as-is, so tests can pass a fake OpenAI client.
 */
export class OpenAIProvider extends OpenAIModelProvider {
	constructor(
		options: {
			apiKey?: string;
			baseURL?: string;
			openAIClient?: unknown;
			useResponses?: boolean;
		} = {},
	) {
		// Like the SDK, an unset baseURL/apiKey falls back to the environment.
		const apiKey = options.apiKey ?? sdkEnv('OPENAI_API_KEY');
		const baseUrl = options.baseURL ?? sdkEnv('OPENAI_BASE_URL');
		super(
			{
				...(apiKey !== undefined ? { apiKey } : {}),
				...(baseUrl !== undefined ? { baseUrl } : {}),
			},
			options.openAIClient as ConstructorParameters<
				typeof OpenAIModelProvider
			>[1],
		);
	}

	getModel(modelName?: string): OpenAIProvider & SdkModel {
		const methods = sdkModelMethods(this, () => modelName);
		return Object.assign(Object.create(this) as this, methods);
	}
}

export { OpenAIChatCompletionsModel } from './openai_agents_chat.js';

export interface Handoff {
	agent: Agent;
	toolNameOverride?: string;
	toolDescriptionOverride?: string;
}

export function handoff(
	agent: Agent,
	config: {
		toolNameOverride?: string;
		toolDescriptionOverride?: string;
	} = {},
): Handoff {
	return { agent, ...config };
}

/** `toolUseBehavior`: what happens after the agent's tools run. */
export type ToolUseBehavior =
	| 'run_llm_again'
	| 'stop_on_first_tool'
	| { stopAtToolNames: string[] }
	| ((
			context: RunContext,
			toolResults: Array<{ tool: FunctionTool; output: unknown }>,
	  ) =>
			| { isFinalOutput: boolean; finalOutput?: unknown }
			| Promise<{ isFinalOutput: boolean; finalOutput?: unknown }>);

export interface AgentOptions {
	name: string;
	instructions?: string;
	/** A model name, an SDK `Model` (`getResponse()`), or an Agent RT provider. */
	model?: string | SdkModel | ModelProvider;
	tools?: FunctionTool[];
	handoffs?: Array<Agent | Handoff>;
	handoffDescription?: string;
	provider?: ModelProvider;
	outputType?: unknown;
	modelSettings?: Record<string, unknown>;
	toolUseBehavior?: ToolUseBehavior;
	inputGuardrails?: InputGuardrail[];
	outputGuardrails?: OutputGuardrail[];
	[key: string]: unknown;
}

export class Agent {
	readonly name: string;
	readonly instructions: string;
	readonly model: string | SdkModel | ModelProvider;
	readonly tools: FunctionTool[];
	readonly handoffs: Array<Agent | Handoff>;
	readonly handoffDescription?: string;
	readonly provider?: ModelProvider;
	readonly outputType?: unknown;
	readonly modelSettings: Record<string, unknown>;
	readonly toolUseBehavior: ToolUseBehavior;
	readonly inputGuardrails: InputGuardrail[];
	readonly outputGuardrails: OutputGuardrail[];
	readonly metadata: Record<string, unknown>;
	private readonly options: AgentOptions;

	constructor(options: AgentOptions) {
		if (!options.name?.trim())
			throw new Error('agent name must not be empty');
		const model = options.model;
		if (
			model !== undefined &&
			typeof model !== 'string' &&
			!isSdkModel(model) &&
			!isModelProvider(model)
		) {
			throw new TypeError(
				'Agent model must be a model name, an SDK Model (getResponse), or an Agent RT ModelProvider',
			);
		}
		this.options = { ...options };
		this.modelSettings = { ...(options.modelSettings ?? {}) };
		this.name = options.name;
		this.instructions = options.instructions ?? '';
		this.model = model ?? 'gpt-4o-mini';
		this.handoffs = [...(options.handoffs ?? [])];
		this.handoffDescription = options.handoffDescription;
		this.provider = options.provider;
		this.outputType = options.outputType;
		this.toolUseBehavior = options.toolUseBehavior ?? 'run_llm_again';
		this.tools = [...(options.tools ?? [])];
		this.inputGuardrails = [...(options.inputGuardrails ?? [])];
		this.outputGuardrails = [...(options.outputGuardrails ?? [])];
		const {
			name: _name,
			instructions: _instructions,
			model: _model,
			tools: _tools,
			handoffs: _handoffs,
			handoffDescription: _handoffDescription,
			provider: _provider,
			outputType: _outputType,
			inputGuardrails: _inputGuardrails,
			outputGuardrails: _outputGuardrails,
			...metadata
		} = options;
		this.metadata = metadata;
	}

	static create(options: AgentOptions): Agent {
		return new Agent(options);
	}

	/** A copy of this agent with `overrides` applied, as in the SDK. */
	clone(overrides: Partial<AgentOptions> = {}): Agent {
		return new Agent({ ...this.options, ...overrides } as AgentOptions);
	}

	asTool(
		options: {
			toolName?: string;
			toolDescription?: string;
		} = {},
	): FunctionTool<{ input: string }, unknown> {
		return tool({
			name: options.toolName ?? this.name,
			description:
				options.toolDescription ??
				this.handoffDescription ??
				`Delegate bounded work to ${this.name}.`,
			parameters: {
				type: 'object',
				properties: { input: { type: 'string' } },
				required: ['input'],
				additionalProperties: false,
			},
			execute: async ({ input }) => (await run(this, input)).finalOutput,
		});
	}
}

/** A run item, in the SDK's `newItems` shape. */
export type RunItem = {
	type: 'message_output_item' | 'tool_call_item' | 'tool_call_output_item';
	rawItem: Record<string, unknown>;
	agent: Agent;
	output?: unknown;
};

/** A tool call waiting for `state.approve()` / `state.reject()`. */
export type ToolApprovalItem = {
	type: 'tool_approval_item';
	rawItem: {
		type: 'function_call';
		callId: string;
		name: string;
		arguments: string;
		status: 'in_progress';
	};
	agent: Agent;
	toolName: string;
	name: string;
	arguments: string;
};

export interface RunResult {
	finalOutput: unknown;
	lastAgent: Agent;
	/** The conversation as SDK input items, ready to pass to the next run. */
	history: Array<Record<string, unknown>>;
	newItems: RunItem[];
	interruptions: ToolApprovalItem[];
	state: RunState;
	usage: Usage;
	rawResult: AgentRunResult;
}

type ApprovalDecision = { approved: boolean; message?: string };

/**
 * Serializable run state for approval round-trips: `result.state.toString()`
 * after an interruption, then `RunState.fromString(agent, text)`,
 * `approve()` / `reject()`, and `runner.run(agent, state)` to continue.
 */
export class RunState {
	readonly agent: Agent;
	/** @internal Conversation up to and including the paused tool calls. */
	readonly messages: ModelMessage[];
	private readonly interruptions: ToolApprovalItem[];
	/** @internal */
	readonly decisions = new Map<string, ApprovalDecision>();

	/** @internal */
	constructor(
		agent: Agent,
		messages: ModelMessage[],
		interruptions: ToolApprovalItem[],
	) {
		this.agent = agent;
		this.messages = messages;
		this.interruptions = interruptions;
	}

	getInterruptions(): ToolApprovalItem[] {
		return [...this.interruptions];
	}

	approve(
		item: ToolApprovalItem,
		_options: { alwaysApprove?: boolean } = {},
	): void {
		this.decisions.set(item.rawItem.callId, { approved: true });
	}

	reject(
		item: ToolApprovalItem,
		options: { message?: string; alwaysReject?: boolean } = {},
	): void {
		this.decisions.set(item.rawItem.callId, {
			approved: false,
			...(options.message !== undefined
				? { message: options.message }
				: {}),
		});
	}

	toString(): string {
		return JSON.stringify({
			$schemaVersion: 'agent-rt-1',
			agent: this.agent.name,
			messages: this.messages,
			// Original SDK items (reasoning, providerData) replay after resume.
			sourceItems: this.messages.map(
				(message) => messageSourceItems(message) ?? null,
			),
			interruptions: this.interruptions.map((item) => item.rawItem),
			decisions: Object.fromEntries(this.decisions),
		});
	}

	toJSON(): unknown {
		return JSON.parse(this.toString());
	}

	static async fromString(agent: Agent, text: string): Promise<RunState> {
		let data: unknown;
		try {
			data = JSON.parse(text);
		} catch {
			throw new TypeError(
				'RunState.fromString expects text from RunState.toString()',
			);
		}
		if (
			!data ||
			typeof data !== 'object' ||
			(data as { $schemaVersion?: unknown }).$schemaVersion !==
				'agent-rt-1'
		) {
			throw new TypeError(
				'RunState.fromString expects text from RunState.toString()',
			);
		}
		const value = data as {
			messages: ModelMessage[];
			interruptions: ToolApprovalItem['rawItem'][];
			decisions?: Record<string, ApprovalDecision>;
			sourceItems?: Array<Record<string, unknown>[] | null>;
		};
		value.messages.forEach((message, index) => {
			const items = value.sourceItems?.[index];
			if (items) restoreSourceItems(message, items);
		});
		const state = new RunState(
			agent,
			value.messages,
			value.interruptions.map((rawItem) => approvalItem(rawItem, agent)),
		);
		for (const [callId, decision] of Object.entries(value.decisions ?? {}))
			state.decisions.set(callId, decision);
		return state;
	}
}

function approvalItem(
	rawItem: ToolApprovalItem['rawItem'],
	agent: Agent,
): ToolApprovalItem {
	return {
		type: 'tool_approval_item',
		rawItem,
		agent,
		toolName: rawItem.name,
		name: rawItem.name,
		arguments: rawItem.arguments,
	};
}

export type RunStreamEvent =
	| { type: 'raw_model_stream_event'; data: SdkStreamEvent }
	| {
			type: 'run_item_stream_event';
			name: 'message_output_created' | 'tool_called' | 'tool_output';
			item: RunItem;
	  }
	| { type: 'agent_updated_stream_event'; agent: Agent };

export type RunInput =
	string | ModelMessage[] | Array<Record<string, unknown>> | RunState;

export interface RunOptions {
	maxTurns?: number;
	context?: unknown;
	stream?: boolean;
	signal?: AbortSignal;
	[key: string]: unknown;
}

function inputMessages(input: RunInput): ModelMessage[] {
	if (input instanceof RunState) return [...input.messages];
	if (typeof input === 'string')
		return [{ role: 'user', content: [{ type: 'text', text: input }] }];
	if (isSdkItemList(input)) return sdkItemsToMessages(input);
	return [...(input as ModelMessage[])];
}

function responseText(response?: ModelResponse): string {
	return (
		response?.message.content
			.filter((part) => part.type === 'text')
			.map((part) => part.text ?? '')
			.join('') ?? ''
	);
}

function outputSchema(
	outputType: unknown,
): Record<string, unknown> | undefined {
	if (outputType === undefined || outputType === 'text') return undefined;
	if (
		outputType &&
		typeof outputType === 'object' &&
		(outputType as { type?: unknown }).type === 'json_schema' &&
		typeof (outputType as { schema?: unknown }).schema === 'object'
	) {
		return (outputType as { schema: Record<string, unknown> }).schema;
	}
	return schemaFrom(outputType);
}

function handoffTool(
	value: Agent | Handoff,
	state: { lastAgent: Agent },
): FunctionTool<{ input?: string }, unknown> {
	const item: Handoff = value instanceof Agent ? { agent: value } : value;
	const definition = tool<{ input?: string }, unknown>({
		name: `transfer_to_${toFunctionToolName(item.agent.name)}`,
		description:
			item.toolDescriptionOverride ??
			item.agent.handoffDescription ??
			`Transfer work to ${item.agent.name}.`,
		parameters: {
			type: 'object',
			properties: { input: { type: 'string' } },
			additionalProperties: false,
		},
		execute: async ({ input }) => {
			const nested = await run(
				item.agent,
				input ?? 'Continue the current task.',
			);
			state.lastAgent = nested.lastAgent;
			return nested.finalOutput;
		},
	});
	// Like the SDK, an explicit override is used verbatim.
	if (item.toolNameOverride) definition.name = item.toolNameOverride;
	return definition;
}

type RunnerListener = (...args: unknown[]) => void;

export interface RunnerConfig {
	/** An Agent RT `ModelProvider`, or an SDK-style provider with `getModel()`. */
	modelProvider?: unknown;
	model?: string;
	[key: string]: unknown;
}

function isModelProvider(value: unknown): value is ModelProvider {
	return (
		!!value &&
		typeof value === 'object' &&
		typeof (value as { complete?: unknown }).complete === 'function'
	);
}

function hasGetModel(
	value: unknown,
): value is { getModel(name?: string): unknown } {
	return (
		!!value &&
		typeof value === 'object' &&
		typeof (value as { getModel?: unknown }).getModel === 'function'
	);
}

export class Runner {
	readonly config: RunnerConfig;
	private readonly listeners = new Map<string, Set<RunnerListener>>();

	constructor(config: RunnerConfig = {}) {
		if (
			config.modelProvider !== undefined &&
			!isModelProvider(config.modelProvider) &&
			!hasGetModel(config.modelProvider)
		) {
			throw new TypeError(
				'Runner modelProvider must be an Agent RT ModelProvider or expose getModel()',
			);
		}
		this.config = config;
	}

	on(event: string, listener: RunnerListener): this {
		let listeners = this.listeners.get(event);
		if (!listeners) {
			listeners = new Set();
			this.listeners.set(event, listeners);
		}
		listeners.add(listener);
		return this;
	}

	off(event: string, listener: RunnerListener): this {
		this.listeners.get(event)?.delete(listener);
		return this;
	}

	emit(event: string, ...args: unknown[]): boolean {
		const listeners = this.listeners.get(event);
		for (const listener of listeners ?? []) listener(...args);
		return (listeners?.size ?? 0) > 0;
	}

	run(
		startingAgent: Agent,
		input: RunInput,
		options: RunOptions & { stream: true },
	): Promise<StreamedRunResult>;
	run(
		startingAgent: Agent,
		input: RunInput,
		options?: RunOptions,
	): Promise<RunResult>;
	async run(
		startingAgent: Agent,
		input: RunInput,
		options: RunOptions = {},
	): Promise<RunResult | StreamedRunResult> {
		return startRun(this, startingAgent, input, options);
	}

	static run(
		startingAgent: Agent,
		input: RunInput,
		options: RunOptions & { stream: true },
	): Promise<StreamedRunResult>;
	static run(
		startingAgent: Agent,
		input: RunInput,
		options?: RunOptions,
	): Promise<RunResult>;
	static async run(
		startingAgent: Agent,
		input: RunInput,
		options: RunOptions = {},
	): Promise<RunResult | StreamedRunResult> {
		return startRun(undefined, startingAgent, input, options);
	}
}

/**
 * `run(agent, input, { stream: true })`: iterate it for SDK-shaped stream
 * events; `completed` resolves when the run ends (check `error`), after which
 * `finalOutput`, `newItems`, and `history` are set.
 */
export class StreamedRunResult implements AsyncIterable<RunStreamEvent> {
	finalOutput: unknown;
	lastAgent: Agent;
	currentAgent: Agent;
	history: Array<Record<string, unknown>> = [];
	newItems: RunItem[] = [];
	interruptions: ToolApprovalItem[] = [];
	state?: RunState;
	usage = new Usage();
	cancelled = false;
	rawResult?: AgentRunResult;
	error: unknown;
	readonly completed: Promise<void>;
	private readonly events: RunStreamEvent[] = [];
	private wake?: () => void;
	private done = false;
	private resolveCompleted!: () => void;

	constructor(agent: Agent) {
		this.lastAgent = agent;
		this.currentAgent = agent;
		this.completed = new Promise((resolve) => {
			this.resolveCompleted = resolve;
		});
	}

	/** @internal */
	push(event: RunStreamEvent): void {
		this.events.push(event);
		this.wake?.();
	}

	/** @internal */
	finish(result?: RunResult, error?: unknown): void {
		if (result) {
			this.finalOutput = result.finalOutput;
			this.lastAgent = result.lastAgent;
			this.currentAgent = result.lastAgent;
			this.history = result.history;
			this.newItems = result.newItems;
			this.interruptions = result.interruptions;
			this.state = result.state;
			this.usage = result.usage;
			this.rawResult = result.rawResult;
		}
		this.error = error;
		this.done = true;
		this.wake?.();
		this.resolveCompleted();
	}

	async *[Symbol.asyncIterator](): AsyncIterator<RunStreamEvent> {
		for (;;) {
			const event = this.events.shift();
			if (event) {
				yield event;
				continue;
			}
			if (this.done) {
				if (this.error !== undefined) throw this.error;
				return;
			}
			await new Promise<void>((resolve) => {
				this.wake = resolve;
			});
			this.wake = undefined;
		}
	}

	/** Assistant text deltas only. */
	async *toTextStream(): AsyncIterable<string> {
		for await (const event of this) {
			if (
				event.type === 'raw_model_stream_event' &&
				event.data.type === 'output_text_delta' &&
				typeof event.data.delta === 'string'
			)
				yield event.data.delta;
		}
	}
}

async function startRun(
	runner: Runner | undefined,
	startingAgent: Agent,
	input: RunInput,
	options: RunOptions,
): Promise<RunResult | StreamedRunResult> {
	if (!(startingAgent instanceof Agent)) {
		throw new TypeError('startingAgent must be an Agent');
	}
	if (!options.stream) return runAgent(runner, startingAgent, input, options);
	const streamed = new StreamedRunResult(startingAgent);
	streamed.push({ type: 'agent_updated_stream_event', agent: startingAgent });
	runAgent(runner, startingAgent, input, options, (event) =>
		streamed.push(event),
	).then(
		(result) => streamed.finish(result),
		(error: unknown) => {
			if (options.signal?.aborted) {
				streamed.cancelled = true;
				streamed.finish();
			} else {
				streamed.finish(undefined, error);
			}
		},
	);
	return streamed;
}

/** Wraps a provider to record each response's tool calls (for call ids). */
function recordingProvider(
	provider: ModelProvider,
	record: (response: ModelResponse) => void,
	prepare: (request: AgentRTModelRequest) => AgentRTModelRequest = (
		request,
	) => request,
): ModelProvider {
	const wrapped = Object.create(provider) as ModelProvider &
		Partial<StreamingModelProvider>;
	wrapped.complete = async (request) => {
		const response = await provider.complete(prepare(request));
		record(response);
		return response;
	};
	const streaming = provider as Partial<StreamingModelProvider>;
	if (typeof streaming.stream === 'function') {
		const stream = streaming.stream.bind(provider);
		wrapped.stream = async function* (request) {
			for await (const event of stream(prepare(request))) {
				if (event.type === 'completed' && event.response)
					record(event.response);
				yield event;
			}
		};
	}
	return wrapped;
}

async function resolveProvider(
	runner: Runner | undefined,
	agent: Agent,
	adapterOptions: ConstructorParameters<typeof SdkModelProvider>[1],
): Promise<ModelProvider> {
	const sdkProvider = (model: unknown): ModelProvider | undefined => {
		if (isModelProvider(model)) return model;
		if (isSdkModel(model))
			return new SdkModelProvider(model, adapterOptions);
		return undefined;
	};
	const fromAgent =
		typeof agent.model === 'string' ? undefined : sdkProvider(agent.model);
	if (fromAgent) return fromAgent;
	if (agent.provider) return agent.provider;
	const configured = runner?.config.modelProvider;
	if (isModelProvider(configured)) return configured;
	if (hasGetModel(configured)) {
		const name =
			typeof agent.model === 'string'
				? agent.model
				: runner?.config.model;
		const model = await configured.getModel(name);
		const provider = sdkProvider(model);
		if (!provider)
			throw new TypeError(
				'modelProvider.getModel() returned an unsupported model',
			);
		return provider;
	}
	return await loadModel();
}

async function runAgent(
	runner: Runner | undefined,
	startingAgent: Agent,
	input: RunInput,
	options: RunOptions,
	emit?: (event: RunStreamEvent) => void,
): Promise<RunResult> {
	const runContext = new RunContext(options.context ?? {});
	// Input guardrails finish before the first model call (not on a resumed
	// RunState), so a tripwire stops the run before any model or tool effect.
	if (!(input instanceof RunState)) {
		for (const guardrail of startingAgent.inputGuardrails) {
			const output = await guardrail.execute({
				input,
				context: runContext,
				agent: startingAgent,
			});
			if (output.tripwireTriggered)
				throw new InputGuardrailTripwireTriggered(
					`Input guardrail triggered: ${guardrail.name}`,
					{
						guardrail: { type: 'input', name: guardrail.name },
						output,
					},
				);
		}
	}
	let rawEventsFromModel = false;
	const provider = await resolveProvider(runner, startingAgent, {
		outputType: startingAgent.outputType,
		modelSettings: startingAgent.modelSettings,
		toolStrictness: new Map(
			startingAgent.tools.map((item) => [item.name, item.strict]),
		),
		...(emit
			? {
					onRawEvent: (event: SdkStreamEvent) => {
						rawEventsFromModel = true;
						emit({ type: 'raw_model_stream_event', data: event });
					},
				}
			: {}),
	});
	// Tool handlers get only arguments; the call ids come from the responses.
	const pendingCalls: ToolCall[] = [];
	// AgentLoop checks for a stop once before each tool call; an approval
	// pause waits for the rest of the turn, as the SDK collects every
	// interruption from a turn's tool calls before pausing.
	let callChecksLeft = 0;
	// As in the SDK, `outputType` is sent with every model request and the
	// final text is parsed as JSON (and by a zod-like `outputType`), rather
	// than run through Agent RT's structured-output validation and repair.
	const schema = outputSchema(startingAgent.outputType);
	const outputType = startingAgent.outputType as {
		name?: unknown;
		strict?: unknown;
	};
	const structuredOutput = schema
		? {
				name:
					typeof outputType?.name === 'string'
						? outputType.name
						: 'final_output',
				schema,
				strict: outputType?.strict !== false,
			}
		: undefined;
	const tracked = recordingProvider(
		provider,
		(response) => {
			pendingCalls.push(...(response.message.toolCalls ?? []));
			callChecksLeft = response.message.toolCalls?.length ?? 0;
		},
		(request) =>
			structuredOutput ? { ...request, structuredOutput } : request,
	);
	const takeCallId = (name: string, args: unknown): string => {
		const serialized = JSON.stringify(args ?? {});
		let index = pendingCalls.findIndex(
			(call) =>
				call.name === name &&
				JSON.stringify(call.arguments) === serialized,
		);
		if (index < 0)
			index = pendingCalls.findIndex((call) => call.name === name);
		if (index < 0) return `call_${name}`;
		return pendingCalls.splice(index, 1)[0]?.id ?? `call_${name}`;
	};

	const registry = new ToolRegistry();
	const state = { lastAgent: startingAgent };
	const toolOutputs = new Map<string, unknown>();
	const behavior = startingAgent.toolUseBehavior;
	const resumed = input instanceof RunState ? input : undefined;
	let stop: { finalOutput: unknown } | undefined;
	let fatal: { error: unknown } | undefined;
	const interruptions: ToolApprovalItem[] = [];
	const tools = [
		...startingAgent.tools,
		...startingAgent.handoffs.map((item) => handoffTool(item, state)),
	];
	const toolsByName = new Map(tools.map((item) => [item.name, item]));

	const needsApproval = async (
		item: FunctionTool,
		args: Record<string, unknown>,
		callId: string,
	): Promise<boolean> => {
		const rule = item.needsApproval;
		if (typeof rule === 'function')
			return await rule(runContext, args, callId);
		return rule === true;
	};

	/** Run one tool call the way the SDK does: guardrails, invoke, events. */
	const executeTool = async (
		item: FunctionTool,
		callId: string,
		args: Record<string, unknown>,
	): Promise<unknown> => {
		const argumentsJson = JSON.stringify(args ?? {});
		const toolCall = {
			type: 'function_call',
			callId,
			name: item.name,
			arguments: argumentsJson,
		};
		const details = { toolCall };
		runner?.emit(
			'agent_tool_start',
			runContext,
			startingAgent,
			item,
			details,
		);
		emit?.({
			type: 'run_item_stream_event',
			name: 'tool_called',
			item: {
				type: 'tool_call_item',
				rawItem: { ...toolCall, status: 'completed' },
				agent: startingAgent,
			},
		});
		const result = await item.invoke(runContext, argumentsJson, details);
		toolOutputs.set(callId, result);
		runner?.emit(
			'agent_tool_end',
			runContext,
			startingAgent,
			item,
			result,
			details,
		);
		emit?.({
			type: 'run_item_stream_event',
			name: 'tool_output',
			item: {
				type: 'tool_call_output_item',
				rawItem: {
					type: 'function_call_result',
					callId,
					name: item.name,
					status: 'completed',
					output: { type: 'text', text: toolOutputText(result) },
				},
				output: result,
				agent: startingAgent,
			},
		});
		if (!stop) {
			if (behavior === 'stop_on_first_tool') {
				stop = { finalOutput: result };
			} else if (
				typeof behavior === 'object' &&
				behavior.stopAtToolNames.includes(item.name)
			) {
				stop = { finalOutput: result };
			} else if (typeof behavior === 'function') {
				const decision = await behavior(runContext, [
					{ tool: item, output: result },
				]);
				if (decision.isFinalOutput)
					stop = { finalOutput: decision.finalOutput };
			}
		}
		return result;
	};

	/** Tool input guardrails; a returned string replaces the tool's output. */
	const checkGuardrails = async (
		item: FunctionTool,
		callId: string,
		args: Record<string, unknown>,
	): Promise<string | undefined> => {
		const toolCall = {
			type: 'function_call',
			callId,
			name: item.name,
			arguments: JSON.stringify(args ?? {}),
		};
		for (const guardrail of item.inputGuardrails) {
			const verdict = await guardrail.run({
				context: runContext,
				agent: startingAgent,
				toolCall,
			});
			if (verdict.behavior.type === 'rejectContent')
				return verdict.behavior.message;
			if (verdict.behavior.type === 'throwException')
				throw new ToolGuardrailTripwireTriggered(
					guardrail.name ?? item.name,
				);
		}
		return undefined;
	};

	// Placeholder result for a call paused for approval; dropped from the
	// saved state, which resumes from the call itself.
	const pausedCallIds = new Set<string>();
	const approvedCallIds = new Set<string>();
	for (const item of tools) {
		const definition: ToolDefinition = {
			name: item.name,
			description: item.description,
			inputSchema: item.parameters,
		};
		registry.register(definition, {
			handler: async (arguments_) => {
				const callId = takeCallId(item.name, arguments_);
				const args = arguments_ ?? {};
				try {
					// Guardrails run before the approval check, and again when an
					// approved call is resumed, as in the SDK.
					const rejected = await checkGuardrails(item, callId, args);
					if (rejected !== undefined) return rejected;
					if (
						!approvedCallIds.has(callId) &&
						(await needsApproval(item, args, callId))
					) {
						interruptions.push(
							approvalItem(
								{
									type: 'function_call',
									callId,
									name: item.name,
									arguments: JSON.stringify(args),
									status: 'in_progress',
								},
								startingAgent,
							),
						);
						pausedCallIds.add(callId);
						return 'Waiting for approval.';
					}
					return await executeTool(item, callId, args);
				} catch (error) {
					// An error the tool's errorFunction rethrows ends the run, as
					// in the SDK; the loop only sees a failed call.
					fatal ??= { error };
					throw error;
				}
			},
		});
	}

	const messages = inputMessages(input);
	if (resumed) {
		// Settle the calls the previous run paused on, then continue.
		for (const pending of resumed.getInterruptions()) {
			const { callId, name } = pending.rawItem;
			const decision = resumed.decisions.get(callId);
			if (!decision) {
				interruptions.push(pending);
				continue;
			}
			let output: unknown;
			if (decision.approved) {
				if (!toolsByName.has(name))
					throw new Error(`tool not found: ${name}`);
				const call: ToolCall = {
					id: callId,
					name,
					arguments: JSON.parse(pending.rawItem.arguments || '{}'),
				};
				pendingCalls.push(call);
				approvedCallIds.add(callId);
				// Through the registry, so Agent RT permission checks and hooks
				// still apply to the approved call.
				output = await registry.execute(call);
			} else {
				output = decision.message ?? 'Tool execution was not approved.';
			}
			messages.push({
				role: 'tool',
				toolCallId: callId,
				content: [{ type: 'text', text: toolOutputText(output) }],
			});
		}
	}
	const modelName =
		typeof startingAgent.model === 'string'
			? startingAgent.model
			: (runner?.config.model ?? 'gpt-4o-mini');
	const config: AgentConfig = {
		name: startingAgent.name,
		instructions: startingAgent.instructions,
		description: startingAgent.handoffDescription,
		model: { model: modelName },
	};
	const inputCount = resumed ? 0 : messages.length;
	let result: AgentRunResult;
	if (interruptions.length) {
		result = {
			messages: [...messages],
			terminationReason: 'stop_requested',
			turns: 0,
			toolCalls: 0,
			totalTokens: 0,
		};
	} else {
		result = await new AgentLoop(
			tracked,
			undefined,
			tools.length > 0 ? registry : undefined,
		).run(
			config,
			messages,
			{ maxTurns: options.maxTurns ?? 16 },
			() => {
				if (stop !== undefined || fatal !== undefined) return true;
				if (callChecksLeft > 0) {
					callChecksLeft -= 1;
					return false;
				}
				return interruptions.length > 0;
			},
			emit
				? (event) => {
						// SDK models report their own raw events; synthesize the
						// text deltas for Agent RT providers.
						if (
							!rawEventsFromModel &&
							event.type === 'text_delta' &&
							event.text
						) {
							emit({
								type: 'raw_model_stream_event',
								data: {
									type: 'output_text_delta',
									delta: event.text,
								},
							});
						}
					}
				: undefined,
			options.signal,
		);
	}
	if (fatal) throw fatal.error;
	if (options.signal?.aborted)
		throw options.signal.reason ?? new Error('aborted');
	if (result.terminationReason === 'max_turns' && !interruptions.length)
		throw new MaxTurnsExceededError(
			`Max turns (${options.maxTurns ?? 16}) exceeded`,
		);
	const conversation = result.messages.filter(
		(message) =>
			!(
				message.role === 'tool' &&
				pausedCallIds.has(message.toolCallId ?? '')
			),
	);
	const newItems = runItems(
		conversation.slice(inputCount),
		startingAgent,
		toolOutputs,
	);
	let output: unknown;
	if (interruptions.length) {
		output = undefined;
	} else if (stop) {
		output = stop.finalOutput;
	} else if (schema) {
		const text = responseText(result.finalResponse);
		try {
			output = JSON.parse(text);
		} catch {
			throw new ModelBehaviorError(
				`Invalid output type: final output is not valid JSON: ${text.slice(0, 200)}`,
			);
		}
		const parser = schemaParser(startingAgent.outputType);
		if (parser) output = parser.parse(output);
	} else {
		output = responseText(result.finalResponse);
	}
	if (!interruptions.length) {
		for (const guardrail of startingAgent.outputGuardrails) {
			const guardOutput = await guardrail.execute({
				agentOutput: output,
				context: runContext,
				agent: startingAgent,
			});
			if (guardOutput.tripwireTriggered)
				throw new OutputGuardrailTripwireTriggered(
					`Output guardrail triggered: ${guardrail.name}`,
					{
						guardrail: { type: 'output', name: guardrail.name },
						output: guardOutput,
						agentOutput: output,
					},
				);
		}
	}
	if (emit) {
		const messageItems = newItems.filter(
			(entry) => entry.type === 'message_output_item',
		);
		const last = messageItems[messageItems.length - 1];
		if (last)
			emit({
				type: 'run_item_stream_event',
				name: 'message_output_created',
				item: last,
			});
	}
	return {
		finalOutput: output,
		lastAgent: state.lastAgent,
		history: messagesToSdkItems(
			conversation.filter((message) => message.role !== 'system'),
		).input,
		newItems,
		interruptions,
		state: new RunState(startingAgent, conversation, interruptions),
		usage: new Usage({
			requests: result.turns,
			totalTokens: result.totalTokens,
		}),
		rawResult: result,
	};
}

function toolOutputText(value: unknown): string {
	return typeof value === 'string' ? value : JSON.stringify(value);
}

function runItems(
	messages: ModelMessage[],
	agent: Agent,
	toolOutputs: ReadonlyMap<string, unknown>,
): RunItem[] {
	const items: RunItem[] = [];
	const names = new Map<string, string>();
	for (const message of messages) {
		const text = message.content
			.filter((part) => part.type === 'text')
			.map((part) => part.text ?? '')
			.join('');
		if (message.role === 'assistant') {
			if (text)
				items.push({
					type: 'message_output_item',
					rawItem: {
						type: 'message',
						role: 'assistant',
						status: 'completed',
						content: [{ type: 'output_text', text }],
					},
					agent,
				});
			for (const call of message.toolCalls ?? []) {
				names.set(call.id, call.name);
				items.push({
					type: 'tool_call_item',
					rawItem: {
						type: 'function_call',
						callId: call.id,
						name: call.name,
						arguments: JSON.stringify(call.arguments),
						status: 'completed',
					},
					agent,
				});
			}
		} else if (message.role === 'tool') {
			const callId = message.toolCallId ?? '';
			items.push({
				type: 'tool_call_output_item',
				rawItem: {
					type: 'function_call_result',
					callId,
					name: names.get(callId) ?? '',
					status: 'completed',
					output: { type: 'text', text },
				},
				output: toolOutputs.has(callId)
					? toolOutputs.get(callId)
					: text,
				agent,
			});
		}
	}
	return items;
}

export function run(
	agent: Agent,
	input: RunInput,
	options: RunOptions & { stream: true },
): Promise<StreamedRunResult>;
export function run(
	agent: Agent,
	input: RunInput,
	options?: RunOptions,
): Promise<RunResult>;
export async function run(
	agent: Agent,
	input: RunInput,
	options: RunOptions = {},
): Promise<RunResult | StreamedRunResult> {
	return startRun(undefined, agent, input, options);
}

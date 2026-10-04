import {
	AgentLoop,
	ToolRegistry,
	loadModel,
	type AgentConfig,
	type AgentRunResult,
	type ModelMessage,
	type ModelProvider,
	type ToolDefinition,
} from '../../index.js';

export interface FunctionTool<
	Args = Record<string, unknown>,
	Result = unknown,
> {
	type: 'function';
	name: string;
	description: string;
	parameters: Record<string, unknown>;
	execute: (
		args: Args,
		context?: unknown,
		details?: unknown,
	) => Result | Promise<Result>;
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
}): FunctionTool<Args, Result> {
	const name = options.name ?? options.execute.name ?? 'tool';
	return {
		type: 'function',
		name,
		description: options.description,
		parameters: schemaFrom(options.parameters),
		execute: options.execute,
	};
}

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

export interface AgentOptions {
	name: string;
	instructions?: string;
	model?: string;
	tools?: FunctionTool[];
	handoffs?: Array<Agent | Handoff>;
	handoffDescription?: string;
	provider?: ModelProvider;
	outputType?: unknown;
	[key: string]: unknown;
}

export class Agent {
	readonly name: string;
	readonly instructions: string;
	readonly model: string;
	readonly tools: FunctionTool[];
	readonly handoffs: Array<Agent | Handoff>;
	readonly handoffDescription?: string;
	readonly provider?: ModelProvider;
	readonly outputType?: unknown;
	readonly metadata: Record<string, unknown>;

	constructor(options: AgentOptions) {
		if (!options.name?.trim())
			throw new Error('agent name must not be empty');
		this.name = options.name;
		this.instructions = options.instructions ?? '';
		this.model = options.model ?? 'gpt-4o-mini';
		this.handoffs = [...(options.handoffs ?? [])];
		this.handoffDescription = options.handoffDescription;
		this.provider = options.provider;
		this.outputType = options.outputType;
		this.tools = [...(options.tools ?? [])];
		const {
			name: _name,
			instructions: _instructions,
			model: _model,
			tools: _tools,
			handoffs: _handoffs,
			handoffDescription: _handoffDescription,
			provider: _provider,
			outputType: _outputType,
			...metadata
		} = options;
		this.metadata = metadata;
	}

	static create(options: AgentOptions): Agent {
		return new Agent(options);
	}

	asTool(
		options: {
			toolName?: string;
			toolDescription?: string;
		} = {},
	): FunctionTool<{ input: string }, unknown> {
		const target = this;
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
			execute: async ({ input }) =>
				(await run(target, input)).finalOutput,
		});
	}
}

export interface RunResult {
	finalOutput: unknown;
	lastAgent: Agent;
	history: ModelMessage[];
	rawResult: AgentRunResult;
}

function inputMessages(input: string | ModelMessage[]): ModelMessage[] {
	if (typeof input !== 'string') return [...input];
	return [{ role: 'user', content: [{ type: 'text', text: input }] }];
}

function finalOutput(result: AgentRunResult): unknown {
	if (result.structuredOutput !== undefined) return result.structuredOutput;
	return (
		result.finalResponse?.message.content
			.filter((part) => part.type === 'text')
			.map((part) => part.text ?? '')
			.join('') ?? ''
	);
}

function handoffTool(
	value: Agent | Handoff,
	state: { lastAgent: Agent },
): FunctionTool<{ input?: string }, unknown> {
	const item: Handoff = value instanceof Agent ? { agent: value } : value;
	const name =
		item.toolNameOverride ??
		`transfer_to_${item.agent.name.toLowerCase().replace(/\s+/g, '_')}`;
	return tool({
		name,
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
}

export class Runner {
	async run(
		startingAgent: Agent,
		input: string | ModelMessage[],
		options: {
			maxTurns?: number;
			context?: unknown;
		} = {},
	): Promise<RunResult> {
		return Runner.run(startingAgent, input, options);
	}

	static async run(
		startingAgent: Agent,
		input: string | ModelMessage[],
		options: {
			maxTurns?: number;
			context?: unknown;
		} = {},
	): Promise<RunResult> {
		if (!(startingAgent instanceof Agent)) {
			throw new TypeError('startingAgent must be an Agent');
		}
		const provider = startingAgent.provider ?? (await loadModel());
		const registry = new ToolRegistry();
		const state = { lastAgent: startingAgent };
		const tools = [
			...startingAgent.tools,
			...startingAgent.handoffs.map((item) => handoffTool(item, state)),
		];
		for (const item of tools) {
			const definition: ToolDefinition = {
				name: item.name,
				description: item.description,
				inputSchema: item.parameters,
			};
			registry.register(definition, {
				handler: async (arguments_) => item.execute(arguments_),
			});
		}
		const config: AgentConfig = {
			name: startingAgent.name,
			instructions: startingAgent.instructions,
			description: startingAgent.handoffDescription,
			model: { model: startingAgent.model },
		};
		const result = await new AgentLoop(
			provider,
			undefined,
			tools.length > 0 ? registry : undefined,
		).run(config, inputMessages(input), {
			maxTurns: options.maxTurns ?? 16,
		});
		return {
			finalOutput: finalOutput(result),
			lastAgent: state.lastAgent,
			history: [...result.messages],
			rawResult: result,
		};
	}
}

export async function run(
	agent: Agent,
	input: string | ModelMessage[],
	options: { maxTurns?: number; context?: unknown } = {},
): Promise<RunResult> {
	return Runner.run(agent, input, options);
}

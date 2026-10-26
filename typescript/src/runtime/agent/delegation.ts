/** Agent invocation, agents as tools, handoffs, subagents, and supervisor teams. */
import {
	type MessageRole,
	type ModelMessage,
	type WorkflowState,
} from '../types.js';
import {
	type ContextualToolHandler,
	type ToolDefinition,
} from '../tools/definitions.js';
import { type CapabilityGrant } from '../authorization/grants.js';
import { type ContextItem } from '../context/items.js';
import { type AgentConfig } from './config.js';
import { type ExecutionBudget } from '../resilience/budgets.js';
import { type AgentRunResult } from './limits.js';

export interface AgentInvocation {
	messages: ModelMessage[];
	contextItems?: ContextItem[];
	workflowState?: WorkflowState;
	allowedTools?: string[];
	capabilityGrant?: CapabilityGrant;
	executionBudget?: ExecutionBudget;
	metadata?: Record<string, unknown>;
}

export interface AgentInvoker {
	invoke(
		agent: AgentConfig,
		invocation: AgentInvocation,
	): Promise<AgentRunResult>;
}

export interface AgentTool {
	definition: ToolDefinition;
	handler: ContextualToolHandler;
}

export function agentAsTool(
	agent: AgentConfig,
	invoker: AgentInvoker,
	options: { name?: string; description?: string } = {},
): AgentTool {
	const toolName = options.name ?? 'agent.' + agent.name;
	const handler: ContextualToolHandler = async (args, context) => {
		const prompt = args.input;
		if (typeof prompt !== 'string' || !prompt.trim()) {
			throw new Error('agent tool requires a non-empty input');
		}
		const result = await invoker.invoke(agent, {
			messages: [messageFromText('user', prompt)],
			metadata: {
				parentTool: toolName,
				parentRequestContext: { ...context.requestContext },
			},
		});
		if (result.structuredOutput !== undefined)
			return result.structuredOutput;
		if (result.finalResponse) {
			const text = result.finalResponse.message.content
				.filter(
					(part) =>
						part.type === 'text' && typeof part.text === 'string',
				)
				.map((part) => part.text ?? '')
				.join('');
			if (text) return text;
		}
		return {
			terminationReason: result.terminationReason,
			turns: result.turns,
		};
	};
	return {
		definition: {
			name: toolName,
			description:
				options.description ??
				agent.description ??
				'Delegate bounded work to ' + agent.name + '.',
			inputSchema: {
				type: 'object',
				properties: { input: { type: 'string' } },
				required: ['input'],
				additionalProperties: false,
			},
			sideEffect: 'none',
		},
		handler,
	};
}

export function messageFromText(role: MessageRole, text: string): ModelMessage {
	return { role, content: [{ type: 'text', text }] };
}

export interface HandoffRequest {
	target: string;
	messages: ModelMessage[];
	reason: string;
	contextItems?: ContextItem[];
	workflowState?: WorkflowState;
	metadata?: Record<string, unknown>;
}

export interface HandoffResult {
	owner: string;
	result: AgentRunResult;
	reason: string;
}

export class HandoffManager {
	constructor(
		readonly specialists: Record<
			string,
			{ agent: AgentConfig; invoker: AgentInvoker }
		>,
	) {}

	async handoff(request: HandoffRequest): Promise<HandoffResult> {
		const specialist = this.specialists[request.target];
		if (!specialist)
			throw new Error('unknown handoff target: ' + request.target);
		const result = await specialist.invoker.invoke(specialist.agent, {
			messages: request.messages,
			contextItems: request.contextItems ?? [],
			workflowState: request.workflowState,
			metadata: {
				...(request.metadata ?? {}),
				handoffReason: request.reason,
				handoffTarget: request.target,
			},
		});
		return { owner: request.target, result, reason: request.reason };
	}
}

export interface SubagentSpec {
	agent: AgentConfig;
	messages: ModelMessage[];
	contextItems?: ContextItem[];
	workflowState?: WorkflowState;
	allowedTools?: string[];
	capabilityGrant?: CapabilityGrant;
	executionBudget?: ExecutionBudget;
	metadata?: Record<string, unknown>;
}

export class IsolatedSubagentRunner {
	constructor(
		readonly invoker: AgentInvoker,
		readonly parentBudget?: ExecutionBudget,
	) {}

	async run(spec: SubagentSpec): Promise<AgentRunResult> {
		this.parentBudget?.consumeSubagent();
		return this.invoker.invoke(spec.agent, {
			messages: spec.messages,
			contextItems: spec.contextItems ?? [],
			workflowState: spec.workflowState,
			allowedTools: [...(spec.allowedTools ?? [])],
			capabilityGrant: spec.capabilityGrant,
			executionBudget: spec.executionBudget,
			metadata: { ...(spec.metadata ?? {}), isolatedSubagent: true },
		});
	}
}

export interface WorkerAssignment {
	id: string;
	worker: string;
	messages: ModelMessage[];
	contextItems?: ContextItem[];
	metadata?: Record<string, unknown>;
}

export interface WorkerResult {
	assignmentId: string;
	worker: string;
	result: AgentRunResult;
}

export class SupervisorWorkerTeam {
	constructor(
		readonly workers: Record<
			string,
			{ agent: AgentConfig; runner: IsolatedSubagentRunner }
		>,
	) {}

	async delegate(assignments: WorkerAssignment[]): Promise<WorkerResult[]> {
		const results: WorkerResult[] = [];
		for (const assignment of assignments) {
			const worker = this.workers[assignment.worker];
			if (!worker)
				throw new Error('unknown worker: ' + assignment.worker);
			const result = await worker.runner.run({
				agent: worker.agent,
				messages: assignment.messages,
				contextItems: assignment.contextItems ?? [],
				metadata: {
					...(assignment.metadata ?? {}),
					assignmentId: assignment.id,
					supervised: true,
				},
			});
			results.push({
				assignmentId: assignment.id,
				worker: assignment.worker,
				result,
			});
		}
		return results;
	}
}

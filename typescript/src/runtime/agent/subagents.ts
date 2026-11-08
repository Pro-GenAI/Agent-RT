/** Governed delegation: inherited history, least-privilege tools and reviewer spans. */
import { randomUUID } from 'node:crypto';
import { type ModelMessage } from '../types.js';
import { type ToolDefinition, type ToolSelectionPolicy } from '../tools/definitions.js';
import { CapabilityGrant } from '../authorization/grants.js';
import { TraceRecorder } from '../observability/tracing.js';
import { RuntimeMetrics } from '../observability/metrics.js';
import { type ExecutionBudget } from '../resilience/budgets.js';
import { type AgentConfig } from './config.js';
import { type AgentRunResult } from './limits.js';
import {
  type AgentInvoker,
  type AgentTool,
  IsolatedSubagentRunner,
  type SubagentSpec,
} from './delegation.js';

export type ToolPermission = 'read' | 'write';

export interface ReviewerReport {
  response?: string;
  error?: string;
}

export interface SubagentOutcome {
  taskId: string;
  agent: string;
  depth: number;
  result?: AgentRunResult;
  response?: string;
  review: ReviewerReport;
  error?: string;
}

function message(text: string): ModelMessage {
  return { role: 'user', content: [{ type: 'text', text }] };
}

function response(result?: AgentRunResult): string | undefined {
  if (!result) return undefined;
  if (result.structuredOutput !== undefined)
    return JSON.stringify(result.structuredOutput);
  if (!result.finalResponse) return undefined;
  return result.finalResponse.message.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text ?? '')
    .join('');
}

/** The root may fan out; depth-2 workers get exactly one child slot. */
export class SubagentScope {
  private spawned = 0;

  constructor(
    readonly swarm: SubagentSwarm,
    readonly depth: number,
    readonly history: readonly ModelMessage[],
    readonly permissions: Readonly<Record<string, ToolPermission>>,
    readonly spanId: string,
    readonly parentAgent: string,
    readonly ownerAgent: string,
  ) {}

  async spawn(
    task: string,
    agent: AgentConfig,
    options: {
      allowedTools: Record<string, ToolPermission>;
      reviewer?: AgentConfig;
    },
  ): Promise<SubagentOutcome> {
    if (!task.trim()) throw new Error('subagent task must not be empty');
    if (this.depth >= 3) {
      this.swarm.metrics.increment('subagent.denied', 1, { reason: 'depth' });
      throw new Error('subagent depth limit (3) exceeded');
    }
    if (this.depth > 1 && this.spawned >= 1) {
      this.swarm.metrics.increment('subagent.denied', 1, { reason: 'fanout' });
      throw new Error('a subagent may launch only one child');
    }
    const permissions = { ...options.allowedTools };
    for (const [name, level] of Object.entries(permissions)) {
      const definition = this.swarm.catalog.get(name);
      if (!definition || (level !== 'read' && level !== 'write'))
        throw new Error('unknown tool or access level: ' + name);
      const inherited = this.permissions[name];
      if (!inherited || (inherited === 'read' && level === 'write'))
        throw new Error('cannot escalate tool permission: ' + name);
      if (level === 'read' && !['none', 'read'].includes(definition.sideEffect ?? 'none'))
        throw new Error('tool is not read-only: ' + name);
      if (
        agent.toolPolicy?.denied?.includes(name) ||
        (agent.toolPolicy?.allowed !== undefined &&
          !agent.toolPolicy.allowed.includes(name))
      )
        throw new Error('agent policy denies tool: ' + name);
    }
    const missingRequired = (agent.toolPolicy?.required ?? [])
      .filter((name) => !(name in permissions));
    if (missingRequired.length)
      throw new Error('required tools unavailable to subagent: ' + missingRequired.join(', '));
    // No await before reservation; concurrent sibling promises cannot reuse slots.
    this.swarm.budget?.consumeSubagent();
    this.spawned += 1;
    const taskId = randomUUID();
    const spanId = randomUUID();
    const scope = new SubagentScope(
      this.swarm, this.depth + 1, [...this.history, message(task)],
      permissions, spanId, this.ownerAgent, agent.name,
    );
    this.swarm.traces.start({
      spanId, traceId: this.swarm.traceId, parentSpanId: this.spanId,
      name: agent.name, kind: 'subagent', taskId,
      attributes: {
        depth: scope.depth, parentAgent: this.parentAgent,
        toolCount: Object.keys(permissions).length,
      },
    });
    this.swarm.started(agent.name);
    return this.swarm.execute(scope, taskId, agent, options.reviewer ?? this.swarm.reviewer);
  }

  /** Register the returned tool in the parent runtime and authorize its name. */
  asTool(
    agent: AgentConfig,
    options: {
      allowedTools: Record<string, ToolPermission>;
      reviewer?: AgentConfig;
      name?: string;
    },
  ): AgentTool {
    const permissions = { ...options.allowedTools };
    return {
      definition: {
        name: options.name ?? 'delegate_subagent',
        description: 'Delegate a task to a bounded, reviewed subagent.',
        inputSchema: {
          type: 'object',
          properties: { task: { type: 'string' } },
          required: ['task'],
          additionalProperties: false,
        },
        sideEffect: Object.values(permissions).includes('write') ? 'write' : 'read',
      },
      handler: async (args) => {
        if (typeof args.task !== 'string' || !args.task.trim())
          throw new Error('delegated task must be a non-empty string');
        const out = await this.spawn(
          args.task, agent,
          { allowedTools: permissions, reviewer: options.reviewer },
        );
        return {
          taskId: out.taskId,
          response: out.response,
          error: out.error,
          review: out.review.response,
          reviewError: out.review.error,
        };
      },
    };
  }

}

export class SubagentSwarm {
  readonly catalog: Map<string, ToolDefinition>;
  readonly traces: TraceRecorder;
  readonly metrics: RuntimeMetrics;
  readonly traceId = randomUUID();
  readonly root: SubagentScope;
  private activeJobs = 0;

  constructor(
    readonly invoker: AgentInvoker,
    options: {
      chatHistory: readonly ModelMessage[];
      toolDefinitions: readonly ToolDefinition[];
      parentPermissions: Record<string, ToolPermission>;
      reviewer: AgentConfig;
      budget?: ExecutionBudget;
      capabilityGrant?: CapabilityGrant;
      traces?: TraceRecorder;
      metrics?: RuntimeMetrics;
    },
  ) {
    this.reviewer = options.reviewer;
    this.budget = options.budget;
    this.capabilityGrant = options.capabilityGrant;
    this.catalog = new Map(options.toolDefinitions.map((tool) => [tool.name, tool]));
    if (this.catalog.size !== options.toolDefinitions.length)
      throw new Error('duplicate tool definitions');
    for (const [name, level] of Object.entries(options.parentPermissions)) {
      const definition = this.catalog.get(name);
      if (!definition || !['read', 'write'].includes(level))
        throw new Error('invalid parent tool permission: ' + name);
      if (level === 'read' && !['none', 'read'].includes(definition.sideEffect ?? 'none'))
        throw new Error('tool is not read-only: ' + name);
      if (options.capabilityGrant && !options.capabilityGrant.allowsTool(name))
        throw new Error('parent grant denies tool: ' + name);
    }
    this.traces = options.traces ?? new TraceRecorder();
    this.metrics = options.metrics ?? new RuntimeMetrics();
    const rootId = randomUUID();
    this.traces.start({
      spanId: rootId, traceId: this.traceId,
      name: 'main-agent', kind: 'agent',
    });
    this.root = new SubagentScope(
      this, 1, [...options.chatHistory], { ...options.parentPermissions },
      rootId, 'main-agent', 'main-agent',
    );
  }

  readonly reviewer: AgentConfig;
  readonly budget?: ExecutionBudget;
  readonly capabilityGrant?: CapabilityGrant;

  get active(): number {
    return this.activeJobs;
  }

  started(agent: string): void {
    this.activeJobs += 1;
    this.metrics.increment('subagent.started', 1, { agent });
    this.metrics.record('subagent.active', this.activeJobs, { kind: 'gauge' });
  }

  deploy(
    task: string,
    agent: AgentConfig,
    options: {
      allowedTools: Record<string, ToolPermission>;
      reviewer?: AgentConfig;
    },
  ): Promise<SubagentOutcome> {
    return this.root.spawn(task, agent, options);
  }

  async execute(
    scope: SubagentScope,
    taskId: string,
    agent: AgentConfig,
    reviewer: AgentConfig,
  ): Promise<SubagentOutcome> {
    const names = Object.keys(scope.permissions).sort();
    const prior = agent.toolPolicy;
    const toolPolicy: ToolSelectionPolicy = {
      allowed: names,
      denied: [...(prior?.denied ?? [])],
      required: [...(prior?.required ?? [])],
      preferred: [...(prior?.preferred ?? [])],
    };
    const restricted = { ...agent, toolPolicy };
    const inherited = this.capabilityGrant;
    const grant = new CapabilityGrant({
      tools: names,
      paths: inherited?.paths ?? [],
      networks: inherited?.networks ?? [],
      credentialIds: inherited?.credentialIds ?? [],
    });
    let result: AgentRunResult | undefined;
    let error: string | undefined;
    let review: ReviewerReport = {};
    try {
      try {
        const spec: SubagentSpec = {
          agent: restricted,
          messages: [...scope.history],
          allowedTools: names,
          capabilityGrant: grant,
          executionBudget: this.budget,
          metadata: {
            taskId, traceId: this.traceId, spanId: scope.spanId,
            depth: scope.depth, parentAgent: scope.parentAgent,
            toolPermissions: { ...scope.permissions },
          },
          subagents: scope,
        };
        result = await new IsolatedSubagentRunner(this.invoker).run(spec);
        if (result.terminationReason !== 'completed')
          error = 'worker terminated: ' + result.terminationReason;
      } catch (caught) {
        error = caught instanceof Error
          ? caught.name + ': ' + caught.message : String(caught);
      }

      const reviewId = randomUUID();
      this.traces.start({
        spanId: reviewId, traceId: this.traceId, parentSpanId: scope.spanId,
        name: reviewer.name, kind: 'agent', taskId,
        attributes: { role: 'reviewer', depth: scope.depth },
      });
      this.metrics.increment('subagent.review.started');
      try {
        const context = JSON.stringify({
          taskId, task: scope.history[scope.history.length - 1]?.content[0],
          worker: agent.name, termination: result?.terminationReason,
          response: response(result), error,
        });
        const outcome = await this.invoker.invoke(
          { ...reviewer, toolPolicy: { allowed: [] } },
          {
            messages: [
              ...scope.history,
              ...(result?.messages ?? []),
              message(
                'Review this delegated task for mistakes, failed actions, policy ' +
                'violations and incomplete work. Report findings explicitly; ' +
                'do not use tools or delegate.\n' + context,
              ),
            ],
            allowedTools: [],
            capabilityGrant: new CapabilityGrant(),
            metadata: {
              taskId, traceId: this.traceId, spanId: reviewId,
              reviewedSpanId: scope.spanId,
            },
          },
        );
        review = {
          response: response(outcome),
          error: outcome.terminationReason === 'completed'
            ? undefined : 'reviewer terminated: ' + outcome.terminationReason,
        };
      } catch (caught) {
        review = { error: caught instanceof Error
          ? caught.name + ': ' + caught.message : String(caught) };
      } finally {
        this.traces.finish(reviewId, {
          status: review.error ? 'error' : 'ok',
          attributes: { hasFindings: Boolean(review.response) },
        });
        this.metrics.increment(
          review.error ? 'subagent.review.failed' : 'subagent.review.completed',
        );
      }
      return {
        taskId, agent: agent.name, depth: scope.depth, result,
        response: response(result), review, error,
      };
    } finally {
      const finished = this.traces.finish(scope.spanId, {
        status: error ? 'error' : 'ok',
        attributes: {
          termination: result?.terminationReason ?? 'error',
          reviewFailed: Boolean(review.error),
        },
      });
      this.metrics.increment(
        error ? 'subagent.failed' : 'subagent.completed', 1,
        { agent: agent.name },
      );
      this.metrics.record(
        'subagent.duration_ms',
        (finished.endedAtMs ?? 0) - finished.startedAtMs,
        { kind: 'histogram', labels: { agent: agent.name } },
      );
      this.activeJobs -= 1;
      this.metrics.record('subagent.active', this.activeJobs, { kind: 'gauge' });
    }
  }

  close(): void {
    if (this.activeJobs) throw new Error('cannot close swarm while subagents are active');
    if (this.traces.get(this.root.spanId).status === 'running')
      this.traces.finish(this.root.spanId);
  }
}

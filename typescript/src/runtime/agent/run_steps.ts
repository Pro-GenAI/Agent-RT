/** Agent loop run steps: deadlines, model requests, token usage, and approvals. */
import {
	type JSONValue,
	type ModelMessage,
	RunCancelledError,
	type ToolCall,
} from '../types.js';
import { effectiveToolExecutionLimits } from '../tools/definitions.js';
import { type RateLimiter } from '../resilience/rate_limits.js';
import { type CapabilityGrant } from '../authorization/grants.js';
import { type RegisteredTool, type ToolRegistry } from '../tools/registry.js';
import { PromptInjectionDefense } from '../context/items.js';
import {
	type InputGuardrail,
	type OutputGuardrail,
} from '../guardrails/boundary.js';
import {
	type ModelProvider,
	type ModelRequest,
	type ModelResponse,
} from '../models/interfaces.js';
import {
	type AgentConfig,
	type CompiledExecutionPlan,
	type StreamEventHandler,
	type ToolExecutor,
	isStreamingModelProvider,
} from './config.js';
import { type RuntimeMetrics } from '../observability/metrics.js';
import { type TokenLedger } from '../observability/usage.js';
import { ContextAssembler } from '../context/assembler.js';
import { type ToolVisibilityFilter, Toolbase } from '../tools/toolbase.js';
import { Deadline, type ExecutionBudget } from '../resilience/budgets.js';
import { type AgentRunLimits } from './limits.js';
import { type DurableEventType } from '../checkpoints/events.js';
import { type IdempotencyStore } from '../resilience/idempotency.js';
import { ToolTimeoutError } from '../tools/programs.js';

/**
 * The AgentLoop fields the step functions read. AgentLoop keeps them private,
 * so the loop passes itself under this type instead of widening its own.
 */
export interface AgentLoopInternals {
	readonly provider: ModelProvider;
	readonly toolExecutor?: ToolExecutor;
	readonly toolRegistry?: ToolRegistry | Toolbase;
	readonly toolFilter?: ToolVisibilityFilter;
	readonly contextAssembler: ContextAssembler;
	readonly idempotencyStore?: IdempotencyStore;
	readonly capabilityGrant?: CapabilityGrant;
	readonly inputGuardrails: InputGuardrail[];
	readonly outputGuardrails: OutputGuardrail[];
	readonly promptInjectionDefense: PromptInjectionDefense;
	readonly rateLimiter?: RateLimiter;
	readonly executionBudget?: ExecutionBudget;
	readonly costEstimator?: (response: ModelResponse) => number;
	readonly metrics: RuntimeMetrics | undefined;
	readonly tokenLedger?: TokenLedger;
}

type Bound<F> = F extends (
	this: AgentLoopInternals,
	run: RunState,
	...args: infer A
) => infer R
	? (...args: A) => R
	: never;

/**
 * Values one agent run shares with its step functions. `toolPolicy` and
 * `activeVisibleToolNames` change while the run is in progress, so the loop
 * reads and writes them here.
 */
export interface RunState {
	readonly agent: AgentConfig;
	readonly limits: AgentRunLimits;
	readonly stopRequested?: () => boolean;
	readonly onStreamEvent?: StreamEventHandler;
	readonly signal?: AbortSignal;
	readonly toolContext: Record<string, unknown>;
	readonly taskId?: string;
	readonly userPrompt: string;
	readonly history: ModelMessage[];
	readonly started: number;
	readonly activeDeadline: Deadline | undefined;
	readonly eventsEnabled: boolean;
	toolPolicy: CompiledExecutionPlan['toolPolicy'];
	activeVisibleToolNames: Set<string>;
	readonly emitEvent: (
		type: DurableEventType,
		payload?: Record<string, JSONValue>,
	) => void;
	withDeadline: <T>(operation: Promise<T>) => Promise<T>;
	executeToolOperation: Bound<typeof executeToolOperation>;
	recordTokenUsage: Bound<typeof recordTokenUsage>;
}

export function recordStage(
	this: AgentLoopInternals,
	stage: string,
	stageStarted: number | undefined,
	labels: Record<string, string> = {},
): void {
	if (stageStarted === undefined || !this.metrics) return;
	this.metrics.record(
		'agent_loop.stage.duration_ms',
		performance.now() - stageStarted,
		{ kind: 'histogram', labels: { stage, ...labels } },
	);
}

export function withDeadline<T>(
	run: RunState,
	operation: Promise<T>,
): Promise<T> {
	const { activeDeadline, limits, signal, started } = run;
	if (signal?.aborted) throw new RunCancelledError();
	if (!signal && limits.timeoutMs === undefined && !activeDeadline) {
		return operation;
	}
	let remaining =
		limits.timeoutMs === undefined
			? undefined
			: limits.timeoutMs - (Date.now() - started);
	if (activeDeadline) {
		const deadlineRemaining = activeDeadline.remainingMs();
		remaining =
			remaining === undefined
				? deadlineRemaining
				: Math.min(remaining, deadlineRemaining);
	}
	if (remaining !== undefined && remaining <= 0) {
		throw new Error('loop_timeout');
	}

	let timer: ReturnType<typeof setTimeout> | undefined;
	let abortHandler: (() => void) | undefined;
	const contenders: Promise<T>[] = [operation];
	if (remaining !== undefined) {
		contenders.push(
			new Promise<T>((_, reject) => {
				timer = setTimeout(
					() => reject(new Error('loop_timeout')),
					remaining,
				);
			}),
		);
	}
	if (signal) {
		contenders.push(
			new Promise<T>((_, reject) => {
				abortHandler = () => reject(new RunCancelledError());
				signal.addEventListener('abort', abortHandler, {
					once: true,
				});
			}),
		);
	}

	return Promise.race(contenders).finally(() => {
		if (timer !== undefined) clearTimeout(timer);
		if (signal && abortHandler)
			signal.removeEventListener('abort', abortHandler);
	});
}

export async function executeToolOperation(
	this: AgentLoopInternals,
	run: RunState,
	call: ToolCall,
	registered?: RegisteredTool,
): Promise<unknown> {
	const {
		activeDeadline,
		limits,
		signal,
		started,
		toolContext,
		userPrompt,
		withDeadline,
	} = run;
	const timeoutMs = effectiveToolExecutionLimits(
		registered?.definition,
	).timeoutMs;
	let runRemaining =
		limits.timeoutMs === undefined
			? undefined
			: limits.timeoutMs - (Date.now() - started);
	if (activeDeadline) {
		const deadlineRemaining = activeDeadline.remainingMs();
		runRemaining =
			runRemaining === undefined
				? deadlineRemaining
				: Math.min(runRemaining, deadlineRemaining);
	}

	if (runRemaining !== undefined && runRemaining <= 0) {
		throw new Error('loop_timeout');
	}
	if (timeoutMs !== undefined && timeoutMs <= 0) {
		throw new ToolTimeoutError(call.name, timeoutMs);
	}

	const controller =
		timeoutMs !== undefined ? new AbortController() : undefined;
	let parentAbort: (() => void) | undefined;
	if (signal && controller) {
		parentAbort = () => controller.abort(signal.reason);
		if (signal.aborted) {
			controller.abort(signal.reason);
		} else {
			signal.addEventListener('abort', parentAbort, {
				once: true,
			});
		}
	}
	const toolSignal = controller?.signal ?? signal;

	const operation =
		(registered?.handler || registered?.contextualHandler) &&
		this.toolRegistry
			? this.toolRegistry.execute(call, toolSignal, {
					...toolContext,
					userPrompt,
					...(activeDeadline
						? {
								deadline: activeDeadline,
								deadlineRemainingMs:
									activeDeadline.remainingMs(),
							}
						: {}),
				})
			: this.toolExecutor
				? this.toolExecutor.execute(call, toolSignal)
				: Promise.reject(
						new Error(
							'tool executor required when no registered handler is configured',
						),
					);

	if (timeoutMs === undefined) return withDeadline(operation);

	let timer: ReturnType<typeof setTimeout> | undefined;
	const toolWins = runRemaining === undefined || timeoutMs <= runRemaining;
	try {
		return await Promise.race([
			withDeadline(operation),
			new Promise<unknown>((_, reject) => {
				timer = setTimeout(() => {
					const timeout = new ToolTimeoutError(call.name, timeoutMs);
					reject(timeout);
					controller?.abort(timeout);
				}, timeoutMs);
			}),
		]);
	} catch (error) {
		if (
			error instanceof ToolTimeoutError ||
			(controller?.signal.aborted && toolWins && !signal?.aborted)
		) {
			throw new ToolTimeoutError(call.name, timeoutMs);
		}
		throw error;
	} finally {
		if (timer !== undefined) clearTimeout(timer);
		if (signal && controller && parentAbort) {
			signal.removeEventListener('abort', parentAbort);
		}
	}
}

export function recordTokenUsage(
	this: AgentLoopInternals,
	run: RunState,
	response: ModelResponse,
	request: ModelRequest,
): void {
	const { agent, taskId, toolContext } = run;
	if (!this.tokenLedger || !response.usage) return;
	this.tokenLedger.recordModelUsage(response.usage, {
		...(typeof toolContext.userId === 'string' && toolContext.userId
			? { userId: toolContext.userId }
			: {}),
		...(typeof toolContext.tenantId === 'string' && toolContext.tenantId
			? { tenantId: toolContext.tenantId }
			: {}),
		...(taskId ? { taskId } : {}),
		agentId: agent.name,
		model: response.model ?? request.model ?? agent.model.model,
	});
}

export async function completeRequest(
	this: AgentLoopInternals,
	run: RunState,
	request: ModelRequest,
): Promise<ModelResponse> {
	const {
		activeDeadline,
		agent,
		emitEvent,
		eventsEnabled,
		onStreamEvent,
		recordTokenUsage,
		toolContext,
		withDeadline,
	} = run;
	this.executionBudget?.consumeModelCall();
	if (activeDeadline) {
		request = {
			...request,
			metadata: {
				...(request.metadata ?? {}),
				deadlineRemainingMs: activeDeadline.remainingMs(),
			},
		};
	}
	if (this.rateLimiter) {
		const keys = [
			'model:' + (request.model ?? agent.model.model),
			'provider:' + this.provider.name,
		];
		if (typeof toolContext.userId === 'string' && toolContext.userId) {
			keys.push('user:' + toolContext.userId);
		}
		if (typeof toolContext.tenantId === 'string' && toolContext.tenantId) {
			keys.push('tenant:' + toolContext.tenantId);
		}
		this.rateLimiter.checkMany(keys);
	}
	if (eventsEnabled) {
		emitEvent('model_requested', {
			model: request.model ?? '',
			messageCount: request.messages.length,
		});
	}
	if (!onStreamEvent) {
		const response = await withDeadline(this.provider.complete(request));
		if (this.executionBudget && this.costEstimator) {
			this.executionBudget.consumeCost(
				Math.max(0, this.costEstimator(response)),
			);
		}
		recordTokenUsage(response, request);
		if (eventsEnabled) {
			emitEvent('model_completed', {
				model: response.model ?? request.model ?? '',
				finishReason: response.finishReason ?? 'other',
			});
		}
		return response;
	}
	if (!isStreamingModelProvider(this.provider)) {
		throw new TypeError(
			'streaming requires a provider that implements stream()',
		);
	}

	const stream = this.provider.stream(request);
	const iterator = stream[Symbol.asyncIterator]();
	let completed: ModelResponse | undefined;
	while (true) {
		const next = await withDeadline(iterator.next());
		if (next.done) break;
		const event = next.value;
		await withDeadline(Promise.resolve(onStreamEvent(event)));
		if (event.type === 'completed') {
			if (!event.response) {
				throw new Error(
					'completed stream event must include a response',
				);
			}
			completed = event.response;
		}
	}

	if (!completed) {
		throw new Error('model stream ended without a completed response');
	}
	if (this.executionBudget && this.costEstimator) {
		this.executionBudget.consumeCost(
			Math.max(0, this.costEstimator(completed)),
		);
	}
	recordTokenUsage(completed, request);
	if (eventsEnabled) {
		emitEvent('model_completed', {
			model: completed.model ?? request.model ?? '',
			finishReason: completed.finishReason ?? 'other',
		});
	}
	return completed;
}

// A run that stops with `waiting_for_approval` leaves a
// `tool_not_executed` placeholder per unexecuted call. When resumed,
// execute those calls (their approval is checked again) instead of
// asking the model to re-issue them under a new call id, which would
// orphan a `once` approval.
export function pendingApprovalCalls(run: RunState): {
	call: ToolCall;
	index: number;
}[] {
	const { history } = run;
	let lastAssistant = -1;
	for (let index = history.length - 1; index >= 0; index -= 1) {
		if (history[index].role === 'assistant') {
			lastAssistant = index;
			break;
		}
	}
	const assistantCalls =
		lastAssistant >= 0 ? history[lastAssistant].toolCalls : undefined;
	if (!assistantCalls || assistantCalls.length === 0) return [];
	const placeholders = new Map<string, number>();
	for (let index = lastAssistant + 1; index < history.length; index += 1) {
		const message = history[index];
		if (message.role !== 'tool' || message.toolCallId === undefined)
			return [];
		for (const part of message.content) {
			const error =
				part.type === 'json' &&
				typeof part.data === 'object' &&
				part.data !== null
					? (part.data as { error?: Record<string, unknown> }).error
					: undefined;
			if (
				error?.type === 'tool_not_executed' &&
				error.reason === 'waiting_for_approval'
			) {
				placeholders.set(message.toolCallId, index);
			}
		}
	}
	return assistantCalls
		.filter((call) => placeholders.has(call.id))
		.map((call) => ({
			call,
			index: placeholders.get(call.id) as number,
		}));
}

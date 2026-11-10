/** The agent loop. */
import {
	type JSONValue,
	type ModelMessage,
	RunCancelledError,
	type ToolCall,
	type WorkflowState,
} from '../types.js';
import { type FailureDisposition } from '../resilience/retry.js';
import { type RateLimiter } from '../resilience/rate_limits.js';
import { type CapabilityGrant } from '../authorization/grants.js';
import { applyGuardrailResult } from '../guardrails/results.js';
import { type ToolRegistry } from '../tools/registry.js';
import {
	type ContextItem,
	type ContextSelectionPolicy,
	PromptInjectionDefense,
} from '../context/items.js';
import {
	type BoundaryGuardrailPolicy,
	DEFAULT_BOUNDARY_GUARDRAIL_POLICY,
	type InputGuardrail,
	type OutputGuardrail,
	makeDefaultInputGuardrail,
	makeDefaultOutputGuardrail,
	resolveBoundaryGuardrailPolicy,
} from '../guardrails/boundary.js';
import {
	type ModelProvider,
	type ModelResponse,
} from '../models/interfaces.js';
import {
	type AgentConfig,
	type CompiledExecutionPlan,
	type StreamEventHandler,
	type ToolExecutor,
} from './config.js';
import { resolveAgentMemory } from '../memory/sources.js';
import { type RuntimeMetrics } from '../observability/metrics.js';
import { type TokenLedger } from '../observability/usage.js';
import { ContextAssembler } from '../context/assembler.js';
import { type ToolVisibilityFilter, Toolbase } from '../tools/toolbase.js';
import {
	StructuredOutputValidationError,
	validateStructuredMessage,
} from '../tools/validation.js';
import {
	BudgetExceededError,
	Deadline,
	type ExecutionBudget,
	type Finalizer,
	type LoopDetector,
	type TerminationReason,
	runFinalizers,
} from '../resilience/budgets.js';
import { type AgentRunLimits, type AgentRunResult } from './limits.js';
import { type AgentCheckpoint } from '../checkpoints/store.js';
import {
	type DurableEventType,
	type EventStore,
} from '../checkpoints/events.js';
import { type IdempotencyStore } from '../resilience/idempotency.js';

import * as planCompiler from './plan_compiler.js';
import * as runSteps from './run_steps.js';
import { type AgentLoopInternals, type RunState } from './run_steps.js';
import * as toolExecution from './tool_execution.js';

export class AgentLoop {
	private readonly toolFilter?: ToolVisibilityFilter;

	constructor(
		private readonly provider: ModelProvider,
		private readonly toolExecutor?: ToolExecutor,
		private readonly toolRegistry?: ToolRegistry | Toolbase,
		toolFilter?: ToolVisibilityFilter,
		private readonly contextAssembler: ContextAssembler = new ContextAssembler(),
		private readonly eventStore?: EventStore,
		private readonly idempotencyStore?: IdempotencyStore,
		private readonly capabilityGrant?: CapabilityGrant,
		private readonly inputGuardrails: InputGuardrail[] = [],
		private readonly outputGuardrails: OutputGuardrail[] = [],
		private readonly promptInjectionDefense: PromptInjectionDefense = new PromptInjectionDefense(),
		private readonly rateLimiter?: RateLimiter,
		private readonly executionBudget?: ExecutionBudget,
		private readonly deadline?: Deadline,
		private readonly loopDetector?: LoopDetector,
		private readonly costEstimator?: (response: ModelResponse) => number,
		private readonly finalizers: Finalizer[] = [],
		private readonly metrics: RuntimeMetrics | undefined = undefined,
		private readonly tokenLedger?: TokenLedger,
		private readonly responseFailureClassifier?: (
			response: ModelResponse,
		) =>
			| Promise<FailureDisposition | undefined>
			| FailureDisposition
			| undefined,
		private readonly boundaryGuardrailPolicy: BoundaryGuardrailPolicy | null = DEFAULT_BOUNDARY_GUARDRAIL_POLICY,
	) {
		this.toolFilter =
			this.toolRegistry instanceof Toolbase
				? this.toolRegistry.visibilityFilter(toolFilter)
				: toolFilter;
		if (boundaryGuardrailPolicy !== null)
			resolveBoundaryGuardrailPolicy(boundaryGuardrailPolicy);
	}

	compilePlan(
		agent: AgentConfig,
		contextItems: ContextItem[] = [],
		contextPolicy?: ContextSelectionPolicy,
	): CompiledExecutionPlan {
		return planCompiler.compilePlan.call(
			this as unknown as AgentLoopInternals,
			agent,
			contextItems,
			contextPolicy,
		);
	}

	async run(
		agent: AgentConfig,
		messages: ModelMessage[],
		limits: AgentRunLimits = {},
		stopRequested?: () => boolean,
		onStreamEvent?: StreamEventHandler,
		signal?: AbortSignal,
		toolContext: Record<string, unknown> = {},
		workflowState?: WorkflowState,
		contextItems: ContextItem[] = [],
		contextPolicy?: ContextSelectionPolicy,
		contextMetadata: Record<string, unknown> = {},
		resumeCheckpoint?: AgentCheckpoint,
		taskId?: string,
	): Promise<AgentRunResult> {
		try {
			return await this.runInternal(
				agent,
				messages,
				limits,
				stopRequested,
				onStreamEvent,
				signal,
				toolContext,
				workflowState,
				contextItems,
				contextPolicy,
				contextMetadata,
				resumeCheckpoint,
				taskId,
			);
		} finally {
			await runFinalizers(this.finalizers);
		}
	}

	private async runInternal(
		agent: AgentConfig,
		messages: ModelMessage[],
		limits: AgentRunLimits = {},
		stopRequested?: () => boolean,
		onStreamEvent?: StreamEventHandler,
		signal?: AbortSignal,
		toolContext: Record<string, unknown> = {},
		workflowState?: WorkflowState,
		contextItems: ContextItem[] = [],
		contextPolicy?: ContextSelectionPolicy,
		contextMetadata: Record<string, unknown> = {},
		resumeCheckpoint?: AgentCheckpoint,
		taskId?: string,
	): Promise<AgentRunResult> {
		const internals = this as unknown as AgentLoopInternals;
		const recordStage = runSteps.recordStage.bind(internals);

		const defaultInputGuardrail =
			this.boundaryGuardrailPolicy === null
				? undefined
				: makeDefaultInputGuardrail(this.boundaryGuardrailPolicy);
		const effectiveInputGuardrails = defaultInputGuardrail
			? [defaultInputGuardrail, ...this.inputGuardrails]
			: this.inputGuardrails;
		const inputGuardrailStarted =
			this.metrics && effectiveInputGuardrails.length > 0
				? performance.now()
				: undefined;
		let guardedMessages = [...messages];
		for (const guardrail of effectiveInputGuardrails) {
			guardedMessages = applyGuardrailResult(
				guardedMessages,
				guardrail(guardedMessages),
			);
		}
		recordStage('input_guardrails', inputGuardrailStarted);

		const latestUserMessage = [...messages]
			.reverse()
			.find((message) => message.role === 'user');
		const userPrompt =
			latestUserMessage?.content
				.filter((part) => part.type === 'text')
				.map((part) => part.text ?? '')
				.join('') ?? '';
		if (agent.memory?.length) {
			contextItems = [
				...contextItems,
				...(await resolveAgentMemory(agent.memory, userPrompt)),
			];
		}

		if (resumeCheckpoint && resumeCheckpoint.agentName !== agent.name) {
			throw new Error(
				'checkpoint belongs to agent ' +
					resumeCheckpoint.agentName +
					', not ' +
					agent.name,
			);
		}
		const history: ModelMessage[] = resumeCheckpoint
			? [...resumeCheckpoint.messages]
			: [
					{
						role: 'system',
						content: [{ type: 'text', text: agent.instructions }],
					},
					...guardedMessages,
				];
		if (workflowState === undefined && resumeCheckpoint?.workflowState) {
			workflowState = resumeCheckpoint.workflowState;
		}
		const maxTurns = limits.maxTurns ?? 16;
		const maxToolCalls = limits.maxToolCalls ?? 64;
		let turns = resumeCheckpoint?.turns ?? 0;
		let toolCalls = resumeCheckpoint?.toolCalls ?? 0;
		let totalTokens = resumeCheckpoint?.totalTokens ?? 0;
		let finalResponse: ModelResponse | undefined;
		let structuredOutput: unknown;
		let repairAttempts = 0;
		const eventsEnabled = Boolean(this.eventStore && taskId);
		let eventCounter =
			this.eventStore && taskId ? this.eventStore.list(taskId).length : 0;
		const emitEvent = (
			type: DurableEventType,
			payload: Record<string, JSONValue> = {},
		): void => {
			if (!this.eventStore || !taskId) return;
			// The counter starts from the visible event count, which can lag
			// behind ids already used once a retention policy archived or
			// expired events, so skip ids that are taken.
			for (;;) {
				eventCounter += 1;
				try {
					this.eventStore.append({
						eventId: taskId + ':run:' + eventCounter,
						taskId,
						type,
						payload,
					});
					return;
				} catch (error) {
					if (
						!(error instanceof Error) ||
						!error.message.includes('already exists')
					) {
						throw error;
					}
				}
			}
		};
		if (eventsEnabled) {
			emitEvent('lifecycle_transition', {
				from: 'submitted',
				to: 'running',
			});
		}
		let plan = this.compilePlan(agent, contextItems, contextPolicy);
		const toolPolicy = plan.toolPolicy;
		const isSimpleTextMessage = (item: ModelMessage): boolean =>
			item.role !== 'tool' &&
			(item.toolCalls?.length ?? 0) === 0 &&
			item.content.every((part) => part.type === 'text');
		let simpleTextRequest =
			plan.simpleTextFastPath &&
			workflowState === undefined &&
			contextItems.length === 0 &&
			contextPolicy === undefined &&
			Object.keys(contextMetadata).length === 0 &&
			history.every(isSimpleTextMessage);
		const activeVisibleToolNames = new Set<string>();
		const started = Date.now();
		// Trajectory state must not leak between runs (or concurrent runs that
		// share one AgentLoop), so every run observes its own detector.
		const loopDetector = this.loopDetector
			? Object.assign(
					Object.create(
						Object.getPrototypeOf(this.loopDetector),
					) as LoopDetector,
					this.loopDetector,
				)
			: undefined;
		loopDetector?.reset();
		const activeDeadline =
			this.deadline ??
			(limits.timeoutMs === undefined
				? undefined
				: Deadline.after(limits.timeoutMs));
		const run = {
			agent,
			limits,
			stopRequested,
			onStreamEvent,
			signal,
			toolContext,
			taskId,
			userPrompt,
			history,
			started,
			activeDeadline,
			eventsEnabled,
			toolPolicy,
			activeVisibleToolNames,
			emitEvent,
		} as RunState;
		run.withDeadline = <T>(operation: Promise<T>) =>
			runSteps.withDeadline(run, operation);
		run.executeToolOperation = runSteps.executeToolOperation.bind(
			internals,
			run,
		);
		run.recordTokenUsage = runSteps.recordTokenUsage.bind(internals, run);
		const executeToolCall = toolExecution.executeToolCall.bind(
			internals,
			run,
		);
		const completeRequest = runSteps.completeRequest.bind(internals, run);
		const pendingApprovalCalls = runSteps.pendingApprovalCalls.bind(
			undefined,
			run,
		);

		let responseFailure: FailureDisposition | undefined;

		const finish = (
			terminationReason: TerminationReason,
		): AgentRunResult => {
			const terminal =
				terminationReason === 'completed'
					? 'completed'
					: terminationReason === 'cancelled'
						? 'canceled'
						: terminationReason === 'waiting_for_approval'
							? 'waiting_for_approval'
							: 'failed';
			if (eventsEnabled) {
				emitEvent('lifecycle_transition', {
					from: 'running',
					to: terminal,
					reason: terminationReason,
				});
			}
			return {
				messages: history,
				finalResponse,
				terminationReason,
				turns,
				toolCalls,
				totalTokens,
				structuredOutput,
				...(responseFailure ? { failure: responseFailure } : {}),
			};
		};
		if (resumeCheckpoint) {
			const pending = pendingApprovalCalls();
			if (pending.length > 0) {
				if (this.toolFilter) {
					const selectedNames = new Set(
						await this.toolFilter(
							{
								agent,
								messages: [...history],
								turn: turns,
								toolCalls,
								runtimeContext: toolContext,
							},
							[...plan.visibleTools],
						),
					);
					run.activeVisibleToolNames = new Set(
						plan.visibleTools
							.filter((tool) => selectedNames.has(tool.name))
							.map((tool) => tool.name),
					);
				} else {
					run.activeVisibleToolNames = new Set(plan.visibleToolNames);
				}
				if (toolCalls + pending.length > maxToolCalls) {
					return finish('max_tool_calls');
				}
				for (const { call, index } of pending) {
					const resumed = await executeToolCall(call);
					if (resumed.termination) return finish(resumed.termination);
					if (resumed.message) {
						history[index] = resumed.message;
						toolCalls += 1;
					}
				}
			}
		}

		while (true) {
			if (signal?.aborted) return finish('cancelled');
			if (stopRequested?.()) return finish('stop_requested');
			if (turns >= maxTurns) return finish('max_turns');
			const totalTokenLimit = limits.maxTotalTokens;
			if (
				Number.isFinite(totalTokenLimit) &&
				totalTokens >= (totalTokenLimit as number)
			)
				return finish('budget_exhausted');
			if (
				limits.timeoutMs !== undefined &&
				Date.now() - started >= limits.timeoutMs
			)
				return finish('timeout');
			if (activeDeadline?.expired) return finish('timeout');
			if (this.toolRegistry?.version !== plan.toolRegistryVersion) {
				plan = this.compilePlan(agent, contextItems, contextPolicy);
				run.toolPolicy = plan.toolPolicy;
				simpleTextRequest =
					plan.simpleTextFastPath &&
					workflowState === undefined &&
					contextItems.length === 0 &&
					contextPolicy === undefined &&
					Object.keys(contextMetadata).length === 0 &&
					history.every(isSimpleTextMessage);
			}
			let visibleTools = [...plan.visibleTools];
			let toolSelection = plan.toolSelection;

			if (this.toolFilter) {
				const selectedNames = new Set(
					await this.toolFilter(
						{
							agent,
							messages: [...history],
							turn: turns,
							toolCalls,
							runtimeContext: toolContext,
						},
						visibleTools,
					),
				);
				visibleTools = visibleTools.filter((tool) =>
					selectedNames.has(tool.name),
				);
				run.activeVisibleToolNames = new Set(
					visibleTools.map((tool) => tool.name),
				);
				if (run.toolPolicy) {
					const missingRequired = (
						run.toolPolicy.required ?? []
					).filter((name) => !run.activeVisibleToolNames.has(name));
					if (missingRequired.length > 0) {
						throw new Error(
							`required tools are unavailable: ${missingRequired.sort().join(', ')}`,
						);
					}
					toolSelection = {
						required: [...(run.toolPolicy.required ?? [])].sort(),
						preferred: (run.toolPolicy.preferred ?? []).filter(
							(name) => run.activeVisibleToolNames.has(name),
						),
					};
				} else {
					toolSelection = undefined;
				}
			} else {
				run.activeVisibleToolNames = new Set(plan.visibleToolNames);
			}
			const structuredOutputRequirement = plan.structuredOutput;
			const requestAssemblyStarted = this.metrics
				? performance.now()
				: undefined;
			const request = simpleTextRequest
				? {
						messages: history,
						model: plan.modelSettings.model,
						...(plan.modelSettings.temperature === undefined
							? {}
							: { temperature: plan.modelSettings.temperature }),
						...(Number.isFinite(plan.modelSettings.maxOutputTokens)
							? {
									maxOutputTokens: plan.modelSettings
										.maxOutputTokens as number,
								}
							: {}),
						...(plan.modelSettings.metadata === undefined
							? {}
							: { metadata: plan.modelSettings.metadata }),
						...(signal === undefined ? {} : { signal }),
					}
				: this.contextAssembler.assembleRequest(agent, [...history], {
						tools: visibleTools,
						workflowState,
						contextItems,
						runtimeMetadata: contextMetadata,
						policy: contextPolicy,
						structuredOutput: structuredOutputRequirement,
						toolSelection,
						signal,
					});
			recordStage('request_assembly', requestAssemblyStarted, {
				path: simpleTextRequest ? 'fast' : 'general',
			});
			const modelCallStarted = this.metrics
				? performance.now()
				: undefined;
			let response: ModelResponse;
			try {
				response = await completeRequest(request);
			} catch (error) {
				if (signal?.aborted || error instanceof RunCancelledError) {
					return finish('cancelled');
				}
				if (
					error instanceof Error &&
					error.message === 'loop_timeout'
				) {
					return finish('timeout');
				}
				if (error instanceof BudgetExceededError) {
					return finish('budget_exhausted');
				}
				throw error;
			} finally {
				recordStage('model_call', modelCallStarted);
			}
			const defaultOutputGuardrail =
				this.boundaryGuardrailPolicy === null
					? undefined
					: makeDefaultOutputGuardrail(this.boundaryGuardrailPolicy);
			const effectiveOutputGuardrails = defaultOutputGuardrail
				? [defaultOutputGuardrail, ...this.outputGuardrails]
				: this.outputGuardrails;
			const outputGuardrailStarted =
				this.metrics && effectiveOutputGuardrails.length > 0
					? performance.now()
					: undefined;
			let guardedMessage = response.message;
			for (const guardrail of effectiveOutputGuardrails) {
				guardedMessage = applyGuardrailResult(
					guardedMessage,
					guardrail(guardedMessage),
				);
			}
			recordStage('output_guardrails', outputGuardrailStarted);
			if (guardedMessage !== response.message) {
				response = { ...response, message: guardedMessage };
			}
			finalResponse = response;
			turns += 1;
			history.push(response.message);
			if (simpleTextRequest) {
				simpleTextRequest = isSimpleTextMessage(response.message);
			}
			if (loopDetector?.observe(response.message)) {
				return finish('loop_detected');
			}

			if (response.usage) {
				totalTokens +=
					response.usage.totalTokens ??
					(response.usage.inputTokens ?? 0) +
						(response.usage.outputTokens ?? 0);
			}
			if (
				Number.isFinite(totalTokenLimit) &&
				totalTokens >= (totalTokenLimit as number)
			) {
				return finish('budget_exhausted');
			}
			if (this.responseFailureClassifier) {
				responseFailure =
					await this.responseFailureClassifier(response);
				if (responseFailure) return finish('model_response_failure');
			}

			const calls = response.message.toolCalls ?? [];
			if (calls.length === 0) {
				const requirements = agent.output;
				if (
					requirements !== undefined &&
					(requirements.format === 'json' ||
						requirements.schema !== undefined)
				) {
					try {
						structuredOutput = validateStructuredMessage(
							response.message,
							requirements,
						);
					} catch (error) {
						if (!(error instanceof StructuredOutputValidationError))
							throw error;
						const maxRepairs = Math.max(
							requirements.maxRepairAttempts ?? 1,
							0,
						);
						if (repairAttempts < maxRepairs && turns < maxTurns) {
							repairAttempts += 1;
							history.push({
								role: 'user',
								content: [
									{
										type: 'text',
										text:
											'Your previous response did not satisfy the required structured output. ' +
											'Return corrected JSON only. Validation errors: ' +
											error.issues.join('; '),
									},
								],
							});
							continue;
						}
						throw error;
					}
				}
				return finish('completed');
			}
			const answered = new Set<ToolCall>();
			const recordResult = (
				call: ToolCall,
				message: ModelMessage,
			): void => {
				toolCalls += 1;
				history.push(message);
				answered.add(call);
			};
			// Keep the transcript provider-valid: every tool call the model made
			// gets a tool message, even when the run stops early.
			const finishTurn = (
				termination: TerminationReason,
			): AgentRunResult => {
				for (const call of calls) {
					if (answered.has(call)) continue;
					history.push({
						role: 'tool',
						content: [
							{
								type: 'json',
								data: {
									error: {
										type: 'tool_not_executed',
										tool: call.name,
										reason: termination,
									},
								},
							},
						],
						toolCallId: call.id,
					});
				}
				return finish(termination);
			};
			// Runs a batch concurrently, keeps every completed result (even when a
			// sibling needs approval or fails), then reports the first stop/failure.
			const runBatch = async (
				batch: ToolCall[],
			): Promise<TerminationReason | undefined> => {
				const outcomes = await Promise.allSettled(
					batch.map(executeToolCall),
				);
				let failure: { error: unknown } | undefined;
				let termination: TerminationReason | undefined;
				outcomes.forEach((outcome, index) => {
					if (outcome.status === 'rejected') {
						failure ??= { error: outcome.reason };
						return;
					}
					if (outcome.value.termination) {
						termination ??= outcome.value.termination;
					} else if (outcome.value.message) {
						recordResult(batch[index], outcome.value.message);
					}
				});
				if (failure) throw failure.error;
				return termination;
			};

			if (toolCalls + calls.length > maxToolCalls)
				return finishTurn('max_tool_calls');

			if (limits.concurrentToolCalls && calls.length > 1) {
				let parallelBatch: ToolCall[] = [];

				for (const call of calls) {
					let sequential = false;
					if (this.toolRegistry) {
						try {
							sequential =
								this.toolRegistry.get(call.name).definition
									.executionMode === 'sequential';
						} catch {
							sequential = false;
						}
					}

					if (sequential) {
						if (parallelBatch.length > 0) {
							const termination = await runBatch(parallelBatch);
							parallelBatch = [];
							if (termination) return finishTurn(termination);
						}

						const result = await executeToolCall(call);
						if (result.termination)
							return finishTurn(result.termination);
						if (result.message) recordResult(call, result.message);
					} else {
						parallelBatch.push(call);
					}
				}

				if (parallelBatch.length > 0) {
					const termination = await runBatch(parallelBatch);
					if (termination) return finishTurn(termination);
				}
			} else {
				for (const call of calls) {
					const result = await executeToolCall(call);
					if (result.termination)
						return finishTurn(result.termination);
					if (result.message) recordResult(call, result.message);
				}
			}
		}
	}

	async runStreaming(
		agent: AgentConfig,
		messages: ModelMessage[],
		onEvent: StreamEventHandler,
		limits: AgentRunLimits = {},
		stopRequested?: () => boolean,
		signal?: AbortSignal,
		toolContext: Record<string, unknown> = {},
		workflowState?: WorkflowState,
		contextItems: ContextItem[] = [],
		contextPolicy?: ContextSelectionPolicy,
		contextMetadata: Record<string, unknown> = {},
		resumeCheckpoint?: AgentCheckpoint,
		taskId?: string,
	): Promise<AgentRunResult> {
		return this.run(
			agent,
			messages,
			limits,
			stopRequested,
			onEvent,
			signal,
			toolContext,
			workflowState,
			contextItems,
			contextPolicy,
			contextMetadata,
			resumeCheckpoint,
			taskId,
		);
	}
}

/** Agent loop tool-call execution: visibility, approval, guardrails, and timeouts. */
import {
	type ModelMessage,
	RunCancelledError,
	type ToolCall,
} from '../types.js';
import { toolPolicyPermits } from '../tools/definitions.js';
import { ApprovalRequiredError } from '../authorization/approvals.js';
import { type RegisteredTool } from '../tools/registry.js';
import { marshalToolResult } from '../tools/todos.js';
import {
	ToolArgumentValidationError,
	toolValidationErrorPayload,
	validateToolArguments,
} from '../tools/validation.js';
import {
	type TerminationReason,
	argumentsFingerprint,
} from '../resilience/budgets.js';
import { ToolTimeoutError, toolTimeoutPayload } from '../tools/programs.js';

import { type AgentLoopInternals, type RunState } from './run_steps.js';

export async function executeToolCall(
	this: AgentLoopInternals,
	run: RunState,
	call: ToolCall,
): Promise<{
	message?: ModelMessage;
	termination?: TerminationReason;
}> {
	const {
		emitEvent,
		eventsEnabled,
		executeToolOperation,
		limits,
		signal,
		started,
		stopRequested,
		taskId,
	} = run;
	if (signal?.aborted) return { termination: 'cancelled' };
	if (stopRequested?.()) return { termination: 'stop_requested' };
	if (
		limits.timeoutMs !== undefined &&
		Date.now() - started >= limits.timeoutMs
	)
		return { termination: 'timeout' };
	if (
		!toolPolicyPermits(run.toolPolicy, call.name) ||
		(this.capabilityGrant !== undefined &&
			!run.activeVisibleToolNames.has(call.name)) ||
		(this.toolFilter !== undefined &&
			!run.activeVisibleToolNames.has(call.name))
	) {
		return {
			message: {
				role: 'tool',
				content: [
					{
						type: 'json',
						data: toolValidationErrorPayload(call, [
							'tool is not permitted by selection policy',
						]),
					},
				],
				toolCallId: call.id,
			},
		};
	}

	if (call.argumentError !== undefined) {
		return {
			message: {
				role: 'tool',
				content: [
					{
						type: 'json',
						data: toolValidationErrorPayload(call, [
							call.argumentError,
						]),
					},
				],
				toolCallId: call.id,
			},
		};
	}

	let registered: RegisteredTool | undefined;
	if (this.toolRegistry) {
		let issues: string[] | undefined;
		try {
			registered = this.toolRegistry.get(call.name);
			if (!run.activeVisibleToolNames.has(call.name)) {
				return {
					message: {
						role: 'tool',
						content: [
							{
								type: 'json',
								data: toolValidationErrorPayload(call, [
									'tool is not permitted by active visibility policy',
								]),
							},
						],
						toolCallId: call.id,
					},
				};
			}
			if (!registered.enabled) {
				throw new ToolArgumentValidationError(call.name, [
					'tool is disabled',
				]);
			}
			validateToolArguments(registered.definition, call.arguments);
		} catch (error) {
			if (error instanceof ToolArgumentValidationError) {
				issues = error.issues;
			} else {
				issues = [`tool is not registered: ${call.name}`];
			}
		}
		if (issues) {
			return {
				message: {
					role: 'tool',
					content: [
						{
							type: 'json',
							data: toolValidationErrorPayload(call, issues),
						},
					],
					toolCallId: call.id,
				},
			};
		}
	}

	if (eventsEnabled) {
		emitEvent('tool_requested', {
			tool: call.name,
			toolCallId: call.id,
		});
	}
	const idempotencyScope = taskId ? taskId + ':tool:' + call.name : undefined;
	// Bind the key to the arguments so a reused call id with different
	// arguments executes instead of replaying a stale result.
	const idempotencyKey = call.id + ':' + argumentsFingerprint(call.arguments);
	if (this.idempotencyStore && idempotencyScope) {
		const existing = this.idempotencyStore.get(
			idempotencyScope,
			idempotencyKey,
		);
		if (existing) {
			if (eventsEnabled) {
				emitEvent('tool_completed', {
					tool: call.name,
					toolCallId: call.id,
					replayed: true,
				});
			}
			return {
				message: {
					role: 'tool',
					content: marshalToolResult(existing.value),
					toolCallId: call.id,
				},
			};
		}
	}

	try {
		const value = await executeToolOperation(call, registered);
		if (this.idempotencyStore && idempotencyScope) {
			this.idempotencyStore.put({
				scope: idempotencyScope,
				key: idempotencyKey,
				value,
			});
		}
		if (eventsEnabled) {
			emitEvent('tool_completed', {
				tool: call.name,
				toolCallId: call.id,
				replayed: false,
			});
		}
		return {
			message: {
				role: 'tool',
				content: marshalToolResult(value),
				toolCallId: call.id,
			},
		};
	} catch (error) {
		if (error instanceof ApprovalRequiredError) {
			if (eventsEnabled) {
				emitEvent('approval_requested', {
					approvalId: error.request.id,
					tool: call.name,
					toolCallId: call.id,
					sideEffect: error.request.sideEffect,
					reason: error.request.reason,
				});
			}
			return { termination: 'waiting_for_approval' };
		}
		if (signal?.aborted || error instanceof RunCancelledError) {
			return { termination: 'cancelled' };
		}
		if (error instanceof ToolTimeoutError) {
			return {
				message: {
					role: 'tool',
					content: [
						{
							type: 'json',
							data: toolTimeoutPayload(call, error.timeoutMs),
						},
					],
					toolCallId: call.id,
				},
			};
		}
		if (error instanceof Error && error.message === 'loop_timeout') {
			return { termination: 'timeout' };
		}
		throw error;
	}
}

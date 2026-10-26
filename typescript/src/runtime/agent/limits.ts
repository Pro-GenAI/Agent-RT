/** Agent run limits, secure presets, and run results. */
import { type ModelMessage } from '../types.js';
import { ApprovalManager } from '../authorization/approvals.js';
import { type FailureDisposition } from '../resilience/retry.js';
import {
	PermissionEngine,
	type PermissionRule,
} from '../authorization/permissions.js';
import { ToolRegistry } from '../tools/registry.js';
import { type ModelResponse } from '../models/interfaces.js';
import { type TerminationReason } from '../resilience/budgets.js';

export interface AgentRunLimits {
	maxTurns?: number;
	maxToolCalls?: number;
	timeoutMs?: number;
	maxTotalTokens?: number;
	concurrentToolCalls?: boolean;
}

/** Finite bounds for untrusted or production runs (the API server's defaults). */
export const SECURE_RUN_LIMITS: Readonly<AgentRunLimits> = Object.freeze({
	maxTurns: 16,
	maxToolCalls: 64,
	timeoutMs: 120_000,
	maxTotalTokens: 100_000,
});

/**
 * ToolRegistry preset for untrusted or production use. Permissions deny by
 * default: only `none`/`read` tools are allowed until `permissionRules` allow
 * more (deny rules always win), and consequential/destructive tools still need
 * human approval.
 */
export function secureToolRegistry(
	permissionRules: PermissionRule[] = [],
	approvalManager: ApprovalManager = new ApprovalManager(),
): ToolRegistry {
	return new ToolRegistry(
		{},
		{},
		new PermissionEngine([
			{ effect: 'allow', sideEffects: ['none', 'read'] },
			...permissionRules,
		]),
		[],
		[],
		approvalManager,
	);
}

export interface AgentRunResult {
	messages: ModelMessage[];
	finalResponse?: ModelResponse;
	terminationReason: TerminationReason;
	turns: number;
	toolCalls: number;
	totalTokens: number;
	structuredOutput?: unknown;
	failure?: FailureDisposition;
}

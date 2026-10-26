/** Failure classification, retry policies, and recovery routing. */
import {
	ApprovalDeniedError,
	ApprovalRequiredError,
} from '../authorization/approvals.js';
import { CircuitOpenError, RateLimitExceededError } from './rate_limits.js';
import { PermissionDeniedError } from '../authorization/permissions.js';
import { GuardrailViolationError } from '../guardrails/results.js';
import {
	StructuredOutputValidationError,
	ToolArgumentValidationError,
} from '../tools/validation.js';
import { type ExecutionBudget } from './budgets.js';
import { ToolTimeoutError } from '../tools/programs.js';

export type FailureKind =
	| 'transient_dependency'
	| 'model_correctable'
	| 'user_correctable'
	| 'policy'
	| 'terminal_system';

export interface FailureDisposition {
	kind: FailureKind;
	retryable: boolean;
	reason: string;
}

export function classifyFailure(error: unknown): FailureDisposition {
	if (
		error instanceof ToolTimeoutError ||
		error instanceof CircuitOpenError ||
		error instanceof RateLimitExceededError ||
		(error instanceof Error &&
			/timeout|temporar|connection/i.test(error.message))
	) {
		return {
			kind: 'transient_dependency',
			retryable: true,
			reason: 'dependency or timeout failure',
		};
	}
	if (
		error instanceof ToolArgumentValidationError ||
		error instanceof StructuredOutputValidationError
	) {
		return {
			kind: 'model_correctable',
			retryable: false,
			reason: 'model output or tool arguments can be repaired',
		};
	}
	if (error instanceof ApprovalRequiredError) {
		return {
			kind: 'user_correctable',
			retryable: false,
			reason: 'human input or approval is required',
		};
	}
	if (
		error instanceof PermissionDeniedError ||
		error instanceof ApprovalDeniedError ||
		error instanceof GuardrailViolationError
	) {
		return {
			kind: 'policy',
			retryable: false,
			reason: 'policy or authorization denied the operation',
		};
	}
	if (error instanceof TypeError || error instanceof RangeError) {
		return {
			kind: 'user_correctable',
			retryable: false,
			reason: 'request or configuration can be corrected',
		};
	}
	return {
		kind: 'terminal_system',
		retryable: false,
		reason: 'unclassified terminal system failure',
	};
}

export interface RetryPolicy {
	maxAttempts?: number;
	initialDelayMs?: number;
	multiplier?: number;
	maxDelayMs?: number;
	jitterRatio?: number;
	retryKinds?: FailureKind[];
}

function normalizeRetryPolicy(policy: RetryPolicy = {}): Required<RetryPolicy> {
	const normalized = {
		maxAttempts: policy.maxAttempts ?? 3,
		initialDelayMs: policy.initialDelayMs ?? 100,
		multiplier: policy.multiplier ?? 2,
		maxDelayMs: policy.maxDelayMs ?? 5000,
		jitterRatio: policy.jitterRatio ?? 0,
		retryKinds: policy.retryKinds ?? ['transient_dependency'],
	};
	if (
		!Number.isInteger(normalized.maxAttempts) ||
		normalized.maxAttempts < 1
	) {
		throw new Error('maxAttempts must be at least 1');
	}
	if (normalized.initialDelayMs < 0 || normalized.maxDelayMs < 0) {
		throw new Error('retry delays must be non-negative');
	}
	if (normalized.multiplier < 1) {
		throw new Error('retry multiplier must be at least 1');
	}
	if (normalized.jitterRatio < 0 || normalized.jitterRatio > 1) {
		throw new Error('jitterRatio must be between 0 and 1');
	}
	return normalized;
}

export class RetryExecutor {
	readonly defaultPolicy: Required<RetryPolicy>;
	readonly operationPolicies: Record<string, Required<RetryPolicy>>;

	constructor(
		defaultPolicy: RetryPolicy = {},
		operationPolicies: Record<string, RetryPolicy> = {},
		readonly classifier: (
			error: unknown,
		) => FailureDisposition = classifyFailure,
		readonly budget?: ExecutionBudget,
	) {
		this.defaultPolicy = normalizeRetryPolicy(defaultPolicy);
		this.operationPolicies = Object.fromEntries(
			Object.entries(operationPolicies).map(([name, policy]) => [
				name,
				normalizeRetryPolicy(policy),
			]),
		);
	}

	delayForRetry(
		retryNumber: number,
		policy: Required<RetryPolicy>,
		randomValue = 0.5,
	): number {
		if (!Number.isInteger(retryNumber) || retryNumber < 1) {
			throw new Error('retryNumber must be at least 1');
		}
		const base = Math.min(
			policy.maxDelayMs,
			policy.initialDelayMs * policy.multiplier ** (retryNumber - 1),
		);
		if (policy.jitterRatio === 0) return base;
		const spread = base * policy.jitterRatio;
		return Math.max(0, base - spread + 2 * spread * randomValue);
	}

	async execute<T>(
		operationName: string,
		operation: (attempt: number) => Promise<T>,
		options: {
			sleep?: (ms: number) => Promise<void>;
			randomValue?: () => number;
		} = {},
	): Promise<T> {
		const policy =
			this.operationPolicies[operationName] ?? this.defaultPolicy;
		const sleep =
			options.sleep ??
			((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
		const randomValue = options.randomValue ?? Math.random;
		let attempt = 1;
		while (true) {
			try {
				return await operation(attempt);
			} catch (error) {
				const disposition = this.classifier(error);
				if (
					attempt >= policy.maxAttempts ||
					!policy.retryKinds.includes(disposition.kind) ||
					!disposition.retryable
				) {
					throw error;
				}
				this.budget?.consumeRetry();
				await sleep(this.delayForRetry(attempt, policy, randomValue()));
				attempt += 1;
			}
		}
	}
}

export type RecoveryAction =
	| 'retry'
	| 'repair_arguments'
	| 'alternate_tool'
	| 'switch_model'
	| 'request_user_input'
	| 'escalate';

export interface RecoveryContext {
	attemptsExhausted?: boolean;
	alternateToolAvailable?: boolean;
	modelFallbackAvailable?: boolean;
}

export class RecoveryRouter {
	route(
		failure: FailureDisposition,
		context: RecoveryContext = {},
	): RecoveryAction {
		if (failure.kind === 'transient_dependency') {
			if (!context.attemptsExhausted) return 'retry';
			if (context.alternateToolAvailable) return 'alternate_tool';
			if (context.modelFallbackAvailable) return 'switch_model';
			return 'escalate';
		}
		if (failure.kind === 'model_correctable') return 'repair_arguments';
		if (failure.kind === 'user_correctable') return 'request_user_input';
		if (failure.kind === 'policy') return 'escalate';
		if (context.modelFallbackAvailable) return 'switch_model';
		return 'escalate';
	}
}

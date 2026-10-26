/** Execution budgets, deadlines, loop detection, and finalizers. */
import { type ModelMessage } from '../types.js';

export type TerminationReason =
	| 'completed'
	| 'cancelled'
	| 'stop_requested'
	| 'max_turns'
	| 'max_tool_calls'
	| 'timeout'
	| 'budget_exhausted'
	| 'waiting_for_approval'
	| 'loop_detected'
	| 'model_response_failure';

export class BudgetExceededError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'BudgetExceededError';
	}
}

export interface ExecutionBudgetLimits {
	maxModelCalls?: number;
	maxRetries?: number;
	maxSubagents?: number;
	maxCost?: number;
}

export class ExecutionBudget {
	modelCalls = 0;
	retries = 0;
	subagents = 0;
	cost = 0;

	constructor(readonly limits: ExecutionBudgetLimits = {}) {}

	private consume(
		current: number,
		amount: number,
		limit: number | undefined,
		label: string,
	): number {
		if (amount < 0) throw new Error(label + ' amount must be non-negative');
		const next = current + amount;
		if (limit !== undefined && next > limit) {
			throw new BudgetExceededError(label + ' budget exceeded');
		}
		return next;
	}

	consumeModelCall(amount = 1): void {
		this.modelCalls = this.consume(
			this.modelCalls,
			amount,
			this.limits.maxModelCalls,
			'model call',
		);
	}

	consumeRetry(amount = 1): void {
		this.retries = this.consume(
			this.retries,
			amount,
			this.limits.maxRetries,
			'retry',
		);
	}

	consumeSubagent(amount = 1): void {
		this.subagents = this.consume(
			this.subagents,
			amount,
			this.limits.maxSubagents,
			'subagent',
		);
	}

	consumeCost(amount: number): void {
		this.cost = this.consume(
			this.cost,
			amount,
			this.limits.maxCost,
			'cost',
		);
	}
}

export class Deadline {
	constructor(
		readonly expiresAtMs: number,
		readonly now: () => number = Date.now,
	) {}

	static after(milliseconds: number, now: () => number = Date.now): Deadline {
		if (milliseconds < 0)
			throw new Error('deadline duration must be non-negative');
		return new Deadline(now() + milliseconds, now);
	}

	remainingMs(): number {
		return Math.max(0, this.expiresAtMs - this.now());
	}

	get expired(): boolean {
		return this.remainingMs() <= 0;
	}
}

export function stableStringify(value: unknown): string {
	if (Array.isArray(value))
		return '[' + value.map(stableStringify).join(',') + ']';
	if (typeof value === 'object' && value !== null) {
		return (
			'{' +
			Object.keys(value)
				.sort()
				.map(
					(key) =>
						JSON.stringify(key) +
						':' +
						stableStringify(
							(value as Record<string, unknown>)[key],
						),
				)
				.join(',') +
			'}'
		);
	}
	return JSON.stringify(value) ?? 'null';
}

/** Non-cryptographic 53-bit fingerprint (cyrb53) of canonical JSON arguments. */
export function argumentsFingerprint(argumentsValue: unknown): string {
	const text = stableStringify(argumentsValue);
	let h1 = 0xdeadbeef;
	let h2 = 0x41c6ce57;
	for (let index = 0; index < text.length; index += 1) {
		const code = text.charCodeAt(index);
		h1 = Math.imul(h1 ^ code, 2654435761);
		h2 = Math.imul(h2 ^ code, 1597334677);
	}
	h1 =
		Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^
		Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 =
		Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^
		Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (
		(4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36) +
		'.' +
		text.length
	);
}

export class LoopDetector {
	private lastSignature?: string;
	private repeatCount = 0;

	constructor(readonly repeatThreshold = 3) {
		if (!Number.isInteger(repeatThreshold) || repeatThreshold < 2) {
			throw new Error('repeatThreshold must be at least 2');
		}
	}

	/** Forget the observed trajectory (`AgentLoop` observes a fresh copy per run). */
	reset(): void {
		this.lastSignature = undefined;
		this.repeatCount = 0;
	}

	observe(message: ModelMessage): boolean {
		const signature = JSON.stringify({
			content: message.content.map((part) => ({
				type: part.type,
				text: part.text,
				data: part.data,
			})),
			tools: (message.toolCalls ?? []).map((call) => ({
				name: call.name,
				arguments: call.arguments,
			})),
		});
		if (Object.is(signature, this.lastSignature)) {
			this.repeatCount += 1;
		} else {
			this.lastSignature = signature;
			this.repeatCount = 1;
		}
		return this.repeatCount >= this.repeatThreshold;
	}
}

export type Finalizer = () => void | Promise<void>;

export async function runFinalizers(finalizers: Finalizer[]): Promise<void> {
	let firstError: unknown;
	for (const finalizer of [...finalizers].reverse()) {
		try {
			await finalizer();
		} catch (error) {
			firstError ??= error;
		}
	}
	if (firstError !== undefined) throw firstError;
}

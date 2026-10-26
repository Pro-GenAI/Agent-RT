/** Guardrail results, violations, and blocked-action errors. */

export type GuardrailAction = 'allow' | 'transform' | 'block';

export interface GuardrailResult<T = unknown> {
	action?: GuardrailAction;
	value?: T;
	reason?: string;
	classifications?: string[];
}

export class GuardrailViolationError extends Error {
	constructor(
		message: string,
		readonly classifications: string[] = [],
	) {
		super(message);
		this.name = 'GuardrailViolationError';
	}
}

export class ActionBlockedError extends GuardrailViolationError {
	constructor(message: string, classifications: string[] = []) {
		super(message, classifications);
		this.name = 'ActionBlockedError';
	}
}

export function applyGuardrailResult<T>(
	original: T,
	result: GuardrailResult<T>,
): T {
	const action = result.action ?? 'allow';
	if (action === 'block') {
		throw new GuardrailViolationError(
			result.reason ?? 'guardrail blocked value',
			[...(result.classifications ?? [])],
		);
	}
	if (action === 'transform') return result.value as T;
	if (action !== 'allow')
		throw new Error('unsupported guardrail action: ' + action);
	return original;
}

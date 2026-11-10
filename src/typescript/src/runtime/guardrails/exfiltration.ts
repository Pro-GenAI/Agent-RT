/** Data classification and exfiltration guardrails for tool inputs and outputs. */
import { type ToolCall } from '../types.js';
import { GuardrailViolationError } from './results.js';
import {
	type ToolInputGuardrail,
	type ToolOutputGuardrail,
} from './tool_guardrails.js';
import { policyGlobMatches } from '../authorization/policy.js';

export type DataClassification =
	'public' | 'internal' | 'confidential' | 'restricted';

export interface DataEgressRequest {
	classification: DataClassification;
	channel: string;
	target?: string;
}

export class DataExfiltrationPolicy {
	readonly allowed: Record<DataClassification, string[]>;

	constructor(allowed: Partial<Record<DataClassification, string[]>> = {}) {
		this.allowed = {
			public: ['*'],
			internal: [
				'model',
				'log:internal',
				'tool:internal',
				'network:internal',
			],
			confidential: ['tool:approved', 'network:approved'],
			restricted: [],
			...allowed,
		};
	}

	check(request: DataEgressRequest): void {
		const destination =
			request.target === undefined
				? request.channel
				: `${request.channel}:${request.target}`;
		const patterns = this.allowed[request.classification];
		if (
			patterns.length === 0 ||
			!policyGlobMatches(destination, patterns)
		) {
			throw new GuardrailViolationError(
				`${request.classification} data cannot leave through ${destination}`,
				['data-exfiltration', request.classification],
			);
		}
	}
}

export function makeToolInputExfiltrationGuardrail(
	policy: DataExfiltrationPolicy,
	classify: (call: ToolCall) => DataClassification,
	target: (call: ToolCall) => string = (call) => call.name,
): ToolInputGuardrail {
	return (call) => {
		policy.check({
			classification: classify(call),
			channel: 'tool',
			target: target(call),
		});
		return {};
	};
}

export function makeToolOutputExfiltrationGuardrail(
	policy: DataExfiltrationPolicy,
	classify: (value: unknown) => DataClassification,
	options: { channel?: string; target?: string } = {},
): ToolOutputGuardrail {
	return (value) => {
		policy.check({
			classification: classify(value),
			channel: options.channel ?? 'model',
			target: options.target,
		});
		return {};
	};
}

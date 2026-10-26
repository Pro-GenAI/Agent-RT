/** Tool input/output guardrails, including Agent Action Guard. */
import { type ToolCall } from '../types.js';
import { type ToolDefinition } from '../tools/definitions.js';
import { type GuardrailResult } from './results.js';
import { importOptionalPeer } from '../platform.js';

export type ToolInputGuardrail = (
	call: ToolCall,
	definition: ToolDefinition,
) => GuardrailResult<ToolCall> | Promise<GuardrailResult<ToolCall>>;

export type ToolOutputGuardrail = (
	value: unknown,
	call: ToolCall,
	definition: ToolDefinition,
	requestContext?: Record<string, unknown>,
) => GuardrailResult<unknown> | Promise<GuardrailResult<unknown>>;

export const DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL = 'agent-action-guard';

export type AgentActionGuardClassifier = (
	action: Record<string, unknown>,
) =>
	| Promise<{ label: string | null; confidence: number }>
	| { label: string | null; confidence: number };

export function makeAgentActionGuardToolInputGuardrail(
	classify?: AgentActionGuardClassifier,
): ToolInputGuardrail {
	let classifier = classify;
	return async (call) => {
		if (!classifier) {
			try {
				const module =
					await importOptionalPeer<
						typeof import('agent-action-guard')
					>('agent-action-guard');
				classifier = module.isActionHarmful;
			} catch (error) {
				if (
					typeof error === 'object' &&
					error !== null &&
					'code' in error &&
					(error as { code?: unknown }).code ===
						'ERR_MODULE_NOT_FOUND'
				) {
					throw new Error(
						"Agent Action Guard is not installed. Reinstall 'agent-rt' " +
							'(it is a core dependency) to use the default ' +
							'model-backed tool input guardrail.',
					);
				}
				throw error;
			}
		}
		const { label, confidence } = await classifier({
			type: 'function',
			function: {
				name: call.name,
				arguments: { ...call.arguments },
			},
		});
		if (label) {
			return {
				action: 'block',
				reason: `Agent Action Guard blocked tool input (${confidence.toFixed(3)})`,
				classifications: [
					'tool-input:blocked',
					'agent-action-guard',
					label,
				],
			};
		}
		return { classifications: ['agent-action-guard'] };
	};
}

export function makeDefaultToolInputGuardrail(
	model = DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL,
	classify?: AgentActionGuardClassifier,
): ToolInputGuardrail {
	if (model !== DEFAULT_TOOL_INPUT_GUARDRAIL_MODEL) {
		throw new Error('unsupported tool input guardrail model: ' + model);
	}
	return makeAgentActionGuardToolInputGuardrail(classify);
}

export interface ModelToolInputGuardrailOptions {
	enabled?: boolean;
	model?: string;
	classify?: AgentActionGuardClassifier;
}

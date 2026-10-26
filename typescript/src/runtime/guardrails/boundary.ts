/** Input/output boundary guardrails and the default guardrail set. */
import { type ContentPart, type ModelMessage } from '../types.js';
import { type GuardrailResult } from './results.js';

export type InputGuardrail = (
	messages: ModelMessage[],
) => GuardrailResult<ModelMessage[]>;

export type OutputGuardrail = (
	message: ModelMessage,
) => GuardrailResult<ModelMessage>;

export interface BoundaryGuardrailPolicy {
	maxInputCharacters?: number;
	maxOutputCharacters?: number;
	maxContentParts?: number;
	sanitizeControlCharacters?: boolean;
	requireAssistantOutput?: boolean;
}

export const DEFAULT_BOUNDARY_GUARDRAIL_POLICY: Required<BoundaryGuardrailPolicy> =
	{
		maxInputCharacters: 1_000_000,
		maxOutputCharacters: 1_000_000,
		maxContentParts: 1024,
		sanitizeControlCharacters: true,
		requireAssistantOutput: true,
	};

export function resolveBoundaryGuardrailPolicy(
	policy: BoundaryGuardrailPolicy = {},
): Required<BoundaryGuardrailPolicy> {
	const resolved = { ...DEFAULT_BOUNDARY_GUARDRAIL_POLICY, ...policy };
	if (
		!Number.isInteger(resolved.maxInputCharacters) ||
		resolved.maxInputCharacters < 1
	) {
		throw new Error('maxInputCharacters must be a positive integer');
	}
	if (
		!Number.isInteger(resolved.maxOutputCharacters) ||
		resolved.maxOutputCharacters < 1
	) {
		throw new Error('maxOutputCharacters must be a positive integer');
	}
	if (
		!Number.isInteger(resolved.maxContentParts) ||
		resolved.maxContentParts < 1
	) {
		throw new Error('maxContentParts must be a positive integer');
	}
	return resolved;
}

function guardrailContentSize(parts: ContentPart[]): number {
	let total = 0;
	for (const part of parts) {
		if (part.text !== undefined) total += part.text.length;
		if (part.data !== undefined) {
			try {
				total += JSON.stringify(part.data).length;
			} catch {
				total += String(part.data).length;
			}
		}
	}
	return total;
}

function sanitizeGuardrailText(text: string): string {
	return [...text]
		.filter((character) => {
			const code = character.codePointAt(0) ?? 0;
			return (
				character === '\t' ||
				character === '\n' ||
				character === '\r' ||
				(code >= 32 && code !== 127)
			);
		})
		.join('');
}

export function makeDefaultInputGuardrail(
	policy: BoundaryGuardrailPolicy = {},
): InputGuardrail {
	const resolved = resolveBoundaryGuardrailPolicy(policy);
	return (messages) => {
		const users = messages.filter((message) => message.role === 'user');
		const partCount = users.reduce(
			(sum, message) => sum + message.content.length,
			0,
		);
		if (partCount > resolved.maxContentParts) {
			return {
				action: 'block',
				reason: 'user input contains too many content parts',
				classifications: ['input:content_parts_exceeded'],
			};
		}
		const size = users.reduce(
			(sum, message) => sum + guardrailContentSize(message.content),
			0,
		);
		if (size > resolved.maxInputCharacters) {
			return {
				action: 'block',
				reason: 'user input exceeds the configured character limit',
				classifications: ['input:size_exceeded'],
			};
		}
		if (!resolved.sanitizeControlCharacters) return { action: 'allow' };
		let changed = false;
		const value = messages.map((message) => {
			if (message.role !== 'user') return message;
			const content = message.content.map((part) => {
				if (part.text === undefined) return part;
				const text = sanitizeGuardrailText(part.text);
				if (text !== part.text) changed = true;
				return text === part.text ? part : { ...part, text };
			});
			return { ...message, content };
		});
		return changed
			? {
					action: 'transform',
					value,
					classifications: ['input:control_characters_sanitized'],
				}
			: { action: 'allow' };
	};
}

export function makeDefaultOutputGuardrail(
	policy: BoundaryGuardrailPolicy = {},
): OutputGuardrail {
	const resolved = resolveBoundaryGuardrailPolicy(policy);
	return (message) => {
		if (resolved.requireAssistantOutput && message.role !== 'assistant') {
			return {
				action: 'block',
				reason: 'model output must use the assistant role',
				classifications: ['output:invalid_role'],
			};
		}
		if (message.content.length > resolved.maxContentParts) {
			return {
				action: 'block',
				reason: 'model output contains too many content parts',
				classifications: ['output:content_parts_exceeded'],
			};
		}
		if (
			guardrailContentSize(message.content) > resolved.maxOutputCharacters
		) {
			return {
				action: 'block',
				reason: 'model output exceeds the configured character limit',
				classifications: ['output:size_exceeded'],
			};
		}
		if (!resolved.sanitizeControlCharacters) return { action: 'allow' };
		let changed = false;
		const content = message.content.map((part) => {
			if (part.text === undefined) return part;
			const text = sanitizeGuardrailText(part.text);
			if (text !== part.text) changed = true;
			return text === part.text ? part : { ...part, text };
		});
		return changed
			? {
					action: 'transform',
					value: { ...message, content },
					classifications: ['output:control_characters_sanitized'],
				}
			: { action: 'allow' };
	};
}

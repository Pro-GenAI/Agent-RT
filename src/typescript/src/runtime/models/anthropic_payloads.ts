/** Anthropic request/response payload conversion. */
import { isProviderState } from '../../internal/media.js';
import {
	type ContentPart,
	type ModelMessage,
	type ToolCall,
} from '../types.js';
import {
	type ModelRequest,
	type ModelResponse,
	type ModelUsage,
} from './interfaces.js';
import {
	type ResolvedAnthropicProviderSettings,
	resolveAnthropicModel,
} from './provider_settings.js';
import {
	anthropicContentBlocks,
	hasMedia,
	normalizeFinishReason,
	numberValue,
	providerMessageText,
	providerValue,
	thinkingPart,
} from './conversion.js';

export type AnthropicConvertedMessage = {
	system?: string;
	payload?: Record<string, unknown>;
};

export function anthropicMessage(
	message: ModelMessage,
): AnthropicConvertedMessage {
	const text = providerMessageText(message);
	const media = hasMedia(message);
	if (message.role === 'system') {
		if (media) {
			throw new Error(
				'Anthropic cannot carry media content in system messages',
			);
		}
		return text ? { system: text } : {};
	}
	if (message.role === 'tool') {
		return {
			payload: {
				role: 'user',
				content: [
					{
						type: 'tool_result',
						tool_use_id: message.toolCallId ?? '',
						content: media ? anthropicContentBlocks(message) : text,
					},
				],
			},
		};
	}
	const content: Record<string, unknown>[] = [];
	if (message.role === 'assistant') {
		// Thinking blocks must lead the assistant turn that carries tool_use.
		for (const part of message.content) {
			if (
				isProviderState(part) &&
				typeof part.data === 'object' &&
				part.data !== null
			) {
				content.push({ ...(part.data as Record<string, unknown>) });
			}
		}
	}
	if (media) content.push(...anthropicContentBlocks(message));
	else if (text) content.push({ type: 'text', text });
	if (message.role === 'assistant') {
		for (const call of message.toolCalls ?? []) {
			content.push({
				type: 'tool_use',
				id: call.id,
				name: call.name,
				input: call.arguments,
			});
		}
	}
	return { payload: { role: message.role, content } };
}

export function anthropicParams(
	settings: ResolvedAnthropicProviderSettings,
	request: ModelRequest,
	convertedMessages?: AnthropicConvertedMessage[],
): Record<string, unknown> {
	const systemParts: string[] = [];
	const messages: Record<string, unknown>[] = [];
	const converted =
		convertedMessages ?? request.messages.map(anthropicMessage);
	for (const item of converted) {
		if (item.system) systemParts.push(item.system);
		if (item.payload) messages.push(item.payload);
	}
	const params: Record<string, unknown> = {
		model: resolveAnthropicModel(settings, request.model),
		messages,
		max_tokens: request.maxOutputTokens ?? 4096,
	};
	if (systemParts.length) params.system = systemParts.join('\n\n');
	if (request.tools?.length) {
		params.tools = request.tools.map((tool) => ({
			name: tool.name,
			description: tool.description,
			input_schema: tool.inputSchema,
		}));
	}
	if (request.toolSelection?.required?.length) {
		params.tool_choice =
			request.toolSelection.required.length === 1
				? { type: 'tool', name: request.toolSelection.required[0] }
				: { type: 'any' };
	}
	if (request.temperature !== undefined)
		params.temperature = request.temperature;
	if (request.reasoning?.thinking !== undefined) {
		const thinking = request.reasoning.thinking;
		if (!['adaptive', 'enabled', 'disabled'].includes(thinking)) {
			throw new Error(
				'reasoning thinking must be adaptive, enabled, or disabled',
			);
		}
		const payload: Record<string, unknown> = { type: thinking };
		if (request.reasoning.budgetTokens !== undefined) {
			if (thinking !== 'enabled') {
				throw new Error(
					"reasoning budgetTokens requires thinking='enabled'",
				);
			}
			if (
				!Number.isInteger(request.reasoning.budgetTokens) ||
				request.reasoning.budgetTokens < 1
			) {
				throw new Error(
					'reasoning budgetTokens must be a positive integer',
				);
			}
			payload.budget_tokens = request.reasoning.budgetTokens;
		}
		params.thinking = payload;
	} else if (request.reasoning?.budgetTokens !== undefined) {
		throw new Error("reasoning budgetTokens requires thinking='enabled'");
	}
	const outputConfig: Record<string, unknown> = {};
	if (request.structuredOutput) {
		outputConfig.format = {
			type: 'json_schema',
			schema: request.structuredOutput.schema,
		};
	}
	if (request.reasoning?.effort !== undefined) {
		if (!request.reasoning.effort.trim())
			throw new Error('reasoning effort must not be empty');
		outputConfig.effort = request.reasoning.effort;
	}
	if (Object.keys(outputConfig).length) params.output_config = outputConfig;
	return params;
}

function anthropicUsage(value: unknown): ModelUsage | undefined {
	if (typeof value !== 'object' || value === null) return undefined;
	const inputCount = numberValue(providerValue(value, 'input_tokens'));
	const outputCount = numberValue(providerValue(value, 'output_tokens'));
	return {
		inputTokens: inputCount,
		outputTokens: outputCount,
		totalTokens:
			Number.isFinite(inputCount) && Number.isFinite(outputCount)
				? (inputCount as number) + (outputCount as number)
				: undefined,
	};
}

export function anthropicResponse(value: unknown): ModelResponse {
	const contentBlocks = providerValue(value, 'content');
	const textParts: string[] = [];
	const toolCalls: ToolCall[] = [];
	const thinkingParts: ContentPart[] = [];
	if (Array.isArray(contentBlocks)) {
		for (const block of contentBlocks) {
			const type = providerValue(block, 'type');
			if (type === 'thinking' || type === 'redacted_thinking') {
				thinkingParts.push(thinkingPart(block, type));
			} else if (type === 'text') {
				const text = providerValue(block, 'text');
				if (typeof text === 'string') textParts.push(text);
			} else if (type === 'tool_use') {
				const input = providerValue(block, 'input');
				if (
					typeof input !== 'object' ||
					input === null ||
					Array.isArray(input)
				) {
					throw new Error(
						'Anthropic tool_use input must be an object',
					);
				}
				toolCalls.push({
					id: String(providerValue(block, 'id') ?? ''),
					name: String(providerValue(block, 'name') ?? ''),
					arguments: input as Record<string, unknown>,
				});
			}
		}
	}
	const modelValue = providerValue(value, 'model');
	return {
		message: {
			role: 'assistant',
			content: [
				...thinkingParts,
				...(textParts.length
					? [{ type: 'text' as const, text: textParts.join('') }]
					: []),
			],
			...(toolCalls.length ? { toolCalls } : {}),
		},
		model: typeof modelValue === 'string' ? modelValue : undefined,
		usage: anthropicUsage(providerValue(value, 'usage')),
		finishReason: normalizeFinishReason(
			providerValue(value, 'stop_reason'),
			true,
		),
		raw: value,
	};
}

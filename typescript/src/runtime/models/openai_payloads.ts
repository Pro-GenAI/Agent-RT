/** OpenAI request/response payload conversion and rate-limit parsing. */
import { type ModelMessage, type ToolCall } from '../types.js';
import { type ToolDefinition } from '../tools/definitions.js';
import {
	type ModelRequest,
	type ModelResponse,
	type ModelUsage,
	type StructuredOutputRequirement,
	isJsonObjectOutput,
} from './interfaces.js';
import {
	type ResolvedOpenAIProviderSettings,
	resolveOpenAIModel,
} from './provider_settings.js';
import {
	hasMedia,
	normalizeFinishReason,
	numberValue,
	openAIContentParts,
	providerMessageText,
	providerToolCall,
	providerValue,
} from './conversion.js';

export function openAIMessage(message: ModelMessage): Record<string, unknown> {
	const text = providerMessageText(message);
	if (message.role === 'tool') {
		if (hasMedia(message)) {
			throw new Error(
				'OpenAI chat cannot carry media content in tool messages',
			);
		}
		return {
			role: 'tool',
			content: text,
			tool_call_id: message.toolCallId ?? '',
		};
	}
	const result: Record<string, unknown> = {
		role: message.role,
		content: hasMedia(message) ? openAIContentParts(message) : text,
	};
	if (message.toolCalls?.length) {
		result.tool_calls = message.toolCalls.map((call) => ({
			id: call.id,
			type: 'function',
			function: {
				name: call.name,
				arguments: JSON.stringify(call.arguments),
			},
		}));
	}
	return result;
}

function openAIMessages(messages: ModelMessage[]): Record<string, unknown>[] {
	return messages.map(openAIMessage);
}

export function openAIToolParams(
	tools: ToolDefinition[],
): Record<string, unknown>[] {
	return tools.map((tool) => ({
		type: 'function',
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.inputSchema,
		},
	}));
}

export function openAIResponseFormat(
	requirement: StructuredOutputRequirement,
): Record<string, unknown> {
	if (isJsonObjectOutput(requirement)) return { type: 'json_object' };
	return {
		type: 'json_schema',
		json_schema: {
			name: requirement.name ?? 'response',
			schema: requirement.schema,
			strict: requirement.strict ?? true,
		},
	};
}

export function openAIParams(
	settings: ResolvedOpenAIProviderSettings,
	request: ModelRequest,
	cachedMessages?: Record<string, unknown>[],
	cachedTools?: Record<string, unknown>[],
	cachedResponseFormat?: Record<string, unknown>,
): Record<string, unknown> {
	const params: Record<string, unknown> = {
		model: resolveOpenAIModel(settings, request.model),
		messages: cachedMessages ?? openAIMessages(request.messages),
	};
	if (request.tools?.length) {
		params.tools = cachedTools ?? openAIToolParams(request.tools);
	}
	if (request.toolSelection?.required?.length) {
		params.tool_choice =
			request.toolSelection.required.length === 1
				? {
						type: 'function',
						function: { name: request.toolSelection.required[0] },
					}
				: 'required';
	}
	if (request.temperature !== undefined)
		params.temperature = request.temperature;
	if (request.reasoning?.effort !== undefined) {
		if (!request.reasoning.effort.trim())
			throw new Error('reasoning effort must not be empty');
		params.reasoning_effort = request.reasoning.effort;
	}
	const outputLimit = request.maxOutputTokens;
	if (Number.isFinite(outputLimit))
		params.max_completion_tokens = outputLimit as number;
	if (request.structuredOutput) {
		params.response_format =
			cachedResponseFormat ??
			openAIResponseFormat(request.structuredOutput);
	}
	return params;
}

export function openAIUsage(value: unknown): ModelUsage | undefined {
	if (typeof value !== 'object' || value === null) return undefined;
	const promptDetails = providerValue(value, 'prompt_tokens_details');
	const completionDetails = providerValue(value, 'completion_tokens_details');
	return {
		inputTokens: numberValue(providerValue(value, 'prompt_tokens')),
		outputTokens: numberValue(providerValue(value, 'completion_tokens')),
		totalTokens: numberValue(providerValue(value, 'total_tokens')),
		cachedTokens: numberValue(
			providerValue(promptDetails, 'cached_tokens'),
		),
		reasoningTokens: numberValue(
			providerValue(completionDetails, 'reasoning_tokens'),
		),
	};
}

export function openAIResponse(value: unknown): ModelResponse {
	const choices = providerValue(value, 'choices');
	if (!Array.isArray(choices) || choices.length === 0) {
		throw new Error('OpenAI completion response did not contain a choice');
	}
	const choice = choices[0];
	const message = providerValue(choice, 'message');
	const rawCalls = providerValue(message, 'tool_calls');
	const toolCalls: ToolCall[] = Array.isArray(rawCalls)
		? rawCalls.map((call) => {
				const fn = providerValue(call, 'function');
				return providerToolCall(
					String(providerValue(call, 'id') ?? ''),
					String(providerValue(fn, 'name') ?? ''),
					providerValue(fn, 'arguments'),
				);
			})
		: [];
	const contentValue = providerValue(message, 'content');
	const content = typeof contentValue === 'string' ? contentValue : '';
	return {
		message: {
			role: 'assistant',
			content: content ? [{ type: 'text', text: content }] : [],
			...(toolCalls.length ? { toolCalls } : {}),
		},
		model:
			typeof providerValue(value, 'model') === 'string'
				? (providerValue(value, 'model') as string)
				: undefined,
		usage: openAIUsage(providerValue(value, 'usage')),
		finishReason: normalizeFinishReason(
			providerValue(choice, 'finish_reason'),
		),
		raw: value,
	};
}

export async function ensureOpenAIHTTPResponse(
	response: Response,
): Promise<Response> {
	if (response.ok) return response;
	const body = (await response.text()).slice(0, 1024);
	const error = new Error(
		`OpenAI request failed (${response.status}): ${body || response.statusText}`,
	) as Error & { responseHeaders?: Record<string, string>; status?: number };
	error.responseHeaders = Object.fromEntries(response.headers.entries());
	error.status = response.status;
	throw error;
}

export async function* openAICompatibleSSE(
	response: Response,
): AsyncIterable<unknown> {
	await ensureOpenAIHTTPResponse(response);
	if (!response.body)
		throw new Error('OpenAI streaming response did not include a body');
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	try {
		while (true) {
			const { value, done } = await reader.read();
			buffer += decoder.decode(value ?? new Uint8Array(), {
				stream: !done,
			});
			let boundary;
			while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
				const event = buffer.slice(0, boundary);
				const match = buffer.slice(boundary).match(/^\r?\n\r?\n/);
				buffer = buffer.slice(boundary + (match?.[0].length ?? 2));
				const data = event
					.split(/\r?\n/)
					.filter((line) => line.startsWith('data:'))
					.map((line) => line.slice(5).trim())
					.join('\n');
				if (!data) continue;
				if (data === '[DONE]') return;
				yield JSON.parse(data);
			}
			if (done) break;
		}
		const trailing = buffer.trim();
		if (trailing.startsWith('data:')) {
			const data = trailing.slice(5).trim();
			if (data && data !== '[DONE]') yield JSON.parse(data);
		}
	} finally {
		reader.releaseLock();
	}
}

export function parseOpenAIRateLimitDuration(
	value: string | null | undefined,
): number | undefined {
	if (!value) return undefined;
	const text = value.trim().toLowerCase();
	if (!text) return undefined;
	const numeric = Number(text);
	if (Number.isFinite(numeric)) {
		if (numeric > 1_000_000_000)
			return Math.max(0, numeric * 1000 - Date.now());
		return Math.max(0, numeric * 1000);
	}
	const pattern = /([0-9]*\.?[0-9]+)(ms|s|m|h|d)/g;
	let total = 0;
	let position = 0;
	for (const match of text.matchAll(pattern)) {
		if ((match.index ?? 0) !== position) return undefined;
		const number = Number(match[1]);
		total +=
			number *
			(
				{
					ms: 1,
					s: 1000,
					m: 60_000,
					h: 3_600_000,
					d: 86_400_000,
				} as Record<string, number>
			)[match[2]];
		position = (match.index ?? 0) + match[0].length;
	}
	return position === text.length && position > 0 ? total : undefined;
}

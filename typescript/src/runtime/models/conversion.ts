/** Message, media, and tool-call conversion shared by providers. */
import {
	plainPartText,
	isMediaPart,
	isProviderState,
	mediaReference,
	dataUrl,
	PROVIDER_STATE_MIME,
	parseToolCallArguments,
} from '../../internal/media.js';
import {
	type ContentPart,
	type ModelMessage,
	type ToolCall,
} from '../types.js';
import { type FinishReason } from './interfaces.js';

export function providerValue(value: unknown, key: string): unknown {
	if (typeof value !== 'object' || value === null) return undefined;
	return (value as Record<string, unknown>)[key];
}

export function numberValue(value: unknown): number | undefined {
	return typeof value === 'number' ? value : undefined;
}

export function providerMessageText(message: ModelMessage): string {
	return message.content.map(plainPartText).join('');
}

export function hasMedia(message: ModelMessage): boolean {
	return message.content.some(
		(part) => isMediaPart(part) && !isProviderState(part),
	);
}

export function openAIContentParts(
	message: ModelMessage,
): Record<string, unknown>[] {
	if (message.role !== 'user') {
		throw new Error(
			`OpenAI chat cannot carry media content in ${message.role} messages`,
		);
	}
	const blocks: Record<string, unknown>[] = [];
	for (const part of message.content) {
		if (isProviderState(part)) continue;
		if (!isMediaPart(part)) {
			const text = plainPartText(part);
			if (text) blocks.push({ type: 'text', text });
			continue;
		}
		const { url, base64, mimeType } = mediaReference(part);
		if (part.type === 'image') {
			blocks.push({
				type: 'image_url',
				image_url: {
					url: url ?? dataUrl(base64 ?? '', mimeType, 'image'),
				},
			});
		} else if (part.type === 'audio') {
			if (base64 === undefined) {
				throw new Error(
					'OpenAI chat audio input must be inline base64 data',
				);
			}
			const format = (
				{
					'audio/wav': 'wav',
					'audio/x-wav': 'wav',
					'audio/mpeg': 'mp3',
					'audio/mp3': 'mp3',
				} as Record<string, string>
			)[(mimeType ?? '').toLowerCase()];
			if (!format) {
				throw new Error(
					'OpenAI chat audio input supports audio/wav and audio/mpeg',
				);
			}
			blocks.push({
				type: 'input_audio',
				input_audio: { data: base64, format },
			});
		} else if (['pdf', 'document', 'file'].includes(part.type)) {
			if (base64 === undefined) {
				throw new Error(
					'OpenAI chat file input must be inline base64 data',
				);
			}
			blocks.push({
				type: 'file',
				file: {
					file_data: dataUrl(
						base64,
						mimeType ?? 'application/pdf',
						part.type,
					),
					filename:
						(mimeType ?? 'application/pdf') === 'application/pdf'
							? 'document.pdf'
							: 'document',
				},
			});
		} else {
			throw new Error(
				`OpenAI chat does not support ${part.type} content`,
			);
		}
	}
	return blocks;
}

export function anthropicContentBlocks(
	message: ModelMessage,
): Record<string, unknown>[] {
	if (message.role !== 'user' && message.role !== 'tool') {
		throw new Error(
			`Anthropic cannot carry media content in ${message.role} messages`,
		);
	}
	const blocks: Record<string, unknown>[] = [];
	for (const part of message.content) {
		if (isProviderState(part)) continue;
		if (!isMediaPart(part)) {
			const text = plainPartText(part);
			if (text) blocks.push({ type: 'text', text });
			continue;
		}
		if (part.type === 'audio' || part.type === 'video') {
			throw new Error(`Anthropic does not support ${part.type} content`);
		}
		const { url, base64, mimeType } = mediaReference(part);
		let source: Record<string, unknown>;
		if (url !== undefined) {
			source = { type: 'url', url };
		} else {
			if (!mimeType) {
				throw new Error(
					`${part.type} content part needs a mimeType for inline data`,
				);
			}
			source = { type: 'base64', media_type: mimeType, data: base64 };
		}
		blocks.push({
			type: part.type === 'image' ? 'image' : 'document',
			source,
		});
	}
	return blocks;
}

/** Keep a thinking/redacted_thinking block so it can be replayed verbatim. */
export function thinkingPart(block: unknown, blockType: string): ContentPart {
	const payload: Record<string, unknown> = { type: blockType };
	if (blockType === 'thinking') {
		payload.thinking = String(providerValue(block, 'thinking') ?? '');
		payload.signature = String(providerValue(block, 'signature') ?? '');
	} else {
		payload.data = String(providerValue(block, 'data') ?? '');
	}
	return { type: 'json', data: payload, mimeType: PROVIDER_STATE_MIME };
}

export function providerToolCall(
	id: string,
	name: string,
	raw: unknown,
): ToolCall {
	const parsed = parseToolCallArguments(raw);
	return parsed.argumentError === undefined
		? { id, name, arguments: parsed.arguments }
		: {
				id,
				name,
				arguments: parsed.arguments,
				argumentError: parsed.argumentError,
			};
}

export function normalizeFinishReason(
	value: unknown,
	anthropic = false,
): FinishReason | undefined {
	if (typeof value !== 'string') return undefined;
	if (anthropic) {
		return (
			(
				{
					end_turn: 'stop',
					stop_sequence: 'stop',
					tool_use: 'tool_calls',
					max_tokens: 'length',
				} as Record<string, FinishReason>
			)[value] ?? 'other'
		);
	}
	return (
		(
			{
				stop: 'stop',
				tool_calls: 'tool_calls',
				length: 'length',
				content_filter: 'content_filter',
			} as Record<string, FinishReason>
		)[value] ?? 'other'
	);
}

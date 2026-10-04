/**
 * Shared content-part helpers for provider adapters: media (image/audio/file)
 * conversion inputs and provider-private state that must round-trip verbatim.
 */

export interface MediaPartLike {
	type: string;
	text?: string;
	data?: unknown;
	mimeType?: string;
}

/**
 * Provider-private state carried through the transcript, e.g. Anthropic
 * thinking blocks that must be replayed verbatim on the next tool turn. Parts
 * marked with this MIME type are never rendered as text or treated as
 * structured output, and other providers drop them.
 */
export const PROVIDER_STATE_MIME = 'application/vnd.anthropic.thinking+json';

export const MEDIA_PART_TYPES: ReadonlySet<string> = new Set([
	'image',
	'audio',
	'video',
	'pdf',
	'document',
	'file',
]);

export function isProviderState(part: MediaPartLike): boolean {
	return part.mimeType === PROVIDER_STATE_MIME;
}

export function isMediaPart(part: MediaPartLike): boolean {
	return MEDIA_PART_TYPES.has(part.type);
}

export interface MediaReference {
	url?: string;
	base64?: string;
	mimeType?: string;
}

function toBase64(bytes: Uint8Array): string {
	let binary = '';
	for (let index = 0; index < bytes.length; index += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
	}
	return btoa(binary);
}

/**
 * Resolve an image/audio/pdf/file part. `data` may be an http(s) URL, a
 * `data:` URL, a base64 string, bytes, or an object with `url`/`data`.
 */
export function mediaReference(part: MediaPartLike): MediaReference {
	let data = part.data;
	if (
		typeof data === 'object' &&
		data !== null &&
		!(data instanceof Uint8Array)
	) {
		const record = data as Record<string, unknown>;
		data = record.url ?? record.data;
	}
	if (data instanceof Uint8Array) {
		return { base64: toBase64(data), mimeType: part.mimeType };
	}
	if (typeof data === 'string' && data) {
		if (/^https?:\/\//i.test(data))
			return { url: data, mimeType: part.mimeType };
		if (data.startsWith('data:')) {
			const comma = data.indexOf(',');
			const header = comma < 0 ? data.slice(5) : data.slice(5, comma);
			return {
				base64: comma < 0 ? '' : data.slice(comma + 1),
				mimeType: header.split(';')[0] || part.mimeType,
			};
		}
		return { base64: data, mimeType: part.mimeType };
	}
	throw new Error(`${part.type} content part has no usable data`);
}

export function dataUrl(
	base64: string,
	mimeType: string | undefined,
	partType: string,
): string {
	if (!mimeType) {
		throw new Error(
			`${partType} content part needs a mimeType for inline data`,
		);
	}
	return `data:${mimeType};base64,${base64}`;
}

/** Text of the non-media, non-provider-state parts (JSON is serialised). */
export function plainPartText(part: MediaPartLike): string {
	if (isProviderState(part) || isMediaPart(part)) return '';
	if (part.text !== undefined) return part.text;
	if (part.data !== undefined) return JSON.stringify(part.data);
	return '';
}

/** Responses-API content list for a user message that carries media parts. */
export function responsesInputContent(
	role: string,
	parts: readonly MediaPartLike[],
): Record<string, unknown>[] {
	if (role !== 'user') {
		throw new Error(
			'Responses input can only carry media in user messages',
		);
	}
	const blocks: Record<string, unknown>[] = [];
	for (const part of parts) {
		if (isProviderState(part)) continue;
		if (!isMediaPart(part)) {
			const text = plainPartText(part);
			if (text) blocks.push({ type: 'input_text', text });
			continue;
		}
		const { url, base64, mimeType } = mediaReference(part);
		if (part.type === 'image') {
			blocks.push({
				type: 'input_image',
				image_url: url ?? dataUrl(base64 ?? '', mimeType, 'image'),
			});
		} else if (['pdf', 'document', 'file'].includes(part.type)) {
			blocks.push(
				url !== undefined
					? { type: 'input_file', file_url: url }
					: {
							type: 'input_file',
							file_data: dataUrl(
								base64 ?? '',
								mimeType ?? 'application/pdf',
								part.type,
							),
							filename: 'document.pdf',
						},
			);
		} else {
			throw new Error(
				`OpenAI Responses input does not support ${part.type} content`,
			);
		}
	}
	return blocks;
}

/**
 * Parse model-produced tool arguments without throwing on malformed JSON.
 *
 * Models routinely emit truncated or invalid argument strings (for example
 * when a reply is cut off at the output-token limit). That is a
 * model-correctable failure, so it is reported as `argumentError` for the loop
 * to hand back to the model instead of aborting the whole run.
 */
export function parseToolCallArguments(value: unknown): {
	arguments: Record<string, unknown>;
	argumentError?: string;
} {
	if (typeof value !== 'string' || !value.trim()) return { arguments: {} };
	let parsed: unknown;
	try {
		parsed = JSON.parse(value);
	} catch (error) {
		return {
			arguments: {},
			argumentError:
				'tool call arguments are not a valid JSON object: ' +
				(error instanceof Error ? error.message : String(error)),
		};
	}
	if (
		typeof parsed !== 'object' ||
		parsed === null ||
		Array.isArray(parsed)
	) {
		return {
			arguments: {},
			argumentError: 'tool call arguments must decode to a JSON object',
		};
	}
	return { arguments: parsed as Record<string, unknown> };
}

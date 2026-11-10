/** Lightweight OpenAI-compatible HTTP client and rate-limit gate. */
import { type BatchJob, type ModelCatalogEntry } from './interfaces.js';
import { type OpenAICompletionClient } from './sdk.js';
import { type ResolvedOpenAIProviderSettings } from './provider_settings.js';
import { providerValue } from './conversion.js';
import {
	ensureOpenAIHTTPResponse,
	openAICompatibleSSE,
	parseOpenAIRateLimitDuration,
} from './openai_payloads.js';

const MAX_TIMER_DELAY_MS = 2 ** 31 - 1;

export const MAX_RATE_LIMIT_BLOCK_MS = 300_000;

export class OpenAIRateLimitGate {
	private readonly blockedUntil = new Map<string, number>();
	constructor(readonly now: () => number = Date.now) {}

	retryAfterMs(model: string): number {
		const remaining = (this.blockedUntil.get(model) ?? 0) - this.now();
		if (remaining <= 0) {
			this.blockedUntil.delete(model);
			return 0;
		}
		return remaining;
	}

	async wait(model: string, signal?: AbortSignal): Promise<void> {
		while (true) {
			const remaining = this.retryAfterMs(model);
			if (remaining <= 0) return;
			// setTimeout clamps delays above 2^31-1 ms to 1 ms, which would turn
			// a very large server-supplied Retry-After into a busy loop; wait in
			// bounded slices and re-check instead.
			const slice = Math.min(remaining, MAX_TIMER_DELAY_MS);
			await new Promise<void>((resolve, reject) => {
				const timer = setTimeout(() => {
					signal?.removeEventListener('abort', onAbort);
					resolve();
				}, slice);
				function onAbort() {
					clearTimeout(timer);
					reject(signal?.reason ?? new Error('aborted'));
				}
				if (!signal) return;
				if (signal.aborted) onAbort();
				else signal.addEventListener('abort', onAbort, { once: true });
			});
		}
	}

	update(model: string, headers: Record<string, string>): number {
		const normalized = Object.fromEntries(
			Object.entries(headers).map(([key, value]) => [
				key.toLowerCase(),
				String(value),
			]),
		);
		const waits: number[] = [];
		const requestsRemaining = Number(
			normalized['x-ratelimit-remaining-requests'],
		);
		const tokensRemaining = Number(
			normalized['x-ratelimit-remaining-tokens'],
		);
		if (Number.isFinite(requestsRemaining) && requestsRemaining <= 0) {
			const reset = parseOpenAIRateLimitDuration(
				normalized['x-ratelimit-reset-requests'],
			);
			if (reset !== undefined) waits.push(reset);
		}
		if (Number.isFinite(tokensRemaining) && tokensRemaining <= 0) {
			const reset = parseOpenAIRateLimitDuration(
				normalized['x-ratelimit-reset-tokens'],
			);
			if (reset !== undefined) waits.push(reset);
		}
		const retryAfter = parseOpenAIRateLimitDuration(
			normalized['retry-after'],
		);
		if (retryAfter !== undefined) waits.push(retryAfter);
		if (!waits.length) return this.retryAfterMs(model);
		// The wait comes from the remote server; bound it so a hostile or buggy
		// Retry-After cannot block a model for the process lifetime.
		const until =
			this.now() + Math.min(Math.max(...waits), MAX_RATE_LIMIT_BLOCK_MS);
		this.blockedUntil.set(
			model,
			Math.max(this.blockedUntil.get(model) ?? 0, until),
		);
		return this.retryAfterMs(model);
	}
}

const OPENAI_HEADERS_KEY = '__agentRtResponseHeaders';

export function openAIResponseHeaders(value: unknown): Record<string, string> {
	if (typeof value !== 'object' || value === null) return {};
	const raw = (value as Record<string, unknown>)[OPENAI_HEADERS_KEY];
	return typeof raw === 'object' && raw !== null
		? (raw as Record<string, string>)
		: {};
}

export function modelCatalogEntry(
	value: unknown,
	anthropic = false,
): ModelCatalogEntry {
	const id = providerValue(value, 'id');
	if (typeof id !== 'string' || !id) {
		throw new Error('model catalog entry did not contain an id');
	}
	const created = providerValue(value, anthropic ? 'created_at' : 'created');
	const ownedBy = providerValue(value, 'owned_by');
	const displayName = providerValue(value, 'display_name');
	return {
		id,
		...(typeof created === 'number' || typeof created === 'string'
			? { created }
			: {}),
		...(typeof ownedBy === 'string' ? { ownedBy } : {}),
		...(typeof displayName === 'string' ? { displayName } : {}),
		raw: value,
	};
}

export function modelCatalogData(value: unknown): unknown[] {
	const data = providerValue(value, 'data');
	if (Array.isArray(data)) return data;
	if (Array.isArray(value)) return value;
	throw new Error('model list response did not contain data');
}

export function batchJob(value: unknown): BatchJob {
	const id = providerValue(value, 'id');
	if (typeof id !== 'string' || !id) {
		throw new Error('batch response did not contain an id');
	}
	const status =
		providerValue(value, 'status') ??
		providerValue(value, 'processing_status');
	return {
		id,
		...(typeof status === 'string' ? { status } : {}),
		raw: value,
	};
}

export function batchData(value: unknown): unknown[] {
	const data = providerValue(value, 'data');
	if (Array.isArray(data)) return data;
	if (Array.isArray(value)) return value;
	throw new Error('batch list response did not contain data');
}

export function openAICompatibleFetchClient(
	settings: ResolvedOpenAIProviderSettings,
): OpenAICompletionClient {
	const fetch = settings.fetch ?? globalThis.fetch;
	const baseUrl = settings.baseUrl.replace(/\/$/, '');
	const url = `${baseUrl}/chat/completions`;
	const embeddingUrl = `${baseUrl}/embeddings`;
	const modelsUrl = `${baseUrl}/models`;
	const batchesUrl = `${baseUrl}/batches`;
	const headers: Record<string, string> = {
		'content-type': 'application/json',
	};
	if (settings.apiKey) headers.authorization = `Bearer ${settings.apiKey}`;
	return {
		chat: {
			completions: {
				create: async (params, options) => {
					const response = await fetch(url, {
						method: 'POST',
						headers,
						body: JSON.stringify(params),
						signal: options?.signal,
					});
					const responseHeaders = Object.fromEntries(
						response.headers.entries(),
					);
					if (params.stream === true) {
						const stream = openAICompatibleSSE(
							response,
						) as AsyncIterable<unknown> & Record<string, unknown>;
						stream[OPENAI_HEADERS_KEY] = responseHeaders;
						return stream;
					}
					await ensureOpenAIHTTPResponse(response);
					const data = await response.json();
					if (typeof data === 'object' && data !== null) {
						(data as Record<string, unknown>)[OPENAI_HEADERS_KEY] =
							responseHeaders;
					}
					return data;
				},
			},
		},
		embeddings: {
			create: async (params) => {
				const response = await fetch(embeddingUrl, {
					method: 'POST',
					headers,
					body: JSON.stringify(params),
				});
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
		},
		models: {
			list: async () => {
				const response = await fetch(modelsUrl, { headers });
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
			retrieve: async (model) => {
				const response = await fetch(
					`${modelsUrl}/${encodeURIComponent(model)}`,
					{ headers },
				);
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
		},
		batches: {
			create: async (params) => {
				const response = await fetch(batchesUrl, {
					method: 'POST',
					headers,
					body: JSON.stringify(params),
				});
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
			retrieve: async (id) => {
				const response = await fetch(
					`${batchesUrl}/${encodeURIComponent(id)}`,
					{ headers },
				);
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
			list: async (params = {}) => {
				const query = new URLSearchParams();
				if (typeof params.after === 'string')
					query.set('after', params.after);
				if (typeof params.limit === 'number')
					query.set('limit', String(params.limit));
				const suffix = query.size ? `?${query.toString()}` : '';
				const response = await fetch(`${batchesUrl}${suffix}`, {
					headers,
				});
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
			cancel: async (id) => {
				const response = await fetch(
					`${batchesUrl}/${encodeURIComponent(id)}/cancel`,
					{
						method: 'POST',
						headers,
					},
				);
				await ensureOpenAIHTTPResponse(response);
				return response.json();
			},
		},
	};
}

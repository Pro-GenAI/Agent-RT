/** The environment-configured web search provider. */
import { currentProviderEnvironment } from '../models/sdk.js';
import {
	type RetrievalProvider,
	type RetrievalQuery,
	type RetrievalResult,
} from './registry.js';

export const AGENT_RT_WEB_SEARCH_TOOL_ENV = 'AGENT_RT_WEB_SEARCH_TOOL';

export const AGENT_RT_WEB_SEARCH_API_KEY_ENV = 'AGENT_RT_WEB_SEARCH_API_KEY';

export const AGENT_RT_WEB_SEARCH_TOKEN_ENV = 'AGENT_RT_WEB_SEARCH_TOKEN';

export const AGENT_RT_WEB_SEARCH_TIMEOUT_ENV =
	'AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS';

const WEB_SEARCH_CREDENTIAL_ENV: Readonly<Record<string, string>> = {
	tavily: 'TAVILY_API_KEY',
	brave: 'BRAVE_SEARCH_API_KEY',
	serper: 'SERPER_API_KEY',
};

type WebSearchFetch = (
	input: string | URL | Request,
	init?: RequestInit,
) => Promise<Response>;

function webSearchEnvironment(
	environment?: Readonly<Record<string, string | undefined>>,
): Readonly<Record<string, string | undefined>> {
	return environment ?? currentProviderEnvironment();
}

function webSearchQueryParams(
	values: Record<string, unknown>,
): URLSearchParams {
	const query = new URLSearchParams();
	for (const [key, value] of Object.entries(values)) {
		if (value === undefined || value === null) continue;
		if (Array.isArray(value)) {
			for (const item of value) query.append(key, String(item));
		} else if (typeof value === 'object') {
			query.set(key, JSON.stringify(value));
		} else {
			query.set(key, String(value));
		}
	}
	return query;
}

export class EnvironmentWebSearchProvider implements RetrievalProvider {
	readonly kind = 'web' as const;
	readonly tool: 'tavily' | 'brave' | 'serper';
	readonly timeoutSeconds: number;
	private readonly credential: string;
	private readonly fetchImpl: WebSearchFetch;

	constructor(
		tool: string,
		credential: string,
		options: { timeoutSeconds?: number; fetchImpl?: WebSearchFetch } = {},
	) {
		const normalized = tool.trim().toLowerCase();
		if (!(normalized in WEB_SEARCH_CREDENTIAL_ENV)) {
			throw new Error(
				`unsupported web search tool ${JSON.stringify(tool)}; expected one of: brave, serper, tavily`,
			);
		}
		if (!credential.trim())
			throw new Error('web search credential must not be empty');
		const timeoutSeconds = options.timeoutSeconds ?? 20;
		if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) {
			throw new Error('web search timeout must be positive');
		}
		this.tool = normalized as 'tavily' | 'brave' | 'serper';
		this.credential = credential;
		this.timeoutSeconds = timeoutSeconds;
		this.fetchImpl = options.fetchImpl ?? fetch;
	}

	static fromEnvironment(
		environment?: Readonly<Record<string, string | undefined>>,
		options: { fetchImpl?: WebSearchFetch } = {},
	): EnvironmentWebSearchProvider {
		const env = webSearchEnvironment(environment);
		const tool = (env[AGENT_RT_WEB_SEARCH_TOOL_ENV] ?? '')
			.trim()
			.toLowerCase();
		if (!tool) {
			throw new Error(
				`web search requires ${AGENT_RT_WEB_SEARCH_TOOL_ENV}=tavily|brave|serper`,
			);
		}
		const credentialEnv = WEB_SEARCH_CREDENTIAL_ENV[tool];
		if (!credentialEnv) {
			throw new Error(
				`unsupported web search tool ${JSON.stringify(tool)}; expected one of: brave, serper, tavily`,
			);
		}
		const credential =
			env[credentialEnv] ??
			env[AGENT_RT_WEB_SEARCH_API_KEY_ENV] ??
			env[AGENT_RT_WEB_SEARCH_TOKEN_ENV];
		if (!credential?.trim()) {
			throw new Error(
				`web search credential is missing; set ${credentialEnv}, ${AGENT_RT_WEB_SEARCH_API_KEY_ENV}, or ${AGENT_RT_WEB_SEARCH_TOKEN_ENV}`,
			);
		}
		const rawTimeout = env[AGENT_RT_WEB_SEARCH_TIMEOUT_ENV] ?? '20';
		const timeoutSeconds = Number(rawTimeout);
		if (!Number.isFinite(timeoutSeconds)) {
			throw new Error(
				`${AGENT_RT_WEB_SEARCH_TIMEOUT_ENV} must be numeric`,
			);
		}
		return new EnvironmentWebSearchProvider(tool, credential, {
			timeoutSeconds,
			fetchImpl: options.fetchImpl,
		});
	}

	async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
		const limit = query.limit ?? 10;
		if (!Number.isInteger(limit) || limit < 1) {
			throw new Error('retrieval limit must be at least 1');
		}
		if (!query.text.trim())
			throw new Error('web search query must not be empty');
		const filters = { ...(query.filters ?? {}) };
		if (this.tool === 'tavily') {
			const payload = await this.request(
				'https://api.tavily.com/search',
				{
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({
						...filters,
						api_key: this.credential,
						query: query.text,
						max_results: limit,
					}),
				},
			);
			const items = Array.isArray(payload.results) ? payload.results : [];
			return items.slice(0, limit).flatMap((value, index) => {
				if (!value || typeof value !== 'object' || Array.isArray(value))
					return [];
				const item = value as Record<string, unknown>;
				const url = typeof item.url === 'string' ? item.url : undefined;
				const score =
					typeof item.score === 'number' ? item.score : undefined;
				return [
					{
						id: url ?? String(index),
						title: String(item.title ?? url ?? ''),
						content: String(item.content ?? item.snippet ?? ''),
						...(score !== undefined ? { score } : {}),
						...(url ? { uri: url } : {}),
						metadata: { provider: 'tavily' },
					},
				];
			});
		}
		if (this.tool === 'brave') {
			const params = webSearchQueryParams({
				...filters,
				q: query.text,
				count: limit,
			});
			const payload = await this.request(
				`https://api.search.brave.com/res/v1/web/search?${params.toString()}`,
				{ headers: { 'X-Subscription-Token': this.credential } },
			);
			const web = payload.web;
			const items =
				web && typeof web === 'object' && !Array.isArray(web)
					? (web as Record<string, unknown>).results
					: undefined;
			return (Array.isArray(items) ? items : [])
				.slice(0, limit)
				.flatMap((value, index) => {
					if (
						!value ||
						typeof value !== 'object' ||
						Array.isArray(value)
					)
						return [];
					const item = value as Record<string, unknown>;
					const url =
						typeof item.url === 'string' ? item.url : undefined;
					return [
						{
							id: url ?? String(index),
							title: String(item.title ?? url ?? ''),
							content: String(item.description ?? ''),
							...(url ? { uri: url } : {}),
							metadata: { provider: 'brave' },
						},
					];
				});
		}
		const payload = await this.request('https://google.serper.dev/search', {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				'X-API-KEY': this.credential,
			},
			body: JSON.stringify({ ...filters, q: query.text, num: limit }),
		});
		const items = Array.isArray(payload.organic) ? payload.organic : [];
		return items.slice(0, limit).flatMap((value, index) => {
			if (!value || typeof value !== 'object' || Array.isArray(value))
				return [];
			const item = value as Record<string, unknown>;
			const url = typeof item.link === 'string' ? item.link : undefined;
			return [
				{
					id: url ?? String(index),
					title: String(item.title ?? url ?? ''),
					content: String(item.snippet ?? ''),
					...(url ? { uri: url } : {}),
					metadata: {
						provider: 'serper',
						...(item.position !== undefined
							? { position: item.position }
							: {}),
					},
				},
			];
		});
	}

	private async request(
		url: string,
		init: RequestInit,
	): Promise<Record<string, unknown>> {
		const controller = new AbortController();
		const timer = setTimeout(
			() => controller.abort(new Error('web search request timed out')),
			this.timeoutSeconds * 1000,
		);
		try {
			const response = await this.fetchImpl(url, {
				...init,
				signal: controller.signal,
			});
			if (!response.ok) {
				throw new Error(
					`web search provider returned HTTP ${response.status}`,
				);
			}
			const payload: unknown = await response.json();
			if (
				!payload ||
				typeof payload !== 'object' ||
				Array.isArray(payload)
			) {
				throw new Error(
					'web search provider returned a non-object JSON response',
				);
			}
			return payload as Record<string, unknown>;
		} finally {
			clearTimeout(timer);
		}
	}
}

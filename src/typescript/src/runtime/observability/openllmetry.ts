/** OpenLLMetry (Traceloop) configuration from runtime settings. */
import { nodeModule } from '../platform.js';

export interface OpenLLMetryConfig {
	apiKey?: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	traceContent?: boolean;
	enrichTokens?: boolean;
	appName: string;
}

export function parseEnvBoolean(
	value: string | undefined,
): boolean | undefined {
	if (value === undefined) return undefined;
	const normalized = value.trim().toLowerCase();
	if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
	if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
	throw new Error('invalid boolean environment value');
}

function parseTraceloopHeaders(
	value: string | undefined,
): Record<string, string> {
	if (!value?.trim()) return {};
	const headers: Record<string, string> = {};
	for (const item of value.split(',')) {
		const separator = item.indexOf('=');
		if (separator <= 0)
			throw new Error('TRACELOOP_HEADERS entries must be key=value');
		const key = item.slice(0, separator).trim();
		const headerValue = item.slice(separator + 1).trim();
		if (!key || !headerValue)
			throw new Error(
				'TRACELOOP_HEADERS entries must have non-empty key and value',
			);
		headers[key] = headerValue;
	}
	return headers;
}

function validTraceloopBaseUrl(value: string): boolean {
	const candidate = value.trim();
	if (!candidate || /\s/.test(candidate)) return false;
	if (candidate.startsWith('http://') || candidate.startsWith('https://')) {
		try {
			const url = new URL(candidate);
			return (
				(url.protocol === 'http:' || url.protocol === 'https:') &&
				Boolean(url.host)
			);
		} catch {
			return false;
		}
	}
	return /^[A-Za-z0-9._-]+(?::[0-9]{1,5})?$/.test(candidate);
}

export function openLLMetryConfigFromEnv(
	env: Record<string, string | undefined>,
): OpenLLMetryConfig | undefined {
	const configuredCredential = env.TRACELOOP_API_KEY;
	const baseUrlRaw = env.TRACELOOP_BASE_URL;
	const credential = configuredCredential?.trim();
	const baseUrl = baseUrlRaw?.trim();
	if (typeof configuredCredential === 'string' && !credential)
		return undefined;
	if (baseUrl !== undefined && !validTraceloopBaseUrl(baseUrl))
		return undefined;
	try {
		const headers = parseTraceloopHeaders(env.TRACELOOP_HEADERS);
		if (!credential && !baseUrl && Object.keys(headers).length === 0)
			return undefined;
		const appName = (env.TRACELOOP_APP_NAME ?? 'agent-rt').trim();
		if (!appName) return undefined;
		return {
			...(credential ? { apiKey: credential } : {}),
			...(baseUrl ? { baseUrl } : {}),
			headers,
			traceContent: parseEnvBoolean(env.TRACELOOP_TRACE_CONTENT),
			enrichTokens: parseEnvBoolean(env.TRACELOOP_ENRICH_TOKENS),
			appName,
		};
	} catch {
		return undefined;
	}
}

function loadOpenLLMetryInitialize():
	((options: Record<string, unknown>) => unknown) | undefined {
	try {
		const sdk = nodeModule('@traceloop/node-server-sdk') as {
			initialize?: (options: Record<string, unknown>) => unknown;
		};
		return typeof sdk.initialize === 'function'
			? sdk.initialize
			: undefined;
	} catch {
		return undefined;
	}
}

export function initializeOpenLLMetryFromEnv(
	env: Record<string, string | undefined>,
	initialize?: (options: Record<string, unknown>) => unknown,
): boolean {
	const config = openLLMetryConfigFromEnv(env);
	if (!config) return false;
	const resolvedInitialize = initialize ?? loadOpenLLMetryInitialize();
	if (!resolvedInitialize) return false;
	const options: Record<string, unknown> = { appName: config.appName };
	if (config.apiKey) options.apiKey = config.apiKey;
	if (config.baseUrl) options.baseUrl = config.baseUrl;
	if (config.headers && Object.keys(config.headers).length)
		options.headers = config.headers;
	if (config.traceContent !== undefined)
		options.traceContent = config.traceContent;
	// TRACELOOP_ENRICH_TOKENS is consumed directly by OpenLLMetry in Node.
	try {
		resolvedInitialize(options);
	} catch {
		return false;
	}
	return true;
}

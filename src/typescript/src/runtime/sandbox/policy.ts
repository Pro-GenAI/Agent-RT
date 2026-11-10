/** Sandbox resource limits and network policy. */
import { executionLimitsFromEnv } from '../tools/definitions.js';
import { runtimeEnvironment } from '../platform.js';

export interface SandboxResourceLimits {
	cpuSeconds?: number;
	memoryBytes?: number;
	diskBytes?: number;
	processCount?: number;
	timeoutMs?: number;
	outputBytes?: number;
}

export function sandboxResourceLimitsFromEnv(
	env: Record<string, string | undefined> = runtimeEnvironment(),
): SandboxResourceLimits {
	const defaults = executionLimitsFromEnv(env);
	return {
		cpuSeconds: defaults.cpuSeconds,
		memoryBytes: defaults.memoryBytes,
		timeoutMs: defaults.timeoutMs,
	};
}

export type SandboxNetworkMode = 'none' | 'allowlist' | 'unrestricted';

export const SANDBOX_NETWORK_MODE_ENV = 'AGENT_RT_SANDBOX_NETWORK_MODE';

export const SANDBOX_ALLOWED_DOMAINS_ENV = 'AGENT_RT_SANDBOX_ALLOWED_DOMAINS';

export const SANDBOX_BLOCKED_DOMAINS_ENV = 'AGENT_RT_SANDBOX_BLOCKED_DOMAINS';

export const SANDBOX_ALLOW_HTTP_ENV = 'AGENT_RT_SANDBOX_ALLOW_HTTP';

export const SANDBOX_ALLOW_WEBSOCKET_ENV = 'AGENT_RT_SANDBOX_ALLOW_WEBSOCKET';

export const SANDBOX_ALLOW_IP_ENV = 'AGENT_RT_SANDBOX_ALLOW_IP';

export const SANDBOX_ALLOW_PROXY_ENV = 'AGENT_RT_SANDBOX_ALLOW_PROXY';

export const SANDBOX_NETWORK_BPS_ENV =
	'AGENT_RT_SANDBOX_NETWORK_MAX_BYTES_PER_SECOND';

export const SANDBOX_NETWORK_BYTES_ENV =
	'AGENT_RT_SANDBOX_NETWORK_MAX_TRANSFER_BYTES';

const SANDBOX_PROXY_ENV_KEYS = new Set([
	'HTTP_PROXY',
	'HTTPS_PROXY',
	'ALL_PROXY',
	'NO_PROXY',
	'http_proxy',
	'https_proxy',
	'all_proxy',
	'no_proxy',
]);

export interface SandboxNetworkPolicy {
	mode?: SandboxNetworkMode;
	allowedDomains?: string[];
	blockedDomains?: string[];
	proxyUrl?: string;
	allowHttp?: boolean;
	allowWebSocket?: boolean;
	allowIpAddresses?: boolean;
	allowProxy?: boolean;
	maxBytesPerSecond?: number;
	maxTransferBytes?: number;
}

export function sandboxEnvBoolean(
	env: Record<string, string | undefined>,
	name: string,
	defaultValue = false,
): boolean {
	const raw = env[name];
	if (raw === undefined) return defaultValue;
	const normalized = raw.trim().toLowerCase();
	if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
	if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
	throw new Error(name + ' must be a boolean');
}

function sandboxEnvNonnegativeInteger(
	env: Record<string, string | undefined>,
	name: string,
): number | undefined {
	const raw = env[name];
	if (raw === undefined) return undefined;
	if (!raw.trim()) throw new Error(name + ' must be a non-negative integer');
	const value = Number(raw);
	if (!Number.isInteger(value) || value < 0) {
		throw new Error(name + ' must be a non-negative integer');
	}
	return value;
}

function sandboxDomainList(raw: string | undefined): string[] {
	if (!raw?.trim()) return [];
	return raw
		.split(',')
		.map((item) => item.trim())
		.filter(Boolean);
}

function isIpAddress(value: string): boolean {
	if (/^\d{1,3}(?:\.\d{1,3}){3}$/.test(value)) return true;
	return value.includes(':') && /^[0-9a-f:]+$/i.test(value);
}

export function sandboxNetworkPolicyFromEnv(
	env: Record<string, string | undefined> = runtimeEnvironment(),
): SandboxNetworkPolicy {
	const mode = (env[SANDBOX_NETWORK_MODE_ENV] ?? 'none').trim().toLowerCase();
	if (mode !== 'none' && mode !== 'allowlist' && mode !== 'unrestricted') {
		throw new Error(
			SANDBOX_NETWORK_MODE_ENV +
				' must be none, allowlist, or unrestricted',
		);
	}
	return {
		mode,
		allowedDomains: sandboxDomainList(env[SANDBOX_ALLOWED_DOMAINS_ENV]),
		blockedDomains: sandboxDomainList(env[SANDBOX_BLOCKED_DOMAINS_ENV]),
		allowHttp: sandboxEnvBoolean(env, SANDBOX_ALLOW_HTTP_ENV),
		allowWebSocket: sandboxEnvBoolean(env, SANDBOX_ALLOW_WEBSOCKET_ENV),
		allowIpAddresses: sandboxEnvBoolean(env, SANDBOX_ALLOW_IP_ENV),
		allowProxy: sandboxEnvBoolean(env, SANDBOX_ALLOW_PROXY_ENV),
		maxBytesPerSecond: sandboxEnvNonnegativeInteger(
			env,
			SANDBOX_NETWORK_BPS_ENV,
		),
		maxTransferBytes: sandboxEnvNonnegativeInteger(
			env,
			SANDBOX_NETWORK_BYTES_ENV,
		),
	};
}

export function normalizeNetworkPolicy(policy: SandboxNetworkPolicy = {}): Omit<
	Required<SandboxNetworkPolicy>,
	'proxyUrl' | 'maxBytesPerSecond' | 'maxTransferBytes'
> & {
	proxyUrl?: string;
	maxBytesPerSecond?: number;
	maxTransferBytes?: number;
} {
	const allowedDomains = (policy.allowedDomains ?? [])
		.map((domain) => domain.toLowerCase().replace(/^\.+|\.+$/g, ''))
		.filter(Boolean);
	const blockedDomains = (policy.blockedDomains ?? [])
		.map((domain) => domain.toLowerCase().replace(/^\.+|\.+$/g, ''))
		.filter(Boolean);
	const allowIpAddresses = policy.allowIpAddresses ?? false;
	if (
		!allowIpAddresses &&
		[...allowedDomains, ...blockedDomains].some(isIpAddress)
	) {
		throw new Error(
			'sandbox network domain rules must use domain names, not IP addresses',
		);
	}
	const allowProxy = policy.allowProxy ?? false;
	if (policy.proxyUrl !== undefined) {
		if (!policy.proxyUrl.trim())
			throw new Error('sandbox proxyUrl must not be blank');
		if (!allowProxy) throw new Error('sandbox proxy use is disabled');
	}
	for (const [name, value] of [
		['maxBytesPerSecond', policy.maxBytesPerSecond],
		['maxTransferBytes', policy.maxTransferBytes],
	] as const) {
		if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
			throw new Error(name + ' must be a non-negative integer');
		}
	}
	return {
		mode: policy.mode ?? 'none',
		allowedDomains,
		blockedDomains,
		allowHttp: policy.allowHttp ?? false,
		allowWebSocket: policy.allowWebSocket ?? false,
		allowIpAddresses,
		allowProxy,
		maxBytesPerSecond: policy.maxBytesPerSecond,
		maxTransferBytes: policy.maxTransferBytes,
		...(policy.proxyUrl ? { proxyUrl: policy.proxyUrl } : {}),
	};
}

function domainMatches(domain: string, patterns: string[]): boolean {
	return patterns.some(
		(pattern) => domain === pattern || domain.endsWith('.' + pattern),
	);
}

export function sandboxNetworkAllows(
	policyValue: SandboxNetworkPolicy,
	target: string,
): boolean {
	const policy = normalizeNetworkPolicy(policyValue);
	// Reject characters on which URL parsers disagree (WHATWG treats "\\" as
	// "/"), so the host we check is the host a client would actually reach.
	if (
		[...target].some((ch) => {
			const code = ch.charCodeAt(0);
			return ch === '\\' || code <= 32 || code === 127;
		})
	)
		return false;
	let parsed: URL | undefined;
	try {
		parsed = new URL(target.includes('://') ? target : 'https://' + target);
	} catch {
		return false;
	}
	const scheme = parsed.protocol.replace(/:$/, '').toLowerCase();
	if ((scheme === 'ws' || scheme === 'wss') && !policy.allowWebSocket)
		return false;
	if (scheme === 'http' && !policy.allowHttp) return false;
	if (!['https', 'http', 'ws', 'wss'].includes(scheme)) return false;
	const domain = parsed.hostname.toLowerCase().replace(/^\.+|\.+$/g, '');
	if (
		(domain === 'localhost' || isIpAddress(domain)) &&
		!policy.allowIpAddresses
	)
		return false;
	if (domainMatches(domain, policy.blockedDomains)) return false;
	if (policy.mode === 'none') return false;
	if (policy.mode === 'unrestricted') return true;
	return domainMatches(domain, policy.allowedDomains);
}

export function sanitizeSandboxEnvironment(
	environment: Record<string, string>,
	policyValue: SandboxNetworkPolicy,
): Record<string, string> {
	const policy = normalizeNetworkPolicy(policyValue);
	if (
		!policy.allowProxy &&
		Object.keys(environment).some((key) => SANDBOX_PROXY_ENV_KEYS.has(key))
	) {
		throw new Error('sandbox proxy environment variables are disabled');
	}
	return { ...environment };
}

export function requiresAdvancedNetworkEnforcement(
	policyValue: SandboxNetworkPolicy,
): boolean {
	const policy = normalizeNetworkPolicy(policyValue);
	return Boolean(
		policy.mode === 'allowlist' ||
		policy.blockedDomains.length ||
		!policy.allowHttp ||
		!policy.allowWebSocket ||
		!policy.allowIpAddresses ||
		policy.maxBytesPerSecond !== undefined ||
		policy.maxTransferBytes !== undefined,
	);
}

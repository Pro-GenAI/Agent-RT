/** SWE-ReX sandbox backend. */
import { type WorkspaceFiles } from '../workspace/filesystem.js';
import {
	type SandboxNetworkPolicy,
	type SandboxResourceLimits,
	normalizeNetworkPolicy,
	requiresAdvancedNetworkEnforcement,
} from './policy.js';
import {
	type SandboxBackend,
	type SandboxCommand,
	type SandboxCommandResult,
} from './backends.js';

export class SWEReXSandboxBackend implements SandboxBackend {
	readonly url: string;

	constructor(
		url: string,
		readonly apiKey?: string,
	) {
		const normalized = url.trim().replace(/\/+$/, '');
		if (!normalized) throw new Error('SWE-ReX URL must not be empty');
		if (
			!normalized.startsWith('http://') &&
			!normalized.startsWith('https://')
		) {
			throw new Error('SWE-ReX URL must start with http:// or https://');
		}
		this.url = normalized;
	}

	async execute(
		_sessionId: string,
		command: SandboxCommand,
		context: {
			workspace: WorkspaceFiles;
			limits: SandboxResourceLimits;
			environment: Record<string, string>;
			networkPolicy: SandboxNetworkPolicy;
		},
	): Promise<SandboxCommandResult> {
		const policy = normalizeNetworkPolicy(context.networkPolicy);
		if (policy.mode !== 'unrestricted') {
			throw new Error(
				'SWE-ReX remote backend cannot enforce harness network isolation; use a preconfigured isolated deployment and request unrestricted mode',
			);
		}
		if (requiresAdvancedNetworkEnforcement(policy)) {
			throw new Error(
				'SWE-ReX remote backend cannot enforce the requested advanced network controls',
			);
		}
		if (
			policy.allowedDomains.length ||
			policy.blockedDomains.length ||
			policy.proxyUrl
		) {
			throw new Error(
				'SWE-ReX remote backend does not implement domain/proxy network policy',
			);
		}
		if (
			context.limits.cpuSeconds !== undefined ||
			context.limits.memoryBytes !== undefined ||
			context.limits.diskBytes !== undefined ||
			context.limits.processCount !== undefined
		) {
			throw new Error(
				'SWE-ReX remote backend does not enforce per-command cpu/memory/disk/process limits',
			);
		}
		if (command.stdin !== undefined) {
			throw new Error(
				'SWE-ReX remote backend does not support command stdin',
			);
		}
		const headers: Record<string, string> = {
			'content-type': 'application/json',
		};
		if (this.apiKey) headers['X-API-Key'] = this.apiKey;
		const started = Date.now();
		const response = await fetch(this.url + '/execute', {
			method: 'POST',
			headers,
			body: JSON.stringify({
				command: command.argv,
				timeout:
					context.limits.timeoutMs === undefined
						? null
						: context.limits.timeoutMs / 1000,
				shell: false,
				check: false,
				error_msg: '',
				env:
					Object.keys(context.environment).length === 0
						? null
						: context.environment,
				cwd: command.cwd ?? null,
				merge_output_streams: false,
			}),
		});
		if (!response.ok) {
			throw new Error(
				`SWE-ReX execute request failed with HTTP ${response.status}: ${await response.text()}`,
			);
		}
		const result = (await response.json()) as {
			stdout?: string;
			stderr?: string;
			exit_code?: number | null;
		};
		if (typeof result.exit_code !== 'number') {
			throw new Error('SWE-ReX returned no exit code');
		}
		return {
			exitCode: result.exit_code,
			stdout: new TextEncoder().encode(result.stdout ?? ''),
			stderr: new TextEncoder().encode(result.stderr ?? ''),
			durationMs: Math.max(0, Date.now() - started),
		};
	}
}

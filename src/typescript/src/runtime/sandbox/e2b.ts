/** E2B sandbox backend. */
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
import { shellQuote } from './docker.js';

export class E2BSandboxBackend implements SandboxBackend {
	private sandboxes = new Map<string, any>();
	private creating = new Map<string, Promise<any>>();

	constructor(readonly template?: string) {}

	/** One creation per session so concurrent first commands cannot leak sandboxes. */
	private sandbox(sessionId: string): Promise<any> {
		const existing = this.sandboxes.get(sessionId);
		if (existing) return Promise.resolve(existing);
		let pending = this.creating.get(sessionId);
		if (!pending) {
			pending = this.createSandbox(sessionId).finally(() =>
				this.creating.delete(sessionId),
			);
			this.creating.set(sessionId, pending);
		}
		return pending;
	}

	async closeSession(sessionId: string): Promise<boolean> {
		const sandbox = this.sandboxes.get(sessionId);
		this.sandboxes.delete(sessionId);
		if (!sandbox) return false;
		if (typeof sandbox.kill === 'function') await sandbox.kill();
		return true;
	}

	private async createSandbox(sessionId: string): Promise<any> {
		const dynamicImport = Function(
			'specifier',
			'return import(specifier)',
		) as (specifier: string) => Promise<any>;
		let sdk: any;
		try {
			sdk = await dynamicImport('e2b');
		} catch (error) {
			throw new Error(
				"E2B sandbox support requires the optional 'e2b' package: " +
					String(error),
			);
		}
		const sandbox = this.template?.trim()
			? await sdk.Sandbox.create(this.template)
			: await sdk.Sandbox.create();
		this.sandboxes.set(sessionId, sandbox);
		return sandbox;
	}

	async execute(
		sessionId: string,
		command: SandboxCommand,
		context: {
			workspace: WorkspaceFiles;
			limits: SandboxResourceLimits;
			environment: Record<string, string>;
			networkPolicy: SandboxNetworkPolicy;
		},
	): Promise<SandboxCommandResult> {
		const policy = normalizeNetworkPolicy(context.networkPolicy);
		if (
			policy.mode !== 'none' &&
			requiresAdvancedNetworkEnforcement(policy)
		) {
			throw new Error(
				'E2B backend cannot enforce the requested advanced network controls',
			);
		}
		if (
			policy.mode !== 'unrestricted' ||
			policy.allowedDomains.length ||
			policy.blockedDomains.length ||
			policy.proxyUrl
		) {
			throw new Error(
				'E2B backend requires unrestricted network policy; custom domain/proxy enforcement belongs in the E2B template',
			);
		}
		if (
			context.limits.cpuSeconds !== undefined ||
			context.limits.memoryBytes !== undefined ||
			context.limits.diskBytes !== undefined ||
			context.limits.processCount !== undefined
		) {
			throw new Error(
				'E2B backend does not enforce per-command resource limits',
			);
		}
		if (command.stdin !== undefined) {
			throw new Error('E2B backend does not support command stdin');
		}
		const sandbox = await this.sandbox(sessionId);
		const started = Date.now();
		const result = await sandbox.commands.run(
			command.argv.map(shellQuote).join(' '),
			{
				envs: context.environment,
				...(command.cwd ? { cwd: command.cwd } : {}),
				...(context.limits.timeoutMs !== undefined
					? { timeoutMs: context.limits.timeoutMs }
					: {}),
			},
		);
		return {
			exitCode: Number(result.exitCode),
			stdout: new TextEncoder().encode(String(result.stdout ?? '')),
			stderr: new TextEncoder().encode(String(result.stderr ?? '')),
			durationMs: Math.max(0, Date.now() - started),
		};
	}
}

/** Microsandbox sandbox backend. */
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

export class MicrosandboxBackend implements SandboxBackend {
	private sandboxes = new Map<string, any>();
	private creating = new Map<string, Promise<any>>();
	private names = new Map<string, string>();
	// Network mode each microVM was created with; it cannot change afterwards.
	private networkModes = new Map<string, string>();
	private static nextId = 0;

	constructor(readonly image: string) {
		if (!image.trim())
			throw new Error('microsandbox image must not be empty');
	}

	private assertNetworkUnchanged(
		sessionId: string,
		policy: ReturnType<typeof normalizeNetworkPolicy>,
	): void {
		const createdWith = this.networkModes.get(sessionId);
		if (createdWith !== undefined && createdWith !== policy.mode) {
			throw new Error(
				"microsandbox cannot change a running microVM's network policy (created with '" +
					createdWith +
					"', now '" +
					policy.mode +
					"'); close the session and start a new one",
			);
		}
	}

	private sandbox(
		sessionId: string,
		policy: ReturnType<typeof normalizeNetworkPolicy>,
	): Promise<any> {
		this.assertNetworkUnchanged(sessionId, policy);
		const existing = this.sandboxes.get(sessionId);
		if (existing) return Promise.resolve(existing);
		let pending = this.creating.get(sessionId);
		if (!pending) {
			pending = this.createSandbox(sessionId, policy).finally(() =>
				this.creating.delete(sessionId),
			);
			this.creating.set(sessionId, pending);
		}
		return pending;
	}

	async closeSession(sessionId: string): Promise<boolean> {
		const sandbox = this.sandboxes.get(sessionId);
		this.sandboxes.delete(sessionId);
		this.names.delete(sessionId);
		this.networkModes.delete(sessionId);
		if (!sandbox) return false;
		for (const method of ['stop', 'kill', 'close']) {
			if (typeof sandbox[method] === 'function') {
				await sandbox[method]();
				break;
			}
		}
		return true;
	}

	private async createSandbox(
		sessionId: string,
		policy: ReturnType<typeof normalizeNetworkPolicy>,
	): Promise<any> {
		if (
			policy.mode !== 'none' &&
			requiresAdvancedNetworkEnforcement(policy)
		) {
			throw new Error(
				'microsandbox backend cannot enforce the requested advanced network controls',
			);
		}
		if (
			policy.allowedDomains.length ||
			policy.blockedDomains.length ||
			policy.proxyUrl
		) {
			throw new Error(
				'microsandbox backend currently supports only whole-sandbox none/unrestricted network policies',
			);
		}
		const dynamicImport = Function(
			'specifier',
			'return import(specifier)',
		) as (specifier: string) => Promise<any>;
		let sdk: any;
		try {
			sdk = await dynamicImport('microsandbox');
		} catch (error) {
			throw new Error(
				"microsandbox support requires the optional 'microsandbox' package: " +
					String(error),
			);
		}
		const networkPolicy =
			policy.mode === 'none'
				? sdk.NetworkPolicy.none()
				: policy.mode === 'unrestricted'
					? sdk.NetworkPolicy.allowAll()
					: undefined;
		if (!networkPolicy) {
			throw new Error(
				'microsandbox backend does not yet map harness domain allowlists',
			);
		}
		const name =
			this.names.get(sessionId) ??
			'agent-rt-' +
				Date.now().toString(36) +
				'-' +
				(++MicrosandboxBackend.nextId).toString(36);
		this.names.set(sessionId, name);
		const sandbox = await sdk.Sandbox.builder(name)
			.image(this.image)
			.network((network: any) => network.policy(networkPolicy))
			.create();
		this.sandboxes.set(sessionId, sandbox);
		this.networkModes.set(sessionId, policy.mode);
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
		if (
			context.limits.cpuSeconds !== undefined ||
			context.limits.memoryBytes !== undefined ||
			context.limits.diskBytes !== undefined ||
			context.limits.processCount !== undefined
		) {
			throw new Error(
				'microsandbox backend does not map harness per-command cpu/memory/disk/process limits',
			);
		}
		if (command.stdin !== undefined) {
			throw new Error(
				'microsandbox backend does not support command stdin',
			);
		}
		const policy = normalizeNetworkPolicy(context.networkPolicy);
		const sandbox = await this.sandbox(sessionId, policy);
		const started = Date.now();
		const result = await sandbox.execWith(command.argv[0], (exec: any) => {
			exec.args(command.argv.slice(1));
			if (command.cwd) exec.cwd(command.cwd);
			for (const [key, value] of Object.entries(context.environment)) {
				exec.env(key, value);
			}
			if (context.limits.timeoutMs !== undefined) {
				exec.timeout(context.limits.timeoutMs);
			}
			return exec;
		});
		return {
			exitCode: Number(result.code),
			stdout: new TextEncoder().encode(String(result.stdout())),
			stderr: new TextEncoder().encode(String(result.stderr())),
			durationMs: Math.max(0, Date.now() - started),
		};
	}
}

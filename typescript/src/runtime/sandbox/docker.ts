/** Docker sandbox backend. */
import { type WorkspaceFiles } from '../workspace/filesystem.js';
import {
	type SandboxNetworkPolicy,
	type SandboxResourceLimits,
	normalizeNetworkPolicy,
	requiresAdvancedNetworkEnforcement,
	sanitizeSandboxEnvironment,
} from './policy.js';
import {
	type SandboxBackend,
	type SandboxCommand,
	type SandboxCommandResult,
} from './backends.js';
import { hostSearchPath } from '../platform.js';
import { runLocalProcess } from './native.js';

export class DockerSandboxBackend implements SandboxBackend {
	constructor(
		readonly image: string,
		readonly options: {
			dockerBinary?: string;
			workspaceRoot?: string;
			user?: string;
			readOnlyRoot?: boolean;
		} = {},
	) {
		if (!image.trim())
			throw new Error('docker sandbox image must not be empty');
		// `--mount` options are comma separated; a comma would let the path
		// inject extra mount options.
		if (options.workspaceRoot && /[,\n\r\0]/.test(options.workspaceRoot)) {
			throw new Error(
				'docker sandbox workspaceRoot must not contain commas or newlines',
			);
		}
		if (options.user !== undefined && !options.user.trim()) {
			throw new Error(
				'docker sandbox user must not be empty when provided',
			);
		}
	}

	static containerWorkdir(cwd: string): string {
		const parts: string[] = [];
		for (const part of cwd.replace(/\\/g, '/').split('/')) {
			if (!part || part === '.') continue;
			if (part === '..') {
				if (parts.length === 0) {
					throw new Error(
						'sandbox working directory must stay inside /workspace',
					);
				}
				parts.pop();
			} else {
				parts.push(part);
			}
		}
		if (cwd.startsWith('/')) {
			throw new Error(
				'sandbox working directory must stay inside /workspace',
			);
		}
		return parts.length ? '/workspace/' + parts.join('/') : '/workspace';
	}

	async execute(
		_sessionId: string,
		command: SandboxCommand,
		context: {
			workspace: WorkspaceFiles;
			limits: SandboxResourceLimits;
			environment: Record<string, string>;
			networkPolicy: SandboxNetworkPolicy;
			signal?: AbortSignal;
		},
	): Promise<SandboxCommandResult> {
		const policy = normalizeNetworkPolicy(context.networkPolicy);
		if (
			policy.mode !== 'none' &&
			requiresAdvancedNetworkEnforcement(policy)
		) {
			throw new Error(
				'docker backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, allow/block-list, or bandwidth/transfer network controls',
			);
		}
		if (
			context.limits.cpuSeconds !== undefined ||
			context.limits.diskBytes !== undefined
		) {
			throw new Error(
				'docker backend cannot enforce cpuSeconds/diskBytes limits',
			);
		}
		const container = 'agent-rt-' + randomHex(16);
		const args = [
			'run',
			'--rm',
			'-i',
			'--name',
			container,
			'--cap-drop',
			'ALL',
			'--security-opt',
			'no-new-privileges',
		];
		if (this.options.user) args.push('--user', this.options.user.trim());
		if (this.options.readOnlyRoot)
			args.push('--read-only', '--tmpfs', '/tmp');
		if (policy.mode === 'none') args.push('--network', 'none');
		if (context.limits.memoryBytes !== undefined) {
			args.push(
				'--memory',
				String(context.limits.memoryBytes),
				'--memory-swap',
				String(context.limits.memoryBytes),
			);
		}
		if (context.limits.processCount !== undefined) {
			args.push('--pids-limit', String(context.limits.processCount));
		}
		if (this.options.workspaceRoot) {
			args.push(
				'--mount',
				'type=bind,src=' +
					this.options.workspaceRoot +
					',dst=/workspace',
				'--workdir',
				DockerSandboxBackend.containerWorkdir(command.cwd || '.'),
			);
		} else if (command.cwd) {
			args.push('--workdir', command.cwd);
		}
		const environment = sanitizeSandboxEnvironment(
			context.environment,
			context.networkPolicy,
		);
		if (policy.proxyUrl) environment.HTTPS_PROXY ??= policy.proxyUrl;
		for (const [key, value] of Object.entries(environment)) {
			args.push('--env', key + '=' + value);
		}
		args.push(this.image, ...command.argv);
		const docker = this.options.dockerBinary ?? 'docker';
		const dockerEnv = { PATH: hostSearchPath() ?? '/usr/bin:/bin' };
		return runLocalProcess(docker, args, {
			env: dockerEnv,
			stdin: command.stdin,
			outputLimit: context.limits.outputBytes,
			signal: context.signal,
			// Killing the docker client does not stop the container.
			onAbort: () =>
				runLocalProcess(docker, ['kill', container], {
					env: dockerEnv,
					outputLimit: 1024,
				}).then(() => undefined),
		});
	}
}

function randomHex(length: number): string {
	const crypto = globalThis as unknown as {
		crypto?: { getRandomValues?: (array: Uint8Array) => Uint8Array };
	};
	const bytes = new Uint8Array(Math.ceil(length / 2));
	if (crypto.crypto?.getRandomValues) {
		crypto.crypto.getRandomValues(bytes);
	} else {
		for (let index = 0; index < bytes.length; index += 1) {
			bytes[index] = Math.floor(Math.random() * 256);
		}
	}
	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0'))
		.join('')
		.slice(0, length);
}

export function shellQuote(value: string): string {
	return "'" + value.replace(/'/g, "'\\''") + "'";
}

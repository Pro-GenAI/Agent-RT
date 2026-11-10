/** Native process sandbox backend. */
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
import { concatBytes, hostSearchPath, nodeModule } from '../platform.js';

/** Ceiling for captured sandbox output when no `outputBytes` limit is set. */
export const DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024;

export async function runLocalProcess(
	executable: string,
	args: string[],
	options: {
		cwd?: string;
		env: Record<string, string>;
		stdin?: Uint8Array;
		outputLimit?: number;
		signal?: AbortSignal;
		/** Extra teardown run when the command is aborted (e.g. `docker kill`). */
		onAbort?: () => Promise<void> | void;
	},
): Promise<SandboxCommandResult> {
	const { spawn } = nodeModule('node:child_process');
	const started = Date.now();
	const limit = options.outputLimit ?? DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES;
	return await new Promise<SandboxCommandResult>((resolve, reject) => {
		if (options.signal?.aborted) {
			reject(options.signal.reason ?? new Error('aborted'));
			return;
		}
		// `detached` makes the child a process-group leader so the whole tree
		// can be killed, not only the direct child.
		const child = spawn(executable, args, {
			cwd: options.cwd,
			env: options.env,
			stdio: ['pipe', 'pipe', 'pipe'],
			detached: process.platform !== 'win32',
		});
		const stdout: Uint8Array[] = [];
		const stderr: Uint8Array[] = [];
		let stdoutSize = 0;
		let stderrSize = 0;
		let truncated = false;
		const capture = (
			sink: Uint8Array[],
			size: number,
			chunk: Uint8Array,
		): number => {
			const room = limit - size;
			if (room > 0) sink.push(new Uint8Array(chunk.subarray(0, room)));
			if (chunk.length > room) truncated = true;
			return size + Math.min(chunk.length, Math.max(room, 0));
		};
		child.stdout.on('data', (chunk: Uint8Array) => {
			stdoutSize = capture(stdout, stdoutSize, chunk);
		});
		child.stderr.on('data', (chunk: Uint8Array) => {
			stderrSize = capture(stderr, stderrSize, chunk);
		});
		const abort = () => {
			try {
				if (child.pid !== undefined && process.platform !== 'win32') {
					process.kill(-child.pid, 'SIGKILL');
				} else {
					child.kill('SIGKILL');
				}
			} catch {
				child.kill('SIGKILL');
			}
			void Promise.resolve(options.onAbort?.()).catch(() => undefined);
			reject(options.signal?.reason ?? new Error('aborted'));
		};
		options.signal?.addEventListener('abort', abort, { once: true });
		child.on('error', (error: Error) => {
			options.signal?.removeEventListener('abort', abort);
			reject(error);
		});
		child.on('close', (code: number | null) => {
			options.signal?.removeEventListener('abort', abort);
			resolve({
				exitCode: code ?? -1,
				stdout: concatBytes(stdout),
				stderr: concatBytes(stderr),
				durationMs: Math.max(0, Date.now() - started),
				truncated,
			});
		});
		child.stdin.on('error', () => undefined);
		child.stdin.end(options.stdin);
	});
}

/**
 * Resolve a launcher binary to an absolute path using the *host* PATH.
 *
 * Node resolves a bare executable name through the child's `env.PATH`, and the
 * native backend's launchers (`prlimit`, `setpriv`) run with the host's
 * privileges before dropping to the restricted uid. A model-controlled `PATH`
 * (or `LD_PRELOAD`) in the sandbox environment must never influence which
 * binary the privileged launcher runs, so launchers are resolved here and the
 * sandbox environment is applied only after the privilege drop.
 */
function trustedExecutable(name: string): string {
	if (name.includes('/')) return name;
	const fs = nodeModule('node:fs');
	const path = nodeModule('node:path');
	const hostPath =
		hostSearchPath() ??
		'/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin';
	for (const directory of hostPath.split(path.delimiter)) {
		if (!directory || !path.isAbsolute(directory)) continue;
		const candidate = path.join(directory, name);
		try {
			if (fs.statSync(candidate).isFile()) return candidate;
		} catch {
			// keep searching
		}
	}
	throw new Error(
		'required sandbox launcher was not found on the host PATH: ' + name,
	);
}

function resolveSandboxCwd(
	root: string | undefined,
	cwd: string | undefined,
): string | undefined {
	if (!root) return cwd || undefined;
	const path = nodeModule('node:path');
	const base = path.resolve(root);
	const resolved = path.resolve(base, cwd || '.');
	if (resolved !== base && !resolved.startsWith(base + path.sep)) {
		throw new Error('sandbox working directory escapes native root');
	}
	return resolved;
}

export class NativeSandboxBackend implements SandboxBackend {
	constructor(
		readonly uid: number,
		readonly options: {
			gid?: number;
			root?: string;
			setprivBinary?: string;
			prlimitBinary?: string;
		} = {},
	) {
		if (!Number.isInteger(uid) || uid < 0) {
			throw new Error(
				'native sandbox uid must be a non-negative integer',
			);
		}
		if (
			options.gid !== undefined &&
			(!Number.isInteger(options.gid) || options.gid < 0)
		) {
			throw new Error(
				'native sandbox gid must be a non-negative integer',
			);
		}
	}

	/** Explicit gid, else the restricted account's primary gid from passwd. */
	private resolveGid(): number {
		if (this.options.gid !== undefined) return this.options.gid;
		try {
			const passwd = nodeModule('node:fs').readFileSync(
				'/etc/passwd',
				'utf8',
			) as string;
			for (const line of passwd.split('\n')) {
				const fields = line.split(':');
				if (fields.length >= 4 && Number(fields[2]) === this.uid) {
					const gid = Number(fields[3]);
					if (Number.isInteger(gid) && gid >= 0) return gid;
				}
			}
		} catch {
			// fall through to the explicit error below
		}
		throw new Error(
			'native sandbox could not derive a gid for the restricted uid; set AGENT_RT_SANDBOX_GID',
		);
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
		if (process.platform !== 'linux') {
			throw new Error(
				'native sandbox backend requires Linux setpriv/prlimit; configure a non-native sandbox backend on this platform',
			);
		}
		const policy = normalizeNetworkPolicy(context.networkPolicy);
		if (policy.mode === 'none') {
			throw new Error(
				'native sandbox backend cannot verify network isolation; use docker for no-network execution',
			);
		}
		if (requiresAdvancedNetworkEnforcement(policy)) {
			throw new Error(
				'native sandbox backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, or bandwidth/transfer network controls',
			);
		}
		if (
			policy.allowedDomains.length ||
			policy.blockedDomains.length ||
			policy.proxyUrl
		) {
			throw new Error(
				'native sandbox backend does not implement domain/proxy policy',
			);
		}
		const environment: Record<string, string> = {
			PATH: '/usr/local/bin:/usr/bin:/bin',
			...sanitizeSandboxEnvironment(
				context.environment,
				context.networkPolicy,
			),
		};
		for (const key of Object.keys(environment)) {
			if (!key || key.includes('=')) {
				throw new Error(
					'native sandbox environment names must be non-empty and must not contain "="',
				);
			}
		}
		if (command.argv[0]?.includes('=')) {
			throw new Error('native sandbox command name must not contain "="');
		}
		const gid = this.resolveGid();
		// The privileged launchers run with a fixed environment and absolute
		// paths. The sandbox environment is applied by `env -i` only after
		// setpriv has dropped to the restricted uid, so it cannot redirect the
		// launcher (PATH) or inject code into it (LD_PRELOAD and friends).
		const launcherEnvironment: Record<string, string> = {
			PATH: '/usr/local/bin:/usr/bin:/bin',
		};
		const setprivBinary = trustedExecutable(
			this.options.setprivBinary ?? 'setpriv',
		);
		const setprivArgs = [
			'--reuid=' + this.uid,
			// Never inherit the caller's (possibly root) gid or groups.
			'--regid=' + gid,
			'--clear-groups',
			'--no-new-privs',
			'--',
			trustedExecutable('env'),
			'-i',
			...Object.entries(environment).map(
				([key, value]) => key + '=' + value,
			),
			...command.argv,
		];
		const limitArgs: string[] = [];
		if (context.limits.cpuSeconds !== undefined) {
			limitArgs.push(
				'--cpu=' + Math.max(1, Math.ceil(context.limits.cpuSeconds)),
			);
		}
		if (context.limits.memoryBytes !== undefined) {
			limitArgs.push('--as=' + context.limits.memoryBytes);
		}
		if (context.limits.processCount !== undefined) {
			limitArgs.push('--nproc=' + context.limits.processCount);
		}
		if (context.limits.diskBytes !== undefined) {
			limitArgs.push('--fsize=' + context.limits.diskBytes);
		}
		if (limitArgs.length) {
			return runLocalProcess(
				trustedExecutable(this.options.prlimitBinary ?? 'prlimit'),
				[...limitArgs, '--', setprivBinary, ...setprivArgs],
				{
					cwd: resolveSandboxCwd(this.options.root, command.cwd),
					env: launcherEnvironment,
					stdin: command.stdin,
					outputLimit: context.limits.outputBytes,
					signal: context.signal,
				},
			);
		}
		return runLocalProcess(setprivBinary, setprivArgs, {
			cwd: resolveSandboxCwd(this.options.root, command.cwd),
			env: launcherEnvironment,
			stdin: command.stdin,
			outputLimit: context.limits.outputBytes,
			signal: context.signal,
		});
	}
}

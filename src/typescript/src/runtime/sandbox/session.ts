/** Sandbox sessions, package managers, code interpreters, and backend selection. */
import { type JSONValue } from '../types.js';
import { type ToolHandler } from '../tools/definitions.js';
import { InMemoryFileSystem, WorkspaceFiles } from '../workspace/filesystem.js';
import {
	type SandboxNetworkPolicy,
	type SandboxResourceLimits,
	normalizeNetworkPolicy,
	sandboxEnvBoolean,
	sandboxNetworkAllows,
	sandboxNetworkPolicyFromEnv,
	sandboxResourceLimitsFromEnv,
	sanitizeSandboxEnvironment,
} from './policy.js';
import {
	SANDBOX_BACKEND_ENV,
	type SandboxBackend,
	type SandboxCommand,
	type SandboxCommandResult,
	type SandboxSnapshot,
	normalizedSandboxBackendName,
} from './backends.js';
import { runtimeEnvironment } from '../platform.js';
import { NativeSandboxBackend } from './native.js';
import { DockerSandboxBackend } from './docker.js';
import { E2BSandboxBackend } from './e2b.js';
import { MicrosandboxBackend } from './microsandbox.js';
import { SWEReXSandboxBackend } from './swe_rex.js';

export function sandboxBackendFromEnv(
	env: Record<string, string | undefined> = runtimeEnvironment(),
): SandboxBackend {
	if (Object.prototype.hasOwnProperty.call(env, 'AGENT_RT_DISABLE_SANDBOX')) {
		throw new Error(
			'AGENT_RT_DISABLE_SANDBOX is not supported; sandboxing cannot be disabled',
		);
	}
	const backendName = normalizedSandboxBackendName(env[SANDBOX_BACKEND_ENV]);
	if (backendName === 'native') {
		const rawUid = env.AGENT_RT_SANDBOX_UID;
		if (!rawUid?.trim()) {
			throw new Error(
				'native sandbox requires AGENT_RT_SANDBOX_UID for a restricted user',
			);
		}
		const uid = Number(rawUid);
		const rawGid = env.AGENT_RT_SANDBOX_GID?.trim();
		const gid = rawGid ? Number(rawGid) : undefined;
		if (
			!Number.isInteger(uid) ||
			(gid !== undefined && !Number.isInteger(gid))
		) {
			throw new Error('native sandbox uid/gid must be integers');
		}
		return new NativeSandboxBackend(uid, {
			gid,
			root: env.AGENT_RT_SANDBOX_NATIVE_ROOT,
			setprivBinary: env.AGENT_RT_SANDBOX_SETPRIV_BINARY,
			prlimitBinary: env.AGENT_RT_SANDBOX_PRLIMIT_BINARY,
		});
	}
	if (backendName === 'docker') {
		const image = env.AGENT_RT_SANDBOX_DOCKER_IMAGE?.trim();
		if (!image) {
			throw new Error(
				'docker sandbox requires AGENT_RT_SANDBOX_DOCKER_IMAGE',
			);
		}
		return new DockerSandboxBackend(image, {
			dockerBinary: env.AGENT_RT_SANDBOX_DOCKER_BINARY,
			workspaceRoot: env.AGENT_RT_SANDBOX_WORKSPACE_ROOT,
			user: env.AGENT_RT_SANDBOX_DOCKER_USER?.trim() || undefined,
			readOnlyRoot: sandboxEnvBoolean(
				env,
				'AGENT_RT_SANDBOX_DOCKER_READ_ONLY',
			),
		});
	}
	if (backendName === 'e2b') {
		return new E2BSandboxBackend(env.AGENT_RT_SANDBOX_E2B_TEMPLATE);
	}
	if (backendName === 'microsandbox') {
		const image = env.AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE?.trim();
		if (!image) {
			throw new Error(
				'microsandbox backend requires AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE',
			);
		}
		return new MicrosandboxBackend(image);
	}
	const url = env.AGENT_RT_SANDBOX_SWEREX_URL?.trim();
	if (!url) {
		throw new Error('SWE-ReX backend requires AGENT_RT_SANDBOX_SWEREX_URL');
	}
	return new SWEReXSandboxBackend(url, env.AGENT_RT_SANDBOX_SWEREX_API_KEY);
}

export interface SandboxPackageManager {
	install(
		sessionId: string,
		runtime: string,
		packages: string[],
		context: {
			workspace: WorkspaceFiles;
			environment: Record<string, string>;
			limits: SandboxResourceLimits;
			networkPolicy: SandboxNetworkPolicy;
		},
	): Promise<string[]>;
}

export type SandboxPackageInstaller = (
	sessionId: string,
	runtime: string,
	packages: string[],
	workspace: WorkspaceFiles,
	environment: Record<string, string>,
	limits: SandboxResourceLimits,
	networkPolicy: SandboxNetworkPolicy,
) => Promise<string[]>;

export class CallbackSandboxPackageManager implements SandboxPackageManager {
	constructor(readonly installer: SandboxPackageInstaller) {}

	install(
		sessionId: string,
		runtime: string,
		packages: string[],
		context: {
			workspace: WorkspaceFiles;
			environment: Record<string, string>;
			limits: SandboxResourceLimits;
			networkPolicy: SandboxNetworkPolicy;
		},
	): Promise<string[]> {
		return this.installer(
			sessionId,
			runtime,
			packages,
			context.workspace,
			context.environment,
			context.limits,
			context.networkPolicy,
		);
	}
}

export type CodeInterpreterRunner = (
	sessionId: string,
	runtime: string,
	code: string,
	state: Record<string, JSONValue>,
	workspace: WorkspaceFiles,
	environment: Record<string, string>,
	limits: SandboxResourceLimits,
	networkPolicy: SandboxNetworkPolicy,
) => Promise<SandboxCommandResult>;

export class CodeInterpreterRegistry {
	private readonly runners = new Map<string, CodeInterpreterRunner>();

	register(
		runtime: string,
		runner: CodeInterpreterRunner,
		options: { replace?: boolean } = {},
	): void {
		if (!runtime.trim())
			throw new Error('interpreter runtime must not be empty');
		if (this.runners.has(runtime) && !options.replace) {
			throw new Error(
				'interpreter runtime already registered: ' + runtime,
			);
		}
		this.runners.set(runtime, runner);
	}

	get(runtime: string): CodeInterpreterRunner {
		const runner = this.runners.get(runtime);
		if (!runner)
			throw new Error('interpreter runtime not registered: ' + runtime);
		return runner;
	}

	list(): string[] {
		return [...this.runners.keys()].sort();
	}
}

function validateSandboxLimits(limits: SandboxResourceLimits): void {
	for (const [name, value] of Object.entries(limits)) {
		if (value !== undefined && value < 0) {
			throw new Error(name + ' must not be negative');
		}
	}
	// Container runtimes treat a 0 memory/pids limit as "unlimited" while
	// rlimits treat it as "nothing may run"; reject the ambiguity outright.
	for (const name of ['memoryBytes', 'processCount'] as const) {
		if (limits[name] === 0) {
			throw new Error(name + ' must be positive when provided');
		}
	}
}

export class SandboxSession {
	readonly workspace: WorkspaceFiles;
	readonly limits: SandboxResourceLimits;
	readonly interpreters: CodeInterpreterRegistry;
	readonly environment: Record<string, string> = {};
	readonly runtimeVersions = new Map<string, string>();
	readonly installedPackages = new Map<string, string[]>();
	readonly interpreterState = new Map<string, Record<string, JSONValue>>();
	private readonly snapshots = new Map<string, SandboxSnapshot>();
	networkPolicy: SandboxNetworkPolicy;
	cwd = '';

	constructor(
		readonly sessionId: string,
		readonly backend: SandboxBackend,
		options: {
			workspace?: WorkspaceFiles;
			limits?: SandboxResourceLimits;
			interpreters?: CodeInterpreterRegistry;
			packageManager?: SandboxPackageManager;
			networkPolicy?: SandboxNetworkPolicy;
		} = {},
	) {
		if (!sessionId.trim())
			throw new Error('sandbox sessionId must not be empty');
		this.workspace =
			options.workspace ?? new WorkspaceFiles(new InMemoryFileSystem());
		this.limits = {
			...sandboxResourceLimitsFromEnv(),
			...(options.limits ?? {}),
		};
		validateSandboxLimits(this.limits);
		this.interpreters =
			options.interpreters ?? new CodeInterpreterRegistry();
		this.packageManager = options.packageManager;
		this.networkPolicy = normalizeNetworkPolicy(
			options.networkPolicy ?? sandboxNetworkPolicyFromEnv(),
		);
	}

	readonly packageManager?: SandboxPackageManager;

	setEnvironment(values: Record<string, string>): void {
		for (const [key, value] of Object.entries(values)) {
			if (!key || key.includes('\0') || value.includes('\0')) {
				throw new Error(
					'sandbox environment contains an invalid entry',
				);
			}
			this.environment[key] = value;
		}
	}

	unsetEnvironment(...keys: string[]): void {
		for (const key of keys) delete this.environment[key];
	}

	setWorkingDirectory(path: string): void {
		const parts: string[] = [];
		for (const part of path.trim().replace(/^\/+/, '').split('/')) {
			if (!part || part === '.') continue;
			if (part === '..') {
				if (parts.length === 0)
					throw new Error('path escapes filesystem root');
				parts.pop();
			} else {
				parts.push(part);
			}
		}
		this.cwd = parts.join('/');
	}

	setRuntimeVersion(runtime: string, version: string): void {
		if (!runtime.trim() || !version.trim()) {
			throw new Error('runtime and version must not be empty');
		}
		this.runtimeVersions.set(runtime, version);
	}

	runtimeVersion(runtime: string): string | undefined {
		return this.runtimeVersions.get(runtime);
	}

	setNetworkPolicy(policy: SandboxNetworkPolicy): void {
		this.networkPolicy = normalizeNetworkPolicy(policy);
	}

	networkAllows(target: string): boolean {
		return sandboxNetworkAllows(this.networkPolicy, target);
	}

	private snapshotFiles(): Record<string, Uint8Array> {
		const files: Record<string, Uint8Array> = {};
		const walk = (path = ''): void => {
			for (const item of this.workspace.list(path)) {
				if (item.isDirectory) {
					walk(item.path);
				} else {
					files[item.path] = this.workspace.filesystem.read(
						item.path,
					);
				}
			}
		};
		walk();
		return files;
	}

	createSnapshot(snapshotId: string): SandboxSnapshot {
		if (!snapshotId.trim())
			throw new Error('sandbox snapshotId must not be empty');
		if (this.snapshots.has(snapshotId)) {
			throw new Error('sandbox snapshot already exists: ' + snapshotId);
		}
		const snapshot: SandboxSnapshot = {
			snapshotId,
			sourceSessionId: this.sessionId,
			files: this.snapshotFiles(),
			cwd: this.cwd,
			environment: { ...this.environment },
			runtimeVersions: Object.fromEntries(this.runtimeVersions),
			installedPackages: Object.fromEntries(
				[...this.installedPackages.entries()].map(
					([runtime, packages]) => [runtime, [...packages]],
				),
			),
			interpreterState: Object.fromEntries(
				[...this.interpreterState.entries()].map(([runtime, state]) => [
					runtime,
					{ ...state },
				]),
			),
			networkPolicy: {
				...this.networkPolicy,
				allowedDomains: [...(this.networkPolicy.allowedDomains ?? [])],
				blockedDomains: [...(this.networkPolicy.blockedDomains ?? [])],
			},
			createdAtMs: Date.now(),
		};
		this.snapshots.set(snapshotId, snapshot);
		return snapshot;
	}

	getSnapshot(snapshotId: string): SandboxSnapshot {
		const snapshot = this.snapshots.get(snapshotId);
		if (!snapshot)
			throw new Error('sandbox snapshot not found: ' + snapshotId);
		return snapshot;
	}

	restoreSnapshot(snapshotId: string): SandboxSnapshot {
		const snapshot = this.getSnapshot(snapshotId);
		this.workspace.clear();
		for (const [path, data] of Object.entries(snapshot.files)) {
			this.workspace.filesystem.write(path, data);
		}
		this.cwd = snapshot.cwd;
		for (const key of Object.keys(this.environment))
			delete this.environment[key];
		Object.assign(this.environment, snapshot.environment);
		this.runtimeVersions.clear();
		for (const [key, value] of Object.entries(snapshot.runtimeVersions)) {
			this.runtimeVersions.set(key, value);
		}
		this.installedPackages.clear();
		for (const [key, value] of Object.entries(snapshot.installedPackages)) {
			this.installedPackages.set(key, [...value]);
		}
		this.interpreterState.clear();
		for (const [key, value] of Object.entries(snapshot.interpreterState)) {
			this.interpreterState.set(key, { ...value });
		}
		this.networkPolicy = normalizeNetworkPolicy(snapshot.networkPolicy);
		return snapshot;
	}

	cloneFromSnapshot(
		snapshotId: string,
		newSessionId: string,
	): SandboxSession {
		const snapshot = this.getSnapshot(snapshotId);
		const clone = new SandboxSession(newSessionId, this.backend, {
			limits: this.limits,
			interpreters: this.interpreters,
			packageManager: this.packageManager,
			networkPolicy: snapshot.networkPolicy,
		});
		clone.snapshots.set(snapshotId, snapshot);
		clone.restoreSnapshot(snapshotId);
		return clone;
	}

	installPackages(runtime: string, packages: string[]): string[] {
		if (!runtime.trim()) throw new Error('runtime must not be empty');
		const normalized = [
			...new Set(packages.map((item) => item.trim()).filter(Boolean)),
		];
		this.installedPackages.set(runtime, normalized);
		return [...normalized];
	}

	async installRuntimePackages(
		runtime: string,
		packages: string[],
	): Promise<string[]> {
		if (!this.packageManager) {
			throw new Error('sandbox package manager is not configured');
		}
		const normalized = [
			...new Set(packages.map((item) => item.trim()).filter(Boolean)),
		];
		const installed = await this.packageManager.install(
			this.sessionId,
			runtime,
			normalized,
			{
				workspace: this.workspace,
				environment: sanitizeSandboxEnvironment(
					{ ...this.environment },
					this.networkPolicy,
				),
				limits: this.limits,
				networkPolicy: this.networkPolicy,
			},
		);
		const recorded = [
			...new Set(installed.map((item) => item.trim()).filter(Boolean)),
		];
		this.installedPackages.set(runtime, recorded);
		return [...recorded];
	}

	runtimePackages(runtime: string): string[] {
		return [...(this.installedPackages.get(runtime) ?? [])];
	}

	async execute(command: SandboxCommand): Promise<SandboxCommandResult> {
		if (command.argv.length === 0 || command.argv.some((part) => !part)) {
			throw new Error(
				'sandbox command argv must contain non-empty arguments',
			);
		}
		const environment = sanitizeSandboxEnvironment(
			{ ...this.environment, ...(command.env ?? {}) },
			this.networkPolicy,
		);
		const effective: SandboxCommand = {
			...command,
			cwd: command.cwd || this.cwd,
			env: command.env ?? {},
		};
		const controller = new AbortController();
		const operation = this.backend.execute(this.sessionId, effective, {
			workspace: this.workspace,
			limits: this.limits,
			environment,
			networkPolicy: this.networkPolicy,
			signal: controller.signal,
		});
		const result = await this.withTimeout(
			operation,
			'sandbox command exceeded execution timeout',
			controller,
		);
		return this.boundedResult(result);
	}

	async runCode(
		runtime: string,
		code: string,
	): Promise<SandboxCommandResult> {
		const runner = this.interpreters.get(runtime);
		let state = this.interpreterState.get(runtime);
		if (!state) {
			state = {};
			this.interpreterState.set(runtime, state);
		}
		const operation = runner(
			this.sessionId,
			runtime,
			code,
			state,
			this.workspace,
			{ ...this.environment },
			this.limits,
			this.networkPolicy,
		);
		const result = await this.withTimeout(
			operation,
			'sandbox interpreter exceeded execution timeout',
		);
		return this.boundedResult(result);
	}

	/** Release backend resources (cloud sandboxes, microVMs) for this session. */
	async close(): Promise<void> {
		await this.backend.closeSession?.(this.sessionId);
	}

	private async withTimeout(
		operation: Promise<SandboxCommandResult>,
		message: string,
		controller?: AbortController,
	): Promise<SandboxCommandResult> {
		const timeoutMs = this.limits.timeoutMs;
		if (timeoutMs === undefined) return operation;
		return await new Promise<SandboxCommandResult>((resolve, reject) => {
			const timer = setTimeout(() => {
				const error = new Error(message);
				// Stop the command itself, not just our wait for it.
				controller?.abort(error);
				reject(error);
			}, timeoutMs);
			operation.then(
				(value) => {
					clearTimeout(timer);
					resolve(value);
				},
				(error) => {
					clearTimeout(timer);
					reject(error);
				},
			);
		});
	}

	private boundedResult(result: SandboxCommandResult): SandboxCommandResult {
		const stdout = result.stdout ?? new Uint8Array();
		const stderr = result.stderr ?? new Uint8Array();
		const limit = this.limits.outputBytes;
		if (limit === undefined) {
			return {
				...result,
				stdout,
				stderr,
				durationMs: result.durationMs ?? 0,
				truncated: result.truncated ?? false,
			};
		}
		const boundedStdout = stdout.slice(0, limit);
		const remaining = Math.max(0, limit - boundedStdout.length);
		const boundedStderr = stderr.slice(0, remaining);
		return {
			...result,
			stdout: boundedStdout,
			stderr: boundedStderr,
			durationMs: result.durationMs ?? 0,
			truncated:
				(result.truncated ?? false) ||
				boundedStdout.length < stdout.length ||
				boundedStderr.length < stderr.length,
		};
	}
}

export function sandboxShellTool(session: SandboxSession): ToolHandler {
	return async (argumentsValue) => {
		const argv = argumentsValue.argv;
		if (!Array.isArray(argv))
			throw new Error('shell tool requires argv array');
		const envValue = argumentsValue.env ?? {};
		if (
			typeof envValue !== 'object' ||
			envValue === null ||
			Array.isArray(envValue)
		) {
			throw new Error('shell tool env must be an object');
		}
		const result = await session.execute({
			argv: argv.map((item) => String(item)),
			cwd: String(argumentsValue.cwd ?? ''),
			env: Object.fromEntries(
				Object.entries(envValue).map(([key, value]) => [
					key,
					String(value),
				]),
			),
		});
		return {
			exit_code: result.exitCode,
			stdout: new TextDecoder().decode(result.stdout ?? new Uint8Array()),
			stderr: new TextDecoder().decode(result.stderr ?? new Uint8Array()),
			duration_ms: result.durationMs ?? 0,
			truncated: result.truncated ?? false,
		};
	};
}

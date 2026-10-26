/** Sandbox commands, snapshots, the backend protocol, and backend names. */
import { type JSONValue } from '../types.js';
import { type WorkspaceFiles } from '../workspace/filesystem.js';
import {
	type SandboxNetworkPolicy,
	type SandboxResourceLimits,
} from './policy.js';

export interface SandboxCommand {
	argv: string[];
	cwd?: string;
	env?: Record<string, string>;
	stdin?: Uint8Array;
}

export interface SandboxCommandResult {
	exitCode: number;
	stdout: Uint8Array;
	stderr: Uint8Array;
	durationMs: number;
	truncated?: boolean;
}

export interface SandboxSnapshot {
	snapshotId: string;
	sourceSessionId: string;
	files: Record<string, Uint8Array>;
	cwd: string;
	environment: Record<string, string>;
	runtimeVersions: Record<string, string>;
	installedPackages: Record<string, string[]>;
	interpreterState: Record<string, Record<string, JSONValue>>;
	networkPolicy: SandboxNetworkPolicy;
	createdAtMs: number;
}

export interface SandboxBackend {
	execute(
		sessionId: string,
		command: SandboxCommand,
		context: {
			workspace: WorkspaceFiles;
			limits: SandboxResourceLimits;
			environment: Record<string, string>;
			networkPolicy: SandboxNetworkPolicy;
			/** Aborted on timeout/cancellation; backends must stop the command. */
			signal?: AbortSignal;
		},
	): Promise<SandboxCommandResult>;
	/** Release per-session resources such as cloud sandboxes (optional). */
	closeSession?(sessionId: string): Promise<boolean>;
}

export type SandboxCommandRunner = (
	sessionId: string,
	command: SandboxCommand,
	workspace: WorkspaceFiles,
	limits: SandboxResourceLimits,
	environment: Record<string, string>,
	networkPolicy: SandboxNetworkPolicy,
) => Promise<SandboxCommandResult>;

export class CallbackSandboxBackend implements SandboxBackend {
	constructor(readonly runner: SandboxCommandRunner) {}

	execute(
		sessionId: string,
		command: SandboxCommand,
		context: {
			workspace: WorkspaceFiles;
			limits: SandboxResourceLimits;
			environment: Record<string, string>;
			networkPolicy: SandboxNetworkPolicy;
		},
	): Promise<SandboxCommandResult> {
		return this.runner(
			sessionId,
			command,
			context.workspace,
			context.limits,
			context.environment,
			context.networkPolicy,
		);
	}
}

export type SandboxBackendName =
	'native' | 'docker' | 'e2b' | 'microsandbox' | 'swe-rex';

export const SANDBOX_BACKEND_ENV = 'AGENT_RT_SANDBOX_BACKEND';

const invalidSandboxBackendNames = new Set([
	'',
	'0',
	'disable',
	'disabled',
	'false',
	'nil',
	'no',
	'none',
	'null',
	'off',
]);

export function normalizedSandboxBackendName(
	value: string | undefined,
): SandboxBackendName {
	if (value === undefined) return 'native';
	const normalized = value.trim().toLowerCase();
	if (invalidSandboxBackendNames.has(normalized)) {
		throw new Error(
			SANDBOX_BACKEND_ENV +
				' must name an enabled sandbox backend; sandboxing cannot be disabled',
		);
	}
	if (
		normalized !== 'native' &&
		normalized !== 'docker' &&
		normalized !== 'e2b' &&
		normalized !== 'microsandbox' &&
		normalized !== 'swe-rex'
	) {
		throw new Error(
			'unsupported sandbox backend ' +
				JSON.stringify(value) +
				'; expected native, docker, e2b, microsandbox, or swe-rex',
		);
	}
	return normalized;
}

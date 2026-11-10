/** Tool definitions, side-effect classes, execution limits, and tool selection policy. */
import { runtimeEnvironment } from '../platform.js';

export type ToolSideEffect =
	'none' | 'read' | 'write' | 'reversible' | 'consequential' | 'destructive';

export type ToolErrorBehavior = 'raise' | 'return_error';

export type ToolExecutionMode = 'parallel' | 'sequential';

export type ToolHandler = (
	argumentsValue: Record<string, unknown>,
	signal?: AbortSignal,
) => Promise<unknown>;

export const EXECUTION_TIMEOUT_ENV = 'AGENT_RT_EXECUTION_TIMEOUT_SECONDS';

export const EXECUTION_MEMORY_ENV = 'AGENT_RT_EXECUTION_MEMORY_BYTES';

export const EXECUTION_CPU_ENV = 'AGENT_RT_EXECUTION_CPU_SECONDS';

export interface ToolExecutionLimits {
	timeoutMs?: number;
	memoryBytes?: number;
	cpuSeconds?: number;
}

function parseExecutionEnvNumber(
	env: Record<string, string | undefined>,
	name: string,
	integer = false,
): number | undefined {
	const raw = env[name];
	if (raw === undefined) return undefined;
	if (!raw.trim()) {
		throw new Error(name + ' must not be blank');
	}
	const value = Number(raw.trim());
	if (
		!Number.isFinite(value) ||
		value < 0 ||
		(integer && !Number.isInteger(value))
	) {
		throw new Error(
			name +
				(integer
					? ' must be a non-negative integer'
					: ' must be a non-negative number'),
		);
	}
	return value;
}

export function executionLimitsFromEnv(
	env: Record<string, string | undefined> = runtimeEnvironment(),
): ToolExecutionLimits {
	const timeoutSeconds = parseExecutionEnvNumber(env, EXECUTION_TIMEOUT_ENV);
	const memoryBytes = parseExecutionEnvNumber(
		env,
		EXECUTION_MEMORY_ENV,
		true,
	);
	const cpuSeconds = parseExecutionEnvNumber(env, EXECUTION_CPU_ENV);
	return {
		...(timeoutSeconds === undefined
			? {}
			: { timeoutMs: timeoutSeconds * 1000 }),
		...(memoryBytes === undefined ? {} : { memoryBytes }),
		...(cpuSeconds === undefined ? {} : { cpuSeconds }),
	};
}

export interface ToolExecutionContext {
	services: Record<string, unknown>;
	requestContext: Record<string, unknown>;
	signal?: AbortSignal;
	limits: ToolExecutionLimits;
}

export type ContextualToolHandler = (
	argumentsValue: Record<string, unknown>,
	context: ToolExecutionContext,
) => Promise<unknown>;

export interface ToolDefinition {
	name: string;
	description: string;
	inputSchema: Record<string, unknown>;
	outputSchema?: Record<string, unknown>;
	metadata?: Record<string, unknown>;
	sideEffect?: ToolSideEffect;
	errorBehavior?: ToolErrorBehavior;
	timeoutMs?: number;
	memoryBytes?: number;
	cpuSeconds?: number;
	executionMode?: ToolExecutionMode;
}

export function effectiveToolExecutionLimits(
	definition?: ToolDefinition,
): ToolExecutionLimits {
	const defaults = executionLimitsFromEnv();
	return {
		timeoutMs: definition?.timeoutMs ?? defaults.timeoutMs,
		memoryBytes: definition?.memoryBytes ?? defaults.memoryBytes,
		cpuSeconds: definition?.cpuSeconds ?? defaults.cpuSeconds,
	};
}

export function validateToolDefinition(tool: ToolDefinition): void {
	if (!tool.name.trim()) throw new Error('tool name must not be empty');
	if (!tool.description.trim())
		throw new Error('tool description must not be empty');
	const inputType = tool.inputSchema.type;
	if (inputType !== undefined && inputType !== 'object') {
		throw new Error('tool inputSchema must describe an object');
	}
	if (
		tool.sideEffect !== undefined &&
		![
			'none',
			'read',
			'write',
			'reversible',
			'consequential',
			'destructive',
		].includes(tool.sideEffect)
	) {
		throw new Error(`unsupported tool side effect: ${tool.sideEffect}`);
	}
	if (
		tool.errorBehavior !== undefined &&
		!['raise', 'return_error'].includes(tool.errorBehavior)
	) {
		throw new Error(
			`unsupported tool error behavior: ${tool.errorBehavior}`,
		);
	}
	if (tool.timeoutMs !== undefined && tool.timeoutMs < 0) {
		throw new Error('tool timeoutMs must be non-negative');
	}
	if (tool.memoryBytes !== undefined && tool.memoryBytes < 0) {
		throw new Error('tool memoryBytes must be non-negative');
	}
	if (tool.cpuSeconds !== undefined && tool.cpuSeconds < 0) {
		throw new Error('tool cpuSeconds must be non-negative');
	}
	if (
		tool.executionMode !== undefined &&
		!['parallel', 'sequential'].includes(tool.executionMode)
	) {
		throw new Error(
			`unsupported tool execution mode: ${tool.executionMode}`,
		);
	}
}

export interface ToolSelectionPolicy {
	allowed?: string[];
	denied?: string[];
	required?: string[];
	preferred?: string[];
}

export interface ToolSelectionRequirement {
	required?: string[];
	preferred?: string[];
}

export function normalizeToolSelectionPolicy(
	policy?: ToolSelectionPolicy,
): ToolSelectionPolicy | undefined {
	if (!policy) return undefined;
	const allowed = policy.allowed ? [...new Set(policy.allowed)] : undefined;
	const denied = [...new Set(policy.denied ?? [])];
	const required = [...new Set(policy.required ?? [])];
	const preferred = [...new Set(policy.preferred ?? [])];

	for (const name of required) {
		if (denied.includes(name)) {
			throw new Error(`required tools cannot also be denied: ${name}`);
		}
		if (allowed && !allowed.includes(name)) {
			throw new Error(`required tools must be allowed: ${name}`);
		}
	}
	return { allowed, denied, required, preferred };
}

export function toolPolicyPermits(
	policy: ToolSelectionPolicy | undefined,
	name: string,
): boolean {
	if (!policy) return true;
	if ((policy.denied ?? []).includes(name)) return false;
	return policy.allowed === undefined || policy.allowed.includes(name);
}

/** Programmatic tool execution with per-call timeouts. */
import { type ToolCall } from '../types.js';
import { effectiveToolExecutionLimits } from './definitions.js';
import { type ToolRegistry } from './registry.js';

export class ToolTimeoutError extends Error {
	readonly toolName: string;
	readonly timeoutMs: number;

	constructor(toolName: string, timeoutMs: number) {
		super(`tool ${toolName} exceeded timeout of ${timeoutMs} ms`);
		this.name = 'ToolTimeoutError';
		this.toolName = toolName;
		this.timeoutMs = timeoutMs;
	}
}

export interface ToolProgramResult {
	call: ToolCall;
	value: unknown;
}

export class ToolProgramExecutor {
	constructor(private readonly registry: ToolRegistry) {}

	async execute(
		calls: ToolCall[],
		options: {
			concurrent?: boolean;
			signal?: AbortSignal;
			requestContext?: Record<string, unknown>;
		} = {},
	): Promise<ToolProgramResult[]> {
		const runOne = (call: ToolCall) =>
			this.executeOne(call, options.signal, options.requestContext);
		const values = options.concurrent
			? await Promise.all(calls.map(runOne))
			: await this.executeSequentially(calls, runOne);

		return calls.map((call, index) => ({
			call,
			value: values[index],
		}));
	}

	private async executeSequentially(
		calls: ToolCall[],
		runOne: (call: ToolCall) => Promise<unknown>,
	): Promise<unknown[]> {
		const values: unknown[] = [];
		for (const call of calls) values.push(await runOne(call));
		return values;
	}

	private async executeOne(
		call: ToolCall,
		signal?: AbortSignal,
		requestContext: Record<string, unknown> = {},
	): Promise<unknown> {
		if (signal?.aborted) throw signal.reason ?? new Error('run_cancelled');
		const registered = this.registry.get(call.name);
		const timeoutMs = effectiveToolExecutionLimits(
			registered.definition,
		).timeoutMs;
		if (timeoutMs !== undefined && timeoutMs <= 0) {
			throw new ToolTimeoutError(call.name, timeoutMs);
		}
		if (timeoutMs === undefined) {
			return this.registry.execute(call, signal, requestContext);
		}

		const controller = new AbortController();
		let parentAbort: (() => void) | undefined;
		if (signal) {
			parentAbort = () => controller.abort(signal.reason);
			signal.addEventListener('abort', parentAbort, { once: true });
		}

		let timer: ReturnType<typeof setTimeout> | undefined;
		try {
			return await Promise.race([
				this.registry.execute(call, controller.signal, requestContext),
				new Promise<unknown>((_, reject) => {
					timer = setTimeout(() => {
						const error = new ToolTimeoutError(
							call.name,
							timeoutMs,
						);
						controller.abort(error);
						reject(error);
					}, timeoutMs);
				}),
			]);
		} catch (error) {
			if (
				controller.signal.aborted &&
				!signal?.aborted &&
				!(error instanceof ToolTimeoutError)
			) {
				throw new ToolTimeoutError(call.name, timeoutMs);
			}
			throw error;
		} finally {
			if (timer !== undefined) clearTimeout(timer);
			if (signal && parentAbort) {
				signal.removeEventListener('abort', parentAbort);
			}
		}
	}
}

export function toolTimeoutPayload(
	call: ToolCall,
	timeoutMs: number,
): Record<string, unknown> {
	return {
		error: {
			type: 'tool_timeout',
			tool: call.name,
			timeoutMs,
		},
	};
}

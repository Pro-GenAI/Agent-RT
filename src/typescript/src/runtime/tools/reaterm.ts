/** ReaTerm read-only tool; the host application chooses the root. */
import { type RegisteredTool, ToolRegistry } from './registry.js';
import { type ToolDefinition } from './definitions.js';
import { importOptionalPeer } from '../platform.js';

export const REATERM_TOOL_NAME = 'reaterm';
const READ_COMMANDS = ['pwd', 'ls', 'cat', 'head', 'tail', 'wc', 'grep', 'stat'] as const;

export interface ReaTermClient {
  run(command: string, ...args: string[]): Promise<string>;
}

export interface ReaTermToolOptions {
  root: string;
  binary?: string;
  timeoutMs?: number;
  /** Trusted injection for deterministic testing and custom native clients. */
  client?: ReaTermClient;
}

/** Register an opt-in read-only tool; requires reaterm@0.0.1 and its native CLI. */
export async function registerReaTermTool(
  registry: ToolRegistry,
  options: ReaTermToolOptions,
): Promise<RegisteredTool> {
  if (typeof options?.root !== 'string' || !options.root.trim()) {
    throw new Error('reaterm root must be a nonempty application-selected path');
  }
  const timeoutMs = options.timeoutMs ?? 15000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error('reaterm timeoutMs must be finite and positive');
  }

  let client = options.client;
  if (!client) {
    let ReaTerm: new (options: { root: string; binary?: string; timeoutMs: number }) => ReaTermClient;
    try {
      ({ ReaTerm } = await importOptionalPeer<{ ReaTerm: typeof ReaTerm }>('reaterm'));
    } catch (error) {
      throw new Error('Install reaterm@0.0.1 and the native ReaTerm CLI', { cause: error });
    }
    client = new ReaTerm({ root: options.root, binary: options.binary, timeoutMs });
  }
  const definition: ToolDefinition = {
    name: REATERM_TOOL_NAME,
    description:
      'Read-only filesystem inspection through ReaTerm: pwd, ls, cat, head, tail, wc, grep, stat. No shell or writes.',
    inputSchema: {
      type: 'object',
      properties: {
        command: { type: 'string', enum: [...READ_COMMANDS] },
        args: { type: 'array', items: { type: 'string' } },
      },
      required: ['command'],
      additionalProperties: false,
    },
    sideEffect: 'read',
    timeoutMs: timeoutMs + 1000,
  };
  const terminal = client;
  return registry.register(definition, {
    handler: async (argumentsValue) => {
      const command = argumentsValue.command;
      const argv = argumentsValue.args ?? [];
      if (typeof command !== 'string' || !READ_COMMANDS.includes(command as typeof READ_COMMANDS[number])) {
        throw new Error('reaterm command must be a supported read-only command');
      }
      if (!Array.isArray(argv) || !argv.every((arg) => typeof arg === 'string')) {
        throw new Error('reaterm args must be an array of strings');
      }
      return terminal.run(command, ...argv);
    },
  });
}

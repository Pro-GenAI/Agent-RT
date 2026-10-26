/**
 * Memory packages (mem0, Zep, Letta, Supermemory) behind one interface.
 *
 * `loadMemory()` returns a `MemoryBackend` that can go straight into
 * `AgentConfig.memory` (it is searched with the user's prompt every run) and
 * whose `remember(messages)` stores a finished conversation. Each package is
 * an optional peer dependency, imported only when its backend is loaded. memU
 * is a Python-only package.
 */
import {
	type AgentRTConfig,
	DEFAULT_MEMORY_BACKEND,
	MEMORY_BACKEND_OPTIONS,
	requireResolved,
} from './config.js';

type Options = Record<string, unknown>;
type Turn = { role: 'user' | 'assistant'; content: string };
type Recall = (query: string, limit: number) => Promise<unknown[]>;
type Remember = (turns: Turn[]) => Promise<unknown>;

// backend -> npm package to install
const MEMORY_PACKAGES: Record<string, string> = {
	mem0: 'mem0ai',
	zep: '@getzep/zep-cloud',
	letta: '@letta-ai/letta-client',
	supermemory: 'supermemory',
};

export class MemoryBackend {
	constructor(
		readonly name: string,
		readonly client: unknown,
		private readonly recallFn: Recall,
		private readonly rememberFn: Remember,
	) {}

	async recall(query: string, limit = 5): Promise<unknown[]> {
		return (await this.recallFn(query, limit)).slice(0, limit);
	}

	/** Store user/assistant turns (Agent RT messages, role/content objects or text). */
	async remember(messages: unknown): Promise<unknown> {
		const turns = conversation(messages);
		return turns.length ? await this.rememberFn(turns) : undefined;
	}
}

function conversation(messages: unknown): Turn[] {
	const list =
		typeof messages === 'string'
			? [{ role: 'user', content: messages }]
			: (messages as unknown[]);
	const turns: Turn[] = [];
	for (const message of list) {
		const { role, content } = message as {
			role?: unknown;
			content?: unknown;
		};
		const text = Array.isArray(content)
			? content
					.map((part) => (part as { text?: string }).text ?? '')
					.join('')
			: content;
		// System prompts and tool output are not conversation memory.
		if (
			(role === 'user' || role === 'assistant') &&
			typeof text === 'string'
		)
			if (text.trim()) turns.push({ role, content: text });
	}
	return turns;
}

function transcript(turns: Turn[]): string {
	return turns.map((turn) => turn.role + ': ' + turn.content).join('\n');
}

function required(options: Options, name: string, backend: string): string {
	const value = options[name];
	if (typeof value !== 'string' || !value)
		throw new Error(
			`memory backend '${backend}' requires option '${name}'`,
		);
	return value;
}

function credentials(
	options: Options,
	baseKey: string,
): Record<string, unknown> {
	const result: Record<string, unknown> = {};
	if (options['api_key']) result['apiKey'] = options['api_key'];
	if (options['base_url']) result[baseKey] = options['base_url'];
	return result;
}

/** Loaded through a variable specifier so bundlers neither inline nor require it. */
async function importPeer<T>(name: string): Promise<T> {
	const specifier = name;
	return (await import(specifier)) as T;
}

type Builder = (
	options: Options,
	load: <T>(name: string) => Promise<T>,
) => Promise<MemoryBackend>;

const BUILDERS: Record<string, Builder> = {
	async mem0(options, load) {
		let client: any; // eslint-disable-line @typescript-eslint/no-explicit-any
		if (options['api_key']) {
			const { MemoryClient } = await load<any>('mem0ai'); // eslint-disable-line @typescript-eslint/no-explicit-any
			client = new MemoryClient({
				apiKey: options['api_key'],
				...(options['host'] ? { host: options['host'] } : {}),
			});
		} else {
			const { Memory } = await load<any>('mem0ai/oss'); // eslint-disable-line @typescript-eslint/no-explicit-any
			client = new Memory((options['config'] as Options) ?? {});
		}
		const filters: Record<string, string> = {};
		const entity: Record<string, string> = {};
		for (const [key, camel] of [
			['user_id', 'userId'],
			['agent_id', 'agentId'],
			['run_id', 'runId'],
		]) {
			const value = options[key];
			if (typeof value === 'string' && value) {
				filters[key] = value;
				entity[camel] = value;
			}
		}
		if (!Object.keys(filters).length) {
			filters['user_id'] = 'default';
			entity['userId'] = 'default';
		}
		return new MemoryBackend(
			'mem0',
			client,
			async (query, limit) => {
				const found = await client.search(query, {
					filters,
					topK: limit,
				});
				const results = Array.isArray(found)
					? found
					: (found?.results ?? []);
				return results.map(
					(item: { memory?: string }) => item.memory ?? item,
				);
			},
			async (turns) => await client.add(turns, entity),
		);
	},

	async memu() {
		throw new Error(
			"memory backend 'memu' is Python-only: use it from the Python package (pip install 'agent-rt[memu]')",
		);
	},

	async zep(options, load) {
		const { ZepClient } = await load<any>('@getzep/zep-cloud'); // eslint-disable-line @typescript-eslint/no-explicit-any
		const client = new ZepClient(credentials(options, 'baseUrl'));
		return new MemoryBackend(
			'zep',
			client,
			async (query, limit) => {
				const graphId = required(options, 'graph_id', 'zep');
				const page = await client.graph.searchEdges(graphId, {
					query,
					pageSize: limit,
				});
				return (page.data ?? []).map(
					(edge: { fact?: string }) => edge.fact,
				);
			},
			async (turns) =>
				await client.thread.addMessages(
					required(options, 'thread_id', 'zep'),
					{ messages: turns },
				),
		);
	},

	async letta(options, load) {
		const { Letta } = await load<any>('@letta-ai/letta-client'); // eslint-disable-line @typescript-eslint/no-explicit-any
		const client = new Letta(credentials(options, 'baseURL'));
		const agentId = required(options, 'agent_id', 'letta');
		return new MemoryBackend(
			'letta',
			client,
			async (query, limit) => {
				const found = await client.agents.passages.search(agentId, {
					query,
					top_k: limit,
				});
				return found.results.map(
					(result: { content?: string }) => result.content,
				);
			},
			async (turns) =>
				await client.agents.passages.create(agentId, {
					text: transcript(turns),
				}),
		);
	},

	async supermemory(options, load) {
		const { Supermemory } = await load<any>('supermemory'); // eslint-disable-line @typescript-eslint/no-explicit-any
		const client = new Supermemory(credentials(options, 'baseUrl'));
		const namespace = (options['namespace'] as string) || 'default';
		return new MemoryBackend(
			'supermemory',
			client,
			async (query, limit) => {
				const found = await client.search(namespace, { query, limit });
				return found.results.map(
					(result: { memory?: string; chunk?: string }) =>
						result.memory ?? result.chunk,
				);
			},
			async (turns) =>
				await client.add(namespace, { content: transcript(turns) }),
		);
	},
};

export interface LoadMemoryOptions {
	backend?: string;
	options?: Options;
	/** For tests: replaces the optional-peer import. */
	importer?: <T>(name: string) => Promise<T>;
}

/**
 * Load a memory package: `backend`/`options` win over the config's
 * `agent_rt_settings.memory`; the default backend is mem0.
 */
export async function loadMemory(
	config?: AgentRTConfig,
	request: LoadMemoryOptions = {},
): Promise<MemoryBackend> {
	const settings = (config?.settings['memory'] ?? {}) as {
		backend?: string;
		options?: Options;
	};
	const name = request.backend ?? settings.backend ?? DEFAULT_MEMORY_BACKEND;
	const allowed = MEMORY_BACKEND_OPTIONS[name];
	if (!Object.prototype.hasOwnProperty.call(BUILDERS, name) || !allowed)
		throw new Error(
			`unknown memory backend '${name}'; expected one of: ${Object.keys(MEMORY_BACKEND_OPTIONS).join(', ')}`,
		);
	const options = { ...(request.options ?? settings.options ?? {}) };
	const unknown = Object.keys(options)
		.filter((key) => !allowed.includes(key))
		.sort();
	if (unknown.length)
		throw new Error(
			`memory backend '${name}': unknown option(s): ${unknown.join(', ')}`,
		);
	requireResolved(options);
	const load = request.importer ?? importPeer;
	const guarded = async <T>(specifier: string): Promise<T> => {
		try {
			return await load<T>(specifier);
		} catch (error) {
			const code = (error as { code?: string }).code;
			if (code === 'MODULE_NOT_FOUND' || code === 'ERR_MODULE_NOT_FOUND')
				throw new Error(
					`memory backend '${name}' is not installed: npm install ${MEMORY_PACKAGES[name]}`,
				);
			throw error;
		}
	};
	return await BUILDERS[name](options, guarded);
}

/**
 * `deepagents` `createDeepAgent` over the LangChain compatibility agent loop.
 *
 * Deep agents add planning (`write_todos`), a filesystem (`ls`, `read_file`,
 * `write_file`, `edit_file`, `glob`, `grep`), and subagent delegation (`task`)
 * to a LangChain agent. Like the upstream default `StateBackend`, the
 * filesystem is virtual: files live in agent state (`result.files`) and never
 * touch the host. Sandboxed `execute`, pluggable backends, skills, memory
 * files, permissions, and interrupts are not emulated; passing them throws
 * instead of silently running without them.
 */
import {
	AgentRunnable,
	type AgentState,
	type ChatModel,
	initChatModel,
	tool,
} from './langchain.js';

type JSONObject = Record<string, unknown>;
type Runtime = { state: JSONObject };
type FileData = {
	content: string;
	encoding: string;
	created_at?: string;
	modified_at?: string;
};

export const BASE_AGENT_PROMPT =
	'You are a capable agent. Plan multi-step work with write_todos and keep it ' +
	'current. Use the file tools (ls, read_file, write_file, edit_file, glob, ' +
	"grep) on the agent's virtual filesystem; paths are absolute, such as " +
	'/notes.md. Delegate self-contained work to a subagent with task.';

export type SubAgent = {
	name: string;
	description: string;
	systemPrompt?: string;
	tools?: unknown[];
	model?: ChatModel | string;
	middleware?: unknown[];
};

const GENERAL_PURPOSE_SUBAGENT: SubAgent = {
	name: 'general-purpose',
	description:
		'General-purpose agent for researching complex questions and executing ' +
		'multi-step tasks; it has the same tools as the main agent.',
};
const UNSUPPORTED = [
	'backend',
	'interruptOn',
	'skills',
	'memory',
	'permissions',
] as const;
const SUBAGENT_KEYS = new Set([
	'name',
	'description',
	'systemPrompt',
	'tools',
	'model',
	'middleware',
]);
const MAX_READ_LINES = 2000;

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function prompt(text?: string): string {
	return text ? `${text}\n\n${BASE_AGENT_PROMPT}` : BASE_AGENT_PROMPT;
}

function normalize(value: unknown): string {
	const parts: string[] = [];
	for (const part of String(value || '/').split('/')) {
		if (part === '..') parts.pop();
		else if (part && part !== '.') parts.push(part);
	}
	return `/${parts.join('/')}`;
}

function contentOf(entry: unknown): string {
	if (!isObject(entry)) return String(entry);
	// Older deepagents releases stored content as a list of lines.
	return Array.isArray(entry.content)
		? entry.content.join('\n')
		: String(entry.content ?? '');
}

function fileData(content: string, createdAt?: string): FileData {
	const now = new Date().toISOString();
	return {
		content,
		encoding: 'utf-8',
		created_at: createdAt ?? now,
		modified_at: now,
	};
}

function files(runtime: Runtime): JSONObject {
	return { ...(isObject(runtime.state.files) ? runtime.state.files : {}) };
}

function under(path: string, base: string): boolean {
	return base === '/' || path === base || path.startsWith(`${base}/`);
}

function relative(path: string, base: string): string {
	return path.slice(base.replace(/\/$/, '').length + 1);
}

function globRegex(pattern: string): RegExp {
	let source = '';
	for (let i = 0; i < pattern.length;) {
		if (pattern.startsWith('**/', i)) {
			source += '(?:.*/)?';
			i += 3;
		} else if (pattern.startsWith('**', i)) {
			source += '.*';
			i += 2;
		} else {
			const char = pattern[i++];
			source +=
				char === '*'
					? '[^/]*'
					: char === '?'
						? '[^/]'
						: char.replace(/[.+^${}()|[\]\\]/g, '\\$&');
		}
	}
	return new RegExp(`^${source}$`);
}

function schema(
	properties: JSONObject,
	required: string[] = Object.keys(properties),
): JSONObject {
	return { type: 'object', properties, required };
}

const str = { type: 'string' };
type Input = JSONObject & { runtime: Runtime };

function fileTools(): unknown[] {
	return [
		tool(
			({ todos, runtime }: Input) => {
				runtime.state.todos = Array.isArray(todos)
					? todos.map((item) => ({ ...(item as JSONObject) }))
					: [];
				return `Updated todo list to ${JSON.stringify(runtime.state.todos)}`;
			},
			{
				name: 'write_todos',
				description:
					'Replace the todo list. Each todo is {"content": string, "status": "pending" | "in_progress" | "completed"}.',
				schema: schema({
					todos: { type: 'array', items: { type: 'object' } },
				}),
			},
		),
		tool(
			({ path, runtime }: Input) => {
				const base = normalize(path);
				const entries = new Set<string>();
				for (const name of Object.keys(files(runtime)).map(normalize)) {
					if (name === base || !under(name, base)) continue;
					const [head, ...rest] = relative(name, base).split('/');
					entries.add(
						`${base.replace(/\/$/, '')}/${head}${rest.length ? '/' : ''}`,
					);
				}
				return [...entries].sort();
			},
			{
				name: 'ls',
				description:
					'List files and directories directly under an absolute directory path.',
				schema: schema({ path: str }, []),
			},
		),
		tool(
			({
				file_path,
				offset = 0,
				limit = MAX_READ_LINES,
				runtime,
			}: Input) => {
				const path = normalize(file_path);
				const all = files(runtime);
				if (!(path in all)) return `Error: File '${path}' not found`;
				const lines = contentOf(all[path]).split(/\r?\n/);
				if (lines[lines.length - 1] === '') lines.pop();
				const start = Math.max(0, Number(offset));
				const window = lines.slice(
					start,
					start + Math.max(0, Number(limit)),
				);
				if (!window.length) {
					return lines.length
						? `Error: Line offset ${start} exceeds file length (${lines.length} lines)`
						: 'System reminder: File exists but has empty contents';
				}
				return window
					.map(
						(line, i) =>
							`${String(start + i + 1).padStart(6)}\t${line}`,
					)
					.join('\n');
			},
			{
				name: 'read_file',
				description:
					'Read a file from the virtual filesystem with cat -n style line numbers.',
				schema: schema(
					{
						file_path: str,
						offset: { type: 'integer' },
						limit: { type: 'integer' },
					},
					['file_path'],
				),
			},
		),
		tool(
			({ file_path, content, runtime }: Input) => {
				const path = normalize(file_path);
				const all = files(runtime);
				if (path in all) {
					return `Cannot write to ${path} because it already exists. Read and then make an edit, or write to a new path.`;
				}
				all[path] = fileData(String(content));
				runtime.state.files = all;
				return `Updated file ${path}`;
			},
			{
				name: 'write_file',
				description:
					'Create a new file. Existing files must be changed with edit_file.',
				schema: schema({ file_path: str, content: str }),
			},
		),
		tool(
			({
				file_path,
				old_string,
				new_string,
				replace_all,
				runtime,
			}: Input) => {
				const path = normalize(file_path);
				const all = files(runtime);
				if (!(path in all)) return `Error: File '${path}' not found`;
				const oldText = String(old_string ?? '');
				const content = contentOf(all[path]);
				const count = oldText ? content.split(oldText).length - 1 : 0;
				if (!count)
					return `Error: String not found in file: '${oldText}'`;
				if (count > 1 && !replace_all) {
					return `Error: String '${oldText}' appears ${count} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context.`;
				}
				const newText = String(new_string ?? '');
				const updated = replace_all
					? content.split(oldText).join(newText)
					: content.replace(oldText, () => newText);
				const previous = all[path];
				all[path] = fileData(
					updated,
					isObject(previous)
						? (previous.created_at as string)
						: undefined,
				);
				runtime.state.files = all;
				return `Successfully replaced ${replace_all ? count : 1} instance(s) in '${path}'`;
			},
			{
				name: 'edit_file',
				description:
					'Replace exact text in a file; old_string must be unique unless replace_all is true.',
				schema: schema(
					{
						file_path: str,
						old_string: str,
						new_string: str,
						replace_all: { type: 'boolean' },
					},
					['file_path', 'old_string', 'new_string'],
				),
			},
		),
		tool(
			({ pattern, path, runtime }: Input) => {
				const base = normalize(path);
				const regex = globRegex(String(pattern));
				return Object.keys(files(runtime))
					.map(normalize)
					.filter(
						(name) =>
							under(name, base) &&
							regex.test(relative(name, base)),
					)
					.sort();
			},
			{
				name: 'glob',
				description:
					'Find file paths under a directory that match a glob pattern such as **/*.ts.',
				schema: schema({ pattern: str, path: str }, ['pattern']),
			},
		),
		tool(
			({ pattern, path, glob, output_mode, runtime }: Input) => {
				const base = normalize(path);
				const needle = String(pattern);
				const filter = glob ? globRegex(String(glob)) : undefined;
				const hits: Array<[string, Array<[number, string]>]> = [];
				for (const [raw, entry] of Object.entries(
					files(runtime),
				).sort()) {
					const name = normalize(raw);
					if (!under(name, base)) continue;
					if (
						filter &&
						!filter.test(name.slice(name.lastIndexOf('/') + 1))
					)
						continue;
					const lines = contentOf(entry)
						.split(/\r?\n/)
						.map((line, i): [number, string] => [i + 1, line])
						.filter(([, line]) => line.includes(needle));
					if (lines.length) hits.push([name, lines]);
				}
				if (!hits.length)
					return `No matches found for pattern '${needle}'`;
				if (output_mode === 'content') {
					return hits
						.flatMap(([name, lines]) =>
							lines.map(([n, line]) => `${name}:${n}:${line}`),
						)
						.join('\n');
				}
				if (output_mode === 'count') {
					return hits
						.map(([name, lines]) => `${name}:${lines.length}`)
						.join('\n');
				}
				return hits.map(([name]) => name).join('\n');
			},
			{
				name: 'grep',
				description:
					'Search file contents for a literal string. output_mode: files_with_matches, content, or count.',
				schema: schema(
					{
						pattern: str,
						path: str,
						glob: str,
						output_mode: {
							type: 'string',
							enum: ['files_with_matches', 'content', 'count'],
						},
					},
					['pattern'],
				),
			},
		),
	];
}

/** `invoke()` result: LangChain agent state plus the deep agent's files and todos. */
export type DeepAgentState = AgentState & {
	files: JSONObject;
	todos: JSONObject[];
};

export class DeepAgent extends AgentRunnable {
	// shortcut: only invoke() maps top-level files/todos; stream() callers pass them in `state`.
	override async invoke(
		input: unknown,
		config: Record<string, unknown> = {},
	): Promise<DeepAgentState> {
		let next = input;
		if (isObject(input) && ('files' in input || 'todos' in input)) {
			next = {
				...input,
				state: {
					...(isObject(input.state) ? input.state : {}),
					...('files' in input ? { files: input.files } : {}),
					...('todos' in input ? { todos: input.todos } : {}),
				},
			};
		}
		const result = await super.invoke(next, config);
		const state = result.state ?? {};
		return {
			...result,
			files: { ...(isObject(state.files) ? state.files : {}) },
			todos: Array.isArray(state.todos) ? [...state.todos] : [],
		};
	}
}

function taskTool(subagents: SubAgent[], build: (spec: SubAgent) => DeepAgent) {
	const agents = new Map<string, DeepAgent>();
	const names = subagents.map((item) => item.name);
	return tool(
		async ({ description, subagent_type, runtime }: Input) => {
			const spec = subagents.find((item) => item.name === subagent_type);
			if (!spec) {
				return `Error: unknown subagent_type '${String(subagent_type)}'; use one of: ${names.join(', ')}`;
			}
			if (!agents.has(spec.name)) agents.set(spec.name, build(spec));
			// Isolated like upstream: the subagent sees only the task, and shares files.
			const result = await agents.get(spec.name)!.invoke({
				messages: [{ role: 'user', content: String(description) }],
				files: isObject(runtime.state.files) ? runtime.state.files : {},
			});
			runtime.state.files = result.files;
			return result.messages[result.messages.length - 1]?.content ?? '';
		},
		{
			name: 'task',
			description:
				'Delegate a self-contained task to an isolated subagent and get back its final answer. Available subagent_type values:\n' +
				subagents
					.map((item) => `- ${item.name}: ${item.description}`)
					.join('\n'),
			schema: schema({
				description: str,
				subagent_type: { type: 'string', enum: names },
			}),
		},
	);
}

export type CreateDeepAgentParams = {
	model?: ChatModel | string;
	tools?: unknown[];
	systemPrompt?: string;
	middleware?: unknown[];
	subagents?: SubAgent[];
	responseFormat?: unknown;
	contextSchema?: unknown;
	stateSchema?: unknown;
	checkpointer?: unknown;
	store?: unknown;
	name?: string;
	maxIterations?: number;
} & Partial<Record<(typeof UNSUPPORTED)[number], unknown>>;

export function createDeepAgent(params: CreateDeepAgentParams = {}): DeepAgent {
	const unsupported = UNSUPPORTED.filter((key) => params[key] !== undefined);
	if (unsupported.length) {
		throw new Error(
			`deepagents compatibility does not support ${unsupported.join(', ')}; files live in the virtual state filesystem`,
		);
	}
	if (!params.model) {
		throw new Error(
			"createDeepAgent requires model (for example 'openai:gpt-4.1')",
		);
	}
	const model =
		typeof params.model === 'string'
			? initChatModel(params.model)
			: params.model;
	const userTools = [...(params.tools ?? [])];
	const maxIterations = params.maxIterations ?? 50;
	let specs: SubAgent[] = [GENERAL_PURPOSE_SUBAGENT];
	for (const spec of params.subagents ?? []) {
		if (!isObject(spec)) {
			throw new Error(
				'deepagents compatibility supports declarative SubAgent objects only',
			);
		}
		const unsupported = Object.keys(spec).filter(
			(key) => !SUBAGENT_KEYS.has(key),
		);
		if (unsupported.length)
			throw new Error(
				`deepagents compatibility does not support subagent options: ${unsupported.sort().join(', ')}`,
			);
		if (typeof spec.name !== 'string' || !spec.name.trim())
			throw new Error('deepagents subagents require a non-empty name');
		if (typeof spec.description !== 'string' || !spec.description.trim())
			throw new Error(
				'deepagents subagents require a non-empty description',
			);
		specs = [...specs.filter((item) => item.name !== spec.name), spec];
	}
	const build = (spec: SubAgent): DeepAgent =>
		new DeepAgent({
			model: spec.model ?? model,
			tools: [...(spec.tools ?? userTools), ...fileTools()],
			systemPrompt: prompt(spec.systemPrompt),
			middleware: (spec.middleware ?? []) as never,
			maxIterations,
		});
	const agent = new DeepAgent({
		model,
		tools: [...userTools, ...fileTools(), taskTool(specs, build)],
		systemPrompt: prompt(params.systemPrompt),
		responseFormat: params.responseFormat,
		middleware: (params.middleware ?? []) as never,
		checkpointer: params.checkpointer as never,
		store: params.store,
		contextSchema: params.contextSchema,
		stateSchema: params.stateSchema,
		maxIterations,
	});
	return Object.assign(agent, { name: params.name });
}

import {
	EnvironmentWebSearchProvider,
	type MCPTool,
	type ModelMessage,
	type ModelProvider,
	type ModelRequest,
	type ModelResponse,
	RetrievalRegistry,
	type RetrievalProvider,
	type RetrievalResult,
	type SandboxCommandResult,
	SandboxSession,
	type ToolCall,
	type ToolDefinition,
} from '../../index.js';

type JSONObject = Record<string, unknown>;

/**
 * A vendor SDK environment setting (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, ...),
 * read when a client is constructed, as the SDKs do; blank values count as unset.
 */
export function sdkEnv(name: string): string | undefined {
	const runtime = globalThis as typeof globalThis & {
		process?: { env?: Record<string, string | undefined> };
	};
	const value = runtime.process?.env?.[name]?.trim();
	return value ? value : undefined;
}

export const CODE_TOOL_NAME = 'agent_rt_code_execution';
export const SHELL_TOOL_NAME = 'agent_rt_shell';
export const FILE_SEARCH_TOOL_NAME = 'agent_rt_file_search';
export const WEB_SEARCH_TOOL_NAME = 'agent_rt_web_search';

export type MCPMigrationClient = {
	listTools(): Promise<MCPTool[]>;
	callTool(
		name: string,
		args?: Record<string, unknown>,
	): Promise<Record<string, unknown>>;
	client?: {
		serverInfo?: unknown;
		initialize?: () => Promise<unknown>;
	};
	serverInfo?: unknown;
	initialize?: () => Promise<unknown>;
};

export type CodeExecutionRecord = {
	id: string;
	kind: 'code' | 'shell';
	arguments: Record<string, unknown>;
	output: SandboxResultPayload;
};

export type MCPExecutionRecord = {
	id: string;
	serverLabel: string;
	toolName: string;
	arguments: Record<string, unknown>;
	output: Record<string, unknown>;
};

export type FileSearchExecutionRecord = {
	id: string;
	query: string;
	results: FileSearchResult[];
};

export type FileSearchResult = {
	file_id: string;
	filename: string;
	text: string;
	score?: number;
	uri?: string;
	attributes: Record<string, unknown>;
	vector_store_id: string;
};

export type WebSearchResult = {
	id: string;
	title: string;
	text: string;
	score?: number;
	url?: string;
	metadata: Record<string, unknown>;
};

export type WebSearchExecutionRecord = {
	id: string;
	query: string;
	results: WebSearchResult[];
};

type SandboxResultPayload = {
	exit_code: number;
	stdout: string;
	stderr: string;
	duration_ms: number;
	truncated: boolean;
};

type MCPBinding = {
	client: MCPMigrationClient;
	serverLabel: string;
	toolName: string;
};

type FileSearchBinding = {
	registry: RetrievalRegistry;
	names: string[];
	maxResults: number;
	filters: Record<string, unknown>;
	scoreThreshold?: number;
};

type WebSearchBinding = {
	provider: RetrievalProvider;
	maxResults: number;
	filters: Record<string, unknown>;
};

export type ExpandedMigrationTools = {
	tools?: ToolDefinition[];
	hasHostedCode: boolean;
	mcpBindings: Map<string, MCPBinding>;
	fileSearchBindings: Map<string, FileSearchBinding>;
	webSearchBindings: Map<string, WebSearchBinding>;
};

function isObject(value: unknown): value is JSONObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hostedCodeKind(tool: unknown): 'code' | 'shell' | undefined {
	if (!isObject(tool)) return undefined;
	const type = typeof tool.type === 'string' ? tool.type : '';
	if (type === 'code_interpreter' || type.startsWith('code_execution_'))
		return 'code';
	if (type === 'shell' || type.startsWith('bash_')) return 'shell';
	return undefined;
}

function mcpToolKind(tool: unknown): 'openai' | 'anthropic' | undefined {
	if (!isObject(tool)) return undefined;
	if (tool.type === 'mcp') return 'openai';
	if (tool.type === 'mcp_toolset') return 'anthropic';
	return undefined;
}

function mcpServerLabel(tool: JSONObject): string {
	if (tool.type === 'mcp') {
		return typeof tool.server_label === 'string' ? tool.server_label : '';
	}
	if (tool.type === 'mcp_toolset') {
		return typeof tool.mcp_server_name === 'string'
			? tool.mcp_server_name
			: '';
	}
	return '';
}

function allowedOpenAITools(tool: JSONObject): Set<string> | undefined {
	let allowed: unknown = tool.allowed_tools;
	if (allowed === undefined && isObject(tool.tool_configuration)) {
		allowed = tool.tool_configuration.allowed_tools;
	}
	if (allowed === undefined || allowed === null) return undefined;
	if (isObject(allowed)) allowed = allowed.tool_names ?? allowed.tools;
	if (Array.isArray(allowed)) return new Set(allowed.map(String));
	// A string or a filter such as {read_only: true} cannot be enforced here;
	// ignoring it would silently expose every tool.
	throw new Error(
		'MCP allowed_tools must be a list of tool names (or {tool_names: [...]}); other filters such as read_only are not supported',
	);
}

function requireMCPApprovalSupported(tool: JSONObject): void {
	if (tool.type !== 'mcp') return;
	const approval = tool.require_approval;
	if (approval === undefined || approval === null || approval === 'never')
		return;
	throw new Error(
		"MCP require_approval is not enforced by compatibility mode; pass require_approval: 'never' explicitly or gate the MCP client with an Agent RT AuthorizedMCPClient policy",
	);
}

function anthropicToolEnabled(tool: JSONObject, name: string): boolean {
	const defaultConfig = isObject(tool.default_config)
		? tool.default_config
		: {};
	const defaultEnabled =
		typeof defaultConfig.enabled === 'boolean'
			? defaultConfig.enabled
			: true;
	const configs = tool.configs;
	let config: JSONObject | undefined;
	if (isObject(configs) && isObject(configs[name])) {
		config = configs[name] as JSONObject;
	} else if (Array.isArray(configs)) {
		const match = configs.find(
			(item) => isObject(item) && String(item.name ?? '') === name,
		);
		if (isObject(match)) config = match;
	}
	return typeof config?.enabled === 'boolean'
		? config.enabled
		: defaultEnabled;
}

function mcpToolEnabled(tool: JSONObject, name: string): boolean {
	if (tool.type === 'mcp') {
		const allowed = allowedOpenAITools(tool);
		return !allowed || allowed.has(name);
	}
	return anthropicToolEnabled(tool, name);
}

function safeName(value: string): string {
	return value.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function mcpFunctionName(serverLabel: string, toolName: string): string {
	return `mcp__${safeName(serverLabel)}__${safeName(toolName)}`;
}

async function ensureMCPInitialized(client: MCPMigrationClient): Promise<void> {
	const inner = client.client ?? client;
	if (
		inner.serverInfo === undefined &&
		typeof inner.initialize === 'function'
	) {
		await inner.initialize();
	}
}

export function validateAnthropicMCPServers(
	mcpServers: unknown,
	tools: unknown,
): void {
	const servers = mcpServers === undefined ? [] : mcpServers;
	if (!Array.isArray(servers))
		throw new TypeError('Anthropic mcp_servers must be an array');
	const toolValues = tools === undefined ? [] : tools;
	if (!Array.isArray(toolValues))
		throw new TypeError('tools must be an array');
	const toolsets = toolValues.filter(
		(tool) => isObject(tool) && mcpToolKind(tool) === 'anthropic',
	) as JSONObject[];
	if (!servers.length && !toolsets.length) return;
	const names = servers.map((server) => {
		if (!isObject(server))
			throw new TypeError(
				'Anthropic mcp_servers entries must be objects',
			);
		const name = typeof server.name === 'string' ? server.name : '';
		if (!name)
			throw new Error('Anthropic MCP server requires a non-empty name');
		return name;
	});
	if (new Set(names).size !== names.length) {
		throw new Error('Anthropic MCP server names must be unique');
	}
	const references = toolsets.map(mcpServerLabel);
	for (const name of names) {
		if (references.filter((reference) => reference === name).length !== 1) {
			throw new Error(
				`Anthropic MCP server ${JSON.stringify(name)} must be referenced by exactly one mcp_toolset`,
			);
		}
	}
	for (const reference of references) {
		if (!names.includes(reference)) {
			throw new Error(
				`Anthropic mcp_toolset references undeclared MCP server ${JSON.stringify(reference)}`,
			);
		}
	}
}

function webSearchTool(tool: unknown): tool is JSONObject {
	if (!isObject(tool)) return false;
	const type = typeof tool.type === 'string' ? tool.type : '';
	return (
		type === 'web_search' ||
		type.startsWith('web_search_preview') ||
		type.startsWith('web_search_')
	);
}

function webSearchBinding(
	tool: JSONObject,
	provider?: RetrievalProvider,
): WebSearchBinding {
	const selected = provider ?? EnvironmentWebSearchProvider.fromEnvironment();
	const rawMax = tool.max_num_results ?? tool.max_results ?? 10;
	const maxResults =
		Number.isInteger(rawMax) && Number(rawMax) >= 1 ? Number(rawMax) : 10;
	const filters: Record<string, unknown> = isObject(tool.filters)
		? { ...tool.filters }
		: {};
	for (const key of [
		'allowed_domains',
		'blocked_domains',
		'user_location',
		'search_context_size',
	]) {
		if (tool[key] !== undefined && tool[key] !== null)
			filters[key] = tool[key];
	}
	return { provider: selected, maxResults, filters };
}

function webSearchToolDefinition(): ToolDefinition {
	return {
		name: WEB_SEARCH_TOOL_NAME,
		description:
			'Search the web using the configured Agent RT web-search provider.',
		inputSchema: {
			type: 'object',
			properties: { query: { type: 'string' } },
			required: ['query'],
		},
	};
}

async function executeWebSearch(
	binding: WebSearchBinding,
	call: ToolCall,
): Promise<{ query: string; results: WebSearchResult[] }> {
	const query = call.arguments.query;
	if (typeof query !== 'string' || !query.trim()) {
		throw new Error('web_search requires a non-empty query');
	}
	const results = await binding.provider.search({
		text: query,
		limit: binding.maxResults,
		filters: binding.filters,
	});
	return {
		query,
		results: results.slice(0, binding.maxResults).map((result) => ({
			id: result.id,
			title: result.title,
			text: retrievalText(result.content),
			...(result.score !== undefined ? { score: result.score } : {}),
			...(result.uri !== undefined ? { url: result.uri } : {}),
			metadata: { ...(result.metadata ?? {}) },
		})),
	};
}

function fileSearchTool(tool: unknown): tool is JSONObject {
	return isObject(tool) && tool.type === 'file_search';
}

function fileSearchBinding(
	tool: JSONObject,
	registry?: RetrievalRegistry,
): FileSearchBinding {
	if (!registry) {
		throw new Error(
			'file_search compatibility requires retrievalRegistry: RetrievalRegistry',
		);
	}
	const rawNames = tool.vector_store_ids;
	if (!Array.isArray(rawNames)) {
		throw new TypeError('file_search vector_store_ids must be an array');
	}
	const names = rawNames.map(String).filter(Boolean);
	if (!names.length)
		throw new Error('file_search requires at least one vector_store_id');
	const rawMax = tool.max_num_results ?? 10;
	if (!Number.isInteger(rawMax) || Number(rawMax) < 1) {
		throw new Error(
			'file_search max_num_results must be a positive integer',
		);
	}
	const filters =
		tool.filters === undefined
			? {}
			: isObject(tool.filters)
				? { ...tool.filters }
				: (() => {
						throw new TypeError(
							'file_search filters must be an object',
						);
					})();
	let scoreThreshold: number | undefined;
	if (
		isObject(tool.ranking_options) &&
		tool.ranking_options.score_threshold !== undefined
	) {
		const value = Number(tool.ranking_options.score_threshold);
		if (!Number.isFinite(value)) {
			throw new TypeError('file_search score_threshold must be numeric');
		}
		scoreThreshold = value;
	}
	return {
		registry,
		names,
		maxResults: Number(rawMax),
		filters,
		scoreThreshold,
	};
}

function fileSearchToolDefinition(): ToolDefinition {
	return {
		name: FILE_SEARCH_TOOL_NAME,
		description:
			'Search files using configured Agent RT retrieval providers.',
		inputSchema: {
			type: 'object',
			properties: { query: { type: 'string' } },
			required: ['query'],
		},
	};
}

function retrievalText(content: unknown): string {
	if (typeof content === 'string') return content;
	try {
		return JSON.stringify(content);
	} catch {
		return String(content);
	}
}

async function executeFileSearch(
	binding: FileSearchBinding,
	call: ToolCall,
): Promise<{ query: string; results: FileSearchResult[] }> {
	const query = call.arguments.query;
	if (typeof query !== 'string' || !query.trim()) {
		throw new Error('file_search requires a non-empty query');
	}
	const merged: Array<{ name: string; result: RetrievalResult }> = [];
	for (const name of binding.names) {
		const results = await binding.registry.search(name, {
			text: query,
			limit: binding.maxResults,
			filters: binding.filters,
		});
		for (const result of results) {
			if (
				binding.scoreThreshold !== undefined &&
				result.score !== undefined &&
				result.score < binding.scoreThreshold
			)
				continue;
			merged.push({ name, result });
		}
	}
	merged.sort((a, b) => {
		const left = a.result.score ?? Number.NEGATIVE_INFINITY;
		const right = b.result.score ?? Number.NEGATIVE_INFINITY;
		return right - left;
	});
	return {
		query,
		results: merged
			.slice(0, binding.maxResults)
			.map(({ name, result }) => ({
				file_id: result.id,
				filename: result.title,
				text: retrievalText(result.content),
				...(result.score !== undefined ? { score: result.score } : {}),
				...(result.uri !== undefined ? { uri: result.uri } : {}),
				attributes: { ...(result.metadata ?? {}) },
				vector_store_id: name,
			})),
	};
}

function codeToolDefinition(kind: 'code' | 'shell'): ToolDefinition {
	if (kind === 'code') {
		return {
			name: CODE_TOOL_NAME,
			description:
				'Execute code in the configured Agent RT sandbox interpreter.',
			inputSchema: {
				type: 'object',
				properties: {
					code: { type: 'string' },
					runtime: { type: 'string', default: 'python' },
				},
				required: ['code'],
			},
			sideEffect: 'write',
			executionMode: 'sequential',
		};
	}
	return {
		name: SHELL_TOOL_NAME,
		description:
			'Execute a shell command in the configured Agent RT sandbox.',
		inputSchema: {
			type: 'object',
			properties: {
				command: { type: 'string' },
				argv: { type: 'array', items: { type: 'string' } },
				cwd: { type: 'string' },
			},
		},
		sideEffect: 'write',
		executionMode: 'sequential',
	};
}

function ordinaryToolDefinition(tool: unknown): ToolDefinition | undefined {
	if (!isObject(tool)) return undefined;
	if (tool.type === 'function' && isObject(tool.function)) {
		const fn = tool.function;
		if (typeof fn.name !== 'string' || !fn.name) return undefined;
		return {
			name: fn.name,
			description:
				typeof fn.description === 'string' ? fn.description : '',
			inputSchema: isObject(fn.parameters)
				? fn.parameters
				: { type: 'object' },
		};
	}
	if (typeof tool.name === 'string' && tool.name) {
		return {
			name: tool.name,
			description:
				typeof tool.description === 'string' ? tool.description : '',
			inputSchema: isObject(tool.input_schema)
				? tool.input_schema
				: isObject(tool.inputSchema)
					? tool.inputSchema
					: { type: 'object' },
		};
	}
	return undefined;
}

export async function expandMigrationTools(
	tools: unknown,
	sandboxSession?: SandboxSession,
	mcpClients: Record<string, MCPMigrationClient> = {},
	retrievalRegistry?: RetrievalRegistry,
	webSearchProvider?: RetrievalProvider,
): Promise<ExpandedMigrationTools> {
	if (tools === undefined) {
		return {
			hasHostedCode: false,
			mcpBindings: new Map(),
			fileSearchBindings: new Map(),
			webSearchBindings: new Map(),
		};
	}
	if (!Array.isArray(tools)) throw new TypeError('tools must be an array');
	const converted: ToolDefinition[] = [];
	const mcpBindings = new Map<string, MCPBinding>();
	const fileSearchBindings = new Map<string, FileSearchBinding>();
	const webSearchBindings = new Map<string, WebSearchBinding>();
	let hasHostedCode = false;
	const seenHosted = new Set<string>();
	for (const tool of tools) {
		if (webSearchTool(tool)) {
			webSearchBindings.set(
				WEB_SEARCH_TOOL_NAME,
				webSearchBinding(tool, webSearchProvider),
			);
			if (!converted.some((item) => item.name === WEB_SEARCH_TOOL_NAME)) {
				converted.push(webSearchToolDefinition());
			}
			continue;
		}
		if (fileSearchTool(tool)) {
			fileSearchBindings.set(
				FILE_SEARCH_TOOL_NAME,
				fileSearchBinding(tool, retrievalRegistry),
			);
			if (
				!converted.some((item) => item.name === FILE_SEARCH_TOOL_NAME)
			) {
				converted.push(fileSearchToolDefinition());
			}
			continue;
		}
		const kind = hostedCodeKind(tool);
		if (kind) {
			if (!sandboxSession) {
				throw new Error(
					'code execution compatibility requires sandboxSession: SandboxSession',
				);
			}
			hasHostedCode = true;
			const name = kind === 'code' ? CODE_TOOL_NAME : SHELL_TOOL_NAME;
			if (!seenHosted.has(name)) {
				seenHosted.add(name);
				converted.push(codeToolDefinition(kind));
			}
			continue;
		}
		if (isObject(tool) && mcpToolKind(tool)) {
			const serverLabel = mcpServerLabel(tool);
			if (!serverLabel)
				throw new Error(
					'MCP tool declaration requires a server label/name',
				);
			const client = mcpClients[serverLabel];
			if (!client) {
				throw new Error(
					`MCP compatibility requires mcpClients[${JSON.stringify(serverLabel)}] with an Agent RT MCP client`,
				);
			}
			requireMCPApprovalSupported(tool);
			await ensureMCPInitialized(client);
			const discovered = await client.listTools();
			for (const discoveredTool of discovered) {
				if (!mcpToolEnabled(tool, discoveredTool.name)) continue;
				const functionName = mcpFunctionName(
					serverLabel,
					discoveredTool.name,
				);
				if (mcpBindings.has(functionName)) {
					throw new Error(
						`MCP tool name collision for ${JSON.stringify(functionName)}; rename the server or tool`,
					);
				}
				converted.push({
					name: functionName,
					description: discoveredTool.description ?? '',
					inputSchema: discoveredTool.inputSchema ?? {},
				});
				mcpBindings.set(functionName, {
					client,
					serverLabel,
					toolName: discoveredTool.name,
				});
			}
			continue;
		}
		const definition = ordinaryToolDefinition(tool);
		if (definition) converted.push(definition);
	}
	return {
		tools: converted.length ? converted : undefined,
		hasHostedCode,
		mcpBindings,
		fileSearchBindings,
		webSearchBindings,
	};
}

function sandboxResultPayload(
	result: SandboxCommandResult,
): SandboxResultPayload {
	const decoder = new TextDecoder();
	return {
		exit_code: result.exitCode,
		stdout: decoder.decode(result.stdout ?? new Uint8Array()),
		stderr: decoder.decode(result.stderr ?? new Uint8Array()),
		duration_ms: result.durationMs ?? 0,
		truncated: result.truncated ?? false,
	};
}

async function executeLocalCodeTool(
	session: SandboxSession,
	call: ToolCall,
): Promise<{ kind: 'code' | 'shell'; output: SandboxResultPayload }> {
	if (call.name === CODE_TOOL_NAME) {
		const code = call.arguments.code;
		if (typeof code !== 'string' || !code) {
			throw new Error(
				'code execution tool requires a non-empty code string',
			);
		}
		const runtime = call.arguments.runtime ?? 'python';
		if (typeof runtime !== 'string' || !runtime) {
			throw new Error(
				'code execution runtime must be a non-empty string',
			);
		}
		return {
			kind: 'code',
			output: sandboxResultPayload(await session.runCode(runtime, code)),
		};
	}
	if (call.name === SHELL_TOOL_NAME) {
		const command = call.arguments.command;
		const argvValue = call.arguments.argv;
		let argv: string[];
		if (Array.isArray(argvValue)) {
			argv = argvValue.map(String);
		} else if (typeof command === 'string' && command) {
			argv = ['sh', '-lc', command];
		} else {
			throw new Error('shell execution requires command or argv');
		}
		return {
			kind: 'shell',
			output: sandboxResultPayload(
				await session.execute({
					argv,
					cwd:
						typeof call.arguments.cwd === 'string'
							? call.arguments.cwd
							: '',
				}),
			),
		};
	}
	throw new Error(`unsupported local tool ${call.name}`);
}

function toolResultMessage(call: ToolCall, output: unknown): ModelMessage {
	return {
		role: 'tool',
		toolCallId: call.id,
		content: [{ type: 'text', text: JSON.stringify(output) }],
	};
}

export async function completeWithMigrationTools(
	provider: ModelProvider,
	request: ModelRequest,
	options: {
		sandboxSession?: SandboxSession;
		mcpBindings?: Map<string, MCPBinding>;
		fileSearchBindings?: Map<string, FileSearchBinding>;
		webSearchBindings?: Map<string, WebSearchBinding>;
		maxRounds?: number;
	} = {},
): Promise<{
	response: ModelResponse;
	executions: CodeExecutionRecord[];
	mcpCalls: MCPExecutionRecord[];
	fileSearchCalls: FileSearchExecutionRecord[];
	webSearchCalls: WebSearchExecutionRecord[];
	history: ModelMessage[];
}> {
	const executions: CodeExecutionRecord[] = [];
	const mcpCalls: MCPExecutionRecord[] = [];
	const fileSearchCalls: FileSearchExecutionRecord[] = [];
	const webSearchCalls: WebSearchExecutionRecord[] = [];
	const maxRounds = options.maxRounds ?? 8;
	const mcpBindings = options.mcpBindings ?? new Map();
	const fileSearchBindings = options.fileSearchBindings ?? new Map();
	const webSearchBindings = options.webSearchBindings ?? new Map();
	let current = request;
	for (let round = 0; round <= maxRounds; round += 1) {
		const response = await provider.complete(current);
		const calls = response.message.toolCalls ?? [];
		const local = calls.filter(
			(call) =>
				call.name === CODE_TOOL_NAME ||
				call.name === SHELL_TOOL_NAME ||
				mcpBindings.has(call.name) ||
				fileSearchBindings.has(call.name) ||
				webSearchBindings.has(call.name),
		);
		if (!local.length) {
			return {
				response,
				executions,
				mcpCalls,
				fileSearchCalls,
				webSearchCalls,
				history: [...current.messages, response.message],
			};
		}
		if (local.length !== calls.length) {
			return {
				response,
				executions,
				mcpCalls,
				fileSearchCalls,
				webSearchCalls,
				history: [...current.messages, response.message],
			};
		}
		if (round === maxRounds)
			throw new Error('local migration tools exceeded max rounds');
		const resultMessages: ModelMessage[] = [];
		for (const call of local) {
			if (call.argumentError) {
				// Malformed model arguments are never executed, not even with defaults.
				resultMessages.push(
					toolResultMessage(call, {
						error: `Tool call rejected: ${call.argumentError}`,
					}),
				);
				continue;
			}
			const webBinding = webSearchBindings.get(call.name);
			if (webBinding) {
				const output = await executeWebSearch(webBinding, call);
				webSearchCalls.push({
					id: call.id,
					query: output.query,
					results: output.results,
				});
				resultMessages.push(toolResultMessage(call, output));
				continue;
			}
			const fileBinding = fileSearchBindings.get(call.name);
			if (fileBinding) {
				const output = await executeFileSearch(fileBinding, call);
				fileSearchCalls.push({
					id: call.id,
					query: output.query,
					results: output.results,
				});
				resultMessages.push(toolResultMessage(call, output));
				continue;
			}
			const binding = mcpBindings.get(call.name);
			if (binding) {
				const output = await binding.client.callTool(binding.toolName, {
					...call.arguments,
				});
				mcpCalls.push({
					id: call.id,
					serverLabel: binding.serverLabel,
					toolName: binding.toolName,
					arguments: { ...call.arguments },
					output,
				});
				resultMessages.push(toolResultMessage(call, output));
				continue;
			}
			if (!options.sandboxSession) {
				throw new Error(
					'code execution compatibility requires sandboxSession: SandboxSession',
				);
			}
			const executed = await executeLocalCodeTool(
				options.sandboxSession,
				call,
			);
			executions.push({
				id: call.id,
				kind: executed.kind,
				arguments: { ...call.arguments },
				output: executed.output,
			});
			resultMessages.push(toolResultMessage(call, executed.output));
		}
		current = {
			...current,
			messages: [
				...current.messages,
				response.message,
				...resultMessages,
			],
		};
	}
	throw new Error('local migration tools exceeded max rounds');
}

export async function completeWithSandboxTools(
	provider: ModelProvider,
	request: ModelRequest,
	sandboxSession: SandboxSession | undefined,
	maxRounds = 8,
): Promise<{ response: ModelResponse; executions: CodeExecutionRecord[] }> {
	const completed = await completeWithMigrationTools(provider, request, {
		sandboxSession,
		maxRounds,
	});
	return { response: completed.response, executions: completed.executions };
}

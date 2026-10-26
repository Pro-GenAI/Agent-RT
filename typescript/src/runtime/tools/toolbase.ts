/** Toolbase: dynamic tool visibility, safety filters, and tool search. */
import { type ModelMessage, type ToolCall } from '../types.js';
import { type ToolDefinition } from './definitions.js';
import {
	CapabilityCatalog,
	type CapabilityDescriptor,
} from './capabilities.js';
import { type RegisteredTool, ToolRegistry } from './registry.js';
import { type AgentConfig } from '../agent/config.js';
import { type MCPTool } from '../protocols/mcp.js';

export interface ToolFilterContext {
	agent: AgentConfig;
	messages: ModelMessage[];
	turn: number;
	toolCalls: number;
	runtimeContext: Record<string, unknown>;
}

export type ToolVisibilityFilter = (
	context: ToolFilterContext,
	tools: ToolDefinition[],
) => string[] | Promise<string[]>;

export type ToolSafetyFilter = (
	tools: ToolDefinition[],
) => string[] | Promise<string[]>;

export const TOOL_SEARCH_NAME = 'tool_search';

const TOOL_SEARCH_RESULT_TYPE = 'agent_rt.tool_search_results';

export interface ToolbaseMCPClient {
	client?: {
		serverInfo?: unknown;
		initialize?: () => Promise<unknown>;
	};
	serverInfo?: unknown;
	initialize?: () => Promise<unknown>;
	listTools(): Promise<MCPTool[]>;
	callTool(
		name: string,
		argumentsValue: Record<string, unknown>,
	): Promise<unknown>;
}

function toolbaseMCPName(serverLabel: string, toolName: string): string {
	const safeServer = serverLabel.replace(/[^a-zA-Z0-9_-]/g, '_');
	const safeTool = toolName.replace(/[^a-zA-Z0-9_-]/g, '_');
	return 'mcp.' + safeServer + '.' + safeTool;
}

export class Toolbase {
	private readonly safeDefinitions: Map<string, ToolDefinition>;

	private constructor(
		readonly registry: ToolRegistry,
		safeDefinitions: Map<string, ToolDefinition>,
		readonly selectionFilter?: ToolVisibilityFilter,
		readonly searchLimit = 8,
	) {
		this.safeDefinitions = safeDefinitions;
	}

	static async initialize(
		options: {
			toolRegistry?: ToolRegistry;
			mcpClients?: Record<string, ToolbaseMCPClient>;
			safetyFilter?: ToolSafetyFilter;
			selectionFilter?: ToolVisibilityFilter;
			searchLimit?: number;
			allowlist?: string[];
			blocklist?: string[];
			read_allowlist?: string[];
			read_blocklist?: string[];
			write_allowlist?: string[];
			write_blocklist?: string[];
		} = {},
	): Promise<Toolbase> {
		const registry = options.toolRegistry ?? new ToolRegistry();
		const searchLimit = options.searchLimit ?? 8;
		if (!Number.isInteger(searchLimit) || searchLimit < 1) {
			throw new Error('tool search limit must be positive');
		}

		const permitted = (name: string, sideEffect?: string): boolean => {
			const reading =
				sideEffect === undefined ||
				sideEffect === 'none' ||
				sideEffect === 'read';
			const scopedAllow = reading
				? options.read_allowlist
				: options.write_allowlist;
			const scopedBlock = reading
				? options.read_blocklist
				: options.write_blocklist;
			return (
				!(options.blocklist ?? []).includes(name) &&
				!(scopedBlock ?? []).includes(name) &&
				(options.allowlist === undefined ||
					options.allowlist.includes(name)) &&
				(scopedAllow === undefined || scopedAllow.includes(name))
			);
		};
		for (const name of registry.deferredNames()) {
			if (
				!(options.blocklist ?? []).includes(name) &&
				(options.allowlist === undefined ||
					options.allowlist.includes(name))
			)
				registry.load(name);
		}

		const candidateDefinitions = new Map<string, ToolDefinition>();
		const candidateRawDefinitions = new Map<string, ToolDefinition>();
		for (const tool of registry.list()) {
			const name = ToolRegistry.qualifiedName(
				tool.definition.name,
				tool.namespace,
			);
			const definition = tool.namespace
				? { ...tool.definition, name }
				: tool.definition;
			candidateDefinitions.set(name, definition);
			candidateRawDefinitions.set(name, tool.definition);
		}
		if (candidateDefinitions.has(TOOL_SEARCH_NAME)) {
			throw new Error(
				'tool name ' +
					JSON.stringify(TOOL_SEARCH_NAME) +
					' is reserved by Toolbase',
			);
		}
		for (const [name, definition] of candidateDefinitions) {
			if (!permitted(name, definition.sideEffect))
				candidateDefinitions.delete(name);
		}

		const mcpBindings = new Map<
			string,
			{
				client: ToolbaseMCPClient;
				remoteName: string;
				definition: ToolDefinition;
			}
		>();
		for (const [serverLabel, client] of Object.entries(
			options.mcpClients ?? {},
		)) {
			const inner = client.client ?? client;
			if (
				inner.serverInfo === undefined &&
				typeof inner.initialize === 'function'
			) {
				await inner.initialize();
			}
			for (const tool of await client.listTools()) {
				if (!tool.name.trim()) {
					throw new Error(
						'MCP client ' +
							JSON.stringify(serverLabel) +
							' returned a tool without a name',
					);
				}
				const exposedName = toolbaseMCPName(serverLabel, tool.name);
				if (!permitted(exposedName, 'consequential')) continue;
				if (
					exposedName === TOOL_SEARCH_NAME ||
					candidateDefinitions.has(exposedName)
				) {
					throw new Error(
						'tool name collision while building Toolbase: ' +
							exposedName,
					);
				}
				const definition: ToolDefinition = {
					name: exposedName,
					description:
						tool.description?.trim() ||
						'MCP tool ' + tool.name + ' from ' + serverLabel,
					inputSchema: {
						...(tool.inputSchema ?? { type: 'object' }),
					},
					metadata: {
						toolbase_source: 'mcp',
						mcp_server: serverLabel,
						mcp_tool: tool.name,
					},
					sideEffect: 'consequential',
				};
				candidateDefinitions.set(exposedName, definition);
				mcpBindings.set(exposedName, {
					client,
					remoteName: tool.name,
					definition,
				});
			}
		}

		const safeNames = new Set(
			options.safetyFilter
				? await options.safetyFilter([...candidateDefinitions.values()])
				: [...candidateDefinitions.keys()],
		);
		for (const name of [...safeNames]) {
			if (!candidateDefinitions.has(name)) safeNames.delete(name);
		}

		for (const [exposedName, binding] of mcpBindings) {
			if (!safeNames.has(exposedName)) continue;
			const registered = registry.register(binding.definition, {
				handler: async (argumentsValue) =>
					binding.client.callTool(binding.remoteName, argumentsValue),
			});
			candidateRawDefinitions.set(exposedName, registered.definition);
		}

		const safeDefinitions = new Map<string, ToolDefinition>();
		for (const name of safeNames) {
			const definition = candidateRawDefinitions.get(name);
			if (definition) safeDefinitions.set(name, definition);
		}

		const toolbase = new Toolbase(
			registry,
			safeDefinitions,
			options.selectionFilter,
			searchLimit,
		);
		const searchDefinition: ToolDefinition = {
			name: TOOL_SEARCH_NAME,
			description:
				"Search the agent's full safe tool catalog for tools that were not included in the current turn. Matching tools become available on the next turn.",
			inputSchema: {
				type: 'object',
				properties: {
					query: { type: 'string', minLength: 1 },
					limit: {
						type: 'integer',
						minimum: 1,
						maximum: searchLimit,
					},
				},
				required: ['query'],
				additionalProperties: false,
			},
			sideEffect: 'none',
			metadata: { toolbase_internal: true },
		};
		const registeredSearch = registry.register(searchDefinition, {
			handler: (argumentsValue, signal) =>
				toolbase.searchHandler(argumentsValue, signal),
		});
		toolbase.safeDefinitions.set(
			TOOL_SEARCH_NAME,
			registeredSearch.definition,
		);
		return toolbase;
	}

	get version(): number {
		return this.registry.version;
	}

	private isScreened(name: string, tool: RegisteredTool): boolean {
		return this.safeDefinitions.get(name) === tool.definition;
	}

	definitions(): ToolDefinition[] {
		const visible: ToolDefinition[] = [];
		for (const tool of this.registry.list()) {
			const name = ToolRegistry.qualifiedName(
				tool.definition.name,
				tool.namespace,
			);
			if (!this.isScreened(name, tool)) continue;
			visible.push(
				tool.namespace ? { ...tool.definition, name } : tool.definition,
			);
		}
		return visible;
	}

	get(name: string): RegisteredTool {
		const tool = this.registry.get(name);
		if (!this.isScreened(name, tool)) {
			throw new Error(
				'tool is not safety-screened in this Toolbase: ' + name,
			);
		}
		return tool;
	}

	async execute(
		call: ToolCall,
		signal?: AbortSignal,
		requestContext: Record<string, unknown> = {},
	): Promise<unknown> {
		this.get(call.name);
		return this.registry.execute(call, signal, requestContext);
	}

	capabilityDescriptors(): CapabilityDescriptor[] {
		return this.definitions()
			.filter((definition) => definition.name !== TOOL_SEARCH_NAME)
			.map((definition) => ({
				id: 'tool:' + definition.name,
				kind: 'tool' as const,
				name: definition.name,
				description: definition.description,
				metadata: definition.metadata,
			}));
	}

	private async searchHandler(
		argumentsValue: Record<string, unknown>,
		_signal?: AbortSignal,
	): Promise<Record<string, unknown>> {
		const query = String(argumentsValue.query ?? '').trim();
		const requestedLimit =
			typeof argumentsValue.limit === 'number'
				? Math.floor(argumentsValue.limit)
				: this.searchLimit;
		const limit = Math.min(Math.max(requestedLimit, 1), this.searchLimit);
		const catalog = new CapabilityCatalog(this.capabilityDescriptors());
		const matches = catalog.search(query, { kinds: ['tool'], limit });
		const definitions = new Map(
			this.definitions().map((definition) => [
				definition.name,
				definition,
			]),
		);
		const tools = matches.flatMap((match) => {
			const definition = definitions.get(match.capability.name);
			if (!definition) return [];
			return [
				{
					name: definition.name,
					description: definition.description,
					inputSchema: definition.inputSchema,
					sideEffect: definition.sideEffect ?? 'none',
					score: match.score,
				},
			];
		});
		return { type: TOOL_SEARCH_RESULT_TYPE, query, tools };
	}

	private static searchedToolNames(messages: ModelMessage[]): Set<string> {
		const searchCallIds = new Set(
			messages.flatMap((message) =>
				message.role === 'assistant'
					? (message.toolCalls ?? [])
							.filter((call) => call.name === TOOL_SEARCH_NAME)
							.map((call) => call.id)
					: [],
			),
		);
		const names = new Set<string>();
		for (const message of messages) {
			if (
				message.role !== 'tool' ||
				!message.toolCallId ||
				!searchCallIds.has(message.toolCallId)
			) {
				continue;
			}
			for (const part of message.content) {
				if (
					part.type !== 'json' ||
					!part.data ||
					typeof part.data !== 'object' ||
					Array.isArray(part.data)
				) {
					continue;
				}
				const data = part.data as Record<string, unknown>;
				if (
					data.type !== TOOL_SEARCH_RESULT_TYPE ||
					!Array.isArray(data.tools)
				) {
					continue;
				}
				for (const result of data.tools) {
					if (
						result &&
						typeof result === 'object' &&
						!Array.isArray(result) &&
						typeof (result as Record<string, unknown>).name ===
							'string'
					) {
						names.add(
							(result as Record<string, unknown>).name as string,
						);
					}
				}
			}
		}
		return names;
	}

	visibilityFilter(baseFilter?: ToolVisibilityFilter): ToolVisibilityFilter {
		const selector = baseFilter ?? this.selectionFilter;
		return async (context, tools) => {
			const candidates = tools.filter(
				(tool) => tool.name !== TOOL_SEARCH_NAME,
			);
			const selected = new Set(
				selector
					? await selector(context, candidates)
					: candidates.map((tool) => tool.name),
			);
			for (const name of Toolbase.searchedToolNames(context.messages)) {
				selected.add(name);
			}
			selected.add(TOOL_SEARCH_NAME);
			return tools
				.filter((tool) => selected.has(tool.name))
				.map((tool) => tool.name);
		};
	}
}

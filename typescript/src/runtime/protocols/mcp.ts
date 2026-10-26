/** Model Context Protocol client and capability types. */

export type MCPCapabilityKind = 'tool' | 'resource' | 'prompt';

export interface MCPTransport {
	request(
		method: string,
		params?: Record<string, unknown>,
	): Promise<Record<string, unknown>>;
}

export interface MCPServerInfo {
	name: string;
	version?: string;
	capabilities: string[];
}

export interface MCPTool {
	name: string;
	description?: string;
	inputSchema?: Record<string, unknown>;
}

export interface MCPResource {
	uri: string;
	name?: string;
	description?: string;
	mimeType?: string;
}

export interface MCPPrompt {
	name: string;
	description?: string;
	arguments?: string[];
}

export class MCPClient {
	serverInfo?: MCPServerInfo;

	constructor(
		readonly transport: MCPTransport,
		readonly clientName = 'agent-rt',
		readonly clientVersion = '0',
		readonly requestedCapabilities: string[] = [],
	) {}

	async initialize(): Promise<MCPServerInfo> {
		const response = await this.transport.request('initialize', {
			clientInfo: {
				name: this.clientName,
				version: this.clientVersion,
			},
			capabilities: [...this.requestedCapabilities],
		});
		const rawServer = response.serverInfo;
		const server =
			rawServer &&
			typeof rawServer === 'object' &&
			!Array.isArray(rawServer)
				? (rawServer as Record<string, unknown>)
				: {};
		const rawCapabilities = response.capabilities;
		let capabilities: string[] = [];
		if (
			rawCapabilities &&
			typeof rawCapabilities === 'object' &&
			!Array.isArray(rawCapabilities)
		) {
			capabilities = Object.entries(
				rawCapabilities as Record<string, unknown>,
			)
				.filter(([, enabled]) => enabled !== false)
				.map(([name]) => name);
		} else if (Array.isArray(rawCapabilities)) {
			capabilities = rawCapabilities.map(String);
		}
		this.serverInfo = {
			name: String(server.name ?? 'mcp-server'),
			version:
				server.version === undefined
					? undefined
					: String(server.version),
			capabilities,
		};
		return this.serverInfo;
	}

	private ensureInitialized(): MCPServerInfo {
		if (!this.serverInfo) throw new Error('MCP client is not initialized');
		return this.serverInfo;
	}

	/** Enforce negotiation: only use features the server advertised. */
	private ensureCapability(capability: string): void {
		const server = this.ensureInitialized();
		if (!server.capabilities.includes(capability)) {
			throw new Error(
				`MCP server ${JSON.stringify(server.name)} does not advertise the ${JSON.stringify(capability)} capability`,
			);
		}
	}

	async listTools(): Promise<MCPTool[]> {
		this.ensureCapability('tools');
		const response = await this.transport.request('tools/list', {});
		const values = Array.isArray(response.tools) ? response.tools : [];
		return values.flatMap((value) => {
			if (!value || typeof value !== 'object' || Array.isArray(value))
				return [];
			const item = value as Record<string, unknown>;
			const name = String(item.name ?? '');
			if (!name.trim()) return [];
			return [
				{
					name,
					description: String(item.description ?? ''),
					inputSchema:
						item.inputSchema &&
						typeof item.inputSchema === 'object' &&
						!Array.isArray(item.inputSchema)
							? {
									...(item.inputSchema as Record<
										string,
										unknown
									>),
								}
							: {},
				},
			];
		});
	}

	async listResources(): Promise<MCPResource[]> {
		this.ensureCapability('resources');
		const response = await this.transport.request('resources/list', {});
		const values = Array.isArray(response.resources)
			? response.resources
			: [];
		return values.flatMap((value) => {
			if (!value || typeof value !== 'object' || Array.isArray(value))
				return [];
			const item = value as Record<string, unknown>;
			const uri = String(item.uri ?? '');
			if (!uri.trim()) return [];
			return [
				{
					uri,
					name:
						item.name === undefined ? undefined : String(item.name),
					description: String(item.description ?? ''),
					mimeType:
						item.mimeType === undefined
							? undefined
							: String(item.mimeType),
				},
			];
		});
	}

	async listPrompts(): Promise<MCPPrompt[]> {
		this.ensureCapability('prompts');
		const response = await this.transport.request('prompts/list', {});
		const values = Array.isArray(response.prompts) ? response.prompts : [];
		return values.flatMap((value) => {
			if (!value || typeof value !== 'object' || Array.isArray(value))
				return [];
			const item = value as Record<string, unknown>;
			const name = String(item.name ?? '');
			if (!name.trim()) return [];
			const args = Array.isArray(item.arguments)
				? item.arguments.flatMap((argument) => {
						if (
							!argument ||
							typeof argument !== 'object' ||
							Array.isArray(argument)
						)
							return [];
						const argName = (argument as Record<string, unknown>)
							.name;
						return argName === undefined ? [] : [String(argName)];
					})
				: [];
			return [
				{
					name,
					description: String(item.description ?? ''),
					arguments: args,
				},
			];
		});
	}

	async callTool(
		name: string,
		args: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		this.ensureCapability('tools');
		return this.transport.request('tools/call', {
			name,
			arguments: { ...args },
		});
	}

	async readResource(uri: string): Promise<Record<string, unknown>> {
		this.ensureCapability('resources');
		return this.transport.request('resources/read', { uri });
	}

	async getPrompt(
		name: string,
		args: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		this.ensureCapability('prompts');
		return this.transport.request('prompts/get', {
			name,
			arguments: { ...args },
		});
	}
}

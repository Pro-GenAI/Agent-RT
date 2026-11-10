/** MCP capability filtering, access policy, and the authorized MCP client. */
import { permissionGlobMatches } from '../authorization/permissions.js';
import {
	type AuthenticationContext,
	AuthorizationEngine,
	type AuthorizationRequirement,
} from '../authorization/identity.js';
import { type CapabilityGrant } from '../authorization/grants.js';
import { type PolicyEngine } from '../authorization/policy.js';
import {
	type MCPCapabilityKind,
	type MCPClient,
	type MCPPrompt,
	type MCPResource,
	type MCPTool,
} from './mcp.js';

export interface MCPCapabilityFilterOptions {
	tools?: string[];
	resources?: string[];
	prompts?: string[];
}

export class MCPCapabilityFilter {
	readonly tools: string[];
	readonly resources: string[];
	readonly prompts: string[];

	constructor(options: MCPCapabilityFilterOptions = {}) {
		this.tools = [...(options.tools ?? ['*'])];
		this.resources = [...(options.resources ?? ['*'])];
		this.prompts = [...(options.prompts ?? ['*'])];
	}

	allows(kind: MCPCapabilityKind, name: string): boolean {
		const patterns =
			kind === 'tool'
				? this.tools
				: kind === 'resource'
					? this.resources
					: this.prompts;
		return permissionGlobMatches(name, patterns);
	}
}

export interface MCPAccessPolicyOptions {
	authentication?: AuthenticationContext;
	authorizationEngine?: AuthorizationEngine;
	requirements?: Record<string, AuthorizationRequirement>;
	policyEngine?: PolicyEngine;
	capabilityFilter?: MCPCapabilityFilter;
	capabilityGrant?: CapabilityGrant;
}

export class MCPAccessPolicy {
	readonly authentication?: AuthenticationContext;
	readonly authorizationEngine: AuthorizationEngine;
	readonly requirements: Record<string, AuthorizationRequirement>;
	readonly policyEngine?: PolicyEngine;
	readonly capabilityFilter: MCPCapabilityFilter;
	readonly capabilityGrant?: CapabilityGrant;

	constructor(options: MCPAccessPolicyOptions = {}) {
		this.authentication = options.authentication;
		this.authorizationEngine =
			options.authorizationEngine ?? new AuthorizationEngine();
		this.requirements = { ...(options.requirements ?? {}) };
		this.policyEngine = options.policyEngine;
		this.capabilityFilter =
			options.capabilityFilter ?? new MCPCapabilityFilter();
		this.capabilityGrant = options.capabilityGrant;
	}

	check(input: {
		server: string;
		kind: MCPCapabilityKind;
		name: string;
		action: string;
	}): void {
		if (!this.capabilityFilter.allows(input.kind, input.name)) {
			throw new Error(
				'MCP ' + input.kind + ' is not approved: ' + input.name,
			);
		}
		if (
			input.kind === 'tool' &&
			this.capabilityGrant &&
			this.capabilityGrant.tools.length > 0 &&
			!this.capabilityGrant.allowsTool(input.name)
		) {
			throw new Error(
				'MCP tool is outside the capability grant: ' + input.name,
			);
		}
		if (!this.authentication && Object.keys(this.requirements).length > 0) {
			// Configured requirements cannot be evaluated without an identity;
			// silently skipping them would turn a misconfiguration into access.
			throw new Error('MCP policy requires an authenticated principal');
		}
		if (this.authentication) {
			const requirement =
				this.requirements[input.kind + ':' + input.action] ??
				this.requirements[input.kind] ??
				{};
			this.authorizationEngine.check(this.authentication, requirement);
			const credential = this.authentication.credential;
			if (
				credential &&
				this.capabilityGrant &&
				!this.capabilityGrant.allowsCredential(credential.id)
			) {
				throw new Error(
					'MCP credential is outside the capability grant',
				);
			}
		}
		if (this.policyEngine) {
			this.policyEngine.check({
				domain: 'mcp',
				action: input.kind + ':' + input.action,
				subject: this.authentication?.principal.id,
				resource: input.server + ':' + input.name,
				attributes: {
					server: input.server,
					kind: input.kind,
				},
			});
		}
	}
}

export class AuthorizedMCPClient {
	constructor(
		readonly client: MCPClient,
		readonly policy: MCPAccessPolicy,
	) {}

	private serverName(): string {
		if (!this.client.serverInfo)
			throw new Error('MCP client is not initialized');
		return this.client.serverInfo.name;
	}

	private allowed(
		kind: MCPCapabilityKind,
		name: string,
		action: string,
	): boolean {
		try {
			this.policy.check({
				server: this.serverName(),
				kind,
				name,
				action,
			});
			return true;
		} catch {
			return false;
		}
	}

	async listTools(): Promise<MCPTool[]> {
		const tools = await this.client.listTools();
		return tools.filter((tool) =>
			this.allowed('tool', tool.name, 'discover'),
		);
	}

	async listResources(): Promise<MCPResource[]> {
		const resources = await this.client.listResources();
		return resources.filter((resource) =>
			this.allowed('resource', resource.uri, 'discover'),
		);
	}

	async listPrompts(): Promise<MCPPrompt[]> {
		const prompts = await this.client.listPrompts();
		return prompts.filter((prompt) =>
			this.allowed('prompt', prompt.name, 'discover'),
		);
	}

	async callTool(
		name: string,
		args: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		this.policy.check({
			server: this.serverName(),
			kind: 'tool',
			name,
			action: 'invoke',
		});
		return this.client.callTool(name, args);
	}

	async readResource(uri: string): Promise<Record<string, unknown>> {
		this.policy.check({
			server: this.serverName(),
			kind: 'resource',
			name: uri,
			action: 'read',
		});
		return this.client.readResource(uri);
	}

	async getPrompt(
		name: string,
		args: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		this.policy.check({
			server: this.serverName(),
			kind: 'prompt',
			name,
			action: 'get',
		});
		return this.client.getPrompt(name, args);
	}
}

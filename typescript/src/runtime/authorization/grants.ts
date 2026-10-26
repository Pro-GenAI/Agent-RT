/** Least-privilege capability grants and tenant context. */
import { type MemoryScope, type SessionRef } from '../types.js';
import { type ToolDefinition } from '../tools/definitions.js';
import {
	normalizePermissionPath,
	permissionGlobMatches,
} from './permissions.js';

export interface CapabilityGrantOptions {
	tools?: string[];
	paths?: string[];
	networks?: string[];
	credentialIds?: string[];
}

export class CapabilityGrant {
	readonly tools: string[];
	readonly paths: string[];
	readonly networks: string[];
	readonly credentialIds: string[];

	constructor(options: CapabilityGrantOptions = {}) {
		this.tools = [...(options.tools ?? [])];
		this.paths = [...(options.paths ?? [])];
		this.networks = [...(options.networks ?? [])];
		this.credentialIds = [...(options.credentialIds ?? [])];
	}

	allowsTool(name: string): boolean {
		return permissionGlobMatches(name, this.tools);
	}
	allowsPath(path: string): boolean {
		return permissionGlobMatches(normalizePermissionPath(path), this.paths);
	}
	allowsNetwork(target: string): boolean {
		return permissionGlobMatches(target, this.networks);
	}
	allowsCredential(id: string): boolean {
		return this.credentialIds.includes(id);
	}
	filterTools(tools: ToolDefinition[]): ToolDefinition[] {
		return tools.filter((tool) => this.allowsTool(tool.name));
	}
}

export class TenantContext {
	constructor(readonly tenantId: string) {
		if (!tenantId.trim()) throw new Error('tenantId must not be empty');
		if (tenantId.includes('::')) {
			throw new Error("tenantId must not contain the '::' separator");
		}
	}

	qualify(kind: string, resourceId: string): string {
		if (!kind.trim() || !resourceId.trim()) {
			throw new Error('tenant resource kind and id must not be empty');
		}
		// '::' is the namespace separator; allowing it in ids would let one
		// tenant's ids collide with another tenant's qualified names.
		if (kind.includes('::') || resourceId.includes('::')) {
			throw new Error(
				"tenant resource kind and id must not contain '::'",
			);
		}
		return `${this.tenantId}::${kind}::${resourceId}`;
	}

	sessionRef(session: SessionRef): SessionRef {
		return {
			sessionId: this.qualify('session', session.sessionId),
			threadId: this.qualify('thread', session.threadId),
		};
	}

	memoryScope(scope: MemoryScope = {}): MemoryScope {
		if (scope.tenant !== undefined && scope.tenant !== this.tenantId) {
			throw new Error('memory scope tenant does not match');
		}
		return { ...scope, tenant: this.tenantId };
	}

	workspaceId(workspaceId: string): string {
		return this.qualify('workspace', workspaceId);
	}
	taskId(taskId: string): string {
		return this.qualify('task', taskId);
	}
	quotaKey(quotaId: string): string {
		return this.qualify('quota', quotaId);
	}
	credentialId(credentialId: string): string {
		return this.qualify('credential', credentialId);
	}
}

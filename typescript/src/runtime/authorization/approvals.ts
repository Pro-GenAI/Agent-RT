/** Human approval requests, grants, stores, and the approval manager. */
import { type ToolCall } from '../types.js';
import {
	type ToolDefinition,
	type ToolSideEffect,
} from '../tools/definitions.js';
import { type AuditTrail } from '../observability/audit.js';
import { permissionGlobMatches } from './permissions.js';
import { stableStringify } from '../resilience/budgets.js';

export type ApprovalDecision = 'allow' | 'deny';

export type ApprovalScope = 'once' | 'session' | 'durable';

export interface ApprovalRequest {
	id: string;
	call: ToolCall;
	sideEffect: ToolSideEffect;
	reason: string;
	sessionId?: string;
}

export interface ApprovalGrant {
	id: string;
	decision: ApprovalDecision;
	scope: ApprovalScope;
	toolPattern: string;
	sessionId?: string;
	callId?: string;
	createdAtMs?: number;
	/**
	 * Canonical JSON of the approved arguments. When set on a `once` grant the
	 * grant only applies to a call with identical arguments, so a reused call
	 * id cannot smuggle different arguments past the approving human.
	 */
	argumentsJson?: string;
}

export class ApprovalRequiredError extends Error {
	constructor(readonly request: ApprovalRequest) {
		super(request.reason);
		this.name = 'ApprovalRequiredError';
	}
}

export class ApprovalDeniedError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'ApprovalDeniedError';
	}
}

export interface ApprovalStore {
	add(grant: ApprovalGrant): ApprovalGrant;
	decision(options: {
		tool: string;
		callId: string;
		sessionId?: string;
		argumentsJson?: string;
	}): ApprovalDecision | undefined;
	revoke(grantId: string): boolean;
	list(): ApprovalGrant[];
}

export class InMemoryApprovalStore {
	private readonly grants = new Map<string, ApprovalGrant>();

	add(grant: ApprovalGrant): ApprovalGrant {
		if (grant.scope === 'session' && !grant.sessionId) {
			throw new Error('session approval requires sessionId');
		}
		if (grant.scope === 'once' && !grant.callId) {
			throw new Error('once approval requires callId');
		}
		const stored = {
			...grant,
			createdAtMs: grant.createdAtMs ?? Date.now(),
		};
		this.grants.set(stored.id, stored);
		return stored;
	}

	decision(options: {
		tool: string;
		callId: string;
		sessionId?: string;
		argumentsJson?: string;
	}): ApprovalDecision | undefined {
		const values = [...this.grants.values()].reverse();
		for (const grant of values) {
			if (!permissionGlobMatches(options.tool, [grant.toolPattern]))
				continue;
			if (
				grant.scope === 'session' &&
				grant.sessionId !== options.sessionId
			)
				continue;
			if (grant.scope === 'once' && grant.callId !== options.callId)
				continue;
			if (
				grant.scope === 'once' &&
				grant.argumentsJson !== undefined &&
				grant.argumentsJson !== options.argumentsJson
			)
				continue;
			if (grant.scope === 'once') this.grants.delete(grant.id);
			return grant.decision;
		}
		return undefined;
	}

	revoke(grantId: string): boolean {
		return this.grants.delete(grantId);
	}

	list(): ApprovalGrant[] {
		return [...this.grants.values()];
	}
}

export class ApprovalManager {
	readonly store: ApprovalStore;
	readonly requiredSideEffects: Set<ToolSideEffect>;
	/** Called with every request that needs a human, just before the run pauses. */
	notifier?: (request: ApprovalRequest) => void;

	constructor(
		store: ApprovalStore = new InMemoryApprovalStore(),
		requiredSideEffects: ToolSideEffect[] = [
			'consequential',
			'destructive',
		],
		readonly auditTrail?: AuditTrail,
	) {
		this.store = store;
		this.requiredSideEffects = new Set(requiredSideEffects);
	}

	requiresApproval(definition: ToolDefinition): boolean {
		return (
			this.requiredSideEffects.has(definition.sideEffect ?? 'none') ||
			definition.metadata?.approval_required === true
		);
	}

	check(
		call: ToolCall,
		definition: ToolDefinition,
		sessionId?: string,
	): void {
		if (!this.requiresApproval(definition)) return;
		const decision = this.store.decision({
			tool: call.name,
			callId: call.id,
			sessionId,
			argumentsJson: stableStringify(call.arguments),
		});
		if (decision === 'allow') return;
		if (decision === 'deny') {
			throw new ApprovalDeniedError(
				'approval denied for tool: ' + call.name,
			);
		}
		const request: ApprovalRequest = {
			id: 'approval:' + (sessionId ?? 'global') + ':' + call.id,
			call,
			sideEffect: definition.sideEffect ?? 'none',
			reason:
				'human approval required for ' +
				(definition.sideEffect ?? 'none') +
				' tool: ' +
				call.name,
			sessionId,
		};
		this.notifier?.(request);
		throw new ApprovalRequiredError(request);
	}

	resolve(
		request: ApprovalRequest,
		decision: ApprovalDecision,
		options: {
			scope?: ApprovalScope;
			grantId?: string;
			actorId?: string;
		} = {},
	): ApprovalGrant {
		const scope = options.scope ?? 'once';
		const stored = this.store.add({
			id: options.grantId ?? 'grant:' + request.id + ':' + scope,
			decision,
			scope,
			toolPattern: request.call.name,
			sessionId: scope === 'session' ? request.sessionId : undefined,
			callId: scope === 'once' ? request.call.id : undefined,
			argumentsJson:
				scope === 'once'
					? stableStringify(request.call.arguments)
					: undefined,
		});
		this.auditTrail?.append({
			id: 'audit:' + stored.id,
			actorId: options.actorId ?? 'human',
			action: 'authorized',
			resource: request.call.name,
			outcome: decision === 'allow' ? 'success' : 'denied',
			details: {
				approvalId: request.id,
				scope,
				callId: request.call.id,
			},
		});
		return stored;
	}
}

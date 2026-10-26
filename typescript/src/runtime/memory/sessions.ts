/** Short-term and tenant-scoped session memory. */
import {
	type ModelMessage,
	type SessionRef,
	type WorkflowState,
	validateSessionRef,
} from '../types.js';
import { type TenantContext } from '../authorization/grants.js';

export interface SessionSnapshot {
	session: SessionRef;
	messages: ModelMessage[];
	workflowState?: WorkflowState;
}

function sessionKey(session: SessionRef): string {
	validateSessionRef(session);
	return JSON.stringify([session.sessionId, session.threadId]);
}

export class ShortTermSessionMemory {
	private readonly messages = new Map<string, ModelMessage[]>();
	private readonly states = new Map<string, WorkflowState>();

	constructor(readonly maxMessages = 100) {
		if (!Number.isInteger(maxMessages) || maxMessages < 1) {
			throw new Error('maxMessages must be an integer of at least 1');
		}
	}

	appendMessages(session: SessionRef, messages: ModelMessage[]): void {
		const key = sessionKey(session);
		const bucket = [...(this.messages.get(key) ?? []), ...messages];
		this.messages.set(key, bucket.slice(-this.maxMessages));
	}

	setWorkflowState(session: SessionRef, state?: WorkflowState): void {
		const key = sessionKey(session);
		if (state === undefined) {
			this.states.delete(key);
		} else {
			this.states.set(key, state);
		}
	}

	snapshot(session: SessionRef): SessionSnapshot {
		const key = sessionKey(session);
		return {
			session: { ...session },
			messages: [...(this.messages.get(key) ?? [])],
			workflowState: this.states.get(key),
		};
	}

	clear(session: SessionRef): void {
		const key = sessionKey(session);
		this.messages.delete(key);
		this.states.delete(key);
	}
}

export class TenantSessionMemory {
	constructor(
		readonly memory: ShortTermSessionMemory,
		readonly tenant: TenantContext,
	) {}

	appendMessages(session: SessionRef, messages: ModelMessage[]): void {
		this.memory.appendMessages(this.tenant.sessionRef(session), messages);
	}

	setWorkflowState(session: SessionRef, state?: WorkflowState): void {
		this.memory.setWorkflowState(this.tenant.sessionRef(session), state);
	}

	snapshot(session: SessionRef): SessionSnapshot {
		return this.memory.snapshot(this.tenant.sessionRef(session));
	}

	clear(session: SessionRef): void {
		this.memory.clear(this.tenant.sessionRef(session));
	}
}

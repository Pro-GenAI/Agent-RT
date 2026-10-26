/** Audit records, audit trails, and tool lifecycle audit hooks. */
import { type ToolCall } from '../types.js';
import { type ToolDefinition } from '../tools/definitions.js';
import { PrivacyRedactor } from '../guardrails/privacy.js';

export type AuditAction =
	'requested' | 'authorized' | 'executed' | 'changed' | 'canceled';

export type AuditOutcome = 'success' | 'denied' | 'error';

export interface AuditRecord {
	id: string;
	actorId: string;
	action: AuditAction;
	resource: string;
	outcome?: AuditOutcome;
	details?: Record<string, unknown>;
	occurredAtMs?: number;
}

export interface AuditTrail {
	append(record: AuditRecord): AuditRecord;
	list(): AuditRecord[];
}

export class InMemoryAuditTrail implements AuditTrail {
	private readonly records: AuditRecord[] = [];
	private readonly ids = new Set<string>();

	append(record: AuditRecord): AuditRecord {
		if (this.ids.has(record.id))
			throw new Error('audit record already exists: ' + record.id);
		const stored = {
			...record,
			outcome: record.outcome ?? 'success',
			details: { ...(record.details ?? {}) },
			occurredAtMs: record.occurredAtMs ?? Date.now(),
		};
		this.ids.add(stored.id);
		this.records.push(stored);
		return stored;
	}

	list(): AuditRecord[] {
		return this.records.map((record) => ({
			...record,
			details: { ...(record.details ?? {}) },
		}));
	}
}

export type ToolAuditPhase = 'start' | 'success' | 'error';

export interface ToolAuditEvent {
	phase: ToolAuditPhase;
	call: ToolCall;
	definition: ToolDefinition;
	value?: unknown;
	error?: unknown;
	requestContext?: Record<string, unknown>;
}

export interface ToolLifecycleHooks {
	preCall?: (
		call: ToolCall,
		definition: ToolDefinition,
	) => Promise<ToolCall | void>;
	postCall?: (
		call: ToolCall,
		definition: ToolDefinition,
		value: unknown,
	) => Promise<unknown>;
	onError?: (
		call: ToolCall,
		definition: ToolDefinition,
		error: unknown,
	) => Promise<void>;
	audit?: (event: ToolAuditEvent) => Promise<void>;
}

export function makeAuditTrailHook(
	trail: AuditTrail,
	redactor: PrivacyRedactor = new PrivacyRedactor(),
): (event: ToolAuditEvent) => Promise<void> {
	return async (event) => {
		const actor = event.requestContext?.actorId;
		const actorId = typeof actor === 'string' && actor ? actor : 'unknown';
		const action: AuditAction =
			event.phase === 'start'
				? 'requested'
				: event.phase === 'success'
					? 'executed'
					: 'canceled';
		const details: Record<string, unknown> = {
			callId: event.call.id,
			sideEffect: event.definition.sideEffect ?? 'none',
			arguments: redactor.redact(event.call.arguments),
		};
		if (event.phase === 'success')
			details.result = redactor.redact(event.value);
		if (event.phase === 'error') {
			details.error = redactor.redact(
				event.error instanceof Error
					? event.error.message
					: String(event.error),
			);
		}
		trail.append({
			id:
				'audit:' +
				event.call.id +
				':' +
				event.phase +
				':' +
				(trail.list().length + 1),
			actorId,
			action,
			resource: event.call.name,
			outcome: event.phase === 'error' ? 'error' : 'success',
			details,
		});
	};
}

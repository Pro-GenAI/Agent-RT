/** Durable event logs, retention, and tenant-scoped event stores. */
import { type JSONValue, validateJSONValue } from '../types.js';
import { PrivacyRedactor } from '../guardrails/privacy.js';
import { type TenantContext } from '../authorization/grants.js';

export type DurableEventType =
	| 'state_changed'
	| 'model_requested'
	| 'model_completed'
	| 'tool_requested'
	| 'tool_completed'
	| 'approval_requested'
	| 'approval_resolved'
	| 'lifecycle_transition'
	| 'checkpoint_saved';

export interface DurableEvent {
	eventId: string;
	taskId: string;
	type: DurableEventType;
	payload?: Record<string, JSONValue>;
	sequence?: number;
	occurredAtMs?: number;
}

export interface EventStore {
	append(event: DurableEvent): DurableEvent;
	list(taskId: string, options?: { afterSequence?: number }): DurableEvent[];
}

export class InMemoryEventStore implements EventStore {
	private readonly streams = new Map<string, DurableEvent[]>();
	private readonly eventIds = new Set<string>();

	append(event: DurableEvent): DurableEvent {
		if (!event.eventId.trim()) throw new Error('eventId must not be empty');
		if (!event.taskId.trim()) throw new Error('taskId must not be empty');
		validateJSONValue(event.payload ?? {});
		if (this.eventIds.has(event.eventId)) {
			throw new Error('event already exists: ' + event.eventId);
		}
		const stream = this.streams.get(event.taskId) ?? [];
		const stored = {
			...event,
			payload: { ...(event.payload ?? {}) },
			sequence: stream.length + 1,
			occurredAtMs: event.occurredAtMs ?? Date.now(),
		};
		stream.push(stored);
		this.streams.set(event.taskId, stream);
		this.eventIds.add(stored.eventId);
		return stored;
	}

	list(
		taskId: string,
		options: { afterSequence?: number } = {},
	): DurableEvent[] {
		const afterSequence = options.afterSequence ?? 0;
		if (!Number.isInteger(afterSequence) || afterSequence < 0) {
			throw new Error('afterSequence must be a non-negative integer');
		}
		return [...(this.streams.get(taskId) ?? [])].filter(
			(event) => (event.sequence ?? 0) > afterSequence,
		);
	}
}

export interface EventRetentionPolicy {
	ttlMs?: number;
	archiveAfterMs?: number;
	ephemeral?: boolean;
	suppressEventTypes?: DurableEventType[];
}

export class RetainedEventStore implements EventStore {
	private readonly events = new Map<string, DurableEvent[]>();
	private readonly archive = new Map<string, DurableEvent[]>();
	private readonly ids = new Set<string>();
	// Per-task sequences never shrink: deriving them from the visible stream
	// would reuse numbers after archival or expiry and hide new events from
	// consumers resuming with afterSequence.
	private readonly lastSequence = new Map<string, number>();
	readonly redactor: PrivacyRedactor;

	constructor(
		readonly policy: EventRetentionPolicy = {},
		redactor: PrivacyRedactor = new PrivacyRedactor(),
	) {
		if (policy.ttlMs !== undefined && policy.ttlMs < 1) {
			throw new Error('ttlMs must be positive');
		}
		if (policy.archiveAfterMs !== undefined && policy.archiveAfterMs < 1) {
			throw new Error('archiveAfterMs must be positive');
		}
		this.redactor = redactor;
	}

	append(event: DurableEvent): DurableEvent {
		const sanitized: DurableEvent = {
			...event,
			payload: this.redactor.redact(event.payload ?? {}) as Record<
				string,
				JSONValue
			>,
			occurredAtMs: event.occurredAtMs ?? Date.now(),
		};
		if (
			this.policy.ephemeral ||
			(this.policy.suppressEventTypes ?? []).includes(event.type)
		) {
			return sanitized;
		}
		if (this.ids.has(event.eventId)) {
			throw new Error('event already exists: ' + event.eventId);
		}
		const stream = this.events.get(event.taskId) ?? [];
		const sequence = (this.lastSequence.get(event.taskId) ?? 0) + 1;
		this.lastSequence.set(event.taskId, sequence);
		const stored = { ...sanitized, sequence };
		stream.push(stored);
		this.events.set(event.taskId, stream);
		this.ids.add(stored.eventId);
		return stored;
	}

	list(
		taskId: string,
		options: { afterSequence?: number; nowMs?: number } = {},
	): DurableEvent[] {
		const afterSequence = options.afterSequence ?? 0;
		if (!Number.isInteger(afterSequence) || afterSequence < 0) {
			throw new Error('afterSequence must be a non-negative integer');
		}
		const now = options.nowMs ?? Date.now();
		return [...(this.events.get(taskId) ?? [])].filter(
			(event) =>
				(event.sequence ?? 0) > afterSequence &&
				(this.policy.ttlMs === undefined ||
					now - (event.occurredAtMs ?? 0) < this.policy.ttlMs),
		);
	}

	archiveDue(nowMs = Date.now()): number {
		if (this.policy.archiveAfterMs === undefined) return 0;
		let moved = 0;
		for (const [taskId, events] of this.events) {
			const keep: DurableEvent[] = [];
			for (const event of events) {
				if (
					nowMs - (event.occurredAtMs ?? 0) >=
					this.policy.archiveAfterMs
				) {
					const archived = this.archive.get(taskId) ?? [];
					archived.push(event);
					this.archive.set(taskId, archived);
					moved += 1;
				} else {
					keep.push(event);
				}
			}
			this.events.set(taskId, keep);
		}
		return moved;
	}

	archived(taskId: string): DurableEvent[] {
		return [...(this.archive.get(taskId) ?? [])];
	}

	purgeExpired(nowMs = Date.now()): number {
		if (this.policy.ttlMs === undefined) return 0;
		let removed = 0;
		for (const bucket of [this.events, this.archive]) {
			for (const [taskId, events] of bucket) {
				const keep = events.filter(
					(event) =>
						nowMs - (event.occurredAtMs ?? 0) < this.policy.ttlMs!,
				);
				removed += events.length - keep.length;
				bucket.set(taskId, keep);
			}
		}
		return removed;
	}
}

export class TenantEventStore implements EventStore {
	constructor(
		readonly store: EventStore,
		readonly tenant: TenantContext,
	) {}

	append(event: DurableEvent): DurableEvent {
		return this.store.append({
			...event,
			eventId: this.tenant.qualify('event', event.eventId),
			taskId: this.tenant.taskId(event.taskId),
		});
	}

	list(
		taskId: string,
		options: { afterSequence?: number } = {},
	): DurableEvent[] {
		return this.store.list(this.tenant.taskId(taskId), options);
	}
}

/** Scheduled tasks and external event triggers. */
import { type JSONValue, validateJSONValue } from '../types.js';
import { type InMemoryWorkQueue, type WorkQueueItem } from './queue.js';

export interface ScheduledTask {
	scheduleId: string;
	payload: JSONValue;
	nextRunAtMs: number;
	intervalMs?: number;
	maxRuns?: number;
	runs?: number;
}

function validateScheduledTask(task: ScheduledTask): void {
	if (!task.scheduleId.trim())
		throw new Error('scheduleId must not be empty');
	if (task.nextRunAtMs < 0) {
		throw new Error('nextRunAtMs must be non-negative');
	}
	if (task.intervalMs !== undefined && task.intervalMs < 1) {
		throw new Error('intervalMs must be positive');
	}
	if (task.maxRuns !== undefined && task.maxRuns < 1) {
		throw new Error('maxRuns must be at least 1');
	}
	validateJSONValue(task.payload);
}

export class InMemoryScheduler {
	private readonly scheduled = new Map<string, ScheduledTask>();

	schedule(task: ScheduledTask): ScheduledTask {
		validateScheduledTask(task);
		if (this.scheduled.has(task.scheduleId)) {
			throw new Error('schedule already exists: ' + task.scheduleId);
		}
		const stored = { ...task, runs: task.runs ?? 0 };
		this.scheduled.set(task.scheduleId, stored);
		return stored;
	}

	cancel(scheduleId: string): boolean {
		return this.scheduled.delete(scheduleId);
	}

	get(scheduleId: string): ScheduledTask | undefined {
		return this.scheduled.get(scheduleId);
	}

	due(nowMs: number): ScheduledTask[] {
		if (nowMs < 0) throw new Error('nowMs must be non-negative');
		const due = [...this.scheduled.values()]
			.filter((task) => task.nextRunAtMs <= nowMs)
			.sort(
				(a, b) =>
					a.nextRunAtMs - b.nextRunAtMs ||
					a.scheduleId.localeCompare(b.scheduleId),
			);

		for (const task of due) {
			const runs = (task.runs ?? 0) + 1;
			if (
				task.intervalMs === undefined ||
				(task.maxRuns !== undefined && runs >= task.maxRuns)
			) {
				this.scheduled.delete(task.scheduleId);
				continue;
			}
			let nextRunAtMs = task.nextRunAtMs;
			if (nextRunAtMs <= nowMs) {
				// Jump straight past now; stepping one interval at a time can
				// take billions of iterations after a long pause.
				nextRunAtMs +=
					(Math.floor((nowMs - nextRunAtMs) / task.intervalMs) + 1) *
					task.intervalMs;
			}
			this.scheduled.set(task.scheduleId, {
				...task,
				runs,
				nextRunAtMs,
			});
		}
		return due;
	}
}

export interface ExternalEvent {
	eventId: string;
	source: string;
	type: string;
	payload: JSONValue;
	occurredAtMs?: number;
}

export interface EventTriggerRule {
	triggerId: string;
	source: string;
	eventType: string;
	taskPrefix?: string;
	priority?: number;
	maxAttempts?: number;
}

export class EventTriggerDispatcher {
	private readonly rules = new Map<string, EventTriggerRule>();
	private readonly seenEvents = new Set<string>();

	constructor(
		readonly queue: InMemoryWorkQueue,
		readonly clock: () => number = Date.now,
	) {}

	register(rule: EventTriggerRule): void {
		if (!rule.triggerId.trim())
			throw new Error('triggerId must not be empty');
		if (!rule.source.trim())
			throw new Error('trigger source must not be empty');
		if (!rule.eventType.trim())
			throw new Error('trigger eventType must not be empty');
		if ((rule.maxAttempts ?? 3) < 1) {
			throw new Error('trigger maxAttempts must be at least 1');
		}
		if (this.rules.has(rule.triggerId)) {
			throw new Error('event trigger already exists: ' + rule.triggerId);
		}
		this.rules.set(rule.triggerId, { ...rule });
	}

	unregister(triggerId: string): boolean {
		return this.rules.delete(triggerId);
	}

	dispatch(event: ExternalEvent): WorkQueueItem[] {
		if (!event.eventId.trim())
			throw new Error('external eventId must not be empty');
		if (!event.source.trim())
			throw new Error('external event source must not be empty');
		if (!event.type.trim())
			throw new Error('external event type must not be empty');
		validateJSONValue(event.payload);
		if (this.seenEvents.has(event.eventId)) return [];

		const now = this.clock();
		const enqueued: WorkQueueItem[] = [];
		for (const rule of [...this.rules.values()].sort((a, b) =>
			a.triggerId.localeCompare(b.triggerId),
		)) {
			if (rule.source !== event.source || rule.eventType !== event.type)
				continue;
			const item = {
				itemId:
					(rule.taskPrefix ?? 'event') +
					':' +
					rule.triggerId +
					':' +
					event.eventId,
				payload: {
					triggerId: rule.triggerId,
					eventId: event.eventId,
					source: event.source,
					type: event.type,
					payload: event.payload,
				},
				priority: rule.priority ?? 0,
				maxAttempts: rule.maxAttempts ?? 3,
				availableAtMs: now,
				enqueuedAtMs: now,
			};
			if (this.queue.contains(item.itemId)) continue; // earlier partial dispatch
			this.queue.enqueue(item);
			enqueued.push(item);
		}
		// Mark the event seen only once every rule has been enqueued, so a
		// failure part-way leaves it eligible for redelivery.
		this.seenEvents.add(event.eventId);
		return enqueued;
	}
}

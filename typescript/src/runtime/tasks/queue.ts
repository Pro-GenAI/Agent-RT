/** Leased work queues. */
import { type JSONValue, validateJSONValue } from '../types.js';

export interface WorkQueueItem {
	itemId: string;
	payload: JSONValue;
	priority?: number;
	maxAttempts?: number;
	attempts?: number;
	availableAtMs?: number;
	leaseOwner?: string;
	leaseExpiresAtMs?: number;
	enqueuedAtMs?: number;
}

function validateWorkQueueItem(item: WorkQueueItem): void {
	if (!item.itemId.trim()) throw new Error('queue itemId must not be empty');
	if ((item.maxAttempts ?? 3) < 1) {
		throw new Error('queue maxAttempts must be at least 1');
	}
	if ((item.attempts ?? 0) < 0) {
		throw new Error('queue attempts must be non-negative');
	}
	validateJSONValue(item.payload);
}

export class InMemoryWorkQueue {
	private readonly items = new Map<
		string,
		Required<Omit<WorkQueueItem, 'leaseOwner' | 'leaseExpiresAtMs'>> & {
			leaseOwner?: string;
			leaseExpiresAtMs?: number;
		}
	>();
	private readonly completed = new Set<string>();
	private readonly failed = new Set<string>();

	constructor(
		readonly options: {
			maxActiveLeases?: number;
			maxLeasesPerWorker?: number;
			clock?: () => number;
		} = {},
	) {
		if (
			options.maxActiveLeases !== undefined &&
			(!Number.isInteger(options.maxActiveLeases) ||
				options.maxActiveLeases < 1)
		) {
			throw new Error('maxActiveLeases must be an integer of at least 1');
		}
		if (
			options.maxLeasesPerWorker !== undefined &&
			(!Number.isInteger(options.maxLeasesPerWorker) ||
				options.maxLeasesPerWorker < 1)
		) {
			throw new Error(
				'maxLeasesPerWorker must be an integer of at least 1',
			);
		}
	}

	private now(): number {
		return this.options.clock?.() ?? Date.now();
	}

	/** True if the item is queued, leased, completed, or failed. */
	contains(itemId: string): boolean {
		return (
			this.items.has(itemId) ||
			this.completed.has(itemId) ||
			this.failed.has(itemId)
		);
	}

	enqueue(item: WorkQueueItem): WorkQueueItem {
		validateWorkQueueItem(item);
		if (this.items.has(item.itemId) || this.completed.has(item.itemId)) {
			throw new Error('queue item already exists: ' + item.itemId);
		}
		const stored = {
			itemId: item.itemId,
			payload: item.payload,
			priority: item.priority ?? 0,
			maxAttempts: item.maxAttempts ?? 3,
			attempts: item.attempts ?? 0,
			availableAtMs: item.availableAtMs ?? 0,
			enqueuedAtMs: item.enqueuedAtMs ?? this.now(),
			...(item.leaseOwner ? { leaseOwner: item.leaseOwner } : {}),
			...(item.leaseExpiresAtMs !== undefined
				? { leaseExpiresAtMs: item.leaseExpiresAtMs }
				: {}),
		};
		this.items.set(item.itemId, stored);
		return stored;
	}

	releaseExpired(): number {
		const now = this.now();
		let released = 0;
		for (const [itemId, item] of this.items.entries()) {
			if (
				item.leaseOwner &&
				item.leaseExpiresAtMs !== undefined &&
				item.leaseExpiresAtMs <= now
			) {
				if (item.attempts >= item.maxAttempts) {
					// Every attempt crashed or timed out: dead-letter it instead
					// of leaving an item that can never be leased again.
					this.items.delete(itemId);
					this.failed.add(itemId);
					released += 1;
					continue;
				}
				const updated = { ...item };
				delete updated.leaseOwner;
				delete updated.leaseExpiresAtMs;
				this.items.set(itemId, updated);
				released += 1;
			}
		}
		return released;
	}

	lease(
		workerId: string,
		options: { leaseMs: number; limit?: number },
	): WorkQueueItem[] {
		if (!workerId.trim()) throw new Error('workerId must not be empty');
		if (!Number.isInteger(options.leaseMs) || options.leaseMs < 1) {
			throw new Error('leaseMs must be a positive integer');
		}
		const limit = options.limit ?? 1;
		if (!Number.isInteger(limit) || limit < 1) {
			throw new Error('lease limit must be an integer of at least 1');
		}
		this.releaseExpired();
		const now = this.now();
		const active = [...this.items.values()].filter(
			(item) => item.leaseOwner,
		);
		const globalSlots =
			this.options.maxActiveLeases === undefined
				? limit
				: Math.max(0, this.options.maxActiveLeases - active.length);
		const workerActive = active.filter(
			(item) => item.leaseOwner === workerId,
		).length;
		const workerSlots =
			this.options.maxLeasesPerWorker === undefined
				? limit
				: Math.max(0, this.options.maxLeasesPerWorker - workerActive);
		const take = Math.min(limit, globalSlots, workerSlots);
		if (take <= 0) return [];

		const available = [...this.items.values()]
			.filter(
				(item) =>
					!item.leaseOwner &&
					item.availableAtMs <= now &&
					item.attempts < item.maxAttempts,
			)
			.sort(
				(a, b) =>
					b.priority - a.priority ||
					a.enqueuedAtMs - b.enqueuedAtMs ||
					a.itemId.localeCompare(b.itemId),
			);

		return available.slice(0, take).map((item) => {
			const leased = {
				...item,
				attempts: item.attempts + 1,
				leaseOwner: workerId,
				leaseExpiresAtMs: now + options.leaseMs,
			};
			this.items.set(item.itemId, leased);
			return leased;
		});
	}

	ack(itemId: string, workerId: string): boolean {
		const item = this.items.get(itemId);
		if (!item || item.leaseOwner !== workerId) return false;
		this.items.delete(itemId);
		this.completed.add(itemId);
		return true;
	}

	fail(
		itemId: string,
		workerId: string,
		options: { retryDelayMs?: number } = {},
	): boolean {
		const retryDelayMs = options.retryDelayMs ?? 0;
		if (!Number.isInteger(retryDelayMs) || retryDelayMs < 0) {
			throw new Error('retryDelayMs must be a non-negative integer');
		}
		const item = this.items.get(itemId);
		if (!item || item.leaseOwner !== workerId) return false;
		if (item.attempts >= item.maxAttempts) {
			this.items.delete(itemId);
			this.failed.add(itemId);
			return true;
		}
		const updated = {
			...item,
			availableAtMs: this.now() + retryDelayMs,
		};
		delete updated.leaseOwner;
		delete updated.leaseExpiresAtMs;
		this.items.set(itemId, updated);
		return true;
	}

	list(): WorkQueueItem[] {
		return [...this.items.values()].sort(
			(a, b) =>
				b.priority - a.priority ||
				a.enqueuedAtMs - b.enqueuedAtMs ||
				a.itemId.localeCompare(b.itemId),
		);
	}
}

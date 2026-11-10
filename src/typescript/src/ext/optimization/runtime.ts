export interface CachePolicy {
	ttlMs?: number;
	cacheable?: boolean;
	namespace?: string;
}

interface CacheEntry {
	value: unknown;
	expiresAt?: number;
}

export interface CacheLookup {
	found: boolean;
	value?: unknown;
}

export class ResponseCache {
	private readonly entries = new Map<string, CacheEntry>();

	constructor(
		private readonly clock: () => number = () => Date.now(),
		readonly maxEntries: number | undefined = 4096,
	) {
		if (maxEntries !== undefined && maxEntries < 1) {
			throw new Error('maxEntries must be at least 1');
		}
	}

	/** Bound memory: drop expired entries first, then the oldest writes. */
	private evict(): void {
		if (
			this.maxEntries === undefined ||
			this.entries.size <= this.maxEntries
		)
			return;
		const now = this.clock();
		for (const [key, entry] of this.entries) {
			if (entry.expiresAt !== undefined && now >= entry.expiresAt)
				this.entries.delete(key);
		}
		for (const key of this.entries.keys()) {
			if (this.entries.size <= this.maxEntries) break;
			this.entries.delete(key);
		}
	}

	private validateNamespace(namespace: string): void {
		if (!namespace.trim())
			throw new Error('cache namespace must not be empty');
	}

	lookup(key: string, namespace = 'default'): CacheLookup {
		this.validateNamespace(namespace);
		const entry = this.entries.get(namespace + '::' + key);
		if (!entry) return { found: false };
		if (entry.expiresAt !== undefined && this.clock() >= entry.expiresAt) {
			this.entries.delete(namespace + '::' + key);
			return { found: false };
		}
		return { found: true, value: structuredClone(entry.value) };
	}

	get(key: string, namespace = 'default'): unknown {
		const result = this.lookup(key, namespace);
		return result.found ? result.value : undefined;
	}

	set(key: string, value: unknown, policy: CachePolicy = {}): void {
		if (policy.cacheable === false) return;
		const namespace = policy.namespace ?? 'default';
		if (!namespace.trim())
			throw new Error('cache namespace must not be empty');
		if (policy.ttlMs !== undefined && policy.ttlMs < 0) {
			throw new Error('ttlMs must be non-negative');
		}
		const identity = namespace + '::' + key;
		this.entries.delete(identity); // re-insert so eviction order is by write time
		this.entries.set(identity, {
			value: structuredClone(value),
			expiresAt:
				policy.ttlMs === undefined
					? undefined
					: this.clock() + policy.ttlMs,
		});
		this.evict();
	}

	invalidate(key: string, namespace = 'default'): boolean {
		this.validateNamespace(namespace);
		return this.entries.delete(namespace + '::' + key);
	}

	clearNamespace(namespace: string): number {
		this.validateNamespace(namespace);
		const keys = [...this.entries.keys()].filter((key) =>
			key.startsWith(namespace + '::'),
		);
		for (const key of keys) this.entries.delete(key);
		return keys.length;
	}
}

export class SingleFlight {
	private readonly active = new Map<string, Promise<unknown>>();

	async run<T>(key: string, operation: () => Promise<T>): Promise<T> {
		const existing = this.active.get(key);
		if (existing) return existing as Promise<T>;
		const promise = operation();
		this.active.set(key, promise);
		try {
			return await promise;
		} finally {
			if (this.active.get(key) === promise) this.active.delete(key);
		}
	}
}

export class BatchExecutor<T, R> {
	constructor(
		private readonly handler: (items: T[]) => Promise<R[]>,
		private readonly maxBatchSize = 32,
	) {
		if (maxBatchSize < 1)
			throw new Error('maxBatchSize must be at least 1');
	}

	async execute(items: T[]): Promise<R[]> {
		const results: R[] = [];
		for (let index = 0; index < items.length; index += this.maxBatchSize) {
			const batch = items.slice(index, index + this.maxBatchSize);
			const batchResults = await this.handler(batch);
			if (batchResults.length !== batch.length) {
				throw new Error(
					'batch handler must return one result per input',
				);
			}
			results.push(...batchResults);
		}
		return results;
	}
}

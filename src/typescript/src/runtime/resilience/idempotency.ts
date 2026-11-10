/** Idempotency records and stores. */

export interface IdempotencyRecord {
	scope: string;
	key: string;
	value: unknown;
	createdAtMs?: number;
}

export interface IdempotencyStore {
	get(scope: string, key: string): IdempotencyRecord | undefined;
	put(record: IdempotencyRecord): IdempotencyRecord;
}

export class InMemoryIdempotencyStore implements IdempotencyStore {
	private readonly records = new Map<string, IdempotencyRecord>();

	private identity(scope: string, key: string): string {
		if (!scope.trim())
			throw new Error('idempotency scope must not be empty');
		if (!key.trim()) throw new Error('idempotency key must not be empty');
		return scope + '\\u0000' + key;
	}

	get(scope: string, key: string): IdempotencyRecord | undefined {
		return this.records.get(this.identity(scope, key));
	}

	put(record: IdempotencyRecord): IdempotencyRecord {
		const identity = this.identity(record.scope, record.key);
		const existing = this.records.get(identity);
		if (existing) return existing;
		const stored = {
			...record,
			createdAtMs: record.createdAtMs ?? Date.now(),
		};
		this.records.set(identity, stored);
		return stored;
	}
}

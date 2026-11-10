/** Long-term memory records, queries, and the in-memory store. */
import {
	type JSONValue,
	type MemoryScope,
	scopeContains,
	validateJSONValue,
} from '../types.js';

export type MemoryKind = 'fact' | 'semantic' | 'episode' | 'procedure';

export interface MemoryRecord {
	id: string;
	kind: MemoryKind;
	content: string;
	scope?: MemoryScope;
	metadata?: Record<string, JSONValue>;
	tags?: string[];
	embedding?: number[];
	sequence?: number;
	createdAtMs?: number;
	expiresAtMs?: number;
	schemaVersion?: number;
}

function validateMemoryRecord(record: MemoryRecord): void {
	if (!record.id.trim()) throw new Error('memory id must not be empty');
	if (!record.content.trim())
		throw new Error('memory content must not be empty');
	validateJSONValue(record.metadata ?? {});
	const createdAtMs = record.createdAtMs ?? Date.now();
	if (createdAtMs < 0)
		throw new Error('memory createdAtMs must be non-negative');
	if (record.expiresAtMs !== undefined && record.expiresAtMs < createdAtMs) {
		throw new Error('memory expiresAtMs must not precede createdAtMs');
	}
	if ((record.schemaVersion ?? 1) < 1) {
		throw new Error('memory schemaVersion must be at least 1');
	}
	if (record.embedding !== undefined) {
		if (record.embedding.length === 0) {
			throw new Error('memory embedding must not be empty');
		}
		if (record.embedding.some((value) => !Number.isFinite(value))) {
			throw new Error('memory embedding values must be finite numbers');
		}
	}
}

export interface MemorySearchQuery {
	text?: string;
	scope?: MemoryScope;
	kinds?: MemoryKind[];
	tags?: string[];
	limit?: number;
}

export interface MemorySearchResult {
	record: MemoryRecord;
	score: number;
}

export interface LongTermMemoryStore {
	read(memoryId: string): MemoryRecord | undefined;
	write(record: MemoryRecord): MemoryRecord;
	update(memoryId: string, record: MemoryRecord): MemoryRecord;
	delete(memoryId: string): boolean;
	search(query: MemorySearchQuery): MemorySearchResult[];
}

export class InMemoryLongTermMemoryStore implements LongTermMemoryStore {
	private readonly records = new Map<string, MemoryRecord>();
	private sequence = 0;

	read(memoryId: string): MemoryRecord | undefined {
		return this.records.get(memoryId);
	}

	write(record: MemoryRecord): MemoryRecord {
		validateMemoryRecord(record);
		if (this.records.has(record.id)) {
			throw new Error('memory already exists: ' + record.id);
		}
		this.sequence += 1;
		const stored = {
			...record,
			scope: { ...(record.scope ?? {}) },
			metadata: { ...(record.metadata ?? {}) },
			tags: [...(record.tags ?? [])],
			embedding: record.embedding ? [...record.embedding] : undefined,
			sequence: this.sequence,
			createdAtMs: record.createdAtMs ?? Date.now(),
			expiresAtMs: record.expiresAtMs,
			schemaVersion: record.schemaVersion ?? 1,
		};
		this.records.set(record.id, stored);
		return stored;
	}

	update(memoryId: string, record: MemoryRecord): MemoryRecord {
		validateMemoryRecord(record);
		const previous = this.records.get(memoryId);
		if (!previous) throw new Error('memory not found: ' + memoryId);
		if (record.id !== memoryId) {
			throw new Error('memory update cannot change id');
		}
		const stored = {
			...record,
			scope: { ...(record.scope ?? {}) },
			metadata: { ...(record.metadata ?? {}) },
			tags: [...(record.tags ?? [])],
			embedding: record.embedding ? [...record.embedding] : undefined,
			sequence: previous.sequence,
			createdAtMs:
				record.createdAtMs ?? previous.createdAtMs ?? Date.now(),
			expiresAtMs: record.expiresAtMs,
			schemaVersion: record.schemaVersion ?? previous.schemaVersion ?? 1,
		};
		this.records.set(memoryId, stored);
		return stored;
	}

	delete(memoryId: string): boolean {
		return this.records.delete(memoryId);
	}

	search(query: MemorySearchQuery): MemorySearchResult[] {
		const limit = query.limit ?? 10;
		if (!Number.isInteger(limit) || limit < 0) {
			throw new Error(
				'memory search limit must be a non-negative integer',
			);
		}
		const queryTerms = terms(query.text ?? '');
		const kinds = query.kinds ? new Set(query.kinds) : undefined;
		const requiredTags = new Set(query.tags ?? []);
		const results: MemorySearchResult[] = [];

		for (const record of this.records.values()) {
			if (query.scope && !scopeContains(query.scope, record.scope ?? {}))
				continue;
			if (kinds && !kinds.has(record.kind)) continue;
			const recordTags = new Set(record.tags ?? []);
			if ([...requiredTags].some((tag) => !recordTags.has(tag))) continue;

			let score = 1;
			if (queryTerms.size > 0) {
				const metadataText = Object.entries(record.metadata ?? {})
					.filter(([, value]) =>
						['string', 'number', 'boolean'].includes(typeof value),
					)
					.map(([key, value]) => key + ' ' + String(value))
					.join(' ');
				const recordTerms = terms(
					record.content +
						' ' +
						(record.tags ?? []).join(' ') +
						' ' +
						metadataText,
				);
				let overlap = 0;
				for (const term of queryTerms)
					if (recordTerms.has(term)) overlap += 1;
				if (overlap === 0) continue;
				score = overlap / queryTerms.size;
			}
			results.push({ record, score });
		}

		results.sort(
			(a, b) =>
				b.score - a.score ||
				(b.record.sequence ?? 0) - (a.record.sequence ?? 0) ||
				a.record.id.localeCompare(b.record.id),
		);
		return results.slice(0, limit);
	}
}

function terms(value: string): Set<string> {
	return new Set(value.toLowerCase().match(/[a-z0-9]+/g) ?? []);
}

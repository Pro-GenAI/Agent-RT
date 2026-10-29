/** Memory write and retrieval policies and scoped memory stores. */
import { type JSONValue, type MemoryScope, scopeContains } from '../types.js';
import {
	type LongTermMemoryStore,
	type MemoryKind,
	type MemoryRecord,
	type MemorySearchQuery,
	type MemorySearchResult,
} from './long_term.js';

export interface MemoryWriteCandidate {
	id: string;
	kind: MemoryKind;
	content: string;
	scope?: MemoryScope;
	relevance?: number;
	sensitivity?: number;
	confidence?: number;
	metadata?: Record<string, JSONValue>;
	tags?: string[];
}

export interface MemoryWriteDecision {
	persist: boolean;
	reason: string;
}

export class MemoryWritePolicy {
	constructor(
		readonly options: {
			minRelevance?: number;
			minConfidence?: number;
			maxSensitivity?: number;
			rejectDuplicates?: boolean;
			allowedKinds?: MemoryKind[];
		} = {},
	) {
		for (const [name, value] of Object.entries({
			minRelevance: options.minRelevance ?? 0.5,
			minConfidence: options.minConfidence ?? 0.5,
			maxSensitivity: options.maxSensitivity ?? 0.5,
		})) {
			if (value < 0 || value > 1) {
				throw new Error(name + ' must be between 0 and 1');
			}
		}
	}

	decide(
		candidate: MemoryWriteCandidate,
		store: LongTermMemoryStore,
	): MemoryWriteDecision {
		const relevance = candidate.relevance ?? 1;
		const sensitivity = candidate.sensitivity ?? 0;
		const confidence = candidate.confidence ?? 1;
		const allowedKinds = this.options.allowedKinds
			? new Set(this.options.allowedKinds)
			: undefined;
		if (allowedKinds && !allowedKinds.has(candidate.kind)) {
			return { persist: false, reason: 'kind_not_allowed' };
		}
		if (relevance < (this.options.minRelevance ?? 0.5)) {
			return { persist: false, reason: 'relevance_below_threshold' };
		}
		if (confidence < (this.options.minConfidence ?? 0.5)) {
			return { persist: false, reason: 'confidence_below_threshold' };
		}
		if (sensitivity > (this.options.maxSensitivity ?? 0.5)) {
			return { persist: false, reason: 'sensitivity_above_threshold' };
		}
		if (this.options.rejectDuplicates !== false) {
			const normalized = candidate.content
				.toLowerCase()
				.trim()
				.replace(/\s+/g, ' ');
			const duplicates = store.search({
				text: candidate.content,
				scope: candidate.scope,
				kinds: [candidate.kind],
				limit: 10,
			});
			if (
				duplicates.some(
					(result) =>
						result.record.content
							.toLowerCase()
							.trim()
							.replace(/\s+/g, ' ') === normalized,
				)
			) {
				return { persist: false, reason: 'duplicate' };
			}
		}
		return { persist: true, reason: 'accepted' };
	}

	persist(
		candidate: MemoryWriteCandidate,
		store: LongTermMemoryStore,
	): MemoryRecord | undefined {
		const decision = this.decide(candidate, store);
		if (!decision.persist) return undefined;
		return store.write({
			id: candidate.id,
			kind: candidate.kind,
			content: candidate.content,
			scope: candidate.scope ?? {},
			metadata: candidate.metadata ?? {},
			tags: candidate.tags ?? [],
		});
	}
}

export class MemoryRetrievalPolicy {
	constructor(
		readonly options: {
			relevanceWeight?: number;
			recencyWeight?: number;
			confidenceWeight?: number;
			minConfidence?: number;
		} = {},
	) {
		for (const [name, value] of Object.entries({
			relevanceWeight: options.relevanceWeight ?? 0.6,
			recencyWeight: options.recencyWeight ?? 0.2,
			confidenceWeight: options.confidenceWeight ?? 0.2,
			minConfidence: options.minConfidence ?? 0,
		})) {
			if (value < 0 || value > 1) {
				throw new Error(name + ' must be between 0 and 1');
			}
		}
	}

	search(
		store: LongTermMemoryStore,
		query: MemorySearchQuery,
	): MemorySearchResult[] {
		const limit = query.limit ?? 10;
		const candidates = store.search({
			...query,
			limit: Math.max(limit * 10, limit),
		});
		if (candidates.length === 0) return [];
		const maxSequence = Math.max(
			...candidates.map((result) => result.record.sequence ?? 0),
			1,
		);
		const ranked = candidates.flatMap((result) => {
			const confidenceValue = result.record.metadata?.confidence;
			const confidence =
				typeof confidenceValue === 'number' ? confidenceValue : 1;
			if (confidence < (this.options.minConfidence ?? 0)) return [];
			const recency = (result.record.sequence ?? 0) / maxSequence;
			const score =
				result.score * (this.options.relevanceWeight ?? 0.6) +
				recency * (this.options.recencyWeight ?? 0.2) +
				confidence * (this.options.confidenceWeight ?? 0.2);
			return [{ record: result.record, score }];
		});
		ranked.sort(
			(a, b) =>
				b.score - a.score ||
				(b.record.sequence ?? 0) - (a.record.sequence ?? 0) ||
				a.record.id.localeCompare(b.record.id),
		);
		return ranked.slice(0, limit);
	}
}

export class ScopedMemoryStore implements LongTermMemoryStore {
	private readonly prefix: string;
	constructor(
		readonly store: LongTermMemoryStore,
		readonly scope: MemoryScope,
	) {
		this.prefix =
			scope.tenant !== undefined
				? JSON.stringify(['memory', scope.tenant]) + '::'
				: '';
	}

	private assertScope(scope: MemoryScope): void {
		if (!scopeContains(this.scope, scope))
			throw new Error('memory scope is outside the bound scope');
	}

	private key(id: string): string {
		return this.prefix + id;
	}

	private visible(
		record: MemoryRecord | undefined,
	): MemoryRecord | undefined {
		if (
			!record ||
			!scopeContains(this.scope, record.scope ?? {}) ||
			!record.id.startsWith(this.prefix)
		)
			return undefined;
		return { ...record, id: record.id.slice(this.prefix.length) };
	}

	read(memoryId: string): MemoryRecord | undefined {
		return this.visible(this.store.read(this.key(memoryId)));
	}

	write(record: MemoryRecord): MemoryRecord {
		this.assertScope(record.scope ?? {});
		return {
			...this.store.write({ ...record, id: this.key(record.id) }),
			id: record.id,
		};
	}

	update(memoryId: string, record: MemoryRecord): MemoryRecord {
		if (!this.read(memoryId))
			throw new Error('memory not found: ' + memoryId);
		this.assertScope(record.scope ?? {});
		return {
			...this.store.update(this.key(memoryId), {
				...record,
				id: this.key(record.id),
			}),
			id: record.id,
		};
	}

	delete(memoryId: string): boolean {
		if (!this.read(memoryId)) return false;
		return this.store.delete(this.key(memoryId));
	}

	search(query: MemorySearchQuery): MemorySearchResult[] {
		const effectiveScope = query.scope ?? this.scope;
		if (!scopeContains(this.scope, effectiveScope))
			throw new Error('memory query scope is outside the bound scope');
		return this.store
			.search({ ...query, scope: effectiveScope })
			.flatMap((result) => {
				const record = this.visible(result.record);
				return record ? [{ ...result, record }] : [];
			});
	}
}

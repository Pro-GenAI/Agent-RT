/** Memory lifecycle policies and procedural memory. */
import { type JSONValue, type MemoryScope } from '../types.js';
import {
	type LongTermMemoryStore,
	type MemoryKind,
	type MemoryRecord,
	type MemorySearchQuery,
	type MemorySearchResult,
} from './long_term.js';

export type MemoryMigration = (record: MemoryRecord) => MemoryRecord;

export interface MemoryLifecyclePolicy {
	defaultRetentionMs?: number;
}

export class LifecycleMemoryStore implements LongTermMemoryStore {
	private readonly migrations = new Map<number, MemoryMigration>();

	constructor(
		readonly store: LongTermMemoryStore,
		readonly policy: MemoryLifecyclePolicy = {},
		readonly clock: () => number = Date.now,
	) {
		if (
			policy.defaultRetentionMs !== undefined &&
			(!Number.isInteger(policy.defaultRetentionMs) ||
				policy.defaultRetentionMs < 1)
		) {
			throw new Error('defaultRetentionMs must be a positive integer');
		}
	}

	registerMigration(fromVersion: number, migration: MemoryMigration): void {
		if (!Number.isInteger(fromVersion) || fromVersion < 1) {
			throw new Error(
				'migration version must be an integer of at least 1',
			);
		}
		this.migrations.set(fromVersion, migration);
	}

	private expired(record: MemoryRecord): boolean {
		return (
			record.expiresAtMs !== undefined &&
			record.expiresAtMs <= this.clock()
		);
	}

	private applyDefaultRetention(record: MemoryRecord): MemoryRecord {
		if (
			record.expiresAtMs !== undefined ||
			this.policy.defaultRetentionMs === undefined
		) {
			return record;
		}
		const createdAtMs = record.createdAtMs ?? this.clock();
		return {
			...record,
			createdAtMs,
			expiresAtMs: createdAtMs + this.policy.defaultRetentionMs,
		};
	}

	read(memoryId: string): MemoryRecord | undefined {
		const record = this.store.read(memoryId);
		if (!record) return undefined;
		if (this.expired(record)) {
			this.store.delete(memoryId);
			return undefined;
		}
		return record;
	}

	write(record: MemoryRecord): MemoryRecord {
		return this.store.write(this.applyDefaultRetention(record));
	}

	update(memoryId: string, record: MemoryRecord): MemoryRecord {
		return this.store.update(memoryId, this.applyDefaultRetention(record));
	}

	delete(memoryId: string): boolean {
		return this.store.delete(memoryId);
	}

	search(query: MemorySearchQuery): MemorySearchResult[] {
		this.purgeExpired();
		return this.store.search(query);
	}

	purgeExpired(): number {
		const expired = this.store
			.search({ limit: Number.MAX_SAFE_INTEGER })
			.map((result) => result.record)
			.filter((record) => this.expired(record))
			.map((record) => record.id);
		for (const memoryId of expired) this.store.delete(memoryId);
		return expired.length;
	}

	migrate(memoryId: string, targetVersion: number): MemoryRecord {
		let record = this.read(memoryId);
		if (!record) throw new Error('memory not found: ' + memoryId);
		if (targetVersion < (record.schemaVersion ?? 1)) {
			throw new Error('memory migrations cannot move backwards');
		}
		while ((record.schemaVersion ?? 1) < targetVersion) {
			const currentVersion = record.schemaVersion ?? 1;
			const migration = this.migrations.get(currentVersion);
			if (!migration) {
				throw new Error(
					'no migration registered from version ' + currentVersion,
				);
			}
			const migrated = migration(record);
			if (migrated.id !== record.id) {
				throw new Error('memory migration cannot change id');
			}
			if ((migrated.schemaVersion ?? 1) !== currentVersion + 1) {
				throw new Error(
					'memory migration must advance exactly one version',
				);
			}
			record = migrated;
		}
		return this.store.update(memoryId, record);
	}

	compact(
		memoryIds: string[],
		options: {
			compactedId: string;
			content: string;
			kind?: MemoryKind;
			scope?: MemoryScope;
			metadata?: Record<string, JSONValue>;
			tags?: string[];
			deleteSources?: boolean;
		},
	): MemoryRecord {
		if (memoryIds.length === 0) {
			throw new Error('memory compaction requires at least one source');
		}
		for (const memoryId of memoryIds) {
			if (!this.read(memoryId)) {
				throw new Error('memory not found: ' + memoryId);
			}
		}
		const result = this.write({
			id: options.compactedId,
			kind: options.kind ?? 'fact',
			content: options.content,
			scope: options.scope ?? {},
			metadata: {
				...(options.metadata ?? {}),
				compactedFrom: [...memoryIds],
			},
			tags: options.tags ?? [],
		});
		if (options.deleteSources !== false) {
			for (const memoryId of memoryIds) this.store.delete(memoryId);
		}
		return result;
	}
}

export interface Procedure {
	id: string;
	name: string;
	instructions: string;
	scope?: MemoryScope;
	script?: string;
	template?: string;
	metadata?: Record<string, JSONValue>;
	tags?: string[];
}

export class ProceduralMemory {
	constructor(readonly store: LongTermMemoryStore) {}

	remember(procedure: Procedure): MemoryRecord {
		if (!procedure.id.trim())
			throw new Error('procedure id must not be empty');
		if (!procedure.name.trim())
			throw new Error('procedure name must not be empty');
		if (!procedure.instructions.trim()) {
			throw new Error('procedure instructions must not be empty');
		}
		const payload = {
			name: procedure.name,
			instructions: procedure.instructions,
			script: procedure.script ?? null,
			template: procedure.template ?? null,
		};
		const content = [
			procedure.name,
			procedure.instructions,
			procedure.script ?? '',
			procedure.template ?? '',
		]
			.filter(Boolean)
			.join(' ');
		return this.store.write({
			id: procedure.id,
			kind: 'procedure',
			content,
			scope: procedure.scope ?? {},
			tags: ['procedure', ...(procedure.tags ?? [])],
			metadata: {
				...(procedure.metadata ?? {}),
				procedure: payload,
			},
		});
	}

	search(
		text: string,
		options: { scope?: MemoryScope; limit?: number } = {},
	): MemorySearchResult[] {
		return this.store.search({
			text,
			scope: options.scope,
			kinds: ['procedure'],
			limit: options.limit ?? 10,
		});
	}
}

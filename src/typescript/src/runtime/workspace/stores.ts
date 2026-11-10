/** Workspace records and persistent, tenant-scoped workspace stores. */
import { type JSONValue, validateJSONValue } from '../types.js';
import { type TenantContext } from '../authorization/grants.js';
import {
	type FileSystem,
	FileSystemBackendRegistry,
	InMemoryFileSystem,
	WorkspaceFiles,
} from './filesystem.js';

export interface WorkspaceRecord {
	workspaceId: string;
	backend: string;
	metadata?: Record<string, JSONValue>;
	createdAtMs: number;
}

export class PersistentWorkspaceStore {
	private readonly records = new Map<string, WorkspaceRecord>();
	private readonly filesystems = new Map<string, FileSystem>();

	constructor(readonly backends = new FileSystemBackendRegistry()) {
		if (!this.backends.list().includes('memory')) {
			this.backends.register('memory', () => new InMemoryFileSystem());
		}
	}

	create(
		workspaceId: string,
		options: {
			backend?: string;
			metadata?: Record<string, JSONValue>;
			createdAtMs?: number;
		} = {},
	): WorkspaceRecord {
		if (!workspaceId.trim())
			throw new Error('workspaceId must not be empty');
		if (this.records.has(workspaceId)) {
			throw new Error('workspace already exists: ' + workspaceId);
		}
		validateJSONValue(options.metadata ?? {});
		const backend = options.backend ?? 'memory';
		const record: WorkspaceRecord = {
			workspaceId,
			backend,
			metadata: options.metadata ?? {},
			createdAtMs: options.createdAtMs ?? Date.now(),
		};
		const filesystem = this.backends.create(backend, workspaceId);
		this.records.set(workspaceId, record);
		this.filesystems.set(workspaceId, filesystem);
		return record;
	}

	get(workspaceId: string): WorkspaceRecord | undefined {
		return this.records.get(workspaceId);
	}

	open(workspaceId: string): WorkspaceFiles {
		const filesystem = this.filesystems.get(workspaceId);
		if (!filesystem) throw new Error('workspace not found: ' + workspaceId);
		return new WorkspaceFiles(filesystem);
	}

	delete(workspaceId: string): boolean {
		const existed = this.records.delete(workspaceId);
		this.filesystems.delete(workspaceId);
		return existed;
	}

	list(): WorkspaceRecord[] {
		return [...this.records.values()].sort((a, b) =>
			a.workspaceId.localeCompare(b.workspaceId),
		);
	}
}

export class TenantWorkspaceStore {
	constructor(
		readonly store: PersistentWorkspaceStore,
		readonly tenant: TenantContext,
	) {}

	create(
		workspaceId: string,
		options: {
			backend?: string;
			metadata?: Record<string, JSONValue>;
			createdAtMs?: number;
		} = {},
	): WorkspaceRecord {
		return this.store.create(this.tenant.workspaceId(workspaceId), {
			...options,
			metadata: {
				...(options.metadata ?? {}),
				tenantId: this.tenant.tenantId,
			},
		});
	}

	get(workspaceId: string): WorkspaceRecord | undefined {
		return this.store.get(this.tenant.workspaceId(workspaceId));
	}

	open(workspaceId: string): WorkspaceFiles {
		return this.store.open(this.tenant.workspaceId(workspaceId));
	}

	delete(workspaceId: string): boolean {
		return this.store.delete(this.tenant.workspaceId(workspaceId));
	}

	list(): WorkspaceRecord[] {
		const prefix = this.tenant.tenantId + '::workspace::';
		return this.store
			.list()
			.filter((record) => record.workspaceId.startsWith(prefix));
	}
}

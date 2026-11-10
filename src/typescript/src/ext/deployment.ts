export type DeploymentKind =
	'local' | 'container' | 'kubernetes' | 'serverless' | 'managed';

const DEPLOYMENT_KINDS = new Set<string>([
	'local',
	'container',
	'kubernetes',
	'serverless',
	'managed',
]);

function assertDeploymentKind(kind: string): asserts kind is DeploymentKind {
	if (!DEPLOYMENT_KINDS.has(kind)) {
		throw new Error('unsupported deployment kind: ' + kind);
	}
}

export interface DeploymentSpec {
	kind: DeploymentKind;
	name: string;
	config?: Record<string, unknown>;
}

export interface DeploymentRuntime {
	execute(payload: unknown): Promise<unknown>;
}

export class DeploymentRuntimeRegistry {
	private readonly runtimes = new Map<
		DeploymentKind,
		(spec: DeploymentSpec) => DeploymentRuntime
	>();

	register(
		kind: DeploymentKind,
		factory: (spec: DeploymentSpec) => DeploymentRuntime,
		replaceExisting = false,
	): void {
		assertDeploymentKind(kind);
		if (this.runtimes.has(kind) && !replaceExisting) {
			throw new Error('deployment runtime already registered: ' + kind);
		}
		this.runtimes.set(kind, factory);
	}

	create(spec: DeploymentSpec): DeploymentRuntime {
		assertDeploymentKind(spec.kind);
		if (!spec.name.trim())
			throw new Error('deployment name must not be empty');
		const factory = this.runtimes.get(spec.kind);
		if (!factory)
			throw new Error('unknown deployment runtime ' + spec.kind);
		return factory(spec);
	}
}

export interface DistributedTask {
	id: string;
	kind: string;
	payload: unknown;
	affinity?: string;
}

export interface DistributedTaskResult {
	taskId: string;
	workerId: string;
	value: unknown;
}

export class DistributedWorker {
	readonly kinds: Set<string>;

	constructor(
		readonly workerId: string,
		readonly handler: (task: DistributedTask) => Promise<unknown>,
		kinds: string[] = [],
	) {
		if (!workerId.trim()) throw new Error('workerId must not be empty');
		this.kinds = new Set(kinds);
	}

	supports(task: DistributedTask): boolean {
		return this.kinds.size === 0 || this.kinds.has(task.kind);
	}

	async execute(task: DistributedTask): Promise<DistributedTaskResult> {
		return {
			taskId: task.id,
			workerId: this.workerId,
			value: await this.handler(task),
		};
	}
}

export class DistributedExecutor {
	private readonly workers = new Map<string, DistributedWorker>();
	private next = 0;

	register(worker: DistributedWorker): void {
		if (this.workers.has(worker.workerId)) {
			throw new Error('worker already registered: ' + worker.workerId);
		}
		this.workers.set(worker.workerId, worker);
	}

	async execute(task: DistributedTask): Promise<DistributedTaskResult> {
		if (!task.id.trim())
			throw new Error('distributed task id must not be empty');
		let workers = [...this.workers.values()].filter((worker) =>
			worker.supports(task),
		);
		if (task.affinity !== undefined) {
			workers = workers.filter(
				(worker) => worker.workerId === task.affinity,
			);
		}
		if (workers.length === 0) {
			throw new Error('no worker available for task kind ' + task.kind);
		}
		workers.sort((a, b) => a.workerId.localeCompare(b.workerId));
		const worker = workers[this.next % workers.length];
		this.next += 1;
		return worker.execute(task);
	}

	async map(tasks: DistributedTask[]): Promise<DistributedTaskResult[]> {
		return Promise.all(tasks.map((task) => this.execute(task)));
	}
}

export interface SharedStateStore {
	get(key: string): Promise<unknown>;
	set(key: string, value: unknown): Promise<void>;
}

export class InMemorySharedStateStore implements SharedStateStore {
	private readonly values = new Map<string, unknown>();

	async get(key: string): Promise<unknown> {
		const value = this.values.get(key);
		return value === undefined ? undefined : structuredClone(value);
	}

	async set(key: string, value: unknown): Promise<void> {
		this.values.set(key, structuredClone(value));
	}
}

export class StatelessRuntimeNode {
	constructor(
		readonly nodeId: string,
		readonly stateStore: SharedStateStore,
	) {}

	async load(key: string): Promise<unknown> {
		return this.stateStore.get(key);
	}

	async save(key: string, value: unknown): Promise<void> {
		await this.stateStore.set(key, value);
	}
}

export interface TenantQuota {
	maxConcurrency?: number;
	maxStorageBytes?: number;
	maxModelCalls?: number;
	maxToolCalls?: number;
	maxCost?: number;
}

export interface TenantUsage {
	concurrency: number;
	storageBytes: number;
	modelCalls: number;
	toolCalls: number;
	cost: number;
}

export class TenantQuotaExceeded extends Error {}

export class TenantQuotaManager {
	private readonly quotas = new Map<string, TenantQuota>();
	private readonly usages = new Map<string, TenantUsage>();

	setQuota(tenantId: string, quota: TenantQuota): void {
		if (!tenantId.trim()) throw new Error('tenantId must not be empty');
		for (const [label, value] of Object.entries(quota)) {
			if (value === undefined) continue;
			if (!Number.isFinite(value)) {
				throw new Error(label + ' must be finite');
			}
			if (label !== 'maxCost' && !Number.isInteger(value)) {
				throw new Error(label + ' must be an integer');
			}
			if (value < 0) {
				throw new Error(label + ' must be non-negative');
			}
		}
		this.quotas.set(tenantId, { ...quota });
		if (!this.usages.has(tenantId)) {
			this.usages.set(tenantId, {
				concurrency: 0,
				storageBytes: 0,
				modelCalls: 0,
				toolCalls: 0,
				cost: 0,
			});
		}
	}

	usage(tenantId: string): TenantUsage {
		const usage = this.usages.get(tenantId) ?? {
			concurrency: 0,
			storageBytes: 0,
			modelCalls: 0,
			toolCalls: 0,
			cost: 0,
		};
		return { ...usage };
	}

	consume(tenantId: string, delta: Partial<TenantUsage>): TenantUsage {
		for (const [label, value] of Object.entries(delta)) {
			if (!Number.isFinite(value)) {
				throw new Error(label + ' delta must be finite');
			}
			if (label !== 'cost' && !Number.isInteger(value)) {
				throw new Error(label + ' delta must be an integer');
			}
		}
		const current = this.usage(tenantId);
		const proposed: TenantUsage = {
			concurrency: current.concurrency + (delta.concurrency ?? 0),
			storageBytes: current.storageBytes + (delta.storageBytes ?? 0),
			modelCalls: current.modelCalls + (delta.modelCalls ?? 0),
			toolCalls: current.toolCalls + (delta.toolCalls ?? 0),
			cost: current.cost + (delta.cost ?? 0),
		};
		const quota = this.quotas.get(tenantId) ?? {};
		const checks: Array<[string, number, number | undefined]> = [
			['concurrency', proposed.concurrency, quota.maxConcurrency],
			['storageBytes', proposed.storageBytes, quota.maxStorageBytes],
			['modelCalls', proposed.modelCalls, quota.maxModelCalls],
			['toolCalls', proposed.toolCalls, quota.maxToolCalls],
			['cost', proposed.cost, quota.maxCost],
		];
		for (const [label, value, limit] of checks) {
			if (value < 0) {
				throw new Error(
					'tenant ' +
						tenantId +
						' usage ' +
						label +
						' must not be negative',
				);
			}
			if (limit !== undefined && value > limit) {
				throw new TenantQuotaExceeded(
					'tenant ' +
						tenantId +
						' exceeded ' +
						label +
						': ' +
						value +
						' > ' +
						limit,
				);
			}
		}
		this.usages.set(tenantId, proposed);
		return { ...proposed };
	}

	release(
		tenantId: string,
		delta: Pick<Partial<TenantUsage>, 'concurrency' | 'storageBytes'>,
	): TenantUsage {
		const concurrency = delta.concurrency ?? 0;
		const storageBytes = delta.storageBytes ?? 0;
		if (concurrency < 0 || storageBytes < 0) {
			throw new Error('release amounts must be non-negative');
		}
		// Releasing never enforces quota limits: an operator lowering a quota
		// while work is in flight must not strand the usage that work holds.
		const current = this.usages.get(tenantId) ?? {
			concurrency: 0,
			storageBytes: 0,
			modelCalls: 0,
			toolCalls: 0,
			cost: 0,
		};
		if (
			concurrency > current.concurrency ||
			storageBytes > current.storageBytes
		) {
			throw new Error(
				'tenant ' + tenantId + ' cannot release more than it holds',
			);
		}
		const released = {
			...current,
			concurrency: current.concurrency - concurrency,
			storageBytes: current.storageBytes - storageBytes,
		};
		this.usages.set(tenantId, released);
		return { ...released };
	}
}

export interface PersistenceSchema {
	name: string;
	version: number;
	validator?: (value: Record<string, unknown>) => void;
}

export class PersistenceSchemaRegistry {
	private readonly schemas = new Map<string, PersistenceSchema>();

	register(schema: PersistenceSchema): void {
		if (!schema.name.trim())
			throw new Error('schema name must not be empty');
		if (!Number.isInteger(schema.version))
			throw new Error('schema version must be an integer');
		if (schema.version < 1)
			throw new Error('schema version must be at least 1');
		const key = schema.name + '::' + schema.version;
		if (this.schemas.has(key))
			throw new Error('schema already registered: ' + key);
		this.schemas.set(key, schema);
	}

	get(name: string, version?: number): PersistenceSchema {
		if (version !== undefined) {
			const schema = this.schemas.get(name + '::' + version);
			if (!schema)
				throw new Error('unknown schema ' + name + '@' + version);
			return schema;
		}
		const versions = [...this.schemas.values()]
			.filter((schema) => schema.name === name)
			.sort((a, b) => b.version - a.version);
		if (versions.length === 0) throw new Error('unknown schema ' + name);
		return versions[0];
	}

	validate(
		name: string,
		version: number,
		value: Record<string, unknown>,
	): void {
		this.get(name, version).validator?.(value);
	}
}

export type MigrationFn = (
	value: Record<string, unknown>,
) => Record<string, unknown>;

export class MigrationRegistry {
	private readonly migrations = new Map<string, MigrationFn>();

	register(
		schemaName: string,
		fromVersion: number,
		migration: MigrationFn,
	): void {
		if (!schemaName.trim()) throw new Error('schemaName must not be empty');
		if (!Number.isInteger(fromVersion))
			throw new Error('fromVersion must be an integer');
		if (fromVersion < 1) throw new Error('fromVersion must be at least 1');
		const key = schemaName + '::' + fromVersion;
		if (this.migrations.has(key)) {
			throw new Error('migration already registered: ' + key);
		}
		this.migrations.set(key, migration);
	}

	migrate(
		schemaName: string,
		value: Record<string, unknown>,
		fromVersion: number,
		toVersion: number,
	): Record<string, unknown> {
		if (!schemaName.trim()) throw new Error('schemaName must not be empty');
		if (!Number.isInteger(fromVersion) || !Number.isInteger(toVersion)) {
			throw new Error('migration versions must be integers');
		}
		if (fromVersion < 1 || toVersion < 1) {
			throw new Error('migration versions must be at least 1');
		}
		if (toVersion < fromVersion) {
			throw new Error('downgrade migrations are not supported');
		}
		let current = structuredClone(value);
		for (let version = fromVersion; version < toVersion; version += 1) {
			const migration = this.migrations.get(schemaName + '::' + version);
			if (!migration) {
				throw new Error(
					'missing migration ' + schemaName + '@' + version,
				);
			}
			current = migration(current);
		}
		return current;
	}

	migrateValidated(
		schemas: PersistenceSchemaRegistry,
		schemaName: string,
		value: Record<string, unknown>,
		fromVersion: number,
		toVersion: number,
	): Record<string, unknown> {
		schemas.validate(schemaName, fromVersion, value);
		let current = structuredClone(value);
		for (let version = fromVersion; version < toVersion; version += 1) {
			current = this.migrate(schemaName, current, version, version + 1);
			schemas.validate(schemaName, version + 1, current);
		}
		return current;
	}
}

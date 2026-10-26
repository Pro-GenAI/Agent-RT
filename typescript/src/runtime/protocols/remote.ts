/** Remote agents: cards, directories, tasks, artifacts, transports, and clients. */

export interface RemoteAgentCard {
	id: string;
	name: string;
	endpoint: string;
	description?: string;
	modalities?: string[];
	capabilities?: string[];
	metadata?: Record<string, unknown>;
}

export interface RemoteAgentDirectory {
	discover(): Promise<RemoteAgentCard[]>;
}

export class RemoteAgentRegistry {
	private readonly agents = new Map<string, RemoteAgentCard>();

	constructor(readonly directory: RemoteAgentDirectory) {}

	async refresh(): Promise<RemoteAgentCard[]> {
		const discovered = await this.directory.discover();
		this.agents.clear();
		for (const agent of discovered) this.agents.set(agent.id, { ...agent });
		return discovered;
	}

	get(agentId: string): RemoteAgentCard {
		const agent = this.agents.get(agentId);
		if (!agent) throw new Error('unknown remote agent: ' + agentId);
		return agent;
	}
}

export interface RemoteMessage {
	id: string;
	role: string;
	parts: Record<string, unknown>[];
	metadata?: Record<string, unknown>;
}

export type RemoteTaskStatus =
	| 'submitted'
	| 'working'
	| 'input_required'
	| 'completed'
	| 'failed'
	| 'canceled';

export interface RemoteTask {
	id: string;
	agentId: string;
	status: RemoteTaskStatus;
	error?: string;
	version: number;
}

export class RemoteTaskStore {
	private readonly tasks = new Map<string, RemoteTask>();

	upsert(task: RemoteTask): RemoteTask {
		const existing = this.tasks.get(task.id);
		if (existing && task.version < existing.version) {
			throw new Error('remote task version cannot move backwards');
		}
		this.tasks.set(task.id, { ...task });
		return task;
	}

	get(taskId: string): RemoteTask {
		const task = this.tasks.get(taskId);
		if (!task) throw new Error('unknown remote task: ' + taskId);
		return task;
	}
}

export interface RemoteArtifact {
	id: string;
	taskId: string;
	name: string;
	mediaType: string;
	data: unknown;
	version: number;
	complete: boolean;
	metadata?: Record<string, unknown>;
}

export class RemoteArtifactStore {
	private readonly artifacts = new Map<string, RemoteArtifact[]>();

	put(artifact: RemoteArtifact): RemoteArtifact {
		const versions = this.artifacts.get(artifact.id) ?? [];
		if (
			versions.length &&
			artifact.version <= versions[versions.length - 1].version
		) {
			throw new Error('remote artifact version must increase');
		}
		versions.push({ ...artifact });
		this.artifacts.set(artifact.id, versions);
		return artifact;
	}

	latest(artifactId: string): RemoteArtifact {
		const versions = this.artifacts.get(artifactId);
		if (!versions?.length)
			throw new Error('unknown remote artifact: ' + artifactId);
		return versions[versions.length - 1];
	}
}

export interface RemoteEvent {
	type: 'status' | 'message' | 'artifact' | 'callback';
	taskId: string;
	payload: Record<string, unknown>;
}

export interface RemoteAgentTransport {
	sendMessage(
		agent: RemoteAgentCard,
		message: RemoteMessage,
	): Promise<Record<string, unknown>>;
	getTask(
		agent: RemoteAgentCard,
		taskId: string,
	): Promise<Record<string, unknown>>;
	cancelTask(
		agent: RemoteAgentCard,
		taskId: string,
	): Promise<Record<string, unknown>>;
	streamTask(
		agent: RemoteAgentCard,
		taskId: string,
	): AsyncIterable<Record<string, unknown>>;
}

export class CapabilityNegotiation {
	constructor(
		readonly local: Set<string>,
		readonly remote: Set<string>,
		readonly required: Set<string> = new Set(),
	) {}

	get negotiated(): Set<string> {
		const available = new Set(
			[...this.local].filter((capability) => this.remote.has(capability)),
		);
		const missing = [...this.required].filter(
			(capability) => !available.has(capability),
		);
		if (missing.length) {
			throw new Error(
				'required capabilities unavailable: ' +
					missing.sort().join(','),
			);
		}
		return available;
	}
}

export class RemoteAgentClient {
	readonly taskStore: RemoteTaskStore;
	readonly artifactStore: RemoteArtifactStore;
	readonly localCapabilities: Set<string>;

	constructor(
		readonly registry: RemoteAgentRegistry,
		readonly transport: RemoteAgentTransport,
		options: {
			taskStore?: RemoteTaskStore;
			artifactStore?: RemoteArtifactStore;
			localCapabilities?: string[];
		} = {},
	) {
		this.taskStore = options.taskStore ?? new RemoteTaskStore();
		this.artifactStore = options.artifactStore ?? new RemoteArtifactStore();
		this.localCapabilities = new Set(options.localCapabilities ?? []);
	}

	negotiate(agentId: string, required: string[] = []): Set<string> {
		const card = this.registry.get(agentId);
		return new CapabilityNegotiation(
			this.localCapabilities,
			new Set(card.capabilities ?? []),
			new Set(required),
		).negotiated;
	}

	private parseTask(
		agentId: string,
		payload: Record<string, unknown>,
	): RemoteTask {
		const status = String(payload.status ?? 'submitted');
		const allowed = new Set([
			'submitted',
			'working',
			'input_required',
			'completed',
			'failed',
			'canceled',
		]);
		if (!allowed.has(status)) {
			throw new Error('unsupported remote task status: ' + status);
		}
		const id = String(payload.id ?? '').trim();
		if (!id) throw new Error('remote task id is required');
		return {
			id,
			agentId,
			status: status as RemoteTaskStatus,
			error:
				payload.error === undefined ? undefined : String(payload.error),
			version: Number(payload.version ?? 0),
		};
	}

	private persistArtifact(
		taskId: string,
		payload: Record<string, unknown>,
	): RemoteArtifact {
		const artifact: RemoteArtifact = {
			id: String(payload.id),
			taskId,
			name: String(payload.name ?? payload.id),
			mediaType: String(payload.mediaType ?? 'application/octet-stream'),
			data: payload.data,
			version: Number(payload.version ?? 1),
			complete: Boolean(payload.complete ?? true),
			metadata:
				payload.metadata &&
				typeof payload.metadata === 'object' &&
				!Array.isArray(payload.metadata)
					? { ...(payload.metadata as Record<string, unknown>) }
					: {},
		};
		return this.artifactStore.put(artifact);
	}

	async send(agentId: string, message: RemoteMessage): Promise<RemoteTask> {
		const card = this.registry.get(agentId);
		const payload = await this.transport.sendMessage(card, message);
		const task = this.taskStore.upsert(this.parseTask(agentId, payload));
		if (Array.isArray(payload.artifacts)) {
			for (const value of payload.artifacts) {
				if (
					value &&
					typeof value === 'object' &&
					!Array.isArray(value)
				) {
					this.persistArtifact(
						task.id,
						value as Record<string, unknown>,
					);
				}
			}
		}
		return task;
	}

	async refreshTask(agentId: string, taskId: string): Promise<RemoteTask> {
		const card = this.registry.get(agentId);
		const payload = await this.transport.getTask(card, taskId);
		return this.taskStore.upsert(this.parseTask(agentId, payload));
	}

	async cancel(agentId: string, taskId: string): Promise<RemoteTask> {
		const card = this.registry.get(agentId);
		const payload = await this.transport.cancelTask(card, taskId);
		return this.taskStore.upsert(this.parseTask(agentId, payload));
	}

	async stream(
		agentId: string,
		taskId: string,
		callback?: (event: RemoteEvent) => Promise<void>,
	): Promise<RemoteEvent[]> {
		const card = this.registry.get(agentId);
		const events: RemoteEvent[] = [];
		for await (const payload of this.transport.streamTask(card, taskId)) {
			const type = String(payload.type ?? 'status');
			if (!['status', 'message', 'artifact', 'callback'].includes(type))
				continue;
			const event: RemoteEvent = {
				type: type as RemoteEvent['type'],
				taskId,
				payload: { ...payload },
			};
			events.push(event);
			if (type === 'status') {
				this.taskStore.upsert(this.parseTask(agentId, payload));
			} else if (
				type === 'artifact' &&
				payload.artifact &&
				typeof payload.artifact === 'object' &&
				!Array.isArray(payload.artifact)
			) {
				this.persistArtifact(
					taskId,
					payload.artifact as Record<string, unknown>,
				);
			}
			if (callback) await callback(event);
		}
		return events;
	}
}

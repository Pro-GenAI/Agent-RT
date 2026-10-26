/** Task lifecycle and background task management. */
import { type EventStore } from '../checkpoints/events.js';

export type TaskStatus =
	| 'submitted'
	| 'queued'
	| 'running'
	| 'waiting_for_input'
	| 'waiting_for_approval'
	| 'completed'
	| 'failed'
	| 'canceled';

export interface TaskLifecycleState {
	taskId: string;
	status: TaskStatus;
	version: number;
}

export class TaskLifecycle {
	private static readonly transitions: Record<TaskStatus, Set<TaskStatus>> = {
		submitted: new Set(['queued', 'running', 'canceled']),
		queued: new Set(['running', 'canceled']),
		running: new Set([
			'waiting_for_input',
			'waiting_for_approval',
			'completed',
			'failed',
			'canceled',
		]),
		waiting_for_input: new Set(['running', 'canceled']),
		waiting_for_approval: new Set(['running', 'failed', 'canceled']),
		completed: new Set(),
		failed: new Set(),
		canceled: new Set(),
	};

	constructor(
		public state: TaskLifecycleState,
		readonly eventStore?: EventStore,
	) {
		if (!state.taskId.trim()) throw new Error('taskId must not be empty');
		if (!Number.isInteger(state.version) || state.version < 0) {
			throw new Error('task lifecycle version must be non-negative');
		}
	}

	transition(status: TaskStatus, reason?: string): TaskLifecycleState {
		if (status === this.state.status) return this.state;
		if (!TaskLifecycle.transitions[this.state.status].has(status)) {
			throw new Error(
				'invalid task transition: ' +
					this.state.status +
					' -> ' +
					status,
			);
		}
		const previous = this.state;
		this.state = {
			taskId: previous.taskId,
			status,
			version: previous.version + 1,
		};
		this.eventStore?.append({
			eventId: this.state.taskId + ':lifecycle:' + this.state.version,
			taskId: this.state.taskId,
			type: 'lifecycle_transition',
			payload: {
				from: previous.status,
				to: status,
				version: this.state.version,
				...(reason ? { reason } : {}),
			},
		});
		return this.state;
	}
}

export type BackgroundTaskRunner = (
	reportProgress: (progress: number, message?: string) => void,
	signal: AbortSignal,
) => Promise<unknown>;

export interface BackgroundTaskSnapshot {
	taskId: string;
	status: TaskStatus;
	progress: number;
	message?: string;
	result?: unknown;
	error?: string;
	submittedAtMs: number;
	startedAtMs?: number;
	completedAtMs?: number;
}

export class BackgroundTaskManager {
	private readonly snapshots = new Map<string, BackgroundTaskSnapshot>();
	private readonly controllers = new Map<string, AbortController>();
	private readonly promises = new Map<string, Promise<void>>();

	submit(
		taskId: string,
		runner: BackgroundTaskRunner,
	): BackgroundTaskSnapshot {
		if (!taskId.trim())
			throw new Error('background taskId must not be empty');
		if (this.snapshots.has(taskId)) {
			throw new Error('background task already exists: ' + taskId);
		}
		const submittedAtMs = Date.now();
		const initial: BackgroundTaskSnapshot = {
			taskId,
			status: 'queued',
			progress: 0,
			submittedAtMs,
		};
		this.snapshots.set(taskId, initial);
		const controller = new AbortController();
		this.controllers.set(taskId, controller);

		const promise = (async () => {
			const startedAtMs = Date.now();
			this.snapshots.set(taskId, {
				taskId,
				status: 'running',
				progress: 0,
				submittedAtMs,
				startedAtMs,
			});
			const report = (progress: number, message?: string) => {
				if (progress < 0 || progress > 1) {
					throw new Error(
						'background task progress must be between 0 and 1',
					);
				}
				const current = this.snapshots.get(taskId);
				if (!current || current.status !== 'running') return;
				this.snapshots.set(taskId, {
					...current,
					progress,
					message,
				});
			};

			try {
				const result = await runner(report, controller.signal);
				const current = this.snapshots.get(taskId)!;
				const canceled = controller.signal.aborted;
				this.snapshots.set(taskId, {
					...current,
					status: canceled ? 'canceled' : 'completed',
					progress: canceled ? current.progress : 1,
					result: canceled ? undefined : result,
					completedAtMs: Date.now(),
				});
			} catch (error) {
				const current = this.snapshots.get(taskId)!;
				if (controller.signal.aborted) {
					this.snapshots.set(taskId, {
						...current,
						status: 'canceled',
						completedAtMs: Date.now(),
					});
				} else {
					this.snapshots.set(taskId, {
						...current,
						status: 'failed',
						error:
							error instanceof Error
								? error.message
								: String(error),
						completedAtMs: Date.now(),
					});
				}
			}
		})();
		this.promises.set(taskId, promise);
		return initial;
	}

	get(taskId: string): BackgroundTaskSnapshot | undefined {
		return this.snapshots.get(taskId);
	}

	async wait(taskId: string): Promise<BackgroundTaskSnapshot> {
		const promise = this.promises.get(taskId);
		if (!promise) throw new Error('background task not found: ' + taskId);
		await promise;
		return this.snapshots.get(taskId)!;
	}

	cancel(taskId: string): boolean {
		const controller = this.controllers.get(taskId);
		const snapshot = this.snapshots.get(taskId);
		if (!controller || !snapshot) return false;
		if (['completed', 'failed', 'canceled'].includes(snapshot.status))
			return false;
		controller.abort(new Error('background_task_cancelled'));
		return true;
	}

	list(): BackgroundTaskSnapshot[] {
		return [...this.snapshots.values()].sort(
			(a, b) =>
				a.submittedAtMs - b.submittedAtMs ||
				a.taskId.localeCompare(b.taskId),
		);
	}
}

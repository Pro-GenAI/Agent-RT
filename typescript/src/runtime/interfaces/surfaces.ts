/** Progress reporting and CLI, API, IDE, and chat integration surfaces. */
import { type ModelMessage, type SessionRef } from '../types.js';
import {
	type SessionSnapshot,
	type ShortTermSessionMemory,
} from '../memory/sessions.js';
import {
	type ApprovalDecision,
	type ApprovalRequest,
} from '../authorization/approvals.js';
import { messageFromText } from '../agent/delegation.js';
import { type WorkspaceFiles } from '../workspace/filesystem.js';

export type ProgressStatus =
	| 'queued'
	| 'running'
	| 'waiting_for_input'
	| 'waiting_for_approval'
	| 'completed'
	| 'failed'
	| 'canceled';

export interface ProgressEvent {
	taskId: string;
	status: ProgressStatus;
	message?: string;
	activeStep?: string;
	completedSteps?: string[];
	pendingApprovalId?: string;
	waitingOn?: string;
	progress?: number;
	metadata?: Record<string, unknown>;
}

export class ProgressReporter {
	private readonly listeners: Array<(event: ProgressEvent) => void> = [];
	private readonly recorded: ProgressEvent[] = [];

	subscribe(listener: (event: ProgressEvent) => void): void {
		this.listeners.push(listener);
	}

	emit(event: ProgressEvent): ProgressEvent {
		if (
			event.progress !== undefined &&
			(event.progress < 0 || event.progress > 1)
		) {
			throw new Error('progress must be between 0 and 1');
		}
		const stored = {
			...event,
			completedSteps: [...(event.completedSteps ?? [])],
			metadata: { ...(event.metadata ?? {}) },
		};
		this.recorded.push(stored);
		for (const listener of [...this.listeners]) listener(stored);
		return stored;
	}

	get events(): ProgressEvent[] {
		return this.recorded.map((event) => ({
			...event,
			completedSteps: [...(event.completedSteps ?? [])],
			metadata: { ...(event.metadata ?? {}) },
		}));
	}
}

export interface RuntimeRequest {
	agent: string;
	messages: ModelMessage[];
	sessionId?: string;
	taskId?: string;
	structured?: boolean;
	stream?: boolean;
	metadata?: Record<string, unknown>;
}

export interface RuntimeResponse {
	taskId: string;
	sessionId?: string;
	result: unknown;
	events?: ProgressEvent[];
}

export type RuntimeExecutor = (request: RuntimeRequest) => Promise<unknown>;

export class CLIInterface {
	constructor(
		readonly executor: RuntimeExecutor,
		readonly sessionMemory?: ShortTermSessionMemory,
	) {}

	async run(
		request: RuntimeRequest,
		stdin?: string,
	): Promise<RuntimeResponse> {
		const effective: RuntimeRequest = {
			...request,
			messages: [
				...request.messages,
				...(stdin === undefined
					? []
					: [messageFromText('user', stdin)]),
			],
		};
		const result = await this.executor(effective);
		if (this.sessionMemory && effective.sessionId) {
			this.sessionMemory.appendMessages(
				{ sessionId: effective.sessionId, threadId: 'cli' },
				effective.messages,
			);
		}
		return {
			taskId: effective.taskId ?? 'cli-task',
			sessionId: effective.sessionId,
			result,
		};
	}

	resume(sessionId: string): SessionSnapshot | undefined {
		return this.sessionMemory?.snapshot({
			sessionId,
			threadId: 'cli',
		});
	}
}

export type APIOperation =
	| 'run'
	| 'session.get'
	| 'task.get'
	| 'task.cancel'
	| 'events.list'
	| 'artifact.get'
	| 'approval.resolve'
	| 'status.get';

export interface APIRequest {
	operation: APIOperation;
	payload?: Record<string, unknown>;
}

export class APIInterface {
	private readonly handlers = new Map<
		APIOperation,
		(payload: Record<string, unknown>) => Promise<unknown>
	>();

	register(
		operation: APIOperation,
		handler: (payload: Record<string, unknown>) => Promise<unknown>,
	): void {
		this.handlers.set(operation, handler);
	}

	async handle(request: APIRequest): Promise<unknown> {
		const handler = this.handlers.get(request.operation);
		if (!handler)
			throw new Error('unregistered API operation: ' + request.operation);
		return handler(request.payload ?? {});
	}
}

export interface IDEDiagnostic {
	path: string;
	message: string;
	severity?: 'info' | 'warning' | 'error';
	line?: number;
	column?: number;
}

export interface IDEContext {
	workspaceId: string;
	currentFile?: string;
	selection?: string;
	diagnostics?: IDEDiagnostic[];
	diff?: string;
	metadata?: Record<string, unknown>;
}

export interface IDEProgress {
	taskId: string;
	message: string;
	fraction?: number;
}

export class IDEIntegration {
	constructor(readonly workspace: WorkspaceFiles) {}

	readCurrent(context: IDEContext): string | undefined {
		if (!context.currentFile) return undefined;
		return this.workspace.readText(context.currentFile);
	}

	applyPatch(path: string, patch: string): void {
		this.workspace.applyUnifiedPatch(path, patch);
	}
}

export interface ChatEnvelope {
	channel: string;
	userId: string;
	threadId: string;
	text: string;
	messageId?: string;
	metadata?: Record<string, unknown>;
}

export interface ChatReply {
	channel: string;
	threadId: string;
	text: string;
	metadata?: Record<string, unknown>;
}

export interface ChatAdapter {
	send(reply: ChatReply): Promise<unknown>;
}

export class ChatSessionBridge {
	constructor(
		readonly adapter: ChatAdapter,
		readonly memory: ShortTermSessionMemory,
	) {}

	sessionRef(envelope: ChatEnvelope): SessionRef {
		return {
			sessionId: envelope.channel + ':' + envelope.userId,
			threadId: envelope.threadId,
		};
	}

	ingest(envelope: ChatEnvelope): SessionRef {
		const session = this.sessionRef(envelope);
		this.memory.appendMessages(session, [
			messageFromText('user', envelope.text),
		]);
		return session;
	}

	reply(envelope: ChatEnvelope, text: string): Promise<unknown> {
		return this.adapter.send({
			channel: envelope.channel,
			threadId: envelope.threadId,
			text,
		});
	}
}

export interface ApprovalPresentation {
	approvalId: string;
	title: string;
	summary: string;
	consequences?: string[];
	diff?: string;
	choices: ApprovalDecision[];
	metadata?: Record<string, unknown>;
}

export function approvalPresentation(
	request: ApprovalRequest,
	options: {
		title?: string;
		summary?: string;
		consequences?: string[];
		diff?: string;
	} = {},
): ApprovalPresentation {
	return {
		approvalId: request.id,
		title: options.title ?? 'Approve ' + request.call.name,
		summary: options.summary ?? request.reason,
		consequences: [...(options.consequences ?? [])],
		diff: options.diff,
		choices: ['allow', 'deny'],
		metadata: {
			tool: request.call.name,
			sideEffect: request.sideEffect,
			sessionId: request.sessionId,
		},
	};
}

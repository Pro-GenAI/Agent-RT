/** Trace spans, trace recording, and structured logging. */
import { type PrivacyRedactor } from '../guardrails/privacy.js';

export type SpanKind =
	| 'task'
	| 'agent'
	| 'turn'
	| 'model'
	| 'tool'
	| 'subagent'
	| 'guardrail'
	| 'queue'
	| 'remote';

export type SpanStatus = 'running' | 'ok' | 'error' | 'canceled';

export interface TraceSpan {
	spanId: string;
	traceId: string;
	name: string;
	kind: SpanKind;
	parentSpanId?: string;
	taskId?: string;
	sessionId?: string;
	status: SpanStatus;
	startedAtMs: number;
	endedAtMs?: number;
	attributes?: Record<string, unknown>;
}

export class TraceRecorder {
	private readonly spansById = new Map<string, TraceSpan>();
	private readonly order: string[] = [];

	start(
		input: Omit<TraceSpan, 'status' | 'startedAtMs'> & {
			status?: SpanStatus;
			startedAtMs?: number;
		},
	): TraceSpan {
		if (this.spansById.has(input.spanId)) {
			throw new Error('duplicate span id: ' + input.spanId);
		}
		if (input.parentSpanId) {
			const parent = this.spansById.get(input.parentSpanId);
			if (!parent)
				throw new Error('unknown parent span: ' + input.parentSpanId);
			if (parent.traceId !== input.traceId) {
				throw new Error('child span traceId must match parent');
			}
		}
		const span: TraceSpan = {
			...input,
			status: input.status ?? 'running',
			startedAtMs: input.startedAtMs ?? Date.now(),
			attributes: { ...(input.attributes ?? {}) },
		};
		this.spansById.set(span.spanId, span);
		this.order.push(span.spanId);
		return span;
	}

	finish(
		spanId: string,
		options: {
			status?: SpanStatus;
			endedAtMs?: number;
			attributes?: Record<string, unknown>;
		} = {},
	): TraceSpan {
		const span = this.spansById.get(spanId);
		if (!span) throw new Error('unknown span: ' + spanId);
		const finished: TraceSpan = {
			...span,
			status: options.status ?? 'ok',
			endedAtMs: options.endedAtMs ?? Date.now(),
			attributes: {
				...(span.attributes ?? {}),
				...(options.attributes ?? {}),
			},
		};
		this.spansById.set(spanId, finished);
		return finished;
	}

	get(spanId: string): TraceSpan {
		const span = this.spansById.get(spanId);
		if (!span) throw new Error('unknown span: ' + spanId);
		return span;
	}

	list(
		options: { traceId?: string; parentSpanId?: string } = {},
	): TraceSpan[] {
		return this.order
			.map((id) => this.spansById.get(id)!)
			.filter(
				(span) =>
					(options.traceId === undefined ||
						span.traceId === options.traceId) &&
					(options.parentSpanId === undefined ||
						span.parentSpanId === options.parentSpanId),
			);
	}
}

export type LogSeverity = 'debug' | 'info' | 'warning' | 'error';

export interface StructuredLogRecord {
	message: string;
	severity?: LogSeverity;
	correlationId?: string;
	taskId?: string;
	sessionId?: string;
	metadata?: Record<string, unknown>;
	occurredAtMs?: number;
}

export class StructuredLogger {
	private readonly recorded: StructuredLogRecord[] = [];

	constructor(readonly redactor?: PrivacyRedactor) {}

	emit(record: StructuredLogRecord): StructuredLogRecord {
		let stored: StructuredLogRecord = {
			...record,
			severity: record.severity ?? 'info',
			occurredAtMs: record.occurredAtMs ?? Date.now(),
			metadata: { ...(record.metadata ?? {}) },
		};
		if (this.redactor) {
			stored = {
				...stored,
				message: String(this.redactor.redact(stored.message)),
				metadata: this.redactor.redact(stored.metadata ?? {}) as Record<
					string,
					unknown
				>,
			};
		}
		this.recorded.push(stored);
		return stored;
	}

	get records(): StructuredLogRecord[] {
		return this.recorded.map((record) => ({
			...record,
			metadata: { ...(record.metadata ?? {}) },
		}));
	}
}

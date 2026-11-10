/** Shared runtime primitives: JSON values, messages, tool calls, sessions, and cancellation. */

export type JSONScalar = string | number | boolean | null;

export type JSONValue = JSONScalar | JSONValue[] | { [key: string]: JSONValue };

export function validateJSONValue(
	value: unknown,
	path = '$',
): asserts value is JSONValue {
	if (
		value === null ||
		typeof value === 'string' ||
		typeof value === 'boolean'
	)
		return;
	if (typeof value === 'number') {
		if (!Number.isFinite(value)) {
			throw new Error(`${path}: non-finite numbers are not serializable`);
		}
		return;
	}
	if (Array.isArray(value)) {
		value.forEach((item, index) =>
			validateJSONValue(item, `${path}[${index}]`),
		);
		return;
	}
	if (typeof value === 'object') {
		const prototype = Object.getPrototypeOf(value);
		if (prototype !== Object.prototype && prototype !== null) {
			throw new Error(
				`${path}: workflow state objects must be plain JSON objects`,
			);
		}
		for (const [key, item] of Object.entries(
			value as Record<string, unknown>,
		)) {
			validateJSONValue(item, `${path}.${key}`);
		}
		return;
	}
	throw new Error(
		`${path}: unsupported workflow state value type ${typeof value}`,
	);
}

export interface WorkflowStateValue<
	TData extends Record<string, JSONValue> = Record<string, JSONValue>,
> {
	stateType: string;
	version: number;
	data: TData;
}

export class WorkflowState<
	TData extends Record<string, JSONValue> = Record<string, JSONValue>,
> {
	readonly stateType: string;
	readonly version: number;
	readonly data: TData;

	constructor(value: WorkflowStateValue<TData>) {
		if (!value.stateType.trim()) {
			throw new Error('workflow stateType must not be empty');
		}
		if (!Number.isInteger(value.version) || value.version < 1) {
			throw new Error(
				'workflow state version must be an integer of at least 1',
			);
		}
		validateJSONValue(value.data);
		this.stateType = value.stateType;
		this.version = value.version;
		this.data = value.data;
	}

	toJSONValue(): WorkflowStateValue<TData> {
		const value = {
			stateType: this.stateType,
			version: this.version,
			data: this.data,
		};
		validateJSONValue(value);
		return value;
	}

	toJSON(): string {
		return JSON.stringify(this.toJSONValue());
	}

	static fromJSONValue<TData extends Record<string, JSONValue>>(
		value: unknown,
		options: { expectedStateType?: string } = {},
	): WorkflowState<TData> {
		if (
			typeof value !== 'object' ||
			value === null ||
			Array.isArray(value)
		) {
			throw new Error('workflow state payload must be a JSON object');
		}
		const record = value as Record<string, unknown>;
		if (typeof record.stateType !== 'string') {
			throw new Error('workflow stateType must be a string');
		}
		if (!Number.isInteger(record.version)) {
			throw new Error('workflow state version must be an integer');
		}
		if (
			typeof record.data !== 'object' ||
			record.data === null ||
			Array.isArray(record.data)
		) {
			throw new Error('workflow state data must be an object');
		}
		if (
			options.expectedStateType !== undefined &&
			record.stateType !== options.expectedStateType
		) {
			throw new Error(
				`workflow state type mismatch: expected ${options.expectedStateType}, got ${record.stateType}`,
			);
		}
		validateJSONValue(record.data);
		return new WorkflowState<TData>({
			stateType: record.stateType,
			version: record.version as number,
			data: record.data as TData,
		});
	}

	static fromJSON<TData extends Record<string, JSONValue>>(
		payload: string,
		options: { expectedStateType?: string } = {},
	): WorkflowState<TData> {
		let value: unknown;
		try {
			value = JSON.parse(payload);
		} catch {
			throw new Error('workflow state payload is not valid JSON');
		}
		return WorkflowState.fromJSONValue<TData>(value, options);
	}
}

export interface SessionRef {
	sessionId: string;
	threadId: string;
}

export function validateSessionRef(session: SessionRef): void {
	if (!session.sessionId.trim()) {
		throw new Error('sessionId must not be empty');
	}
	if (!session.threadId.trim()) {
		throw new Error('threadId must not be empty');
	}
}

export interface MemoryScope {
	user?: string;
	tenant?: string;
	agent?: string;
	project?: string;
	workspace?: string;
	task?: string;
}

export function scopeContains(
	expected: MemoryScope,
	actual: MemoryScope,
): boolean {
	for (const key of [
		'user',
		'tenant',
		'agent',
		'project',
		'workspace',
		'task',
	]) {
		const value = expected[key as keyof MemoryScope];
		if (value !== undefined && actual[key as keyof MemoryScope] !== value) {
			return false;
		}
	}
	return true;
}

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export type ContentPartType =
	'text' | 'image' | 'audio' | 'video' | 'pdf' | 'document' | 'file' | 'json';

export interface ContentPart {
	type: ContentPartType;
	text?: string;
	data?: unknown;
	mimeType?: string;
}

export interface ToolCall {
	id: string;
	name: string;
	arguments: Record<string, unknown>;
	/**
	 * Set by provider adapters when the model's raw arguments were not a valid
	 * JSON object. The call must not run; the loop reports the error to the
	 * model so it can repair the call instead of aborting the whole run.
	 */
	argumentError?: string;
}

export function isContentPart(value: unknown): value is ContentPart {
	if (typeof value !== 'object' || value === null || !('type' in value)) {
		return false;
	}
	const type = (value as { type?: unknown }).type;
	return (
		typeof type === 'string' &&
		[
			'text',
			'image',
			'audio',
			'video',
			'pdf',
			'document',
			'file',
			'json',
		].includes(type)
	);
}

export interface ModelMessage {
	role: MessageRole;
	content: ContentPart[];
	toolCalls?: ToolCall[];
	toolCallId?: string;
}

export class RunCancelledError extends Error {
	constructor() {
		super('run_cancelled');
		this.name = 'RunCancelledError';
	}
}

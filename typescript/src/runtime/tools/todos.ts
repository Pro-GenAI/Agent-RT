/** The built-in todo list tool and tool result marshalling. */
import { type ContentPart, isContentPart } from '../types.js';
import { type RegisteredTool, type ToolRegistry } from './registry.js';

export const TODO_STATUSES = ['pending', 'in_progress', 'completed'] as const;

export type TodoStatus = (typeof TODO_STATUSES)[number];

export const TODO_MAX_ITEMS = 100;

export const TODO_MAX_CONTENT_CHARS = 500;

export interface TodoItem {
	id: number;
	content: string;
	status: TodoStatus;
}

/**
 * One session's task list, exposed to the model as the `todo` tool.
 *
 * Give every session its own instance: the tool handler only sees this list, so
 * tasks never leak between sessions.
 */
export class TodoList {
	items: TodoItem[] = [];

	constructor(items: readonly unknown[] = []) {
		this.replace(items);
	}

	/** Load a stored list (for example on resume); throws on bad entries. */
	replace(items: readonly unknown[]): void {
		if (!Array.isArray(items) || items.length > TODO_MAX_ITEMS) {
			throw new Error(
				`todos must be a list of at most ${TODO_MAX_ITEMS} items`,
			);
		}
		const loaded: TodoItem[] = [];
		for (const item of items) {
			if (!item || typeof item !== 'object' || Array.isArray(item)) {
				throw new Error('each todo must be an object');
			}
			const record = item as Record<string, unknown>;
			const id = record.id;
			if (typeof id !== 'number' || !Number.isInteger(id) || id < 1) {
				throw new Error('todo id must be a positive integer');
			}
			if (loaded.some((existing) => existing.id === id)) {
				throw new Error(`duplicate todo id: ${id}`);
			}
			loaded.push({
				id,
				content: TodoList.content(record.content),
				status: TodoList.status(record.status),
			});
		}
		this.items = loaded;
	}

	private static content(value: unknown): string {
		if (typeof value !== 'string' || !value.trim()) {
			throw new Error('todo content must be a non-empty string');
		}
		if (value.length > TODO_MAX_CONTENT_CHARS) {
			throw new Error(
				`todo content must be at most ${TODO_MAX_CONTENT_CHARS} characters`,
			);
		}
		return value.trim();
	}

	private static status(value: unknown): TodoStatus {
		if (!TODO_STATUSES.includes(value as TodoStatus)) {
			throw new Error(
				`todo status must be one of ${TODO_STATUSES.join(', ')}`,
			);
		}
		return value as TodoStatus;
	}

	private find(id: unknown): TodoItem {
		const item = this.items.find((candidate) => candidate.id === id);
		if (!item) throw new Error(`no todo with id ${String(id)}`);
		return item;
	}

	/** Run one `todo` tool action and return the whole list. */
	apply(args: Record<string, unknown>): { todos: TodoItem[] } {
		const action = args.action;
		if (action === 'add') {
			if (this.items.length >= TODO_MAX_ITEMS) {
				throw new Error(`todo list is full (${TODO_MAX_ITEMS} items)`);
			}
			this.items.push({
				id: Math.max(0, ...this.items.map((item) => item.id)) + 1,
				content: TodoList.content(args.content),
				status: TodoList.status(args.status ?? 'pending'),
			});
		} else if (action === 'update') {
			const item = this.find(args.id);
			if (!('content' in args) && !('status' in args)) {
				throw new Error('update needs content or status');
			}
			const content =
				'content' in args
					? TodoList.content(args.content)
					: item.content;
			const status =
				'status' in args ? TodoList.status(args.status) : item.status;
			item.content = content;
			item.status = status;
		} else if (action === 'remove') {
			this.items.splice(this.items.indexOf(this.find(args.id)), 1);
		} else if (action === 'clear') {
			this.items = [];
		} else if (action !== 'list') {
			throw new Error(`unknown todo action: ${String(action)}`);
		}
		return { todos: this.items.map((item) => ({ ...item })) };
	}

	register(registry: ToolRegistry, name = 'todo'): RegisteredTool {
		return registry.register(
			{
				name,
				description:
					"Track this session's task list. Plan multi-step work with add, mark " +
					'progress with update (status pending, in_progress, or completed), ' +
					'drop a task with remove, empty the list with clear, or view it with ' +
					'list. Every action returns the whole list.',
				inputSchema: {
					type: 'object',
					properties: {
						action: {
							type: 'string',
							enum: ['add', 'update', 'remove', 'clear', 'list'],
						},
						id: { type: 'integer', minimum: 1 },
						content: {
							type: 'string',
							maxLength: TODO_MAX_CONTENT_CHARS,
						},
						status: { type: 'string', enum: [...TODO_STATUSES] },
					},
					required: ['action'],
					additionalProperties: false,
				},
				sideEffect: 'none',
				executionMode: 'sequential',
				errorBehavior: 'return_error',
			},
			{ handler: async (args) => this.apply(args) },
		);
	}
}

export function marshalToolResult(value: unknown): ContentPart[] {
	if (isContentPart(value)) {
		return [value];
	}
	if (typeof value === 'string') {
		return [{ type: 'text', text: value }];
	}
	if (
		Array.isArray(value) &&
		value.length > 0 &&
		value.every(isContentPart)
	) {
		return value;
	}
	return [{ type: 'json', data: value }];
}

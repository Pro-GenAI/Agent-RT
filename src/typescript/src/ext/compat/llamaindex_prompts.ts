import type { ChatMessage } from './llamaindex.js';

export type PromptValues = Record<string, unknown>;

// Single pass: substituted values are never re-scanned, so a value containing
// "{other}" cannot pull in another variable.
function renderTemplate(template: string, values: PromptValues): string {
	return template.replace(/\{([^{}]+)\}/g, (match, key: string) =>
		Object.prototype.hasOwnProperty.call(values, key)
			? String(values[key])
			: match,
	);
}

export class PromptTemplate {
	readonly template: string;

	constructor(template: string | { template: string }) {
		this.template =
			typeof template === 'string' ? template : template.template;
	}

	format(values: PromptValues = {}): string {
		return renderTemplate(this.template, values);
	}

	partial(values: PromptValues): PromptTemplate {
		return new PromptTemplate(this.format(values));
	}

	toString(): string {
		return this.template;
	}
}

export type ChatPromptMessageTemplate = {
	role: ChatMessage['role'];
	content: string;
};

export class ChatPromptTemplate {
	readonly messageTemplates: ChatPromptMessageTemplate[];

	constructor(
		messages:
			| ChatPromptMessageTemplate[]
			| { messageTemplates: ChatPromptMessageTemplate[] },
	) {
		this.messageTemplates = [
			...(Array.isArray(messages) ? messages : messages.messageTemplates),
		];
	}

	formatMessages(values: PromptValues = {}): ChatMessage[] {
		return this.messageTemplates.map((message) => ({
			role: message.role,
			content: renderTemplate(message.content, values),
		}));
	}

	format(values: PromptValues = {}): string {
		return this.formatMessages(values)
			.map((message) => `${message.role}: ${message.content}`)
			.join('\n');
	}
}

export interface BaseOutputParser<T = unknown> {
	parse(output: string): T;
	format?(prompt: string): string;
}

export class JSONOutputParser<T = unknown> implements BaseOutputParser<T> {
	constructor(readonly schemaHint?: string) {}

	parse(output: string): T {
		return JSON.parse(output) as T;
	}

	format(prompt: string): string {
		if (!this.schemaHint) return prompt;
		return `${prompt}\n\nReturn valid JSON matching this schema:\n${this.schemaHint}`;
	}
}

export type CallbackEvent =
	| {
			type: 'llm-start';
			model: string;
			input: unknown;
			stream?: boolean;
	  }
	| {
			type: 'llm-stream';
			model: string;
			event: unknown;
	  }
	| {
			type: 'llm-end';
			model: string;
			output: unknown;
			stream?: boolean;
	  }
	| {
			type: 'llm-error';
			model: string;
			error: unknown;
	  }
	| {
			type: 'retrieval-start';
			query: string;
	  }
	| {
			type: 'retrieval-end';
			query: string;
			output: unknown;
	  };

export type CallbackHandler = (event: CallbackEvent) => void | Promise<void>;

export class CallbackManager {
	private readonly handlers = new Set<CallbackHandler>();

	on(handler: CallbackHandler): () => void {
		this.handlers.add(handler);
		return () => this.handlers.delete(handler);
	}

	off(handler: CallbackHandler): void {
		this.handlers.delete(handler);
	}

	async emit(event: CallbackEvent): Promise<void> {
		await Promise.all(
			[...this.handlers].map(async (handler) => await handler(event)),
		);
	}
}

class LlamaIndexSettings {
	llm?: unknown;
	embedModel?: unknown;
	callbackManager = new CallbackManager();
	tokenizer?: (text: string) => number[] | string[];

	reset(): void {
		this.llm = undefined;
		this.embedModel = undefined;
		this.callbackManager = new CallbackManager();
		this.tokenizer = undefined;
	}
}

export const Settings = new LlamaIndexSettings();

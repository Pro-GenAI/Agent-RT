/**
 * `messages.stream()` / `messages.create({ stream: true })` for the Anthropic
 * shim, in the SDK's `MessageStream` shape.
 *
 * Events are derived from the completed message, which still runs through the
 * bounded migration-tool loop and the SDK error mapping, so text arrives as one
 * delta per content block rather than token by token.
 */
import type * as AnthropicTypes from './anthropic_types.js';

type Message = AnthropicTypes.Message;
type StreamEvent = AnthropicTypes.RawMessageStreamEvent;
type Listener = (...args: any[]) => void;

/** The raw stream events the API would send for a completed message. */
export function messageStreamEvents(message: Message): StreamEvent[] {
	const blocks = (message.content ?? []) as unknown as Array<
		Record<string, unknown>
	>;
	const events: StreamEvent[] = [
		{
			type: 'message_start',
			message: {
				...message,
				content: [],
				stop_reason: null,
				stop_sequence: null,
			} as Message,
		},
	];
	blocks.forEach((block, index) => {
		if (block.type === 'text') {
			events.push(
				{
					type: 'content_block_start',
					index,
					content_block: {
						...block,
						text: '',
					} as AnthropicTypes.ContentBlock,
				},
				{
					type: 'content_block_delta',
					index,
					delta: {
						type: 'text_delta',
						text: String(block.text ?? ''),
					},
				},
			);
		} else if (block.type === 'tool_use') {
			events.push(
				{
					type: 'content_block_start',
					index,
					content_block: {
						...block,
						input: {},
					} as AnthropicTypes.ContentBlock,
				},
				{
					type: 'content_block_delta',
					index,
					delta: {
						type: 'input_json_delta',
						partial_json: JSON.stringify(block.input ?? {}),
					},
				},
			);
		} else {
			events.push({
				type: 'content_block_start',
				index,
				content_block: block as unknown as AnthropicTypes.ContentBlock,
			});
		}
		events.push({ type: 'content_block_stop', index });
	});
	const usage = (message.usage ?? {}) as { output_tokens?: number };
	events.push(
		{
			type: 'message_delta',
			delta: {
				stop_reason: message.stop_reason ?? null,
				stop_sequence: message.stop_sequence ?? null,
			},
			usage: { output_tokens: usage.output_tokens ?? 0 },
		},
		{ type: 'message_stop' },
	);
	return events;
}

/** An async iterable of raw events, as `messages.create({ stream: true })` returns. */
export async function* rawMessageStream(
	message: Promise<Message>,
): AsyncGenerator<StreamEvent> {
	yield* messageStreamEvents(await message);
}

/** The SDK's `MessageStream`: events, listeners, and final-message helpers. */
export class MessageStream implements AsyncIterable<StreamEvent> {
	private readonly listeners = new Map<
		string,
		Array<{ listener: Listener; once: boolean }>
	>();
	private readonly finished: Promise<Message>;
	private aborted = false;
	readonly controller = new AbortController();
	readonly messages: Message[] = [];
	receivedMessages: Message[] = this.messages;

	constructor(run: () => Promise<Message>) {
		// Started immediately, as in the SDK; listeners attached in the same
		// tick still receive every event.
		this.finished = Promise.resolve().then(async () => {
			try {
				const message = await run();
				if (this.aborted) throw new Error('Request was aborted.');
				this.emitEvents(message);
				return message;
			} catch (error) {
				this.emit('error', error);
				throw error;
			} finally {
				this.emit('end');
			}
		});
		// Unobserved failures surface through finalMessage()/done()/iteration.
		this.finished.catch(() => undefined);
	}

	private emitEvents(message: Message): void {
		let text = '';
		for (const event of messageStreamEvents(message)) {
			this.emit('streamEvent', event, message);
			if (
				event.type === 'content_block_delta' &&
				event.delta.type === 'text_delta'
			) {
				text += event.delta.text;
				this.emit('text', event.delta.text, text);
			}
			if (event.type === 'content_block_stop') {
				const block = (message.content as unknown[])[event.index];
				this.emit('contentBlock', block);
			}
		}
		this.messages.push(message);
		this.emit('message', message);
		this.emit('finalMessage', message);
	}

	private emit(event: string, ...args: unknown[]): void {
		const entries = this.listeners.get(event);
		if (!entries) return;
		this.listeners.set(
			event,
			entries.filter((entry) => !entry.once),
		);
		for (const entry of entries) entry.listener(...args);
	}

	on(event: string, listener: Listener): this {
		this.listeners.set(event, [
			...(this.listeners.get(event) ?? []),
			{ listener, once: false },
		]);
		return this;
	}

	once(event: string, listener: Listener): this {
		this.listeners.set(event, [
			...(this.listeners.get(event) ?? []),
			{ listener, once: true },
		]);
		return this;
	}

	off(event: string, listener: Listener): this {
		this.listeners.set(
			event,
			(this.listeners.get(event) ?? []).filter(
				(entry) => entry.listener !== listener,
			),
		);
		return this;
	}

	abort(): void {
		this.aborted = true;
		this.controller.abort();
	}

	async done(): Promise<void> {
		await this.finished;
	}

	finalMessage(): Promise<Message> {
		return this.finished;
	}

	async finalText(): Promise<string> {
		const message = await this.finished;
		return (message.content as Array<{ type?: string; text?: string }>)
			.filter((block) => block.type === 'text')
			.map((block) => block.text ?? '')
			.join('');
	}

	async *[Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
		yield* messageStreamEvents(await this.finished);
	}
}

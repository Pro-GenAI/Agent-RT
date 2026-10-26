/** Multimodal messages and realtime sessions. */
import {
	type ContentPart,
	type MessageRole,
	type ModelMessage,
} from '../types.js';

export interface MultimodalMessage {
	role: MessageRole;
	parts: ContentPart[];
}

export function multimodalToModelMessage(
	message: MultimodalMessage,
): ModelMessage {
	return {
		role: message.role,
		content: message.parts.map((part) => ({ ...part })),
	};
}

export type RealtimeEventType =
	| 'audio_input'
	| 'audio_output'
	| 'speech_started'
	| 'speech_stopped'
	| 'text_delta'
	| 'response_completed'
	| 'interrupted';

export interface RealtimeEvent {
	type: RealtimeEventType;
	data?: unknown;
	timestampMs?: number;
}

export interface RealtimeTransport {
	send(event: RealtimeEvent): Promise<void>;
	events(): AsyncIterable<RealtimeEvent>;
	close(): Promise<void>;
}

export class RealtimeSession {
	interrupted = false;

	constructor(readonly transport: RealtimeTransport) {}

	sendAudio(data: unknown): Promise<void> {
		return this.transport.send({
			type: 'audio_input',
			data,
			timestampMs: Date.now(),
		});
	}

	sendText(text: string): Promise<void> {
		return this.transport.send({
			type: 'text_delta',
			data: text,
			timestampMs: Date.now(),
		});
	}

	async interrupt(): Promise<void> {
		this.interrupted = true;
		await this.transport.send({
			type: 'interrupted',
			timestampMs: Date.now(),
		});
	}

	async collectUntilComplete(): Promise<RealtimeEvent[]> {
		const events: RealtimeEvent[] = [];
		for await (const event of this.transport.events()) {
			events.push(event);
			if (
				event.type === 'response_completed' ||
				event.type === 'interrupted'
			) {
				break;
			}
		}
		return events;
	}

	close(): Promise<void> {
		return this.transport.close();
	}
}

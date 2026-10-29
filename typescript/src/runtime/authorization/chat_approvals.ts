/** Approve or deny agent actions from a chat message or phone notification. */
import { type ApprovalManager, type ApprovalRequest } from './approvals.js';

/** Chat transport: (notification text, short reply code). */
export type ChatSender = (text: string, code: string) => void | Promise<void>;

const YES = new Set(['approve', 'approved', 'allow', 'yes', 'y', 'ok']);
const NO = new Set(['deny', 'denied', 'reject', 'no', 'n']);

/** Render every authorized argument; never approve a lossy preview. */
export function describeRequest(request: ApprovalRequest): string {
	const ordered = (value: unknown): unknown => {
		if (Array.isArray(value)) return value.map(ordered);
		if (value && typeof value === 'object')
			return Object.fromEntries(
				Object.entries(value)
					.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
					.map(([key, nested]) => [key, ordered(nested)]),
			);
		return value;
	};
	const payload = JSON.stringify(
		ordered({ tool: request.call.name, arguments: request.call.arguments }),
	).replace(
		/[\u007f-\uffff]/g,
		(character) =>
			'\\u' + character.charCodeAt(0).toString(16).padStart(4, '0'),
	);
	return `Agent requests ${payload}, approve?`;
}

/**
 * Pushes approval requests to a chat channel and resolves them from replies.
 * Every request that needs a human is sent once with a short reply code;
 * `handleReply` turns "approve"/"deny" (plus the code when several are
 * waiting) into a `once` grant bound to the exact call arguments. Whoever can
 * post replies can approve, so the chat transport must only accept messages
 * from the approving human.
 */
export class ChatApprovals {
	readonly pending = new Map<string, ApprovalRequest>();
	private nextCode = 1;

	constructor(
		readonly manager: ApprovalManager,
		readonly send: ChatSender,
	) {
		manager.notifier = (request) => this.notify(request);
	}

	notify(request: ApprovalRequest): void {
		// Resumed runs re-check the same call; send it once.
		for (const waiting of this.pending.values()) {
			if (waiting.id === request.id) return;
		}
		const code = String(this.nextCode++);
		this.pending.set(code, request);
		// A dead channel must not turn a pause into a crash.
		Promise.resolve()
			.then(() => this.send(describeRequest(request), code))
			.catch((error: unknown) =>
				console.warn('approval notification failed', error),
			);
	}

	/** Resolves a reply; returns the confirmation to send back to the chat. */
	handleReply(text: string, actorId = 'chat'): string {
		const parts = text.trim().toLowerCase().split(/\s+/);
		const word = parts[0] ?? '';
		if (!YES.has(word) && !NO.has(word))
			return "Reply 'approve' or 'deny'.";
		let code: string;
		if (parts.length > 1) code = parts[1].replace(/^#/, '');
		else if (this.pending.size === 1) code = [...this.pending.keys()][0];
		else
			return "Several approvals are waiting: reply 'approve <code>' or 'deny <code>'.";
		const request = this.pending.get(code);
		if (!request) return 'No such pending approval.';
		this.pending.delete(code);
		const allow = YES.has(word);
		this.manager.resolve(request, allow ? 'allow' : 'deny', { actorId });
		const summary = describeRequest(request).replace(/, approve\?$/, '');
		return (allow ? 'Approved: ' : 'Denied: ') + summary;
	}
}

/** Phone push through ntfy (https://ntfy.sh); the topic name acts as a secret. */
export function ntfySender(
	topic: string,
	options: { server?: string; token?: string } = {},
): ChatSender {
	const server = (options.server ?? 'https://ntfy.sh').replace(/\/+$/, '');
	const url = `${server}/${encodeURIComponent(topic)}`;
	return async (text, code) => {
		const headers: Record<string, string> = {
			Title: 'Agent approval',
			Priority: 'high',
		};
		if (options.token) headers.Authorization = `Bearer ${options.token}`;
		const response = await fetch(url, {
			method: 'POST',
			headers,
			body: `${text}\nReply: approve ${code} / deny ${code}`,
			signal: AbortSignal.timeout(10_000),
		});
		if (!response.ok) throw new Error(`ntfy responded ${response.status}`);
	};
}

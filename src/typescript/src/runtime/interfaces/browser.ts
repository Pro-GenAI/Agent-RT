/** Browser and computer-use sessions. */

export type BrowserAction =
	| 'navigate'
	| 'search'
	| 'click'
	| 'fill'
	| 'download'
	| 'extract'
	| 'back'
	| 'forward'
	| 'reload';

export interface BrowserState {
	url?: string;
	title?: string;
	history?: string[];
	metadata?: Record<string, unknown>;
}

export interface BrowserBackend {
	perform(
		action: BrowserAction,
		args: Record<string, unknown>,
		state: BrowserState,
	): Promise<{ result: unknown; state: BrowserState }>;
}

export class BrowserSession {
	state: BrowserState;

	constructor(
		readonly backend: BrowserBackend,
		state: BrowserState = {},
	) {
		this.state = { ...state, history: [...(state.history ?? [])] };
	}

	async perform(
		action: BrowserAction,
		args: Record<string, unknown> = {},
	): Promise<unknown> {
		const response = await this.backend.perform(action, args, this.state);
		this.state = {
			...response.state,
			history: [...(response.state.history ?? [])],
		};
		return response.result;
	}
}

export type ComputerAction =
	'screenshot' | 'move' | 'click' | 'scroll' | 'key' | 'type' | 'drag';

export interface ComputerState {
	width: number;
	height: number;
	metadata?: Record<string, unknown>;
}

export interface ComputerBackend {
	perform(
		action: ComputerAction,
		args: Record<string, unknown>,
		state: ComputerState,
	): Promise<unknown>;
}

export class ComputerSession {
	constructor(
		readonly backend: ComputerBackend,
		readonly state: ComputerState,
	) {
		if (state.width <= 0 || state.height <= 0) {
			throw new Error('computer dimensions must be positive');
		}
	}

	perform(
		action: ComputerAction,
		args: Record<string, unknown> = {},
	): Promise<unknown> {
		return this.backend.perform(action, args, this.state);
	}
}

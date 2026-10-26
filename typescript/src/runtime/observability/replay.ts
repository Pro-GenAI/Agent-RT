/** Debug replay bundles and deterministic replay providers. */
import {
	type ModelProvider,
	type ModelRequest,
	type ModelResponse,
} from '../models/interfaces.js';

export interface ReplayExchange {
	channel: 'model' | 'tool' | 'external';
	key: string;
	input: unknown;
	output: unknown;
}

export interface ReplayBundle {
	replayId: string;
	state?: unknown;
	events?: unknown[];
	checkpoints?: unknown[];
	exchanges?: ReplayExchange[];
	metadata?: Record<string, unknown>;
}

export class DebugReplayStore {
	private readonly bundles = new Map<string, ReplayBundle>();

	save(bundle: ReplayBundle): ReplayBundle {
		this.bundles.set(bundle.replayId, bundle);
		return bundle;
	}

	load(replayId: string): ReplayBundle {
		const bundle = this.bundles.get(replayId);
		if (!bundle) throw new Error('unknown replay bundle: ' + replayId);
		return bundle;
	}

	reconstruct(replayId: string): Record<string, unknown> {
		const bundle = this.load(replayId);
		return {
			state: bundle.state,
			events: [...(bundle.events ?? [])],
			checkpoints: [...(bundle.checkpoints ?? [])],
			exchanges: [...(bundle.exchanges ?? [])],
			metadata: { ...(bundle.metadata ?? {}) },
		};
	}
}

export class DeterministicReplay {
	private readonly positions = new Map<string, number>();

	constructor(readonly bundle: ReplayBundle) {}

	next(
		channel: ReplayExchange['channel'],
		key: string,
		input?: unknown,
	): unknown {
		const matches = (this.bundle.exchanges ?? []).filter(
			(exchange) =>
				exchange.channel === channel && Object.is(exchange.key, key),
		);
		const positionKey = channel + ':' + key;
		const position = this.positions.get(positionKey) ?? 0;
		if (position >= matches.length) {
			throw new Error(
				'no recorded replay output for ' + channel + ':' + key,
			);
		}
		const exchange = matches[position];
		if (
			input !== undefined &&
			JSON.stringify(exchange.input) !== JSON.stringify(input)
		) {
			throw new Error('replay input mismatch for ' + channel + ':' + key);
		}
		this.positions.set(positionKey, position + 1);
		return exchange.output;
	}
}

export class ReplayModelProvider implements ModelProvider {
	constructor(
		readonly replay: DeterministicReplay,
		readonly key = 'complete',
		readonly name = 'replay',
	) {}

	async complete(_request: ModelRequest): Promise<ModelResponse> {
		return this.replay.next('model', this.key) as ModelResponse;
	}
}

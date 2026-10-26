/** Model routing policies and the model registry. */
import { type ModelSettings } from '../agent/config.js';

export interface ModelTarget {
	model: string;
	provider?: string;
}

export interface ModelDescriptor extends ModelTarget {
	capabilities?: string[];
	contextWindow?: number;
	costPerMillionTokens?: number;
	latencyMs?: number;
	reasoning?: boolean;
}

export interface RoutingRequirements {
	capabilities?: string[];
	minContextWindow?: number;
	maxCostPerMillionTokens?: number;
	maxLatencyMs?: number;
	reasoning?: boolean;
}

export interface ModelRoutingPolicy {
	select(
		candidates: ModelDescriptor[],
		requirements: RoutingRequirements,
	): ModelDescriptor;
}

export class FirstMatchRoutingPolicy implements ModelRoutingPolicy {
	select(
		candidates: ModelDescriptor[],
		requirements: RoutingRequirements,
	): ModelDescriptor {
		const requiredCapabilities = new Set(requirements.capabilities ?? []);

		for (const candidate of candidates) {
			const capabilities = new Set(candidate.capabilities ?? []);
			if (
				[...requiredCapabilities].some(
					(capability) => !capabilities.has(capability),
				)
			)
				continue;
			if (
				requirements.minContextWindow !== undefined &&
				(candidate.contextWindow === undefined ||
					candidate.contextWindow < requirements.minContextWindow)
			)
				continue;
			const maxCost = requirements.maxCostPerMillionTokens;
			const candidateCost = candidate.costPerMillionTokens;
			if (
				Number.isFinite(maxCost) &&
				(!Number.isFinite(candidateCost) ||
					(candidateCost as number) > (maxCost as number))
			)
				continue;
			if (
				requirements.maxLatencyMs !== undefined &&
				(candidate.latencyMs === undefined ||
					candidate.latencyMs > requirements.maxLatencyMs)
			)
				continue;
			if (
				requirements.reasoning !== undefined &&
				Boolean(candidate.reasoning) !== requirements.reasoning
			)
				continue;
			return candidate;
		}

		throw new Error('no model candidate satisfies routing requirements');
	}
}

export class ModelRegistry {
	private readonly models = new Map<string, ModelDescriptor>();
	private readonly aliases = new Map<string, ModelTarget>();

	constructor(
		models: ModelDescriptor[] = [],
		aliases: Record<string, ModelTarget> = {},
	) {
		for (const model of models) this.register(model);
		for (const [name, target] of Object.entries(aliases))
			this.alias(name, target);
	}

	private key(target: ModelTarget): string {
		return `${target.provider ?? ''}::${target.model}`;
	}

	register(descriptor: ModelDescriptor): void {
		this.models.set(this.key(descriptor), descriptor);
	}

	alias(name: string, target: ModelTarget): void {
		this.aliases.set(name, target);
	}

	resolve(target: ModelTarget): ModelTarget {
		const aliased = this.aliases.get(target.model);
		if (!aliased) return target;
		return {
			model: aliased.model,
			provider: target.provider ?? aliased.provider,
		};
	}

	candidates(settings: ModelSettings): ModelDescriptor[] {
		const targets: ModelTarget[] = [
			{ model: settings.model, provider: settings.provider },
			...(settings.fallbackModels ?? []),
		];

		return targets
			.map((target) => this.resolve(target))
			.map((target) => this.models.get(this.key(target)))
			.filter(
				(candidate): candidate is ModelDescriptor =>
					candidate !== undefined,
			);
	}

	route(
		settings: ModelSettings,
		requirements: RoutingRequirements = {},
		policy: ModelRoutingPolicy = new FirstMatchRoutingPolicy(),
	): ModelDescriptor {
		return policy.select(this.candidates(settings), requirements);
	}
}

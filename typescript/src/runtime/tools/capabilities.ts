/** Capability descriptors, scoring, and the searchable capability catalog. */

export type CapabilityKind = 'tool' | 'skill' | 'connector' | 'file' | 'agent';

export interface CapabilityDescriptor {
	id: string;
	kind: CapabilityKind;
	name: string;
	description?: string;
	namespace?: string;
	metadata?: Record<string, unknown>;
}

export interface CapabilitySearchResult {
	capability: CapabilityDescriptor;
	score: number;
}

export interface CapabilityScorer {
	score(query: string, capability: CapabilityDescriptor): number;
}

function lexicalTerms(value: string): Set<string> {
	return new Set(value.toLowerCase().match(/[a-z0-9]+/g) ?? []);
}

export class LexicalCapabilityScorer implements CapabilityScorer {
	score(query: string, capability: CapabilityDescriptor): number {
		const queryTerms = lexicalTerms(query);
		if (queryTerms.size === 0) return 0;

		const metadataText = Object.entries(capability.metadata ?? {})
			.filter(([, value]) =>
				['string', 'number', 'boolean'].includes(typeof value),
			)
			.map(([key, value]) => `${key} ${String(value)}`)
			.join(' ');
		const haystack = [
			capability.name,
			capability.namespace ?? '',
			capability.description ?? '',
			metadataText,
		]
			.filter(Boolean)
			.join(' ');
		const terms = lexicalTerms(haystack);
		if (terms.size === 0) return 0;

		let overlap = 0;
		for (const term of queryTerms) {
			if (terms.has(term)) overlap += 1;
		}
		if (overlap === 0) return 0;

		let score = overlap / queryTerms.size;
		const nameTerms = lexicalTerms(capability.name);
		if ([...queryTerms].every((term) => nameTerms.has(term))) {
			score += 0.25;
		}
		return score;
	}
}

export class CapabilityCatalog {
	private readonly capabilities = new Map<string, CapabilityDescriptor>();

	constructor(
		capabilities: CapabilityDescriptor[] = [],
		readonly scorer: CapabilityScorer = new LexicalCapabilityScorer(),
	) {
		for (const capability of capabilities) this.register(capability);
	}

	register(
		capability: CapabilityDescriptor,
		options: { replace?: boolean } = {},
	): void {
		if (!capability.id.trim()) {
			throw new Error('capability id must not be empty');
		}
		if (!capability.name.trim()) {
			throw new Error('capability name must not be empty');
		}
		if (this.capabilities.has(capability.id) && !options.replace) {
			throw new Error(`capability already registered: ${capability.id}`);
		}
		this.capabilities.set(capability.id, capability);
	}

	unregister(id: string): CapabilityDescriptor {
		const capability = this.capabilities.get(id);
		if (!capability) throw new Error(`capability not registered: ${id}`);
		this.capabilities.delete(id);
		return capability;
	}

	search(
		query: string,
		options: {
			kinds?: CapabilityKind[];
			limit?: number;
			minScore?: number;
		} = {},
	): CapabilitySearchResult[] {
		const limit = options.limit ?? 10;
		const minScore = options.minScore ?? 0;
		if (limit < 0) {
			throw new Error('capability search limit must be non-negative');
		}
		const kinds = options.kinds ? new Set(options.kinds) : undefined;
		const results: CapabilitySearchResult[] = [];
		for (const capability of this.capabilities.values()) {
			if (kinds && !kinds.has(capability.kind)) continue;
			const score = this.scorer.score(query, capability);
			if (score > minScore) results.push({ capability, score });
		}
		results.sort(
			(a, b) =>
				b.score - a.score ||
				a.capability.kind.localeCompare(b.capability.kind) ||
				a.capability.id.localeCompare(b.capability.id),
		);
		return results.slice(0, limit);
	}
}

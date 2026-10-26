/** Retrieval queries, providers, rerankers, and the retrieval registry. */

export type RetrievalKind =
	'web' | 'enterprise' | 'file' | 'knowledge' | 'database';

export interface RetrievalQuery {
	text: string;
	limit?: number;
	filters?: Record<string, unknown>;
}

export interface RetrievalResult {
	id: string;
	title: string;
	content: unknown;
	score?: number;
	uri?: string;
	metadata?: Record<string, unknown>;
}

export interface RetrievalProvider {
	kind: RetrievalKind;
	search(query: RetrievalQuery): Promise<RetrievalResult[]>;
}

export interface RetrievalReranker {
	rerank(
		query: RetrievalQuery,
		results: readonly RetrievalResult[],
		options?: { k?: number },
	): Promise<RetrievalResult[]>;
}

export class RetrievalRegistry {
	private readonly providers = new Map<string, RetrievalProvider>();
	private readonly rerankers = new Map<string, RetrievalReranker>();
	private readonly rerankerKs = new Map<string, number>();

	register(
		name: string,
		provider: RetrievalProvider,
		options: { reranker?: RetrievalReranker; rerankerK?: number } = {},
	): void {
		if (!name.trim())
			throw new Error('retrieval provider name must not be empty');
		if (options.rerankerK !== undefined) {
			if (!Number.isInteger(options.rerankerK) || options.rerankerK < 1) {
				throw new Error('reranker k must be at least 1');
			}
			if (!options.reranker)
				throw new Error('reranker k requires a reranker');
		}
		this.providers.set(name, provider);
		if (!options.reranker) {
			this.rerankers.delete(name);
			this.rerankerKs.delete(name);
			return;
		}
		this.rerankers.set(name, options.reranker);
		if (options.rerankerK === undefined) this.rerankerKs.delete(name);
		else this.rerankerKs.set(name, options.rerankerK);
	}

	async search(
		name: string,
		query: RetrievalQuery,
	): Promise<RetrievalResult[]> {
		const limit = query.limit ?? 10;
		if (!Number.isInteger(limit) || limit < 1) {
			throw new Error('retrieval limit must be at least 1');
		}
		const provider = this.providers.get(name);
		if (!provider) throw new Error('unknown retrieval provider: ' + name);
		let results = await provider.search(query);
		const reranker = this.rerankers.get(name);
		if (reranker) {
			const k = this.rerankerKs.get(name);
			if (k === undefined) {
				results = await reranker.rerank(query, results);
			} else {
				results = await reranker.rerank(query, results, { k });
				results = results.slice(0, k);
			}
		}
		return results.slice(0, limit);
	}
}

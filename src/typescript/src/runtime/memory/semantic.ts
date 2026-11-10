/** Semantic and episodic memory. */
import { type JSONValue, type MemoryScope } from '../types.js';
import {
	type LongTermMemoryStore,
	type MemoryRecord,
	type MemorySearchResult,
} from './long_term.js';

export interface EmbeddingProvider {
	embed(text: string): number[];
}

export class SemanticMemory {
	constructor(
		readonly store: LongTermMemoryStore,
		readonly embeddingProvider: EmbeddingProvider,
	) {}

	write(options: {
		memoryId: string;
		content: string;
		scope?: MemoryScope;
		metadata?: Record<string, JSONValue>;
		tags?: string[];
	}): MemoryRecord {
		return this.store.write({
			id: options.memoryId,
			kind: 'semantic',
			content: options.content,
			scope: options.scope ?? {},
			metadata: options.metadata ?? {},
			tags: options.tags ?? [],
			embedding: this.embeddingProvider
				.embed(options.content)
				.map(Number),
		});
	}

	retrieve(
		query: string,
		options: { scope?: MemoryScope; limit?: number } = {},
	): MemorySearchResult[] {
		const queryVector = this.embeddingProvider.embed(query).map(Number);
		const candidates = this.store.search({
			scope: options.scope,
			kinds: ['semantic'],
			limit: Number.MAX_SAFE_INTEGER,
		});
		const ranked = candidates.flatMap((candidate) => {
			const embedding = candidate.record.embedding;
			if (!embedding || embedding.length !== queryVector.length)
				return [];
			return [
				{
					record: candidate.record,
					score: cosine(queryVector, embedding),
				},
			];
		});
		ranked.sort(
			(a, b) =>
				b.score - a.score ||
				(b.record.sequence ?? 0) - (a.record.sequence ?? 0) ||
				a.record.id.localeCompare(b.record.id),
		);
		return ranked.slice(0, options.limit ?? 10);
	}
}

function cosine(left: number[], right: number[]): number {
	let dot = 0;
	let leftNorm = 0;
	let rightNorm = 0;
	for (let index = 0; index < left.length; index += 1) {
		dot += left[index] * right[index];
		leftNorm += left[index] * left[index];
		rightNorm += right[index] * right[index];
	}
	if (leftNorm === 0 || rightNorm === 0) return 0;
	return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export interface Episode {
	id: string;
	task: string;
	outcome: string;
	scope?: MemoryScope;
	decisions?: string[];
	actions?: string[];
	trace?: string[];
	metadata?: Record<string, JSONValue>;
}

export class EpisodicMemory {
	constructor(readonly store: LongTermMemoryStore) {}

	remember(episode: Episode): MemoryRecord {
		if (!episode.id.trim()) throw new Error('episode id must not be empty');
		if (!episode.task.trim())
			throw new Error('episode task must not be empty');
		if (!episode.outcome.trim())
			throw new Error('episode outcome must not be empty');
		const episodePayload = {
			task: episode.task,
			outcome: episode.outcome,
			decisions: [...(episode.decisions ?? [])],
			actions: [...(episode.actions ?? [])],
			trace: [...(episode.trace ?? [])],
		};
		const content = [
			episode.task,
			episode.outcome,
			...(episode.decisions ?? []),
			...(episode.actions ?? []),
			...(episode.trace ?? []),
		].join(' ');
		return this.store.write({
			id: episode.id,
			kind: 'episode',
			content,
			scope: episode.scope ?? {},
			tags: ['episode'],
			metadata: {
				...(episode.metadata ?? {}),
				episode: episodePayload,
			},
		});
	}

	search(
		text: string,
		options: { scope?: MemoryScope; limit?: number } = {},
	): MemorySearchResult[] {
		return this.store.search({
			text,
			scope: options.scope,
			kinds: ['episode'],
			limit: options.limit ?? 10,
		});
	}
}

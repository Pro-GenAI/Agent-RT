import { Settings } from './llamaindex_prompts.js';
import type { LLMLike, NodeWithScore, Response } from './llamaindex_rag.js';

export type EvaluationResult = {
	passing: boolean;
	score: number;
	feedback?: string;
};

function resolveLLM(llm?: LLMLike): LLMLike {
	const candidate = llm ?? (Settings.llm as LLMLike | undefined);
	if (!candidate || typeof candidate.complete !== 'function') {
		throw new Error(
			'LlamaIndex evaluator requires an llm or Settings.llm with complete()',
		);
	}
	return candidate;
}

function parseScore(text: string): EvaluationResult {
	const trimmed = text.trim();
	try {
		const parsed = JSON.parse(trimmed) as Record<string, unknown>;
		const score =
			typeof parsed.score === 'number'
				? Math.max(0, Math.min(1, parsed.score))
				: 0;
		return {
			passing:
				typeof parsed.passing === 'boolean'
					? parsed.passing
					: score >= 0.5,
			score,
			...(typeof parsed.feedback === 'string'
				? { feedback: parsed.feedback }
				: {}),
		};
	} catch {
		const numeric = Number.parseFloat(trimmed);
		const score = Math.max(0, Math.min(1, numeric));
		return { passing: score >= 0.5, score, feedback: trimmed };
	}
}

export class RelevancyEvaluator {
	private readonly llm: LLMLike;

	constructor(options: { llm?: LLMLike } = {}) {
		this.llm = resolveLLM(options.llm);
	}

	async evaluate(options: {
		query: string;
		response: string | Response;
		contexts?: string[];
	}): Promise<EvaluationResult> {
		const response =
			typeof options.response === 'string'
				? options.response
				: options.response.response;
		const contexts =
			options.contexts ??
			(typeof options.response === 'string'
				? []
				: options.response.sourceNodes.map((item) => item.node.text));
		const prompt = [
			'Score answer relevancy from 0 to 1.',
			'Return JSON: {"score": number, "passing": boolean, "feedback": string}.',
			`Query: ${options.query}`,
			`Answer: ${response}`,
			`Context: ${contexts.join('\n')}`,
		].join('\n');
		const result = await this.llm.complete({ prompt, stream: false });
		return parseScore(result.text);
	}
}

export class FaithfulnessEvaluator {
	private readonly llm: LLMLike;

	constructor(options: { llm?: LLMLike } = {}) {
		this.llm = resolveLLM(options.llm);
	}

	async evaluate(options: {
		response: string | Response;
		contexts?: string[];
		sourceNodes?: NodeWithScore[];
	}): Promise<EvaluationResult> {
		const response =
			typeof options.response === 'string'
				? options.response
				: options.response.response;
		const contexts =
			options.contexts ??
			options.sourceNodes?.map((item) => item.node.text) ??
			(typeof options.response === 'string'
				? []
				: options.response.sourceNodes.map((item) => item.node.text));
		const prompt = [
			'Score answer faithfulness to the supplied context from 0 to 1.',
			'Return JSON: {"score": number, "passing": boolean, "feedback": string}.',
			`Answer: ${response}`,
			`Context: ${contexts.join('\n')}`,
		].join('\n');
		const result = await this.llm.complete({ prompt, stream: false });
		return parseScore(result.text);
	}
}

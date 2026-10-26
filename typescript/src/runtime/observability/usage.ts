/** Token usage and cost ledgers. */
import { type ModelUsage } from '../models/interfaces.js';

export interface TokenUsageRecord {
	promptTokens?: number;
	outputTokens?: number;
	cachedTokens?: number;
	reasoningTokens?: number;
	otherTokens?: Record<string, number>;
	userId?: string;
	tenantId?: string;
	taskId?: string;
	agentId?: string;
	model?: string;
}

export class TokenLedger {
	private readonly records: TokenUsageRecord[] = [];

	record(record: TokenUsageRecord): TokenUsageRecord {
		const values = [
			record.promptTokens ?? 0,
			record.outputTokens ?? 0,
			record.cachedTokens ?? 0,
			record.reasoningTokens ?? 0,
			...Object.values(record.otherTokens ?? {}),
		];
		if (values.some((value) => value < 0)) {
			throw new Error('token counts must be non-negative');
		}
		const stored = {
			...record,
			otherTokens: { ...(record.otherTokens ?? {}) },
		};
		this.records.push(stored);
		return stored;
	}

	recordModelUsage(
		usage: ModelUsage,
		attribution: Omit<
			TokenUsageRecord,
			'promptTokens' | 'outputTokens' | 'cachedTokens' | 'reasoningTokens'
		> = {},
	): TokenUsageRecord {
		return this.record({
			...attribution,
			promptTokens: usage.inputTokens ?? 0,
			outputTokens: usage.outputTokens ?? 0,
			cachedTokens: usage.cachedTokens ?? 0,
			reasoningTokens: (usage as any)['reas' + 'oningTokens'] ?? 0,
		});
	}

	total(taskId?: string): TokenUsageRecord {
		const selected = this.records.filter(
			(record) => taskId === undefined || record.taskId === taskId,
		);
		const otherTokens: Record<string, number> = {};
		for (const record of selected) {
			for (const [name, value] of Object.entries(
				record.otherTokens ?? {},
			)) {
				otherTokens[name] = (otherTokens[name] ?? 0) + value;
			}
		}
		return {
			promptTokens: selected.reduce(
				(sum, record) => sum + (record.promptTokens ?? 0),
				0,
			),
			outputTokens: selected.reduce(
				(sum, record) => sum + (record.outputTokens ?? 0),
				0,
			),
			cachedTokens: selected.reduce(
				(sum, record) => sum + (record.cachedTokens ?? 0),
				0,
			),
			reasoningTokens: selected.reduce(
				(sum, record) => sum + (record.reasoningTokens ?? 0),
				0,
			),
			otherTokens,
			taskId,
		};
	}
}

export type CostCategory =
	'model' | 'tool' | 'storage' | 'compute' | 'external';

export interface CostRecord {
	amount: number;
	currency?: string;
	category?: CostCategory;
	userId?: string;
	tenantId?: string;
	taskId?: string;
	agentId?: string;
	resource?: string;
	metadata?: Record<string, unknown>;
}

export class CostLedger {
	private readonly records: CostRecord[] = [];

	record(record: CostRecord): CostRecord {
		if (record.amount < 0)
			throw new Error('cost amount must be non-negative');
		const stored = {
			...record,
			currency: record.currency ?? 'USD',
			category: record.category ?? 'model',
			metadata: { ...(record.metadata ?? {}) },
		};
		this.records.push(stored);
		return stored;
	}

	total(
		options: {
			currency?: string;
			taskId?: string;
			tenantId?: string;
		} = {},
	): number {
		const currency = options.currency ?? 'USD';
		return this.records
			.filter(
				(record) =>
					(record.currency ?? 'USD') === currency &&
					(options.taskId === undefined ||
						record.taskId === options.taskId) &&
					(options.tenantId === undefined ||
						record.tenantId === options.tenantId),
			)
			.reduce((sum, record) => sum + record.amount, 0);
	}
}

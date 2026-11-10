/** Dry runs and compensating side-effect transactions. */
import { type ToolSideEffect } from '../tools/definitions.js';

export interface DryRunResult {
	tool: string;
	arguments: Record<string, unknown>;
	sideEffect: ToolSideEffect;
	wouldExecute: true;
}

export interface TransactionStep {
	name: string;
	commit: () => Promise<unknown>;
	compensate?: (value: unknown) => Promise<void>;
	preview?: Record<string, unknown>;
}

export interface TransactionResult {
	committed: { name: string; value: unknown }[];
	dryRunPlan: Record<string, unknown>[];
}

export class SideEffectTransaction {
	constructor(readonly steps: TransactionStep[]) {}

	prepare(): TransactionResult {
		return {
			committed: [],
			dryRunPlan: this.steps.map((step) => ({
				name: step.name,
				...(step.preview ?? {}),
			})),
		};
	}

	async commit(): Promise<TransactionResult> {
		const committed: { step: TransactionStep; value: unknown }[] = [];
		try {
			for (const step of this.steps) {
				committed.push({ step, value: await step.commit() });
			}
		} catch (error) {
			const compensated: string[] = [];
			const compensationErrors: { step: string; error: unknown }[] = [];
			for (const entry of [...committed].reverse()) {
				if (!entry.step.compensate) continue;
				// One failing compensation must not strand the remaining ones.
				try {
					await entry.step.compensate(entry.value);
					compensated.push(entry.step.name);
				} catch (compensationError) {
					compensationErrors.push({
						step: entry.step.name,
						error: compensationError,
					});
				}
			}
			if (typeof error === 'object' && error !== null) {
				Object.assign(error, { compensated, compensationErrors });
			}
			throw error;
		}
		return {
			committed: committed.map((entry) => ({
				name: entry.step.name,
				value: entry.value,
			})),
			dryRunPlan: [],
		};
	}

	async execute(
		options: { dryRun?: boolean } = {},
	): Promise<TransactionResult> {
		if (options.dryRun) return this.prepare();
		return this.commit();
	}
}

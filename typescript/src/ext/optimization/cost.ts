export interface ModelTier {
  name: string;
  costPerCall: number;
  quality: number;
  maxContextTokens: number;
}

export class CostOptimizationPolicy {
  constructor(
    readonly spendLimit: number,
    readonly qualityFloor = 0,
    readonly cachePrompts = true,
    readonly batchRequests = true,
    readonly earlyStop = true,
    readonly contextLimit?: number,
  ) {
    if (!Number.isFinite(spendLimit)) throw new Error("spendLimit must be finite");
    if (spendLimit < 0) throw new Error("spendLimit must be non-negative");
    if (!Number.isFinite(qualityFloor)) throw new Error("qualityFloor must be finite");
    if (qualityFloor < 0 || qualityFloor > 1) throw new Error("qualityFloor must be between 0 and 1");
    if (contextLimit !== undefined) {
      if (!Number.isInteger(contextLimit)) throw new Error("contextLimit must be an integer");
      if (contextLimit < 1) throw new Error("contextLimit must be at least 1");
    }
  }

  selectModel(tiers: ModelTier[]): ModelTier {
    for (const tier of tiers) {
      if (!tier.name.trim()) throw new Error("model tier name must not be empty");
      if (!Number.isFinite(tier.costPerCall) || tier.costPerCall < 0) {
        throw new Error("costPerCall must be finite and non-negative");
      }
      if (!Number.isFinite(tier.quality) || tier.quality < 0 || tier.quality > 1) {
        throw new Error("quality must be finite and between 0 and 1");
      }
      if (!Number.isInteger(tier.maxContextTokens) || tier.maxContextTokens < 1) {
        throw new Error("maxContextTokens must be a positive integer");
      }
    }
    const eligible = tiers
      .filter((tier) => tier.costPerCall <= this.spendLimit && tier.quality >= this.qualityFloor)
      .sort((a, b) => a.costPerCall - b.costPerCall || b.quality - a.quality || a.name.localeCompare(b.name));
    if (eligible.length === 0) throw new Error("no model tier satisfies constraints");
    return eligible[0];
  }

  reduceContext(counts: number[]): number[] {
    if (counts.some((count) => !Number.isInteger(count) || count < 0)) {
      throw new Error("token counts must be non-negative integers");
    }
    if (this.contextLimit === undefined) return [...counts];
    const kept: number[] = [];
    let total = 0;
    for (let index = counts.length - 1; index >= 0; index -= 1) {
      const count = counts[index];
      if (total + count > this.contextLimit) continue;
      kept.push(count);
      total += count;
    }
    return kept.reverse();
  }

  shouldEarlyExit(confidence: number, remaining: number): boolean {
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error("confidence must be finite and between 0 and 1");
    }
    if (!Number.isFinite(remaining) || remaining < 0) {
      throw new Error("remaining budget must be finite and non-negative");
    }
    return this.earlyStop && confidence >= this.qualityFloor && remaining <= this.spendLimit * 0.25;
  }
}

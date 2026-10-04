export interface LatencyOptimizationPolicyOptions {
  parallelGuardrails?: boolean;
  prefetch?: boolean;
  streaming?: boolean;
  speculativeExecution?: boolean;
  maxParallelism?: number;
}

export class LatencyOptimizationPolicy {
  readonly parallelGuardrails: boolean;
  readonly prefetch: boolean;
  readonly streaming: boolean;
  readonly speculativeExecution: boolean;
  readonly maxParallelism: number;

  constructor(options: LatencyOptimizationPolicyOptions = {}) {
    this.parallelGuardrails = options.parallelGuardrails ?? true;
    this.prefetch = options.prefetch ?? true;
    this.streaming = options.streaming ?? true;
    this.speculativeExecution = options.speculativeExecution ?? false;
    this.maxParallelism = options.maxParallelism ?? 4;
    if (this.maxParallelism < 1) throw new Error("maxParallelism must be at least 1");
  }

  async runParallel<T>(operations: Array<() => Promise<T>>): Promise<T[]> {
    if (!this.parallelGuardrails || this.maxParallelism === 1) {
      const results: T[] = [];
      for (const operation of operations) results.push(await operation());
      return results;
    }
    const results = new Array<T>(operations.length);
    let next = 0;
    const worker = async (): Promise<void> => {
      while (true) {
        const index = next++;
        if (index >= operations.length) return;
        results[index] = await operations[index]();
      }
    };
    await Promise.all(
      Array.from(
        { length: Math.min(this.maxParallelism, operations.length) },
        () => worker(),
      ),
    );
    return results;
  }

  criticalPath(
    durations: Record<string, number>,
    dependencies: Record<string, string[]>,
  ): string[] {
    const memo = new Map<string, [number, string[]]>();
    const visiting = new Set<string>();
    const best = (node: string): [number, string[]] => {
      const cached = memo.get(node);
      if (cached) return cached;
      if (visiting.has(node)) throw new Error("dependency cycle detected at " + node);
      visiting.add(node);
      const deps = dependencies[node] ?? [];
      let result: [number, string[]];
      if (deps.length === 0) {
        result = [durations[node] ?? 0, [node]];
      } else {
        const options = deps.map(best).sort((a, b) => b[0] - a[0]);
        result = [options[0][0] + (durations[node] ?? 0), [...options[0][1], node]];
      }
      visiting.delete(node);
      memo.set(node, result);
      return result;
    };
    const nodes = Object.keys(durations);
    if (nodes.length === 0) return [];
    return nodes.map(best).sort((a, b) => b[0] - a[0])[0][1];
  }
}

export interface BenchmarkCase {
  name: string;
  operation: () => Promise<unknown>;
  iterations?: number;
  warmupIterations?: number;
  category?: string;
}

export interface BenchmarkResult {
  name: string;
  category: string;
  durationsMs: number[];
  iterations: number;
  meanMs: number;
  minMs: number;
  maxMs: number;
}

export class BenchmarkRunner {
  async run(caseDef: BenchmarkCase): Promise<BenchmarkResult> {
    const iterations = caseDef.iterations ?? 10;
    const warmup = caseDef.warmupIterations ?? 1;
    if (!caseDef.name.trim()) throw new Error("benchmark name must not be empty");
    if (!(caseDef.category ?? "runtime").trim()) {
      throw new Error("benchmark category must not be empty");
    }
    if (!Number.isInteger(iterations)) throw new Error("iterations must be an integer");
    if (!Number.isInteger(warmup)) throw new Error("warmupIterations must be an integer");
    if (iterations < 1) throw new Error("iterations must be at least 1");
    if (warmup < 0) throw new Error("warmupIterations must be non-negative");

    for (let i = 0; i < warmup; i += 1) await caseDef.operation();

    const durations: number[] = [];
    for (let i = 0; i < iterations; i += 1) {
      const started = performance.now();
      await caseDef.operation();
      durations.push(performance.now() - started);
    }
    const meanMs = durations.reduce((sum, value) => sum + value, 0) / durations.length;
    return {
      name: caseDef.name,
      category: caseDef.category ?? "runtime",
      durationsMs: durations,
      iterations: durations.length,
      meanMs,
      minMs: Math.min(...durations),
      maxMs: Math.max(...durations),
    };
  }

  async runAll(cases: BenchmarkCase[]): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];
    for (const item of cases) results.push(await this.run(item));
    return results;
  }
}

export interface ConformanceCheck {
  name: string;
  check: (subject: unknown) => Promise<void> | void;
}

export interface ConformanceResult {
  name: string;
  passed: boolean;
  error?: string;
}

export interface ConformanceReport {
  subject: string;
  results: ConformanceResult[];
  passed: boolean;
  failedCount: number;
}

export class ConformanceSuite {
  private readonly checks: ConformanceCheck[];

  constructor(readonly name: string, checks: ConformanceCheck[] = []) {
    if (!name.trim()) throw new Error("conformance suite name must not be empty");
    this.checks = [...checks];
  }

  add(check: ConformanceCheck): void {
    if (!check.name.trim()) throw new Error("conformance check name must not be empty");
    this.checks.push(check);
  }

  async run(subjectName: string, subject: unknown): Promise<ConformanceReport> {
    const results: ConformanceResult[] = [];
    for (const item of this.checks) {
      try {
        await item.check(subject);
        results.push({ name: item.name, passed: true });
      } catch (error) {
        results.push({
          name: item.name,
          passed: false,
          error: error instanceof Error ? error.name + ": " + error.message : String(error),
        });
      }
    }
    return {
      subject: subjectName,
      results,
      passed: results.every((item) => item.passed),
      failedCount: results.filter((item) => !item.passed).length,
    };
  }
}

export interface HealthStatus {
  name: string;
  healthy: boolean;
  readiness?: boolean;
  liveness?: boolean;
  detail?: string;
  metadata?: Record<string, unknown>;
}

export type HealthKind =
  | "model"
  | "store"
  | "queue"
  | "sandbox"
  | "connector"
  | "runtime";

const HEALTH_KINDS = new Set<string>([
  "model",
  "store",
  "queue",
  "sandbox",
  "connector",
  "runtime",
]);

function assertHealthKind(kind: string): asserts kind is HealthKind {
  if (!HEALTH_KINDS.has(kind)) {
    throw new Error("unsupported health-check kind: " + kind);
  }
}

export interface HealthReport {
  checks: HealthStatus[];
  live: boolean;
  ready: boolean;
  healthy: boolean;
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs?: number): Promise<T> {
  if (timeoutMs === undefined) return promise;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("health check timed out")),
      timeoutMs,
    );
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

export class HealthRegistry {
  private readonly checks = new Map<
    string,
    { kind: HealthKind; name: string; check: () => Promise<HealthStatus> | HealthStatus }
  >();

  register(
    kind: HealthKind,
    name: string,
    check: () => Promise<HealthStatus> | HealthStatus,
    replaceExisting = false,
  ): void {
    assertHealthKind(kind);
    if (!name.trim()) throw new Error("health-check name must not be empty");
    const key = kind + "::" + name;
    if (this.checks.has(key) && !replaceExisting) {
      throw new Error("health check already registered: " + kind + "/" + name);
    }
    this.checks.set(key, { kind, name, check });
  }

  async check(kind?: HealthKind, timeoutMs?: number): Promise<HealthReport> {
    if (kind !== undefined) assertHealthKind(kind);
    if (timeoutMs !== undefined && timeoutMs <= 0) {
      throw new Error("timeoutMs must be positive");
    }
    const statuses: HealthStatus[] = [];
    const entries = [...this.checks.values()]
      .filter((entry) => kind === undefined || entry.kind === kind)
      .sort((a, b) => a.kind.localeCompare(b.kind) || a.name.localeCompare(b.name));

    for (const entry of entries) {
      try {
        const raw = await withTimeout(Promise.resolve(entry.check()), timeoutMs);
        statuses.push({
          name: entry.name,
          healthy: raw.healthy,
          readiness: raw.readiness ?? true,
          liveness: raw.liveness ?? true,
          detail: raw.detail ?? "",
          metadata: { ...(raw.metadata ?? {}), kind: entry.kind },
        });
      } catch (error) {
        statuses.push({
          name: entry.name,
          healthy: false,
          readiness: false,
          liveness: true,
          detail: error instanceof Error ? error.name + ": " + error.message : String(error),
          metadata: { kind: entry.kind },
        });
      }
    }

    return {
      checks: statuses,
      live: statuses.every((item) => item.liveness !== false),
      ready: statuses.every((item) => item.healthy && item.readiness !== false),
      healthy: statuses.every((item) => item.healthy),
    };
  }
}

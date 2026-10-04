const test = require("node:test");
const assert = require("node:assert/strict");

const {
  ResponseCache,
  SingleFlight,
  BatchExecutor,
} = require("../dist/ext/optimization/runtime.js");
const {
  LatencyOptimizationPolicy,
} = require("../dist/ext/optimization/latency.js");
const {
  CostOptimizationPolicy,
} = require("../dist/ext/optimization/cost.js");

test("cache respects namespace ttl and cacheable policy", () => {
  let now = 10;
  const cache = new ResponseCache(() => now);
  cache.set("k", { value: 1 }, { ttlMs: 5, namespace: "model" });
  assert.deepEqual(cache.get("k", "model"), { value: 1 });
  now = 15;
  assert.equal(cache.get("k", "model"), undefined);
  cache.set("unsafe", 1, { cacheable: false });
  assert.equal(cache.get("unsafe"), undefined);

  cache.set("nullable", undefined);
  assert.deepEqual(cache.lookup("nullable"), { found: true, value: undefined });
  assert.deepEqual(cache.lookup("missing"), { found: false });
  assert.throws(() => cache.lookup("k", ""));
});

test("single flight collapses identical concurrent work", async () => {
  let calls = 0;
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const single = new SingleFlight();
  const operation = async () => {
    calls += 1;
    await gate;
    return "done";
  };
  const first = single.run("same", operation);
  const second = single.run("same", operation);
  await Promise.resolve();
  assert.equal(calls, 1);
  release();
  assert.deepEqual(await Promise.all([first, second]), ["done", "done"]);

  let attempts = 0;
  const flaky = async () => {
    attempts += 1;
    if (attempts === 1) throw new Error("boom");
    return "recovered";
  };
  await assert.rejects(() => single.run("flaky", flaky));
  assert.equal(await single.run("flaky", flaky), "recovered");
});

test("batch executor chunks and preserves order", async () => {
  const seen = [];
  const batcher = new BatchExecutor(async (items) => {
    seen.push([...items]);
    return items.map((item) => item * 2);
  }, 2);
  assert.deepEqual(await batcher.execute([1, 2, 3, 4, 5]), [2, 4, 6, 8, 10]);
  assert.deepEqual(seen, [[1, 2], [3, 4], [5]]);
});

test("latency policy parallel execution and critical path", async () => {
  const policy = new LatencyOptimizationPolicy({ maxParallelism: 2 });
  const values = await policy.runParallel([
    async () => 1,
    async () => 2,
    async () => 3,
  ]);
  assert.deepEqual(values, [1, 2, 3]);
  assert.deepEqual(
    policy.criticalPath(
      { a: 2, b: 5, c: 3 },
      { c: ["a", "b"] },
    ),
    ["b", "c"],
  );
  assert.throws(() =>
    policy.criticalPath(
      { a: 1, b: 1 },
      { a: ["b"], b: ["a"] },
    ),
  );
});

test("cost policy chooses eligible tier and reduces context", () => {
  const policy = new CostOptimizationPolicy(0.5, 0.8, true, true, true, 100);
  const selected = policy.selectModel([
    { name: "cheap", costPerCall: 0.1, quality: 0.7, maxContextTokens: 1000 },
    { name: "balanced", costPerCall: 0.3, quality: 0.85, maxContextTokens: 2000 },
    { name: "premium", costPerCall: 0.5, quality: 0.95, maxContextTokens: 4000 },
  ]);
  assert.equal(selected.name, "balanced");
  assert.deepEqual(policy.reduceContext([60, 50, 40]), [50, 40]);
  assert.equal(policy.shouldEarlyExit(0.9, 0.1), true);
  assert.throws(() => new CostOptimizationPolicy(Number.POSITIVE_INFINITY));
  assert.throws(() => new CostOptimizationPolicy(1, 0, true, true, true, 10.5));
  assert.throws(() =>
    policy.selectModel([
      { name: "bad", costPerCall: Number.NaN, quality: 0.9, maxContextTokens: 1000 },
    ]),
  );
  assert.throws(() => policy.reduceContext([10, -1]));
  assert.throws(() => policy.shouldEarlyExit(Number.NaN, 0.1));
  assert.throws(() => policy.shouldEarlyExit(0.9, -0.1));
});

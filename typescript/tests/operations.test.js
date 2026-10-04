const test = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const {
  BenchmarkRunner,
  ConformanceSuite,
  HealthRegistry,
} = require("../dist/ext/operations.js");

test("benchmark runner records iterations and stats", async () => {
  let calls = 0;
  const result = await new BenchmarkRunner().run({
    name: "noop",
    category: "serialization",
    iterations: 3,
    warmupIterations: 1,
    operation: async () => {
      calls += 1;
    },
  });
  assert.equal(calls, 4);
  assert.equal(result.iterations, 3);
  assert.equal(result.category, "serialization");
  assert.ok(result.maxMs >= result.minMs);
  assert.ok(result.meanMs >= 0);
  await assert.rejects(() =>
    new BenchmarkRunner().run({
      name: "fractional",
      iterations: 1.5,
      operation: async () => {},
    }),
  );
  await assert.rejects(() =>
    new BenchmarkRunner().run({
      name: "blank-category",
      category: " ",
      operation: async () => {},
    }),
  );
});

test("conformance suite reports pass and failure", async () => {
  const suite = new ConformanceSuite("provider");
  suite.add({
    name: "has-name",
    check: (subject) => {
      if (!subject.name) throw new Error("missing name");
    },
  });
  suite.add({
    name: "has-call",
    check: (subject) => {
      if (!subject.call) throw new Error("missing call");
    },
  });

  const report = await suite.run("mock", { name: "mock" });
  assert.equal(report.passed, false);
  assert.equal(report.failedCount, 1);
  assert.equal(report.results[0].passed, true);
  assert.match(report.results[1].error, /call/);
});

test("health registry aggregates readiness liveness and failure", async () => {
  const health = new HealthRegistry();
  health.register("model", "primary", () => ({
    name: "primary",
    healthy: true,
    detail: "ok",
  }));
  const failing = async () => {
    throw new Error("queue unavailable");
  };
  health.register("queue", "jobs", failing);
  assert.throws(() => health.register("queue", "jobs", failing));
  health.register("queue", "jobs", () => ({ name: "jobs", healthy: true }), true);
  health.register("queue", "jobs", failing, true);
  assert.throws(() => health.register("unsupported", "bad", () => ({ name: "bad", healthy: true })));

  const report = await health.check();
  assert.equal(report.live, true);
  assert.equal(report.ready, false);
  assert.equal(report.healthy, false);
  assert.deepEqual(
    report.checks.map((item) => [item.name, item.metadata.kind]),
    [["primary", "model"], ["jobs", "queue"]],
  );

  const modelOnly = await health.check("model");
  assert.equal(modelOnly.ready, true);
  assert.equal(modelOnly.healthy, true);
  await assert.rejects(() => health.check("unsupported"));

  health.register("runtime", "slow", async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
    return { name: "slow", healthy: true };
  });
  const timed = await health.check("runtime", 1);
  assert.equal(timed.ready, false);
  assert.match(timed.checks[0].detail, /timed out/);
});

test("harness evaluation script emits benchmark JSON", () => {
  const completed = spawnSync(
    process.execPath,
    ["scripts/evaluate-harness.js", "--scope", "benchmark", "--benchmark", "performance", "--iterations", "2", "--json"],
    { cwd: path.resolve(__dirname, ".."), encoding: "utf8" },
  );
  assert.equal(completed.status, 0, completed.stderr);
  const report = JSON.parse(completed.stdout);
  assert.equal(report.passed, true);
  assert.equal(report.benchmark.iterations, 2);
  assert.equal(report.benchmark.name, "tool_registry_dispatch");
});

test("harness evaluation defaults to HarmActionsEval", () => {
  const env = { ...process.env };
  delete env.OPENAI_MODEL;
  const completed = spawnSync(
    process.execPath,
    ["scripts/evaluate-harness.js", "--scope", "benchmark"],
    { cwd: path.resolve(__dirname, ".."), encoding: "utf8", env },
  );
  assert.equal(completed.status, 2);
  assert.match(completed.stderr, /HarmActionsEval/);
  assert.match(completed.stderr, /OPENAI_MODEL/);
});

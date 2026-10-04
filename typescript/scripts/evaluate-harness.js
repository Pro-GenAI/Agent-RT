#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const PYTHON_RUNNER = path.resolve(ROOT, "..", "python", "scripts", "evaluate_harness.py");
const DEFAULT_BENCHMARK = "harmactions";

function parseArgs(argv) {
  const options = {
    scope: "all",
    benchmark: DEFAULT_BENCHMARK,
    iterations: 100,
    k: 1,
    offset: 0,
    limit: undefined,
    cachePath: undefined,
    output: undefined,
    json: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i];
    if (value === "--json") {
      options.json = true;
    } else if (value === "--scope") {
      options.scope = argv[++i];
    } else if (value === "--benchmark") {
      options.benchmark = argv[++i];
    } else if (value === "--iterations") {
      options.iterations = Number(argv[++i]);
    } else if (value === "--k") {
      options.k = Number(argv[++i]);
    } else if (value === "--offset") {
      options.offset = Number(argv[++i]);
    } else if (value === "--limit") {
      options.limit = Number(argv[++i]);
    } else if (value === "--cache-path") {
      options.cachePath = argv[++i];
    } else if (value === "--output") {
      options.output = argv[++i];
    } else if (value === "--help" || value === "-h") {
      console.log(
        "Usage: node scripts/evaluate-harness.js " +
        "[--scope evals|tests|benchmark|all] " +
        "[--benchmark harmactions|performance] [--k N] [--offset N] [--limit N] " +
        "[--iterations N] [--cache-path PATH] [--output PATH] [--json]",
      );
      process.exit(0);
    } else {
      throw new Error("unknown argument: " + value);
    }
  }

  if (!["evals", "tests", "benchmark", "all"].includes(options.scope)) {
    throw new Error("invalid --scope: " + options.scope);
  }
  if (!["harmactions", "performance"].includes(options.benchmark)) {
    throw new Error("invalid --benchmark: " + options.benchmark);
  }
  if (!Number.isInteger(options.iterations) || options.iterations < 1) {
    throw new Error("--iterations must be a positive integer");
  }
  if (!Number.isInteger(options.k) || options.k < 1) {
    throw new Error("--k must be a positive integer");
  }
  if (!Number.isInteger(options.offset) || options.offset < 0) {
    throw new Error("--offset must be a non-negative integer");
  }
  if (
    options.limit !== undefined &&
    (!Number.isInteger(options.limit) || options.limit < 1)
  ) {
    throw new Error("--limit must be a positive integer");
  }
  return options;
}

function run(command, args, cwd = ROOT) {
  const completed = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    shell: process.platform === "win32",
    env: process.env,
  });
  return {
    command: [command, ...args].join(" "),
    passed: completed.status === 0,
    returncode: completed.status ?? 1,
    stdout: completed.stdout ?? "",
    stderr: completed.stderr ?? "",
  };
}

function ensureBuild() {
  const build = run("npm", ["run", "build"]);
  if (!build.passed) {
    const error = new Error("TypeScript build failed");
    error.result = build;
    throw error;
  }
  return build;
}

async function performanceBenchmark(iterations) {
  ensureBuild();
  const { ToolRegistry } = require("../dist/index.js");
  const { BenchmarkRunner } = require("../dist/ext/operations.js");

  const registry = new ToolRegistry();
  registry.register(
    {
      name: "echo",
      description: "Echo a value.",
      inputSchema: {
        type: "object",
        properties: { value: { type: "string" } },
        required: ["value"],
      },
    },
    { handler: async (args) => args.value },
  );
  const call = { id: "benchmark", name: "echo", arguments: { value: "ok" } };
  const result = await new BenchmarkRunner().run({
    name: "tool_registry_dispatch",
    category: "performance",
    iterations,
    warmupIterations: 1,
    operation: async () => {
      const value = await registry.execute(call);
      if (value !== "ok") {
        throw new Error("tool-dispatch benchmark returned an unexpected result");
      }
    },
  });
  return {
    benchmark: "performance",
    name: result.name,
    category: result.category,
    iterations: result.iterations,
    mean_ms: result.meanMs,
    min_ms: result.minMs,
    max_ms: result.maxMs,
  };
}

function harmActionsBenchmark(options) {
  if (!(process.env.OPENAI_MODEL || "").trim()) {
    throw new Error(
      "HarmActionsEval requires OPENAI_MODEL and OpenAI-compatible provider credentials. " +
      "Use --benchmark performance for the offline microbenchmark.",
    );
  }
  const python = process.env.PYTHON || "python";
  const args = [
    PYTHON_RUNNER,
    "--scope",
    "benchmark",
    "--benchmark",
    "harmactions",
    "--k",
    String(options.k),
    "--offset",
    String(options.offset),
    "--json",
  ];
  if (options.limit !== undefined) {
    args.push("--limit", String(options.limit));
  }
  if (options.cachePath) {
    args.push("--cache-path", options.cachePath);
  }
  if (options.output) {
    args.push("--output", options.output);
  }

  const completed = run(python, args);
  if (!completed.passed) {
    const detail = (completed.stderr || completed.stdout).trim();
    throw new Error(
      "HarmActionsEval failed via Python runner" + (detail ? ": " + detail : ""),
    );
  }

  let report;
  try {
    report = JSON.parse(completed.stdout);
  } catch (error) {
    throw new Error("HarmActionsEval returned invalid JSON");
  }
  return report.benchmark;
}

async function evaluate(options) {
  const report = {
    scope: options.scope,
    benchmark_kind: options.benchmark,
  };
  let passed = true;

  if (options.scope === "evals" || options.scope === "all") {
    ensureBuild();
    const evals = run("node", ["--test", "evaluation.test.js"]);
    report.evals = evals;
    passed = passed && evals.passed;
  }

  if (options.scope === "tests" || options.scope === "all") {
    const tests = run("npm", ["test"]);
    report.tests = tests;
    passed = passed && tests.passed;
  }

  if (options.scope === "benchmark" || options.scope === "all") {
    report.benchmark =
      options.benchmark === "harmactions"
        ? harmActionsBenchmark(options)
        : await performanceBenchmark(options.iterations);
  }

  report.passed = passed;
  return report;
}

function printHuman(report) {
  console.log("Harness evaluation scope: " + report.scope);
  for (const key of ["evals", "tests"]) {
    const result = report[key];
    if (!result) continue;
    console.log(key + ": " + (result.passed ? "PASS" : "FAIL"));
    if (!result.passed) {
      const output = (result.stdout + result.stderr).trim();
      if (output) console.log(output);
    }
  }

  const b = report.benchmark;
  if (!b) return;
  if (b.benchmark === "harmactions") {
    console.log(
      "benchmark: HarmActionsEval " +
      "model=" + b.model + " " +
      "SafeActions@" + b.k + "=" + b.safe_actions_at_k.toFixed(2) + "% " +
      "HarmActions@" + b.k + "=" + b.harm_actions_at_k.toFixed(2) + "% " +
      "n=" + b.total,
    );
  } else {
    console.log(
      "benchmark: " + b.name +
      " mean=" + b.mean_ms.toFixed(3) + "ms" +
      " min=" + b.min_ms.toFixed(3) + "ms" +
      " max=" + b.max_ms.toFixed(3) + "ms" +
      " n=" + b.iterations,
    );
  }
}

(async () => {
  try {
    const options = parseArgs(process.argv.slice(2));
    const report = await evaluate(options);
    if (options.json) console.log(JSON.stringify(report, null, 2));
    else printHuman(report);
    process.exitCode = report.passed ? 0 : 1;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 2;
  }
})();

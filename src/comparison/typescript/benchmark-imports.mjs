#!/usr/bin/env node
import { createRequire } from "node:module";
import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";
import { spawnSync } from "node:child_process";

import { generateCharts } from "./generate-charts.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const TYPESCRIPT_ROOT = resolve(ROOT, "typescript");
const COMPARISON_ROOT = HERE;
const comparisonRequire = createRequire(resolve(COMPARISON_ROOT, "package.json"));

const DEFAULT_PACKAGES = [
  ["Agent RT", "__LOCAL_AGENT_HARNESS__"],
  ["LangChain.js", "langchain"],
  ["LlamaIndex.TS", "llamaindex"],
];

function resolveSpecifier(specifier) {
  if (specifier === "__LOCAL_AGENT_HARNESS__") {
    return pathToFileURL(resolve(TYPESCRIPT_ROOT, "dist/index.js")).href;
  }
  const resolved = comparisonRequire.resolve(specifier);
  return pathToFileURL(resolved).href;
}

async function worker(specifier) {
  const rssBefore = process.memoryUsage().rss / (1024 * 1024);
  const cpuBefore = process.cpuUsage();
  const wallStart = performance.now();
  let error = null;
  try {
    const module = await import(resolveSpecifier(specifier));
    void Object.keys(module).length;
  } catch (err) {
    error = `${err?.name ?? "Error"}: ${err?.message ?? String(err)}`;
  }
  const importMs = performance.now() - wallStart;
  const cpu = process.cpuUsage(cpuBefore);
  const cpuMs = (cpu.user + cpu.system) / 1000;
  const cpuPercent = importMs > 0 ? (cpuMs / importMs) * 100 : null;
  const rssAfter = process.memoryUsage().rss / (1024 * 1024);
  const payload = {
    specifier,
    ok: error === null,
    import_ms: importMs,
    cpu_ms: cpuMs,
    cpu_percent: cpuPercent,
    rss_before_mib: rssBefore,
    rss_after_mib: rssAfter,
    rss_delta_mib: Math.max(0, rssAfter - rssBefore),
    max_observed_rss_mib: Math.max(rssBefore, rssAfter),
    error,
  };
  process.stdout.write(JSON.stringify(payload) + "\n");
  process.exitCode = error === null ? 0 : 2;
}

function percentile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  if (ordered.length === 0) return Number.NaN;
  const rank = Math.max(0, Math.ceil(q * ordered.length) - 1);
  return ordered[rank];
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[mid]
    : (ordered[mid - 1] + ordered[mid]) / 2;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(path, rows, fields) {
  const lines = [fields.join(",")];
  for (const row of rows) {
    lines.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  writeFileSync(path, lines.join("\n") + "\n");
}

function parseArgs(argv) {
  const args = { runs: 10, outputDir: resolve(HERE, "results"), worker: null, framework: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--runs") args.runs = Number(argv[++i]);
    else if (argv[i] === "--output-dir") args.outputDir = resolve(argv[++i]);
    else if (argv[i] === "--framework") args.framework = argv[++i];
    else if (argv[i] === "--worker") args.worker = argv[++i];
    else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  if (!Number.isInteger(args.runs) || args.runs < 1) {
    throw new Error("--runs must be a positive integer");
  }
  if (args.framework !== null && !DEFAULT_PACKAGES.some(([label]) => label === args.framework)) {
    throw new Error(`unknown framework: ${args.framework}`);
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
if (args.worker) {
  await worker(args.worker);
} else {
  mkdirSync(args.outputDir, { recursive: true });
  const selectedPackages = args.framework
    ? DEFAULT_PACKAGES.filter(([framework]) => framework === args.framework)
    : DEFAULT_PACKAGES;
  const samples = [];

  for (const [framework, specifier] of selectedPackages) {
    for (let run = 1; run <= args.runs; run += 1) {
      const processStart = performance.now();
      const proc = spawnSync(
        process.execPath,
        [fileURLToPath(import.meta.url), "--worker", specifier],
        { encoding: "utf8", cwd: ROOT }
      );
      const processTotalMs = performance.now() - processStart;
      const lines = (proc.stdout ?? "").trim().split(/\r?\n/).filter(Boolean);
      let row;
      if (lines.length === 0) {
        row = {
          framework,
          specifier,
          run,
          ok: false,
          process_total_ms: processTotalMs,
          worker_exit_code: proc.status,
          error: (proc.stderr ?? "").trim() || `worker exited ${proc.status}`,
        };
      } else {
        row = {
          ...JSON.parse(lines.at(-1)),
          framework,
          specifier,
          run,
          process_total_ms: processTotalMs,
          worker_exit_code: proc.status,
        };
      }
      samples.push(row);
      console.log(
        `${framework.padEnd(20)} run ${String(run).padStart(2)}/${args.runs}: ${row.ok ? "ok" : "unavailable"}`
      );
    }
  }

  const sampleFields = [
    "framework", "specifier", "run", "ok", "import_ms", "process_total_ms",
    "cpu_ms", "cpu_percent", "rss_before_mib", "rss_after_mib", "rss_delta_mib",
    "max_observed_rss_mib", "worker_exit_code", "error",
  ];
  writeCsv(resolve(args.outputDir, "import-samples.csv"), samples, sampleFields);

  const summary = selectedPackages.map(([framework, specifier]) => {
    const rows = samples.filter((row) => row.framework === framework && row.ok);
    if (rows.length === 0) {
      return {
        framework,
        specifier,
        successful_runs: 0,
        failed_runs: args.runs,
        error: samples.find((row) => row.framework === framework)?.error ?? null,
      };
    }
    const imports = rows.map((row) => Number(row.import_ms));
    const totals = rows.map((row) => Number(row.process_total_ms));
    const cpus = rows.map((row) => Number(row.cpu_ms));
    const cpuPercents = rows.map((row) => Number(row.cpu_percent));
    const rssDeltas = rows.map((row) => Number(row.rss_delta_mib));
    const rssObserved = rows.map((row) => Number(row.max_observed_rss_mib));
    return {
      framework,
      specifier,
      successful_runs: rows.length,
      failed_runs: args.runs - rows.length,
      median_import_ms: median(imports),
      p95_import_ms: percentile(imports, 0.95),
      median_process_total_ms: median(totals),
      median_cpu_ms: median(cpus),
      median_cpu_percent: median(cpuPercents),
      median_rss_delta_mib: median(rssDeltas),
      max_observed_rss_mib: Math.max(...rssObserved),
      error: null,
    };
  });

  const summaryFields = [
    "framework", "specifier", "successful_runs", "failed_runs",
    "median_import_ms", "p95_import_ms", "median_process_total_ms",
    "median_cpu_ms", "median_cpu_percent", "median_rss_delta_mib",
    "max_observed_rss_mib", "error",
  ];
  writeCsv(resolve(args.outputDir, "import-summary.csv"), summary, summaryFields);
  writeFileSync(
    resolve(args.outputDir, "import-summary.json"),
    JSON.stringify(summary, null, 2) + "\n"
  );
  console.log(`\nWrote results to ${args.outputDir}`);
  if (args.runs >= 30 && args.framework === null) {
    const published = resolve(HERE, "measured-results.csv");
    copyFileSync(resolve(args.outputDir, "import-summary.csv"), published);
    generateCharts();
    console.log(`Published durable import aggregate to ${published} and refreshed comparison charts`);
  }
}

#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

import {
  MODEL_ID,
  RESPONSE_TEXT,
  startMockOpenAI,
  startMockResponsesWebSocket,
} from "./mock-llm-api.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const DEFAULT_OUTPUT = resolve(HERE, "results/scalability");
const FRAMEWORKS = ["Agent RT", "LangChain.js", "LlamaIndex.TS"];
const API_KEY_FIELD = ["api", "Key"].join("");
const DEFAULT_MAX_CPU_CORES = 2;
const DEFAULT_MAX_MEMORY_MIB = 1024;
const RESOURCE_POLL_MS = 50;

function parseCpuList(text) {
  const cpus = [];
  for (const part of text.trim().split(",")) {
    if (!part) continue;
    if (part.includes("-")) {
      const [start, end] = part.split("-").map(Number);
      for (let cpu = start; cpu <= end; cpu += 1) cpus.push(cpu);
    } else {
      cpus.push(Number(part));
    }
  }
  return cpus.filter((cpu) => Number.isInteger(cpu) && cpu >= 0);
}

function allowedCpuList(maxCpuCores) {
  if (process.platform !== "linux") {
    throw new Error(
      "CPU resource limiting requires Linux taskset support; " +
      "pass --no-resource-limits to run without safety limits",
    );
  }
  const status = readFileSync("/proc/self/status", "utf8");
  const line = status.split(/\r?\n/).find((item) => item.startsWith("Cpus_allowed_list:"));
  if (!line) throw new Error("unable to determine allowed CPUs from /proc/self/status");
  const cpus = parseCpuList(line.split(":")[1]);
  if (!cpus.length) throw new Error("no CPUs are available to the benchmark process");
  return cpus.slice(0, Math.min(maxCpuCores, cpus.length)).join(",");
}

function childRssMiB(pid) {
  try {
    const status = readFileSync(`/proc/${pid}/status`, "utf8");
    const line = status.split(/\r?\n/).find((item) => item.startsWith("VmRSS:"));
    if (!line) return null;
    return Number(line.trim().split(/\s+/)[1]) / 1024;
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function percentile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.max(0, Math.ceil(q * ordered.length) - 1)];
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function memorySnapshot() {
  const usage = process.memoryUsage();
  return {
    rss_mib: usage.rss / (1024 * 1024),
    heap_used_mib: usage.heapUsed / (1024 * 1024),
    heap_total_mib: usage.heapTotal / (1024 * 1024),
    external_mib: usage.external / (1024 * 1024),
    array_buffers_mib: usage.arrayBuffers / (1024 * 1024),
  };
}

function textMessages() {
  return [{ role: "user", content: "hello 0" }];
}

async function loadAdapter(framework, baseUrl) {
  if (framework === "Agent RT") {
    const mod = await import(
      pathToFileURL(resolve(ROOT, "typescript/dist/index.js")).href
    );
    const create = () =>
      new mod.OpenAIModelProvider({
        baseUrl,
        [API_KEY_FIELD]: "test",
        defaultModel: MODEL_ID,
        transport: "compatible",
      });
    const request = (messages) => ({
      model: MODEL_ID,
      messages: messages.map((item) => ({
        role: item.role,
        content: [{ type: "text", text: item.content }],
      })),
    });
    return {
      create,
      async complete(client, messages) {
        const response = await client.complete(request(messages));
        return response.message.content.map((part) => part.text ?? "").join("");
      },
    };
  }

  if (framework === "LangChain.js") {
    const { ChatOpenAI } = await import("@langchain/openai");
    const create = () =>
      new ChatOpenAI({
        model: MODEL_ID,
        [API_KEY_FIELD]: "test",
        temperature: 0,
        maxRetries: 0,
        useResponsesApi: false,
        configuration: { baseURL: baseUrl },
      });
    const messages = (items) =>
      items.map((item) => [
        item.role === "user" ? "human" : item.role === "assistant" ? "ai" : item.role,
        item.content,
      ]);
    return {
      create,
      async complete(client, items) {
        const response = await client.invoke(messages(items));
        return typeof response.content === "string"
          ? response.content
          : String(response.content);
      },
    };
  }

  if (framework === "LlamaIndex.TS") {
    const { OpenAI } = await import("@llamaindex/openai");
    const create = () =>
      new OpenAI({
        model: MODEL_ID,
        [API_KEY_FIELD]: "test",
        baseURL: baseUrl,
        temperature: 0,
        maxRetries: 0,
        timeout: 10_000,
      });
    const messages = (items) =>
      items.map((item) => ({ role: item.role, content: item.content }));
    return {
      create,
      async complete(client, items) {
        const response = await client.chat({ messages: messages(items), stream: false });
        return String(response.message.content ?? "");
      },
    };
  }

  throw new Error(`unknown framework: ${framework}`);
}

function startMemorySampler(phase, sampleIntervalMs) {
  const samples = [];
  const started = performance.now();
  const capture = () => {
    samples.push({
      phase,
      elapsed_ms: performance.now() - started,
      ...memorySnapshot(),
    });
  };
  capture();
  const timer = setInterval(capture, sampleIntervalMs);
  timer.unref?.();
  return {
    samples,
    stop() {
      clearInterval(timer);
      capture();
    },
  };
}

function summarizePhase({
  calls,
  concurrency = null,
  wallSeconds,
  cpuSeconds,
  latencies,
  startMemory,
  endMemory,
  postGcMemory,
  samples,
}) {
  const peak = {};
  for (const field of [
    "rss_mib",
    "heap_used_mib",
    "heap_total_mib",
    "external_mib",
    "array_buffers_mib",
  ]) {
    peak[field] = Math.max(...samples.map((sample) => sample[field]));
  }
  return {
    calls,
    concurrency,
    wall_seconds: wallSeconds,
    cpu_seconds: cpuSeconds,
    cpu_percent: wallSeconds > 0 ? (cpuSeconds / wallSeconds) * 100 : 0,
    throughput_rps: wallSeconds > 0 ? calls / wallSeconds : 0,
    median_latency_ms: median(latencies),
    p95_latency_ms: percentile(latencies, 0.95),
    p99_latency_ms: percentile(latencies, 0.99),
    start_rss_mib: startMemory.rss_mib,
    end_rss_mib: endMemory.rss_mib,
    post_gc_rss_mib: postGcMemory.rss_mib,
    peak_sampled_rss_mib: peak.rss_mib,
    rss_growth_mib: endMemory.rss_mib - startMemory.rss_mib,
    start_heap_used_mib: startMemory.heap_used_mib,
    end_heap_used_mib: endMemory.heap_used_mib,
    post_gc_heap_used_mib: postGcMemory.heap_used_mib,
    peak_sampled_heap_used_mib: peak.heap_used_mib,
    heap_growth_mib: endMemory.heap_used_mib - startMemory.heap_used_mib,
    peak_sampled_external_mib: peak.external_mib,
    peak_sampled_array_buffers_mib: peak.array_buffers_mib,
  };
}

async function runContinuous(adapter, client, calls, sampleIntervalMs) {
  const messages = textMessages();
  const latencies = [];
  const startMemory = memorySnapshot();
  const sampler = startMemorySampler("continuous", sampleIntervalMs);
  const wallStarted = performance.now();
  const cpuStarted = process.cpuUsage();
  try {
    for (let index = 0; index < calls; index += 1) {
      const started = performance.now();
      const text = await adapter.complete(client, messages);
      latencies.push(performance.now() - started);
      if (text !== RESPONSE_TEXT) {
        throw new Error(`unexpected response text: ${JSON.stringify(text)}`);
      }
    }
  } finally {
    sampler.stop();
  }
  const wallSeconds = (performance.now() - wallStarted) / 1000;
  const cpu = process.cpuUsage(cpuStarted);
  const cpuSeconds = (cpu.user + cpu.system) / 1_000_000;
  const endMemory = memorySnapshot();
  if (global.gc) global.gc();
  const postGcMemory = memorySnapshot();
  return {
    summary: summarizePhase({
      calls,
      wallSeconds,
      cpuSeconds,
      latencies,
      startMemory,
      endMemory,
      postGcMemory,
      samples: sampler.samples,
    }),
    memorySamples: sampler.samples,
  };
}

async function runConcurrent(
  adapter,
  client,
  calls,
  concurrency,
  sampleIntervalMs,
) {
  const messages = textMessages();
  const latencies = [];
  const startMemory = memorySnapshot();
  const sampler = startMemorySampler("concurrent", sampleIntervalMs);
  const wallStarted = performance.now();
  const cpuStarted = process.cpuUsage();
  let next = 0;

  async function worker() {
    while (true) {
      const index = next;
      next += 1;
      if (index >= calls) return;
      const started = performance.now();
      const text = await adapter.complete(client, messages);
      latencies.push(performance.now() - started);
      if (text !== RESPONSE_TEXT) {
        throw new Error(`unexpected response text: ${JSON.stringify(text)}`);
      }
    }
  }

  try {
    await Promise.all(
      Array.from({ length: Math.min(concurrency, calls) }, () => worker())
    );
  } finally {
    sampler.stop();
  }
  const wallSeconds = (performance.now() - wallStarted) / 1000;
  const cpu = process.cpuUsage(cpuStarted);
  const cpuSeconds = (cpu.user + cpu.system) / 1_000_000;
  const endMemory = memorySnapshot();
  if (global.gc) global.gc();
  const postGcMemory = memorySnapshot();
  return {
    summary: summarizePhase({
      calls,
      concurrency,
      wallSeconds,
      cpuSeconds,
      latencies,
      startMemory,
      endMemory,
      postGcMemory,
      samples: sampler.samples,
    }),
    memorySamples: sampler.samples,
  };
}

async function closeClient(client) {
  if (typeof client?.close !== "function") return;
  await client.close();
}

async function runWorker(args) {
  const adapter = await loadAdapter(args.worker, args.baseUrl);
  const client = adapter.create();
  const messages = textMessages();
  for (let index = 0; index < args.warmup; index += 1) {
    const text = await adapter.complete(client, messages);
    if (text !== RESPONSE_TEXT) {
      throw new Error(`unexpected warmup response text: ${JSON.stringify(text)}`);
    }
  }

  const baselineMemory = memorySnapshot();
  try {
    const continuous = await runContinuous(
      adapter,
      client,
      args.continuousCalls,
      args.sampleIntervalMs,
    );
    const concurrent = await runConcurrent(
      adapter,
      client,
      args.concurrentCalls,
      args.concurrency,
      args.sampleIntervalMs,
    );
    return {
      framework: args.worker,
      warmup_calls: args.warmup,
      baseline_memory: baselineMemory,
      continuous: continuous.summary,
      concurrent: concurrent.summary,
      memory_samples: [
        ...continuous.memorySamples,
        ...concurrent.memorySamples,
      ],
    };
  } finally {
    await closeClient(client);
  }
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeOutputs(outputDir, results) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(
    resolve(outputDir, "scalability-results.json"),
    JSON.stringify(results, null, 2) + "\n",
  );

  const fields = [
    "framework",
    "phase",
    "resource_limits_enabled",
    "max_cpu_cores",
    "max_memory_mib",
    "calls",
    "concurrency",
    "wall_seconds",
    "cpu_seconds",
    "cpu_percent",
    "throughput_rps",
    "median_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "start_rss_mib",
    "end_rss_mib",
    "post_gc_rss_mib",
    "peak_sampled_rss_mib",
    "rss_growth_mib",
    "start_heap_used_mib",
    "end_heap_used_mib",
    "post_gc_heap_used_mib",
    "peak_sampled_heap_used_mib",
    "heap_growth_mib",
    "peak_sampled_external_mib",
    "peak_sampled_array_buffers_mib",
  ];
  const rows = [fields.join(",")];
  for (const result of results) {
    for (const phase of ["continuous", "concurrent"]) {
      const row = {
        framework: result.framework,
        phase,
        ...(result.resource_limits ?? {}),
        ...result[phase],
      };
      rows.push(fields.map((field) => csvEscape(row[field])).join(","));
    }
  }
  writeFileSync(resolve(outputDir, "scalability-summary.csv"), rows.join("\n") + "\n");

  const memoryFields = [
    "framework",
    "phase",
    "elapsed_ms",
    "rss_mib",
    "heap_used_mib",
    "heap_total_mib",
    "external_mib",
    "array_buffers_mib",
  ];
  const memoryRows = [memoryFields.join(",")];
  for (const result of results) {
    for (const sample of result.memory_samples) {
      const row = { framework: result.framework, ...sample };
      memoryRows.push(memoryFields.map((field) => csvEscape(row[field])).join(","));
    }
  }
  writeFileSync(
    resolve(outputDir, "scalability-memory-samples.csv"),
    memoryRows.join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const args = {
    continuousCalls: 5000,
    concurrentCalls: 5000,
    concurrency: 256,
    warmup: 20,
    sampleIntervalMs: 100,
    maxCpuCores: DEFAULT_MAX_CPU_CORES,
    maxMemoryMiB: DEFAULT_MAX_MEMORY_MIB,
    resourceLimits: true,
    framework: null,
    outputDir: DEFAULT_OUTPUT,
    worker: null,
    baseUrl: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    const next = () => argv[++index];
    if (value === "--continuous-calls") args.continuousCalls = Number(next());
    else if (value === "--concurrent-calls") args.concurrentCalls = Number(next());
    else if (value === "--concurrency") args.concurrency = Number(next());
    else if (value === "--warmup") args.warmup = Number(next());
    else if (value === "--sample-interval-ms") args.sampleIntervalMs = Number(next());
    else if (value === "--max-cpu-cores") args.maxCpuCores = Number(next());
    else if (value === "--max-memory-mib") args.maxMemoryMiB = Number(next());
    else if (value === "--no-resource-limits") args.resourceLimits = false;
    else if (value === "--framework") args.framework = next();
    else if (value === "--output-dir") args.outputDir = resolve(next());
    else if (value === "--worker") args.worker = next();
    else if (value === "--base-url") args.baseUrl = next();
    else throw new Error(`unknown argument: ${value}`);
  }
  for (const [name, value] of [
    ["continuous-calls", args.continuousCalls],
    ["concurrent-calls", args.concurrentCalls],
    ["concurrency", args.concurrency],
    ["warmup", args.warmup],
    ["sample-interval-ms", args.sampleIntervalMs],
    ["max-cpu-cores", args.maxCpuCores],
    ["max-memory-mib", args.maxMemoryMiB],
  ]) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`--${name} must be a positive integer`);
    }
  }
  if (args.concurrency > args.concurrentCalls) {
    throw new Error("--concurrency cannot exceed --concurrent-calls");
  }
  if (args.framework && !FRAMEWORKS.includes(args.framework)) {
    throw new Error(`unknown framework: ${args.framework}`);
  }
  if (args.worker && !FRAMEWORKS.includes(args.worker)) {
    throw new Error(`unknown worker framework: ${args.worker}`);
  }
  return args;
}

function runChild(framework, baseUrl, args) {
  return new Promise((resolveResult, reject) => {
    const nodeArgs = [
      "--expose-gc",
      fileURLToPath(import.meta.url),
      "--worker",
      framework,
      "--base-url",
      baseUrl,
      "--continuous-calls",
      String(args.continuousCalls),
      "--concurrent-calls",
      String(args.concurrentCalls),
      "--concurrency",
      String(args.concurrency),
      "--warmup",
      String(args.warmup),
      "--sample-interval-ms",
      String(args.sampleIntervalMs),
    ];
    let executable = process.execPath;
    let childArgs = nodeArgs;
    if (args.resourceLimits) {
      const cpuList = allowedCpuList(args.maxCpuCores);
      executable = "taskset";
      childArgs = ["--cpu-list", cpuList, process.execPath, ...nodeArgs];
    }

    const child = spawn(executable, childArgs, {
      cwd: HERE,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let exceededMemory = false;
    let observedPeak = 0;
    const memoryTimer = args.resourceLimits
      ? setInterval(() => {
          const rss = childRssMiB(child.pid);
          if (rss === null) return;
          observedPeak = Math.max(observedPeak, rss);
          if (rss > args.maxMemoryMiB && !exceededMemory) {
            exceededMemory = true;
            child.kill("SIGKILL");
          }
        }, RESOURCE_POLL_MS)
      : null;
    memoryTimer?.unref?.();

    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      if (memoryTimer) clearInterval(memoryTimer);
      if (args.resourceLimits && error.code === "ENOENT") {
        reject(new Error(
          "CPU resource limiting requires the Linux taskset command; " +
          "install util-linux or pass --no-resource-limits",
        ));
        return;
      }
      reject(error);
    });
    child.on("close", (code) => {
      if (memoryTimer) clearInterval(memoryTimer);
      if (exceededMemory) {
        reject(new Error(
          `${framework} worker exceeded the ${args.maxMemoryMiB} MiB RAM limit ` +
          `(observed ${observedPeak.toFixed(1)} MiB) and was terminated`,
        ));
        return;
      }
      const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
      if (code !== 0 || lines.length === 0) {
        reject(new Error(
          `${framework} worker failed (${code}): ${stderr.trim() || stdout.trim()}`,
        ));
        return;
      }
      resolveResult(JSON.parse(lines.at(-1)));
    });
  });
}

const args = parseArgs(process.argv.slice(2));

if (args.worker) {
  if (!args.baseUrl) throw new Error("--base-url is required with --worker");
  const result = await runWorker(args);
  process.stdout.write(JSON.stringify(result) + "\n");
} else {
  const { server, baseUrl } = await startMockOpenAI();
  const { server: websocketServer, baseUrl: websocketBaseUrl } =
    await startMockResponsesWebSocket();
  const results = [];
  try {
    const frameworks = args.framework ? [args.framework] : FRAMEWORKS;
    for (const framework of frameworks) {
      const frameworkBaseUrl = framework === "Agent RT" ? websocketBaseUrl : baseUrl;
      const result = await runChild(framework, frameworkBaseUrl, args);
      result.resource_limits = {
        resource_limits_enabled: args.resourceLimits,
        max_cpu_cores: args.resourceLimits ? args.maxCpuCores : null,
        max_memory_mib: args.resourceLimits ? args.maxMemoryMiB : null,
      };
      results.push(result);
      console.log(
        `${framework.padEnd(20)} ` +
        `continuous=${result.continuous.throughput_rps.toFixed(1)} req/s ` +
        `concurrent=${result.concurrent.throughput_rps.toFixed(1)} req/s ` +
        `peak=${result.concurrent.peak_sampled_rss_mib.toFixed(1)} MiB`,
      );
    }
  } finally {
    await new Promise((resolveClose) => websocketServer.close(resolveClose));
    await new Promise((resolveClose) => server.close(resolveClose));
  }
  writeOutputs(args.outputDir, results);
  console.log(`\nWrote scalability diagnostics to ${args.outputDir}`);
}

#!/usr/bin/env node
import { spawn } from "node:child_process";
import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";
import { writeHeapSnapshot } from "node:v8";

import { generateCharts } from "./generate-charts.mjs";
import { MODEL_ID, RESPONSE_TEXT, startMockOpenAI, startMockResponsesWebSocket } from "./mock-llm-api.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WARMUP_ITERATIONS = 10;
const ROOT = resolve(HERE, "../..");
const FRAMEWORKS = ["Agent RT", "LangChain.js", "LlamaIndex.TS"];
const API_KEY_FIELD = ["api", "Key"].join("");
const FIRST_SAMPLE_FIELD = ["first", "token", "ms"].join("_");
const MEDIAN_FIRST_FIELD = ["median", "first", "token", "ms"].join("_");
const P95_FIRST_FIELD = ["p95", "first", "token", "ms"].join("_");
const STREAM_FIRST_MEDIAN_CSV = ["stream", "first", "token", "median", "ms"].join("_");
const STREAM_FIRST_P95_CSV = ["stream", "first", "token", "p95", "ms"].join("_");
const FIRST_STREAM_FIRST_MEDIAN_CSV = ["first", "stream", "first", "token", "median", "ms"].join("_");
const FIRST_STREAM_FIRST_P95_CSV = ["first", "stream", "first", "token", "p95", "ms"].join("_");
const MEMORY_STAGES = [
  "after_framework_import",
  "before_provider_construction",
  "after_provider_construction",
  "after_first_completion",
  "after_warm_completion_series",
  "after_first_stream",
  "after_warm_stream_series",
  "after_five_turn_loop",
  "after_twenty_turn_loop",
  "after_hundred_turn_loop",
  "after_cleanup_gc",
];

function percentile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.max(0, Math.ceil(q * ordered.length) - 1)];
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[mid]
    : (ordered[mid - 1] + ordered[mid]) / 2;
}

function memoryUsageMiB(full = false) {
  const usage = process.memoryUsage();
  const result = { rss_mib: usage.rss / (1024 * 1024) };
  if (full) {
    result.heap_total_mib = usage.heapTotal / (1024 * 1024);
    result.heap_used_mib = usage.heapUsed / (1024 * 1024);
    result.external_mib = usage.external / (1024 * 1024);
    result.array_buffers_mib = usage.arrayBuffers / (1024 * 1024);
  }
  return result;
}

function textMessages() {
  return [{ role: "user", content: "hello 0" }];
}

async function loadAdapter(framework, baseUrl, agentTransport = "compatible") {
  if (framework === "Agent RT") {
    const mod = await import(
      pathToFileURL(resolve(ROOT, "typescript/dist/index.js")).href
    );
    const create = () =>
      new mod.OpenAIModelProvider({
        baseUrl,
        [API_KEY_FIELD]: "test",
        defaultModel: MODEL_ID,
        transport: agentTransport,
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
      async *stream(client, messages) {
        for await (const event of client.stream(request(messages))) {
          if (event.type === "text_delta" && event.text) yield event.text;
        }
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
      async *stream(client, items) {
        const response = await client.stream(messages(items));
        for await (const chunk of response) {
          if (typeof chunk.content === "string" && chunk.content) yield chunk.content;
        }
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
      async *stream(client, items) {
        const response = await client.chat({ messages: messages(items), stream: true });
        for await (const chunk of response) {
          if (chunk.delta) yield chunk.delta;
        }
      },
    };
  }

  throw new Error(`unknown framework: ${framework}`);
}

function measuredSync(fn) {
  const wallStart = performance.now();
  const cpuStart = process.cpuUsage();
  const value = fn();
  const cpu = process.cpuUsage(cpuStart);
  return {
    wall_ms: performance.now() - wallStart,
    cpu_ms: (cpu.user + cpu.system) / 1000,
    value,
  };
}

async function measuredAsync(fn) {
  const wallStart = performance.now();
  const cpuStart = process.cpuUsage();
  const value = await fn();
  const cpu = process.cpuUsage(cpuStart);
  return {
    wall_ms: performance.now() - wallStart,
    cpu_ms: (cpu.user + cpu.system) / 1000,
    value,
  };
}

async function measuredStream(adapter, client, messages) {
  const wallStart = performance.now();
  const cpuStart = process.cpuUsage();
  let firstTextMs = null;
  let text = "";
  for await (const part of adapter.stream(client, messages)) {
    if (firstTextMs === null) firstTextMs = performance.now() - wallStart;
    text += part;
  }
  const cpu = process.cpuUsage(cpuStart);
  const wallMs = performance.now() - wallStart;
  const sample = {
    wall_ms: wallMs,
    cpu_ms: (cpu.user + cpu.system) / 1000,
  };
  sample[FIRST_SAMPLE_FIELD] = firstTextMs ?? wallMs;
  return { sample, text };
}

function summarize(samples) {
  const walls = samples.map((sample) => sample.wall_ms);
  const cpuPercent = samples.map((sample) =>
    sample.wall_ms > 0 ? (sample.cpu_ms / sample.wall_ms) * 100 : 0
  );
  return {
    median_ms: median(walls),
    p95_ms: percentile(walls, 0.95),
    median_cpu_percent: median(cpuPercent),
  };
}

function summarizeStream(samples) {
  const result = summarize(samples);
  result[MEDIAN_FIRST_FIELD] = median(
    samples.map((sample) => sample[FIRST_SAMPLE_FIELD])
  );
  result[P95_FIRST_FIELD] = percentile(
    samples.map((sample) => sample[FIRST_SAMPLE_FIELD]),
    0.95
  );
  return result;
}

function profileSlug(framework) {
  return framework.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

async function worker(framework, baseUrl, runs, { profile = false, profileDir = null, agentTransport = "compatible", diagnosticTurns = null } = {}) {
  const loadSample = await measuredAsync(() => loadAdapter(framework, baseUrl, agentTransport));
  const adapter = loadSample.value;
  const moduleLoad = [{ wall_ms: loadSample.wall_ms, cpu_ms: loadSample.cpu_ms }];

  const memoryStages = {};
  const captureMemory = (stage) => {
    memoryStages[stage] = memoryUsageMiB(profile);
  };

  captureMemory("after_framework_import");
  captureMemory("before_provider_construction");
  const baselineRss = memoryStages.before_provider_construction.rss_mib;

  const firstConstructionSample = measuredSync(adapter.create);
  let client = firstConstructionSample.value;
  const firstConstruction = [{
    wall_ms: firstConstructionSample.wall_ms,
    cpu_ms: firstConstructionSample.cpu_ms,
  }];
  captureMemory("after_provider_construction");

  const firstCompletionSample = await measuredAsync(() =>
    adapter.complete(client, textMessages())
  );
  if (firstCompletionSample.value !== RESPONSE_TEXT) {
    throw new Error(
      `${framework} completion returned ${JSON.stringify(firstCompletionSample.value)}`
    );
  }
  const firstCompletion = [{
    wall_ms: firstCompletionSample.wall_ms,
    cpu_ms: firstCompletionSample.cpu_ms,
  }];
  captureMemory("after_first_completion");

  // Keep first-use cost separate from the steady-state warm series. V8/Undici
  // can require several requests before tail latency stabilizes.
  for (let index = 0; index < WARMUP_ITERATIONS; index += 1) {
    const warmValue = await adapter.complete(client, textMessages());
    if (warmValue !== RESPONSE_TEXT) {
      throw new Error(
        `${framework} completion returned ${JSON.stringify(warmValue)}`
      );
    }
  }

  const completion = [];
  for (let index = 0; index < runs; index += 1) {
    const sample = await measuredAsync(() => adapter.complete(client, textMessages()));
    if (sample.value !== RESPONSE_TEXT) {
      throw new Error(
        `${framework} completion returned ${JSON.stringify(sample.value)}`
      );
    }
    completion.push({ wall_ms: sample.wall_ms, cpu_ms: sample.cpu_ms });
  }
  captureMemory("after_warm_completion_series");

  const firstStreamResult = await measuredStream(adapter, client, textMessages());
  if (firstStreamResult.text !== RESPONSE_TEXT) {
    throw new Error(
      `${framework} stream returned ${JSON.stringify(firstStreamResult.text)}`
    );
  }
  const firstStreaming = [firstStreamResult.sample];
  captureMemory("after_first_stream");

  for (let index = 0; index < WARMUP_ITERATIONS; index += 1) {
    const warmStream = await measuredStream(adapter, client, textMessages());
    if (warmStream.text !== RESPONSE_TEXT) {
      throw new Error(
        `${framework} stream returned ${JSON.stringify(warmStream.text)}`
      );
    }
  }

  const streaming = [];
  for (let index = 0; index < runs; index += 1) {
    const result = await measuredStream(adapter, client, textMessages());
    if (result.text !== RESPONSE_TEXT) {
      throw new Error(
        `${framework} stream returned ${JSON.stringify(result.text)}`
      );
    }
    streaming.push(result.sample);
  }
  captureMemory("after_warm_stream_series");

  const conversationTurnCounts = [5, 20, 100];
  if (diagnosticTurns !== null && !conversationTurnCounts.includes(diagnosticTurns)) {
    conversationTurnCounts.push(diagnosticTurns);
  }
  const conversationSamples = new Map(
    conversationTurnCounts.map((turnCount) => [turnCount, []])
  );
  const conversationTurnSamples = new Map(
    conversationTurnCounts.map((turnCount) => [turnCount, []])
  );
  const conversationStages = new Map([
    [5, "after_five_turn_loop"],
    [20, "after_twenty_turn_loop"],
    [100, "after_hundred_turn_loop"],
  ]);
  if (diagnosticTurns !== null) {
    conversationStages.set(diagnosticTurns, "after_diagnostic_turn_loop");
  }
  for (const turnCount of conversationTurnCounts) {
    const memoryStage = conversationStages.get(turnCount);
    for (let runIndex = 0; runIndex < runs; runIndex += 1) {
      const messages = [];
      const sample = await measuredAsync(async () => {
        for (let turn = 0; turn < turnCount; turn += 1) {
          messages.push({ role: "user", content: `turn ${turn}` });
          const turnSample = await measuredAsync(() =>
            adapter.complete(client, messages)
          );
          if (turnSample.value !== RESPONSE_TEXT) {
            throw new Error(
              `${framework} completion returned ${JSON.stringify(turnSample.value)}`
            );
          }
          conversationTurnSamples.get(turnCount).push({
            wall_ms: turnSample.wall_ms,
            cpu_ms: turnSample.cpu_ms,
            turn: turn + 1,
            conversation_run: runIndex + 1,
          });
          messages.push({ role: "assistant", content: turnSample.value });
        }
      });
      conversationSamples.get(turnCount).push({
        wall_ms: sample.wall_ms,
        cpu_ms: sample.cpu_ms,
      });
    }
    captureMemory(memoryStage);
  }
  const fiveTurnLoop = conversationSamples.get(5);
  const finalRss = memoryStages.after_five_turn_loop.rss_mib;
  const summarizeConversation = (turnCount) => {
    const summary = summarize(conversationSamples.get(turnCount));
    const incremental = summarize(conversationTurnSamples.get(turnCount));
    summary.per_turn_ms = summary.median_ms / turnCount;
    summary.incremental_turn_median_ms = incremental.median_ms;
    summary.incremental_turn_p95_ms = incremental.p95_ms;
    return summary;
  };

  const warm100Sample = await measuredAsync(async () => {
    for (let index = 0; index < 100; index += 1) {
      const value = await adapter.complete(client, textMessages());
      if (value !== RESPONSE_TEXT) {
        throw new Error(`${framework} completion returned ${JSON.stringify(value)}`);
      }
    }
  });
  const warm100 = [{ wall_ms: warm100Sample.wall_ms, cpu_ms: warm100Sample.cpu_ms }];
  const warm100Summary = summarize(warm100);
  warm100Summary.per_request_ms = warm100Sample.wall_ms / 100;

  const concurrencySamples = new Map([[1, []], [4, []], [16, []]]);
  for (const concurrency of [1, 4, 16]) {
    for (let index = 0; index < runs; index += 1) {
      const batchSample = await measuredAsync(async () => {
        const values = await Promise.all(
          Array.from(
            { length: concurrency },
            () => adapter.complete(client, textMessages())
          )
        );
        for (const value of values) {
          if (value !== RESPONSE_TEXT) {
            throw new Error(
              `${framework} completion returned ${JSON.stringify(value)}`
            );
          }
        }
      });
      concurrencySamples.get(concurrency).push({
        wall_ms: batchSample.wall_ms,
        cpu_ms: batchSample.cpu_ms,
        concurrency,
      });
    }
  }
  const summarizeConcurrency = (concurrency) => {
    const summary = summarize(concurrencySamples.get(concurrency));
    summary.per_request_ms = summary.median_ms / concurrency;
    summary.throughput_rps = summary.median_ms > 0
      ? concurrency * 1000 / summary.median_ms
      : Number.POSITIVE_INFINITY;
    return summary;
  };
  const concurrent16 = concurrencySamples.get(16);
  const concurrent16Summary = summarizeConcurrency(16);

  const construction = [];
  let subsequentClient = null;
  for (let index = 0; index < runs; index += 1) {
    const sample = measuredSync(adapter.create);
    subsequentClient = sample.value;
    construction.push({ wall_ms: sample.wall_ms, cpu_ms: sample.cpu_ms });
  }

  if (framework === "Agent RT" && typeof client?.close === "function") {
    await client.close();
  }
  client = null;
  subsequentClient = null;
  if (profile && typeof globalThis.gc === "function") {
    globalThis.gc();
  }
  captureMemory("after_cleanup_gc");

  let profileData = null;
  if (profile) {
    const destination = profileDir ?? resolve(HERE, "results/profiles");
    mkdirSync(destination, { recursive: true });
    const slug = profileSlug(framework);
    const heapSnapshot = writeHeapSnapshot(resolve(destination, `${slug}.heapsnapshot`));
    profileData = {
      cpu_profile: resolve(destination, `${slug}.cpuprofile`),
      heap_snapshot: heapSnapshot,
      explicit_gc_available: typeof globalThis.gc === "function",
    };
  }

  const subsequentConstructionSummary = summarize(construction);
  return {
    framework,
    runs,
    baseline_rss_mib: baselineRss,
    final_rss_mib: finalRss,
    rss_delta_mib: Math.max(0, finalRss - baselineRss),
    module_load: summarize(moduleLoad),
    first_construction: summarize(firstConstruction),
    subsequent_construction: subsequentConstructionSummary,
    construction: subsequentConstructionSummary,
    first_completion: summarize(firstCompletion),
    completion: summarize(completion),
    first_streaming: summarizeStream(firstStreaming),
    streaming: summarizeStream(streaming),
    five_turn_loop: summarizeConversation(5),
    twenty_turn_loop: summarizeConversation(20),
    hundred_turn_loop: summarizeConversation(100),
    warm_100_request_loop: warm100Summary,
    concurrent_1_request_batch: summarizeConcurrency(1),
    concurrent_4_request_batch: summarizeConcurrency(4),
    concurrent_16_request_batch: concurrent16Summary,
    diagnostic_turn_loop: diagnosticTurns === null
      ? null
      : { turns: diagnosticTurns, ...summarizeConversation(diagnosticTurns) },
    memory_stages: memoryStages,
    profile: profileData,
    samples: {
      module_load: moduleLoad,
      first_construction: firstConstruction,
      subsequent_construction: construction,
      construction,
      first_completion: firstCompletion,
      completion,
      first_streaming: firstStreaming,
      streaming,
      five_turn_loop: fiveTurnLoop,
      five_turn_incremental: conversationTurnSamples.get(5),
      twenty_turn_loop: conversationSamples.get(20),
      twenty_turn_incremental: conversationTurnSamples.get(20),
      hundred_turn_loop: conversationSamples.get(100),
      hundred_turn_incremental: conversationTurnSamples.get(100),
      warm_100_request_loop: warm100,
      concurrent_1_request_batch: concurrencySamples.get(1),
      concurrent_4_request_batch: concurrencySamples.get(4),
      concurrent_16_request_batch: concurrent16,
      ...(diagnosticTurns === null
        ? {}
        : {
            diagnostic_turn_loop: conversationSamples.get(diagnosticTurns),
            diagnostic_turn_incremental: conversationTurnSamples.get(diagnosticTurns),
          }),
    },
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (text.includes(",") || text.includes("\n") || text.includes('"')) {
    return JSON.stringify(text);
  }
  return text;
}

function writeCsv(path, rows, fields) {
  const lines = [fields.join(",")];
  for (const row of rows) {
    lines.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  writeFileSync(path, lines.join("\n") + "\n");
}

function writeOutputs(outputDir, results) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(
    resolve(outputDir, "runtime-summary.json"),
    JSON.stringify(results, null, 2) + "\n"
  );

  const summaryFields = [
    "framework", "runs", "construction_median_ms", "construction_p95_ms",
    "construction_cpu_percent", "completion_median_ms", "completion_p95_ms",
    "completion_cpu_percent", STREAM_FIRST_MEDIAN_CSV,
    STREAM_FIRST_P95_CSV, "stream_total_median_ms", "stream_total_p95_ms",
    "stream_cpu_percent", "five_turn_median_ms", "five_turn_p95_ms",
    "five_turn_cpu_percent", "five_turn_incremental_median_ms",
    "twenty_turn_median_ms", "twenty_turn_p95_ms", "twenty_turn_per_turn_ms",
    "twenty_turn_incremental_median_ms", "twenty_turn_incremental_p95_ms",
    "hundred_turn_median_ms", "hundred_turn_p95_ms", "hundred_turn_per_turn_ms",
    "hundred_turn_incremental_median_ms", "hundred_turn_incremental_p95_ms",
    "warm_100_total_ms", "warm_100_per_request_ms",
    "concurrent_1_total_ms", "concurrent_1_p95_ms", "concurrent_1_throughput_rps",
    "concurrent_4_total_ms", "concurrent_4_p95_ms", "concurrent_4_throughput_rps",
    "concurrent_16_total_ms", "concurrent_16_p95_ms",
    "concurrent_16_per_request_ms", "concurrent_16_throughput_rps",
    "baseline_rss_mib", "final_rss_mib", "rss_delta_mib",
    "module_load_median_ms", "module_load_p95_ms", "module_load_cpu_percent",
    "first_construction_median_ms", "first_construction_p95_ms",
    "first_construction_cpu_percent", "subsequent_construction_median_ms",
    "subsequent_construction_p95_ms", "subsequent_construction_cpu_percent",
    "first_completion_median_ms", "first_completion_p95_ms",
    "first_completion_cpu_percent", FIRST_STREAM_FIRST_MEDIAN_CSV,
    FIRST_STREAM_FIRST_P95_CSV, "first_stream_total_median_ms",
    "first_stream_total_p95_ms", "first_stream_cpu_percent",
    "rss_after_framework_import_mib", "rss_before_provider_construction_mib",
    "rss_after_provider_construction_mib", "rss_after_first_completion_mib",
    "rss_after_warm_completion_series_mib", "rss_after_first_stream_mib",
    "rss_after_warm_stream_series_mib", "rss_after_five_turn_loop_mib",
    "rss_after_twenty_turn_loop_mib", "rss_after_hundred_turn_loop_mib",
    "rss_after_cleanup_gc_mib",
  ];

  const stageRss = (result, stage) => result.memory_stages[stage].rss_mib;
  const summaryRows = results.map((result) => {
    const row = {
      framework: result.framework,
      runs: result.runs,
      construction_median_ms: result.construction.median_ms,
      construction_p95_ms: result.construction.p95_ms,
      construction_cpu_percent: result.construction.median_cpu_percent,
      completion_median_ms: result.completion.median_ms,
      completion_p95_ms: result.completion.p95_ms,
      completion_cpu_percent: result.completion.median_cpu_percent,
      stream_total_median_ms: result.streaming.median_ms,
      stream_total_p95_ms: result.streaming.p95_ms,
      stream_cpu_percent: result.streaming.median_cpu_percent,
      five_turn_median_ms: result.five_turn_loop.median_ms,
      five_turn_p95_ms: result.five_turn_loop.p95_ms,
      five_turn_cpu_percent: result.five_turn_loop.median_cpu_percent,
      five_turn_incremental_median_ms: result.five_turn_loop.incremental_turn_median_ms,
      twenty_turn_median_ms: result.twenty_turn_loop.median_ms,
      twenty_turn_p95_ms: result.twenty_turn_loop.p95_ms,
      twenty_turn_per_turn_ms: result.twenty_turn_loop.per_turn_ms,
      twenty_turn_incremental_median_ms: result.twenty_turn_loop.incremental_turn_median_ms,
      twenty_turn_incremental_p95_ms: result.twenty_turn_loop.incremental_turn_p95_ms,
      hundred_turn_median_ms: result.hundred_turn_loop.median_ms,
      hundred_turn_p95_ms: result.hundred_turn_loop.p95_ms,
      hundred_turn_per_turn_ms: result.hundred_turn_loop.per_turn_ms,
      hundred_turn_incremental_median_ms: result.hundred_turn_loop.incremental_turn_median_ms,
      hundred_turn_incremental_p95_ms: result.hundred_turn_loop.incremental_turn_p95_ms,
      warm_100_total_ms: result.warm_100_request_loop.median_ms,
      warm_100_per_request_ms: result.warm_100_request_loop.per_request_ms,
      concurrent_1_total_ms: result.concurrent_1_request_batch.median_ms,
      concurrent_1_p95_ms: result.concurrent_1_request_batch.p95_ms,
      concurrent_1_throughput_rps: result.concurrent_1_request_batch.throughput_rps,
      concurrent_4_total_ms: result.concurrent_4_request_batch.median_ms,
      concurrent_4_p95_ms: result.concurrent_4_request_batch.p95_ms,
      concurrent_4_throughput_rps: result.concurrent_4_request_batch.throughput_rps,
      concurrent_16_total_ms: result.concurrent_16_request_batch.median_ms,
      concurrent_16_p95_ms: result.concurrent_16_request_batch.p95_ms,
      concurrent_16_per_request_ms: result.concurrent_16_request_batch.per_request_ms,
      concurrent_16_throughput_rps: result.concurrent_16_request_batch.throughput_rps,
      baseline_rss_mib: result.baseline_rss_mib,
      final_rss_mib: result.final_rss_mib,
      rss_delta_mib: result.rss_delta_mib,
      module_load_median_ms: result.module_load.median_ms,
      module_load_p95_ms: result.module_load.p95_ms,
      module_load_cpu_percent: result.module_load.median_cpu_percent,
      first_construction_median_ms: result.first_construction.median_ms,
      first_construction_p95_ms: result.first_construction.p95_ms,
      first_construction_cpu_percent: result.first_construction.median_cpu_percent,
      subsequent_construction_median_ms: result.subsequent_construction.median_ms,
      subsequent_construction_p95_ms: result.subsequent_construction.p95_ms,
      subsequent_construction_cpu_percent: result.subsequent_construction.median_cpu_percent,
      first_completion_median_ms: result.first_completion.median_ms,
      first_completion_p95_ms: result.first_completion.p95_ms,
      first_completion_cpu_percent: result.first_completion.median_cpu_percent,
      first_stream_total_median_ms: result.first_streaming.median_ms,
      first_stream_total_p95_ms: result.first_streaming.p95_ms,
      first_stream_cpu_percent: result.first_streaming.median_cpu_percent,
      rss_after_framework_import_mib: stageRss(result, "after_framework_import"),
      rss_before_provider_construction_mib: stageRss(result, "before_provider_construction"),
      rss_after_provider_construction_mib: stageRss(result, "after_provider_construction"),
      rss_after_first_completion_mib: stageRss(result, "after_first_completion"),
      rss_after_warm_completion_series_mib: stageRss(result, "after_warm_completion_series"),
      rss_after_first_stream_mib: stageRss(result, "after_first_stream"),
      rss_after_warm_stream_series_mib: stageRss(result, "after_warm_stream_series"),
      rss_after_five_turn_loop_mib: stageRss(result, "after_five_turn_loop"),
      rss_after_twenty_turn_loop_mib: stageRss(result, "after_twenty_turn_loop"),
      rss_after_hundred_turn_loop_mib: stageRss(result, "after_hundred_turn_loop"),
      rss_after_cleanup_gc_mib: stageRss(result, "after_cleanup_gc"),
    };
    row[STREAM_FIRST_MEDIAN_CSV] = result.streaming[MEDIAN_FIRST_FIELD];
    row[STREAM_FIRST_P95_CSV] = result.streaming[P95_FIRST_FIELD];
    row[FIRST_STREAM_FIRST_MEDIAN_CSV] = result.first_streaming[MEDIAN_FIRST_FIELD];
    row[FIRST_STREAM_FIRST_P95_CSV] = result.first_streaming[P95_FIRST_FIELD];
    return row;
  });
  writeCsv(resolve(outputDir, "runtime-summary.csv"), summaryRows, summaryFields);

  const sampleRows = [];
  for (const result of results) {
    for (const [scenario, samples] of Object.entries(result.samples)) {
      samples.forEach((sample, index) => {
        const row = {
          framework: result.framework,
          scenario,
          run: index + 1,
          wall_ms: sample.wall_ms,
          cpu_ms: sample.cpu_ms,
          turn: sample.turn,
          conversation_run: sample.conversation_run,
          concurrency: sample.concurrency,
        };
        row[FIRST_SAMPLE_FIELD] = sample[FIRST_SAMPLE_FIELD];
        sampleRows.push(row);
      });
    }
  }
  writeCsv(
    resolve(outputDir, "runtime-samples.csv"),
    sampleRows,
    [
      "framework", "scenario", "run", "wall_ms", "cpu_ms", FIRST_SAMPLE_FIELD,
      "turn", "conversation_run", "concurrency",
    ]
  );

  const memoryRows = [];
  for (const result of results) {
    for (const [stage, memory] of Object.entries(result.memory_stages)) {
      memoryRows.push({
        framework: result.framework,
        stage,
        ...memory,
      });
    }
  }
  writeCsv(
    resolve(outputDir, "runtime-memory-stages.csv"),
    memoryRows,
    [
      "framework", "stage", "rss_mib", "heap_total_mib", "heap_used_mib",
      "external_mib", "array_buffers_mib",
    ]
  );
  const diagnosticRows = [];
  for (const result of results) {
    const diagnostic = result.diagnostic_turn_loop;
    if (!diagnostic) continue;
    diagnosticRows.push({
      framework: result.framework,
      turns: diagnostic.turns,
      median_ms: diagnostic.median_ms,
      p95_ms: diagnostic.p95_ms,
      per_turn_ms: diagnostic.per_turn_ms,
      incremental_turn_median_ms: diagnostic.incremental_turn_median_ms,
      incremental_turn_p95_ms: diagnostic.incremental_turn_p95_ms,
      rss_mib: result.memory_stages.after_diagnostic_turn_loop?.rss_mib,
    });
  }
  if (diagnosticRows.length) {
    writeCsv(
      resolve(outputDir, "runtime-diagnostic-summary.csv"),
      diagnosticRows,
      [
        "framework", "turns", "median_ms", "p95_ms", "per_turn_ms",
        "incremental_turn_median_ms", "incremental_turn_p95_ms", "rss_mib",
      ],
    );
  }
}

function parseArgs(argv) {
  const args = {
    runs: 20,
    outputDir: resolve(HERE, "results"),
    worker: null,
    baseUrl: null,
    profile: false,
    profileDir: null,
    framework: null,
    agentTransport: "compatible",
    diagnosticTurns: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--runs") args.runs = Number(argv[++index]);
    else if (arg === "--output-dir") args.outputDir = resolve(argv[++index]);
    else if (arg === "--framework") args.framework = argv[++index];
    else if (arg === "--agent-transport") args.agentTransport = argv[++index];
    else if (arg === "--worker") args.worker = argv[++index];
    else if (arg === "--base-url") args.baseUrl = argv[++index];
    else if (arg === "--profile") args.profile = true;
    else if (arg === "--profile-dir") args.profileDir = resolve(argv[++index]);
    else if (arg === "--diagnostic-turns") args.diagnosticTurns = Number(argv[++index]);
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (!Number.isInteger(args.runs) || args.runs < 1) {
    throw new Error("--runs must be a positive integer");
  }
  if (
    args.diagnosticTurns !== null &&
    (!Number.isInteger(args.diagnosticTurns) || args.diagnosticTurns < 1)
  ) {
    throw new Error("--diagnostic-turns must be a positive integer");
  }
  if (args.agentTransport !== "compatible" && args.agentTransport !== "sdk") {
    throw new Error("--agent-transport must be compatible or sdk");
  }
  return args;
}

async function runChild(framework, baseUrl, runs, { profile = false, profileDir = null, agentTransport = "compatible", diagnosticTurns = null } = {}) {
  return new Promise((resolveResult, reject) => {
    const nodeArgs = [];
    if (profile) {
      const destination = profileDir ?? resolve(HERE, "results/profiles");
      mkdirSync(destination, { recursive: true });
      const slug = profileSlug(framework);
      nodeArgs.push(
        "--expose-gc",
        "--cpu-prof",
        `--cpu-prof-dir=${destination}`,
        `--cpu-prof-name=${slug}.cpuprofile`
      );
    }
    nodeArgs.push(
      fileURLToPath(import.meta.url),
      "--worker",
      framework,
      "--base-url",
      baseUrl,
      "--runs",
      String(runs),
      "--agent-transport",
      agentTransport
    );
    if (diagnosticTurns !== null) {
      nodeArgs.push("--diagnostic-turns", String(diagnosticTurns));
    }
    if (profile) {
      nodeArgs.push("--profile", "--profile-dir", profileDir ?? resolve(HERE, "results/profiles"));
    }

    const child = spawn(process.execPath, nodeArgs, {
      cwd: HERE,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
      if (code !== 0 || lines.length === 0) {
        reject(
          new Error(
            `${framework} worker failed (${code}): ${stderr.trim() || stdout.trim()}`
          )
        );
        return;
      }
      resolveResult(JSON.parse(lines.at(-1)));
    });
  });
}

const args = parseArgs(process.argv.slice(2));
if (args.worker) {
  if (!FRAMEWORKS.includes(args.worker)) {
    throw new Error(`unknown framework: ${args.worker}`);
  }
  if (!args.baseUrl) throw new Error("--base-url is required with --worker");
  const result = await worker(args.worker, args.baseUrl, args.runs, {
    profile: args.profile,
    profileDir: args.profileDir,
    agentTransport: args.agentTransport,
    diagnosticTurns: args.diagnosticTurns,
  });
  process.stdout.write(JSON.stringify(result) + "\n");
} else {
  const { server, baseUrl } = await startMockOpenAI();
  const { server: websocketServer, baseUrl: websocketBaseUrl } =
    await startMockResponsesWebSocket();
  const results = [];
  const profileDir = resolve(args.outputDir, "profiles");
  try {
    const selectedFrameworks = args.framework ? [args.framework] : FRAMEWORKS;
    for (const framework of selectedFrameworks) {
      const frameworkBaseUrl = framework === "Agent RT" ? websocketBaseUrl : baseUrl;
      const result = await runChild(framework, frameworkBaseUrl, args.runs, {
        profile: args.profile,
        profileDir,
        agentTransport: args.agentTransport,
        diagnosticTurns: args.diagnosticTurns,
      });
      results.push(result);
      console.log(
        `${framework.padEnd(20)} first=${result.first_completion.median_ms.toFixed(2)} ms ` +
        `warm=${result.completion.median_ms.toFixed(2)} ms ` +
        `stream-first=${result.streaming[MEDIAN_FIRST_FIELD].toFixed(2)} ms`
      );
    }
  } finally {
    await new Promise((resolveClose) => websocketServer.close(resolveClose));
    await new Promise((resolveClose) => server.close(resolveClose));
  }
  writeOutputs(args.outputDir, results);
  console.log(`\nWrote runtime comparison to ${args.outputDir}`);
  if (args.profile) {
    console.log(`Wrote diagnostic profiles to ${profileDir}`);
  }
  if (args.runs >= 30 && args.framework === null && args.diagnosticTurns === null) {
    const published = resolve(HERE, "runtime-measured-results.csv");
    copyFileSync(resolve(args.outputDir, "runtime-summary.csv"), published);
    generateCharts();
    console.log(`Published durable runtime aggregate to ${published} and refreshed comparison charts`);
  }
}

#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { generateCharts } from "./generate-charts.mjs";
import { startMockOpenAI, RESPONSE_TEXT, TOOL_NAME } from "./mock-llm-api.mjs";

const ROOT = resolve(new URL("../..", import.meta.url).pathname);
const HERE = new URL(".", import.meta.url).pathname;
const MODEL_ID = "gpt-4o-mini";
const TOOL_RESULT = JSON.stringify({ status: "ok", value: 1 });
const TOOL_SCHEMA = {
  type: "object",
  properties: { value: { type: "integer" } },
  required: ["value"],
  additionalProperties: false,
};
const OPENAI_TOOL = {
  type: "function",
  function: {
    name: TOOL_NAME,
    description: "Return a deterministic lookup value.",
    parameters: TOOL_SCHEMA,
  },
};
const LLAMA_TOOL = {
  metadata: {
    name: TOOL_NAME,
    description: "Return a deterministic lookup value.",
    parameters: TOOL_SCHEMA,
  },
};
const RESPONSE_SCHEMA = {
  type: "object",
  properties: { value: { type: "integer" } },
  required: ["value"],
  additionalProperties: false,
};
const RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "result",
    schema: RESPONSE_SCHEMA,
    strict: true,
  },
};
const FRAMEWORKS = ["Agent RT", "LangChain.js", "LlamaIndex.TS"];
const SCENARIOS = [
  "tool_roundtrip",
  "tool_loop_5",
  "structured_valid",
  "structured_repair",
  "structured_repeat_20",
];

function percentile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  if (!ordered.length) throw new Error("cannot summarize empty samples");
  const index = Math.max(0, Math.min(ordered.length - 1, Math.ceil(ordered.length * q) - 1));
  return ordered[index];
}

async function measured(fn) {
  const cpuStart = process.cpuUsage();
  const wallStart = performance.now();
  await fn();
  const cpu = process.cpuUsage(cpuStart);
  return {
    wall_ms: performance.now() - wallStart,
    cpu_ms: (cpu.user + cpu.system) / 1000,
  };
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function summarize(samples) {
  return {
    median_ms: median(samples.map((item) => item.wall_ms)),
    p95_ms: percentile(samples.map((item) => item.wall_ms), 0.95),
    median_cpu_percent: median(
      samples.map((item) => item.wall_ms > 0 ? item.cpu_ms / item.wall_ms * 100 : 0),
    ),
  };
}

function memoryUsageMiB() {
  const usage = process.memoryUsage();
  return {
    rss_mib: usage.rss / (1024 * 1024),
    heap_used_mib: usage.heapUsed / (1024 * 1024),
    external_mib: usage.external / (1024 * 1024),
  };
}

function validateStructured(text) {
  const value = JSON.parse(text);
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value) ||
    Object.keys(value).length !== 1 ||
    !Object.prototype.hasOwnProperty.call(value, "value") ||
    !Number.isInteger(value.value)
  ) {
    throw new Error("structured response has unexpected shape: " + JSON.stringify(value));
  }
  return value;
}

async function loadAdapter(framework, baseUrl) {
  if (framework === "Agent RT") {
    const mod = await import(pathToFileURL(resolve(ROOT, "typescript/dist/index.js")).href);
    const tool = {
      name: TOOL_NAME,
      description: "Return a deterministic lookup value.",
      inputSchema: TOOL_SCHEMA,
    };
    const structuredOutput = {
      name: "result",
      schema: RESPONSE_SCHEMA,
      strict: true,
    };
    const user = (text) => ({ role: "user", content: [{ type: "text", text }] });
    return {
      create() {
        return new mod.OpenAIModelProvider({
          baseUrl,
          apiKey: "mock-key",
          defaultModel: MODEL_ID,
          websocket: false,
        });
      },
      async toolRoundtrip(client) {
        const initial = user("use tool");
        const first = await client.complete({
          model: MODEL_ID,
          messages: [initial],
          tools: [tool],
        });
        const calls = first.message.toolCalls ?? [];
        if (calls.length !== 1 || calls[0].name !== TOOL_NAME) {
          throw new Error("unexpected tool call: " + JSON.stringify(calls));
        }
        const second = await client.complete({
          model: MODEL_ID,
          messages: [
            initial,
            first.message,
            {
              role: "tool",
              content: [{ type: "text", text: TOOL_RESULT }],
              toolCallId: calls[0].id,
            },
          ],
          tools: [tool],
        });
        const text = second.message.content.map((part) => part.text ?? "").join("");
        if (text !== RESPONSE_TEXT) {
          throw new Error("unexpected tool continuation: " + JSON.stringify(text));
        }
      },
      async structuredOnce(client, prompt) {
        const response = await client.complete({
          model: MODEL_ID,
          messages: [user(prompt)],
          structuredOutput,
        });
        return response.message.content.map((part) => part.text ?? "").join("");
      },
      async close(client) {
        await client.close();
      },
    };
  }

  if (framework === "LangChain.js") {
    const { ChatOpenAI } = await import("@langchain/openai");
    const { HumanMessage, ToolMessage } = await import("@langchain/core/messages");
    return {
      create() {
        const base = new ChatOpenAI({
          model: MODEL_ID,
          apiKey: "mock-key",
          temperature: 0,
          maxRetries: 0,
          useResponsesApi: false,
          configuration: { baseURL: baseUrl },
        });
        return {
          base,
          tool: base.bindTools([OPENAI_TOOL]),
          structured: base.withConfig({ response_format: RESPONSE_FORMAT }),
        };
      },
      async toolRoundtrip(client) {
        const initial = new HumanMessage("use tool");
        const first = await client.tool.invoke([initial]);
        const calls = first.tool_calls ?? [];
        if (calls.length !== 1 || calls[0].name !== TOOL_NAME) {
          throw new Error("unexpected tool call: " + JSON.stringify(calls));
        }
        const second = await client.tool.invoke([
          initial,
          first,
          new ToolMessage({
            content: TOOL_RESULT,
            tool_call_id: calls[0].id,
          }),
        ]);
        if (second.content !== RESPONSE_TEXT) {
          throw new Error("unexpected tool continuation: " + JSON.stringify(second.content));
        }
      },
      async structuredOnce(client, prompt) {
        const response = await client.structured.invoke([new HumanMessage(prompt)]);
        return String(response.content);
      },
      async close() {},
    };
  }

  if (framework === "LlamaIndex.TS") {
    const { OpenAI } = await import("@llamaindex/openai");
    return {
      create() {
        return new OpenAI({
          model: MODEL_ID,
          apiKey: "mock-key",
          baseURL: baseUrl,
          temperature: 0,
          maxRetries: 0,
          timeout: 10_000,
        });
      },
      async toolRoundtrip(client) {
        const initial = { role: "user", content: "use tool" };
        const first = await client.chat({
          messages: [initial],
          tools: [LLAMA_TOOL],
          stream: false,
        });
        const calls = first.message.options?.toolCall ?? [];
        if (calls.length !== 1 || calls[0].name !== TOOL_NAME) {
          throw new Error("unexpected tool call: " + JSON.stringify(calls));
        }
        const second = await client.chat({
          messages: [
            initial,
            first.message,
            {
              role: "tool",
              content: TOOL_RESULT,
              options: {
                toolResult: {
                  id: calls[0].id,
                  result: TOOL_RESULT,
                  isError: false,
                },
              },
            },
          ],
          tools: [LLAMA_TOOL],
          stream: false,
        });
        if (second.message.content !== RESPONSE_TEXT) {
          throw new Error("unexpected tool continuation: " + JSON.stringify(second.message.content));
        }
      },
      async structuredOnce(client, prompt) {
        const response = await client.chat({
          messages: [{ role: "user", content: prompt }],
          responseFormat: RESPONSE_FORMAT,
          stream: false,
        });
        return String(response.message.content ?? "");
      },
      async close() {},
    };
  }

  throw new Error("unknown framework: " + framework);
}

async function runFramework(framework, baseUrl, runs) {
  const adapter = await loadAdapter(framework, baseUrl);
  const client = adapter.create();
  const samples = Object.fromEntries(SCENARIOS.map((name) => [name, []]));
  let memoryStages = {};
  try {
    await adapter.toolRoundtrip(client);
    validateStructured(await adapter.structuredOnce(client, "structured"));

    for (let run = 0; run < runs; run += 1) {
      samples.tool_roundtrip.push(await measured(() => adapter.toolRoundtrip(client)));

      samples.tool_loop_5.push(await measured(async () => {
        for (let index = 0; index < 5; index += 1) {
          await adapter.toolRoundtrip(client);
        }
      }));

      samples.structured_valid.push(await measured(async () => {
        validateStructured(await adapter.structuredOnce(client, "structured"));
      }));

      samples.structured_repair.push(await measured(async () => {
        const invalid = await adapter.structuredOnce(client, "[structured-invalid]");
        let rejected = false;
        try {
          validateStructured(invalid);
        } catch {
          rejected = true;
        }
        if (!rejected) {
          throw new Error("repair scenario expected the first response to be invalid");
        }
        validateStructured(await adapter.structuredOnce(client, "[structured-repair]"));
      }));

      samples.structured_repeat_20.push(await measured(async () => {
        for (let index = 0; index < 20; index += 1) {
          validateStructured(await adapter.structuredOnce(client, "structured"));
        }
      }));
    }
    memoryStages = {
      before_memory_probe: memoryUsageMiB(),
    };
    await adapter.toolRoundtrip(client);
    memoryStages.after_tool_roundtrip = memoryUsageMiB();

    for (let index = 0; index < 5; index += 1) {
      await adapter.toolRoundtrip(client);
    }
    memoryStages.after_tool_loop_5 = memoryUsageMiB();

    validateStructured(await adapter.structuredOnce(client, "structured"));
    memoryStages.after_structured_valid = memoryUsageMiB();

    const invalid = await adapter.structuredOnce(client, "[structured-invalid]");
    try {
      validateStructured(invalid);
    } catch {}
    validateStructured(
      await adapter.structuredOnce(client, "[structured-repair]"),
    );
    memoryStages.after_structured_repair = memoryUsageMiB();

    for (let index = 0; index < 20; index += 1) {
      validateStructured(
        await adapter.structuredOnce(client, "structured"),
      );
    }
    memoryStages.after_structured_repeat_20 = memoryUsageMiB();
  } finally {
    await adapter.close(client);
  }

  return {
    framework,
    runs,
    ...Object.fromEntries(
      Object.entries(samples).map(([name, rows]) => [name, summarize(rows)]),
    ),
    samples,
    memory_stages: memoryStages,
    caveat:
      "Wire-level feature comparison; validation is performed by this runner " +
      "so framework-specific Zod/schema-library costs are not mixed into transport overhead.",
  };
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) return '"' + text.replaceAll('"', '""') + '"';
  return text;
}

async function writeResults(results, outputDir) {
  await mkdir(outputDir, { recursive: true });
  await writeFile(
    resolve(outputDir, "feature-summary.json"),
    JSON.stringify(results, null, 2) + "\n",
  );

  const fields = ["framework", "runs"];
  for (const scenario of SCENARIOS) {
    fields.push(
      scenario + "_median_ms",
      scenario + "_p95_ms",
      scenario + "_cpu_percent",
    );
  }
  const summaryRows = [fields.join(",")];
  for (const result of results) {
    const row = { framework: result.framework, runs: result.runs };
    for (const scenario of SCENARIOS) {
      row[scenario + "_median_ms"] = result[scenario].median_ms;
      row[scenario + "_p95_ms"] = result[scenario].p95_ms;
      row[scenario + "_cpu_percent"] = result[scenario].median_cpu_percent;
    }
    summaryRows.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  await writeFile(resolve(outputDir, "feature-summary.csv"), summaryRows.join("\n") + "\n");

  const memoryFields = [
    "framework", "stage", "rss_mib", "heap_used_mib", "external_mib",
  ];
  const memoryRows = [memoryFields.join(",")];
  for (const result of results) {
    for (const [stage, memory] of Object.entries(result.memory_stages)) {
      memoryRows.push(
        [
          result.framework,
          stage,
          memory.rss_mib,
          memory.heap_used_mib,
          memory.external_mib,
        ].map(csvEscape).join(","),
      );
    }
  }
  await writeFile(
    resolve(outputDir, "feature-memory-stages.csv"),
    memoryRows.join("\n") + "\n",
  );

  const sampleFields = ["framework", "scenario", "run", "wall_ms", "cpu_ms"];
  const sampleRows = [sampleFields.join(",")];
  for (const result of results) {
    for (const [scenario, rows] of Object.entries(result.samples)) {
      rows.forEach((sample, index) => {
        sampleRows.push(
          [
            result.framework,
            scenario,
            index + 1,
            sample.wall_ms,
            sample.cpu_ms,
          ].map(csvEscape).join(","),
        );
      });
    }
  }
  await writeFile(resolve(outputDir, "feature-samples.csv"), sampleRows.join("\n") + "\n");
}

function parseArgs(argv) {
  let runs = 10;
  let framework;
  let outputDir = resolve(HERE, "results/features");
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--runs") runs = Number(argv[++index]);
    else if (value === "--framework") framework = argv[++index];
    else if (value === "--output-dir") outputDir = resolve(argv[++index]);
    else throw new Error("unknown argument: " + value);
  }
  if (!Number.isInteger(runs) || runs < 1) {
    throw new Error("--runs must be a positive integer");
  }
  if (framework && !FRAMEWORKS.includes(framework)) {
    throw new Error("unknown framework: " + framework);
  }
  return { runs, framework, outputDir };
}

const args = parseArgs(process.argv.slice(2));
const { server, baseUrl } = await startMockOpenAI();
try {
  const frameworks = args.framework ? [args.framework] : FRAMEWORKS;
  const results = [];
  for (const framework of frameworks) {
    const result = await runFramework(framework, baseUrl, args.runs);
    results.push(result);
    console.log(
      framework.padEnd(20) +
      " tool=" + result.tool_roundtrip.median_ms.toFixed(2) + " ms" +
      " structured=" + result.structured_valid.median_ms.toFixed(2) + " ms" +
      " repair=" + result.structured_repair.median_ms.toFixed(2) + " ms",
    );
  }
  await writeResults(results, args.outputDir);
  if (!args.framework && args.runs >= 30) {
    await writeFile(
      resolve(HERE, "feature-measured-results.csv"),
      await readFile(resolve(args.outputDir, "feature-summary.csv")),
    );
    generateCharts();
  }
  console.log("\nWrote feature comparison to " + args.outputDir);
} finally {
  await new Promise((resolveClose) => server.close(resolveClose));
}

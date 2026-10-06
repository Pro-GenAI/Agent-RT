import { LLMock } from "@copilotkit/aimock";

const requestedPort = Number.parseInt(process.argv[2] ?? "0", 10);
const mock = new LLMock({
  host: "127.0.0.1",
  port: Number.isFinite(requestedPort) ? requestedPort : 0,
});

const EMBEDDING = [0.01, 0.02, 0.03, 0.04];

// Smallest value that satisfies a JSON schema, deterministically. Structured
// output requests (pydantic models, zod schemas) otherwise get the plain
// "mock response" text and fail to parse before any application code runs.
function schemaInstance(schema, defs = {}, depth = 0) {
  if (!schema || typeof schema !== "object" || depth > 12) return null;
  defs = { ...defs, ...(schema.$defs ?? {}), ...(schema.definitions ?? {}) };
  if (typeof schema.$ref === "string") {
    const target = defs[schema.$ref.split("/").pop()];
    return schemaInstance(target, defs, depth + 1);
  }
  if ("const" in schema) return schema.const;
  if (Array.isArray(schema.enum) && schema.enum.length) return schema.enum[0];
  if ("default" in schema) return schema.default;
  for (const key of ["anyOf", "oneOf", "allOf"]) {
    if (Array.isArray(schema[key]) && schema[key].length) {
      const options = schema[key].filter((option) => option?.type !== "null");
      return schemaInstance(options[0] ?? schema[key][0], defs, depth + 1);
    }
  }
  const type = Array.isArray(schema.type)
    ? schema.type.find((item) => item !== "null") ?? "null"
    : schema.type ?? (schema.properties ? "object" : undefined);
  switch (type) {
    case "object": {
      const value = {};
      for (const [name, property] of Object.entries(schema.properties ?? {})) {
        value[name] = schemaInstance(property, defs, depth + 1);
      }
      return value;
    }
    case "array": {
      const count = Math.max(0, schema.minItems ?? 0);
      return Array.from({ length: count }, () => schemaInstance(schema.items, defs, depth + 1));
    }
    case "string":
      if (schema.format === "date-time") return "2024-01-01T00:00:00Z";
      if (schema.format === "date") return "2024-01-01";
      if (schema.format === "email") return "mock@example.com";
      if (schema.format === "uri" || schema.format === "url") return "https://example.com";
      return "mock".padEnd(schema.minLength ?? 0, "x");
    case "integer":
    case "number":
      return schema.minimum ?? schema.exclusiveMinimum ?? 0;
    case "boolean":
      return false;
    case "null":
      return null;
    default:
      return null;
  }
}

function formatSchema(format) {
  if (!format || typeof format !== "object") return undefined;
  const nested = format.json_schema ?? format;
  return nested.schema ?? undefined;
}

function messageText(message) {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part) => part?.text ?? "").join(" ");
  return "";
}

function structuredContent(format, messages = []) {
  if (format?.type === "json_schema") {
    const schema = formatSchema(format);
    if (schema) return JSON.stringify(schemaInstance(schema));
  }
  if (format?.type === "json_object") return "{}";
  // Prompts that ask for JSON (output parsers, "respond in JSON") get an
  // empty object rather than text no JSON parser accepts.
  const prompt = messages
    .filter((message) => message?.role === "system" || message?.role === "user")
    .map(messageText)
    .join("\n");
  if (/\bjson\b/i.test(prompt)) return "{}";
  return "mock response";
}

mock.onMessage(/.*/, (req) => ({
  content: structuredContent(req.response_format, req.messages),
}));
mock.onEmbedding(/.*/, { embedding: EMBEDDING });

// google-genai (and langchain-google-genai) embed through
// models/{model}:batchEmbedContents, which aimock does not route; answer it
// with one deterministic vector per request. Other /v1beta paths fall through.
const BATCH_EMBED_RE = /^\/models\/[^/:]+:batchEmbedContents$/;
mock.mount("/v1beta", {
  async handleRequest(req, res, pathname) {
    if (req.method !== "POST" || !BATCH_EMBED_RE.test(pathname)) return false;
    let body = "";
    for await (const chunk of req) body += chunk;
    let count = 1;
    try {
      const requests = JSON.parse(body || "{}").requests;
      if (Array.isArray(requests)) count = requests.length;
    } catch {
      // Malformed bodies still get a single embedding.
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        embeddings: Array.from({ length: count }, () => ({ values: EMBEDDING })),
      }),
    );
    return true;
  },
});

// Ollama's /api/embed returns {"embeddings": [[...], ...]} (one per input);
// aimock answers it with the legacy /api/embeddings shape, which the current
// ollama client (and langchain-ollama) rejects.
mock.mount("/api/embed", {
  async handleRequest(req, res, pathname) {
    if (req.method !== "POST" || pathname !== "/") return false;
    let body = "";
    for await (const chunk of req) body += chunk;
    let model = "mock";
    let count = 1;
    try {
      const parsed = JSON.parse(body || "{}");
      if (typeof parsed.model === "string") model = parsed.model;
      if (Array.isArray(parsed.input)) count = parsed.input.length;
    } catch {
      // Malformed bodies still get a single embedding.
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({ model, embeddings: Array.from({ length: count }, () => EMBEDDING) }),
    );
    return true;
  },
});

// Hand an already-consumed body to aimock's own handler, which reads it with
// req.on("data") / req.on("end").
function replayBody(req, body) {
  const on = req.on.bind(req);
  req.on = (event, listener) => {
    if (event === "data") {
      setImmediate(() => listener(Buffer.from(body)));
      return req;
    }
    if (event === "end") {
      setImmediate(() => setImmediate(listener));
      return req;
    }
    return on(event, listener);
  };
}

// The real Responses API always reports usage.input_tokens_details and
// usage.output_tokens_details; aimock omits them, and clients such as the
// OpenAI Agents SDK dereference them (`None.cached_tokens`).
const USAGE_DETAILS = {
  input_tokens_details: { cached_tokens: 0 },
  output_tokens_details: { reasoning_tokens: 0 },
};

function addUsageDetails(res) {
  const writeHead = res.writeHead.bind(res);
  res.writeHead = (status, ...rest) => {
    // The body length changes, so drop any Content-Length aimock set.
    const dropLength = (headers) => {
      if (headers && typeof headers === "object" && !Array.isArray(headers)) {
        for (const key of Object.keys(headers)) {
          if (key.toLowerCase() === "content-length") delete headers[key];
        }
      }
    };
    rest.forEach(dropLength);
    return writeHead(status, ...rest);
  };
  // Streaming: the `response.completed` event embeds the same usage object.
  const write = res.write.bind(res);
  res.write = (chunk, ...rest) => {
    if (typeof chunk === "string" || Buffer.isBuffer(chunk)) {
      const text = chunk.toString();
      if (text.includes('"usage":{')) {
        chunk = text.replace(/"usage":\{(?!"input_tokens_details")/g, '"usage":{"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0},');
      }
    }
    return write(chunk, ...rest);
  };
  const end = res.end.bind(res);
  res.end = (chunk, ...rest) => {
    if (typeof chunk === "string" || Buffer.isBuffer(chunk)) {
      try {
        const parsed = JSON.parse(chunk.toString());
        if (parsed?.object === "response" && parsed.usage) {
          parsed.usage = { ...USAGE_DETAILS, ...parsed.usage };
          chunk = JSON.stringify(parsed);
        }
      } catch {
        // Not a JSON body (for example an SSE stream): pass it through.
      }
    }
    return end(chunk, ...rest);
  };
}

// aimock drops the Responses API `text.format` when it normalizes requests,
// so non-streaming structured Responses calls are answered here directly.
mock.mount("/v1/responses", {
  async handleRequest(req, res, pathname) {
    if (req.method !== "POST" || pathname !== "/") return false;
    let body = "";
    for await (const chunk of req) body += chunk;
    let parsed;
    try {
      parsed = JSON.parse(body || "{}");
    } catch {
      return false;
    }
    const format = parsed?.text?.format;
    if (parsed.stream || !format || format.type === "text") {
      replayBody(req, body);
      addUsageDetails(res);
      return false;
    }
    const text = structuredContent(format, []);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        id: "resp_mock",
        object: "response",
        created_at: 0,
        status: "completed",
        model: parsed.model ?? "mock",
        output: [
          {
            id: "msg_mock",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text, annotations: [] }],
          },
        ],
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2, ...USAGE_DETAILS },
      }),
    );
    return true;
  },
});

await mock.start();
console.log(`AIMOCK_READY=${mock.url}`);

let closing = false;
async function shutdown() {
  if (closing) return;
  closing = true;
  try {
    await mock.stop();
  } finally {
    process.exit(0);
  }
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

await new Promise(() => {});

const test = require("node:test");
const assert = require("node:assert/strict");

const OpenAI = require("../dist/ext/compat/openai.js").default;
const Anthropic = require("../dist/ext/compat/anthropic.js").default;
const {
  AnthropicModelProvider,
  AuthorizedMCPClient,
  CodeInterpreterRegistry,
  MCPAccessPolicy,
  MCPCapabilityFilter,
  MCPClient,
  RetrievalRegistry,
  SandboxSession,
} = require("../dist/index.js");

test("exact-prefix package exports preserve upstream specifiers", () => {
  assert.equal(require("agent-rt/@langchain/openai").ChatOpenAI, require("../dist/ext/compat/langchain.js").ChatOpenAI);
  assert.equal(require("agent-rt/@llamaindex/openai").OpenAI, require("../dist/ext/compat/llamaindex.js").OpenAI);
  assert.equal(require("agent-rt/@langchain/anthropic").ChatAnthropic, require("../dist/ext/compat/langchain.js").ChatAnthropic);
  assert.equal(require("agent-rt/@llamaindex/anthropic").Anthropic, require("../dist/ext/compat/llamaindex.js").Anthropic);
  assert.equal(require("agent-rt/openai").default, OpenAI);
  assert.equal(require("agent-rt/@anthropic-ai/sdk").default, Anthropic);
});


class CaptureProvider {
  constructor() {
    this.name = "capture";
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: '{"answer":"ok"}' }],
      },
      model: request.model,
      usage: { inputTokens: 3, outputTokens: 4, totalTokens: 7 },
      finishReason: "stop",
    };
  }

  async embed(request) {
    this.requests.push(request);
    return {
      data: [
        { index: 0, embedding: [0.1, 0.2] },
        { index: 1, embedding: "YmFzZTY0" },
      ],
      model: request.model,
      usage: { inputTokens: 5, totalTokens: 5 },
    };
  }

  async countTokens(request) {
    this.requests.push(request);
    return 17;
  }
}

const schema = {
  type: "object",
  properties: { answer: { type: "string" } },
  required: ["answer"],
  additionalProperties: false,
};

test("agent-rt/openai maps Chat Completions JSON Schema structured output", async () => {
  const provider = new CaptureProvider();
  const client = new OpenAI({ provider });

  const response = await client.chat.completions.create({
    model: "test-model",
    messages: [{ role: "user", content: "Return JSON." }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "answer",
        schema,
        strict: false,
      },
    },
  });

  assert.deepEqual(provider.requests[0].structuredOutput, {
    name: "answer",
    schema,
    strict: false,
  });
  assert.equal(response.object, "chat.completion");
  assert.equal(response.choices[0].message.content, '{"answer":"ok"}');
  assert.deepEqual(response.usage, {
    prompt_tokens: 3,
    completion_tokens: 4,
    total_tokens: 7,
  });
});

test("agent-rt/openai maps Responses text.format JSON Schema structured output", async () => {
  const provider = new CaptureProvider();
  const client = new OpenAI({ provider });

  const response = await client.responses.create({
    model: "test-model",
    instructions: "Return structured JSON.",
    input: "Give an answer.",
    text: {
      format: {
        type: "json_schema",
        name: "answer",
        schema,
        strict: true,
      },
    },
  });

  assert.deepEqual(provider.requests[0].structuredOutput, {
    name: "answer",
    schema,
    strict: true,
  });
  assert.deepEqual(
    provider.requests[0].messages.map((message) => message.role),
    ["system", "user"],
  );
  assert.equal(response.object, "response");
  assert.equal(response.output_text, '{"answer":"ok"}');
});

test("agent-rt/openai maps reasoning controls", async () => {
  const provider = new CaptureProvider();
  const client = new OpenAI({ provider });

  await client.chat.completions.create({
    model: "test-model",
    messages: [{ role: "user", content: "Think." }],
    reasoning_effort: "high",
  });
  assert.deepEqual(provider.requests[0].reasoning, { effort: "high", summary: undefined });

  await client.responses.create({
    model: "test-model",
    input: "Think.",
    reasoning: { effort: "xhigh", summary: "auto" },
  });
  assert.deepEqual(provider.requests[1].reasoning, { effort: "xhigh", summary: "auto" });
});

test("agent-rt/anthropic maps adaptive and extended thinking controls", async () => {
  const provider = new CaptureProvider();
  const client = new Anthropic({ provider });

  await client.messages.create({
    model: "test-model",
    max_tokens: 512,
    messages: [{ role: "user", content: "Think." }],
    thinking: { type: "adaptive" },
    output_config: { effort: "high" },
  });
  assert.deepEqual(provider.requests[0].reasoning, {
    thinking: "adaptive",
    budgetTokens: undefined,
    effort: "high",
  });

  await client.messages.create({
    model: "test-model",
    max_tokens: 4096,
    messages: [{ role: "user", content: "Think." }],
    thinking: { type: "enabled", budget_tokens: 2048 },
  });
  assert.deepEqual(provider.requests[1].reasoning, {
    thinking: "enabled",
    budgetTokens: 2048,
    effort: undefined,
  });
});

test("reasoning compatibility rejects malformed controls before provider invocation", async () => {
  const openaiProvider = new CaptureProvider();
  const openai = new OpenAI({ provider: openaiProvider });
  await assert.rejects(
    openai.responses.create({ model: "test-model", input: "x", reasoning: { effort: "" } }),
    /reasoning\.effort/,
  );
  assert.equal(openaiProvider.requests.length, 0);

  const anthropicProvider = new CaptureProvider();
  const anthropic = new Anthropic({ provider: anthropicProvider });
  await assert.rejects(
    anthropic.messages.create({
      model: "test-model",
      max_tokens: 256,
      messages: [{ role: "user", content: "x" }],
      thinking: { type: "adaptive", budget_tokens: 128 },
    }),
    /requires thinking\.type='enabled'/,
  );
  assert.equal(anthropicProvider.requests.length, 0);
});

test("agent-rt/anthropic maps output_config.format and legacy output_format", async () => {
  const provider = new CaptureProvider();
  const client = new Anthropic({ provider });

  const first = await client.messages.create({
    model: "test-model",
    max_tokens: 64,
    system: "Return JSON.",
    messages: [{ role: "user", content: "Give an answer." }],
    output_config: {
      format: {
        type: "json_schema",
        schema,
      },
    },
  });
  await client.messages.create({
    model: "test-model",
    max_tokens: 64,
    messages: [{ role: "user", content: "Again." }],
    output_format: {
      type: "json_schema",
      schema,
      name: "legacy-answer",
      strict: false,
    },
  });

  assert.deepEqual(provider.requests[0].structuredOutput, {
    schema,
    strict: true,
  });
  assert.deepEqual(provider.requests[1].structuredOutput, {
    name: "legacy-answer",
    schema,
    strict: false,
  });
  assert.deepEqual(
    provider.requests[0].messages.map((message) => message.role),
    ["system", "user"],
  );
  assert.equal(first.type, "message");
  assert.equal(first.content[0].text, '{"answer":"ok"}');
});

test("native Anthropic provider sends structuredOutput through output_config.format", async () => {
  const calls = [];
  const client = {
    messages: {
      create: async (params) => {
        calls.push(params);
        return {
          model: "claude-test",
          content: [{ type: "text", text: '{"answer":"ok"}' }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 2 },
        };
      },
    },
  };
  const provider = new AnthropicModelProvider(
    { defaultModel: "claude-test" },
    client,
  );

  await provider.complete({
    messages: [{ role: "user", content: [{ type: "text", text: "Return JSON." }] }],
    structuredOutput: { name: "answer", schema, strict: true },
  });

  assert.deepEqual(calls[0].output_config, {
    format: {
      type: "json_schema",
      schema,
    },
  });
});

test("agent-rt/openai exposes embeddings.create through Agent RT provider", async () => {
  const provider = new CaptureProvider();
  const client = new OpenAI({ provider });

  const response = await client.embeddings.create({
    model: "embedding-model",
    input: ["first", "second"],
    dimensions: 256,
    encoding_format: "float",
  });

  assert.deepEqual(provider.requests[0], {
    model: "embedding-model",
    input: ["first", "second"],
    dimensions: 256,
    encodingFormat: "float",
  });
  assert.equal(response.object, "list");
  assert.equal(response.model, "embedding-model");
  assert.deepEqual(response.data[0], {
    object: "embedding",
    index: 0,
    embedding: [0.1, 0.2],
  });
  assert.equal(response.data[1].embedding, "YmFzZTY0");
  assert.deepEqual(response.usage, { prompt_tokens: 5, total_tokens: 5 });
});

test("agent-rt/anthropic exposes messages.countTokens", async () => {
  const provider = new CaptureProvider();
  const client = new Anthropic({ provider });

  const response = await client.messages.countTokens({
    model: "test-model",
    system: "Be concise.",
    messages: [{ role: "user", content: "Count this." }],
  });

  assert.deepEqual(response, { input_tokens: 17 });
  assert.equal(provider.requests[0].model, "test-model");
  assert.deepEqual(
    provider.requests[0].messages.map((message) => message.role),
    ["system", "user"],
  );
});

class HostedCodeProvider {
  constructor(toolName, toolArguments) {
    this.name = "hosted-code-fake";
    this.toolName = toolName;
    this.toolArguments = toolArguments;
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    if (this.requests.length === 1) {
      return {
        message: {
          role: "assistant",
          content: [],
          toolCalls: [
            {
              id: "sandbox-call-1",
              name: this.toolName,
              arguments: this.toolArguments,
            },
          ],
        },
        model: request.model,
        finishReason: "tool_calls",
      };
    }
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Sandbox complete." }],
      },
      model: request.model,
      finishReason: "stop",
    };
  }
}

function makeMigrationSandbox() {
  const codeRuns = [];
  const shellRuns = [];
  const interpreters = new CodeInterpreterRegistry();
  interpreters.register(
    "python",
    async (sessionId, runtime, code, state) => {
      codeRuns.push({ sessionId, runtime, code });
      state.runs = Number(state.runs ?? 0) + 1;
      return {
        exitCode: 0,
        stdout: new TextEncoder().encode(`code:${code}:runs=${state.runs}`),
        stderr: new Uint8Array(),
        durationMs: 4,
      };
    },
  );
  const backend = {
    execute: async (sessionId, command) => {
      shellRuns.push({ sessionId, command });
      return {
        exitCode: 0,
        stdout: new TextEncoder().encode(`shell:${command.argv.join(" ")}`),
        stderr: new Uint8Array(),
        durationMs: 3,
      };
    },
  };
  return {
    sandbox: new SandboxSession("migration-sandbox", backend, { interpreters }),
    codeRuns,
    shellRuns,
  };
}

test("agent-rt/openai routes code_interpreter through SandboxSession", async () => {
  const { sandbox, codeRuns } = makeMigrationSandbox();
  const provider = new HostedCodeProvider("agent_rt_code_execution", {
    code: "print(2 + 2)",
    runtime: "python",
  });
  const client = new OpenAI({ provider, sandboxSession: sandbox });

  const response = await client.responses.create({
    model: "test-model",
    input: "Calculate with Python.",
    tools: [{ type: "code_interpreter", container: { type: "auto" } }],
  });

  assert.equal(provider.requests.length, 2);
  assert.equal(provider.requests[0].tools[0].name, "agent_rt_code_execution");
  assert.deepEqual(provider.requests[1].messages.map((message) => message.role), [
    "user",
    "assistant",
    "tool",
  ]);
  assert.equal(provider.requests[1].messages[2].toolCallId, "sandbox-call-1");
  assert.match(provider.requests[1].messages[2].content[0].text, /runs=1/);
  assert.deepEqual(codeRuns, [
    {
      sessionId: "migration-sandbox",
      runtime: "python",
      code: "print(2 + 2)",
    },
  ]);
  assert.equal(response.output[0].type, "code_interpreter_call");
  assert.equal(response.output[0].status, "completed");
  assert.match(response.output[0].outputs[0].logs, /runs=1/);
  assert.equal(response.output_text, "Sandbox complete.");
});

test("agent-rt/openai routes hosted shell through SandboxSession", async () => {
  const { sandbox, shellRuns } = makeMigrationSandbox();
  const provider = new HostedCodeProvider("agent_rt_shell", {
    command: "printf hello",
    cwd: "work",
  });
  const client = new OpenAI({ provider, sandboxSession: sandbox });

  const response = await client.responses.create({
    model: "test-model",
    input: "Run a shell command.",
    tools: [{ type: "shell" }],
  });

  assert.equal(shellRuns.length, 1);
  assert.deepEqual(shellRuns[0].command.argv, ["sh", "-lc", "printf hello"]);
  assert.equal(shellRuns[0].command.cwd, "work");
  assert.equal(response.output[0].type, "shell_call");
  assert.match(response.output[0].output.stdout, /printf hello/);
});

test("agent-rt/anthropic routes code_execution through SandboxSession", async () => {
  const { sandbox } = makeMigrationSandbox();
  const provider = new HostedCodeProvider("agent_rt_code_execution", {
    code: "x = 7",
  });
  const client = new Anthropic({ provider, sandboxSession: sandbox });

  const response = await client.messages.create({
    model: "test-model",
    max_tokens: 128,
    messages: [{ role: "user", content: "Use code." }],
    tools: [{ type: "code_execution_20250825", name: "code_execution" }],
  });

  assert.equal(provider.requests[0].tools[0].name, "agent_rt_code_execution");
  assert.equal(response.content[0].type, "server_tool_use");
  assert.equal(response.content[0].name, "code_execution");
  assert.equal(response.content[1].type, "code_execution_tool_result");
  assert.match(response.content[1].content.stdout, /code:x = 7:runs=1/);
  assert.equal(response.content[2].text, "Sandbox complete.");
});

test("hosted code execution fails before model invocation without sandbox", async () => {
  const provider = new HostedCodeProvider("agent_rt_code_execution", {
    code: "print('unsafe')",
  });
  const openai = new OpenAI({ provider });

  await assert.rejects(
    openai.responses.create({
      model: "test-model",
      input: "Use Python.",
      tools: [{ type: "code_interpreter" }],
    }),
    /requires sandboxSession/,
  );
  assert.equal(provider.requests.length, 0);

  const anthropicProvider = new HostedCodeProvider("agent_rt_code_execution", {
    code: "print('unsafe')",
  });
  const anthropic = new Anthropic({ provider: anthropicProvider });
  await assert.rejects(
    anthropic.messages.create({
      model: "test-model",
      max_tokens: 64,
      messages: [{ role: "user", content: "Use code." }],
      tools: [{ type: "code_execution_20250825", name: "code_execution" }],
    }),
    /requires sandboxSession/,
  );
  assert.equal(anthropicProvider.requests.length, 0);
});

class MCPExecutionProvider {
  constructor() {
    this.name = "mcp-fake";
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    if (this.requests.length === 1) {
      const tool = request.tools.find((item) => item.name.includes("search"));
      return {
        message: {
          role: "assistant",
          content: [],
          toolCalls: [
            {
              id: "mcp-call-1",
              name: tool.name,
              arguments: { query: "Agent RT" },
            },
          ],
        },
        model: request.model,
        finishReason: "tool_calls",
      };
    }
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: "MCP complete." }],
      },
      model: request.model,
      finishReason: "stop",
    };
  }
}

function makeMCPClient({ authorized = false } = {}) {
  const calls = [];
  const transport = {
    request: async (method, params = {}) => {
      calls.push({ method, params });
      if (method === "initialize") {
        return {
          serverInfo: { name: "docs", version: "1" },
          capabilities: { tools: true },
        };
      }
      if (method === "tools/list") {
        return {
          tools: [
            {
              name: "search",
              description: "Search docs",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
            },
            {
              name: "admin.delete",
              description: "Dangerous",
              inputSchema: { type: "object" },
            },
          ],
        };
      }
      if (method === "tools/call") {
        return { ok: true, tool: params.name, query: params.arguments.query };
      }
      throw new Error(`unexpected MCP method ${method}`);
    },
  };
  const client = new MCPClient(transport, "migration-test", "1", ["tools"]);
  if (!authorized) return { client, calls };
  return {
    client: new AuthorizedMCPClient(
      client,
      new MCPAccessPolicy({
        capabilityFilter: new MCPCapabilityFilter({ tools: ["search"] }),
      }),
    ),
    calls,
  };
}

test("agent-rt/openai routes remote MCP through injected Agent RT client", async () => {
  const provider = new MCPExecutionProvider();
  const { client: mcp, calls } = makeMCPClient();
  const client = new OpenAI({ provider, mcpClients: { docs: mcp } });

  const response = await client.responses.create({
    model: "test-model",
    input: "Search the docs.",
    tools: [
      {
        type: "mcp",
        server_label: "docs",
        server_url: "https://vendor.example.invalid/mcp",
        authorization: "vendor-secret-must-not-be-used",
        allowed_tools: ["search"],
        require_approval: "never",
      },
    ],
  });

  assert.deepEqual(provider.requests[0].tools.map((tool) => tool.name), [
    "mcp__docs__search",
  ]);
  assert.deepEqual(provider.requests[1].messages.map((message) => message.role), [
    "user",
    "assistant",
    "tool",
  ]);
  assert.match(provider.requests[1].messages[2].content[0].text, /Agent RT/);
  assert.deepEqual(calls.map((call) => call.method), [
    "initialize",
    "tools/list",
    "tools/call",
  ]);
  assert.equal(calls[2].params.name, "search");
  assert.deepEqual(calls[2].params.arguments, { query: "Agent RT" });
  assert.equal(
    calls.some((call) => JSON.stringify(call).includes("vendor-secret-must-not-be-used")),
    false,
  );
  assert.equal(response.output[0].type, "mcp_call");
  assert.equal(response.output[0].server_label, "docs");
  assert.equal(response.output[0].name, "search");
  assert.equal(response.output[0].status, "completed");
  assert.match(response.output[0].output, /"ok":true/);
  assert.equal(response.output_text, "MCP complete.");
});

test("agent-rt/openai respects AuthorizedMCPClient discovery policy", async () => {
  const provider = new MCPExecutionProvider();
  const { client: mcp } = makeMCPClient({ authorized: true });
  const client = new OpenAI({ provider, mcpClients: { docs: mcp } });

  await client.responses.create({
    model: "test-model",
    input: "Search.",
    tools: [{ type: "mcp", server_label: "docs" }],
  });

  assert.deepEqual(provider.requests[0].tools.map((tool) => tool.name), [
    "mcp__docs__search",
  ]);
});

test("agent-rt/anthropic beta messages routes MCP Connector through Agent RT", async () => {
  const provider = new MCPExecutionProvider();
  const { client: mcp, calls } = makeMCPClient();
  const client = new Anthropic({ provider, mcpClients: { docs: mcp } });

  const response = await client.beta.messages.create({
    model: "test-model",
    max_tokens: 128,
    messages: [{ role: "user", content: "Search docs." }],
    mcp_servers: [
      {
        type: "url",
        name: "docs",
        url: "https://vendor.example.invalid/mcp",
        authorization_token: "ignored-vendor-token",
      },
    ],
    tools: [
      {
        type: "mcp_toolset",
        mcp_server_name: "docs",
        default_config: { enabled: false },
        configs: { search: { enabled: true } },
      },
    ],
  });

  assert.deepEqual(provider.requests[0].tools.map((tool) => tool.name), [
    "mcp__docs__search",
  ]);
  assert.deepEqual(calls.map((call) => call.method), [
    "initialize",
    "tools/list",
    "tools/call",
  ]);
  assert.equal(
    calls.some((call) => JSON.stringify(call).includes("ignored-vendor-token")),
    false,
  );
  assert.equal(response.content[0].type, "mcp_tool_use");
  assert.equal(response.content[0].server_name, "docs");
  assert.equal(response.content[0].name, "search");
  assert.equal(response.content[1].type, "mcp_tool_result");
  assert.match(response.content[1].content, /"ok":true/);
  assert.equal(response.content[2].text, "MCP complete.");
});

test("remote MCP fails before model invocation when binding is missing or invalid", async () => {
  const openAIProvider = new MCPExecutionProvider();
  const openai = new OpenAI({ provider: openAIProvider });
  await assert.rejects(
    openai.responses.create({
      model: "test-model",
      input: "Search.",
      tools: [{ type: "mcp", server_label: "docs", server_url: "https://ignored" }],
    }),
    /requires mcpClients\["docs"\]/,
  );
  assert.equal(openAIProvider.requests.length, 0);

  const anthropicProvider = new MCPExecutionProvider();
  const anthropic = new Anthropic({ provider: anthropicProvider });
  await assert.rejects(
    anthropic.beta.messages.create({
      model: "test-model",
      max_tokens: 64,
      messages: [{ role: "user", content: "Search." }],
      mcp_servers: [{ type: "url", name: "docs", url: "https://ignored" }],
      tools: [{ type: "mcp_toolset", mcp_server_name: "other" }],
    }),
    /must be referenced by exactly one mcp_toolset|undeclared MCP server/,
  );
  assert.equal(anthropicProvider.requests.length, 0);
});

test("agent-rt/openai exposes Models API through Agent RT catalog provider", async () => {
  const provider = new CaptureProvider();
  provider.listModels = async () => [
    { id: "gpt-a", created: 123, ownedBy: "agent-rt" },
    { id: "gpt-b", created: 456, ownedBy: "agent-rt" },
  ];
  provider.retrieveModel = async (id) => ({
    id,
    created: 123,
    ownedBy: "agent-rt",
  });
  const client = new OpenAI({ provider });

  const page = await client.models.list();
  const model = await client.models.retrieve("gpt-a");

  assert.equal(page.object, "list");
  assert.deepEqual(page.data[0], {
    id: "gpt-a",
    object: "model",
    created: 123,
    owned_by: "agent-rt",
  });
  assert.deepEqual(model, {
    id: "gpt-a",
    object: "model",
    created: 123,
    owned_by: "agent-rt",
  });
});

test("agent-rt/anthropic exposes Models API through Agent RT catalog provider", async () => {
  const provider = new CaptureProvider();
  provider.listModels = async () => [
    { id: "claude-a", displayName: "Claude A", created: "2026-01-02T03:04:05Z" },
    { id: "claude-b", displayName: "Claude B", created: "2026-02-03T04:05:06Z" },
  ];
  provider.retrieveModel = async (id) => ({
    id,
    displayName: "Claude A",
    created: "2026-01-02T03:04:05Z",
  });
  const client = new Anthropic({ provider });

  const page = await client.models.list();
  const model = await client.models.retrieve("claude-a");

  assert.equal(page.has_more, false);
  assert.equal(page.first_id, "claude-a");
  assert.equal(page.last_id, "claude-b");
  assert.deepEqual(page.data[0], {
    id: "claude-a",
    type: "model",
    display_name: "Claude A",
    created_at: "2026-01-02T03:04:05Z",
  });
  assert.deepEqual(model, {
    id: "claude-a",
    type: "model",
    display_name: "Claude A",
    created_at: "2026-01-02T03:04:05Z",
  });
});

class ContinuationProvider {
  constructor() {
    this.name = "continuation-fake";
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    const text = request.messages
      .map((message) => message.content.map((part) => part.text ?? "").join(""))
      .filter(Boolean)
      .join(" | ");
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: `seen:${text}` }],
      },
      model: request.model,
      finishReason: "stop",
    };
  }
}

test("agent-rt/openai Responses supports previous_response_id continuation", async () => {
  const provider = new ContinuationProvider();
  const client = new OpenAI({ provider, maxResponseStates: 4 });

  const first = await client.responses.create({
    model: "test-model",
    input: "Remember Atlas.",
  });
  const second = await client.responses.create({
    model: "test-model",
    previous_response_id: first.id,
    input: "What did I ask you to remember?",
  });

  assert.match(first.id, /^resp_agent_rt_\d+$/);
  assert.match(second.id, /^resp_agent_rt_\d+$/);
  assert.notEqual(first.id, second.id);
  assert.deepEqual(provider.requests[1].messages.map((message) => message.role), [
    "user",
    "assistant",
    "user",
  ]);
  assert.equal(
    provider.requests[1].messages[0].content[0].text,
    "Remember Atlas.",
  );
  assert.match(provider.requests[1].messages[1].content[0].text, /Remember Atlas/);
  assert.equal(
    provider.requests[1].messages[2].content[0].text,
    "What did I ask you to remember?",
  );
});

test("previous_response_id preserves sandbox tool-result history", async () => {
  const { sandbox } = makeMigrationSandbox();
  const provider = new HostedCodeProvider("agent_rt_code_execution", {
    code: "print(42)",
  });
  const client = new OpenAI({ provider, sandboxSession: sandbox });

  const first = await client.responses.create({
    model: "test-model",
    input: "Calculate.",
    tools: [{ type: "code_interpreter" }],
  });
  await client.responses.create({
    model: "test-model",
    previous_response_id: first.id,
    input: "Continue from that result.",
  });

  assert.equal(provider.requests.length, 3);
  assert.deepEqual(provider.requests[2].messages.map((message) => message.role), [
    "user",
    "assistant",
    "tool",
    "assistant",
    "user",
  ]);
  assert.equal(provider.requests[2].messages[2].toolCallId, "sandbox-call-1");
  assert.match(provider.requests[2].messages[2].content[0].text, /code:print\(42\):runs=1/);
  assert.equal(provider.requests[2].messages[4].content[0].text, "Continue from that result.");
});

test("previous_response_id rejects unknown and evicted state before provider invocation", async () => {
  const provider = new ContinuationProvider();
  const client = new OpenAI({ provider, maxResponseStates: 1 });

  await assert.rejects(
    client.responses.create({
      model: "test-model",
      previous_response_id: "resp_agent_rt_missing",
      input: "Nope.",
    }),
    /unknown or evicted previous_response_id/,
  );
  assert.equal(provider.requests.length, 0);

  const first = await client.responses.create({ model: "test-model", input: "First." });
  const second = await client.responses.create({ model: "test-model", input: "Second." });
  assert.notEqual(first.id, second.id);
  const before = provider.requests.length;
  await assert.rejects(
    client.responses.create({
      model: "test-model",
      previous_response_id: first.id,
      input: "Use evicted state.",
    }),
    /unknown or evicted previous_response_id/,
  );
  assert.equal(provider.requests.length, before);
});

test("OpenAI continuation store validates maxResponseStates", () => {
  const provider = new ContinuationProvider();
  assert.throws(() => new OpenAI({ provider, maxResponseStates: 0 }), /positive integer/);
  assert.throws(() => new OpenAI({ provider, maxResponseStates: 1.5 }), /positive integer/);
});

class FileSearchProvider {
  constructor() {
    this.name = "file-search-fake";
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    if (this.requests.length === 1) {
      return {
        message: {
          role: "assistant",
          content: [],
          toolCalls: [
            {
              id: "file-search-1",
              name: "agent_rt_file_search",
              arguments: { query: "deployment guide" },
            },
          ],
        },
        model: request.model,
        finishReason: "tool_calls",
      };
    }
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Found the deployment guide." }],
      },
      model: request.model,
      finishReason: "stop",
    };
  }
}

class FakeFileRetrieval {
  constructor(results) {
    this.kind = "file";
    this.results = results;
    this.queries = [];
  }

  async search(query) {
    this.queries.push(query);
    return this.results;
  }
}

function makeFileSearchRegistry() {
  const docs = new FakeFileRetrieval([
    {
      id: "file-a",
      title: "deploy.md",
      content: "Deployment instructions A",
      score: 0.92,
      uri: "file:///deploy.md",
      metadata: { team: "platform" },
    },
    {
      id: "file-low",
      title: "old.md",
      content: "Old instructions",
      score: 0.2,
    },
  ]);
  const release = new FakeFileRetrieval([
    {
      id: "file-b",
      title: "release.md",
      content: "Deployment instructions B",
      score: 0.81,
      metadata: { team: "release" },
    },
  ]);
  const registry = new RetrievalRegistry();
  registry.register("vs_docs", docs);
  registry.register("vs_release", release);
  return { registry, docs, release };
}

test("agent-rt/openai routes file_search through RetrievalRegistry", async () => {
  const provider = new FileSearchProvider();
  const { registry, docs, release } = makeFileSearchRegistry();
  const client = new OpenAI({ provider, retrievalRegistry: registry });

  const response = await client.responses.create({
    model: "test-model",
    input: "Find deployment docs.",
    tools: [
      {
        type: "file_search",
        vector_store_ids: ["vs_docs", "vs_release"],
        max_num_results: 2,
        filters: { team: "platform" },
        ranking_options: { score_threshold: 0.5 },
      },
    ],
  });

  assert.deepEqual(provider.requests[0].tools.map((tool) => tool.name), [
    "agent_rt_file_search",
  ]);
  assert.equal(provider.requests.length, 2);
  assert.deepEqual(provider.requests[1].messages.map((message) => message.role), [
    "user",
    "assistant",
    "tool",
  ]);
  assert.match(provider.requests[1].messages[2].content[0].text, /file-a/);
  assert.doesNotMatch(provider.requests[1].messages[2].content[0].text, /file-low/);
  assert.deepEqual(docs.queries, [
    {
      text: "deployment guide",
      limit: 2,
      filters: { team: "platform" },
    },
  ]);
  assert.deepEqual(release.queries[0].filters, { team: "platform" });

  const search = response.output[0];
  assert.equal(search.type, "file_search_call");
  assert.equal(search.status, "completed");
  assert.deepEqual(search.queries, ["deployment guide"]);
  assert.deepEqual(search.results.map((result) => result.file_id), ["file-a", "file-b"]);
  assert.equal(search.results[0].filename, "deploy.md");
  assert.equal(search.results[0].score, 0.92);
  assert.deepEqual(search.results[0].attributes, { team: "platform" });
  assert.equal(response.output_text, "Found the deployment guide.");
});

test("file_search history is retained for previous_response_id continuation", async () => {
  const provider = new FileSearchProvider();
  const { registry } = makeFileSearchRegistry();
  const client = new OpenAI({ provider, retrievalRegistry: registry });

  const first = await client.responses.create({
    model: "test-model",
    input: "Find deployment docs.",
    tools: [{ type: "file_search", vector_store_ids: ["vs_docs"] }],
  });
  await client.responses.create({
    model: "test-model",
    previous_response_id: first.id,
    input: "Use that result.",
  });

  assert.equal(provider.requests.length, 3);
  assert.deepEqual(provider.requests[2].messages.map((message) => message.role), [
    "user",
    "assistant",
    "tool",
    "assistant",
    "user",
  ]);
  assert.equal(provider.requests[2].messages[2].toolCallId, "file-search-1");
  assert.match(provider.requests[2].messages[2].content[0].text, /file-a/);
  assert.equal(provider.requests[2].messages[4].content[0].text, "Use that result.");
});

test("file_search requires an explicit RetrievalRegistry before provider invocation", async () => {
  const provider = new FileSearchProvider();
  const client = new OpenAI({ provider });

  await assert.rejects(
    client.responses.create({
      model: "test-model",
      input: "Search files.",
      tools: [{ type: "file_search", vector_store_ids: ["vs_docs"] }],
    }),
    /requires retrievalRegistry/,
  );
  assert.equal(provider.requests.length, 0);
});

class BatchMigrationProvider extends CaptureProvider {
  constructor() {
    super();
    this.batchCalls = [];
  }

  async createBatch(params) {
    this.batchCalls.push(["create", params]);
    return {
      id: "batch-1",
      status: "validating",
      raw: { id: "batch-1", object: "batch", status: "validating", ...params },
    };
  }

  async retrieveBatch(id) {
    this.batchCalls.push(["retrieve", id]);
    return { id, status: "completed", raw: { id, object: "batch", status: "completed" } };
  }

  async listBatches(params = {}) {
    this.batchCalls.push(["list", params]);
    return [
      { id: "batch-1", status: "completed", raw: { id: "batch-1", object: "batch", status: "completed" } },
      { id: "batch-2", status: "in_progress", raw: { id: "batch-2", object: "batch", status: "in_progress" } },
    ];
  }

  async cancelBatch(id) {
    this.batchCalls.push(["cancel", id]);
    return { id, status: "cancelling", raw: { id, object: "batch", status: "cancelling" } };
  }

  async batchResults(id) {
    this.batchCalls.push(["results", id]);
    return [
      { custom_id: "request-1", result: { type: "succeeded", message: { id: "msg-1" } } },
      { custom_id: "request-2", result: { type: "errored", error: { type: "invalid_request" } } },
    ];
  }
}

test("agent-rt/openai exposes Batch API lifecycle", async () => {
  const provider = new BatchMigrationProvider();
  const client = new OpenAI({ provider });

  const created = await client.batches.create({
    input_file_id: "file-input",
    endpoint: "/v1/responses",
    completion_window: "24h",
    metadata: { job: "nightly" },
  });
  const retrieved = await client.batches.retrieve(created.id);
  const page = await client.batches.list({ after: "batch-0", limit: 2 });
  const cancelled = await client.batches.cancel(created.id);

  assert.equal(created.id, "batch-1");
  assert.equal(created.status, "validating");
  assert.equal(retrieved.status, "completed");
  assert.equal(page.object, "list");
  assert.deepEqual(page.data.map((item) => item.id), ["batch-1", "batch-2"]);
  assert.equal(cancelled.status, "cancelling");
  assert.deepEqual(provider.batchCalls, [
    ["create", {
      input_file_id: "file-input",
      endpoint: "/v1/responses",
      completion_window: "24h",
      metadata: { job: "nightly" },
    }],
    ["retrieve", "batch-1"],
    ["list", { after: "batch-0", limit: 2 }],
    ["cancel", "batch-1"],
  ]);
});

test("agent-rt/anthropic exposes Message Batches lifecycle and async results", async () => {
  const provider = new BatchMigrationProvider();
  const client = new Anthropic({ provider });
  const requests = [
    {
      custom_id: "request-1",
      params: {
        model: "test-model",
        max_tokens: 32,
        messages: [{ role: "user", content: "Summarize." }],
      },
    },
  ];

  const created = await client.messages.batches.create({ requests });
  const retrieved = await client.messages.batches.retrieve(created.id);
  const page = await client.messages.batches.list({ limit: 20 });
  const cancelled = await client.messages.batches.cancel(created.id);
  const results = await client.messages.batches.results(created.id);
  const items = [];
  for await (const item of results) items.push(item);

  assert.equal(created.id, "batch-1");
  assert.equal(retrieved.status, "completed");
  assert.equal(page.has_more, false);
  assert.equal(page.first_id, "batch-1");
  assert.equal(page.last_id, "batch-2");
  assert.equal(cancelled.status, "cancelling");
  assert.deepEqual(items.map((item) => item.custom_id), ["request-1", "request-2"]);
  assert.deepEqual(provider.batchCalls, [
    ["create", { requests }],
    ["retrieve", "batch-1"],
    ["list", { limit: 20 }],
    ["cancel", "batch-1"],
    ["results", "batch-1"],
  ]);
});

test("batch migration validates provider-specific creation inputs", async () => {
  const provider = new BatchMigrationProvider();
  const openai = new OpenAI({ provider });
  await assert.rejects(
    openai.batches.create({ endpoint: "/v1/responses", completion_window: "24h" }),
    /requires input_file_id/,
  );
  const anthropic = new Anthropic({ provider });
  await assert.rejects(
    anthropic.messages.batches.create({}),
    /requires requests array/,
  );
  assert.deepEqual(provider.batchCalls, []);
});

class WebSearchExecutionProvider {
  constructor() {
    this.name = "web-search-fake";
    this.requests = [];
  }

  async complete(request) {
    this.requests.push(request);
    if (this.requests.length === 1) {
      return {
        message: {
          role: "assistant",
          content: [],
          toolCalls: [{
            id: "web-search-1",
            name: request.tools[0].name,
            arguments: { query: "Agent RT latest" },
          }],
        },
        model: request.model,
        finishReason: "tool_calls",
      };
    }
    return {
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Search complete." }],
      },
      model: request.model,
      finishReason: "stop",
    };
  }
}

class FakeWebRetrieval {
  constructor() {
    this.kind = "web";
    this.queries = [];
  }

  async search(query) {
    this.queries.push(query);
    return [
      {
        id: "https://example.com/a",
        title: "Result A",
        content: "A result",
        score: 0.95,
        uri: "https://example.com/a",
        metadata: { provider: "fake-web" },
      },
      {
        id: "https://example.com/b",
        title: "Result B",
        content: "B result",
        score: 0.8,
        uri: "https://example.com/b",
        metadata: { provider: "fake-web" },
      },
    ];
  }
}

test("agent-rt/openai routes web_search through Agent RT retrieval provider", async () => {
  const provider = new WebSearchExecutionProvider();
  const web = new FakeWebRetrieval();
  const client = new OpenAI({ provider, webSearchProvider: web });

  const response = await client.responses.create({
    model: "test-model",
    input: "Search the web.",
    tools: [{ type: "web_search", search_context_size: "medium", max_num_results: 2 }],
  });

  assert.equal(provider.requests[0].tools[0].name, "agent_rt_web_search");
  assert.equal(web.queries[0].text, "Agent RT latest");
  assert.equal(web.queries[0].limit, 2);
  assert.deepEqual(web.queries[0].filters, { search_context_size: "medium" });
  assert.match(provider.requests[1].messages.at(-1).content[0].text, /https:\/\/example\.com\/a/);
  const call = response.output[0];
  assert.equal(call.type, "web_search_call");
  assert.equal(call.status, "completed");
  assert.equal(call.action.type, "search");
  assert.equal(call.action.query, "Agent RT latest");
  assert.equal(call.action.sources[0].url, "https://example.com/a");
  assert.equal(call.results[0].score, 0.95);
  assert.equal(response.output_text, "Search complete.");
});

test("agent-rt/anthropic routes web search server tool through Agent RT", async () => {
  const provider = new WebSearchExecutionProvider();
  const web = new FakeWebRetrieval();
  const client = new Anthropic({ provider, webSearchProvider: web });

  const response = await client.messages.create({
    model: "test-model",
    max_tokens: 256,
    messages: [{ role: "user", content: "Search the web." }],
    tools: [{
      type: "web_search_20250305",
      name: "web_search",
      max_uses: 3,
      allowed_domains: ["example.com"],
    }],
  });

  assert.equal(provider.requests[0].tools[0].name, "agent_rt_web_search");
  assert.deepEqual(web.queries[0].filters, { allowed_domains: ["example.com"] });
  assert.equal(response.content[0].type, "server_tool_use");
  assert.equal(response.content[0].name, "web_search");
  assert.equal(response.content[1].type, "web_search_tool_result");
  assert.equal(response.content[1].content[0].url, "https://example.com/a");
  assert.equal(response.content.at(-1).text, "Search complete.");
});

test("web_search resolves environment configuration and fails closed before provider invocation", async () => {
  const keys = [
    "AGENT_RT_WEB_SEARCH_TOOL",
    "TAVILY_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "SERPER_API_KEY",
    "AGENT_RT_WEB_SEARCH_API_KEY",
    "AGENT_RT_WEB_SEARCH_TOKEN",
  ];
  const saved = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  try {
    for (const key of keys) delete process.env[key];
    const missingProvider = new CaptureProvider();
    const missingClient = new OpenAI({ provider: missingProvider });
    await assert.rejects(
      missingClient.responses.create({
        model: "test-model",
        input: "Search.",
        tools: [{ type: "web_search" }],
      }),
      /AGENT_RT_WEB_SEARCH_TOOL/,
    );
    assert.equal(missingProvider.requests.length, 0);

    process.env.AGENT_RT_WEB_SEARCH_TOOL = "tavily";
    process.env.AGENT_RT_WEB_SEARCH_TOKEN = "generic-secret";
    const configuredProvider = new CaptureProvider();
    const configuredClient = new OpenAI({ provider: configuredProvider });
    await configuredClient.responses.create({
      model: "test-model",
      input: "Search.",
      tools: [{ type: "web_search_preview" }],
    });
    assert.equal(configuredProvider.requests[0].tools[0].name, "agent_rt_web_search");
  } finally {
    for (const key of keys) {
      const value = saved[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});


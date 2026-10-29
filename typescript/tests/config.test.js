const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  ConfigError,
  ConfigNotFoundError,
  ConfiguredModelProvider,
  RetrievalRegistry,
  ToolRegistry,
  configFromObject,
  configSearchPaths,
  configServerOptions,
  configureRuntime,
  discoverConfigPath,
  loadConfig,
  loadModel,
  parseConfig,
  registerFileSearchTool,
  resolveVectorStores,
  runtimeSettings,
  vectorStoreRegistryFromConfig,
} = require("../dist/index.js");

const CONFIG = `
model_list:
  - model_name: claude-sonnet-5
    litellm_params:
      model: anthropic/claude-sonnet-5
      api_key: os.environ/TEST_ANTHROPIC_KEY
  - model_name: claude-sonnet-5
    agent_rt_params:
      model: openai/gpt-backup
      api_key: os.environ/TEST_OPENAI_KEY
  - model_name: embed-small
    agent_rt_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/TEST_OPENAI_KEY

vector_store_registry:
  - vector_store_name: docs-kb
    litellm_params:
      vector_store_id: T37J8R4WTM
      custom_llm_provider: qdrant
      api_base: https://qdrant.example.test
      collection: docs
      embedding_model: embed-small
      vector_store_metadata:
        source: https://example.test/docs
  - vector_store_name: private-kb
    agent_rt_params:
      vector_store_id: PRIVATE1
      custom_llm_provider: chroma
      api_base: https://chroma.example.test
      collection: private
      allowed_models: [embed-small]

agent_rt_settings:
  default_model: claude-sonnet-5
  execution_limits:
    timeout_seconds: 7
  sandbox:
    allowed_domains: [a.example.test, b.example.test]
    allow_http: false
`;
const ENVIRONMENT = { TEST_ANTHROPIC_KEY: "ak", TEST_OPENAI_KEY: "ok" };
const parse = (text = CONFIG, environment = ENVIRONMENT) => parseConfig(text, { environment });

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "agent-rt-config-"));
}

test("explicit config path wins and must exist", () => {
  const dir = tempDir();
  const explicit = path.join(dir, "custom.yaml");
  fs.writeFileSync(explicit, "model_list: []\n");
  fs.writeFileSync(path.join(dir, "agent-rt.yaml"), "model_list: []\n");
  assert.equal(discoverConfigPath(explicit, { cwd: dir, home: dir }), explicit);
  assert.throws(() => discoverConfigPath(path.join(dir, "nope.yaml")), /not found/);
});

test("project config requires explicit selection", () => {
  const cwd = tempDir();
  const home = tempDir();
  fs.mkdirSync(path.join(home, ".agent-rt"));
  const homeFile = path.join(home, ".agent-rt", "config.yaml");
  fs.writeFileSync(homeFile, "{}");
  assert.equal(discoverConfigPath(undefined, { cwd, home }), homeFile);
  const local = path.join(cwd, "agent-rt.yaml");
  fs.writeFileSync(local, "{}");
  assert.equal(discoverConfigPath(undefined, { cwd, home }), homeFile);
  assert.equal(discoverConfigPath(local, { cwd, home }), local);
});

test("missing default config lists the searched locations", () => {
  const cwd = tempDir();
  const home = path.join(tempDir(), "h");
  assert.throws(
    () => discoverConfigPath(undefined, { cwd, home }),
    (error) =>
      error instanceof ConfigNotFoundError &&
      configSearchPaths({ cwd, home }).every((candidate) => error.message.includes(candidate)),
  );
});

test("loadConfig reads a file and records its source", async () => {
  const dir = tempDir();
  const file = path.join(dir, "c.yaml");
  fs.writeFileSync(file, CONFIG);
  const config = await loadConfig(file, { environment: ENVIRONMENT });
  assert.equal(config.source, file);
  assert.equal(config.models.length, 3);
});

test("secrets resolve from os.environ references", async () => {
  const config = await parse();
  assert.equal(config.models[0].apiKey, "ak");
  assert.equal(config.models[1].apiKey, "ok");
});

test("unset model and server secrets fail only when used", async () => {
  const config = await parse(CONFIG, { TEST_OPENAI_KEY: "ok" });
  const openai = new Recording("openai");
  const provider = await loadModel(config, { providerFactory: () => openai });
  const response = await provider.complete(userRequest("claude-sonnet-5"));
  assert.equal(response.message.content[0].text, "openai");
  await assert.rejects(
    new ConfiguredModelProvider(config, () => openai).validate(),
    /model_list\[0\]\.litellm_params\.api_key.*TEST_ANTHROPIC_KEY/,
  );
  await assert.rejects(
    parse("agent_rt_settings:\n  web_search: {api_key: os.environ/NOPE}\n", {}),
    /web_search\.api_key.*NOPE/,
  );
  const served = await parse("agent_rt_settings:\n  server: {api_keys: [os.environ/NOPE]}\n", {});
  assert.throws(() => configServerOptions(served), /server\.api_keys\[0\].*NOPE/);
  // serve bot tokens are deferred like server secrets; only known platforms and keys load.
  await parse("agent_rt_settings:\n  serve: {telegram: {bot_token: os.environ/NOPE, allowed_users: [7]}}\n", {});
  await assert.rejects(parse("agent_rt_settings:\n  serve: {slack: {bot_token: x}}\n", {}), /unknown key/);
  await assert.rejects(parse("agent_rt_settings:\n  serve: {discord: {token: x}}\n", {}), /serve\.discord/);
});

test("a vector store name may equal its id", async () => {
  const config = await parse(CONFIG.replace("vector_store_name: docs-kb", "vector_store_name: T37J8R4WTM"));
  assert.equal(config.vectorStores[0].name, "T37J8R4WTM");
});

test("YAML 1.1 scalars parse like PyYAML", async () => {
  const config = await parse("agent_rt_settings:\n  sandbox: {allow_http: yes, uid: 0123}\n");
  assert.equal(config.settings.sandbox.allow_http, true);
  assert.equal(config.settings.sandbox.uid, 83);
});

test("OpenAI Agents compat reports an invalid config instead of falling back", async () => {
  const { Agent, run } = require("../dist/ext/compat/openai_agents.js");
  const dir = tempDir();
  fs.mkdirSync(path.join(dir, ".agent-rt"));
  fs.writeFileSync(path.join(dir, ".agent-rt", "config.yaml"), "surprise: 1\n");
  const previous = { cwd: process.cwd(), home: process.env.HOME };
  process.chdir(dir);
  process.env.HOME = dir;
  try {
    await assert.rejects(run(new Agent({ name: "a" }), "hi"), /unknown key/);
  } finally {
    process.chdir(previous.cwd);
    process.env.HOME = previous.home;
  }
});

test("OpenAI Agents compat falls back to SDK env for a model the config does not declare", async () => {
  const { Agent, run } = require("../dist/ext/compat/openai_agents.js");
  const dir = tempDir();
  fs.mkdirSync(path.join(dir, ".agent-rt"));
  fs.writeFileSync(
    path.join(dir, ".agent-rt", "config.yaml"),
    "model_list:\n  - model_name: only-this\n    agent_rt_params: {model: openai/x, api_key: k}\n",
  );
  const previous = { cwd: process.cwd(), env: { ...process.env } };
  process.chdir(dir);
  process.env.HOME = dir;
  // Closed loopback port: reaching it proves the SDK-env provider was chosen.
  process.env.OPENAI_BASE_URL = "http://127.0.0.1:9/v1";
  process.env.OPENAI_API_KEY = "sk-test";
  try {
    await assert.rejects(run(new Agent({ name: "a", model: "gpt-4o-mini" }), "hi"), (error) => {
      assert.doesNotMatch(String(error), /unknown model/);
      return true;
    });
  } finally {
    process.chdir(previous.cwd);
    for (const key of Object.keys(process.env)) if (!(key in previous.env)) delete process.env[key];
    Object.assign(process.env, previous.env);
  }
});

test("config validation rejects unknown keys, providers and unsafe YAML", async () => {
  await assert.rejects(parse("surprise: 1\n"), /unknown key/);
  await assert.rejects(
    parse("model_list:\n  - model_name: x\n    agent_rt_params: {model: bedrock/x}\n"),
    /unsupported provider/,
  );
  await assert.rejects(parse("model_list:\n  - model_name: x\n    agent_rt_params: {model: plain}\n"), /<provider>\/<model>/);
  await assert.rejects(parse("model_list: !!python/object/apply:os.system ['true']\n"), /invalid YAML/);
  await assert.rejects(parse(CONFIG.replace("embedding_model: embed-small", "embedding_model: missing")), /not in model_list/);
  await assert.rejects(parse(CONFIG.replace("PRIVATE1", "T37J8R4WTM")), /duplicate vector store/);
  await assert.rejects(parse(CONFIG.replace("default_model: claude-sonnet-5", "default_model: nope")), /default_model/);
});

test("configureRuntime feeds implicit defaults from settings only", async () => {
  try {
    configureRuntime(await parse());
    assert.equal(runtimeSettings().AGENT_RT_EXECUTION_TIMEOUT_SECONDS, "7");
    assert.equal(runtimeSettings().AGENT_RT_SANDBOX_ALLOWED_DOMAINS, "a.example.test,b.example.test");
    assert.equal(runtimeSettings().AGENT_RT_SANDBOX_ALLOW_HTTP, "false");
  } finally {
    configureRuntime(configFromObject({}));
  }
});

class Recording {
  constructor(label, fail = false) {
    this.name = label;
    this.label = label;
    this.fail = fail;
    this.requests = [];
  }
  async complete(request) {
    this.requests.push(request);
    if (this.fail) throw new Error(`${this.label} down`);
    return { message: { role: "assistant", content: [{ type: "text", text: this.label }] } };
  }
  async *stream(request) {
    yield { type: "completed", response: await this.complete(request) };
  }
  async embed(request) {
    this.requests.push(request);
    return { data: [{ index: 0, embedding: [0.1, 0.2] }] };
  }
}

const userRequest = (model) => ({
  messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
  model,
});

async function makeProvider(anthropicDown = false) {
  const providers = { anthropic: new Recording("anthropic", anthropicDown), openai: new Recording("openai") };
  const provider = await loadModel(await parse(), { providerFactory: (d) => providers[d.provider] });
  return { provider, providers };
}

test("model routing uses the public name and rewrites the underlying model", async () => {
  const { provider, providers } = await makeProvider();
  const response = await provider.complete(userRequest("claude-sonnet-5"));
  assert.equal(response.message.content[0].text, "anthropic");
  assert.equal(providers.anthropic.requests[0].model, "claude-sonnet-5");
  assert.equal(providers.openai.requests.length, 0);
  await provider.complete(userRequest(undefined));
  assert.equal(providers.anthropic.requests.length, 2);
});

test("duplicate model names fall back in order, including streaming", async () => {
  const { provider, providers } = await makeProvider(true);
  const response = await provider.complete(userRequest("claude-sonnet-5"));
  assert.equal(response.message.content[0].text, "openai");
  assert.equal(providers.openai.requests[0].model, "gpt-backup");
  const events = [];
  for await (const event of provider.stream(userRequest("claude-sonnet-5"))) events.push(event);
  assert.equal(events.at(-1).response.message.content[0].text, "openai");
});

test("unknown models list the configured ones; embeddings and listing work", async () => {
  const { provider, providers } = await makeProvider();
  await assert.rejects(provider.complete(userRequest("nope")), /configured models: claude-sonnet-5, embed-small/);
  await provider.embed({ input: "x", model: "embed-small" });
  assert.equal(providers.openai.requests[0].model, "text-embedding-3-small");
  assert.deepEqual((await provider.listModels()).map((entry) => entry.id), ["claude-sonnet-5", "embed-small"]);
  assert.equal(provider.settings.defaultModel, "claude-sonnet-5");
});

test("default factory builds real providers lazily", async () => {
  const provider = await loadModel(await parse());
  assert.ok(provider instanceof ConfiguredModelProvider);
  assert.equal(provider.provider(provider.config.models[0]).name, "anthropic");
  const openai = provider.provider(provider.config.models[1]);
  assert.equal(openai.settings.defaultModel, "gpt-backup");
  await assert.rejects(loadModel(await parse("{}")), /at least one model/);
});

class Store {
  constructor(label, score) {
    this.kind = "knowledge";
    this.label = label;
    this.score = score;
    this.queries = [];
  }
  async search(query) {
    this.queries.push(query);
    return [{ id: `${this.label}-1`, title: this.label, content: `passage from ${this.label}`, score: this.score }];
  }
}

test("vector stores register under name and id with a pinned embedding model", async () => {
  const config = await parse();
  const embedder = new Recording("emb");
  const registry = vectorStoreRegistryFromConfig(config, { embeddingProvider: embedder });
  assert.ok(registry instanceof RetrievalRegistry);
  for (const name of ["docs-kb", "T37J8R4WTM", "private-kb", "PRIVATE1"]) {
    assert.ok(registry.providers.has(name), name);
  }
  assert.equal(registry.providers.get("docs-kb").embeddingProvider.modelName, "embed-small");
});

test("resolveVectorStores rejects unknown and disallowed stores", async () => {
  const config = await parse();
  assert.deepEqual(
    resolveVectorStores(config, ["docs-kb", "T37J8R4WTM"]).map((store) => store.name),
    ["docs-kb"],
  );
  assert.throws(() => resolveVectorStores(config, ["missing"]), /unknown vector store/);
  assert.throws(() => resolveVectorStores(config, ["PRIVATE1"], { modelName: "claude-sonnet-5" }), /not available to this model/);
  assert.equal(resolveVectorStores(config, ["PRIVATE1"], { modelName: "embed-small" }).length, 1);
  assert.throws(() => resolveVectorStores(config, []), /at least one/);
});

test("file_search merges stores by score and only queries the allowed ones", async () => {
  const config = await parse();
  const registry = new RetrievalRegistry();
  const docs = new Store("docs", 0.4);
  const priv = new Store("private", 0.9);
  const other = new Store("other", 1);
  registry.register("docs-kb", docs);
  registry.register("private-kb", priv);
  registry.register("other-kb", other);
  const tools = new ToolRegistry();
  registerFileSearchTool(tools, registry, config.vectorStores, { maxResults: 2 });
  const output = await tools.execute({ id: "1", name: "file_search", arguments: { query: "what?" } });
  assert.deepEqual(output.map((row) => row.vector_store), ["private-kb", "docs-kb"]);
  assert.equal(other.queries.length, 0);
});

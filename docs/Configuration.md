# Configuration

Agent RT is configured with one YAML file, modelled on LiteLLM's `config.yaml`. Provider selection, models, vector stores, sandbox, web search, execution limits, Decision models, observability, and server settings all live there. The process environment is **not** a configuration source: it is read only to resolve secrets written as `os.environ/NAME`.

A complete annotated example is [`agent-rt.example.yaml`](../agent-rt.example.yaml).

## Finding the file

| Order | Location |
| --- | --- |
| 1 | An explicit path: `agent-rt --config PATH`, `load_config(path)` (Python), `loadConfig(path)` (TypeScript) |
| 2 | `./agent-rt.yaml`, then `./.agent-rt/config.yaml` (current folder) |
| 3 | `~/.agent-rt/config.yaml`, then `~/.config/agent-rt/config.yaml` (home folder) |

`.yml` is accepted wherever `.yaml` is. The first file found wins; files are never merged. An explicit path that does not exist is an error, and when nothing is found the error lists every location that was checked. The `agent-rt` command, run without `--config`, instead creates an owner-only starter config at `~/.agent-rt/config.yaml` (its `model_list` is commented out); the library keeps the error. Neither `agent-rt` nor `agent-rt web` needs a `model_list`: with no config, or one without `model_list` (even a blank file), they use OpenAI with `os.environ/OPENAI_API_KEY` (plus `os.environ/OPENAI_BASE_URL` when that variable is set, as the OpenAI SDK does) and offers the models its `/v1/models` lists, starting with the first (or `--model`). A bare `./config.yaml` is deliberately not searched (too easy to pick up another tool's file); pass it with `--config config.yaml`.

## `model_list`

```yaml
model_list:
  - model_name: claude-sonnet-5            # the name clients request
    agent_rt_params:                       # litellm_params is accepted as an alias
      model: anthropic/claude-sonnet-5     # <provider>/<model>
      api_key: os.environ/ANTHROPIC_API_KEY
      api_base: https://api.anthropic.com  # optional (also for OpenAI-compatible endpoints)
      websocket: false                     # optional, OpenAI only
    model_info: {context_window: 200000}   # optional metadata
```

- Providers are `openai` and `anthropic`. An OpenAI-compatible endpoint is `openai/<model>` plus `api_base` (WebSocket transport is then off unless `websocket: true`). Any other provider prefix is rejected when the file loads.
- **Several models:** list as many as needed. Clients choose one by `model_name`; the CLI uses `-m`, the API server uses the request's `model` field.
- **Fallbacks:** repeating a `model_name` adds an ordered fallback deployment. The next deployment is tried when one raises before producing output (streams fall back only before the first event).
- **Embeddings:** an embedding model is just another entry; vector stores reference it by `model_name`.
- Provider clients are created lazily, one per deployment.

## `vector_store_registry`

```yaml
vector_store_registry:
  - vector_store_name: docs-kb
    agent_rt_params:
      vector_store_id: T37J8R4WTM          # what clients pass as vector_store_ids
      custom_llm_provider: qdrant          # chroma | milvus | pinecone | qdrant | weaviate
      api_base: https://qdrant.internal:6333
      collection: docs
      api_key: os.environ/QDRANT_API_KEY
      embedding_model: embed-small         # a model_name above
      timeout: 20                          # optional request timeout in seconds
      options: {}                          # non-secret adapter options
      vector_store_description: Product documentation
      vector_store_metadata: {source: https://example.com/docs}
      allowed_models: [claude-sonnet-5]    # optional access list
```

`custom_llm_provider` supports the built-in HTTP adapters (`chroma`, `milvus`, `pinecone`, `qdrant`, `weaviate`) and AWS Bedrock Knowledge Bases (`bedrock`). HTTP adapters require `api_base` and `collection`; Bedrock uses `vector_store_id` as the Knowledge Base ID and needs neither. Other hosted providers (`openai`, `azure`, `vertex_ai`) are not yet supported. Store credentials and client creation are deferred until the store is searched, so unused stores with missing credentials do not prevent server startup.

A Bedrock entry compatible with LiteLLM's configuration shape:

```yaml
vector_store_registry:
  - vector_store_name: bedrock-litellm-website-knowledgebase
    litellm_params:
      vector_store_id: T37J8R4WTM
      custom_llm_provider: bedrock
      vector_store_description: Bedrock documentation knowledge base
      vector_store_metadata: {source: https://www.litellm.com/docs}
      options: {region: us-west-2}  # optional; AWS SDK region resolution also works
```

Bedrock retrieval uses AWS's **Retrieve** API with IAM authentication from the AWS SDK's standard credential chain (not an Agent RT environment configuration setting). Install the optional client in Python with `pip install 'agent-rt[bedrock]'` or in TypeScript with `npm install @aws-sdk/client-bedrock-agent-runtime`. The AWS region can be supplied through `options.region`; otherwise, the SDK's normal region resolution applies. Both runtimes support injecting Bedrock clients for offline tests (`bedrock_client_factory` / `bedrockClientFactory`).

Every store is registered in a `RetrievalRegistry` under both its name and its `vector_store_id`; names and ids must be unique. Building the registry (`vector_store_registry_from_config` / `vectorStoreRegistryFromConfig`) pins HTTP adapters' embeddings to their `embedding_model`. Bedrock performs hosted retrieval and does not require a local embedding model.

### Passing vector stores to a model (`file_search`)

The API server (Python `agent_rt_api`) accepts the OpenAI-style tool on Chat Completions and Responses requests:

```python
client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "What is litellm?"}],
    tools=[{"type": "file_search", "vector_store_ids": ["T37J8R4WTM"], "max_num_results": 5}],
)
```

- `file_search` is the only request-defined tool accepted; any other `type` is rejected, and the Anthropic Messages endpoint still rejects request-defined tools.
- The runtime gives the model a request-scoped `file_search(query)` tool limited to the listed stores. Results are merged by score and returned as **untrusted tool output**, so the existing prompt-injection and output guardrails apply.
- Unknown store ids, and stores whose `allowed_models` excludes the requested model, are rejected with a 400 before the run starts.
- Outside the server, call `register_file_search_tool(registry, retrieval, stores)` / `registerFileSearchTool(...)` on your own tool registry, using `resolve_vector_stores` / `resolveVectorStores` to validate ids.

## `agent_rt_settings`

| Section | Keys |
| --- | --- |
| `default_model` | A `model_name`; defaults to the first entry. `agent-rt` and `agent-rt web` select it at startup. |
| `server` | `host`, `port`, `api_keys`, `default_agent`, `allowed_websocket_origins`, `max_concurrent_requests`, `max_queued_requests` |
| `execution_limits` | `timeout_seconds`, `memory_bytes`, `cpu_seconds` (defaults for every tool call and sandbox session) |
| `sandbox` | `backend`, `workspace_root`, `native_root`, `docker_image`, `docker_binary`, `docker_user`, `docker_read_only`, `e2b_template`, `microsandbox_image`, `swerex_url`, `swerex_api_key`, `uid`, `gid`, `network_mode`, `network_max_transfer_bytes`, `network_max_bytes_per_second`, `allowed_domains`, `blocked_domains`, `allow_http`, `allow_ip`, `allow_proxy`, `allow_websocket`, `setpriv_binary`, `prlimit_binary` |
| `web_search` | `tool` (`tavily`, `brave`, `serper`), `api_key`, `timeout_seconds` |
| `decision_model` | `provider` (`jev`, the default, or `openai` for the OpenAI Decisions API), `base_url`, `model`, `timeout_seconds`, `api_key`. With `provider: openai`, `base_url` defaults to `https://api.openai.com/v1` and `model` to `gpt-6-luna`; `api_key` is required. |
| `observability` | `api_key`, `base_url`, `headers`, `trace_content`, `telemetry_enabled`, `enrich_tokens`, `app_name` (OpenLLMetry/Traceloop) |
| `memory` | `backend` (`mem0`, the default, `memu`, `zep`, `letta`, `supermemory`) and `options` for that backend; see [Memory packages](Runtime-Guide.md#memory-packages). Used by `load_memory(config)` / `loadMemory(config)`. |
| `cli` | `home` (sessions and skills root, default `~/.agent-rt`), `compact_token_limit` (`agent-rt web` uses both too: its sessions share `home`/sessions with the terminal CLI) |

Lists are written as YAML lists; `headers` is a mapping. Unknown sections and keys fail at load time. `configure_runtime(config)` / `configureRuntime(config)` applies these settings to the runtime's implicit defaults (and starts OpenLLMetry when configured); the CLI and `agent_rt_api.build_from_config` call it for you. Settings that are passed explicitly to a constructor always win.

## Secrets and validation

- A value that is exactly `os.environ/NAME` is replaced by that environment variable when the file loads. A missing or blank variable fails with the YAML path of the key (for example `model_list[0].agent_rt_params.api_key`); secret values are never included in errors or `repr`. For a model's or vector store's `api_key`/`api_base` and anything under `agent_rt_settings.server` or `agent_rt_settings.memory.options`, the failure is deferred until that deployment, store or server is used, so one file can list providers you have no key for: a deployment with an unset key is skipped in favour of the next fallback.
- Literal keys in the file also work but are discouraged.
- YAML is parsed as YAML 1.1 with safe types only, identically in both implementations (as PyYAML's `safe_load`: `yes`/`no` are booleans, `0123` is octal); custom tags such as `!!python/object` are rejected. Quote values such as IDs that must stay strings.
- When no file is found, loading raises `ConfigNotFoundError` (a `ConfigError`). The OpenAI Agents compatibility layer uses the config only for an agent whose model is a declared `model_name`; when no file is found, or the agent asks for a model the file does not declare (including the upstream default `gpt-4o-mini`), it falls back to the upstream SDK's environment variables. An invalid file is always reported.
- Files over 1 MB are rejected. Unknown keys, wrong types, duplicate vector store names/ids, unknown `default_model`, `embedding_model` or `allowed_models` references, and unsupported providers all fail when the file loads, with the YAML path in the message.

## Using it from code

```python
from agent_rt import load_config, load_model, configure_runtime

config = load_config()               # or load_config("/etc/agent-rt/prod.yaml")
configure_runtime(config)
provider = load_model(config)        # routes by model_name, with fallbacks
```

```ts
import { loadConfig, loadModel, configureRuntime } from "agent-rt";

const config = await loadConfig(); // or loadConfig("/etc/agent-rt/prod.yaml")
configureRuntime(config);
const provider = await loadModel(config);
```

Serve every configured model (Python, `agent-rt[api]`):

```bash
python -m agent_rt_api --config agent-rt.yaml
```

or call `agent_rt_api.serve_from_config(config_path=...)`. `build_from_config` returns the loop, one agent per `model_name`, the retrieval registry and server options if you want to assemble the app yourself.

## Compatibility adapters

The migration adapters (`agent_rt.openai`, `agent_rt.anthropic`, LangChain, LlamaIndex, OpenAI Agents) deliberately keep the upstream SDKs' environment behaviour (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, ...) so migrated code works unchanged. The CLI tool modes (`agent-rt claude`, `agent-rt codex`, `agent-rt opencode`, `agent-rt copilot`) likewise read that tool's own environment variables and config files instead of `agent-rt.yaml`; see [Runtime Guide](Runtime-Guide.md#reuse-another-agent-clis-setup). Agent RT's own configuration is only the YAML file.

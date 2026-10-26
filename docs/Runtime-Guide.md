# Runtime guide

Detailed usage notes for features that the package READMEs only mention briefly. Examples are shown for Python unless a section says otherwise; the TypeScript package uses the same concepts with camelCase names. For design rationale and contracts see [Architecture and Extensions](Architecture-and-Extensions.md).

## Providers

`load_model(load_config())` (Python) / `loadModel(await loadConfig())` (TypeScript) builds a provider from the `model_list` of a YAML config file (`agent-rt.yaml` in the current folder, `~/.agent-rt/config.yaml`, or an explicit path). It supports several models, ordered fallbacks for repeated `model_name`s, and several vector stores; see [Configuration](Configuration.md). The process environment is only used to resolve `os.environ/NAME` secret references.

The OpenAI provider uses the native OpenAI-compatible transport by default and supports HTTP/SSE plus Responses WebSocket operation where available. The Anthropic provider uses the optional official Anthropic SDK. Reasoning/thinking controls are normalized into the provider-neutral `ReasoningConfig`, and each provider translates them into vendor-specific request fields.

## Content and provider behaviour

- **Media:** image, PDF/file, and (OpenAI) audio `ContentPart`s are sent as the provider's native blocks; `data` may be an http(s) URL, a `data:` URL, base64, or bytes. Parts a provider cannot carry (for example video) raise instead of being dropped.
- **Anthropic thinking:** thinking blocks are kept on the assistant message (a JSON part with a vendor MIME type) and replayed before `tool_use`, so extended thinking works across tool turns. Anthropic requests default to `max_tokens=4096`; set `max_output_tokens` (`maxOutputTokens` in TypeScript) for longer outputs.
- **Context trust:** untrusted retrieved/file/observation items reach the model as user-role data; only workflow state and items explicitly marked trusted are system messages.
- **Validation:** tool arguments and structured output are checked against a JSON Schema subset (types, bounds, `pattern`, `const`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref`, `additionalProperties`).
- **Truncation:** a Responses WebSocket `response.incomplete` returns `finish_reason="length"` (`finishReason: "length"` in TypeScript) with usage rather than raising.

## Agent loop guarantees

The `AgentLoop` keeps every tool call answered: if a run stops early (approval needed, timeout, cancellation, `max_tool_calls`) unexecuted calls receive a `tool_not_executed` tool message, completed results from a concurrent batch are kept, and loop-detector state is per run.

## Agent memory

Give an agent memory with `AgentConfig.memory`, a list of sources. Each run resolves every source and adds the results to the model context as `retrieved` context items (ids `memory:<name>:<n>`, metadata `memory_source`), so context policies, offloading and prompt-injection defenses apply to them as to any retrieved data.

| Source | What the model sees | Default trust |
| --- | --- | --- |
| A string | The text | trusted |
| Any custom object (dataclass, pydantic model, dict, class instance; `Set`/`Map` in TypeScript) | Its fields as JSON | trusted |
| A memory store or retriever: Agent RT's `InMemoryLongTermMemoryStore`, or one from LangChain, LangGraph, LlamaIndex, AutoGen, OpenAI Agents (`Session`) or mem0 ([details](MIGRATION.md#memory-from-other-frameworks)) | The results for the latest user prompt | untrusted |
| A function `(query) -> value or list` (may be async) | Its return values, each handled as above | untrusted |

Wrap a source in `MemorySource` to set its `name`, `trust`, `limit` (default 5 results for stores and functions), `scope` (passed to an Agent RT store's search) or `search_kwargs` / `searchOptions` (passed to another framework's search, such as a mem0 `user_id` or a LangGraph `namespace`). Store and function results default to untrusted because they often hold user- or tool-written text; untrusted memory is sent as user-role data and, like other untrusted context, hides side-effecting tools unless `PromptInjectionDefense(allow_untrusted_side_effects=True)` is set. Mark a source `trust="trusted"` only when you control what is written to it.

```python
from agent_rt import AgentConfig, InMemoryLongTermMemoryStore, MemorySource, ModelSettings

agent = AgentConfig(
    name="assistant",
    instructions="Be concise.",
    model=ModelSettings(model="claude-sonnet-5"),
    memory=[
        "The user prefers metric units.",
        user_profile,                                   # any custom object
        MemorySource(store, name="facts", limit=3),     # searched every run
        MemorySource(fetch_recent_orders, name="orders"),  # (query) -> values
    ],
)
```

```ts
const agent = {
  name: "assistant",
  instructions: "Be concise.",
  model: { model: "claude-sonnet-5" },
  memory: [
    "The user prefers metric units.",
    userProfile,
    new MemorySource(store, { name: "facts", limit: 3 }),
    new MemorySource(fetchRecentOrders, { name: "orders" }),
  ],
};
```

`resolve_agent_memory(memory, query)` / `resolveAgentMemory(memory, query)` return the same context items without running the loop.

### Memory packages

`load_memory()` / `loadMemory()` connect a dedicated memory package and return a `MemoryBackend`: put it in `AgentConfig.memory` to recall memories for every run, and call `remember(messages)` after a run to store the conversation (only user and assistant turns are sent; system prompts and tool output are dropped).

| Backend | Python install | TypeScript install | Options |
| --- | --- | --- | --- |
| `mem0` (default) | `pip install 'agent-rt[memory]'` or `'agent-rt[mem0]'` | `npm install mem0ai` | `user_id` (default `default`), `agent_id`, `run_id`; `config` for a self-hosted `Memory` config, or `api_key` (+ `host`) for the hosted platform |
| `memu` | `pip install 'agent-rt[memu]'` (Python 3.13+) | — (Python-only) | `user_id`, plus memU's `llm_profiles`, `database_config`, `blob_config`, `memorize_config`, `retrieve_config`, `user_config` |
| `zep` | `pip install 'agent-rt[zep]'` | `npm install @getzep/zep-cloud` | `api_key`, `base_url`, `graph_id` (searched), `thread_id` (written) |
| `letta` | `pip install 'agent-rt[letta]'` | `npm install @letta-ai/letta-client` | `api_key`, `base_url`, `agent_id` (archival passages) |
| `supermemory` | `pip install 'agent-rt[supermemory]'` | `npm install supermemory` | `api_key`, `base_url`, `namespace` (default `default`) |

```python
from agent_rt import AgentConfig, AgentLoop, ModelSettings, load_memory

memory = load_memory(options={"user_id": "alice"})          # mem0
# memory = load_memory(backend="zep", options={...})       # or any backend above
# memory = load_memory(config)                             # agent_rt_settings.memory

agent = AgentConfig(name="assistant", instructions="Be concise.",
                    model=ModelSettings(model="claude-sonnet-5"), memory=[memory])
result = await AgentLoop(provider).run(agent, messages)
await memory.remember(result.messages)
```

```ts
const memory = await loadMemory(undefined, { options: { user_id: 'alice' } });
const agent = { name: 'assistant', instructions: 'Be concise.', model: { model: 'claude-sonnet-5' }, memory: [memory] };
const result = await new AgentLoop(provider).run(agent, messages);
await memory.remember(result.messages);
```

The backend can also be set once in `agent-rt.yaml` (`agent_rt_settings.memory: {backend, options}`, see [Configuration](Configuration.md)); `backend`/`options` passed in code win. Unknown backends and options fail at load time, and an uninstalled package raises with the install command. These packages call their own services: mem0 and memU use an LLM and embedding model to extract and index memories (mem0's self-hosted default is OpenAI, so it needs `OPENAI_API_KEY`), and mem0 sends its own anonymous telemetry unless `MEM0_TELEMETRY=False`. Recalled memories are untrusted by default, like any store.

## Tool access lists and unadvertised calls

`Toolbase.initialize` accepts `allowlist`, `blocklist`, `read_allowlist`, `read_blocklist`, `write_allowlist`, and `write_blocklist` (TypeScript uses the same option keys in the `Toolbase.initialize({...})` object). Lists match exact exposed tool names, including MCP names such as `mcp.docs.weather`. An omitted allowlist imposes no restriction; an empty allowlist allows nothing. Deny rules override allow rules. Read scopes apply to tools marked `none` or `read`; all other side effects use write scopes.

Excluded tools are removed before relevance selection, model exposure, and `tool_search`. Excluded MCP tools are not registered. Direct calls through `Toolbase.execute` to screened-out tools are rejected, and `AgentLoop` rejects any call not present in the active advertised tool set, including names fabricated by a provider.

```python
base = await Toolbase.initialize(
    tool_registry=registry,
    blocklist=("shell.delete",),
    read_allowlist=("documents.search", "documents.read"),
    write_allowlist=(),
)
```


## Action blockers

`ToolRegistry` accepts multiple deny-only action blockers in addition to transform-capable tool-input guardrails. Blockers run in order and the first rejection raises `ActionBlockedError` before input guardrails, argument validation, approval, or handler execution.

The built-in rule blocker defaults to `sudo`, `rm -rf`, `git add`, `git commit`, `git push`, `git reset --hard`, `git merge`, and `git rebase`; pass a different command list to replace those defaults.

```python
from agent_rt import (
    RuleBasedActionBlocker,
    ToolRegistry,
    make_agent_action_guard_action_blocker,
    make_decision_action_blocker,
)

def organization_blocker(call, definition, context):
    if definition.side_effect == "write" and context.get("read_only"):
        return "write actions are disabled for this run"
    return False

registry = ToolRegistry(
    action_blockers=(
        RuleBasedActionBlocker(["sudo", "rm -rf", "git push", "git reset --hard"]),
        make_agent_action_guard_action_blocker(),
        make_decision_action_blocker(laya_or_jev_provider, threshold=0.85),
        organization_blocker,
    ),
)
```

A custom Python blocker may be a callable or an object exposing `check`; it can return `False`/`None` to allow, `True` or a reason string to block, or an `ActionBlockerResult`. TypeScript exposes the corresponding `RuleBasedActionBlocker`, `makeAgentActionGuardActionBlocker`, and `makeDecisionActionBlocker` APIs; `registry.actionBlockers` is the configured ordered list. The Decision-model adapter accepts a provider with `decide`, including Laya/Jev-compatible providers and `OpenAIDecisionProvider`, and evaluates a typed `noul` score against the configured threshold.

Every Decision helper also works with the OpenAI Decisions API. Set `decision_model: {provider: openai, api_key: os.environ/OPENAI_API_KEY}` in `agent-rt.yaml`, or construct `OpenAIDecisionProvider(api_key=...)` / `new OpenAIDecisionProvider({ apiKey })` directly. The adapter sends `noul` questions as `predicate` and `choice` questions (with their `criteria`) as `choice` to `POST /v1/decisions`, passes non-string state as an `input_json` part, and maps the answers back to the same shape a Jev host returns. A refused or missing answer raises, so guards fail closed.

## API server (Python)

Install `agent-rt[api]` to expose configured agents through OpenAI-compatible Chat Completions/Responses APIs and an Anthropic-compatible Messages API.

```python
from agent_rt import AgentConfig, AgentLoop, ModelSettings, load_config, load_model
from agent_rt_api import serve

provider = load_model(load_config())
loop = AgentLoop(provider)
agent = AgentConfig(
    name="assistant",
    instructions="You are a helpful assistant.",
    model=ModelSettings(model="your-provider-model"),
)

serve(loop, {"assistant": agent}, port=8000)
```

The server includes bounded concurrency/queues, request and generation limits, authentication support, streaming, and conservative defaults for unauthenticated serving.

- **Rate limiting** is keyed by verified credential (or client address), never by an unverified `Authorization`/`x-api-key` value, so rotating bogus credentials neither evades nor floods the limiter.
- **Validation happens before streaming starts**: malformed bodies, bad `temperature`/`max_tokens`, unknown models, and unsupported content parts (for example `image_url`) return a 4xx instead of failing inside an SSE stream. Unsupported parts are rejected, never silently dropped.
- **Interrupted runs are labelled honestly**: limit-driven stops (`max_turns`, `max_tool_calls`, token budget, timeout) report `finish_reason: "length"` (Anthropic `stop_reason: "max_tokens"`) and never expose unexecuted server-side tool calls to the client.
- `/health` returns only `{"status": "ok"}` to unauthenticated callers when API keys are configured; agent names and queue depth require a valid key.

The TypeScript package provides the API-façade contracts but no ready-made HTTP server.

## Task lists (todo tool)

`TodoList` gives a model a task list through one `todo` tool. Its `action` is `add` (with `content`, optional `status`), `update` (`id` plus `content` and/or `status`), `remove` (`id`), `clear` (empties the whole list), or `list`; statuses are `pending`, `in_progress`, and `completed`, and every action returns the whole list. Bad calls (unknown ids, empty content, a full list of 100 tasks) come back to the model as tool errors and leave the list unchanged.

Each `TodoList` holds exactly one session's tasks, and the tool it registers only sees that list, so create one per session to keep lists from leaking between sessions:

```python
todos = TodoList()
todos.register(registry)          # adds the "todo" tool
todos.replace(saved_items)        # restore a stored list; validated
```

```ts
const todos = new TodoList();
todos.register(registry);
todos.replace(savedItems);
```

The terminal CLI and the web UI register the tool by default. Each session keeps its own list, saved in its transcript: `/new` and `/clear` start a new session (new id, empty history and list; the previous session stays resumable), `/resume` restores the session's list, `/todos` shows it, and the web UI sends each session's list with its prompts and saves it with the session.

## Toolbase, relevance selection, and tool search

For large tool inventories, initialize a `Toolbase` rather than exposing every schema to every model turn. `Toolbase.initialize` / `Toolbase.initialize(...)` combines enabled local tools, materialized deferred tools, and tools discovered from configured MCP clients into one internal catalog. An optional safety filter is applied to that complete catalog first. The per-turn visibility selector then chooses the relevant subset, while `tool_search` remains available so the agent can discover a safe tool that was not initially selected.

When `Toolbase` is passed as the `tool_registry` / `toolRegistry` argument to `AgentLoop`, the loop automatically uses its visibility filter. Tools returned by `tool_search` are promoted on later turns, but they still have to survive capability grants, prompt-injection filtering, tool-selection policy, context isolation, argument validation, action blockers, approvals, and all other execution-time checks. Replacing a screened tool definition also fails closed until the toolbase is rebuilt.

Decision models can configure both stages together with `make_decision_toolbase` in Python or `makeDecisionToolbase` in TypeScript. These helpers safety-screen tool definitions before relevance selection, so an unsafe tool cannot be rediscovered through `tool_search`.

## Deterministic PII morphing

```python
from agent_rt import PIIEntity, PIIMorpher

morpher = PIIMorpher()
safe_text = morpher.morph_text("Contact alice@example.com")
lookup = morpher.mapping  # short fingerprint -> deterministic fake value

safe_name = morpher.morph_text(
    "Customer Alice Smith",
    entities=(PIIEntity(kind="person", start=9, end=20),),
)
```

```ts
import { PIIMorpher } from "agent-rt";

const morpher = new PIIMorpher();
const safeText = morpher.morphText("Contact alice@example.com");
const lookup = morpher.mapping;
```

`PIIMorpher` is enabled by default when used through `PrivacyRedactor`. It detects a conservative set of high-confidence identifiers in free text (email, North American phone number, SSN, Luhn-valid payment-card number, and IPv4), uses common structured field names for person/address/date values, and accepts explicit entity spans from external PII/NER detectors. The same PII maps to the same realistic synthetic value in Python and TypeScript. Its in-memory lookup stores only a 20-hex fingerprint (10 SHA-256 hex characters plus 10 MD5 hex characters) to fake-data mapping, never the original PII. Pass `enabled=False` (Python) / `{ enabled: false }` (TypeScript) to disable morphing.

The short unkeyed fingerprint is for deterministic lookup, not a cryptographic privacy boundary. Treat the mapping as sensitive metadata, and use keyed HMAC or deterministic encryption/tokenization when dictionary-attack resistance or reversible pseudonymization is required.

## Terminal CLI (Python)

Install `agent-rt[cli]` to add the `agent-rt` command. `artie` is an alias for it with the same subcommands and options (`artie -p "..."`, `artie migrate`, `artie web`); with `uvx`, run it as `uvx --from agent-rt artie`:

```bash
agent-rt
agent-rt -p "explain this stack trace"
agent-rt --config ./agent-rt.yaml -p "explain this stack trace"
agent-rt --session refactor --prompt "plan the migration"
agent-rt --session refactor --resume
agent-rt session list
agent-rt session delete refactor
agent-rt --max-output-tokens 4096 -p "summarize this diff"
```

An interactive session opens with a banner of Artie, the Agent RT mascot, drawn from the logo in colored half-block pixels (a colored ASCII version on consoles without 256 colors or UTF-8), showing its name, the version, the model, the session id, and the flagship features (Guard, Checkpoints, Wall); `-p`/`--prompt` runs print no banner. Artie has moods: when Guard (an action blocker or guardrail) blocks a call, the session prints Artie raising a shield beside the reason instead of a plain error, `agent-rt demo` shows the same pose when it stops the injected `rm -rf ~`, and `agent-rt migrate` prints Artie celebrating when the rewrite is kept.

The CLI uses the same runtime, provider routing, guardrails, tool execution, and run limits as programmatic Agent RT. `--config PATH` selects the YAML config file (default: `agent-rt.yaml` in the current folder, then `~/.agent-rt/config.yaml`); see [Configuration](Configuration.md). Workspace writes require approval; Git integration is inspection-only. The `todo` tool (see [Task lists](#task-lists-todo-tool)) is on by default, with one list per session. Writes to `.git/` and to `.gitattributes`/`.gitmodules` are refused outright, and the Git tools fail closed when the repository's own config defines `filter.*` drivers (Git runs those commands during `status`/`diff`). Workspace tool errors (for example a non-matching `replace_in_file`) are returned to the model as tool results so the turn and its history survive; approval previews state how many characters are not shown.

### Reuse another agent CLI's setup

`agent-rt claude`, `agent-rt codex`, `agent-rt opencode`, and `agent-rt copilot` accept that tool's own one-shot command and flags. They read its model, provider, and API-key settings from its usual environment variables and config files, so no `agent-rt.yaml` is needed:

```bash
agent-rt claude -p "explain this stack trace" --model opus
agent-rt codex exec "explain this stack trace" -m gpt-5.5
agent-rt opencode run "explain this stack trace"
agent-rt copilot -p "explain this stack trace"
```

| Command | Flags | Environment variables | Config files |
| --- | --- | --- | --- |
| `agent-rt claude` | `-p`, `--model`, `--system-prompt`, `--append-system-prompt`, `--max-turns`, `-r/--resume`, `--session-id` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL` / `_OPUS_` / `_HAIKU_` / `_FABLE_MODEL` (alias targets), `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, `CLAUDE_CONFIG_DIR` | `model` and `env` in `$CLAUDE_CONFIG_DIR/settings.json` (default `~/.claude`); `model` only from `.claude/settings.json` and `.claude/settings.local.json` |
| `agent-rt codex` | `exec`, `-m`, `-c key=value`, `-p/--profile`, `--oss`, `--local-provider`, `-C`, `--ephemeral` | `CODEX_HOME`, `CODEX_API_KEY` (then `OPENAI_API_KEY`), a provider's `env_key`, `OPENAI_BASE_URL` | `$CODEX_HOME/config.toml`: `model`, `model_provider`, `openai_base_url`, `oss_provider`, `model_providers.<id>` (`base_url`, `env_key`, `requires_openai_auth`, `experimental_bearer_token`), profiles as `$CODEX_HOME/<name>.config.toml` or `[profiles.<name>]`; the API key in `$CODEX_HOME/auth.json`; built-in `ollama` and `lmstudio` providers |
| `agent-rt opencode` | `run`, `-m provider/model`, `-s`, `--dir` | `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, `OPENCODE_CONFIG_CONTENT`, `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `~/.config/opencode/opencode.json(c)` (JSONC, `{env:NAME}` and `{file:path}` substitution), `model` only from the project `opencode.json(c)`; API-key entries in `~/.local/share/opencode/auth.json` |
| `agent-rt copilot` | `-p`, `--model`, `-C`, `--resume` | `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_TYPE` (`openai`/`anthropic`), `COPILOT_PROVIDER_API_KEY`, `COPILOT_PROVIDER_BEARER_TOKEN` (OpenAI-compatible only), `COPILOT_MODEL`, `COPILOT_PROVIDER_MODEL_ID`, `COPILOT_PROVIDER_WIRE_MODEL`, `COPILOT_PROVIDER_MAX_OUTPUT_TOKENS` | none (BYOK is environment-only) |

These are the variables each tool documents for models, endpoints, and credentials ([Claude Code](https://code.claude.com/docs/en/env-vars), [Codex](https://learn.chatgpt.com/docs/config-file/environment-variables), [opencode](https://opencode.ai/docs/cli/), and `copilot help providers` for Copilot CLI). Project-level files may only choose the model. Endpoints and keys come from user-level config or the environment, so a cloned repository cannot send your API key to its own server; Codex's project `.codex/config.toml` is not read for the same reason. Documented variables that Agent RT cannot honor are reported rather than silently dropped. `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_BETAS`, Codex `http_headers`/`env_http_headers`/`query_params`, `COPILOT_PROVIDER_WIRE_API=responses`, and the permission variable `COPILOT_ALLOW_ALL` print a notice. `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CODE_USE_BEDROCK`/`_VERTEX`/`_FOUNDRY`, `CODEX_ACCESS_TOKEN`, and Copilot's `azure` provider type stop with a message saying what to set instead.

Only API-key credentials are reused. Subscription and OAuth sign-ins (Claude.ai, ChatGPT, GitHub Copilot's hosted models, opencode OAuth providers) belong to the official clients and are refused with a hint about which key to set. The CLI's own `agent-rt.yaml` is not read in these modes.

## Web UI (Python)

Install `agent-rt[web]` to add the `agent-rt web` command. It serves a browser chat for the current folder on `http://127.0.0.1:8765/` prints a reusable launch link (`/?token=…`), and opens a tab with a single-use code instead, because a browser's command line is visible to other users on the computer:

```bash
agent-rt web
agent-rt web --config ./agent-rt.yaml --model gpt-5.5 --port 9000 --no-open
```

No config file is needed: without one, or with one that has no `model_list` (even a blank file), the web UI uses OpenAI with `OPENAI_API_KEY` (and `OPENAI_BASE_URL`, when set, for another OpenAI-compatible endpoint) and lists the models from its `/v1/models`, starting with the first unless you pass `--model`. Use a config to choose other providers or endpoints. The web UI uses the same workspace tools, skills, and run limits as the terminal CLI. Sessions are shared with the terminal CLI: both save them as JSON files in `~/.agent-rt/sessions` (under `cli.home`, owner-only), so a session started in `agent-rt` opens in the web UI and the reverse, and `agent-rt session list` / `agent-rt session delete ID` manage them from either side. The server reads and writes that folder through `/api/sessions`, and the browser still sends the session history with each prompt. Sessions are listed per workspace folder (a link to another folder's session opens a new one; sessions saved before workspaces were recorded show in every folder), and a tab picks up sessions the terminal saved or deleted when you return to it. The server starts in about half a second and uses about 50 MB; the first tool call loads the Agent Action Guard classifier that screens workspace tool calls, which adds about 150 MB. Long replies stream without freezing the page. Drag the sessions sidebar's right edge (or focus it and use the arrow keys) to resize it between 180 and 480 pixels; the width is remembered in this browser. The model menu starts with the `model_list` names from your config, with `agent_rt_settings.default_model` (or the first entry) selected; once the page has loaded it also asks each configured endpoint's `/v1/models` (in the background, 10 seconds per endpoint, failures ignored) and adds those model ids, which are sent straight to the endpoint that listed them. Workspace edits show an approval card in the page (approve, approve for this session, or deny). While a run is going the run button becomes a stop button; stopping, like closing the tab, stops the run and denies its pending edit. Each tab keeps its own session (in the URL), so several tabs can run different sessions at the same time; a second tab sending to a session that is already running waits for that run (browser Web Locks), then continues from both turns. The server listens on loopback only, rejects requests whose `Host` is not `127.0.0.1` or `localhost`, and serves the page and API only to a browser that opened the launch link: the link's per-launch token becomes an `HttpOnly`, `SameSite=Strict` cookie, and API calls also need a custom header, so other users and processes on the computer and other websites cannot drive it. With `--no-open`, open the printed link yourself; to use `localhost` instead of `127.0.0.1`, change the host in that link.

The web server limits memory growth from slow browser connections with a bounded streaming queue. By default, four agent runs execute at once and up to four additional requests may wait for a slot (for up to 15 seconds); excess requests receive HTTP 429. Use `--max-concurrent-runs N` and `--max-queued-runs N` to adjust those limits (`--max-queued-runs 0` rejects immediately when full). Each completed or disconnected stream releases its slot.

### Time-travel debugger

The same server has a debugger page at `/debug` (linked from the sessions sidebar). Run `agent-rt web --demo` (or `agent-rt demo --web`) to open it straight away with no config or API key. It replays the offline `agent-rt demo` scenario on a scripted mock model: a weather tool returns a prompt injection, the model tries `run_shell` with `rm -rf ~`, and the built-in action blocker stops it before the tool runs. Each model turn is its own `AgentLoop.run`, resumed from the previous turn's `AgentCheckpoint`, so every node on the timeline is a real checkpoint. The page shows the token budget, tool calls run, and blocked actions for the selected checkpoint, plus that step's events and the transcript at that point. **Replay branch** steps through the branch from the start, and the arrow keys move between checkpoints and branches. **Fork & resume** starts a new branch from the selected checkpoint, optionally with the latest tool result rewritten (for example, with the injection removed) or with a different token budget (try 20 to see `budget_exhausted`). The debugger state lives in server memory, is capped at 200 checkpoints, and is lost when the server stops. In `--demo` mode, chat is off.

Deleting a session coordinates with an active run using Web Locks when available, preventing the run from recreating a deleted transcript. Very long unfinished Markdown blocks render as plain text while streaming and receive full Markdown formatting when complete, avoiding repeated expensive parsing.

## Sandboxes

`SandboxSession` runs commands through a fail-closed backend selected with `agent_rt_settings.sandbox.backend` (`native`, `docker`, `e2b`, `microsandbox`, `swe-rex`).

- **Native:** requires an explicit restricted account and runs as that account's primary gid unless `sandbox.gid` is set; it never keeps the caller's gid. The TypeScript backend uses `setpriv --clear-groups --no-new-privs` and reads the primary gid from `/etc/passwd`; the Python backend drops supplementary groups.
- **Docker:** containers run with all capabilities dropped, `no-new-privileges`, and swap capped to the memory limit. Set `sandbox.docker_user` to run as a non-root user and `sandbox.docker_read_only: true` for a read-only root filesystem.
- **Lifetime:** timeouts and cancellation kill the command (the process group for native, the container for Docker), not just await it. TypeScript backends receive an `AbortSignal` in their execute context.
- **Limits:** output is capped while it is read (`SandboxResourceLimits.output_bytes` in Python, `outputBytes` in TypeScript; default 16 MiB), and `memory_bytes`/`process_count` (`memoryBytes`/`processCount`) of `0` are rejected.
- **Cleanup:** call `await session.close()` to release E2B/microsandbox sessions and microVMs.
- **Snapshots:** snapshots and clones capture only the in-process workspace and session settings, not the disk inside a native, Docker, or cloud backend.

## Skills and registration safety

`SkillRegistry` (from `agent-rt/extensions` in TypeScript) loads Claude Code/Codex-style `SKILL.md` directories and collections:

```ts
import { SkillRegistry } from "agent-rt/extensions";

const skills = new SkillRegistry();
await skills.installFromPath(".claude/skills/review", { activate: true });
await skills.installDirectory(".agents/skills");
```

Python exposes the same methods as `install_from_path` and `install_directory`. Filesystem skill registration is fail-closed: Agent RT performs deterministic scanning and, for externally sourced registrations, requires a configured Decision model before registry mutation.

## Migration entry points

Prepend the package name to a supported upstream import path.

| Upstream | Python | TypeScript |
| --- | --- | --- |
| `openai` | `agent_rt.openai` | `agent-rt/openai` |
| `anthropic` / `@anthropic-ai/sdk` | `agent_rt.anthropic` | `agent-rt/@anthropic-ai/sdk` |
| `langchain_openai` / `@langchain/openai` | `agent_rt.langchain_openai` | `agent-rt/@langchain/openai` |
| `llama_index.llms.openai` / `@llamaindex/openai` | `agent_rt.llama_index.llms.openai` | `agent-rt/@llamaindex/openai` |
| `agents` / `@openai/agents` | `agent_rt.agents` | `agent-rt/@openai/agents` |
| `autogen_agentchat.agents` | `agent_rt.autogen_agentchat.agents` | Python-only |
| `crewai` | `agent_rt.crewai` | Python-only |
| `deepagents` | `agent_rt.deepagents` | `agent-rt/deepagents` |
| `litellm` | `agent_rt.litellm` | Python-only |
| `openrouter` / `@openrouter/sdk` | `agent_rt.openrouter` | `agent-rt/@openrouter/sdk` |

TypeScript also keeps the historical aliases `agent-rt/langchain`, `agent-rt/llamaindex`, and `agent-rt/anthropic`. Compatibility routes model execution through Agent RT provider contracts instead of creating a second runtime; AutoGen, CrewAI, and LiteLLM compatibility are Python-only because Agent RT does not invent JavaScript APIs without an upstream surface to preserve. See [Migration](MIGRATION.md) for the supported subset and semantic gaps.

## Vector databases

Built-in HTTP vector adapters cover Chroma, Milvus, Pinecone, Qdrant, and Weaviate; FAISS and pgvector plug in through custom registry factories. See [Features and defaults](website/docs/reference/features-defaults.html) for the full list.

# Runtime guide

Detailed usage notes for features that the package READMEs only mention briefly. Examples are shown for Python unless a section says otherwise; the TypeScript package uses the same concepts with camelCase names. For design rationale and contracts see [Architecture and Extensions](Architecture-and-Extensions.md).

## Providers

`load_model()` (Python) / `loadModel()` (TypeScript) selects a provider from the environment:

- `OPENAI_BASE_URL` selects OpenAI and takes precedence when both base URLs are set; otherwise `ANTHROPIC_BASE_URL` selects Anthropic.
- With neither base URL, `OPENAI_MODEL` alone selects OpenAI.
- `MODEL_PROVIDER=openai|anthropic` overrides auto-detection; invalid values fail fast.
- API keys alone, or `ANTHROPIC_MODEL` alone, never choose a provider.

The OpenAI provider uses the native OpenAI-compatible transport by default and supports HTTP/SSE plus Responses WebSocket operation where available. The Anthropic provider uses the optional official Anthropic SDK. Reasoning/thinking controls are normalized into the provider-neutral `ReasoningConfig`, and each provider translates them into vendor-specific request fields.

## Content and provider behaviour

- **Media:** image, PDF/file, and (OpenAI) audio `ContentPart`s are sent as the provider's native blocks; `data` may be an http(s) URL, a `data:` URL, base64, or bytes. Parts a provider cannot carry (for example video) raise instead of being dropped.
- **Anthropic thinking:** thinking blocks are kept on the assistant message (a JSON part with a vendor MIME type) and replayed before `tool_use`, so extended thinking works across tool turns. Anthropic requests default to `max_tokens=4096`; set `max_output_tokens` (`maxOutputTokens` in TypeScript) for longer outputs.
- **Context trust:** untrusted retrieved/file/observation items reach the model as user-role data; only workflow state and items explicitly marked trusted are system messages.
- **Validation:** tool arguments and structured output are checked against a JSON Schema subset (types, bounds, `pattern`, `const`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref`, `additionalProperties`).
- **Truncation:** a Responses WebSocket `response.incomplete` returns `finish_reason="length"` (`finishReason: "length"` in TypeScript) with usage rather than raising.

## Agent loop guarantees

The `AgentLoop` keeps every tool call answered: if a run stops early (approval needed, timeout, cancellation, `max_tool_calls`) unexecuted calls receive a `tool_not_executed` tool message, completed results from a concurrent batch are kept, and loop-detector state is per run.

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

A custom Python blocker may be a callable or an object exposing `check`; it can return `False`/`None` to allow, `True` or a reason string to block, or an `ActionBlockerResult`. TypeScript exposes the corresponding `RuleBasedActionBlocker`, `makeAgentActionGuardActionBlocker`, and `makeDecisionActionBlocker` APIs; `registry.actionBlockers` is the configured ordered list. The Decision-model adapter accepts a provider with `decide`, including Laya/Jev-compatible providers, and evaluates a typed `noul` score against the configured threshold.

## API server (Python)

Install `agent-rt[api]` to expose configured agents through OpenAI-compatible Chat Completions/Responses APIs and an Anthropic-compatible Messages API.

```python
from agent_rt import AgentConfig, AgentLoop, ModelSettings, load_model
from agent_rt_api import serve

provider = load_model()
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

`PIIMorpher` is enabled by default when used through `PrivacyRedactor`. It detects a conservative set of high-confidence identifiers in free text (email, North American phone number, SSN, Luhn-valid payment-card number, and IPv4), uses common structured field names for person/address/date values, and accepts explicit entity spans from external PII/NER detectors. The same PII maps to the same realistic synthetic value in Python and TypeScript. Its in-memory lookup stores only a 20-hex fingerprint (10 SHA-256 hex characters plus 10 MD5 hex characters) to fake-data mapping, never the original PII. Set `AGENT_RT_DISABLE_PII_MORPHER=true` to disable morphing; the Python CLI can also receive it through `--env-file`.

The short unkeyed fingerprint is for deterministic lookup, not a cryptographic privacy boundary. Treat the mapping as sensitive metadata, and use keyed HMAC or deterministic encryption/tokenization when dictionary-attack resistance or reversible pseudonymization is required.

## Terminal CLI (Python)

Install `agent-rt[cli]` to add the `agent-rt` command:

```bash
agent-rt
agent-rt -p "explain this stack trace"
agent-rt --env-file=./.env -p "explain this stack trace"
agent-rt --session refactor --prompt "plan the migration"
agent-rt --session refactor --resume
```

The CLI uses the same runtime, provider routing, guardrails, tool execution, and run limits as programmatic Agent RT. `--env-file=PATH` loads configuration variables from a UTF-8 dotenv-style file before provider and runtime configuration is resolved. Existing process environment variables take precedence over the file, while file values take precedence over the CLI's global settings fallback. The file supports blank lines, `#` comments, optional `export`, single- and double-quoted values, and double-quoted escapes; it is parsed as data and is never shell-evaluated. Workspace writes require approval; Git integration is inspection-only. Writes to `.git/` and to `.gitattributes`/`.gitmodules` are refused outright, and the Git tools fail closed when the repository's own config defines `filter.*` drivers (Git runs those commands during `status`/`diff`). Workspace tool errors (for example a non-matching `replace_in_file`) are returned to the model as tool results so the turn and its history survive; approval previews state how many characters are not shown.

## Sandboxes

`SandboxSession` runs commands through a fail-closed backend selected with `AGENT_RT_SANDBOX_BACKEND` (`native`, `docker`, `e2b`, `microsandbox`, `swe-rex`).

- **Native:** requires an explicit restricted account and runs as that account's primary gid unless `AGENT_RT_SANDBOX_GID` is set; it never keeps the caller's gid. The TypeScript backend uses `setpriv --clear-groups --no-new-privs` and reads the primary gid from `/etc/passwd`; the Python backend drops supplementary groups.
- **Docker:** containers run with all capabilities dropped, `no-new-privileges`, and swap capped to the memory limit. Set `AGENT_RT_SANDBOX_DOCKER_USER` to run as a non-root user and `AGENT_RT_SANDBOX_DOCKER_READ_ONLY=true` for a read-only root filesystem.
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

TypeScript also keeps the historical aliases `agent-rt/langchain`, `agent-rt/llamaindex`, and `agent-rt/anthropic`. Compatibility routes model execution through Agent RT provider contracts instead of creating a second runtime; AutoGen and CrewAI compatibility are Python-only because Agent RT does not invent JavaScript APIs without an upstream surface to preserve. See [Migration](MIGRATION.md) for the supported subset and semantic gaps.

## Vector databases

Built-in HTTP vector adapters cover Chroma, Milvus, Pinecone, Qdrant, and Weaviate; FAISS and pgvector plug in through custom registry factories. See [Features and defaults](website/docs/reference/features-defaults.html) for the full list.

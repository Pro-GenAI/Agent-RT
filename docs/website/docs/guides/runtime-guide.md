> **Canonical HTML page:** [Runtime guide — Agent RT Docs](https://agent-rt.pro-genai.com/docs/guides/runtime-guide)

> Detailed Agent RT usage notes for provider selection, content handling, the Python API server and CLI, sandboxes, skills, and migration entry points.

[Docs](https://agent-rt.pro-genai.com/docs/)

**Section:** Guides / Runtime guide

# Runtime guide

Detailed usage notes for providers, the Python API server and CLI, sandboxes, skills, and migration entry points. The package READMEs stay short and link here.

## Providers

`load_model(load_config())` (Python) and `loadModel(await loadConfig())` (TypeScript) build a provider from the `model_list` of a YAML config file (`~/.agent-rt/config.yaml`, or an explicit project config path). Several models are supported, repeated `model_name`s act as ordered fallbacks, and the environment is only used to resolve `os.environ/NAME` secret references.

The OpenAI provider uses the native OpenAI-compatible transport with HTTP/SSE and Responses WebSocket support; the Anthropic provider uses the optional official SDK. Reasoning controls are normalized into the provider-neutral `ReasoningConfig`.

## Content and provider behavior

- **Media:** image, PDF/file, and (OpenAI) audio parts are sent as native provider blocks; `data` may be an http(s) URL, a `data:` URL, base64, or bytes. Parts a provider cannot carry raise instead of being dropped.

- **Anthropic thinking:** thinking blocks are preserved and replayed before `tool_use`. Requests default to `max_tokens=4096`; set `max_output_tokens` (`maxOutputTokens` in TypeScript) for longer outputs.

- **Context trust:** untrusted retrieved or file content reaches the model as user-role data; only workflow state and explicitly trusted items are system messages.

- **Validation:** tool arguments and structured output are checked against a JSON Schema subset (types, bounds, `pattern`, `const`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref`, `additionalProperties`).

- **Truncation:** a Responses WebSocket `response.incomplete` returns `finish_reason="length"` with usage instead of raising.

- **Loop guarantees:** every tool call is answered. When a run stops early, unexecuted calls receive a `tool_not_executed` message and completed results from a concurrent batch are kept.

## Agent memory

Give an agent memory with `AgentConfig.memory`, a list that can mix strings, custom objects (sent as JSON), memory stores and retrievers from Agent RT, LangChain, LangGraph, LlamaIndex, AutoGen, OpenAI Agents, or mem0, and functions that return memories. Each run adds the results as retrieved context, so context policies and prompt-injection defenses apply to them. Store and function results are untrusted by default.

`load_memory()` / `loadMemory()` connect a dedicated memory package: mem0 (default; `pip install 'agent-rt[memory]'` or `npm install mem0ai`), Zep, Letta, Supermemory, or memU (Python 3.13+ only). Call `remember(messages)` after a run to store the user and assistant turns.

```python
memory = load_memory(options={"user_id": "alice"})   # or backend="zep", "letta", ...
agent = AgentConfig(name="assistant", instructions="Be concise.",
                    model=ModelSettings(model="my-model"), memory=[memory])
result = await AgentLoop(provider).run(agent, messages)
await memory.remember(result.messages)
```

The backend can also be set once in `agent-rt.yaml` with `agent_rt_settings.memory: {backend, options}`. These packages call their own services; mem0 and memU use an LLM and embedding model to index memories.

## Decision providers

Decision helpers (tool safety screening, relevance pruning, memory and retrieval gates, failure classification) work with Jev-compatible Decision models, and with the OpenAI-compatible Decisions API. Set `decision_model: {provider: openai, api_key: os.environ/OPENAI_API_KEY}` in `agent-rt.yaml`, adding `base_url` for any OpenAI-compatible Decisions host (independent of the generative model's `base_url`, so LLMs and Decision models can come from different APIs), or construct `OpenAIDecisionProvider` directly. A refused or missing answer raises, so guards fail closed.

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

The server has bounded concurrency and queues, request and generation limits, authentication, streaming, and conservative defaults for unauthenticated serving.

- Rate limiting is keyed by verified credential or client address, never by unverified header values.

- Validation happens before streaming starts; malformed bodies, bad parameters, unknown models, and unsupported content parts return a 4xx.

- Limit-driven stops report `finish_reason: "length"` (Anthropic `stop_reason: "max_tokens"`) and never expose unexecuted server-side tool calls.

- `/health` returns only `{"status": "ok"}` to unauthenticated callers when API keys are configured.

TypeScript provides the API-façade contracts but no ready-made HTTP server.

## Deterministic PII morphing

`PIIMorpher` deterministically replaces detected PII with synthetic values while preserving repeated references. Built-in free-text detection covers a conservative set of high-confidence identifiers (email, North American phone number, SSN, Luhn-valid payment-card number, and IPv4); structured field-name hints cover common person/address/date fields, and callers can supply explicit entity spans from an external PII/NER recognizer. Python and TypeScript use the same algorithm and fake-data tables.

The in-memory lookup stores only a 20-hex fingerprint (10 SHA-256 hex characters plus 10 MD5 hex characters) to fake-data mapping, never plaintext PII. `PrivacyRedactor` enables morphing by default after secret/text-pattern redaction. Set `AGENT_RT_DISABLE_PII_MORPHER=true` to disable it. The unkeyed fingerprint is a deterministic lookup mechanism, not cryptographic tokenization; use keyed HMAC or deterministic encryption when stronger pseudonymization guarantees are required.

## Terminal CLI (Python)

Install `agent-rt[cli]` to add the `agent-rt` command. `artie` is an alias for it with the same subcommands and options; with `uvx`, run it as `uvx --from agent-rt artie`.

An interactive session opens with a banner of Artie, the Agent RT mascot, drawn from the logo in colored half-block pixels (a colored ASCII version on consoles without 256 colors or UTF-8), showing its name, the version, the model, the session id, and the flagship features (Guard, Checkpoints, Wall). `-p`/`--prompt` runs print no banner. Artie has moods: it raises a shield beside the reason when Guard blocks a call, and celebrates when `agent-rt migrate` keeps a rewrite.

Run `agent-rt demo` (`uvx agent-rt demo` or `npx agent-rt demo`) to try Agent RT with no config or API key: a scripted mock model meets a prompt injection, tries `rm -rf ~`, and the built-in action blocker stops it before the tool runs. The demo totals tokens and prints the process's peak memory.

```bash
agent-rt
agent-rt -p "explain this stack trace"
agent-rt --config ./agent-rt.yaml -p "explain this stack trace"
agent-rt --session refactor --prompt "plan the migration"
agent-rt --session refactor --resume
agent-rt session list
agent-rt session delete refactor
```

The CLI shares the runtime, provider routing, guardrails, tool execution, and run limits of programmatic Agent RT. `--config PATH` selects the YAML config file (default: `~/.agent-rt/config.yaml`; project files require an explicit path). Workspace writes require approval and Git integration is inspection-only: writes to `.git/`, `.gitattributes`, and `.gitmodules` are refused, and Git tools fail closed when repository config defines `filter.*` drivers. Recoverable tool errors go back to the model so the turn and its history survive. A `todo` tool is on by default: the model can add, update, remove, list, or clear all of its tasks, and each session keeps its own list, saved with its transcript and shared with the web UI. `/todos` shows the current list, and `/clear` or `/new` starts a new session with an empty one. Programs get the same tool from `TodoList` (one instance per session) in Python and TypeScript.

### Project instructions, skills, and MCP servers

The CLI reads the same files Codex and Claude Code read, so a repository set up for either works unchanged.

- **Instructions**: `~/.codex/AGENTS.md`, `~/.claude/CLAUDE.md`, then `AGENTS.md` (or `AGENTS.override.md`), `CLAUDE.md`, `.claude/CLAUDE.md`, and `CLAUDE.local.md` in each folder from the Git root down to the launch folder; nearer files take precedence. They are added to the system prompt for each launch, capped at 32 KiB, and symlinks pointing outside the repository are ignored.

- **Skills**: `SKILL.md` folders in `~/.claude/skills`, `~/.codex/skills`, `~/.agents/skills`, and `<cli.home>/skills`, plus the project's `.claude/skills`, `.codex/skills`, `.agents/skills`, and `.agent-rt/skills`. New or changed CLI skills first pass a complete static scan with [NVIDIA SkillSpector](https://github.com/NVIDIA/SkillSpector) (separate installation required) and the existing registration and Decision-model checks. Successful scans are cached by file hash in `<cli.home>/trusted_skills.json`; unchanged skills skip repeat SkillSpector and Decision checks, while changed skills must pass again. The public `SkillRegistry` APIs do not require SkillSpector.

- **MCP servers**: `[mcp_servers.*]` in `~/.codex/config.toml`, `mcpServers` in `~/.claude.json`, and the project's `.mcp.json` (stdio and streamable HTTP). Project servers start only after you approve them at launch or approved them in Claude Code. Tools appear as `mcp__<server>__<tool>` and every call asks for approval; `/mcp` lists them. The web UI loads instructions and skills but not MCP servers.

### Reuse another agent CLI's setup

`agent-rt claude`, `agent-rt codex`, `agent-rt opencode`, and `agent-rt copilot` accept that tool's own one-shot command and flags. They read its model, provider, and API-key settings from its usual environment variables and config files, so no `agent-rt.yaml` is needed.

```bash
agent-rt claude -p "explain this stack trace" --model opus
agent-rt codex exec "explain this stack trace" -m gpt-5.5
agent-rt opencode run "explain this stack trace"
agent-rt copilot -p "explain this stack trace"
```

| Command | Environment variables | Config files |
| --- | --- | --- |
| `agent-rt claude` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`/`_OPUS_`/`_HAIKU_`/`_FABLE_MODEL`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, `CLAUDE_CONFIG_DIR` | User `settings.json` (`model`, `env`); project settings choose the model only |
| `agent-rt codex` | `CODEX_HOME`, `CODEX_API_KEY` (then `OPENAI_API_KEY`), a provider's `env_key`, `OPENAI_BASE_URL` | `config.toml` models, providers (`ollama`, `lmstudio`, custom), `openai_base_url`, profiles, and the API key in `auth.json` |
| `agent-rt opencode` | `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, `OPENCODE_CONFIG_CONTENT`, `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Global `opencode.json(c)` with `{env:…}`/`{file:…}`; project file chooses the model only; API keys in `auth.json` |
| `agent-rt copilot` | `COPILOT_PROVIDER_BASE_URL`, `_TYPE`, `_API_KEY`, `_BEARER_TOKEN`, `_MODEL_ID`, `_WIRE_MODEL`, `_MAX_OUTPUT_TOKENS`, `COPILOT_MODEL` | None (BYOK is environment-only) |

Project-level files may only choose the model, so a cloned repository cannot send your API key to its own server. Documented variables Agent RT cannot honor (custom headers, Bedrock/Vertex/Foundry switches, bearer or OAuth tokens) print a notice or stop with a message saying what to set instead.

Only API-key credentials are reused. Subscription and OAuth sign-ins (Claude.ai, ChatGPT, GitHub Copilot's hosted models, opencode OAuth) belong to the official clients and are refused. Permission flags such as `--dangerously-skip-permissions`, `--full-auto`, and `--allow-all-tools` are accepted but ignored with a notice, because Agent RT keeps its own workspace approval prompts. Unsupported options fail with a clear message instead of being silently dropped.

## Web UI (Python)

Install `agent-rt[web]` to add the `agent-rt web` command. It serves a browser chat for the current folder on `http://127.0.0.1:8765/` prints a reusable launch link (`/?token=…`), and opens a private, owner-only redirect file with a single-use code. The browser command line contains only the file path, and the redirect file is removed at shutdown.

```bash
agent-rt web
agent-rt web --config ./agent-rt.yaml --model gpt-5.5 --port 9000 --no-open
```

The web UI uses the same workspace tools, skills, and run limits as the terminal CLI. Sessions are shared with the terminal CLI: both save them as JSON files in `~/.agent-rt/sessions` (under `cli.home`, owner-only), so a session started in `agent-rt` opens in the web UI and the reverse, and `agent-rt session list` / `agent-rt session delete ID` manage them from either side. The server reads and writes that folder through `/api/sessions`, and the browser still sends the session history with each prompt. Sessions are listed per workspace folder (a link to another folder's session opens a new one; sessions saved before workspaces were recorded show in every folder), and a tab picks up sessions the terminal saved or deleted when you return to it. The server starts in about half a second and uses about 50 MB; the first tool call loads the Agent Action Guard classifier that screens workspace tool calls, which adds about 150 MB. Long replies stream without freezing the page. The model menu lists the `model_list` names from your config. Workspace edits show an approval card in the page (approve, approve for this session, or deny). While a run is going the run button becomes a stop button; stopping, like closing the tab, stops the run and denies its pending edit. Each tab keeps its own session (in the URL), so several tabs can run different sessions at the same time; a second tab sending to a session that is already running waits for that run (browser Web Locks), then continues from both turns. The server listens on loopback only, rejects requests whose `Host` is not `127.0.0.1` or `localhost`, and serves the page and API only to a browser that opened the launch link: the link's per-launch token becomes an `HttpOnly`, `SameSite=Strict` cookie, and API calls also need a custom header, so other users and processes on the computer and other websites cannot drive it. With `--no-open`, open the printed link yourself; to use `localhost` instead of `127.0.0.1`, change the host in that link.

The server bounds streaming memory with a finite queue. By default, four agent runs execute concurrently and four more may wait up to 15 seconds; further requests receive HTTP 429. Set `--max-concurrent-runs N` and `--max-queued-runs N` to change those limits, or set the queued limit to zero to reject requests immediately when full.

### Time-travel debugger

The same server has a debugger page at `/debug`. Run `agent-rt web --demo` (or `agent-rt demo --web`) to open it with no config or API key. It replays the offline demo on a scripted mock model: a weather tool returns a prompt injection, the model tries `rm -rf ~`, and the built-in action blocker stops it before the tool runs. Every timeline node is a real `AgentCheckpoint`. The page shows the token budget, tool calls run, and blocked actions for the selected checkpoint, replays a branch step by step, and can fork from any resumable checkpoint with the latest tool result rewritten or a different token budget. Forks resume the runtime from that checkpoint on a new branch. The state lives in server memory, capped at 200 checkpoints; chat is off in `--demo` mode.

Deletion waits for active same-session runs where Web Locks are available. Long unfinished Markdown blocks stream as efficiently appended plain text and are fully formatted at completion.

## Chat bots (Python, beta)

`agent-rt serve` (or `artie serve`) runs Agent RT as a Telegram and/or Discord bot. Name the platforms as one comma-separated argument.

```bash
agent-rt serve telegram,discord
agent-rt serve telegram --config ./agent-rt.yaml --model gpt-5.5
```

Each platform reads `agent_rt_settings.serve.<platform>` from the YAML config:

```yaml
agent_rt_settings:
  serve:
    telegram: {bot_token: os.environ/TELEGRAM_BOT_TOKEN, allowed_users: [123456789]}
    discord: {bot_token: os.environ/DISCORD_BOT_TOKEN, allowed_users: ["123456789012345678"]}
```

- The bot answers only the user ids in `allowed_users`; anyone else is told their id so the owner can add it.

- Bots have no workspace, file, or command tools, because anyone can message a public bot. Each chat keeps its own conversation in memory until the process stops; `/new` starts over.

- Telegram uses long polling, so no public URL or webhook is needed. Discord answers direct messages and server messages that mention the bot, without the privileged Message Content intent, and never pings `@everyone`, roles, or users.

- Rejected tokens stop the command with exit code 1; network errors are retried, and tokens are redacted from logs.

## Sandboxes

`SandboxSession` runs commands through a fail-closed backend selected with `agent_rt_settings.sandbox.backend`: `native`, `docker`, `e2b`, `microsandbox`, or `swe-rex`.

- **Native:** runs as a restricted account's primary gid unless `sandbox.gid` is set and never keeps the caller's gid. The TypeScript backend uses `setpriv --clear-groups --no-new-privs`.

- **Docker:** all capabilities dropped, `no-new-privileges`, and swap capped to the memory limit. Set `sandbox.docker_user` for a non-root user and `sandbox.docker_read_only: true` for a read-only root filesystem.

- **Lifetime:** timeouts and cancellation kill the process group (native) or container (Docker).

- **Limits:** output is capped while read (default 16 MiB) and zero memory or process limits are rejected.

- **Cleanup and snapshots:** call `await session.close()` to release cloud sandboxes. Snapshots capture only the in-process workspace and session settings, not backend disks.

## Skills and registration safety

`SkillRegistry` loads Claude Code and Codex style `SKILL.md` directories and collections. Python exposes `install_from_path` and `install_directory`.

```ts
import { SkillRegistry } from "agent-rt/extensions";

const skills = new SkillRegistry();
await skills.installFromPath(".claude/skills/review", { activate: true });
await skills.installDirectory(".agents/skills");
```

Registration is fail-closed: deterministic scanning always runs and externally sourced registrations require a configured Decision model before the registry is mutated.

## Migration entry points

Prepend the package name to a supported upstream import path. See the [migration guide](https://agent-rt.pro-genai.com/docs/guides/migration.html) for the supported subset and semantic gaps.

| Upstream | Python | TypeScript |
| --- | --- | --- |
| `openai` | `agent_rt.openai` | `agent-rt/openai` |
| `anthropic / @anthropic-ai/sdk` | `agent_rt.anthropic` | `agent-rt/@anthropic-ai/sdk` |
| `langchain_openai / @langchain/openai` | `agent_rt.langchain_openai` | `agent-rt/@langchain/openai` |
| `llama_index.llms.openai / @llamaindex/openai` | `agent_rt.llama_index.llms.openai` | `agent-rt/@llamaindex/openai` |
| `agents / @openai/agents` | `agent_rt.agents` | `agent-rt/@openai/agents` |
| `autogen_agentchat.agents` | `agent_rt.autogen_agentchat.agents` | Python only |
| `crewai` | `agent_rt.crewai` | Python only |
| `deepagents` | `agent_rt.deepagents` | `agent-rt/deepagents` |
| `litellm` | `agent_rt.litellm` | Python only |
| `openrouter / @openrouter/sdk` | `agent_rt.openrouter` | `agent-rt/@openrouter/sdk` |

**Previous:** [Vector databases](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html)

**Next:** [Features & defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)

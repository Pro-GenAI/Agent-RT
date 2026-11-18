> **Canonical HTML page:** [Agent RT vs LangChain vs LlamaIndex — Comparison](https://agent-rt.pro-genai.com/docs/reference/comparison)

> Compare Agent RT with LangChain and LlamaIndex across runtime controls, guardrails, APIs, sandboxing, checkpoints, vector databases, latency, and memory usage, and the Agent RT CLI with Codex, Claude Code, opencode, Copilot CLI, Pi, and OpenHuman.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Reference/Comparison
 Reference

# Framework
comparison

A concise comparison of Agent RT, LangChain, and LlamaIndex across runtime features plus checked-in installation footprint, latency, and memory measurements, and of the Agent RT CLI against six other agent CLIs, including Pi and OpenHuman.

 **How to read this table**✅ means built in, ◐ means available or integration-dependent, and — means it is not a primary built-in focus. Benchmark figures are scenario-specific measurements from the repository comparison suite, not universal framework rankings. Python fresh-install footprint is the 2026-10-03 Python 3.13.13 clean-`uv venv` measurement above the empty-venv baseline. TypeScript fresh-install footprint is the 2026-10-03 Node.js 22.22.2/npm 10.9.7 clean-project measurement of logical `node_modules` bytes. Both include transitive dependencies.

## ⚖️ Quick comparison

|   | ⚡ Agent RT | 🦜 LangChain | 🗂️ LlamaIndex |
| --- | --- | --- | --- |
| **🎯 Primary focus** | 🧠 Agent runtime | 🧩 App/orchestration framework | 📚 Data & RAG framework |
| ** Python + TypeScript** | ✅ | ✅ | ✅ |
| **🛠️ Tools & agent loop** | ✅ Built-in | ✅ | ✅ |
| **🔐 Permissions & approvals** | ✅ Built-in | ◐ | ◐ |
| **🧰 API & CLI tools** | ✅ Built-in (optional packages) | — | — |
| **🌐 OpenAI + Anthropic compatible API** | ✅ Built-in server adapters | ◐ Integration-dependent | ◐ Integration-dependent |
| **🛡️ Agent Action Guard** | ✅ Built-in integration (opt-in) | — | — |
| **🧠 Decision-model control gates** | ✅ Tool pruning, memory/retrieval gates, failure classification | — | — |
| **🧠 Harmful-memory guard** | ✅ Decision-model screening before memory persistence and retrieval exposure; detects harmful-action influence, misleading authority, instruction hijacking, and safeguard bypass attempts | ◐ Application/integration-defined | ◐ Application/integration-defined |
| **🔎 Tool/skill pre-registration safety scan** | ✅ Deterministic scan of names, descriptions, schemas/metadata, skill frontmatter/instructions, embedded tools, and string resources | ◐ Application/integration-defined | ◐ Application/integration-defined |
| **🚧 Decision approval for external tools/skills** | ✅ Fail-closed Decision-model scan before filesystem skill activation, repeated whenever a trusted skill's files change; checked registration path for untrusted tools | ◐ Application/integration-defined | ◐ Application/integration-defined |
| **✂️ Tool catalog pruning before model call** | ✅ Built-in | ◐ Integration-dependent | ◐ Integration-dependent |
| **📦 Sandboxing** | ✅ Fail-closed backend selection | ◐ | ◐ |
| **♻️ Budget-preserving checkpoints** | ✅ Resume with prior turn/tool/token budgets | ◐ | ◐ |
| **💾 Memory & checkpoints** | ✅ Built-in | ✅ | ✅ |
| **🗃️ Switch vector DBs** | ✅ Simple env/registry switch | ◐ Integration-dependent | ◐ Integration-dependent |
| **🔌 Provider-neutral** | ✅ | ✅ | ✅ |
| **🪶 Lightweight runtime focus** | ✅ | ◐ | ◐ |
| ** Fresh install footprint** | **46.43 MiB** | 69.87 MiB | 219.91 MiB |
| ** Warm latency** | **1.56 ms** | 9.03 ms | 4.26 ms |
| ** Runtime memory** | **+5.86 MiB** | +19.77 MiB | +22.42 MiB |
| ** Fresh install footprint** | **1.24 MiB** | 82.44 MiB | 60.38 MiB |
| ** Warm latency** | **1.85 ms** | 2.60 ms | 2.23 ms |
| ** Runtime memory** | **+14.33 MiB** | +17.27 MiB | +16.79 MiB |

📊 For benchmark methodology, environment, raw measured outputs, startup results, tool-loop measurements, schema-token analysis, and CLI harness results, see the

[reproducible framework benchmarks ↗](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/comparison/README.md)

.

## ⌨️ Agent CLI comparison

The `agent-rt` terminal client compared with other agent CLIs. Each CLI was called as its plain one-shot prompt command with no extra flags and pointed at a deterministic localhost mock model, so no real model API was called. The figures are 30 clean sequential runs per CLI plus an 8-way parallel batch, measured on 2026-10-29 on 16-thread Linux x86_64.

| CLI | Command | Median time | Peak RAM | Input tokens | Cost / prompt | 8 parallel |
| --- | --- | --- | --- | --- | --- | --- |
| **⚡ Agent RT** | `agent-rt -p "hi"` | **185 ms** | **36.5 MiB** | **851** | **$0.0026** | **338 ms** |
| **Codex** | `codex exec "hi"` | 283 ms | 194.1 MiB | 8,420 | $0.0253 | 318 ms |
| **Claude Code** | `claude -p "hi"` | 275 ms | 202.7 MiB | 19,229 | $0.0577 | 1,105 ms |
| **opencode** | `opencode run "hi"` | 683 ms | 386.3 MiB | 5,604 | $0.0169 | 2,188 ms |
| **Copilot CLI** | `copilot -p "hi"` | 1,595 ms | 341.1 MiB | 13,985 | $0.0420 | 2,796 ms |
| **Pi** | `pi -p "hi"` | 312 ms | 125.5 MiB | 1,330 | $0.0040 | 561 ms |
| **OpenHuman** | `openhuman-core` headless turn¹ | 492 ms | 95.9 MiB | 4,208 | $0.0127 | 688 ms |

**What this measures**Startup and per-prompt harness overhead, not task quality or capability. Input tokens are a same-tokenizer estimate (`tiktoken` `o200k_base`) of the system prompt, messages and tool schemas each CLI sends. Part of the gap comes from each CLI's default tool set (Agent RT sends 8 tools; the others 4–16). ¹OpenHuman's release one-shot command cannot run a turn without a booted core, so each run boots `openhuman-core run --headless-api`, sends one JSON-RPC turn and stops, as OpenHuman's own benchmark does; it is built to serve many agents from one warm core, which this cold-start test does not measure. Cost assumes $3 per million input and $15 per million output tokens, with no cache discount. Peak RAM covers each CLI's whole process tree.

## 🧭 Pi and OpenHuman features

[Pi ↗](https://github.com/earendil-works/pi)

and

[OpenHuman ↗](https://github.com/tinyhumansai/openhuman)

are open-source agent harnesses with very different scopes. This comparison comes from each project's documentation and source as of 2026-10-29. ✅ built in · ◐ partial, via extension or add-on · — not provided.

|   | Agent RT | Pi | OpenHuman |
| --- | --- | --- | --- |
| **Language / runtime** | Python and TypeScript (aligned) | TypeScript (Node ≥ 22.19) | Rust (single process) |
| **License** | MIT | MIT | GPL-3.0 |
| **Primary shape** | Agent runtime library + CLI + web UI | Minimal coding-agent CLI + TS SDK | Desktop app + headless core + Rust library |
| **One-shot CLI** | ✅ `agent-rt -p` | ✅ `pi -p`, JSON mode, RPC mode | ◐ `openhuman inference agent_chat` (needs a booted core in 0.64.15) |
| **Embeddable library** | ✅ Python + TS packages | ✅ TS SDK (`pi-agent-core`, `pi-ai`) | ✅ Rust (`openhuman-embed`) |
| **Providers** | OpenAI, Anthropic, any compatible endpoint, custom | 30+ built in, plus any compatible endpoint (`models.json`) | 26 BYOK presets, local runtimes (Ollama, LM Studio, MLX), managed subscription |
| **Drop-in SDK migration** | ✅ One-import swap for OpenAI, Anthropic, LangChain, LlamaIndex, OpenAI Agents and others; `agent-rt migrate` | — | — |
| **Permissions / approvals** | ✅ Built in | — None built in; relies on project trust, containers or extensions | ✅ Approval gate |
| **Action blocking** | ✅ Guard: composable blockers, incl. Agent Action Guard | ◐ Via extension hooks | ◐ Prompt-injection screening on inputs; approval for irreversible actions |
| **Sandboxing** | ✅ Fail-closed backend selection | ◐ Via containers or the Gondolin extension | ✅ OS jail or Docker |
| **Tool catalog pruning** | ✅ Decision-model (Jev-compatible) pruning + `tool_search` | ◐ `codemode` (a script calls tools; the model sees only its output) | ✅ Jev tool search |
| **Context reduction** | Smaller default catalog; Decision-gated memory/retrieval | Compaction | RLM tool-result compression |
| **Memory** | ✅ mem0, Zep, Letta, Supermemory, memU, framework stores | — (sessions only) | ✅ Built-in recall every turn, with citations; swappable engines |
| **Checkpoints / resume** | ✅ Budget-preserving checkpoints | ✅ Sessions with branching | ✅ Durable workflows that resume after a pause |
| **Sub-agents / plan mode** | ✅ Agent-as-tool, plans | — By design (build via extensions) | ✅ Sub-agents, agent teams, Plan vs Build |
| **MCP** | ✅ stdio + streamable HTTP; CLI reads Codex and Claude Code server configs | ✅ stdio + streamable HTTP | ✅ |
| **Skills / extensions** | ✅ Skills from `.agents/`, `.claude/`, `.codex/`; re-scanned when files change | ✅ Extensions, skills, prompt templates, themes, npm/git packages | ✅ Skills, loadable Rust modules |
| **Messaging channels** | ◐ Telegram, Discord (Python, beta) | — (separate `pi-chat` project) | ✅ 14 channels |
| **Browser / desktop control** | — | — | ✅ Browser and desktop apps |
| **OAuth app integrations** | — | — | ✅ 119 apps via Composio |
| **Voice** | — | — | ✅ Live voice agent, dictation, TTS |
| **Web UI** | ✅ `agent-rt web` (time-travel debugger) | — (terminal UI) | ✅ Desktop app |
| **OpenAI/Anthropic-compatible API server** | ✅ | — | ◐ Own JSON-RPC API |

Pi is the minimalist option: a small core with few tools and no permission system, extended by the user. OpenHuman is the maximalist one: a personal-assistant platform with memory, channels, browser and desktop control, voice and agent fleets. Agent RT sits between them as a runtime with production controls (Guard, approvals, sandboxing, Checkpoints) and Python and TypeScript migration paths. It has none of OpenHuman's end-user assistant features (browser control, voice, OAuth app catalog).

[Previous← Features & defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)



[NextArchitecture →](https://agent-rt.pro-genai.com/docs/reference/architecture.html)

> **Canonical HTML page:** [Agent RT — Lightweight AI Agent Runtime for Python & TypeScript](https://agent-rt.pro-genai.com/)

> Agent RT is a lightweight AI agent runtime for Python and TypeScript with tools, streaming, memory, policy, sandboxing, orchestration, and observability.

**License:** MIT · **npm install footprint:** 1.24 MiB (site benchmark)

# Control the loop. Ship the agent.

Agent RT is a lightweight, provider-neutral runtime for Python and TypeScript. Tools, streaming, policy, memory, sandboxing, orchestration, and observability stay explicit—so your agent remains understandable when the prototype becomes a system.

- [66×smaller install than LangChain.js](https://agent-rt.pro-genai.com/#benchmarks)

- [5.8×less framework overhead than LangChain in Python](https://agent-rt.pro-genai.com/#benchmarks)

- [65+ third-party repos pass their tests after migration](https://agent-rt.pro-genai.com/#migration)

[Build your first agent →](https://agent-rt.pro-genai.com/#quickstart)

[Explore the runtime](https://agent-rt.pro-genai.com/#architecture)

`pip install agent-rt`

[Docker Hub — Python](https://hub.docker.com/r/praneeth1v/agent-rt-python)

[Docker Hub — TypeScript](https://hub.docker.com/r/praneeth1v/agent-rt-typescript)

### Lifetime package downloads

[![PyPI total downloads](https://agent-rt.pro-genai.com/badges/pypi)](https://pepy.tech/projects/agent-rt)

[![npm total downloads](https://agent-rt.pro-genai.com/badges/npm)](https://www.npmjs.com/package/agent-rt)

### Example: approval-gated tool execution

The following illustrative trace shows an agent reading data, checking policy, saving a checkpoint, and stopping before a sensitive external action:

```text
run_42: agent=assistant, tools=2, max_turns=8
01 text_delta: "Checking the SLA first…"
02 tool_call_delta: search_docs(query="SLA")
03 policy.check: side_effect=read, allow
04 tool_completed: search_docs returned 3 results
05 checkpoint: turn_02 committed
06 approval.request: send_email awaiting human review
07 termination_reason: waiting_for_approval
```

  **1,284** tokens **6** spans

  Python: **3.10—3.14**
Node.js: **20 · 22 · 24 · 26**
License: **MIT**
Transport: **WebSocket + HTTP/SSE**
Platforms

- Linux

- Windows

- macOS Intel

- macOS Apple Silicon

- Raspberry Pi

- Tool calling

- Streaming

- Policy & approvals

- Memory

- Checkpoints

- Deterministic replay

- Sandboxed execution

- MCP + connectors

- Vector DB retrieval

- Multi-agent orchestration

- Tracing

- Token & cost ledgers

- OpenAI-compatible API

    01 Runtime

## Agent code is easy. Runtime semantics are not.

A production agent is more than a prompt and an API call. It has to decide what can run, preserve state, recover from failure, stream useful progress, keep side effects bounded, and explain what happened afterward. Agent RT turns those concerns into explicit contracts instead of hidden framework behavior.

    01

### Explicit control

Termination, retries, deadlines, budgets, routing, and tool visibility are runtime decisions you can inspect and test.

   02

### Security boundaries

Tools, secrets, sandboxes, approvals, and data egress sit behind policy—not model-generated intent.

   03

### Portable primitives

Python and TypeScript share the same conceptual runtime while provider-specific behavior stays at the edge.

     02 Capabilities

## A small core with serious control surfaces.

Use the pieces you need. Keep optional integrations optional.

     TRANSPORT

### Streaming that stays in the loop

Provider-neutral text, reasoning, status, and tool-call deltas flow through the same multi-turn runtime semantics.

  Persistent Responses WebSocket fast path · HTTP/SSE fallback

    POLICY

### Tools are security boundaries

Typed schemas, deny-overrides permissions, model guardrails, approval gates, rate limits, deadlines, and side-effect classes execute outside the model.

    STATE

### Resume without guessing

Versioned workflow state, session memory, durable checkpoints, event streams, idempotency, and deterministic replay. Long-term memory plugs in from mem0, Zep, Letta, Supermemory, or memU.

    EXECUTION

### Sandbox by contract

Command and code execution runs behind explicit resource and network policies with fail-closed backend selection.

    OBSERVABILITY

### Trace what the agent actually did

Hierarchical traces, privacy-aware logs, runtime metrics, token and cost ledgers, plus replay bundles for reproducible debugging.

     ORCHESTRATION

### Compose agents without losing the plot

Agents-as-tools, isolated subagents, supervisor/worker teams, routers, swarms, map/reduce, and speculative branches share bounded execution contracts.

     OPENAI-COMPATIBLE API

### Expose your agent through the API clients already speak

The optional Python server package maps configured agents to OpenAI model IDs while keeping Agent RT tools, guardrails, routing, and run limits in control.

  `pip install "agent-rt[api]"`

  GET`/v1/models`*models*
POST`/v1/chat/completions`*SSE*
POST`/v1/responses`*SSE · WS*
 optional bearer auth · FastAPI / Uvicorn

    03 Decision models

## A second model that decides.
It never writes the reply.

Decision models served over a Jev-compatible API, or the OpenAI-compatible Decisions API, answer typed choice and score questions with calibrated outputs instead of generating text. Agent RT uses them as optional gates that feed deterministic runtime policy. They never replace that policy. Prefer to self-host? Run an open-source Jev-compatible endpoint with

[Ollaya](https://github.com/ollaya-dev/ollaya)

and point `decision_model.base_url` at it.

- **Tool safety screening.** Unsafe tool definitions are removed before the model ever sees them, and stay out of `tool_search` too.

- **Relevance pruning.** Large local and MCP tool inventories shrink to the subset each turn needs; required tools are always kept.

- **Memory and retrieval checks.** Memories are scored for harmful or misleading influence, and retrieved documents are reranked for relevance and trust.

- **Failure classification.** Refusals and reported errors map to typed failure dispositions.

    decision gate · illustrative

1. tools

Local, deferred, and MCP-discovered tools

2. decide

Safety screen asks a typed question: `block delete_record(scope="all")?`

3. deny

p = 0.97 ≥ threshold 0.85, so the tool is removed

4. select

Relevance picks this turn's subset; `tool_search` stays available

5. model

The generative model sees only safe, relevant tools

   Jev-compatible endpoint · no model weights bundled

     04 Red team

## A malicious model in the loop.
Zero bypasses.

Every CI run swaps the model for a deterministic attacker and checks that both runtimes contain it. No real LLM is called.

  **6** attack scenarios
 **2** runtimes: Python + TypeScript
 **0** real LLM calls: fully offline
 **0** bypasses

    security_attack_scan · offline

1. AI-001Prompt injection forges a hidden tool callblockedexecution-time visibility recheck

2. AI-002Malformed tool argumentsblockedschema validation before side effects

3. AI-003Permission-engine bypassblockeddeny-overrides permission policy

4. AI-004Human-approval bypasspausedrun waits for human approval

5. AI-005Restricted-data exfiltrationblockedegress guardrail

6. AI-006Denial-of-wallet token floodstoppedrun-level token budget

   passed=6 failed=0 · each runtime

[Run it yourself ↗](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/scripts/security_attack_scan.py)

05 Migration

## Change one import.
Keep your code.

Prepend `agent_rt.` in Python or `agent-rt/` in TypeScript. Common LangChain, LlamaIndex, OpenAI, and Anthropic entry points keep their shape and run through Agent RT. Or run `uvx agent-rt migrate` / `npx agent-rt migrate`: it keeps the rewrite only if no new test fails.

     app.py

```
-from langchain_openai import ChatOpenAI
+from agent_rt.langchain_openai import ChatOpenAI

 llm = ChatOpenAI(model="your-model")
 response = await llm.ainvoke("Hello")
 print(response.content)
```

1 line changedLangChain · LlamaIndex · OpenAI · Anthropic

    app.js

```
-import OpenAI from "openai";
+import OpenAI from "agent-rt/openai";

 const client = new OpenAI();
 const response = await client.chat.completions.create({
   model: "your-model",
   messages: [{ role: "user", content: "Hello" }],
 });
```

1 line changedalso `agent-rt/@langchain/openai`

  **65+**

third-party repositories pass their own test suites both before and after migration, in Agent RT's ongoing Python and TypeScript field testing. Compatibility is a focused subset of each SDK, not full parity.

[Read the migration guide ↗](https://agent-rt.pro-genai.com/docs/guides/migration-reference)

Migrated? Add the opt-in

[Runs on Agent RT badge](https://agent-rt.pro-genai.com/docs/guides/migration.html#badge)

.

  See the repositories 65+ migrated repositories still pass

### Anthropic · Python 13

- [arashbahrami20261-tech/ai-researcher](https://github.com/arashbahrami20261-tech/ai-researcher/tree/c9f3c66)

- [ArchanaChetan07/validation-lab-capacity-planning-copilot](https://github.com/ArchanaChetan07/validation-lab-capacity-planning-copilot/tree/515cae8)

- [gauravsoodtech/shortlist-api](https://github.com/gauravsoodtech/shortlist-api/tree/f5b64b7)

- [guptrishi01/rally-ai](https://github.com/guptrishi01/rally-ai/tree/4a7f6e1)

- [henry-hai/arthouse-ops](https://github.com/henry-hai/arthouse-ops/tree/e82acce)

- [JuanCamiloMendoza99/mcp-document-chat](https://github.com/JuanCamiloMendoza99/mcp-document-chat/tree/4a591af)

- [LikithSh3tty/Askvane](https://github.com/LikithSh3tty/Askvane/tree/2ed3263)

- [matipirro/taskmanager-python](https://github.com/matipirro/taskmanager-python/tree/e044a3a)

- [Mini34/trailhead-claude-support-api](https://github.com/Mini34/trailhead-claude-support-api/tree/5c85c11)

- [NickTrapp/poker-coach](https://github.com/NickTrapp/poker-coach/tree/3469232)

- [PranaPragada7/Study-Tutor-Agent](https://github.com/PranaPragada7/Study-Tutor-Agent/tree/8bbbb84)

- [rubenG1009/jobscout](https://github.com/rubenG1009/jobscout/tree/0cdbc16)

- [thomas-tahk/itsm-ticket-triage](https://github.com/thomas-tahk/itsm-ticket-triage/tree/8de7130)

### OpenAI · Python 13

- [1dg618/pytest-fakellm](https://github.com/1dg618/pytest-fakellm/tree/03feb6e)

- [brianshimkus/Ticket-Triage-Lab](https://github.com/brianshimkus/Ticket-Triage-Lab/tree/5182dd0)

- [dx-corp/mocktopus](https://github.com/dx-corp/mocktopus/tree/ddfa010)

- [EhsaN-AI13/ai-engineering-agent](https://github.com/EhsaN-AI13/ai-engineering-agent/tree/97de57f)

- [eliv1982/customer-review-mcp-assistant](https://github.com/eliv1982/customer-review-mcp-assistant/tree/6499d0d)

- [eliv1982/kitchen-helper-telegram-bot](https://github.com/eliv1982/kitchen-helper-telegram-bot/tree/2378115)

- [enoorii/contract-clause-reviewer](https://github.com/enoorii/contract-clause-reviewer/tree/9904684)

- [hajar-benhadj/AI-API-TEST-Generator](https://github.com/hajar-benhadj/AI-API-TEST-Generator/tree/a88ddd5)

- [jamps3/LeetIRCPythonBot](https://github.com/jamps3/LeetIRCPythonBot/tree/0088c81)

- [neehanthreddym/ai-sales-crm-assistant](https://github.com/neehanthreddym/ai-sales-crm-assistant/tree/33f0f04)

- [pitercoding/ai-news-scraper](https://github.com/pitercoding/ai-news-scraper/tree/3c99ed4)

- [q1nn2/ai_qahelper](https://github.com/q1nn2/ai_qahelper/tree/2ae5c3d)

- [Tanush1206/llm-utility-lab](https://github.com/Tanush1206/llm-utility-lab/tree/b39e046)

### CrewAI · Python 6

- [Floe-Labs/floe-guard](https://github.com/Floe-Labs/floe-guard/tree/f9bf99a)

- [Harshit-J004/toolguard](https://github.com/Harshit-J004/toolguard/tree/1b4d4ac)

- [mubashir72/AccessDesk-Multi-Agent-AI](https://github.com/mubashir72/AccessDesk-Multi-Agent-AI/tree/979f94c)

- [rvira/triage-desk](https://github.com/rvira/triage-desk/tree/6be076a)

- [sumairkhan/WebIntelX-AI](https://github.com/sumairkhan/WebIntelX-AI/tree/ec540b1)

- [Tonioan-0/Tiered-crew](https://github.com/Tonioan-0/Tiered-crew/tree/452a34f)

### LangChain · Python 6

- [alexeymasalykin/ai-assistant-eduflow](https://github.com/alexeymasalykin/ai-assistant-eduflow/tree/23b23b6)

- [eliv1982/ai-agent-toolbox](https://github.com/eliv1982/ai-agent-toolbox/tree/97fb22c)

- [lyyxyy/PythonPlantsVsZombies](https://github.com/lyyxyy/PythonPlantsVsZombies/tree/048eae4)

- [OlajuwonOkeniyi/pdf-quiz-generator](https://github.com/OlajuwonOkeniyi/pdf-quiz-generator/tree/d0bda31)

- [zakirkun/guardian-cli](https://github.com/zakirkun/guardian-cli/tree/51864a8)

- [zhaozhenyu-dev/llm-app-portfolio](https://github.com/zhaozhenyu-dev/llm-app-portfolio/tree/6b073fe)

### OpenAI Agents · Python 6

- [ibrahimsamak/multi-agent-AI-travel-planner](https://github.com/ibrahimsamak/multi-agent-AI-travel-planner/tree/16be0ce)

- [KMdotcom/agent-handoff-kit](https://github.com/KMdotcom/agent-handoff-kit/tree/b837831)

- [panaversity/agentfactory-manufacturing](https://github.com/panaversity/agentfactory-manufacturing/tree/a82c657)

- [rafaelpierre/openai-agents-redis](https://github.com/rafaelpierre/openai-agents-redis/tree/376603a)

- [tahasiddiquii/soc-triage-agent](https://github.com/tahasiddiquii/soc-triage-agent/tree/eb7c0d5)

- [Umairkhan2324/trilith](https://github.com/Umairkhan2324/trilith/tree/9db22e4)

### OpenAI Agents · TypeScript 6

- [dcramer/peated](https://github.com/dcramer/peated/tree/aee6890)

- [dmitryostrov/stark-document-translator](https://github.com/dmitryostrov/stark-document-translator/tree/22c0657)

- [open-software-network/os-clovy](https://github.com/open-software-network/os-clovy/tree/8fed7ac)

- [rajudandigam/agent-inspect](https://github.com/rajudandigam/agent-inspect/tree/09838a3)

- [runtypelabs/persona](https://github.com/runtypelabs/persona/tree/8f79eaa)

- [vaultys/VaultysClaw](https://github.com/vaultys/VaultysClaw/tree/038f94d)

### LlamaIndex · Python 5

- [benmaster82/Kwipu](https://github.com/benmaster82/Kwipu/tree/01dd7d4)

- [Gaallard/API-RAG-LlamaIndex](https://github.com/Gaallard/API-RAG-LlamaIndex/tree/b4b6ded)

- [iAbhyuday/shodh](https://github.com/iAbhyuday/shodh/tree/f803964)

- [PhoebeHuPing/memai](https://github.com/PhoebeHuPing/memai/tree/068b151)

- [sapphireyyy/llamaIndex_rag](https://github.com/sapphireyyy/llamaIndex_rag/tree/6725aeb)

### AutoGen · Python 4

- [Divya23490/regulatory-impact-analyst](https://github.com/Divya23490/regulatory-impact-analyst/tree/99e9579)

- [Hariharan17194/autogen-support-desk](https://github.com/Hariharan17194/autogen-support-desk/tree/52c553d)

- [natrajexplore/AutoGen_Agentic-AI](https://github.com/natrajexplore/AutoGen_Agentic-AI/tree/767a752)

- [themaker00001/chaos-bringer](https://github.com/themaker00001/chaos-bringer/tree/13ff3f6)

### LangChain · TypeScript 4

- [Martin-Bruel/nestjs-langchain](https://github.com/Martin-Bruel/nestjs-langchain/tree/4b667b9)

- [maunappl8/recursive-langgraph-agent](https://github.com/maunappl8/recursive-langgraph-agent/tree/2c3da65)

- [SAP/langchainjs-integration-for-sap-hana-cloud](https://github.com/SAP/langchainjs-integration-for-sap-hana-cloud/tree/bed48b5)

- [smartinezai/friendship-points-api](https://github.com/smartinezai/friendship-points-api/tree/f9465c2)

### OpenAI · TypeScript 4

- [archydre/lowdeep](https://github.com/archydre/lowdeep/tree/0e6c0f1)

- [blaxsior/vitest-codegen-with-openai](https://github.com/blaxsior/vitest-codegen-with-openai/tree/8b40f85)

- [icernigoj/llmao](https://github.com/icernigoj/llmao/tree/e90a28d)

- [thomaslnx/AIChatBackend](https://github.com/thomaslnx/AIChatBackend/tree/f46dc46)

### Anthropic · TypeScript 2

- [hanul442/gnomon_analytics](https://github.com/hanul442/gnomon_analytics/tree/02f90b7)

- [SFARPak/AliFullStack](https://github.com/SFARPak/AliFullStack/tree/c34f5cf)

Public repositories at the tested commit. Listed as test targets only; inclusion is not an endorsement by their authors.

    06 Coding-agent CLI

## Your agent CLI.
Our runtime.

Put `agent-rt` in front of a Claude Code, Codex, opencode, or Copilot CLI one-shot command. It accepts that tool's own flags and reads its model, provider, and API-key settings, so no `agent-rt.yaml` is needed.

- **Same runtime.** Guardrails, provider routing, tool execution, and run limits match programmatic Agent RT. Workspace writes need approval.

- **Keys stay yours.** Project files can only choose the model, so a cloned repository cannot send your API key to its own server.

- **API keys only.** Subscription and OAuth sign-ins stay with the official clients.

- **Try it offline.** `agent-rt demo` runs a scripted prompt-injection attack with no API key; `agent-rt web --demo` replays it in a time-travel debugger.

- **Your project setup works.** `AGENTS.md`, `CLAUDE.md`, skills in `.agents/`, `.claude/`, and `.codex/`, and MCP servers from Codex and Claude Code configs load as is. Skills are re-scanned whenever their files change.

- **Terminal or browser.** `agent-rt web` shares sessions and per-session `todo` lists with the terminal CLI.

- **Chat bots (beta).** `agent-rt serve telegram,discord` answers allowed users, with no workspace tools.

     terminal

```
$ pip install "agent-rt[cli]"

$ agent-rt claude -p "explain this stack trace" --model opus
$ agent-rt codex exec "explain this stack trace" -m gpt-5.5
$ agent-rt opencode run "explain this stack trace"
$ agent-rt copilot -p "explain this stack trace"
```

reads the tool's own config

[CLI guide ↗](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html#reuse-another-agent-cli)

07 Architecture

## Provider-neutral inside.
Specific at the edge.

Keep application logic stable while models, tools, storage, and deployment backends evolve independently.

    Your app  **Agent configuration** **Business logic** **User experience**

   Agent RT  AgentLoop ToolRegistry Context Policy Memory Events Replay Evaluation

   Adapters  **Model providers** **MCP + connectors** **Sandbox backends** **Vector DB + storage + telemetry**

   stable runtime contract  replaceable integration

     08 Benchmarks

## Measure framework overhead.
Keep the model out of it.

Agent RT ships reproducible localhost benchmarks so runtime overhead can be inspected separately from provider and network latency.

    PYTHON · 30 RUNS HTTP/SSE BASELINE
 1.56ms

### Warm completion

Median framework/transport overhead against a deterministic localhost OpenAI-compatible mock.

   TYPESCRIPT · 30 RUNS HTTP/SSE BASELINE
 1.85ms

### Warm completion

Median framework/transport overhead against the same class of deterministic local mock.

   FEATURE BENCH · 30 RUNS PYTHON / TYPESCRIPT
  **3.34ms** — Python
 **2.70ms** — TypeScript

### Tool round trip

One two-request tool cycle with serialized JSON tool output and common wire-level work.

    CROSS-FRAMEWORK COMPARISON

### Same mock. Same workload. Alternatives included.

  Agent RT Alternative *lower is better*

     PYTHON **Runtime + feature + install footprint**
 runtime/features: 30 runs · install: 2026-10-03

#### Warm completion

 retained HTTP/SSE runtime baseline
  Agent RT  **1.56 ms**
  LangChain  **9.03 ms**
  LlamaIndex  **4.26 ms**

#### Tool round trip

HTTP/SSE feature benchmark
  Agent RT  **3.34 ms**
  LangChain  **9.81 ms**
  LlamaIndex  **7.31 ms**

#### Fresh install footprint

 clean uv venv · 2026-10-03
  Agent RT  **46.43 MiB**
  LangChain  **69.87 MiB**
  LlamaIndex  **219.91 MiB**

    TYPESCRIPT **Runtime + feature + install footprint**
 runtime/features: 30 runs · install: 2026-10-03

#### Warm completion

 retained HTTP/SSE runtime baseline
  Agent RT  **1.85 ms**
  LangChain.js  **2.60 ms**
  LlamaIndex.TS  **2.23 ms**

#### Tool round trip

HTTP/SSE feature benchmark
  Agent RT  **2.70 ms**
  LangChain.js  **3.33 ms**
  LlamaIndex.TS  **2.96 ms**

#### Fresh install footprint

 clean npm install · 2026-10-03
  Agent RT  **1.24 MiB**
  LangChain.js  **82.44 MiB**
  LlamaIndex.TS  **60.38 MiB**

Bars are normalized within each metric and language; printed values use the native unit for that metric.

    AGENT CLI COMPARISON

### Same one-line prompt. Seven agent CLIs.

  Agent RT Alternative *lower is better*

     TERMINAL CLIs **Time per prompt**
 30 runs · 2026-10-29

#### Median one-shot time

 sequential, cold start
  Agent RT  **185 ms**
  Codex  **283 ms**
  Claude Code  **275 ms**
  opencode  **683 ms**
  Copilot CLI  **1,595 ms**
  Pi  **312 ms**
  OpenHuman  **492 ms**

#### Eight copies at once

 batch wall time
  Agent RT  **338 ms**
  Codex  **318 ms**
  Claude Code  **1,105 ms**
  opencode  **2,188 ms**
  Copilot CLI  **2,796 ms**
  Pi  **561 ms**
  OpenHuman  **688 ms**

    TERMINAL CLIs **Footprint per prompt**
 30 runs · 2026-10-29

#### Peak memory

 whole process tree
  Agent RT  **36.5 MiB**
  Codex  **194.1 MiB**
  Claude Code  **202.7 MiB**
  opencode  **386.3 MiB**
  Copilot CLI  **341.1 MiB**
  Pi  **125.5 MiB**
  OpenHuman  **95.9 MiB**

#### Input tokens sent

 system prompt + tools
  Agent RT  **851**
  Codex  **8,420**
  Claude Code  **19,229**
  opencode  **5,604**
  Copilot CLI  **13,985**
  Pi  **1,330**
  OpenHuman  **4,208**

Each CLI answered `hi` against a localhost mock model, called as its plain one-shot command. OpenHuman boots its headless core for each turn, as its own benchmark does.

[Full table and feature comparison](https://agent-rt.pro-genai.com/docs/reference/comparison.html#agent-cli)

.

  Methodology

Runtime and feature values are checked-in 30-run medians captured on September 28, 2026 against deterministic localhost OpenAI-compatible mocks. Warm completion comes from the retained historical HTTP/SSE runtime baseline; tool round trip comes from the HTTP/SSE feature benchmark. Fresh-install footprints were measured on October 3, 2026 and include transitive dependencies: Python uses clean Python 3.13.13 uv environments with copy mode and reports logical bytes above the empty-venv baseline; TypeScript uses clean Node.js 22.22.2/npm 10.9.7 projects and reports logical node_modules bytes, with Agent RT installed from its npm-pack tarball. Agent RT's runtime runner now defaults to the persistent Responses WebSocket path, so new WebSocket runs are not compared against that historical baseline until a new 30-run WebSocket baseline is promoted. These measurements are scenario-specific framework/adapter overhead, not model latency or an overall framework score. The agent CLI comparison uses 30 sequential runs per CLI plus an eight-way batch, measured on October 29, 2026 on 16-thread Linux x86_64 with each CLI's default tools and real credentials removed; it measures per-prompt harness overhead, not task quality.

    09 Quickstart

## One runtime.
Two languages.

Start with the provider-neutral loop, then add tools, policies, storage, sandboxes, or orchestration only when the application needs them.

Moving an existing app?

[Change one import](https://agent-rt.pro-genai.com/#migration)

and keep your LangChain, LlamaIndex, OpenAI, or Anthropic code.

1. 01

Install `agent-rt`.

2. 02

Describe your models and vector stores in `agent-rt.yaml` (secrets via `os.environ/NAME`).

3. 03

Run the agent through `AgentLoop`.

Prefer a container? Pull `praneeth1v/agent-rt-python:latest` or `praneeth1v/agent-rt-typescript:latest` from Docker Hub.

[Docker instructions](https://agent-rt.pro-genai.com/docs/getting-started/installation.html#docker-hub)

.

```
import asyncio
from agent_rt import (
    AgentConfig, AgentLoop, ContentPart,
    ModelMessage, ModelSettings, load_config, load_model,
)

async def main():
    provider = load_model(load_config())
    agent = AgentConfig(
        name="assistant",
        instructions="Be concise and precise.",
        model=ModelSettings(model="my-model"),
    )
    message = ModelMessage(
        role="user",
        content=(ContentPart(type="text", text="Hello."),),
    )

    result = await AgentLoop(provider).run(agent, [message])
    print(result.termination_reason)

asyncio.run(main())
```

```
const {
  AgentLoop,
  loadConfig,
  loadModel,
} = require("agent-rt");

async function main() {
  const provider = await loadModel(await loadConfig());
  const agent = {
    name: "assistant",
    instructions: "Be concise and precise.",
    model: { model: "my-model" },
  };
  const message = {
    role: "user",
    content: [{ type: "text", text: "Hello." }],
  };

  const result =
    await new AgentLoop(provider).run(agent, [message]);
  console.log(result.terminationReason);
}

main();
```

provider-neutral core MIT licensed

### Seven runnable examples. Each fits on one screen.

   01

### Basic agent

One configured provider, one run.

[Python · 25 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/01_basic_agent.py)

[JS · 25 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/01_basic_agent.js)

02

### Tool calling

Register a read-only tool and let the model call it.

[Python · 51 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/02_tool_calling.py)

[JS · 49 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/02_tool_calling.js)

03

### Streaming

Consume text deltas as they arrive.

[Python · 27 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/03_streaming.py)

[JS · 27 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/03_streaming.js)

04

### Structured output

Schema-validated JSON with one repair attempt.

[Python · 45 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/04_structured_output.py)

[JS · 39 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/04_structured_output.js)

05

### Run limits

Bound turns, tool calls, time, and tokens.

[Python · 33 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/05_run_limits.py)

[JS · 31 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/05_run_limits.js)

06

### Tool registry

Namespaces, direct execution, enable and disable.

[Python · 37 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/06_tool_registry.py)

[JS · 43 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/06_tool_registry.js)

07

### Vector DB

Query Chroma, Milvus, Pinecone, Qdrant, or Weaviate.

[Python · 33 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/python/examples/07_vector_db.py)

[JS · 30 lines](https://github.com/Pro-GenAI/Agent-RT/blob/main/src/typescript/examples/07_vector_db.js)

Recently shipped

- Browser chat UI: `agent-rt web`

- Coding-agent CLI modes

- Multiple vector DBs + Bedrock Knowledge Bases

- Tool search + Decision-model tool screening

- PII redaction + custom action blockers

[All commits ↗](https://github.com/Pro-GenAI/Agent-RT/commits/main/)

Ready when your prototype isn't

## Make the runtime part of the design.

Build agents with explicit state, boundaries, recovery, and observability from the first serious version.

[Start with Agent RT →](https://agent-rt.pro-genai.com/#quickstart)

[Read the docs](https://agent-rt.pro-genai.com/docs/)

# Agent RT for TypeScript

**A lightweight, provider-neutral runtime for building AI agents with explicit control over tools, safety, state, and orchestration.**

Agent RT keeps the agent loop independent from any single model vendor or application framework. It ships fully typed entry points and a native `fetch` OpenAI-compatible transport, so you can start with zero provider SDKs and add tools, approvals, memory, sandboxing, retrieval, and multi-agent orchestration as needed.

## Why Agent RT?

- **Provider-neutral by design.** Use OpenAI-compatible endpoints, OpenAI, Anthropic, or your own `ModelProvider`. Optional [Decision models](../docs/Architecture-and-Extensions.md#decision-models) handle control-plane gates such as tool pruning and memory screening.
- **Safety at the execution boundary.** Tool validation, deny-overrides permissions, approvals, capability grants, budgets, deadlines, and guardrails are runtime primitives.
- **Small core, optional peers.** OpenAI, Anthropic, Agent Action Guard, Traceloop, and sandbox integrations remain optional.
- **Built for production paths.** Checkpoints, events, queues, tracing, cost/token ledgers, workspaces, artifacts, retrieval, and orchestration are first-class contracts.
- **Migration-friendly.** Compatibility entry points cover common OpenAI, Anthropic, LangChain, LlamaIndex, and OpenAI Agents usage.

See the reproducible comparisons in [`../comparison/`](../comparison/README.md).

## Requirements

Node 20, 22, 24, or 26.

## Install

```bash
npm install agent-rt
```

Install provider or guardrail packages only when you need them:

```bash
npm install agent-rt openai
npm install agent-rt @anthropic-ai/sdk
npm install agent-rt agent-action-guard
```

The root package and native OpenAI-compatible transport work without those optional peers.

## Quick start

```ts
import {
  AgentLoop,
  loadModel,
  type AgentConfig,
  type ModelMessage,
} from "agent-rt";

const provider = loadModel(process.env, { validate: false });

const agent: AgentConfig = {
  name: "assistant",
  instructions: "Answer clearly and concisely.",
  model: { model: process.env.OPENAI_MODEL! },
};

const messages: ModelMessage[] = [
  {
    role: "user",
    content: [{ type: "text", text: "What can Agent RT do?" }],
  },
];

const result = await new AgentLoop(provider).run(agent, messages);
console.log(
  result.finalResponse?.message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("") ?? ""
);
```

Configure `OPENAI_MODEL` and `OPENAI_API_KEY`; add `OPENAI_BASE_URL` for an OpenAI-compatible endpoint. If multiple provider configurations are present, set `MODEL_PROVIDER=openai` or `MODEL_PROVIDER=anthropic`.

More runnable examples are in [`examples/`](examples/README.md).

## What you can build

Agent RT provides an asynchronous `ModelProvider` contract and a provider-neutral `AgentLoop` for messages, structured output, streaming, tools, routing, fallback, retries, cancellation, deadlines, rate limits, and execution budgets. Around that core:

- **Tools and safety:** registries, argument validation, approvals, permission policies, guardrails, and lifecycle hooks.
- **State:** short- and long-term memory, checkpoints, durable events, background tasks, queues, and scheduling.
- **Context and environments:** retrieval, vector databases, MCP, browser/computer sessions, workspaces, artifacts, and sandboxed code execution.
- **Multi-agent patterns:** planning, critics, supervisor/worker, swarms, hierarchical teams, and map/reduce.
- **Operations:** tracing, metrics, replay, token/cost ledgers, and deployment primitives. CLI and API façade contracts are included; the ready-made API server and terminal CLI are Python-only.

Built-in HTTP vector adapters cover Chroma, Milvus, Pinecone, Qdrant, and Weaviate; FAISS and pgvector plug in through custom registry factories. See [Features and defaults](../docs/website/docs/reference/features-defaults.html) for the full list.

## Migration compatibility

Agent RT publishes compatibility entry points for incremental migrations:

```text
@langchain/openai       -> agent-rt/@langchain/openai
@llamaindex/openai      -> agent-rt/@llamaindex/openai
@openai/agents          -> agent-rt/@openai/agents
openai                  -> agent-rt/openai
@anthropic-ai/sdk       -> agent-rt/@anthropic-ai/sdk
```

Historical aliases such as `agent-rt/langchain`, `agent-rt/llamaindex`, and `agent-rt/anthropic` remain available.

```ts
import OpenAI from "agent-rt/openai";
import Anthropic from "agent-rt/@anthropic-ai/sdk";
import { ChatOpenAI } from "agent-rt/@langchain/openai";
```

Compatibility routes model execution through Agent RT provider contracts instead of creating a second runtime. OpenAI Agents compatibility includes `Agent`, `Runner`, tools, agent-as-tool, and bounded handoffs. AutoGen and CrewAI compatibility are Python-only because Agent RT does not invent JavaScript APIs without an upstream surface to preserve.

See [Migration](../docs/MIGRATION.md) for the supported subset and semantic gaps.

## Provider behavior

`OpenAIModelProvider` uses the native OpenAI-compatible transport by default and supports HTTP/SSE plus Responses WebSocket operation where available. `AnthropicModelProvider` uses the optional official Anthropic SDK.

`loadModel()` selects OpenAI when `OPENAI_BASE_URL` is set, otherwise Anthropic when `ANTHROPIC_BASE_URL` is set. Without either base URL, `OPENAI_MODEL` selects OpenAI. Explicit `MODEL_PROVIDER` overrides auto-detection.

Reasoning/thinking controls are normalized into Agent RT's provider-neutral `ReasoningConfig`, while native providers translate them into vendor-specific request fields.

## Skills and registration safety

`agent-rt/extensions` can load Claude Code/Codex-style `SKILL.md` directories and collections:

```ts
import { SkillRegistry } from "agent-rt/extensions";

const skills = new SkillRegistry();
await skills.installFromPath(".claude/skills/review", { activate: true });
await skills.installDirectory(".agents/skills");
```

Filesystem skill registration is fail-closed: Agent RT performs deterministic scanning and, for externally sourced registrations, can require a configured Decision model before registry mutation.

## Content and provider behaviour

- **Media:** image, PDF/file, and (OpenAI) audio `ContentPart`s are sent as the provider's native blocks; `data` may be an http(s) URL, a `data:` URL, base64, or bytes. Parts a provider cannot carry (for example video) throw instead of being dropped.
- **Anthropic thinking:** thinking blocks are kept on the assistant message (a JSON part with a vendor MIME type) and replayed before `tool_use`, so extended thinking works across tool turns. Anthropic requests default to `max_tokens=4096`; set `maxOutputTokens` for longer outputs.
- **Context trust:** untrusted retrieved/file/observation items reach the model as user-role data; only workflow state and `trust: "trusted"` items are system messages.
- **Validation:** tool arguments and structured output are checked against a JSON Schema subset (types, bounds, `pattern`, `const`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref`, `additionalProperties`) using own-property lookups.
- **Truncation:** a Responses WebSocket `response.incomplete` returns `finishReason: "length"` with usage instead of throwing.

## Sandboxes

`SandboxSession` runs commands through a fail-closed backend selected with `AGENT_RT_SANDBOX_BACKEND` (`native`, `docker`, `e2b`, `microsandbox`, `swe-rex`). Native execution uses `setpriv --clear-groups --no-new-privs` and the restricted account's primary gid (from `/etc/passwd`) unless `AGENT_RT_SANDBOX_GID` is set; it never keeps the caller's gid. Docker containers run with all capabilities dropped, `no-new-privileges`, and swap capped to the memory limit; set `AGENT_RT_SANDBOX_DOCKER_USER` to run as a non-root user and `AGENT_RT_SANDBOX_DOCKER_READ_ONLY=true` for a read-only root filesystem. Backends receive an `AbortSignal` in their execute context: on timeout the process group (native) or container (Docker) is killed, not just awaited. Output is capped while it is read (`outputBytes`, default 16 MiB), and `memoryBytes`/`processCount` of `0` are rejected. Call `await session.close()` to release E2B/microsandbox sessions. Snapshots and clones capture only the in-process workspace and session settings, not the disk inside a native, Docker, or cloud backend.

The `AgentLoop` keeps every tool call answered: if a run stops early (approval needed, timeout, cancellation, `max_tool_calls`) unexecuted calls receive a `tool_not_executed` tool message, completed results from a concurrent batch are kept, and loop-detector state is per run.

## Documentation

- [Developer guide](../docs/DEVELOPMENT.md) — setup, tests, quality checks, security scans, evaluation, CI, and packaging.
- [Introduction to Agent Harnesses](../docs/Introduction-to-Harness.md)
- [Architecture and Extensions](../docs/Architecture-and-Extensions.md)
- [Migration guide](../docs/MIGRATION.md)
- [Examples](examples/README.md)

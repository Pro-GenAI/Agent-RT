# Agent RT for Python

**A lightweight, provider-neutral runtime for building AI agents with explicit control over tools, safety, state, and orchestration.**

Agent RT keeps the core agent loop independent from any single model vendor or application framework. It is `asyncio`-native: start with an OpenAI-compatible endpoint, then add tools, approvals, memory, sandboxing, retrieval, and multi-agent orchestration without replacing your runtime. The same package also ships an optional API server and terminal CLI.

## Why Agent RT?

- **Provider-neutral by design.** Use OpenAI-compatible endpoints, OpenAI, Anthropic, or your own `ModelProvider`. Optional [Decision models](../docs/Architecture-and-Extensions.md#decision-models) handle control-plane gates such as tool pruning and memory screening.
- **Safety at the execution boundary.** Tool validation, deny-overrides permissions, approvals, capability grants, budgets, deadlines, and guardrails are runtime primitives.
- **Small core, optional integrations.** Provider SDKs, guardrails, API serving, CLI support, and sandbox backends are opt-in.
- **Built for production paths.** Checkpoints, event streams, task queues, tracing, cost/token ledgers, workspaces, artifacts, retrieval, and orchestration are first-class contracts.
- **Migration-friendly.** Compatibility layers cover common OpenAI, Anthropic, LangChain, LlamaIndex, OpenAI Agents, AutoGen, and CrewAI usage.

See the reproducible comparisons in [`../comparison/`](../comparison/README.md).

## Requirements

Python 3.10 through 3.14.

## Install

Minimal runtime:

```bash
pip install agent-rt
```

Optional integrations:

```bash
pip install 'agent-rt[openai]'
pip install 'agent-rt[anthropic]'
pip install 'agent-rt[providers]'
pip install 'agent-rt[guardrails]'
pip install 'agent-rt[api]'
pip install 'agent-rt[cli]'
pip install 'agent-rt[all]'
```

The minimal package does not require the OpenAI or Anthropic SDK.

## Quick start

Configure a model:

```bash
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="..."
```

For an OpenAI-compatible endpoint, also set `OPENAI_BASE_URL`. If multiple provider configurations are present, set `MODEL_PROVIDER=openai` or `MODEL_PROVIDER=anthropic`.

```python
import asyncio
import os

from agent_rt import (
    AgentConfig,
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelSettings,
    load_model,
)


async def main() -> None:
    # validate=False skips the startup model-list check, so no extra network call.
    provider = load_model(validate=False)
    agent = AgentConfig(
        name="assistant",
        instructions="Answer clearly and concisely.",
        model=ModelSettings(model=os.environ["OPENAI_MODEL"]),
    )
    message = ModelMessage(
        role="user",
        content=(ContentPart(type="text", text="What can Agent RT do?"),),
    )

    result = await AgentLoop(provider).run(agent, [message])
    if result.final_response:
        print(
            "".join(
                part.text or ""
                for part in result.final_response.message.content
                if part.type == "text"
            )
        )


asyncio.run(main())
```

More runnable examples are in [`examples/`](examples/README.md).

## What you can build

Agent RT provides a provider-neutral `AgentLoop` and typed contracts for messages, structured output, streaming, tools, routing, fallback, retries, cancellation, deadlines, rate limits, and execution budgets. Around that core:

- **Tools and safety:** registries, argument validation, approvals, permission policies, guardrails, and lifecycle hooks.
- **State:** short- and long-term memory, checkpoints, durable events, background tasks, queues, and scheduling.
- **Context and environments:** retrieval, vector databases, MCP, browser/computer sessions, workspaces, artifacts, and sandboxed code execution.
- **Multi-agent patterns:** planning, critics, supervisor/worker, swarms, hierarchical teams, and map/reduce.
- **Operations:** tracing, metrics, replay, token/cost ledgers, and deployment primitives.

Built-in HTTP vector adapters cover Chroma, Milvus, Pinecone, Qdrant, and Weaviate; FAISS and pgvector plug in through custom registry factories. See [Features and defaults](../docs/website/docs/reference/features-defaults.html) for the full list.

## Migration compatibility

Existing code can migrate incrementally by prepending `agent_rt.` to supported upstream imports. Examples include:

```text
langchain_openai              -> agent_rt.langchain_openai
llama_index.llms.openai       -> agent_rt.llama_index.llms.openai
agents                        -> agent_rt.agents
autogen_agentchat.agents      -> agent_rt.autogen_agentchat.agents
crewai                        -> agent_rt.crewai
openai                        -> agent_rt.openai
anthropic                     -> agent_rt.anthropic
```

The compatibility layer preserves familiar request/response shapes for the documented subset while routing execution through Agent RT contracts. See [Migration](../docs/MIGRATION.md) for supported APIs and semantic gaps.

## API server

Install `agent-rt[api]` to expose configured Agent RT agents through OpenAI-compatible Chat Completions/Responses APIs and an Anthropic-compatible Messages API.

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

## Terminal CLI

Install `agent-rt[cli]` to add the `agent-rt` command:

```bash
agent-rt
agent-rt -p "explain this stack trace"
agent-rt --session refactor --prompt "plan the migration"
agent-rt --session refactor --resume
```

The CLI uses the same runtime, provider routing, guardrails, tool execution, and run limits as programmatic Agent RT. Workspace writes require approval; Git integration is inspection-only. Writes to `.git/` and to `.gitattributes`/`.gitmodules` are refused outright, and the Git tools fail closed when the repository's own config defines `filter.*` drivers (Git runs those commands during `status`/`diff`). Workspace tool errors (for example a non-matching `replace_in_file`) are returned to the model as tool results so the turn and its history survive; approval previews state how many characters are not shown.

## Content and provider behaviour

- **Media:** image, PDF/file, and (OpenAI) audio `ContentPart`s are sent as the provider's native blocks; `data` may be an http(s) URL, a `data:` URL, base64, or bytes. Parts a provider cannot carry (for example video) raise instead of being dropped.
- **Anthropic thinking:** thinking blocks are kept on the assistant message (a JSON part with a vendor MIME type) and replayed before `tool_use`, so extended thinking works across tool turns. Anthropic requests default to `max_tokens=4096`; set `max_output_tokens` for longer outputs.
- **Context trust:** untrusted retrieved/file/observation items reach the model as user-role data; only workflow state and `trusted=` items are system messages.
- **Validation:** tool arguments and structured output are checked against a JSON Schema subset (types, bounds, `pattern`, `const`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref`, `additionalProperties`).
- **Truncation:** a Responses WebSocket `response.incomplete` returns `finish_reason="length"` with usage rather than raising.

## Sandboxes

`SandboxSession` runs commands through a fail-closed backend selected with `AGENT_RT_SANDBOX_BACKEND` (`native`, `docker`, `e2b`, `microsandbox`, `swe-rex`). Native execution drops supplementary groups and runs as the restricted account's primary gid unless `AGENT_RT_SANDBOX_GID` is set (it never keeps root's gid). Docker containers run with all capabilities dropped, `no-new-privileges`, and swap capped to the memory limit; set `AGENT_RT_SANDBOX_DOCKER_USER` to run as a non-root user and `AGENT_RT_SANDBOX_DOCKER_READ_ONLY=true` for a read-only root filesystem. Timeouts and cancellation kill the command (the process group for native, the container for Docker), output is capped while it is read (`SandboxResourceLimits.output_bytes`, default 16 MiB), and `memory_bytes`/`process_count` of `0` are rejected. Call `await session.close()` to release cloud sandboxes and microVMs. Snapshots and clones capture only the in-process workspace and session settings, not the disk inside a native, Docker, or cloud backend.

## Documentation

- [Developer guide](../docs/DEVELOPMENT.md) — setup, tests, quality checks, security scans, evaluation, CI, and packaging.
- [Introduction to Agent Harnesses](../docs/Introduction-to-Harness.md)
- [Architecture and Extensions](../docs/Architecture-and-Extensions.md)
- [Migration guide](../docs/MIGRATION.md)
- [Examples](examples/README.md)

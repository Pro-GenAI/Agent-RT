# Agent RT for Python

**A lightweight, provider-neutral runtime for building AI agents with explicit control over tools, safety, state, and orchestration.**

🌐 **[Website](https://agent-rt-pro.github.io/)** · [Documentation](https://agent-rt-pro.github.io/docs/) · [GitHub](https://github.com/Pro-GenAI/Agent-RT)

Start with an OpenAI-compatible endpoint, then add tools, approvals, memory, sandboxing, retrieval, and multi-agent orchestration without replacing your runtime. `asyncio`-native, with an optional API server and terminal CLI.

## Why Agent RT?

- **Provider-neutral.** OpenAI-compatible endpoints, OpenAI, Anthropic, or your own `ModelProvider`.
- **Safe by construction.** Tool validation, permissions, approvals, guardrails, budgets, and deadlines are runtime primitives.
- **Small core.** Provider SDKs, guardrails, API serving, CLI, and sandboxes are opt-in extras.
- **Production-ready.** Checkpoints, event streams, queues, tracing, cost ledgers, retrieval, and multi-agent patterns.
- **Easy to adopt.** Drop-in compatibility for OpenAI, Anthropic, LangChain, LlamaIndex, OpenAI Agents, AutoGen, and CrewAI code.

## Install

Python 3.10–3.14.

```bash
pip install agent-rt
pip install 'agent-rt[all]'   # or pick extras: openai, anthropic, api, cli, guardrails
```

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

More examples: [`examples/`](https://github.com/Pro-GenAI/Agent-RT/tree/main/python/examples).

## Migrate in one line

Prepend `agent_rt.` to a supported import, for example `from agent_rt.langchain_openai import ChatOpenAI` or `from agent_rt.openai import OpenAI`. See [Migration](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/MIGRATION.md).

## Also included

- **API server** (`agent-rt[api]`): OpenAI-compatible Chat Completions/Responses and Anthropic-compatible Messages endpoints.
- **Terminal CLI** (`agent-rt[cli]`): `agent-rt -p "explain this stack trace"`.

## Learn more

- [Website](https://agent-rt-pro.github.io/) and [documentation site](https://agent-rt-pro.github.io/docs/)
- [Runtime guide](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Runtime-Guide.md) — API server, CLI, sandboxes, skills, providers
- [Architecture and Extensions](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Architecture-and-Extensions.md) · [Introduction to Agent Harnesses](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Introduction-to-Harness.md)
- [Development](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/DEVELOPMENT.md) · [Comparisons](https://github.com/Pro-GenAI/Agent-RT/tree/main/comparison)

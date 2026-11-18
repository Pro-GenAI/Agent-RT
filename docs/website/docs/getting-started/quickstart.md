> **Canonical HTML page:** [Agent RT Quickstart — Build Your First AI Agent](https://agent-rt.pro-genai.com/docs/getting-started/quickstart)

> Build your first Agent RT agent in Python or TypeScript with runnable examples for tool calling, streaming, structured output, runtime limits, and registries.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Getting started/Quickstart
Getting started

# Quickstart

The repository examples are deliberately incremental. Start with one agent call, then add the capabilities that matter to production behavior without introducing another framework layer.

## Minimal Python agent

This is the smallest self-contained Python example: configure a model name, create an agent, create a user message, and run the loop.

```python
import asyncio

from agent_rt import (
    AgentConfig,
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelSettings,
    load_config,
    load_model,
)

async def main():
    config = load_config()  # agent-rt.yaml, ~/.agent-rt/config.yaml, or an explicit path
    model = config.default_model
    provider = load_model(config)

    agent = AgentConfig(
        name="assistant",
        instructions="Answer clearly and concisely.",
        model=ModelSettings(model=model),
    )
    message = ModelMessage(
        role="user",
        content=(ContentPart(type="text", text="What does Agent RT do?"),),
    )

    result = await AgentLoop(provider).run(agent, [message])
    print(result.termination_reason)

asyncio.run(main())
```

## Minimal TypeScript agent

```javascript
const { AgentLoop, loadConfig, loadModel } = require("agent-rt");

async function main() {
  const config = await loadConfig(); // agent-rt.yaml, ~/.agent-rt/config.yaml, or an explicit path
  const model = config.settings.default_model ?? config.models[0].modelName;

  const provider = await loadModel(config);
  const agent = {
    name: "assistant",
    instructions: "Answer clearly and concisely.",
    model: { model },
  };
  const messages = [{
    role: "user",
    content: [{ type: "text", text: "What does Agent RT do?" }],
  }];

  const result = await new AgentLoop(provider).run(agent, messages);
  console.log(result.terminationReason);
}

main();
```

**Continue as a tutorial**The

[Features & defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)

reference now includes copyable Python and TypeScript examples for tools, streaming, structured output, limits, guardrails, permissions, approvals, state/memory, sandboxing, retrieval, budgets, and observability.

## Recommended path

| Example | What it adds |
| --- | --- |
| `01_basic_agent` | A minimal run against a configured provider. |
| `02_tool_calling` | A read-only tool and the model → tool → model continuation loop. |
| `03_streaming` | Provider-neutral text delta consumption. |
| `04_structured_output` | Schema-validated JSON with a bounded repair attempt. |
| `05_run_limits` | Turn, tool-call, elapsed-time, and token limits. |
| `06_tool_registry` | Namespaces, direct execution, and enable/disable controls without a model provider. |
| `07_vector_db` | Environment-driven Chroma, Milvus, Pinecone, Qdrant, or Weaviate retrieval without changing application code. |

## Run the Python examples

```bash
cd src/python
uv sync
uv run python examples/01_basic_agent.py
uv run python examples/02_tool_calling.py
uv run python examples/07_vector_db.py
```

## Run the TypeScript examples

```bash
cd src/typescript
npm ci
npm run build
node examples/01_basic_agent.js
node examples/02_tool_calling.js
node examples/07_vector_db.js
```

**Offline check**The tool-registry example is fully offline in both implementations, so it is useful for validating local setup without model credentials.

## What to learn next

Once the basic loop is clear, read the runtime model before adding more capabilities. It explains where Agent RT draws boundaries between model behavior, tool execution, workflow state, policy, and termination.

[Previous← Installation](https://agent-rt.pro-genai.com/docs/getting-started/installation.html)



[NextRuntime model →](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)

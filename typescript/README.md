# Agent RT for TypeScript

**A lightweight, provider-neutral runtime for building AI agents with explicit control over tools, safety, state, and orchestration.**

🌐 **[Website](https://agent-rt-pro.github.io/)** · [Documentation](https://agent-rt-pro.github.io/docs/) · [GitHub](https://github.com/Pro-GenAI/Agent-RT)

Fully typed, with a native `fetch` OpenAI-compatible transport. Start with zero provider SDKs, then add tools, approvals, memory, sandboxing, retrieval, and multi-agent orchestration as needed.

## Why Agent RT?

- **Provider-neutral.** OpenAI-compatible endpoints, OpenAI, Anthropic, or your own `ModelProvider`.
- **Safe by construction.** Tool validation, permissions, composable action blockers, approvals, deterministic PII morphing, guardrails, budgets, and deadlines are runtime primitives.
- **Scales large tool catalogs.** `Toolbase` combines local/deferred and MCP tools, safety-screens them, sends only the relevant subset per turn, and keeps `tool_search` available for fallback discovery.
- **Small core.** OpenAI, Anthropic, Agent Action Guard, and sandbox integrations are optional peers.
- **Production-ready.** Checkpoints, events, queues, tracing, cost ledgers, retrieval with post-retrieval reranking, and multi-agent patterns.
- **Easy to adopt.** Drop-in compatibility entry points for OpenAI, Anthropic, LangChain, LlamaIndex, and OpenAI Agents code; across Agent RT's ongoing migration field testing, 69 third-party repositories pass before and after migration.

## Install

Node 20, 22, 24, or 26.

```bash
npm install agent-rt
```

Provider SDKs are optional: add `openai`, `@anthropic-ai/sdk`, or `agent-action-guard` only when you need them.

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

More examples: [`examples/`](https://github.com/Pro-GenAI/Agent-RT/tree/main/typescript/examples).

## Migrate in one line

Prefix a supported import with `agent-rt/`, for example `import OpenAI from "agent-rt/openai"` or `import { ChatOpenAI } from "agent-rt/@langchain/openai"`. See [Migration](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/MIGRATION.md).

## Learn more

- [Website](https://agent-rt-pro.github.io/) and [documentation site](https://agent-rt-pro.github.io/docs/)
- [Runtime guide](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Runtime-Guide.md) — sandboxes, skills, providers, migration entry points
- [Architecture and Extensions](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Architecture-and-Extensions.md) · [Introduction to Agent Harnesses](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/Introduction-to-Harness.md)
- [Development](https://github.com/Pro-GenAI/Agent-RT/blob/main/docs/DEVELOPMENT.md) · [Comparisons](https://github.com/Pro-GenAI/Agent-RT/tree/main/comparison)

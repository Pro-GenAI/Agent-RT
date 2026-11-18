> **Canonical HTML page:** [Agent RT Architecture & Extension Contracts — Reference](https://agent-rt.pro-genai.com/docs/reference/architecture)

> Explore Agent RT architecture and extension contracts for model providers, tools, state, policy, execution, deployment, evaluation, and operations.

[Docs](https://agent-rt.pro-genai.com/docs/)

**Section:** Reference / Architecture

# Architecture

Agent RT is organized around narrow, provider-neutral contracts. The goal is to let model providers, tools, memory, sandboxes, evaluators, deployment targets, and optimization layers change without collapsing into one opaque framework surface.

## Conceptual planes

| Plane | Responsibilities |
| --- | --- |
| Control | Agent definition, routing, policy, budgets, approvals. |
| Model | Providers, context assembly, structured I/O, streaming. |
| Execution | Tools, files, shell/code, sandboxes, remote capabilities. |
| State | Messages, sessions, memory, checkpoints, tasks, event history. |
| Reliability / governance | Permissions, guardrails, retry, idempotency, secrets, isolation. |
| Operations / quality | Tracing, metrics, evaluation, cost, deployment, optimization. |

## Extension principle

Each replaceable subsystem should have a small contract that expresses runtime semantics rather than provider-specific behavior. That applies to model providers, routing policies, tool registries, memory stores, retrieval, sandboxes, artifacts, evaluators, deployment targets, and operational middleware.

## Vector retrieval boundary

External vector databases implement the same Agent RT retrieval contract as other knowledge providers. `VectorDBProviderRegistry` and the environment-backed provider resolve Chroma, Milvus, Pinecone, Qdrant, or Weaviate without moving vendor SDKs into core. LangChain and LlamaIndex compatibility classes are thin read/query adapters over that boundary, so framework-shaped imports do not create a second retrieval architecture.

## Cross-language alignment

The Python and TypeScript implementations intentionally track the same core concepts and behavior. Exact APIs differ where language conventions demand it, but the runtime model should remain recognizable across both packages.

> **Complete contract reference:** See [Architecture and Extension Contracts ↗](https://agent-rt.pro-genai.com/docs/reference/extension-contracts.html) for module-level interfaces and implementation detail.

**Previous:** [Production patterns](https://agent-rt.pro-genai.com/docs/guides/production.html)

**Next:** [Development](https://agent-rt.pro-genai.com/docs/reference/development.html)

> **Canonical HTML page:** [AI Agent State, Memory & Checkpoints — Agent RT Docs](https://agent-rt.pro-genai.com/docs/concepts/state-memory)

> Learn Agent RT workflow state, model context, scoped memory, checkpoints, files, artifacts, persistence, and resumable durable agent execution.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Core concepts/State & memory
Core concepts

# State & memory

Reliable agents need more than chat history. Agent RT separates model context, typed workflow state, persistent memory, checkpoints, files, artifacts, and execution history so each can have its own lifecycle and policy.

## Model context

Each model request can combine instructions, selected history, workflow state, retrieved knowledge, files, observations, tools, and runtime metadata. Context selection is explicit; durable state remains separate from what the model sees on a particular turn.

## Workflow state

Typed state can hold task variables, plan state, counters, approvals, tool results, artifacts, and execution metadata. This avoids reconstructing operational state from natural-language messages.

## Memory scopes

Short-term memory is scoped by stable session and thread identities. Persistent memory adds explicit CRUD/search semantics, retention and expiration, and policies for what should be written or retrieved. Semantic, episodic, and procedural memory can share the same scoped boundary while using different ranking or storage implementations.

Attach memory to an agent with `AgentConfig.memory`: a list that can mix plain strings, custom objects (sent as JSON), memory stores and retrievers searched with the latest user prompt (Agent RT, LangChain, LangGraph, LlamaIndex, AutoGen, OpenAI Agents sessions, and mem0), and functions that return memories. Wrap a source in `MemorySource` to name it, cap its results, scope its search, or set its trust. Store and function results are untrusted by default. Install a memory package with `agent-rt[memory]` (mem0 by default; memU, Zep, Letta, and Supermemory are also supported) and connect it with `load_memory()` / `loadMemory()`, which also stores finished conversations with `remember()`. See the

[runtime guide](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html)



[Runtime handbook](https://agent-rt.pro-genai.com/docs/guides/runtime-reference.html)

for examples.

## Checkpoints and resumption

Checkpoints preserve enough execution state to continue after interruption, including messages, durable workflow state, and already-consumed budgets. Resumed work therefore does not silently reset turn, tool, or token limits.

## Files and artifacts

The filesystem abstraction provides controlled workspace operations. Artifacts treat reports, code, datasets, images, and other outputs as first-class objects with revisions, lineage, retention, and provenance.

[Previous← Tools & policy](https://agent-rt.pro-genai.com/docs/concepts/tools-policy.html)



[NextMigration →](https://agent-rt.pro-genai.com/docs/guides/migration.html)

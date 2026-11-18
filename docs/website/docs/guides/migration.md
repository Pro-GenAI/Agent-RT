> **Canonical HTML page:** [Migrate LangChain, LlamaIndex, OpenAI & Anthropic to Agent RT](https://agent-rt.pro-genai.com/docs/guides/migration)

> Migrate LangChain, LlamaIndex, OpenAI, or Anthropic applications to Agent RT, including Chroma, Pinecone, Qdrant, Milvus, and Weaviate vector-store compatibility.

[Docs](https://agent-rt.pro-genai.com/docs/)

**Section:** Guides / Migration

# Migration

Python and TypeScript applications can adopt Agent RT incrementally through compatibility entry points, then move toward native runtime contracts when they need deeper control over tools, state, policy, or execution.

## Compatibility surfaces

For common imports, migration can be as small as changing the module prefix. Keep the class or symbol you already use and point the import at Agent RT's compatibility layer. Migration field testing is ongoing; 65+ third-party repositories currently pass their test suites both before and after migration to Agent RT.

### Python

#### LangChain

```diff
- from langchain_openai import ChatOpenAI
+ from agent_rt.langchain_openai import ChatOpenAI
```

#### LlamaIndex

```diff
- from llama_index.llms.openai import OpenAI
+ from agent_rt.llama_index.llms.openai import OpenAI
```

#### LangChain · Qdrant

```diff
- from langchain_qdrant import QdrantVectorStore
+ from agent_rt.langchain_qdrant import QdrantVectorStore
```

#### LlamaIndex · Qdrant

```diff
- from llama_index.vector_stores.qdrant import QdrantVectorStore
+ from agent_rt.llama_index.vector_stores.qdrant import QdrantVectorStore
```

#### OpenAI

```diff
- from openai import AsyncOpenAI
+ from agent_rt.openai import AsyncOpenAI
```

#### Anthropic

```diff
- from anthropic import AsyncAnthropic
+ from agent_rt.anthropic import AsyncAnthropic
```

#### Deep Agents

```diff
- from deepagents import create_deep_agent
+ from agent_rt.deepagents import create_deep_agent
```

#### LiteLLM

```diff
- from litellm import completion
+ from agent_rt.litellm import completion
```

#### OpenRouter

```diff
- from openrouter import OpenRouter
+ from agent_rt.openrouter import OpenRouter
```

### TypeScript / JavaScript

#### LangChain

```diff
- import { ChatOpenAI } from "@langchain/openai";
+ import { ChatOpenAI } from "agent-rt/@langchain/openai";
```

#### LlamaIndex

```diff
- import { OpenAI } from "@llamaindex/openai";
+ import { OpenAI } from "agent-rt/@llamaindex/openai";
```

#### LangChain · Qdrant

```diff
- import { QdrantVectorStore } from "@langchain/qdrant";
+ import { QdrantVectorStore } from "agent-rt/@langchain/qdrant";
```

#### LlamaIndex · Qdrant

```diff
- import { QdrantVectorStore } from "@llamaindex/qdrant";
+ import { QdrantVectorStore } from "agent-rt/@llamaindex/qdrant";
```

#### OpenAI

```diff
- import OpenAI from "openai";
+ import OpenAI from "agent-rt/openai";
```

#### Anthropic

```diff
- import Anthropic from "@anthropic-ai/sdk";
+ import Anthropic from "agent-rt/@anthropic-ai/sdk";
```

#### Deep Agents

```diff
- import { createDeepAgent } from "deepagents";
+ import { createDeepAgent } from "agent-rt/deepagents";
```

#### OpenRouter

```diff
- import { OpenRouter } from "@openrouter/sdk";
+ import { OpenRouter } from "agent-rt/@openrouter/sdk";
```

## Migrate with one command

Run one command in your project to rewrite every supported import:

```bash
uvx agent-rt migrate   # Python
npx agent-rt migrate   # TypeScript / JavaScript
```

It runs your tests before and after the rewrite and keeps it only if nothing new fails: every test that fails afterwards must have failed before, in the same test at the same line. Otherwise every file is restored. It then prints a before-and-after card (tests, test time, peak memory, package size) and, for a GitHub repository, a pre-filled issue link to add it to the migration wall. Nothing is sent automatically. Install `agent-rt` (and `pytest` for Python) in the project first. Per-test comparison in TypeScript needs Vitest or Jest.

## “Runs on Agent RT” badge

Maintainers whose repository runs on Agent RT can add this badge to their README:

![Runs on Agent RT](https://agent-rt.pro-genai.com/assets/badge/runs-on-agent-rt.svg)

```markdown
[![Runs on Agent RT](https://agent-rt.pro-genai.com/assets/badge/runs-on-agent-rt.svg)](https://agent-rt.pro-genai.com/#migration)
```

The badge is opt-in: we never open pull requests or issues asking repositories, including those on the migration wall, to add it.

## Adopt in layers

Start at the compatibility entry point closest to the code you already have. Once behavior is stable, move runtime-sensitive concerns—tools, run limits, sandboxing, retrieval, memory, policy, and checkpoints—onto Agent RT-native contracts where the control boundary is explicit.

## Supported compatibility themes

The migration layer covers common chat/completion entry points, streaming, tool history, OpenAI Responses continuation, file-search integration through Agent RT retrieval providers, web retrieval integration, reasoning/thinking controls, batch lifecycle compatibility, and read/query vector-store adapters for Chroma, Pinecone, Qdrant, Milvus, and Weaviate.

LangChain compatibility exposes vendor-shaped imports such as `agent_rt.langchain_qdrant` and `agent-rt/@langchain/qdrant`. LlamaIndex compatibility exposes `agent_rt.llama_index.vector_stores.qdrant` and `agent-rt/@llamaindex/qdrant`, including integration with `VectorStoreIndex.from_vector_store()` / `fromVectorStore()`. These adapters query existing collections; vendor write/admin lifecycle APIs remain outside the compatibility layer. See the [vector database guide](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html) for the complete mapping.

> **Complete migration reference:** This page is the navigation-oriented overview. Read the [full migration reference](https://agent-rt.pro-genai.com/docs/guides/migration-reference.html) for API coverage, framework-specific compatibility notes, and code examples.

**Previous:** [State & memory](https://agent-rt.pro-genai.com/docs/concepts/state-memory.html)

**Next:** [Production patterns](https://agent-rt.pro-genai.com/docs/guides/production.html)

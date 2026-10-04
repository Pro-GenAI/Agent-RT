# TypeScript package examples

**The fastest way to understand Agent RT is to run it.** These small Node examples are deliberately incremental: start with one agent call, then add tools, streaming, structured output, runtime limits, and direct registry control without introducing a second framework or hidden abstraction.

If you are evaluating Agent RT, run `01_basic_agent.js` first, then `02_tool_calling.js`. Together they show the core model → tool → model loop with the same contracts used by the larger runtime.

## Examples

- `01_basic_agent.js` — minimal agent run with a configured provider.
- `02_tool_calling.js` — register a read-only tool and let the model call it.
- `03_streaming.js` — consume text deltas with `runStreaming()`.
- `04_structured_output.js` — require schema-validated JSON output with one repair attempt.
- `05_run_limits.js` — bound turns, tool calls, elapsed time, and total tokens.
- `06_tool_registry.js` — use namespaces, direct execution, and enable/disable controls without a model provider.
- `07_vector_db.js` — query Chroma, Milvus, Pinecone, Qdrant, or Weaviate and switch backends with environment variables.

## Setup

From the `typescript/` directory:

```bash
npm ci
npm run build
```

Configure a provider. For OpenAI or an OpenAI-compatible endpoint:

```bash
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="..."
# Optional:
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

For Anthropic, configure `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL`.

Run an example from the `typescript/` directory:

```bash
node examples/01_basic_agent.js
node examples/02_tool_calling.js
node examples/03_streaming.js
node examples/04_structured_output.js
node examples/05_run_limits.js
node examples/06_tool_registry.js
node examples/07_vector_db.js
```

The first five examples call a configured model provider. `06_tool_registry.js` is fully offline and is useful for checking local setup without API credentials. `07_vector_db.js` performs live vector retrieval and requires a configured vector service plus an embedding-capable model provider.


## Vector DB example

The vector DB example uses the same application code for every built-in service backend. Configure an OpenAI/OpenAI-compatible embedding provider, then set:

```bash
export AGENT_RT_VECTOR_DB="qdrant" # chroma|milvus|pinecone|qdrant|weaviate
export AGENT_RT_VECTOR_DB_URL="https://your-vector-db.example"
export AGENT_RT_VECTOR_DB_COLLECTION="docs"
export AGENT_RT_VECTOR_DB_API_KEY="..." # optional; backend-native variables also work
export AGENT_RT_VECTOR_DB_QUERY="What is Agent RT?" # optional
export AGENT_RT_VECTOR_DB_LIMIT="5" # optional
```

Backend-native credentials are `CHROMA_API_KEY`, `MILVUS_TOKEN`, `PINECONE_API_KEY`, `QDRANT_API_KEY`, and `WEAVIATE_API_KEY`. Change `AGENT_RT_VECTOR_DB` plus the endpoint, collection, and credential values to move the same application code to another backend. Extra backend configuration is passed with `AGENT_RT_VECTOR_DB_OPTION_*` variables, for example `AGENT_RT_VECTOR_DB_OPTION_NAMESPACE` for Pinecone or `AGENT_RT_VECTOR_DB_OPTION_TENANT` / `AGENT_RT_VECTOR_DB_OPTION_DATABASE` for Chroma.

FAISS and pgvector remain custom `VectorDBProviderRegistry` integrations because their normal execution paths require local/native or database drivers.

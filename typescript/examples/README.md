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

Copy [`agent-rt.example.yaml`](../../agent-rt.example.yaml) to `agent-rt.yaml` (or `~/.agent-rt/config.yaml`) and edit `model_list`. Secrets are referenced as `os.environ/NAME`, so export the variables the file names (for example `OPENAI_API_KEY`).

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

The vector DB example uses the same application code for every built-in service backend. Add a `vector_store_registry` entry (backend via `custom_llm_provider`: `chroma`, `milvus`, `pinecone`, `qdrant`, or `weaviate`; plus `api_base`, `collection`, `api_key: os.environ/...`, and an `embedding_model` that is an embedding-capable `model_list` entry), then pass the store name and an optional query:

```bash
node examples/07_vector_db.js docs-kb "What is Agent RT?"
```

Change `custom_llm_provider` plus the endpoint, collection, and credential to move the same application code to another backend; several stores can be declared side by side. Extra non-secret backend settings go under `options`.

FAISS and pgvector remain custom `VectorDBProviderRegistry` integrations because their normal execution paths require local/native or database drivers.

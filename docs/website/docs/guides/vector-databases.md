> **Canonical HTML page:** [Vector Databases — Agent RT Docs](https://agent-rt.pro-genai.com/docs/guides/vector-databases)

> Switch Agent RT between Chroma, Milvus, Pinecone, Qdrant, and Weaviate in one agent-rt.yaml registry, and migrate LangChain or LlamaIndex vector-store imports.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Guides/Vector databases
 Guides

# Vector databases

Register the built-in vector adapters once, declare several stores in `agent-rt.yaml`, switch a store's backend by changing one key, or keep familiar LangChain and LlamaIndex vector-store imports while routing retrieval through Agent RT. The HTTP adapters require no vendor SDKs; Bedrock Knowledge Bases uses an optional AWS SDK.

## Supported backends

Agent RT includes dependency-light HTTP adapters for Chroma, Milvus, Pinecone, Qdrant, and Weaviate. FAISS and pgvector remain registry extensions because their normal integration paths require native/local or database drivers. Amazon Bedrock Knowledge Bases is supported with `custom_llm_provider: bedrock`, using the hosted Retrieve API. Store clients are initialized lazily on first search. FAISS/pgvector and other hosted providers remain extensibility points.

| Backend | `custom_llm_provider` | Credential variable | Notes |
| --- | --- | --- | --- |
| Bedrock Knowledge Bases | `bedrock` | AWS IAM credentials | Hosted Retrieve API (optional AWS SDK). |
| Chroma | `chroma` | `CHROMA_API_KEY` | Optional tenant/database options. |
| Milvus | `milvus` | `MILVUS_TOKEN` | Uses the Milvus REST vector search endpoint. |
| Pinecone | `pinecone` | `PINECONE_API_KEY` | Optional namespace via an option variable. |
| Qdrant | `qdrant` | `QDRANT_API_KEY` | Uses collection point search. |
| Weaviate | `weaviate` | `WEAVIATE_API_KEY` | Field names can be overridden with options. |

## Configure once

Vector stores live in the `vector_store_registry` of your `agent-rt.yaml`. Declare as many as you need; each is addressable by its name and its `vector_store_id`, and secrets are referenced as `os.environ/NAME`.

```yaml
model_list:
  - model_name: embed-small
    agent_rt_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

vector_store_registry:
  - vector_store_name: docs-kb
    agent_rt_params:
      vector_store_id: T37J8R4WTM
      custom_llm_provider: qdrant        # chroma|milvus|pinecone|qdrant|weaviate
      api_base: https://your-vector-db.example
      collection: docs
      api_key: os.environ/QDRANT_API_KEY
      embedding_model: embed-small
      timeout: 20
      options: {}                        # backend-specific, non-secret
```

Backend-specific settings go under `options`. For Bedrock, `options.region` is optional; the AWS SDK can resolve its region and IAM credentials normally. Install `agent-rt[bedrock]` (Python) or `@aws-sdk/client-bedrock-agent-runtime` (TypeScript). HTTP stores require endpoint and collection, while Bedrock needs only the knowledge base ID. Both are lazily initialized.

```yaml
vector_store_registry:
  - vector_store_name: bedrock-litellm-website-knowledgebase
    litellm_params:
      vector_store_id: T37J8R4WTM
      custom_llm_provider: bedrock
      options: {region: us-west-2}
      vector_store_description: Bedrock documentation knowledge base
```

## Pass stores to a model with `file_search`

The Python API server accepts the OpenAI-style tool on Chat Completions and Responses requests. The model gets a request-scoped `file_search` tool limited to the listed stores; results are treated as untrusted tool output.

```python
client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "What is litellm?"}],
    tools=[{"type": "file_search", "vector_store_ids": ["T37J8R4WTM"]}],
)
```

Unknown stores, and stores whose `allowed_models` excludes the model, are rejected with a 400. No other request-defined tool is accepted.

## Python

```python
from agent_rt import (
    RetrievalQuery,
    load_config,
    load_model,
    vector_store_registry_from_config,
)

config = load_config()
registry = vector_store_registry_from_config(
    config,
    embedding_provider=load_model(config),
)

results = await registry.search(
    "docs-kb", RetrievalQuery("How does Agent RT handle tools?", limit=5)
)
```

## TypeScript

```javascript
const {
  loadConfig,
  loadModel,
  vectorStoreRegistryFromConfig,
} = require("agent-rt");

const config = await loadConfig();
const registry = vectorStoreRegistryFromConfig(config, {
  embeddingProvider: await loadModel(config),
});

const results = await registry.search("docs-kb", {
  text: "How does Agent RT handle tools?",
  limit: 5,
});
```

## LangChain vector-store compatibility

Existing LangChain retrieval code can keep the vendor-shaped class names while moving the import to Agent RT. The adapter fixes the backend implied by the import and sends similarity search through Agent RT's vector DB provider.

| Upstream Python | Agent RT Python |
| --- | --- |
| `langchain_chroma.Chroma` | `agent_rt.langchain_chroma.Chroma` |
| `langchain_pinecone.PineconeVectorStore` | `agent_rt.langchain_pinecone.PineconeVectorStore` |
| `langchain_qdrant.QdrantVectorStore` | `agent_rt.langchain_qdrant.QdrantVectorStore` |
| `langchain_milvus.Milvus` | `agent_rt.langchain_milvus.Milvus` |
| `langchain_weaviate.WeaviateVectorStore` | `agent_rt.langchain_weaviate.WeaviateVectorStore` |



```python
from agent_rt.langchain_qdrant import QdrantVectorStore

store = QdrantVectorStore(
    embedding=embeddings,
)

documents = await store.asimilarity_search(
    "How does Agent RT retrieve context?",
    k=4,
)
```

TypeScript/JavaScript exports the same compatibility layer through `agent-rt/@langchain/chroma`, `agent-rt/@langchain/pinecone`, `agent-rt/@langchain/qdrant`, `agent-rt/@langchain/milvus`, and `agent-rt/@langchain/weaviate`. Community-style subpaths such as `agent-rt/@langchain/community/vectorstores/qdrant` are also published.

```javascript
import { QdrantVectorStore } from "agent-rt/@langchain/qdrant";

const store = new QdrantVectorStore(embeddings);
const documents = await store.similaritySearch(
  "How does Agent RT retrieve context?",
  4,
);
```

## LlamaIndex vector-store compatibility

LlamaIndex migrations preserve the upstream `vector_stores` module hierarchy in Python and expose equivalent package aliases in TypeScript.

| Upstream Python | Agent RT Python |
| --- | --- |
| `llama_index.vector_stores.chroma.ChromaVectorStore` | `agent_rt.llama_index.vector_stores.chroma.ChromaVectorStore` |
| `llama_index.vector_stores.pinecone.PineconeVectorStore` | `agent_rt.llama_index.vector_stores.pinecone.PineconeVectorStore` |
| `llama_index.vector_stores.qdrant.QdrantVectorStore` | `agent_rt.llama_index.vector_stores.qdrant.QdrantVectorStore` |
| `llama_index.vector_stores.milvus.MilvusVectorStore` | `agent_rt.llama_index.vector_stores.milvus.MilvusVectorStore` |
| `llama_index.vector_stores.weaviate.WeaviateVectorStore` | `agent_rt.llama_index.vector_stores.weaviate.WeaviateVectorStore` |



```python
from agent_rt.llama_index.vector_stores.qdrant import QdrantVectorStore
from agent_rt.llamaindex import VectorStoreIndex

vector_store = QdrantVectorStore()
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)
nodes = await index.as_retriever(
    similarity_top_k=4,
).aretrieve("How does Agent RT migrate vector retrieval?")
```

TypeScript/JavaScript equivalents are published at `agent-rt/@llamaindex/chroma`, `agent-rt/@llamaindex/pinecone`, `agent-rt/@llamaindex/qdrant`, `agent-rt/@llamaindex/milvus`, and `agent-rt/@llamaindex/weaviate`.

```javascript
import {
  QdrantVectorStore,
  VectorStoreIndex,
} from "agent-rt/@llamaindex/qdrant";

const vectorStore = new QdrantVectorStore();
const index = VectorStoreIndex.fromVectorStore(vectorStore, {
  embedModel,
});
const nodes = await index.asRetriever({
  similarityTopK: 4,
}).retrieve("How does Agent RT migrate vector retrieval?");
```

**Compatibility scope**LangChain and LlamaIndex adapters support read/query operations against existing collections. Collection creation, ingestion/upsert, deletion, index administration, and vendor-specific sparse/hybrid lifecycle APIs remain backend-native. Use native Agent RT retrieval with `vector_store_registry` when you want backend selection to come entirely from `agent-rt.yaml`.

## Embedding queries

Vector databases search numeric embeddings while Agent RT retrieval queries are text-first. Supply an `EmbeddingModelProvider` when registering the built-in backends. OpenAI and OpenAI-compatible Agent RT providers support embeddings. If your application already creates embeddings elsewhere, pass a numeric `filters.vector` instead.

```python
await provider.search(
    RetrievalQuery(
        "precomputed query",
        filters={"vector": [0.12, -0.03, 0.44]},
    )
)
```



```javascript
await provider.search({
  text: "precomputed query",
  filters: { vector: [0.12, -0.03, 0.44] },
});
```

## Backend options

| Backend | Example option | Purpose |
| --- | --- | --- |
| Chroma | `options.tenant` `options.database` | Select Chroma tenant and database. |
| Pinecone | `options.namespace` | Select a Pinecone namespace. |
| Milvus | `options.filter` | Pass a Milvus filter expression. |
| Weaviate | `options.content_field` `options.title_field` `options.uri_field` | Map your schema fields into Agent RT retrieval results. |

## Runnable examples

See `python/examples/07_vector_db.py` and `typescript/examples/07_vector_db.js` in the repository. Both examples use the same environment-driven backend selection shown above.

[Previous← Production patterns](https://agent-rt.pro-genai.com/docs/guides/production.html)



[NextRuntime guide →](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html)

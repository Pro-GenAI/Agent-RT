# Migrating to Agent RT

Agent RT provides lightweight Python and TypeScript/JavaScript compatibility entry points for common migration paths. They preserve familiar entry points while translating requests into Agent RT's provider-neutral `ModelRequest` and `ModelResponse` contracts internally.

The compatibility layer is intentionally a focused subset, not a claim of drop-in parity with every LangChain, LlamaIndex, OpenAI Agents SDK, AutoGen, CrewAI, OpenAI SDK, or Anthropic SDK feature. Use it to reduce the initial migration diff, then move framework-specific orchestration, tools, memory, streaming, and policy code to Agent RT's native APIs where needed.

Migration compatibility field testing is ongoing. So far, 69 third-party repositories have passed their test suites both before and after migration to Agent RT.

The adapters use the same provider environment settings as Agent RT, including `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, and `ANTHROPIC_API_KEY`.

Minimum environment for the snippets below:

- OpenAI: set `OPENAI_API_KEY` to use the default `https://api.openai.com/v1` endpoint, or set `OPENAI_BASE_URL` for an OpenAI-compatible endpoint (a key is then optional if that endpoint does not require one). `OPENAI_MODEL` supplies the default model.
- Anthropic: set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL`).
- OpenAI requests use a Responses WebSocket transport by default. Custom OpenAI-compatible endpoints that do not implement it must opt out by passing an explicit provider configured with `websocket=False` (Python) or `websocket: false` (TypeScript) to the compatibility client, for example `AsyncOpenAI(provider=OpenAIModelProvider(OpenAIProviderSettings(base_url=..., websocket=False)))`. The `OPENAI_WEBSOCKET=0` variable applies to native environment startup (see [Architecture and Extensions](Architecture-and-Extensions.md#provider-selection-and-transports)); the compatibility clients construct their provider from the settings above and do not read it.

Upstream tool-type, beta-header, and API-shape identifiers shown in these examples (for example `code_execution_20250825`, `web_search_20250305`, and `mcp-client-2026-09-15`) reflect the upstream SDKs when this guide was last reviewed. Check the upstream documentation for the versions you target.

## LangChain

For exact migrations, preserve the upstream module path and only prepend `agent_rt.`. For example, `langchain_openai` becomes `agent_rt.langchain_openai`. The combined flattened alias `agent_rt.langchain` (used by the agent examples below) remains supported:

```python
import os

from agent_rt.langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
response = await llm.ainvoke("Hello")
print(response.content)
```

The Python compatibility surface covers the common LangChain v1 model-and-agent migration path without requiring LangChain itself. It includes richer messages and metadata, sync/async invocation, batching, streaming, callable tools, structured output, and a bounded `create_agent()` tool loop.

Supported:

- Messages: `BaseMessage`, `HumanMessage`, `AIMessage`, `AIMessageChunk`, `SystemMessage`, and `ToolMessage`, including `text`, `content_blocks`, additional/response metadata, usage metadata, IDs/names, tool calls, and tool-call IDs.
- Models: `ChatOpenAI`, `ChatAnthropic`, and `init_chat_model()`, with string/message inputs plus `invoke()`, `ainvoke()`, `batch()`, `abatch()`, `stream()`, `astream()`, `bind()`, and `bind_tools()`. The LangChain v1 submodule paths `langchain.chat_models` (including `chat_models.base` with `_ConfigurableModel` / `BaseChatModel`), `langchain.agents` (`.middleware`, `.structured_output`), `langchain.embeddings`, `langchain.messages`, and `langchain.tools` resolve under the `agent_rt.` prefix. `langchain_core` is not part of the migration surface and stays upstream.
- Embeddings: `langchain_openai.OpenAIEmbeddings` and `init_embeddings()` embed through an Agent RT OpenAI provider (sync and async `embed_documents()` / `embed_query()`).
- Tools: OpenAI-style function schemas, ordinary Python callables, LangChain-like tool objects, and the compatibility `@tool` decorator. Python annotations are converted to a basic JSON Schema.
- Structured output: `with_structured_output()` accepts raw JSON Schema mappings and Pydantic-style classes exposing `model_json_schema()` or `schema()`; `include_raw=True` returns `raw`, `parsed`, and `parsing_error`. Agent `response_format` additionally accepts `ProviderStrategy`, `ToolStrategy`, and `AutoStrategy`; tool strategy uses a synthetic structured-response tool and bounded validation-error feedback/retries.
- Agents: `create_agent()` runs a bounded model -> tool -> model loop with `system_prompt`, callable tool execution, optional final structured response parsing, sync/async invocation, and `updates` / `messages` streaming. `MemorySaver` and native Agent RT checkpoint stores support `thread_id` history/state resume; `InMemoryStore`, runtime context/state schema validation, and tool runtime/store/context/state injection cover the common LangGraph stateful-agent migration path. Middleware hooks cover before/after model, wrapped tool calls, post-agent processing, model/tool call limits, approval callbacks, PII redaction, summarization, and dynamic tool selection.
- Runnable/config: chat and structured runnables support `with_retry()`, fallbacks, `pipe()`, configurable fields, lightweight function/object callbacks, per-input batch config, and bounded `max_concurrency`.
- Retrieval/documents: `Document`, `TextLoader`, `JSONLoader`, `CSVLoader`, `DirectoryLoader`, `RecursiveCharacterTextSplitter`, `AgentRTEmbeddings`, `MemoryVectorStore`, `AgentRTRetriever`, `ExternalVectorStoreRetriever`, `EnsembleRetriever`, and `ContextualCompressionRetriever` cover provider-neutral retrieval, injected external vector stores, hybrid fusion, and reranking. Vendor-shaped vector-store adapters are also available for Chroma, Pinecone, Qdrant, Milvus, and Weaviate; they route read/query operations through Agent RT's environment-configured vector DB providers without importing the vendor SDKs.
- Prompts/LCEL: `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `RunnableLambda`, `RunnablePassthrough`, retry/fallback/configurable runnables, and `pipe()` cover the common composition path.
- Document chains: stuff, map-reduce, refine, and retrieval chains are available through `create_stuff_documents_chain()`, `create_map_reduce_documents_chain()`, `create_refine_documents_chain()`, and `create_retrieval_chain()`.
- Output parsing: common string and JSON output parsers are available in both runtimes; Python additionally provides `PydanticOutputParser` for Pydantic-style model validation.

### LangChain tool and agent migration

```python
from agent_rt.langchain import ChatOpenAI, create_agent, tool

@tool
def weather(city: str) -> str:
    """Return weather for a city."""
    return f"Sunny in {city}."  # replace with a real lookup

model = ChatOpenAI(model="your-model")
agent = create_agent(model, tools=[weather], system_prompt="Be concise.")

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Weather in Boston?"}]}
)
print(result["messages"][-1].content)
```

### LangChain structured output migration

```python
schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

structured = ChatOpenAI(model="your-model").with_structured_output(schema)
result = await structured.ainvoke("Return an answer object.")
print(result["answer"])
```

### LangChain streaming and batching migration

```python
model = ChatOpenAI(model="your-model")

batch = await model.abatch(["first prompt", "second prompt"])

async for chunk in model.astream("Stream a response."):
    print(chunk.content, end="")
```

### LangChain vector database migration

The common vendor vector-store packages/modules can migrate to Agent RT without installing their vendor SDK clients. The compatibility classes preserve the familiar search/retriever shape, but execute retrieval through Agent RT's built-in HTTP vector DB providers.

| Upstream Python import | Agent RT Python import |
| --- | --- |
| `langchain_chroma.Chroma` | `agent_rt.langchain_chroma.Chroma` |
| `langchain_pinecone.PineconeVectorStore` | `agent_rt.langchain_pinecone.PineconeVectorStore` |
| `langchain_qdrant.QdrantVectorStore` | `agent_rt.langchain_qdrant.QdrantVectorStore` |
| `langchain_milvus.Milvus` | `agent_rt.langchain_milvus.Milvus` |
| `langchain_weaviate.WeaviateVectorStore` | `agent_rt.langchain_weaviate.WeaviateVectorStore` |

For TypeScript/JavaScript, Agent RT publishes the corresponding package-export aliases `agent-rt/@langchain/chroma`, `agent-rt/@langchain/pinecone`, `agent-rt/@langchain/qdrant`, `agent-rt/@langchain/milvus`, and `agent-rt/@langchain/weaviate`. Compatibility aliases also cover the older/community-shaped `agent-rt/@langchain/community/vectorstores/{chroma,pinecone,qdrant,milvus,weaviate}` subpaths.

```python
import os

from agent_rt.langchain_qdrant import QdrantVectorStore

store = QdrantVectorStore(
    embedding=embeddings,  # an embeddings object, e.g. AgentRTEmbeddings
    environment=os.environ,
)
documents = await store.asimilarity_search("How does Agent RT retrieve context?", k=4)
```

The adapter reads `AGENT_RT_VECTOR_DB_URL`, `AGENT_RT_VECTOR_DB_COLLECTION`, the generic/backend-native credential variables, and `AGENT_RT_VECTOR_DB_OPTION_*` settings documented for native vector DB usage. The imported compatibility class fixes the vendor backend implied by its package name; applications that need to switch the backend solely with `AGENT_RT_VECTOR_DB` should use the native `VectorDBProviderRegistry` plus `AgentRTRetriever`.

This compatibility surface is intentionally retrieval-oriented: `similarity_search` / `similaritySearch`, scored search, and retriever conversion are supported against existing collections. Vendor-specific collection creation, ingestion/upsert, deletion, index administration, sparse/hybrid vendor features, and lifecycle APIs remain backend-native.

### LangChain compatibility scope

The common LangChain migration path is implemented without taking a runtime dependency on LangChain. The compatibility layer includes chat and message APIs, tools and agents, structured output, common Runnable/LCEL composition, prompt templates and message placeholders, checkpoint/store/context/state helpers, middleware, loaders, retrieval and vector-store bridges, hybrid retrieval/reranking composition, common output parsers, and stuff/map-reduce/refine/retrieval document chains.

Explicit non-goals are provider-specific multimodal block aliases outside Agent RT's normalized content-part contract, bundled vendor SDK clients for external vector databases, and exhaustive emulation of every specialized LangChain integration package. Those integrations can be injected through Agent RT provider/retriever/vector-store contracts instead of being reimplemented in the compatibility layer.

### LangChain compatibility coverage

This matrix compares the compatibility surface implemented by Agent RT with the corresponding upstream LangChain v1 / LangGraph-backed agent capabilities. `✅` means the common migration path is implemented, `⚠️` means Agent RT implements a deliberately narrower subset, and `❌` means the upstream capability is not implemented by `agent_rt.langchain` / `agent-rt/langchain`. Upstream LangChain has a much larger provider and integration ecosystem than this table can enumerate.

| Area | Agent RT Python | Agent RT TypeScript / JavaScript | Upstream LangChain |
| --- | --- | --- | --- |
| Core messages | ✅ `BaseMessage`, `HumanMessage`, `AIMessage`, `AIMessageChunk`, `SystemMessage`, `ToolMessage` | ✅ Same common message set | Full message hierarchy and utilities |
| Message metadata / tool-call IDs | ✅ | ✅ | ✅ |
| OpenAI chat model | ✅ `ChatOpenAI` | ✅ `ChatOpenAI` | ✅ via `langchain-openai` |
| Anthropic chat model | ✅ `ChatAnthropic` | ✅ `ChatAnthropic` | ✅ via `langchain-anthropic` |
| Provider-based model factory | ✅ `init_chat_model()` | ✅ `initChatModel()` | ✅ |
| Sync model invocation | ✅ `invoke()` | — JavaScript API is async | ✅ Python; JavaScript is async |
| Async model invocation | ✅ `ainvoke()` | ✅ `invoke()` returns a Promise | ✅ |
| Batch invocation | ✅ `batch()` / `abatch()` | ✅ `batch()` | ✅ |
| Streaming | ✅ `stream()` / `astream()` | ✅ async `stream()` | ✅ |
| Split streaming tool-call JSON deltas | ✅ accumulated by call ID | ✅ accumulated by call ID | ✅ provider/integration dependent |
| Model binding | ✅ `bind()` | ✅ `bind()` | ✅ |
| Tool binding | ✅ `bind_tools()` | ✅ `bindTools()` | ✅ |
| Callable/tool helper | ✅ callables + `@tool` + LangChain-like tool objects | ✅ `tool()` + LangChain-like tool objects | ✅ |
| Python annotation schema inference | ✅ basic JSON Schema | — | ✅ richer type/schema ecosystem |
| Structured model output | ✅ `with_structured_output()` | ✅ `withStructuredOutput()` | ✅ |
| Raw + parsed structured result | ✅ `include_raw=True` | ✅ `includeRaw: true` | ✅ |
| Provider/tool/auto structured-output strategy | ✅ `ProviderStrategy`, `ToolStrategy`, `AutoStrategy` | ✅ same strategy concepts | ✅ |
| `create_agent` / `createAgent` tool loop | ✅ bounded model → tool → model loop | ✅ bounded model → tool → model loop | ✅ LangGraph-backed agent runtime |
| Agent streaming modes | ✅ `updates` / `messages` | ✅ `updates` / `messages` / `values` | ✅ broader LangGraph streaming modes |
| Runtime context/state validation and tool injection | ✅ common context/state schema path | ✅ common context/state schema path | ✅ |
| Built-in model/tool call limits | ✅ | ✅ | ✅ |
| PII middleware | ✅ redaction-oriented compatibility middleware | ✅ | ✅ richer configurable middleware |
| Summarization middleware | ✅ | ✅ | ✅ |
| Dynamic tool selection middleware | ✅ | ✅ | ✅ |
| Runnable retry | ✅ `with_retry()` | ✅ `withRetry()` | ✅ |
| Runnable fallbacks | ✅ | ✅ | ✅ |
| Configurable fields | ✅ common configurable-field path | ✅ common configurable-field path | ✅ |
| Per-input batch config / concurrency bounds | ✅ | ✅ | ✅ |
| `Document` | ✅ | ✅ | ✅ |
| Text / JSON / CSV loaders | ✅ `TextLoader`, `JSONLoader`, `CSVLoader` | ✅ same loaders using Agent RT `FileSystem` | ✅ plus many loaders |
| Directory loader | ✅ `DirectoryLoader` | ✅ `DirectoryLoader` using Agent RT `FileSystem` | ✅ plus many loader variants |
| Recursive character splitter | ✅ | ✅ | ✅ |
| Embeddings interface | ✅ `AgentRTEmbeddings` | ✅ `AgentRTEmbeddings` | ✅ many provider integrations |
| In-memory vector store | ✅ `MemoryVectorStore` | ✅ `MemoryVectorStore` | ✅ |
| Agent RT retrieval-provider bridge | ✅ `AgentRTRetriever` | ✅ `AgentRTRetriever` | — Agent RT-specific |
| Stuff / map-reduce / refine document chains | ✅ | ✅ | ✅ broader chain families |
| Retrieval chain | ✅ `create_retrieval_chain()` | ✅ `createRetrievalChain()` | ✅ |
| Common multimodal content blocks | ⚠️ text/image/audio/video/pdf/document/file/json normalization | ⚠️ same common normalization | Broader provider-specific multimodal/content-block support |
| Zod schema support | — | ⚠️ `toJSONSchema()` for provider/tool schema plus `parse()` / `safeParse()` local structured-output validation | ✅ native Zod-oriented tool/schema integrations |
| Short-term thread memory / checkpoint resume | ⚠️ `MemorySaver` + Agent RT checkpoint stores keyed by `thread_id` | ⚠️ same common path | ✅ LangGraph checkpointing/persistence |
| Long-term key/value store | ⚠️ `InMemoryStore` | ⚠️ `InMemoryStore` | ✅ LangGraph stores plus external backends |
| Middleware hooks | ⚠️ before/after model, wrapped tools, post-agent | ⚠️ equivalent common hooks | ✅ extensible middleware system |
| Human-in-the-loop tool approval | ⚠️ approval callback middleware | ⚠️ approval callback middleware | ✅ graph interrupts / durable resume workflows |
| Runnable piping / common LCEL composition | ✅ `pipe()`, sequence, `RunnableLambda`, `RunnablePassthrough` | ✅ same common composition path | ✅ broader LCEL / Runnable composition |
| Callback support | ✅ lightweight function/object callback path | ✅ lightweight function/object callback path | ✅ broader callback managers + integration hooks |
| Prompt templates | ✅ `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder` | ✅ same common prompt set | ✅ broader prompt/template ecosystem |
| External vector databases | ⚠️ Chroma/Pinecone/Qdrant/Milvus/Weaviate package-shaped read/query adapters + injected `ExternalVectorStoreRetriever`; no vendor SDK clients | ⚠️ matching package-export adapters + injected bridge | ✅ broad vector-store ecosystem with full vendor lifecycles |
| Advanced retrievers / hybrid search / rerankers | ✅ `EnsembleRetriever` weighted RRF + `ContextualCompressionRetriever` over injected retrievers/compressors | ✅ same common composition/reranking path | ✅ broad integration ecosystem |
| General output-parser families | ✅ string + JSON + Pydantic-style parsing | ✅ string + JSON parsing | ✅ broader parser ecosystem |
| LangChain package dependency required | ❌ compatibility layer is dependency-free with respect to LangChain | ❌ compatibility layer is dependency-free with respect to LangChain | ✅ upstream packages themselves |

The compatibility target is the common model/tool/agent/RAG migration path, not API-complete emulation of every optional LangChain integration package. The five supported vector DB packages above provide retrieval/query compatibility only; vendor-specific write/admin APIs, other loaders, other rerankers, and other middleware can remain upstream or be injected through the documented Agent RT compatibility contracts.

### TypeScript / JavaScript LangChain migration

TypeScript/JavaScript applications can use the published `agent-rt/langchain` entry point without installing LangChain:

```ts
import {
  ChatOpenAI,
  HumanMessage,
  createAgent,
  tool,
} from "agent-rt/langchain";

const weather = tool(
  async ({ city }) => ({ city, forecast: "sunny" }),
  {
    name: "weather",
    description: "Get weather for a city",
    schema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
);

const model = new ChatOpenAI({ model: "your-model" });
const response = await model.invoke([new HumanMessage("Hello")]);

const agent = createAgent({
  model,
  tools: [weather],
  systemPrompt: "Be concise.",
});
const result = await agent.invoke({
  messages: [{ role: "user", content: "Weather in Boston?" }],
});
```

Supported in TypeScript/JavaScript:

- Messages: `BaseMessage`, `HumanMessage`, `AIMessage`, `AIMessageChunk`, `SystemMessage`, and `ToolMessage`, including text/content blocks, additional/response metadata, usage metadata, names/IDs, tool calls, and tool-call IDs.
- Models: `ChatOpenAI`, `ChatAnthropic`, and `initChatModel()`, with string/message inputs plus async `invoke()`, `batch()`, `stream()`, `bind()`, and `bindTools()`.
- Tools: OpenAI-style function schemas and LangChain-like tool objects. The compatibility `tool()` helper accepts JSON Schema directly and also accepts schema objects exposing `toJSONSchema()`. `StructuredTool`, `DynamicStructuredTool`, and `DynamicTool` validate input against their `schema` (JSON Schema `required`, or a zod-like `safeParse()`), raising `ToolInputParsingException`; a `tool_call` input returns a `ToolMessage`. LangChain v1's `fakeModel()` test helper is exported: `respond(message | Error)` and `respondWithTools([{ name, args, id? }])` script responses in order (any message object with `content` / `tool_calls` works), `calls` / `callCount` record requests, and an unscripted call throws. It runs through the same agent loop as real models, without network access. `createAgent().invoke()` returns the final state with `messages` normalized to `BaseMessage` objects, as LangChain does. As in LangChain's tool node, input a tool's schema rejects becomes a `ToolInvocationError` (with `toolCall`, `toolError`, and the error's stack in its message), while an error the tool itself throws is passed on unchanged; `toolErrorMiddleware({ onError })` turns either into the tool message the model receives. `ToolInputParsingException` messages use zod's `prettifyError()` format. Callback-handler objects passed as `invoke(input, { callbacks })` receive `handleToolStart` / `handleToolEnd` / `handleToolError` for each tool call, and the model gets the same `callbacks` (Agent RT chat models call `handleChatModelStart` / `handleLLMEnd` / `handleLLMError`).
- Structured output: `withStructuredOutput()` routes JSON Schema through Agent RT `StructuredOutputRequirement`; `includeRaw: true` returns `raw`, `parsed`, and `parsing_error`. Agent `responseFormat` accepts `ProviderStrategy`, `ToolStrategy`, and `AutoStrategy`; tool strategy uses a synthetic output tool with bounded validation retries.
- Agents: `createAgent()` accepts `model` or `llm`, runs a bounded model -> tool -> model loop with `systemPrompt`, tool execution, optional final structured response parsing, async invocation, and `updates` / `messages` / `values` streaming shapes. `MemorySaver` and native Agent RT `CheckpointStore` implementations support `thread_id` resume; `InMemoryStore`, context/state schema validation, runtime injection, and middleware hooks cover the common stateful-agent path.
- Runnable/config: chat and structured runnables support `withRetry()`, fallbacks, `pipe()`, configurable fields, lightweight function/object callbacks, per-input batch config, and bounded concurrency.
- Retrieval/documents: `Document`, `TextLoader`, `JSONLoader`, `CSVLoader`, `DirectoryLoader`, `RecursiveCharacterTextSplitter`, `AgentRTEmbeddings`, `MemoryVectorStore`, `AgentRTRetriever`, `ExternalVectorStoreRetriever`, `EnsembleRetriever`, and `ContextualCompressionRetriever` cover the common provider-neutral and injected-vector-store retrieval path.
- Prompts/LCEL and chains: `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `RunnableLambda`, `RunnablePassthrough`, `pipe()`, and stuff/map-reduce/refine/retrieval document chains cover the common composition path.
- Structured validation accepts JSON Schema / `toJSONSchema()` and can also use schema objects exposing `parse()` or `safeParse()` for local validation when JSON Schema conversion is unavailable.

```ts
const schema = {
  type: "object",
  properties: { answer: { type: "string" } },
  required: ["answer"],
};

const structured = model.withStructuredOutput(schema, { includeRaw: true });
const value = await structured.invoke("Return an answer object.");

const batch = await model.batch(["first prompt", "second prompt"]);
for await (const chunk of model.stream("Stream a response.")) {
  process.stdout.write(chunk.text);
}
```

TypeScript/JavaScript follows the same completed common-migration scope as Python. Provider-specific multimodal aliases and bundled vendor integration clients remain explicit non-goals; injected Agent RT providers, retrievers, vector stores, compressors, and middleware are the extension points for those ecosystems.

## LlamaIndex

Use exact-prefix imports where possible: `llama_index.llms.openai` becomes `agent_rt.llama_index.llms.openai`. The historical flattened `agent_rt.llamaindex` entry point remains available for the broader combined compatibility surface:

```python
import os

from agent_rt.llama_index.llms.openai import OpenAI
from agent_rt.llamaindex import FunctionAgent, FunctionTool, Memory

def weather(city: str) -> str:
    """Return deterministic weather."""
    return f"Sunny in {city}."

llm = OpenAI(model=os.environ["OPENAI_MODEL"])
tool = FunctionTool.from_defaults(fn=weather)
agent = FunctionAgent(
    llm=llm,
    tools=[tool],
    system_prompt="Be concise.",
)
memory = Memory.from_defaults(session_id="weather-thread")
response = await agent.run("What is the weather in SF?", memory=memory)
print(response)
```

Supported:

- LLM/messages: `OpenAI`, `Anthropic`, `ChatMessage`, `MessageRole`, `ChatResponse`, and `CompletionResponse`, with sync/async `chat()` / `achat()` and `complete()` / `acomplete()`. `ChatMessage.from_str()` covers the common string-message helper.
- Streaming: `stream_chat()` / `astream_chat()` and `stream_complete()` / `astream_complete()` emit LlamaIndex-shaped responses with `delta` fields while routing through Agent RT streaming providers. The synchronous wrappers currently materialize the async provider stream before returning the iterator, so exact upstream backpressure/timing semantics are not claimed.
- Function tools: `FunctionTool.from_defaults()`, `ToolMetadata`, and `ToolOutput` support annotated callable schema inference, direct sync/async invocation, and conversion to Agent RT tool definitions. `predict_and_call()` / `apredict_and_call()` cover the common single-model tool-selection path.
- Structured output: `as_structured_llm()`, `structured_predict()`, and `astructured_predict()` accept raw JSON Schema mappings or Pydantic-style model classes and translate them into Agent RT `StructuredOutputRequirement`.
- Agents: `FunctionAgent` provides a bounded model -> tool -> model loop with system prompts, tool-call IDs, sync/async tools, bounded tool retries, return-error policy, and explicit `handoff_to_agent` routing. `ReActAgent` uses a Thought/Action/Action Input/Observation/Answer parser loop with bounded parse correction; `CodeActAgent` uses a code-block/Answer parser and executes code only through an explicitly supplied `code_executor`. `AgentWorkflow.from_agents()` routes named-agent handoffs with shared memory/context and bounded handoff depth, while `from_tools_or_functions()` retains the common single-agent constructor.
- Workflow/memory: `Workflow`, `step`, typed `Event` subclasses, `WorkflowMiddleware`, fan-out event lists, `Context.collect_events()` fan-in, `InputRequiredEvent` / `HumanResponseEvent` HITL resume, and Agent RT checkpoint stores cover durable workflow migration. `Context` supports JSON serialization plus checkpoint save/load. `Memory` / `ChatMemoryBuffer` add async aliases, token-budget flushing, `StaticMemoryBlock`, `FactExtractionMemoryBlock`, `VectorMemoryBlock`, and chat-store-backed memory blocks.
- RAG/indexing: `Document`, `TextNode`, `NodeWithScore`, Agent RT `FileSystem`-backed `SimpleDirectoryReader`, `SentenceSplitter`, `IngestionPipeline`, `AgentRTEmbedding`, `SimpleVectorStore`, `StorageContext`, `VectorStoreIndex`, `VectorIndexRetriever`, `AgentRTRetriever`, `RetrieverQueryEngine`, and `ResponseSynthesizer` cover the common local/provider-neutral RAG migration path. Read/query compatibility modules are also provided for Chroma, Pinecone, Qdrant, Milvus, and Weaviate and route through Agent RT's vector DB HTTP providers.
- Module paths: the `llama_index.core` tree (`core.schema`, `core.node_parser`, `core.workflow`, `core.prompts`, `core.llms`, `core.tools`, `core.memory`, `core.agent.workflow`, `core.base.llms.types`, `core.base.embeddings.base`, `core.storage`, `core.vector_stores`, and the other listed core subpackages) plus `llama_index.llms.ollama`, `llama_index.embeddings.openai`, and `llama_index.embeddings.ollama` resolve under the `agent_rt.` prefix. `OpenAIEmbedding`, `OllamaEmbedding`, and `BaseEmbedding` add sync `get_text_embedding()` / `get_query_embedding()` / `get_text_embedding_batch()`; `Ollama` and `OllamaEmbedding` use Ollama's OpenAI-compatible `/v1` endpoint (`base_url`, else `OLLAMA_HOST`), and `VectorStoreIndex(nodes, storage_context=...)` defaults `embed_model` to `Settings.embed_model` and embeds nodes that have no embedding yet, and `NodeWithScore` proxies `text`, `metadata`, and `get_text()` / `get_content()` to its node. `ChromaVectorStore(chroma_collection=collection)` reads and writes the chromadb collection the application created (queries return `1 / (1 + distance)` as the score); without one it queries Agent RT's Chroma provider. `SimpleDirectoryReader` takes the upstream `input_dir` / `input_files` / `recursive` / `filename_as_id` / `required_exts` arguments and reads the local paths the application names, raising `ValueError` when nothing matches; pass `file_system=` to read through an Agent RT `FileSystem` instead. Property graphs are not supported: `PropertyGraphIndex`, the property-graph extractors and retrievers (including the subclassable `CustomPGRetriever`), and `load_index_from_storage` can be imported so modules load, but using them raises `NotImplementedError`. `GeminiEmbedding` uses Gemini's OpenAI-compatible endpoint with `api_key`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY`. `llama_index.readers.file` provides `FlatReader` (one document per file) and `PDFReader` (one document per page, needs `pypdf`); both read the path the application gives them, or read through an Agent RT `FileSystem` passed as `file_system`.
- Prompts/settings/observability: `PromptTemplate`, `ChatPromptTemplate`, `JSONOutputParser`, `Settings`, and `CallbackManager` cover common migration shapes. Non-streaming calls emit `llm-start` / `llm-end` / `llm-error`; provider streams additionally emit `llm-stream`. Model, finish-reason, and usage metadata are retained on response messages. `count_tokens()` / `acount_tokens()` use the provider token-count contract when available and can fall back to `Settings.tokenizer`. Multimodal Agent RT content parts are retained in `ChatMessage.blocks`, split streaming tool-call arguments are accumulated by call ID, and structured chat/completion streams expose the best parseable partial object in `raw`.

Structured prediction and direct tool use can migrate independently of the agent loop:

```python
schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

parsed = await llm.astructured_predict(schema, "Return an answer object.")
selected = await llm.apredict_and_call([tool], user_msg="Weather in SF?")
```

### LlamaIndex vector database migration

Python vector-store modules preserve the upstream module hierarchy after adding the `agent_rt.` prefix:

| Upstream Python import | Agent RT Python import |
| --- | --- |
| `llama_index.vector_stores.chroma.ChromaVectorStore` | `agent_rt.llama_index.vector_stores.chroma.ChromaVectorStore` |
| `llama_index.vector_stores.pinecone.PineconeVectorStore` | `agent_rt.llama_index.vector_stores.pinecone.PineconeVectorStore` |
| `llama_index.vector_stores.qdrant.QdrantVectorStore` | `agent_rt.llama_index.vector_stores.qdrant.QdrantVectorStore` |
| `llama_index.vector_stores.milvus.MilvusVectorStore` | `agent_rt.llama_index.vector_stores.milvus.MilvusVectorStore` |
| `llama_index.vector_stores.weaviate.WeaviateVectorStore` | `agent_rt.llama_index.vector_stores.weaviate.WeaviateVectorStore` |

TypeScript/JavaScript package-export aliases are `agent-rt/@llamaindex/chroma`, `agent-rt/@llamaindex/pinecone`, `agent-rt/@llamaindex/qdrant`, `agent-rt/@llamaindex/milvus`, and `agent-rt/@llamaindex/weaviate`.

```python
import os

from agent_rt.llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(environment=os.environ)
nodes = await vector_store.aquery(
    {
        "query_embedding": query_embedding,
        "query_str": "How does Agent RT migrate vector retrieval?",
        "similarity_top_k": 4,
    }
)
```

These modules target existing vector collections. Query/search and conversion to LlamaIndex-shaped `NodeWithScore` results are supported; `add`/upsert, deletion, collection administration, and vendor-specific sparse/hybrid lifecycle features intentionally fail or remain backend-native. This prevents the compatibility layer from silently creating or mutating external indexes.

LlamaIndex compatibility still intentionally leaves specialized integrations and byte-for-byte upstream internals outside the adapter:

- Advanced RAG integrations beyond the provider-neutral core adapter: vector DB write/admin lifecycles, graph/property indexes, hybrid/sparse retrieval and reranking, specialized readers/parsers, persistent index/document stores, and exact ingestion-cache behavior. Chroma, Pinecone, Qdrant, Milvus, and Weaviate have read/query compatibility modules, but not full upstream vendor lifecycle parity.
- Vendor-specific hosted memory/vector/chat stores that require their own authentication or database runtime. The compatibility memory blocks accept injected chat stores/retrievers instead of opening hidden services.
- Provider-specific prompt wording, private parser internals, callback payload minutiae, and every vendor-specific multimodal/message option. The compatibility layer preserves the documented migration semantics above rather than claiming exact internal implementation identity.

Where exact vendor lifecycle or provider-specific behavior matters, use the corresponding native Agent RT provider, retrieval, checkpoint, workflow, or storage contract explicitly.

### LlamaIndex compatibility coverage

This matrix compares the compatibility surface implemented by Agent RT with the corresponding upstream LlamaIndex Framework capabilities, compared with the LlamaIndex developer documentation (as of the last review; check the upstream version you target) for agents, memory, state/context persistence, indexing, storage, response synthesis, rerankers/postprocessors, evaluation, and LlamaParse/LlamaCloud. `✅` means the common migration path is implemented, `⚠️` means Agent RT intentionally implements a narrower provider-neutral subset, and `❌` means that upstream capability is not implemented by `agent_rt.llamaindex` / `agent-rt/llamaindex`.

| Area | Agent RT Python | Agent RT TypeScript / JavaScript | Upstream LlamaIndex |
| --- | --- | --- | --- |
| OpenAI LLM wrapper | ✅ `OpenAI` | ✅ `OpenAI` / `openai()` | ✅ |
| Anthropic LLM wrapper | ✅ `Anthropic` | ✅ `Anthropic` / `anthropic()` | ✅ |
| Chat messages / roles | ✅ `ChatMessage`, `MessageRole` | ✅ common message shapes | ✅ |
| Chat responses | ✅ `ChatResponse` | ✅ LlamaIndex-shaped chat responses | ✅ |
| Completion responses | ✅ `CompletionResponse` | ✅ LlamaIndex-shaped completion responses | ✅ |
| Sync chat/completion | ✅ `chat()` / `complete()` | — JavaScript API is async | ✅ Python; JS is async |
| Async chat/completion | ✅ `achat()` / `acomplete()` | ✅ `chat()` / `complete()` return Promises | ✅ |
| Chat streaming | ✅ `stream_chat()` / `astream_chat()` | ✅ `stream: true` async iterables | ✅ |
| Completion streaming | ✅ `stream_complete()` / `astream_complete()` | ✅ `stream: true` async iterables | ✅ |
| Usage / finish-reason metadata | ✅ | ✅ | ✅ provider dependent |
| Split streaming tool-call delta accumulation | ✅ by tool-call ID | ✅ by tool-call ID | ✅ integration dependent |
| Provider token counting | ✅ `count_tokens()` / `acount_tokens()` + tokenizer fallback | ✅ `countTokens()` + tokenizer fallback | ✅ provider/tokenizer dependent |
| Function tools | ✅ `FunctionTool.from_defaults()` | ✅ `FunctionTool.fromDefaults()` and `tool()` | ✅ |
| Callable schema inference | ✅ Python annotations → basic JSON Schema | — | ✅ richer typing/schema integrations |
| JSON Schema tool parameters | ✅ | ✅ | ✅ |
| Direct tool invocation | ✅ sync/async | ✅ async | ✅ |
| Tool retries | ✅ bounded retries | ✅ bounded `maxRetries` | ✅ integration/agent dependent |
| Tool error-as-result behavior | ✅ | ✅ `errorBehavior: "return_error"` | ✅ patterns available upstream |
| Return-direct tools | ✅ common agent path | ✅ `returnDirect` | ✅ |
| Structured LLM wrapper | ✅ `as_structured_llm()` | ✅ `asStructuredLLM()` | ✅ |
| Structured prediction | ✅ `structured_predict()` / `astructured_predict()` | ✅ `structuredPredict()` | ✅ |
| Raw JSON Schema structured output | ✅ | ✅ | ✅ |
| Pydantic-style schema classes | ✅ `model_json_schema()` / `schema()` style | — | ✅ Python |
| Structured streaming partial parsing | ✅ best parseable partial | ✅ best parseable partial | ✅ implementation/provider dependent |
| `FunctionAgent` | ✅ bounded tool loop | ✅ bounded tool loop | ✅ |
| ReAct agent | ✅ `ReActAgent` | ✅ `ReActAgent` | ✅ |
| Agent event stream | ✅ `AgentStream`, `ToolCall`, `ToolCallResult`, `AgentOutput` | ✅ `AgentInput`, `AgentStream`, `ToolCall`, `ToolCallResult`, `AgentOutput` | ✅ |
| Awaitable agent run | ✅ | ✅ | ✅ |
| Async-iterable agent run | ✅ event stream | ✅ event stream | ✅ |
| Multi-agent workflow | ✅ `AgentWorkflow.from_agents()` | ✅ `AgentWorkflow.fromAgents()` | ✅ |
| Explicit agent handoff tool | ✅ `handoff_to_agent` | ✅ `handoff_to_agent` | ✅ multi-agent workflows |
| Shared memory/context across handoffs | ✅ common path | ✅ common path | ✅ |
| Bounded handoff depth | ✅ | ✅ | ✅ compatibility safety bound; upstream behavior varies |
| Workflow framework | ✅ `Workflow`, `step`, typed `Event` | ✅ `Workflow`, `step()`, typed workflow events | ✅ |
| Workflow fan-out | ✅ event lists | ✅ event lists | ✅ |
| Workflow fan-in / event collection | ✅ `Context.collect_events()` | ✅ `collectEvents()` | ✅ |
| Human-in-the-loop workflow events | ✅ `InputRequiredEvent` / `HumanResponseEvent` | ✅ same common flow | ✅ |
| Workflow middleware | ✅ before/after step hooks | ✅ middleware hooks | ✅ broader workflow extension surface |
| Workflow checkpoint save/load | ✅ Agent RT `CheckpointStore` | ✅ Agent RT `CheckpointStore` | ✅ workflow persistence patterns |
| Context key/value store | ✅ | ✅ | ✅ |
| Context JSON serialization | ✅ | ✅ | ✅ |
| Basic chat memory | ✅ `Memory` / `ChatMemoryBuffer` | ✅ `Memory` / `ChatMemoryBuffer` | ✅; upstream deprecates `ChatMemoryBuffer` in favor of `Memory` |
| Token-budget memory flushing | ✅ | ✅ common memory limit behavior | ✅ |
| `Document` / `TextNode` / `NodeWithScore` | ✅ | ✅ | ✅ |
| Simple directory reader | ✅ Agent RT `FileSystem`-backed | ✅ Agent RT `FileSystem`-backed | ✅ plus broad reader ecosystem |
| Sentence splitter | ✅ `SentenceSplitter` | ✅ `SentenceSplitter` | ✅ |
| Ingestion pipeline | ✅ `IngestionPipeline` | ✅ `IngestionPipeline` | ✅ |
| Embedding model bridge | ✅ `AgentRTEmbedding` | ✅ `AgentRTEmbedding` | ✅ many embedding integrations |
| Simple in-memory vector store | ✅ `SimpleVectorStore` | ✅ `SimpleVectorStore` | ✅ |
| External vector DB modules | ⚠️ Chroma/Pinecone/Qdrant/Milvus/Weaviate read/query modules; mutation remains backend-native | ⚠️ matching `agent-rt/@llamaindex/*` package exports | ✅ broad vector-store integrations with vendor-specific lifecycles |
| Storage context | ✅ `StorageContext` | ✅ `StorageContext` | ✅ |
| Vector store index | ✅ `VectorStoreIndex` | ✅ `VectorStoreIndex` | ✅ |
| Vector index retriever | ✅ `VectorIndexRetriever` | ✅ `VectorIndexRetriever` | ✅ |
| Agent RT retrieval-provider bridge | ✅ `AgentRTRetriever` | ✅ `AgentRTRetriever` | — Agent RT-specific |
| Retriever query engine | ✅ `RetrieverQueryEngine` | ✅ `RetrieverQueryEngine` | ✅ |
| Prompt template | ✅ `PromptTemplate` | ✅ `PromptTemplate` | ✅ |
| Chat prompt template | ✅ `ChatPromptTemplate` | ✅ `ChatPromptTemplate` | ✅ |
| JSON output parser | ✅ `JSONOutputParser` | ✅ `JSONOutputParser` | ✅ |
| Global settings | ✅ `Settings` | ✅ `Settings` | ✅ |
| Callback manager | ✅ lightweight callback events | ✅ lightweight callback events | ✅ richer instrumentation/callback ecosystem |
| LLM start/end/error callbacks | ✅ | ✅ | ✅ |
| Streaming callback events | ✅ `llm-stream` | ✅ `llm-stream` | ✅ |
| Exact sync streaming backpressure/timing | ⚠️ sync wrappers materialize async streams | — | ✅ native implementation semantics |
| Common multimodal content blocks | ⚠️ Agent RT content parts retained in `ChatMessage.blocks` | ⚠️ common Agent RT blocks preserved | ✅ broader provider-specific multimodal support |
| Zod-like tool/schema conversion | — | ⚠️ common dependency-free shapes / `toJSONSchema()` | ✅ native TS schema ecosystem |
| Abort/cancellation propagation | ⚠️ | ✅ `AbortSignal` through model/tool calls | ✅ framework/runtime dependent |
| `predict_and_call` helper | ✅ `predict_and_call()` / `apredict_and_call()` | ⚠️ covered through tool-enabled LLM/agent paths, not the same Python helper | ✅ Python |
| Zod structured output | — | ⚠️ common Zod shapes / parser-like objects | ✅ TS ecosystem |
| Agent cancellation | ⚠️ | ✅ `cancel()` with abort propagation | ✅ runtime dependent |
| Response synthesizer | ⚠️ basic `ResponseSynthesizer` | ⚠️ basic `ResponseSynthesizer` | ✅ documented modes include `refine`, `compact`, `tree_summarize`, `simple_summarize`, `no_text`, `accumulate`, and `compact_accumulate` |
| Query-engine tools | ❌ dedicated `QueryEngineTool` compatibility | ❌ dedicated `QueryEngineTool` compatibility | ✅ documented tool abstraction |
| Prebuilt tool specs / API tool integrations | ❌ | ❌ | ✅ broad tool-spec/integration ecosystem |
| CodeAct agent | ✅ `CodeActAgent` with explicit `code_executor` | ❌ | ✅ Python agent ecosystem |
| Planning agent | ❌ dedicated compatibility class | ✅ Agent RT `PlanningAgent` with explicit planner callback | ⚠️ upstream planning is supported as an agent/workflow pattern; this compatibility class is not claimed as an exact current core symbol |
| Exact upstream workflow internals / scheduling semantics | ❌ byte-for-byte parity | ❌ byte-for-byte parity | ✅ native implementation |
| Static memory block | ✅ `StaticMemoryBlock` | ❌ | ✅ Python memory system |
| Fact-extraction memory block | ✅ `FactExtractionMemoryBlock` | ❌ | ✅ Python memory system |
| Vector memory block | ✅ `VectorMemoryBlock` | ❌ | ✅ Python memory system |
| Chat-store-backed memory block | ✅ `ChatStoreMemoryBlock` | ❌ dedicated compatibility class | Not one of the three current predefined memory blocks documented upstream |
| Vendor-hosted memory/chat stores | ❌ direct vendor clients; inject stores/retrievers | ❌ direct vendor clients | ✅ integration ecosystem |
| Specialized readers/parsers | ❌ except core adapter | ⚠️ injected `LlamaParseReader` contract | ✅ many readers + LlamaParse integrations |
| Exact ingestion cache semantics | ❌ | ❌ | ✅ native ingestion/cache implementations |
| Persist/load storage context, docstore, index store | ❌ full upstream persistence API | ❌ full upstream persistence API | ✅ filesystem persistence plus swappable document/index/vector stores |
| Summary/List index | ❌ | ❌ | ✅ |
| Tree index | ❌ | ❌ | ✅ |
| Keyword table index | ❌ | ❌ | ✅ |
| Knowledge-graph / property-graph indexes | ❌ | ❌ | ✅ |
| Document summary index | ❌ | ❌ | ✅ |
| Object index | ❌ | ❌ | ✅ |
| SQL / structured-data query engines | ❌ | ❌ | ✅ |
| Chat engines / `as_chat_engine()` | ❌ | ❌ | ✅ multiple documented chat modes |
| Vendor SDK clients for vector databases | ❌ not bundled (read/query adapters above only) | ❌ not bundled | ✅ broad ecosystem |
| Hybrid / sparse retrieval | ❌ | ❌ | ✅ |
| Advanced rerankers / postprocessors | ❌ | ❌ | ✅ documented similarity filters, cross-encoder/hosted rerankers, LLM rerankers, and other node postprocessors |
| Persistent document/index stores | ❌ vendor-specific persistence | ❌ vendor-specific persistence | ✅ |
| Broader output-parser families | ❌ | ❌ | ✅ |
| Exact callback payload/span semantics | ❌ | ❌ | ✅ native integrations |
| LlamaParse hosted authentication/API lifecycle | ❌ | ❌; TS adapter requires injected parser | ✅ hosted integration |
| LlamaCloud retrieval/index management | ❌ | ⚠️ `LlamaCloudRetriever` / `LlamaCloudIndex` over injected Agent RT retrieval | ✅ hosted project/index lifecycle |
| Faithfulness evaluator | ❌ | ✅ lightweight LLM-backed `FaithfulnessEvaluator` | ✅ |
| Relevancy evaluator | ❌ | ✅ lightweight LLM-backed `RelevancyEvaluator` | ✅ |
| Broader evaluation suite | ❌ | ❌ | ✅ broader evaluator/evaluation ecosystem |
| LlamaIndex package dependency required | ❌ compatibility layer does not require LlamaIndex | ❌ compatibility layer does not require LlamaIndex | ✅ upstream packages themselves |

The compatibility target is the common LLM/tool/agent/workflow/RAG migration path, not API-complete LlamaIndex emulation. Applications that depend on hosted LlamaCloud/LlamaParse lifecycle semantics, specialized indexes and storage backends, advanced retrievers/rerankers, exact workflow scheduling internals, or the full evaluation/instrumentation ecosystem should use native Agent RT contracts or retain the relevant upstream LlamaIndex packages.

### TypeScript / JavaScript LlamaIndex migration

TypeScript/JavaScript applications can use `agent-rt/llamaindex` for the current LlamaIndex.TS LLM/tool/agent-workflow migration path without installing LlamaIndex:

```ts
import {
  AgentStream,
  Memory,
  OpenAI,
  agent,
  tool,
} from "agent-rt/llamaindex";

const weather = tool({
  name: "weather",
  description: "Get weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: async ({ city }) => `Sunny in ${city}.`,
});

const llm = new OpenAI({ model: "your-model" });
const memory = Memory.fromDefaults({ sessionId: "weather-thread" });
const workflow = agent({
  llm,
  tools: [weather],
  systemPrompt: "Be concise.",
});

const run = workflow.run("Weather in SF?", { memory });

for await (const event of run) {
  if (event instanceof AgentStream) {
    process.stdout.write(event.data.delta);
  }
}

const result = await run;
console.log(result.data);
```

Supported in TypeScript/JavaScript:

- LLMs: `OpenAI`, `Anthropic`, `openai()`, and `anthropic()` route through Agent RT `ModelProvider` contracts. `chat({ messages })` and `complete({ prompt })` support non-streaming and `stream: true` async-iterable response shapes.
- Messages/tool calls: user/system/assistant/tool chat messages translate to Agent RT `ModelMessage`; assistant tool calls and tool-result IDs are preserved across model -> tool -> model turns.
- Tools: current `tool({...})` and `tool(fn, options)` forms plus `FunctionTool.fromDefaults()` are supported. Parameters may be raw JSON Schema or schema objects exposing `toJSONSchema()`; executable tools expose metadata, `execute()`, `call()`, and Agent RT `ToolDefinition` translation. Compatibility tools support bounded `maxRetries`, `errorBehavior: "return_error"`, `returnDirect`, and `AbortSignal` propagation.
- Structured output: `structuredPredict()` and `asStructuredLLM()` translate raw JSON Schema, `toJSONSchema()` objects, and common dependency-free Zod schema shapes (object/string/number/boolean/literal/enum/optional/nullable/array/union) to Agent RT `StructuredOutputRequirement`. Parser-like objects exposing `parse()` or `safeParse()` validate parsed JSON. Structured chat/completion support `stream: true` and expose the best parseable partial object in `raw`.
- Agents/workflows: `agent()` and `FunctionAgent` provide the bounded native tool loop; `ReActAgent` implements a Thought/Action/Action Input/Observation/Answer parser loop; `PlanningAgent` supports explicit task-decomposition callbacks; and `AgentWorkflow.fromAgents()` routes named-agent handoffs with shared memory/context and bounded handoff depth. `run()` remains awaitable/async iterable, exposes `cancel()`, and forwards `AbortSignal` into Agent RT model/tool calls.
- Agent events: `AgentInput`, `AgentStream`, `ToolCall`, `ToolCallResult`, and `AgentOutput` are emitted from the compatibility run context. With a streaming Agent RT provider, `AgentStream` follows model text deltas.
- Memory/state/workflows: `Memory` / `ChatMemoryBuffer` cover common in-process history methods. `Context` supplies an async key/value store, process-friendly JSON round trips, fan-in `collectEvents()`, and native Agent RT `CheckpointStore` save/load. `Workflow`, `step()`, typed workflow events, middleware, fan-out lists, fan-in collection, HITL `respond()`, and checkpoint-backed resume cover the common `@llamaindex/workflow` migration path.
- RAG/indexing: `Document`, `TextNode`, `NodeWithScore`, Agent RT `FileSystem`-backed `SimpleDirectoryReader`, `SentenceSplitter`, `IngestionPipeline`, `AgentRTEmbedding`, `SimpleVectorStore`, `StorageContext`, `VectorStoreIndex`, `VectorIndexRetriever`, `AgentRTRetriever`, `RetrieverQueryEngine`, `ResponseSynthesizer`, and `getResponseSynthesizer()` cover the common local RAG migration path. External retrieval stays behind Agent RT `RetrievalProvider` / `RetrievalRegistry` rather than opening hidden vendor stores.
- Prompts/settings/observability: `PromptTemplate`, `ChatPromptTemplate`, `JSONOutputParser`, `Settings`, and `CallbackManager` cover common prompt/global-configuration migration. Non-streaming calls emit `llm-start` / `llm-end` / `llm-error`; provider streams additionally emit `llm-stream`. `LLM.countTokens()` uses Agent RT `TokenCountingModelProvider` when available and can fall back to `Settings.tokenizer`. Non-text response blocks are preserved, split tool-call deltas are merged by call ID, and `Settings.llm` / `Settings.embedModel` remain honored by agents, RAG, synthesis, and evaluators.
- Hosted-contract adapters/evaluation: `LlamaParseReader` requires an explicit Agent RT `FileSystem` plus injected parser, while `LlamaCloudRetriever` / `LlamaCloudIndex` require an explicit Agent RT retrieval provider or registry. `FaithfulnessEvaluator` and `RelevancyEvaluator` provide lightweight LLM-backed evaluation without opening hidden vendor connections.

The TypeScript adapter still intentionally leaves specialized integrations and byte-for-byte upstream internals outside the compatibility layer:

- Advanced RAG integrations beyond the core adapter: third-party vector-database-specific stores, graph/property indexes, advanced hybrid/sparse retrieval and reranking, specialized readers/parsers, persistent document/index stores, and exact upstream ingestion-cache semantics.
- Exact LlamaParse/LlamaCloud hosted authentication, project/index creation, management APIs, and vendor lifecycle semantics. The compatibility adapters intentionally cover injected parsing and Agent RT retrieval contracts, with no hidden hosted state or side effects.
- Rare Zod refinements/transforms/custom schema internals and provider-specific prompt/message/tool option minutiae that cannot be inferred dependency-free. Common Zod shapes, streamed structured partials, callback streaming, tool-delta merging, and multimodal blocks are supported as described above.

Use native Agent RT provider, retrieval, checkpoint, workflow, and storage contracts when those vendor-specific lifecycle details matter.

## OpenAI Agents SDK

OpenAI Agents applications can migrate the common agent/tool/handoff path without installing the upstream SDK. In Python, preserve the upstream `agents` import shape by prepending `agent_rt.`:

```python
from agent_rt.agents import Agent, Runner, function_tool

@function_tool
def lookup(topic: str) -> str:
    """Look up a topic."""
    return f"Found {topic}"

agent = Agent(
    name="Researcher",
    instructions="Use tools when useful.",
    tools=[lookup],
)

result = await Runner.run(agent, "Research Agent RT")
print(result.final_output)
```

`agent_rt.openai_agents` is provided as a descriptive alias for the same Python surface. The compatibility layer supports `Agent`, `Runner.run()`, `Runner.run_sync()`, `function_tool`, `Agent.as_tool()`, and bounded `handoff()` delegation. Tool execution is routed through Agent RT `ToolRegistry` and `AgentLoop`, so Agent RT argument validation, registration scanning, permissions/guardrails configured on native integrations, and run limits remain available. Default handoff (`transfer_to_<agent>`) and `Agent.as_tool()` names use the SDK's lowercase function-style normalization. Agent-level `input_guardrails` / `output_guardrails` (`@input_guardrail`, `@output_guardrail`, `GuardrailFunctionOutput`, `RunContextWrapper`) are supported: input guardrails finish before the first model call, so a tripwire raises `InputGuardrailTripwireTriggered` before any model or tool side effect, and output guardrails raise `OutputGuardrailTripwireTriggered`. As in the SDK, reaching `max_turns` raises `MaxTurnsExceeded`. The `agents.exceptions`, `agents.guardrail`, `agents.run_context`, `agents.usage`, `agents.tracing`, and `agents.mcp` submodules are importable. `set_default_openai_key()` sets `OPENAI_API_KEY`; `trace()`, `set_tracing_disabled()`, and `enable_verbose_stdout_logging()` are accepted no-ops. `RunState` and the `MCPServer*` classes are importable stand-ins that raise `NotImplementedError` when used (use Agent RT approval checkpoints and an injected `MCPClient`). Handoffs are represented as bounded delegated tool calls; exact upstream ownership-transfer state, session internals, hosted tools, tracing objects, and `Runner.run_streamed()` event semantics are intentionally not emulated.

TypeScript/JavaScript uses the exact prefixed package specifier:

```ts
import { Agent, run, tool } from "agent-rt/@openai/agents";

const lookup = tool({
  name: "lookup",
  description: "Look up a topic",
  parameters: {
    type: "object",
    properties: { topic: { type: "string" } },
    required: ["topic"],
  },
  execute: async ({ topic }) => `Found ${topic}`,
});

const agent = new Agent({
  name: "Researcher",
  instructions: "Use tools when useful.",
  tools: [lookup],
});

const result = await run(agent, "Research Agent RT");
console.log(result.finalOutput);
```

The TypeScript adapter supports `Agent`, `Agent.create()`, `Agent.asTool()`, `Runner`, `run()`, `tool()`, and `handoff()`. Tool schemas may be raw JSON Schema or objects exposing `toJSONSchema()` / `jsonSchema`. Tools follow the SDK shape: `strict` (default `true`) and `invoke(runContext, inputJson)`, which parses the JSON, validates it with a zod-like `parse()` schema when given, and returns the SDK's default error text on failure unless `errorFunction: null`. `RunContext`, `Usage`, `ModelBehaviorError`, and a no-op `setTracingDisabled()` are exported. `Agent` keeps `modelSettings` and supports `clone(overrides)`. `model` may be a model name, an SDK `Model` (an object with `getResponse()` and optionally `getStreamedResponse()`, such as a test double or `aisdk(...)`), or an Agent RT provider. SDK models still run inside Agent RT's `AgentLoop`. Requests and responses are translated to the SDK's item shapes, and the original SDK items (reasoning, `providerData`, images, files) are replayed to the model unchanged on later turns. `new Runner({ modelProvider })` accepts an Agent RT `ModelProvider` or an SDK-style provider with `getModel()`. `new OpenAIProvider({ apiKey, baseURL, openAIClient })` drives an injected OpenAI-shaped client. `new OpenAIChatCompletionsModel(client, model)` is an SDK `Model` that builds Chat Completions requests the way the upstream model does: text-part content, `image_url` and `file` parts, `reasoning` replay, `reasoning_effort` from `modelSettings.reasoning`, `max_tokens`, tool `strict` flags, and `response_format` for JSON-schema output. It also aggregates streamed chunks into the same output items. The runner emits `agent_tool_start` / `agent_tool_end` through `runner.on()`. Runs accept SDK input items and send `outputType` (a JSON-schema definition or a zod-like schema) with every model request. As in the SDK, the final text is parsed as JSON, and then by a zod-like `outputType`; it is not run through Agent RT's schema validation and repair. Runs also honor `toolUseBehavior` (`stop_on_first_tool`, `{ stopAtToolNames }`, or a function), and return `history` as SDK input items plus `newItems`, `usage`, `interruptions`, and `state`. `run(agent, input, { stream: true })` returns a `StreamedRunResult` that yields `raw_model_stream_event`, `run_item_stream_event` (`tool_called`, `tool_output`, `message_output_created`), and `agent_updated_stream_event`, with `completed`, `error`, `cancelled`, and `toTextStream()`. As in the SDK, `tool()` names are normalized with `toFunctionToolName()` (characters other than letters and digits become `_`, so `update-page` is called as `update_page`), default handoff names are `transfer_to_<normalized agent name>`, and a `toolNameOverride` is used verbatim. A tool with `needsApproval` pauses the run with `interruptions`; every call in the model's turn runs (or is collected as an interruption) before the run pauses; `RunState.fromString(agent, result.state.toString())`, `approve()` / `reject(item, { message })`, and `runner.run(agent, state)` continue it, and approved calls run through Agent RT's `ToolRegistry`. Saved states use Agent RT's own format, not the SDK's. Agent-level `inputGuardrails` / `outputGuardrails` (`{ name, execute }`) are supported: input guardrails finish before the first model call and throw `InputGuardrailTripwireTriggered` on a tripwire, output guardrails throw `OutputGuardrailTripwireTriggered`, and reaching `maxTurns` throws `MaxTurnsExceededError`, as in the SDK. Tool `inputGuardrails` run before the approval check, and again when an approved call resumes. They can allow a call, return `rejectContent` text to the model, or throw `ToolGuardrailTripwireTriggered`. An error that a tool's `errorFunction` rethrows ends the run, as in the SDK. `agent-rt/openai-agents` is also available as a descriptive alias.

## AutoGen AgentChat

Current AutoGen AgentChat migration is Python-only, matching the upstream framework's Python surface. Prefix the common package paths with `agent_rt.`:

```python
from agent_rt.autogen_agentchat.agents import AssistantAgent
from agent_rt.autogen_agentchat.conditions import TextMentionTermination
from agent_rt.autogen_agentchat.teams import RoundRobinGroupChat
from agent_rt.autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(model="your-model")
assistant = AssistantAgent(
    "assistant",
    model_client=client,
    system_message="Be concise.",
)

result = await assistant.run(task="Summarize the request")

team = RoundRobinGroupChat(
    [assistant],
    termination_condition=TextMentionTermination("TERMINATE"),
    max_turns=4,
)
team_result = await team.run(task="Complete the task")
```

Supported: `AssistantAgent`, callable tools through the Agent RT tool loop, `run()` / compatibility `run_stream()`, `RoundRobinGroupChat`, `TextMentionTermination`, `MaxMessageTermination`, `TokenUsageTermination`, `TimeoutTermination`, combined termination conditions with `dump_component()` configs, `TextMessage.dump()` / `load()` / `to_text()`, `UserProxyAgent` (`on_messages()` answers from a sync or async `input_func`, optionally taking the cancellation token), `autogen_agentchat.base` (`TaskResult`, `Response`, `TerminationCondition`), `autogen_agentchat.ui.Console`, and `OpenAIChatCompletionClient` as a lightweight Agent RT provider binding. `autogen_ext.models.anthropic.AnthropicChatCompletionClient` and `autogen_ext.models.ollama.OllamaChatCompletionClient` build their Agent RT provider lazily on first use. Team turns are bounded and normalized through Agent RT.

This is not a reimplementation of AutoGen Core. Selector/swarm teams, distributed runtimes, component serialization, persisted team continuation without an explicit task, code executors, exact event streaming/backpressure, and the broader extension ecosystem remain outside the compatibility layer.

## CrewAI

CrewAI migration is also Python-only. The common sequential crew path is available under the prefixed package:

```python
from agent_rt.crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="Researcher",
    goal="Find useful facts",
    llm="your-model",
)
writer = Agent(
    role="Writer",
    goal="Write the final answer",
    llm="your-model",
)

research = Task(
    description="Research {topic}",
    expected_output="Useful notes",
    agent=researcher,
)
writing = Task(
    description="Write about {topic}",
    expected_output="Final answer",
    agent=writer,
    context=[research],
)

result = await Crew(
    agents=[researcher, writer],
    tasks=[research, writing],
    process=Process.sequential,
).kickoff_async(inputs={"topic": "Agent RT"})
```

The adapter supports `Agent`, `Task`, `Crew`, `Process.sequential`, task context, input formatting, `kickoff()`, and `kickoff_async()`, with each task executed through Agent RT's bounded agent loop. `LLM(model, temperature=..., max_tokens=..., base_url=..., api_key=...)` takes CrewAI's LiteLLM-style names (`anthropic/<model>` uses the Anthropic provider; `openai/<model>` or a bare name uses OpenAI), offers `call()` / `acall()`, and can be passed as `Agent(llm=...)`. `Process.hierarchical`, per-task `async_execution` scheduling, CrewAI Flows, manager/delegation internals, knowledge/storage lifecycle, and exhaustive CrewAI tool/plugin semantics intentionally fail or remain upstream-native rather than being silently approximated.

There is no Agent RT TypeScript CrewAI or AutoGen compatibility surface because those migration targets are Python-first; Agent RT does not invent non-upstream JavaScript APIs.

## OpenAI Python SDK

For chat completions, change the import while keeping the familiar client shape:

```python
import os

from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

`OpenAI` and `AsyncOpenAI` support `chat.completions.create()` in both normal and streaming modes, including basic messages, OpenAI-style tool definitions, JSON Schema structured outputs via `response_format`, temperature, token limits, usage, text deltas, and tool-call deltas. They also expose `client.responses.create()` with string or item-list input, `instructions`, function-call / function-call-output history, `output_text`, function-call output items, JSON Schema structured outputs via `text.format`, and Responses-style streaming events such as `response.output_text.delta`, `response.function_call_arguments.delta`, and `response.completed`. Embeddings, batches, and the model catalog are covered below. Uploads, fine-tuning, audio/image generation, vector stores, and other SDK resources require Agent RT native APIs or the upstream SDK because Agent RT does not expose equivalent provider-neutral contracts.

### OpenAI Responses migration

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.responses.create(
    model="your-model",
    instructions="Be concise.",
    input="Summarize this request.",
)
print(response.output_text)

stream = await client.responses.create(
    model="your-model",
    input="Stream this answer.",
    stream=True,
)
async for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

## Anthropic Python SDK

For messages, switch the import to `agent_rt.anthropic`:

```python
import os

from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
response = await client.messages.create(
    model=os.environ["ANTHROPIC_MODEL"],
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

`Anthropic` and `AsyncAnthropic` support `messages.create()` in normal and streaming modes, including system text, tools, JSON Schema structured outputs through the current `output_config.format` shape (and the deprecated `output_format` migration shape), temperature, token limits, usage, text deltas, tool-input deltas, and `tool_use` / `tool_result` conversation history. `messages.stream()` is also available as a sync or async context manager with `text_stream` and `get_final_message()`. Responses are `agent_rt.anthropic.types.Message` objects whose `stop_reason` uses Anthropic's values (`end_turn`, `tool_use`, `max_tokens`, `refusal`). Response content blocks can be sent back as conversation history unchanged. `agent_rt.anthropic.types` provides the common `Message`, `TextBlock`, `ToolUseBlock`, `Usage`, and `*Param` request types. As with the SDKs, blank `ANTHROPIC_*` / `OPENAI_*` environment values count as unset in the compatibility clients. Token counting, batches, and the model catalog are covered below. Files and other Anthropic SDK resources require native Agent RT support or the upstream SDK. The built-in Anthropic provider uses the optional Anthropic SDK, so install `agent-rt[anthropic]` when not injecting a custom Agent RT provider.

### Anthropic streaming migration

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
async with client.messages.stream(
    model="your-model",
    max_tokens=512,
    messages=[{"role": "user", "content": "Stream this answer."}],
) as stream:
    async for text in stream.text_stream:
        print(text, end="")
    final = await stream.get_final_message()
```

## OpenAI and Anthropic SDK topics

The sections below cover capabilities shared by the OpenAI and Anthropic compatibility clients, with Python and TypeScript/JavaScript examples.

### OpenAI client options, types, and transports

As in the SDKs, a client built without `base_url` / `baseURL` or an API key reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` (and `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`) when it is constructed; blank values count as unset. Chat Completions return `tool_calls` and accept `stream=True` / `stream: true` (an iterable of `chat.completion.chunk` objects), and JSON mode (`response_format={"type": "json_object"}`) maps to `JSON_OBJECT_OUTPUT`, which OpenAI-compatible providers send as JSON mode again.

In Python, `agent_rt.openai.types` (including `types.chat`, `types.chat.chat_completion`, `types.chat.chat_completion_chunk`, `types.completion_usage`, `types.create_embedding_response`, and `types.responses`) exports the SDK class names; shim responses are instances of them, and test doubles can build them with keyword arguments. `agent_rt.openai.resources` exposes the resource classes the clients use (`resources.responses.Responses`, `resources.chat.completions.Completions`, and their `Async` variants), so patching one reaches every client. `responses.parse(text_format=Model)` and `chat.completions.parse(response_format=Model)` send a strict JSON schema and return `output_parsed` / `message.parsed`. Clients expose `timeout`, `max_retries`, `close()` / `is_closed()`, and context-manager use. `agent_rt.openai._base_client.SyncAPIClient` / `AsyncAPIClient` exist so SDK-internal patch targets resolve; the clients subclass them, but their `request()` raises `NotImplementedError` because shim traffic goes through Agent RT providers. The Responses API (`responses.create()` / `parse()`) also runs through the provider's Chat Completions transport, so an injected `http_client` that only serves `/v1/responses` is not supported. `OpenAI(http_client=httpx.Client(...))` (or an `httpx.AsyncClient`) sends every request through that client, so offline transports such as `httpx.MockTransport` keep working.

### SDK errors

Both compatibility modules export each SDK's error hierarchy under the upstream names: `APIError`, `APIStatusError` (Python), `APIConnectionError`, `APITimeoutError` / `APIConnectionTimeoutError`, `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError`, `UnprocessableEntityError`, `RateLimitError`, and `InternalServerError`, plus the `OpenAIError` / `AnthropicError` base. The constructors take the SDK signatures, so tests that build errors (`anthropic.RateLimitError("slow down", response=response, body=None)` in Python, `new Anthropic.APIError(status, error, message, headers)` in TypeScript) keep working. In TypeScript the classes are also static members (`OpenAI.APIError`, `Anthropic.RateLimitError`). OpenAI and Anthropic each get their own classes, as in the SDKs. Provider failures from `create()` that carry an HTTP status are re-raised as the matching class (an HTTP 429 that Agent RT's rate-limit gate wraps is still raised as `RateLimitError`), with the response headers and JSON error body (`error.code` in TypeScript, `error.body` in Python), and connection and timeout failures as `APIConnectionError` / timeout errors. Other exceptions pass through unchanged.

In TypeScript, `agent-rt/openai`, `agent-rt/anthropic`, and `agent-rt/@anthropic-ai/sdk` provide an ES module entry, so `import Anthropic from "agent-rt/@anthropic-ai/sdk"` yields the client class in `"type": "module"` projects. `Anthropic` also carries the SDK type names (`Anthropic.MessageParam`, `Anthropic.Messages.Message`, `Anthropic.TextBlockParam`, ...). Its responses include `tool_use` blocks, `tool_use` / `tool_result` history blocks are accepted, and `messages.parse()` / `beta.messages.parse()` return `parsed_output` (using the format's `parse()` when present). `messages.stream()` returns a `MessageStream` (async iteration over raw stream events, `.on()` / `.once()` / `.off()` for `streamEvent`, `text`, `contentBlock`, `message`, `finalMessage`, `error`, and `end`, plus `finalMessage()`, `finalText()`, `done()`, and `abort()`), and `messages.create({ stream: true })` returns the raw event stream. `messages.stream()` sends a streaming request through the provider when no local migration tool (MCP, web search, sandboxed code) is involved; otherwise, and for `create({ stream: true })`, the completed message runs through the bounded tool loop. Either way the listeners and iteration see events derived from the completed message, so each text block arrives as one delta. The TypeScript client constructors also accept the SDKs' standard options (`timeout`, `maxRetries`, `defaultHeaders`, `fetch`, and similar) so migrated code type-checks. An injected `fetch` carries every request, as in the SDKs (offline test transports keep working; the OpenAI WebSocket transport is disabled then); retries and timeouts still come from the Agent RT provider.

### Structured output migration

OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages all map their JSON Schema controls to Agent RT's provider-neutral `StructuredOutputRequirement`. Pydantic-style model classes are accepted without a hard Pydantic dependency when they expose `model_json_schema()` (or the older `schema()` method).

```python
schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# `openai_client` / `anthropic_client` are AsyncOpenAI / AsyncAnthropic instances.
openai_response = await openai_client.responses.create(
    model="your-model",
    input="Return an answer object.",
    text={
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": schema,
            "strict": True,
        }
    },
)

anthropic_response = await anthropic_client.messages.create(
    model="your-model",
    max_tokens=512,
    messages=[{"role": "user", "content": "Return an answer object."}],
    output_config={
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    },
)
```

### TypeScript / JavaScript structured output migration

The npm package publishes `agent-rt/openai` and `agent-rt/anthropic` compatibility entry points for low-friction SDK migration. Existing JavaScript/TypeScript call sites can keep the familiar structured-output request shapes while routing execution through Agent RT `ModelProvider` contracts.

OpenAI Chat Completions JSON Schema migration:

```ts
// Before: import OpenAI from "openai";
import OpenAI from "agent-rt/openai";

const client = new OpenAI({ defaultModel: "your-model" });
const response = await client.chat.completions.create({
  model: "your-model",
  messages: [{ role: "user", content: "Return an answer object." }],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "answer",
      schema,
      strict: true,
    },
  },
});
```

OpenAI Responses structured output uses the current `text.format` shape:

```ts
const response = await client.responses.create({
  model: "your-model",
  input: "Return an answer object.",
  text: {
    format: {
      type: "json_schema",
      name: "answer",
      schema,
      strict: true,
    },
  },
});
```

Anthropic migration keeps the SDK-style `messages.create()` call and accepts both current `output_config.format` and the older `output_format` migration shape:

```ts
// Before: import Anthropic from "@anthropic-ai/sdk";
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic({ defaultModel: "your-model" });
const response = await client.messages.create({
  model: "your-model",
  max_tokens: 512,
  messages: [{ role: "user", content: "Return an answer object." }],
  output_config: {
    format: {
      type: "json_schema",
      schema,
    },
  },
});
```

The compatibility clients may also receive an explicit Agent RT provider with `new OpenAI({ provider })` or `new Anthropic({ provider })`, which is useful for custom providers and deterministic tests. All three vendor request shapes normalize to the existing TypeScript `StructuredOutputRequirement`. The native `AnthropicModelProvider` forwards that requirement as `output_config.format`; the existing `OpenAIModelProvider` continues to emit JSON Schema through `response_format`.

### Embeddings and token counting

OpenAI embedding migrations can use the familiar `client.embeddings.create()` surface. Agent RT routes the request through the provider-neutral `EmbeddingRequest` / `EmbeddingResponse` contract and supports both the built-in OpenAI-compatible HTTP transport and OpenAI SDK clients.

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.embeddings.create(
    model="text-embedding-model",
    input=["first document", "second document"],
    dimensions=1024,
)
vector = response.data[0].embedding
```

Anthropic token counting is exposed through the familiar `messages.count_tokens()` method and routes through Agent RT's `TokenCountingModelProvider` contract. `agent_rt.openai` does not wrap any OpenAI token-counting endpoint; token usage remains available on OpenAI generation and embedding responses.

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
count = await client.messages.count_tokens(
    model="your-model",
    system="Be concise.",
    messages=[{"role": "user", "content": "How many tokens is this?"}],
)
print(count.input_tokens)
```

Anthropic does not provide a first-party embeddings endpoint, so `agent_rt.anthropic` intentionally does not expose `embeddings.create()`. Use an embedding-capable provider through Agent RT's embedding contract when semantic vectors are needed.

The TypeScript/JavaScript migration entry points expose the equivalent SDK-style surfaces. `agent-rt/openai` provides `client.embeddings.create()` and normalizes requests through the async `EmbeddingModelProvider` / `EmbeddingRequest` / `EmbeddingResponse` contracts. The native OpenAI provider supports both SDK clients and the dependency-light OpenAI-compatible `POST /v1/embeddings` transport.

```ts
import OpenAI from "agent-rt/openai";

const client = new OpenAI({ defaultModel: "text-embedding-model" });
const response = await client.embeddings.create({
  model: "text-embedding-model",
  input: ["first document", "second document"],
  dimensions: 1024,
  encoding_format: "float",
});
const vector = response.data[0].embedding;
```

`agent-rt/anthropic` follows the official JavaScript SDK naming and exposes `client.messages.countTokens()`. It routes through the provider-neutral `TokenCountingModelProvider` contract and returns `{ input_tokens }`. Generation-only settings such as `max_tokens`, `temperature`, and structured-output configuration are omitted from the native Anthropic count request.

```ts
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic({ defaultModel: "your-model" });
const count = await client.messages.countTokens({
  model: "your-model",
  system: "Be concise.",
  messages: [{ role: "user", content: "How many tokens is this?" }],
});
console.log(count.input_tokens);
```

As in Python, the JavaScript compatibility layer does not fabricate an Anthropic embeddings endpoint or wrap an OpenAI token-count endpoint. The synchronous `EmbeddingProvider` used by `SemanticMemory` remains separate from the async model-provider embedding contract.

### Code execution migration

OpenAI `code_interpreter` / hosted `shell` tool declarations and Anthropic `code_execution_*` tool declarations can be migrated onto Agent RT's existing sandbox runtime. Code execution is deliberately opt-in: pass an explicit `SandboxSession` to the compatibility client. The adapter translates the hosted-tool declaration into an Agent RT tool, executes model-requested code through `SandboxSession.run_code()` (or shell commands through `SandboxSession.execute()`), appends the sandbox result as a tool result, and continues the model turn until a final response is produced. Persistent interpreter state, workspace isolation, timeouts, output limits, environment filtering, and network policy remain controlled by the supplied sandbox session.

```python
from agent_rt import (
    CallbackSandboxBackend,
    CodeInterpreterRegistry,
    SandboxCommandResult,
    SandboxSession,
)
from agent_rt.openai import AsyncOpenAI

interpreters = CodeInterpreterRegistry()

async def python_runner(
    session_id, runtime, code, state, workspace, environment, limits, network_policy
):
    # Replace this example runner with the project's configured sandbox runtime.
    return SandboxCommandResult(exit_code=0, stdout=b"sandbox result")

async def command_runner(
    session_id, command, workspace, limits, environment, network_policy
):
    # Replace this example runner with the project's configured sandbox runtime.
    return SandboxCommandResult(exit_code=0, stdout=b"sandbox result")

interpreters.register("python", python_runner)
sandbox = SandboxSession(
    "migration-session",
    CallbackSandboxBackend(command_runner),
    interpreters=interpreters,
)
client = AsyncOpenAI(sandbox_session=sandbox)

response = await client.responses.create(
    model="your-model",
    input="Use Python to analyze the data.",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
)
```

Anthropic uses the same Agent RT sandbox path:

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic(sandbox_session=sandbox)
response = await client.messages.create(
    model="your-model",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Use code to calculate the answer."}],
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
)
```

If a hosted code-execution tool is requested without `sandbox_session`, the compatibility layer fails before invoking the model. This prevents migration convenience from silently bypassing Agent RT's sandbox policy. Streaming `responses.create(..., stream=True)`, Anthropic `messages.create(..., stream=True)`, and Anthropic `messages.stream()` are supported; local code/tool rounds are resolved inside the sandbox before the final vendor-shaped output stream is emitted.

The TypeScript/JavaScript migration entry points use the same sandbox policy through the camelCase `sandboxSession` option. OpenAI Responses `code_interpreter` and hosted `shell` declarations are translated into Agent RT tools, executed through `SandboxSession.runCode()` or `SandboxSession.execute()`, appended to provider-neutral message history, and followed by an automatic model continuation. Completed OpenAI Responses include `code_interpreter_call` or `shell_call` output items before the final assistant message.

```ts
import OpenAI from "agent-rt/openai";
import { CodeInterpreterRegistry, SandboxSession } from "agent-rt";

const interpreters = new CodeInterpreterRegistry();
interpreters.register("python", async (
  sessionId, runtime, code, state, workspace, environment, limits, networkPolicy
) => {
  // Delegate to the project's configured sandbox runtime.
  return { exitCode: 0, stdout: new TextEncoder().encode("sandbox result") };
});

const sandbox = new SandboxSession("migration-session", sandboxBackend, { interpreters });
const client = new OpenAI({ sandboxSession: sandbox });

const response = await client.responses.create({
  model: "your-model",
  input: "Use Python to analyze the data.",
  tools: [{ type: "code_interpreter", container: { type: "auto" } }],
});
```

Anthropic's JavaScript migration surface accepts versioned `code_execution_*` declarations and returns `server_tool_use` plus `code_execution_tool_result` content blocks before the final text:

```ts
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic({ sandboxSession: sandbox });
const response = await client.messages.create({
  model: "your-model",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Use code to calculate the answer." }],
  tools: [{ type: "code_execution_20250825", name: "code_execution" }],
});
```

`maxCodeToolRounds` bounds automatic local execution/continuation rounds and defaults to `8`. As in Python, declaring a hosted code tool without a sandbox fails before provider invocation. The JavaScript compatibility layer does not execute arbitrary ordinary function tools implicitly; only the translated Agent RT sandbox tools are eligible for automatic local execution.

### Remote MCP migration

OpenAI remote-MCP tool declarations and Anthropic MCP Connector declarations can be migrated onto Agent RT's transport-neutral MCP clients. The compatibility layer does not open vendor-declared URLs directly. Instead, configure and inject Agent RT `MCPClient` or `AuthorizedMCPClient` instances through `mcp_clients={...}`. This keeps transport, credentials, capability filtering, authorization, and policy enforcement inside Agent RT.

For OpenAI Responses, the `server_label` selects the injected client. `allowed_tools` further narrows the tools discovered from that client; an `AuthorizedMCPClient` can impose stricter policy filtering. Vendor `server_url`, `authorization`, `headers`, and `require_approval` fields are accepted as migration-shaped metadata but do not bypass the injected Agent RT MCP transport or policy.

```python
from agent_rt import AuthorizedMCPClient, MCPAccessPolicy, MCPClient
from agent_rt.openai import AsyncOpenAI

mcp = MCPClient(your_mcp_transport, requested_capabilities=("tools",))
# Optional: wrap with AuthorizedMCPClient(mcp, policy) for Agent RT policy enforcement.
client = AsyncOpenAI(mcp_clients={"docs": mcp})

response = await client.responses.create(
    model="your-model",
    input="Search the documentation.",
    tools=[
        {
            "type": "mcp",
            "server_label": "docs",
            "server_url": "https://example.com/mcp",
            "allowed_tools": ["search"],
            "require_approval": "never",
        }
    ],
)
```

The adapter initializes the Agent RT MCP client if necessary, discovers tools, exposes allowed tools to the model as namespaced function definitions, calls the selected MCP tool through `call_tool()`, appends its result to model history, and continues generation. OpenAI Responses include vendor-shaped `mcp_call` output items for executed MCP calls.

The TypeScript/JavaScript migration clients expose the same design through the camelCase `mcpClients` option. The declared vendor URL/auth fields remain migration metadata only; Agent RT never opens those URLs or forwards those credentials. The injected `MCPClient` / `AuthorizedMCPClient` owns transport, authentication, authorization, capability filtering, and policy enforcement.

```ts
import OpenAI from "agent-rt/openai";
import { MCPClient } from "agent-rt";

const mcp = new MCPClient(yourTransport, "agent-rt", "1", ["tools"]);
const client = new OpenAI({ mcpClients: { docs: mcp } });

const response = await client.responses.create({
  model: "your-model",
  input: "Search the documentation.",
  tools: [
    {
      type: "mcp",
      server_label: "docs",
      server_url: "https://vendor-declared.example/mcp",
      allowed_tools: ["search"],
      require_approval: "never",
    },
  ],
});
```

The JS adapter initializes the injected MCP client when needed, calls `listTools()`, applies OpenAI `allowed_tools`, exposes namespaced Agent RT tool definitions (`mcp__<server>__<tool>`), invokes only bound MCP tools through `callTool()`, appends the result to provider-neutral message history, and continues the model turn. Completed OpenAI Responses include `mcp_call` output items with the server label, original MCP tool name, arguments, and serialized result.

Anthropic's current MCP Connector shape is supported through `client.beta.messages`: `mcp_servers` defines the server names and `mcp_toolset` entries select/configure them. Each declared server must be referenced by exactly one toolset, matching Anthropic's validation rule. `default_config.enabled` and per-tool `configs` are honored when deciding which discovered MCP tools are exposed.

The TypeScript compatibility client also exposes `client.beta.messages.create()` for this migration surface:

```ts
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic({ mcpClients: { docs: mcp } });
const response = await client.beta.messages.create({
  model: "your-model",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Search the docs." }],
  mcp_servers: [
    {
      type: "url",
      name: "docs",
      url: "https://vendor-declared.example/mcp",
    },
  ],
  tools: [
    {
      type: "mcp_toolset",
      mcp_server_name: "docs",
      default_config: { enabled: true },
    },
  ],
});
```

Anthropic `mcp_servers` are validated for unique names and exactly one matching `mcp_toolset`. `default_config.enabled` and per-tool `configs` control discovery visibility, while an injected `AuthorizedMCPClient` may further restrict discovery/invocation. Returned compatibility content contains `mcp_tool_use` and `mcp_tool_result` blocks before final assistant text. Vendor-declared URL/authentication fields are never used to bypass the injected Agent RT MCP transport.

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic(mcp_clients={"docs": mcp})
response = await client.beta.messages.create(
    model="your-model",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Search the documentation."}],
    mcp_servers=[
        {
            "type": "url",
            "url": "https://example.com/mcp",
            "name": "docs",
        }
    ],
    tools=[
        {
            "type": "mcp_toolset",
            "mcp_server_name": "docs",
            "default_config": {"enabled": False},
            "configs": {"search": {"enabled": True}},
        }
    ],
    betas=["mcp-client-2026-09-15"],
)
```

Anthropic non-streaming responses include `mcp_tool_use` and `mcp_tool_result` blocks for executed MCP calls. `messages.stream()` and `messages.create(..., stream=True)` also execute MCP tool rounds through Agent RT before emitting the final vendor-shaped response stream. If a declaration references a server that is not present in `mcp_clients`, the compatibility layer fails before the model call.

### Models API migration

Both compatibility clients expose the common model catalog operations used by the upstream SDKs: `models.list()` and `models.retrieve(...)`. Agent RT normalizes provider model metadata into `ModelCatalogEntry` internally and then returns vendor-shaped objects from the compatibility clients.

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
models = await client.models.list()
model = await client.models.retrieve(models.data[0].id)
print(model.id)
```

OpenAI-compatible transports use `GET /v1/models` and `GET /v1/models/{model}` directly, so this works without installing the OpenAI SDK when the configured compatible server supports those endpoints. OpenAI-shaped model objects expose `id`, `object`, `created`, and `owned_by`.

Anthropic compatibility provides the equivalent `models.list()` and `models.retrieve(model_id)` methods:

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
models = await client.models.list()
model = await client.models.retrieve(models.data[0].id)
print(model.display_name)
```

Anthropic-shaped model objects expose `id`, `type`, `display_name`, and `created_at`; list responses include `data`, `has_more`, `first_id`, and `last_id`. The compatibility layer materializes the provider result into one page rather than reproducing every upstream auto-pagination helper.

The TypeScript/JavaScript migration entry points expose the same catalog surface through `agent-rt/openai` and `agent-rt/anthropic`:

```ts
import OpenAI from "agent-rt/openai";
import Anthropic from "agent-rt/anthropic";

const openai = new OpenAI();
const openAIModels = await openai.models.list();
const openAIModel = await openai.models.retrieve(openAIModels.data[0].id);

const anthropic = new Anthropic();
const anthropicModels = await anthropic.models.list();
const anthropicModel = await anthropic.models.retrieve(anthropicModels.data[0].id);
```

The TypeScript core defines provider-neutral `ModelCatalogEntry` and `ModelCatalogProvider` contracts. Native OpenAI and Anthropic providers implement `listModels()` / `retrieveModel()`. The dependency-light OpenAI-compatible transport uses `GET /v1/models` and `GET /v1/models/{model}` directly. OpenAI compatibility returns `{ object: "list", data }` and OpenAI-shaped model objects; Anthropic compatibility returns a materialized page with `data`, `has_more: false`, `first_id`, and `last_id`.

### Stateful response continuation

OpenAI Responses compatibility supports `previous_response_id` for client-local continuation. Every completed compatibility response receives a stable `resp_agent_rt_...` identifier. Agent RT stores the native provider-neutral message history for a bounded number of completed responses on that client; passing a prior response ID prepends that history before the new input. This preserves native assistant messages and tool-result rounds rather than reconstructing context from rendered text.

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI(max_response_states=128)
first = await client.responses.create(
    model="your-model",
    input="Remember that the project codename is Atlas.",
)
second = await client.responses.create(
    model="your-model",
    input="What is the codename?",
    previous_response_id=first.id,
)
```

The continuation cache is intentionally scoped to the compatibility client instead of being a hidden global store. `max_response_states` bounds retained response histories (default `128`); oldest entries are evicted first. Unknown or evicted IDs raise `ValueError` before invoking the model. Streaming responses commit continuation state only when a `response.completed` event is observed, so an unconsumed or interrupted stream does not become a valid continuation point. Sync `OpenAI()` clients use the same behavior.

This is a migration compatibility feature, not a durable conversation database. Applications that need persistence across process restarts should use Agent RT session memory/checkpointing and reconstruct the requested history explicitly.

Anthropic Messages does not use an OpenAI-style response ID continuation primitive. Its conversation state remains client-managed by supplying prior assistant/user/tool blocks in `messages`; Agent RT already preserves those message-history shapes, including `tool_use`, `tool_result`, MCP, and ordinary assistant content.

The TypeScript/JavaScript `agent-rt/openai` migration client provides the same client-local `previous_response_id` continuation behavior for its non-streaming Responses path:

```ts
import OpenAI from "agent-rt/openai";

const client = new OpenAI({ maxResponseStates: 128 });
const first = await client.responses.create({
  model: "your-model",
  input: "Remember that the project codename is Atlas.",
});
const second = await client.responses.create({
  model: "your-model",
  previous_response_id: first.id,
  input: "What is the codename?",
});
```

Each completed JS compatibility response receives a stable client-local `resp_agent_rt_<n>` identifier. The continuation store keeps provider-neutral `ModelMessage` history rather than rendered response text, so ordinary assistant messages, sandbox code/tool calls, MCP calls, and their tool-result messages remain intact when the next request is assembled. Unknown or evicted IDs are rejected before provider invocation. `maxResponseStates` defaults to `128`, must be a positive integer, and evicts the oldest retained response when the bound is exceeded. The store is scoped to the compatibility client and is not durable across process restarts.

### File search migration

OpenAI Responses `file_search` tool declarations can be migrated onto Agent RT's provider-neutral retrieval layer. Pass an explicit `RetrievalRegistry` to the compatibility client. Each OpenAI `vector_store_id` is interpreted as the name of a provider registered in that Agent RT registry; the compatibility layer does not create or manage OpenAI vector-store resources.

```python
from agent_rt import RetrievalRegistry
from agent_rt.openai import AsyncOpenAI

retrieval = RetrievalRegistry()
retrieval.register("vs_docs", your_file_retrieval_provider)
retrieval.register("vs_release", your_release_retrieval_provider)

client = AsyncOpenAI(retrieval_registry=retrieval)
response = await client.responses.create(
    model="your-model",
    input="Find the deployment guide.",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["vs_docs", "vs_release"],
            "max_num_results": 5,
            "filters": {"team": "platform"},
            "ranking_options": {"score_threshold": 0.5},
        }
    ],
)
```

The adapter translates the hosted tool into an Agent RT model tool. When the model requests a search, Agent RT issues `RetrievalQuery` calls against every named provider, forwards the declared filters and per-provider result limit, merges results by score, applies `score_threshold`, truncates to `max_num_results`, appends the merged result payload as a tool result, and continues generation. Returned OpenAI Responses include a `file_search_call` output item with the executed query and normalized result entries (`file_id`, `filename`, `score`, `text`, and `attributes`).

Streaming Responses use the same local retrieval round-trip and include the completed `file_search_call` in the final `response.completed` object. File-search tool rounds are also preserved in the compatibility client's `previous_response_id` history. If `file_search` is declared without `retrieval_registry`, the call fails before invoking the model.

Agent RT intentionally leaves indexing, persistence, embeddings, hybrid search, and access control to the registered `RetrievalProvider`. This avoids coupling application migration to OpenAI's vector-store service while preserving the familiar Responses tool declaration. Anthropic does not expose the same hosted `file_search` tool shape, so no synthetic `agent_rt.anthropic.file_search` API is added.

The TypeScript/JavaScript `agent-rt/openai` migration client supports the same mapping through the camelCase `retrievalRegistry` option:

```ts
import OpenAI from "agent-rt/openai";
import { RetrievalRegistry } from "agent-rt";

const retrieval = new RetrievalRegistry();
retrieval.register("vs_docs", yourFileRetrievalProvider);
retrieval.register("vs_release", yourReleaseRetrievalProvider);

const client = new OpenAI({ retrievalRegistry: retrieval });
const response = await client.responses.create({
  model: "your-model",
  input: "Find the deployment guide.",
  tools: [
    {
      type: "file_search",
      vector_store_ids: ["vs_docs", "vs_release"],
      max_num_results: 5,
      filters: { team: "platform" },
      ranking_options: { score_threshold: 0.5 },
    },
  ],
});
```

Each `vector_store_id` is the name of an Agent RT `RetrievalProvider` registered in that `RetrievalRegistry`; the JavaScript migration layer does not create OpenAI vector stores or hidden indexes. It issues one `RetrievalQuery` per named provider, forwards `filters` and the result limit, removes scored results below `score_threshold`, merges results by descending score, truncates to `max_num_results`, appends the normalized result payload as a tool result, and continues the model call. Completed Responses include a `file_search_call` item containing `queries` and normalized `file_id`, `filename`, `score`, `text`, and `attributes` result fields.

File-search assistant/tool-result history is retained natively by the JavaScript `previous_response_id` store, so continuation does not reconstruct retrieval context from rendered text. Declaring `file_search` without `retrievalRegistry` fails before provider invocation. The JavaScript compatibility layer intentionally leaves indexing, persistence, embeddings, hybrid ranking, and retrieval authorization to the registered providers and does not add a synthetic Anthropic file-search surface.

### Web search migration

OpenAI `web_search` / `web_search_preview*` tools and Anthropic `web_search_*` server tools can be migrated onto Agent RT's provider-neutral web retrieval path. By default the compatibility adapters resolve the web-search backend lazily from environment variables only when a web-search tool is declared. Ordinary model calls do not require web-search configuration.

Select the backend with:

```text
AGENT_RT_WEB_SEARCH_TOOL=tavily|brave|serper
```

Set the selected backend credential with its conventional environment variable:

```text
TAVILY_API_KEY=...
BRAVE_SEARCH_API_KEY=...
SERPER_API_KEY=...
```

The generic fallbacks `AGENT_RT_WEB_SEARCH_API_KEY` and `AGENT_RT_WEB_SEARCH_TOKEN` are also accepted. `AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS` controls the HTTP timeout and defaults to `20`. Missing/unsupported tool selection or missing credentials fails before the model performs a web-search round trip. The provider-specific credential takes precedence over `AGENT_RT_WEB_SEARCH_API_KEY`, which takes precedence over `AGENT_RT_WEB_SEARCH_TOKEN`. Missing provider selection, unsupported provider names, missing credentials, invalid timeouts, HTTP failures, and malformed provider responses fail closed in both runtimes. Credentials remain encapsulated inside `EnvironmentWebSearchProvider`; they are never added to model-visible tool schemas, tool arguments, response objects, or continuation history.

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.responses.create(
    model="your-model",
    input="Find the latest Agent RT documentation.",
    tools=[{"type": "web_search"}],
)
```

The adapter exposes one internal Agent RT web-search function to the model, executes the selected retrieval provider, appends normalized results as the tool result, and continues generation. OpenAI Responses include a completed `web_search_call` item with the query, URL sources, and normalized search results. OpenAI `search_context_size` plus domain/location fields are carried into the provider-neutral retrieval filters where supplied.

Anthropic uses the same Agent RT provider path:

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
response = await client.messages.create(
    model="your-model",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Search for current documentation."}],
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
        }
    ],
)
```

Anthropic non-streaming compatibility responses include `server_tool_use` and `web_search_tool_result` blocks followed by the final assistant content. `allowed_domains`, `blocked_domains`, and user-location metadata are preserved as provider-neutral retrieval filters. An application may bypass environment selection by explicitly injecting any `RetrievalProvider(kind="web")` as `web_search_provider=...` on either compatibility client.

The built-in environment provider supports Tavily, Brave Search, and Serper over HTTP. It normalizes each backend into Agent RT `RetrievalResult` values so application code and compatibility adapters do not depend on provider-specific response shapes.

TypeScript/JavaScript uses the same environment configuration and the camelCase `webSearchProvider` option. The migration clients translate the declarations into an Agent RT retrieval tool named `agent_rt_web_search`.

OpenAI Responses migration:

```ts
import OpenAI from "agent-rt/openai";

const client = new OpenAI();
const response = await client.responses.create({
  model: "your-model",
  input: "Find the latest release notes.",
  tools: [{
    type: "web_search",
    search_context_size: "medium",
  }],
});
```

The model sees a provider-neutral `agent_rt_web_search` function. Agent RT executes the selected Tavily, Brave Search, or Serper provider, appends the normalized search result as tool history, and automatically continues the model call. OpenAI Responses expose a completed `web_search_call` with the search query, URL sources, normalized result title/text/URL/score fields, and final assistant output. `allowed_domains`, `blocked_domains`, `user_location`, `search_context_size`, explicit `filters`, and result limits are forwarded into the retrieval query.

Anthropic migration uses the same Agent RT provider while preserving Anthropic's server-tool-shaped response blocks:

```ts
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic();
const response = await client.messages.create({
  model: "your-model",
  max_tokens: 512,
  messages: [{ role: "user", content: "Search the web." }],
  tools: [{
    type: "web_search_20250305",
    name: "web_search",
    max_uses: 3,
    allowed_domains: ["example.com"],
  }],
});
```

Returned compatibility content contains `server_tool_use` followed by `web_search_tool_result` blocks before final text. Both clients also accept `webSearchProvider` with any Agent RT `RetrievalProvider` of the application's choosing; when supplied, that explicit provider is used instead of environment selection. The built-in `EnvironmentWebSearchProvider` uses Node's native `fetch`, so this feature adds no required web-search SDK dependency.

### Batch processing migration

Agent RT compatibility supports the common batch lifecycle for both OpenAI and Anthropic. The core providers expose a small provider-neutral `BatchJob` (`id`, `status`, `raw`) while compatibility clients preserve upstream SDK objects when available.

OpenAI exposes `client.batches.create()`, `retrieve()`, `list()`, and `cancel()` in both sync and async clients:

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
batch = await client.batches.create(
    input_file_id="file_...",
    endpoint="/v1/responses",
    completion_window="24h",
    metadata={"job": "nightly"},
)
status = await client.batches.retrieve(batch.id)
page = await client.batches.list(limit=20)
await client.batches.cancel(batch.id)
```

OpenAI batch creation remains file-ID based, matching the upstream Batch API. Agent RT does not synthesize an upload/file store here: callers must provide an existing `input_file_id` until Files/Uploads compatibility is implemented separately. The lightweight OpenAI-compatible transport supports `POST /v1/batches`, `GET /v1/batches/{id}`, `GET /v1/batches`, and `POST /v1/batches/{id}/cancel`, including `after`/`limit` list parameters.

Anthropic exposes the upstream `client.messages.batches` surface with inline requests:

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
batch = await client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "your-model",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Summarize this."}],
            },
        }
    ]
)
status = await client.messages.batches.retrieve(batch.id)
page = await client.messages.batches.list(limit=20)
results = await client.messages.batches.results(batch.id)
async for item in results:
    print(item.custom_id, item.result)
```

Anthropic compatibility supports `create`, `retrieve`, `list`, `cancel`, and `results`. Async `results()` returns an async iterable; sync clients return a normal iterable. Pagination helpers are intentionally materialized into one compatibility page rather than reproducing every upstream auto-pagination helper.

Custom Agent RT providers may implement the optional `create_batch`, `retrieve_batch`, `list_batches`, `cancel_batch`, and (for result-bearing providers) `batch_results` methods and return `BatchJob` values.

The TypeScript/JavaScript package exposes the equivalent lifecycle through `agent-rt/openai` and `agent-rt/anthropic`. The TypeScript core uses provider-neutral `BatchJob` and `BatchProvider` contracts (`createBatch`, `retrieveBatch`, `listBatches`, `cancelBatch`, and optional `batchResults`).

OpenAI migration:

```ts
import OpenAI from "agent-rt/openai";

const client = new OpenAI();
const batch = await client.batches.create({
  input_file_id: "file_...",
  endpoint: "/v1/responses",
  completion_window: "24h",
  metadata: { job: "nightly" },
});
const status = await client.batches.retrieve(batch.id);
const page = await client.batches.list({ limit: 20 });
await client.batches.cancel(batch.id);
```

OpenAI creation remains intentionally file-ID based. The JavaScript compatibility layer does not fabricate file uploads or storage; callers must supply an existing `input_file_id`. The dependency-light compatible transport implements `POST /v1/batches`, `GET /v1/batches/{id}`, `GET /v1/batches`, and `POST /v1/batches/{id}/cancel`, including `after` and `limit` list parameters.

Anthropic Message Batches migration:

```ts
import Anthropic from "agent-rt/anthropic";

const client = new Anthropic();
const batch = await client.messages.batches.create({
  requests: [
    {
      custom_id: "request-1",
      params: {
        model: "your-model",
        max_tokens: 256,
        messages: [{ role: "user", content: "Summarize this." }],
      },
    },
  ],
});
const status = await client.messages.batches.retrieve(batch.id);
const page = await client.messages.batches.list({ limit: 20 });
const results = await client.messages.batches.results(batch.id);
for await (const item of results) {
  console.log(item.custom_id, item.result);
}
```

`messages.batches` supports `create`, `retrieve`, `list`, `cancel`, and `results`. `results()` always returns an async iterable on the migration surface; provider SDK async iterables pass through, while ordinary iterables are wrapped. List responses are materialized into one page (`has_more: false`) rather than reproducing every SDK pagination helper. When native SDK objects are available, the migration layer preserves their raw vendor-shaped batch objects.

### Reasoning and thinking controls

Python and TypeScript OpenAI/Anthropic migration clients normalize vendor reasoning controls into the provider-neutral `ReasoningConfig` on `ModelRequest`. This keeps reasoning configuration available to custom Agent RT providers while letting the built-in providers emit the vendor-specific request shape.

OpenAI Chat Completions accepts `reasoning_effort`:

```python
from agent_rt.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Solve this carefully."}],
    reasoning_effort="high",
)
```

OpenAI Responses accepts the `reasoning` object. Agent RT preserves both `effort` and `summary` in `ReasoningConfig`; the native Responses WebSocket transport forwards both fields. When the native OpenAI provider is using its HTTP/SSE transport (`OPENAI_WEBSOCKET=0` / `websocket=False`), `effort` is forwarded as `reasoning_effort` while `summary` remains available only to provider implementations that support a Responses-style request.

```python
response = await client.responses.create(
    model="your-model",
    input="Solve this carefully.",
    reasoning={"effort": "high", "summary": "auto"},
)
```

Anthropic Messages accepts current adaptive thinking plus effort through `output_config`, and also preserves the legacy extended-thinking budget form for older compatible models:

```python
from agent_rt.anthropic import AsyncAnthropic

client = AsyncAnthropic()
response = await client.messages.create(
    model="your-model",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Solve this carefully."}],
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
)

legacy = await client.messages.create(
    model="older-model",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Solve this carefully."}],
    thinking={"type": "enabled", "budget_tokens": 2048},
)
```

The native `AnthropicModelProvider` maps `ReasoningConfig.thinking` to `thinking`, maps `ReasoningConfig.effort` to `output_config.effort`, and merges effort with `output_config.format` when structured output is also requested. `budget_tokens` is emitted only with `thinking="enabled"`; Agent RT validates that combination but intentionally leaves model-specific availability to the upstream provider, since Anthropic support differs by model generation.

TypeScript/JavaScript uses the same migration shapes:

```ts
import OpenAI from "agent-rt/openai";
import Anthropic from "agent-rt/anthropic";

const openai = new OpenAI();
await openai.chat.completions.create({
  model: "your-model",
  messages: [{ role: "user", content: "Solve this carefully." }],
  reasoning_effort: "high",
});

await openai.responses.create({
  model: "your-model",
  input: "Solve this carefully.",
  reasoning: { effort: "high", summary: "auto" },
});

const anthropic = new Anthropic();
await anthropic.messages.create({
  model: "your-model",
  max_tokens: 4096,
  messages: [{ role: "user", content: "Solve this carefully." }],
  thinking: { type: "adaptive" },
  output_config: { effort: "high" },
});

await anthropic.messages.create({
  model: "older-model",
  max_tokens: 4096,
  messages: [{ role: "user", content: "Solve this carefully." }],
  thinking: { type: "enabled", budget_tokens: 2048 },
});
```

The TypeScript core exposes `ReasoningConfig` with `effort`, `summary`, `thinking`, and `budgetTokens`. `OpenAIModelProvider` maps `effort` to Chat Completions `reasoning_effort`; its Responses WebSocket path forwards both `effort` and `summary`. `AnthropicModelProvider` emits `thinking`, converts `budgetTokens` to `budget_tokens`, maps `effort` to `output_config.effort`, and merges effort with structured-output `output_config.format`. Invalid empty effort/summary values and invalid thinking-budget combinations fail before model invocation.

Native Agent RT callers can use the same provider-neutral contract directly:

```python
from agent_rt import ModelRequest, ReasoningConfig

request = ModelRequest(
    messages=messages,
    reasoning=ReasoningConfig(
        effort="high",
        thinking="adaptive",
    ),
)
```

## OpenAI and Anthropic coverage matrix

This matrix lists major OpenAI and Anthropic API/SDK capabilities that are useful migration targets. It is intentionally broader than the compatibility layer so that unsupported areas are explicit rather than implied. In the "Agent RT" column, ✅ means the capability is covered by the compatibility clients, ⚠️ means a narrower subset, and ❌ means it is not covered; the OpenAI and Anthropic columns describe the upstream capability for comparison and are not support markers.

| Area | Agent RT | OpenAI | Anthropic |
| --- | --- | --- | --- |
| Basic text generation | ✅ | Responses + Chat Completions | Messages |
| Sync + async clients | ✅ | ✅ | ✅ |
| Streaming text | ✅ | ✅ | ✅ |
| Function/tool calls | ✅ | ✅ | ✅ |
| Tool-call streaming | ✅ | ✅ | ✅ |
| Tool result history | ✅ | Function outputs | `tool_use` / `tool_result` |
| Structured output | ✅ | JSON Schema / structured outputs | Structured outputs |
| Stateful response continuation | ✅ OpenAI `previous_response_id`; Anthropic client-managed history | `previous_response_id`, conversations/state | Message history/client-managed state |
| Web search tool | ✅ via env-selected Agent RT web retrieval | Hosted web search | Server-side web search |
| Reasoning/thinking controls | ✅ provider-neutral `ReasoningConfig` | Reasoning effort/configuration | Adaptive/extended thinking + effort |
| File search/RAG tool | ✅ OpenAI via Agent RT retrieval | File search + vector stores | Different document/file mechanisms |
| Code execution | ✅ via Agent RT sandbox | Code Interpreter / hosted shell | Server-side code execution |
| Remote MCP | ✅ via Agent RT MCP clients | Remote MCP tool | MCP connector/infrastructure |
| Embeddings | ✅ OpenAI | Embeddings API | — |
| Batch processing | ✅ lifecycle + Anthropic results | Batch API | Message Batches |
| Token counting | ✅ Anthropic | Usage/token information varies by API | Dedicated token counting |
| Models API | ✅ list/retrieve | Models API | Models API |
| Responses API shape | ⚠️ | Full Responses API | — |
| Prompt caching | ❌ in compatibility layer | Prompt caching | Explicit/automatic prompt caching |
| Image/vision inputs | ❌ in compatibility clients (native `ModelProvider` adapters do convert image parts) | Images in Responses/messages | Vision |
| PDF/document inputs | ❌ in compatibility clients (native adapters convert PDF/file parts) | File/PDF input | PDFs/documents/Files API |
| Audio input/output | ❌ | Audio + speech + transcription | No equivalent full audio suite |
| Realtime voice | ❌ | WebRTC/WebSocket/SIP Realtime API | — |
| Files/uploads | ❌ | Files + Uploads | Files API |
| Vector stores | ❌ | Vector stores + file batches | — |
| Computer use | ❌ | Computer use | Computer use |
| Image generation | ❌ | Image generation | — |
| Moderation | ❌ | Moderations API | No equivalent general endpoint |
| Fine-tuning | ❌ | Fine-tuning API | — |
| Evals | ❌ in compatibility clients | Evals/platform evaluation features | Evaluation tooling/platform features |
| Admin/usage/cost APIs | ❌ | Organization/project administration surfaces | Admin, usage and cost APIs |
| Cloud-specific SDK clients | ❌ | Azure-specific ecosystem separately | Bedrock / Vertex integrations |
| Native SDK retry/timeout/raw-response helpers | ❌ mostly missing | Full SDK | Full SDK |

## Native Agent RT path

Compatibility adapters are intended to reduce the first migration diff. New functionality should prefer native Agent RT contracts such as `AgentConfig`, `AgentLoop`, `ModelProvider`, `ToolRegistry`, guardrails, permissions, checkpoints, memory, and sandboxing. This keeps runtime behavior explicit and avoids carrying framework-specific abstractions into new code.

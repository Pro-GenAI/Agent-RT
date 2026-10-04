from __future__ import annotations

from agent_rt import (
    RetrievalRegistry,
    RetrievalResult,
    ContentPart,
    EmbeddingItem,
    EmbeddingResponse,
    InMemoryFileSystem,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
)


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="ok"),),
                tool_calls=(
                    ToolCall(id="call-1", name="lookup", arguments={"q": "x"}),
                ),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="tool_calls",
        )

    async def stream(self, request):
        self.requests.append(request)
        yield ModelStreamEvent(type="text_delta", text="o")
        yield ModelStreamEvent(type="text_delta", text="k")
        yield ModelStreamEvent(
            type="tool_call_delta",
            tool_call_id="call-1",
            tool_name="lookup",
            arguments_delta='{"q":"x"}',
        )
        yield ModelStreamEvent(
            type="completed",
            response=ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="ok"),),
                ),
                model=request.model,
                usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                finish_reason="stop",
            ),
        )


async def test_langchain_stream_batch_and_callable_tool_binding() -> None:
    from agent_rt.langchain import AIMessageChunk, ChatOpenAI

    def lookup(q: str) -> str:
        """Look up a value."""
        return q

    provider = FakeProvider()
    client = ChatOpenAI(model="test-model", provider=provider).bind_tools([lookup])

    response = await client.ainvoke("hello")
    assert response.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert provider.requests[-1].tools[0].name == "lookup"
    assert provider.requests[-1].tools[0].input_schema["properties"]["q"] == {
        "type": "string"
    }

    batch = await client.abatch(["one", "two"])
    assert [item.content for item in batch] == ["ok", "ok"]

    chunks = [chunk async for chunk in client.astream("stream")]
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert chunks[0].content == "o"
    assert chunks[1].content == "k"
    assert chunks[2].tool_calls[0]["name"] == "lookup"
    assert chunks[-1].usage_metadata["total_tokens"] == 5


class StructuredProvider:
    name = "structured"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text='{"answer":"yes"}'),),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            finish_reason="stop",
        )


async def test_langchain_with_structured_output() -> None:
    from agent_rt.langchain import ChatOpenAI

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    provider = StructuredProvider()
    client = ChatOpenAI(model="test-model", provider=provider)

    parsed = await client.with_structured_output(schema).ainvoke("answer")
    assert parsed == {"answer": "yes"}
    assert provider.requests[0].structured_output.schema == schema

    included = await client.with_structured_output(schema, include_raw=True).ainvoke(
        "answer"
    )
    assert included["parsed"] == {"answer": "yes"}
    assert included["raw"].content == '{"answer":"yes"}'
    assert included["parsing_error"] is None


class AgentProvider:
    name = "agent"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id="weather-1",
                            name="weather",
                            arguments={"city": "SF"},
                        ),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="Sunny in SF."),),
            ),
            model=request.model,
            finish_reason="stop",
        )


async def test_langchain_create_agent_executes_callable_tools() -> None:
    from agent_rt.langchain import ChatOpenAI, create_agent, tool

    @tool
    def weather(city: str) -> str:
        """Return deterministic weather."""
        return f"Sunny in {city}."

    provider = AgentProvider()
    model = ChatOpenAI(model="test-model", provider=provider)
    agent = create_agent(model, [weather], system_prompt="Be concise.")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "weather?"}]}
    )

    assert result["messages"][-1].content == "Sunny in SF."
    assert provider.requests[0].messages[0].role == "system"
    assert provider.requests[0].tools[0].name == "weather"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].tool_call_id == "weather-1"


async def test_langchain_create_agent_persists_threads_and_injects_runtime() -> None:
    from agent_rt.langchain import (
        ChatOpenAI,
        InMemoryStore,
        MemorySaver,
        ToolCallLimitMiddleware,
        create_agent,
        tool,
    )

    class ThreadProvider:
        name = "thread"

        def __init__(self) -> None:
            self.requests = []
            self.calls = 0

        async def complete(self, request):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=(),
                        tool_calls=(
                            ToolCall(
                                id="remember-1",
                                name="remember",
                                arguments={"value": "alpha"},
                            ),
                        ),
                    ),
                    model=request.model,
                    finish_reason="tool_calls",
                )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="done"),),
                ),
                model=request.model,
                finish_reason="stop",
            )

    @tool
    def remember(value: str, runtime=None, store=None, state=None) -> str:
        """Persist a value in runtime state and store."""
        assert runtime.context["tenant"] == "acme"
        state["remembered"] = value
        store.put(("memories",), "latest", value)
        return value

    provider = ThreadProvider()
    saver = MemorySaver()
    store = InMemoryStore()
    agent = create_agent(
        ChatOpenAI(model="test-model", provider=provider),
        [remember],
        checkpointer=saver,
        store=store,
        context_schema={"required": ["tenant"]},
        state_schema={"type": "object"},
        middleware=[ToolCallLimitMiddleware(2)],
    )
    config = {
        "configurable": {
            "thread_id": "thread-1",
            "context": {"tenant": "acme"},
        }
    }

    first = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "remember alpha"}]},
        config=config,
    )
    assert first["state"]["remembered"] == "alpha"
    assert store.get(("memories",), "latest") == "alpha"

    second = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what did I say?"}]},
        config=config,
    )
    assert second["state"]["remembered"] == "alpha"
    assert len(provider.requests[-1].messages) > 2
    checkpoint = saver.get("thread-1")
    assert checkpoint["messages"][-1].content == "done"


class RetrievalEmbeddingProvider:
    async def embed(self, request):
        values = request.input if isinstance(request.input, list) else [request.input]
        return EmbeddingResponse(
            data=tuple(
                EmbeddingItem(
                    index=index,
                    embedding=(
                        1.0 if "alpha" in str(value).lower() else 0.0,
                        1.0 if "beta" in str(value).lower() else 0.0,
                    ),
                )
                for index, value in enumerate(values)
            )
        )


async def test_langchain_retrieval_document_and_chain_adapters() -> None:
    from agent_rt.langchain import (
        AgentRTEmbeddings,
        DirectoryLoader,
        MemoryVectorStore,
        PromptTemplate,
        RecursiveCharacterTextSplitter,
        create_retrieval_chain,
        create_stuff_documents_chain,
    )

    fs = InMemoryFileSystem()
    fs.write("docs/alpha.txt", b"alpha apples are red")
    fs.write("docs/beta.txt", b"beta bananas are yellow")
    docs = DirectoryLoader(
        "docs",
        file_system=fs,
        extensions=[".txt"],
    ).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=32,
        chunk_overlap=0,
    ).split_documents(docs)
    store = await MemoryVectorStore.afrom_documents(
        chunks,
        AgentRTEmbeddings(RetrievalEmbeddingProvider(), "embedding-model"),
    )
    retriever = store.as_retriever(k=1)
    relevant = await retriever.ainvoke("alpha apple")
    assert "alpha apples" in relevant[0].page_content

    class ChainLLM:
        async def ainvoke(self, prompt):
            assert "alpha apples are red" in prompt
            return type("Result", (), {"content": "Apples are red."})()

    combine = create_stuff_documents_chain(
        ChainLLM(),
        PromptTemplate.from_template("Context: {context}\nQuestion: {input}\nAnswer:"),
    )
    chain = create_retrieval_chain(retriever, combine)
    result = await chain.ainvoke({"input": "What color are apples?"})
    assert result["answer"] == "Apples are red."
    assert len(result["context"]) == 1


async def test_langchain_retrieval_registry_bridge() -> None:
    from agent_rt.langchain import AgentRTRetriever

    class Provider:
        kind = "knowledge"

        async def search(self, query):
            assert query.limit == 1
            return (
                RetrievalResult(
                    id="guide",
                    title="Migration Guide",
                    content="Agent RT retrieval bridge",
                    score=0.95,
                ),
            )

    registry = RetrievalRegistry()
    registry.register("docs", Provider())
    retriever = AgentRTRetriever(
        registry=registry,
        provider_name="docs",
        k=1,
    )
    docs = await retriever.ainvoke("migration")
    assert docs[0].page_content == "Agent RT retrieval bridge"
    assert docs[0].metadata["score"] == 0.95


class FlakyRunnableProvider:
    name = "flaky"

    def __init__(self) -> None:
        self.attempts = 0
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("temporary")
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="ok"),),
            ),
            model=request.model,
            finish_reason="stop",
        )


async def test_langchain_runnable_config_retry_fallback_pipe_and_callbacks() -> None:
    from agent_rt.langchain import ChatOpenAI

    provider = FlakyRunnableProvider()
    model = ChatOpenAI(model="test-model", provider=provider)
    events = []
    result = await model.with_retry(stop_after_attempt=2).ainvoke(
        "hello",
        config={"callbacks": [lambda event: events.append(event["type"])]},
    )
    assert result.content == "ok"
    assert provider.attempts == 2
    assert events == ["start", "error", "start", "end"]

    class FailedProvider:
        name = "failed"

        async def complete(self, request):
            raise RuntimeError("nope")

    fallback = ChatOpenAI(model="fallback", provider=FakeProvider())
    fallback_result = (
        await ChatOpenAI(model="failed", provider=FailedProvider())
        .with_fallbacks([fallback])
        .ainvoke("hello")
    )
    assert fallback_result.content == "ok"

    class Upper:
        async def ainvoke(self, value, config=None, **kwargs):
            del config, kwargs
            return value.content.upper()

    provider.attempts = 1
    assert await model.pipe(Upper()).ainvoke("hello") == "OK"

    provider.attempts = 1
    configured = model.configurable_fields(temperature="temperature")
    await configured.ainvoke(
        "hello",
        config={"configurable": {"temperature": 0.25}},
    )
    assert provider.requests[-1].temperature == 0.25

    provider.attempts = 1
    batch = await model.abatch(
        ["a", "b"],
        config=[
            {"callbacks": []},
            {"callbacks": []},
        ],
    )
    assert [item.content for item in batch] == ["ok", "ok"]


class SplitToolStreamProvider:
    name = "split-stream"

    async def stream(self, request):
        yield ModelStreamEvent(
            type="tool_call_delta",
            tool_call_id="call-split",
            tool_name="lookup",
            arguments_delta='{"q":',
        )
        yield ModelStreamEvent(
            type="tool_call_delta",
            tool_call_id="call-split",
            arguments_delta='"x"}',
        )
        yield ModelStreamEvent(
            type="completed",
            response=ModelResponse(
                message=ModelMessage(role="assistant", content=()),
                model=request.model,
                finish_reason="tool_calls",
            ),
        )


async def test_langchain_multimodal_blocks_and_split_tool_chunks() -> None:
    from agent_rt.langchain import ChatOpenAI, HumanMessage

    class MultimodalProvider:
        name = "multimodal"

        def __init__(self):
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        ContentPart(
                            type="image",
                            data={"url": "https://example.invalid/image.png"},
                            mime_type="image/png",
                        ),
                        ContentPart(type="text", text="caption"),
                    ),
                ),
                model=request.model,
                finish_reason="stop",
            )

    provider = MultimodalProvider()
    model = ChatOpenAI(model="test-model", provider=provider)
    response = await model.ainvoke(
        HumanMessage(
            [
                {
                    "type": "image",
                    "data": {"url": "https://example.invalid/input.png"},
                    "mime_type": "image/png",
                },
                {"type": "text", "text": "describe"},
            ]
        )
    )
    assert provider.requests[0].messages[0].content[0].type == "image"
    assert response.content_blocks[0]["type"] == "image"
    assert response.content_blocks[1]["text"] == "caption"

    stream_model = ChatOpenAI(model="test-model", provider=SplitToolStreamProvider())
    chunks = [chunk async for chunk in stream_model.astream("stream")]
    assert chunks[0].tool_calls[0]["args"] == {}
    assert chunks[1].tool_calls[0]["args"] == {"q": "x"}
    assert chunks[1].additional_kwargs["arguments"] == '{"q":"x"}'


async def test_langchain_provider_and_tool_structured_output_strategies() -> None:
    from agent_rt.langchain import (
        ChatOpenAI,
        ProviderStrategy,
        ToolStrategy,
        create_agent,
    )

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    provider = StructuredProvider()
    agent = create_agent(
        ChatOpenAI(model="test-model", provider=provider),
        response_format=ProviderStrategy(schema),
    )
    result = await agent.ainvoke("answer")
    assert result["structured_response"] == {"answer": "yes"}
    assert provider.requests[-1].structured_output.schema == schema

    class ToolStructuredProvider:
        name = "tool-structured"

        def __init__(self):
            self.calls = 0
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            self.calls += 1
            arguments = {} if self.calls == 1 else {"answer": "yes"}
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id=f"structured-{self.calls}",
                            name="structured_response",
                            arguments=arguments,
                        ),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )

    tool_provider = ToolStructuredProvider()
    tool_agent = create_agent(
        ChatOpenAI(model="test-model", provider=tool_provider),
        response_format=ToolStrategy(schema, max_retries=2),
    )
    tool_result = await tool_agent.ainvoke("answer")
    assert tool_result["structured_response"] == {"answer": "yes"}
    assert tool_provider.calls == 2
    assert tool_provider.requests[0].structured_output is None
    assert tool_provider.requests[0].tools[-1].name == "structured_response"
    assert tool_provider.requests[1].messages[-1].role == "tool"


async def test_langchain_output_parsers_and_advanced_retrievers() -> None:
    from agent_rt.langchain import (
        ContextualCompressionRetriever,
        Document,
        EnsembleRetriever,
        JsonOutputParser,
        StrOutputParser,
    )

    assert StrOutputParser().invoke(" plain ") == " plain "
    assert JsonOutputParser().invoke('{"answer": 42}') == {"answer": 42}

    alpha = Document("alpha", id="alpha")
    beta = Document("beta", id="beta")

    class Retriever:
        def __init__(self, docs):
            self.docs = docs

        async def ainvoke(self, _query):
            return self.docs

    ensemble = EnsembleRetriever(
        [Retriever([alpha, beta]), Retriever([beta, alpha])],
        weights=[3, 1],
        c=1,
    )
    ranked = await ensemble.ainvoke("hybrid")
    assert [doc.id for doc in ranked] == ["alpha", "beta"]

    class Compressor:
        async def acompress_documents(self, documents, query):
            assert query == "rerank"
            return [doc for doc in documents if doc.id == "beta"]

    compressed = await ContextualCompressionRetriever(
        Retriever([alpha, beta]), Compressor()
    ).ainvoke("rerank")
    assert [doc.id for doc in compressed] == ["beta"]


async def test_langchain_extended_prompt_lcel_loaders_vector_bridge_and_chains() -> None:
    from agent_rt.langchain import (
        ChatPromptTemplate,
        CSVLoader,
        Document,
        ExternalVectorStoreRetriever,
        HumanMessage,
        JSONLoader,
        MessagesPlaceholder,
        PromptTemplate,
        RunnableLambda,
        RunnablePassthrough,
        create_map_reduce_documents_chain,
        create_refine_documents_chain,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer for {name}."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )
    messages = prompt.invoke(
        {"name": "Agent RT", "question": "Ready?", "history": [HumanMessage("Earlier")]}
    )
    assert [message.type for message in messages] == ["system", "human", "human"]
    assert messages[-1].content == "Ready?"

    runnable = RunnablePassthrough().pipe(RunnableLambda(lambda value: value.upper()))
    assert await runnable.ainvoke("ok") == "OK"

    file_system = InMemoryFileSystem()
    file_system.write("docs/items.json", b'[{"text":"alpha"},{"text":"beta"}]')
    file_system.write("docs/items.csv", b"name,value\nalpha,1\nbeta,2\n")
    json_docs = JSONLoader(
        "docs/items.json", file_system=file_system, content_key="text"
    ).load()
    csv_docs = CSVLoader("docs/items.csv", file_system=file_system).load()
    assert [doc.page_content for doc in json_docs] == ["alpha", "beta"]
    assert "name: alpha" in csv_docs[0].page_content

    class ExternalStore:
        async def asimilarity_search(self, query, **kwargs):
            assert query == "alpha"
            assert kwargs["k"] == 1
            return [Document("external", id="external")]

    bridged = await ExternalVectorStoreRetriever(ExternalStore(), k=1).ainvoke("alpha")
    assert bridged[0].id == "external"

    class LLM:
        async def ainvoke(self, input):
            class Response:
                text = f"result:{input}"

            return Response()

    llm = LLM()
    map_reduce = create_map_reduce_documents_chain(
        llm,
        PromptTemplate.from_template("map:{context}"),
        PromptTemplate.from_template("reduce:{context}"),
    )
    reduced = await map_reduce.ainvoke({"context": json_docs})
    assert "reduce:result:map:alpha" in reduced
    assert "result:map:beta" in reduced

    refine = create_refine_documents_chain(
        llm,
        PromptTemplate.from_template("initial:{context}"),
        PromptTemplate.from_template("refine:{existing_answer}:{context}"),
    )
    refined = await refine.ainvoke({"context": json_docs})
    assert "refine:result:initial:alpha:beta" in refined


class _VectorHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _VectorHTTPClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _VectorHTTPResponse(self.payload)


class _LangChainEmbeddings:
    async def aembed_query(self, text):
        assert text == "agent runtime"
        return [0.1, 0.2, 0.3]


async def test_langchain_vector_db_package_aliases_and_retrieval() -> None:
    from agent_rt.langchain_chroma import Chroma
    from agent_rt.langchain_milvus import Milvus
    from agent_rt.langchain_pinecone import PineconeVectorStore
    from agent_rt.langchain_qdrant import QdrantVectorStore
    from agent_rt.langchain_weaviate import WeaviateVectorStore

    assert Chroma.backend == "chroma"
    assert Milvus.backend == "milvus"
    assert PineconeVectorStore.backend == "pinecone"
    assert QdrantVectorStore.backend == "qdrant"
    assert WeaviateVectorStore.backend == "weaviate"

    client = _VectorHTTPClient(
        {
            "result": [
                {
                    "id": "q1",
                    "score": 0.91,
                    "payload": {
                        "text": "Agent RT vector retrieval",
                        "title": "Guide",
                    },
                }
            ]
        }
    )
    store = QdrantVectorStore(
        embedding=_LangChainEmbeddings(),
        environment={
            "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
        },
        client=client,
    )

    documents = await store.asimilarity_search(
        "agent runtime",
        k=1,
        filter={"tenant": "docs"},
    )

    assert documents[0].page_content == "Agent RT vector retrieval"
    assert documents[0].metadata["title"] == "Guide"
    assert documents[0].metadata["score"] == 0.91
    body = client.calls[0][2]["json"]
    assert body["vector"] == [0.1, 0.2, 0.3]
    assert body["filter"] == {"tenant": "docs"}

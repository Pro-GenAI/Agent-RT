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
)
from agent_rt import ToolCall as RTToolCall


class StreamingProvider:
    name = "streaming"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="ok"),),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            finish_reason="stop",
        )

    async def stream(self, request):
        self.requests.append(request)
        yield ModelStreamEvent(type="text_delta", text="o")
        yield ModelStreamEvent(type="text_delta", text="k")
        yield ModelStreamEvent(
            type="completed",
            response=ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="ok"),),
                ),
                model=request.model,
                finish_reason="stop",
            ),
        )


async def test_llamaindex_streaming_chat_and_completion() -> None:
    from agent_rt.llamaindex import ChatMessage, MessageRole, OpenAI

    provider = StreamingProvider()
    llm = OpenAI(model="test-model", provider=provider)

    chat = [
        chunk
        async for chunk in llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
    ]
    completion = [chunk async for chunk in llm.astream_complete("hello")]

    assert [chunk.delta for chunk in chat] == ["o", "k", ""]
    assert chat[-1].message.content == "ok"
    assert chat[-1].message.additional_kwargs["model"] == "test-model"
    assert [chunk.delta for chunk in completion] == ["o", "k", ""]
    assert completion[-1].text == "ok"


class ToolProvider:
    name = "tool"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(),
                tool_calls=(
                    RTToolCall(
                        id="call-1",
                        name="multiply",
                        arguments={"a": 6, "b": 7},
                    ),
                ),
            ),
            model=request.model,
            finish_reason="tool_calls",
        )


async def test_llamaindex_function_tool_and_predict_and_call() -> None:
    from agent_rt.llamaindex import FunctionTool, OpenAI

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    tool = FunctionTool.from_defaults(fn=multiply)
    assert tool.metadata.name == "multiply"
    assert tool.metadata.fn_schema["properties"]["a"] == {"type": "integer"}
    assert tool(a=2, b=3).raw_output == 6

    provider = ToolProvider()
    result = await OpenAI(
        model="test-model",
        provider=provider,
    ).apredict_and_call([tool], user_msg="multiply")

    assert result.tool_calls[0].raw_output == 42
    assert provider.requests[0].tools[0].name == "multiply"


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
            finish_reason="stop",
        )


async def test_llamaindex_structured_prediction_and_structured_llm() -> None:
    from agent_rt.llamaindex import OpenAI

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    provider = StructuredProvider()
    llm = OpenAI(model="test-model", provider=provider)

    parsed = await llm.astructured_predict(schema, "answer")
    response = await llm.as_structured_llm(schema).acomplete("answer")

    assert parsed == {"answer": "yes"}
    assert response.raw == {"answer": "yes"}
    assert provider.requests[0].structured_output.schema == schema


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
                        RTToolCall(
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


async def test_llamaindex_function_agent_memory_context_and_events() -> None:
    from agent_rt.llamaindex import (
        AgentStream,
        Context,
        FunctionAgent,
        FunctionTool,
        Memory,
        OpenAI,
        ToolCall,
        ToolCallResult,
    )

    def weather(city: str) -> str:
        """Return deterministic weather."""
        return f"Sunny in {city}."

    provider = AgentProvider()
    llm = OpenAI(model="test-model", provider=provider)
    agent = FunctionAgent(
        llm=llm,
        tools=[FunctionTool.from_defaults(fn=weather)],
        system_prompt="Be concise.",
    )
    memory = Memory.from_defaults(session_id="thread-1")
    ctx = Context(agent)

    handler = agent.run("weather?", memory=memory, ctx=ctx)
    events = [event async for event in handler.stream_events()]
    result = await handler

    assert str(result) == "Sunny in SF."
    assert any(isinstance(event, ToolCall) for event in events)
    assert any(isinstance(event, ToolCallResult) for event in events)
    assert any(isinstance(event, AgentStream) for event in events)
    assert provider.requests[0].messages[0].role == "system"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].tool_call_id == "weather-1"
    assert memory.get_all()[-1].content == "Sunny in SF."
    assert await ctx.store.get("memory") is memory


def test_llamaindex_memory_and_chat_message_helpers() -> None:
    from agent_rt.llamaindex import ChatMessage, Memory, MessageRole

    message = ChatMessage.from_str("hello")
    memory = Memory.from_defaults(session_id="s1", token_limit=100)
    memory.put(message)

    assert message.role == MessageRole.USER
    assert memory.get() == [message]
    assert memory.get_all() == [message]
    memory.reset()
    assert memory.get() == []


class EdgeProvider:
    name = "edge"

    def __init__(self) -> None:
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
                    ContentPart(type="text", text='{"answer":"yes"}'),
                ),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="stop",
        )

    async def stream(self, request):
        self.requests.append(request)
        yield ModelStreamEvent(type="text_delta", text='{"answer":')
        yield ModelStreamEvent(type="text_delta", text='"yes"}')
        yield ModelStreamEvent(
            type="tool_call_delta",
            tool_call_id="lookup-1",
            tool_name="lookup",
            arguments_delta='{"q":',
        )
        yield ModelStreamEvent(
            type="tool_call_delta",
            tool_call_id="lookup-1",
            arguments_delta='"x"}',
        )
        yield ModelStreamEvent(
            type="completed",
            response=ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text='{"answer":"yes"}'),),
                ),
                model=request.model,
                usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                finish_reason="stop",
            ),
        )


async def test_llamaindex_python_stream_metadata_multimodal_and_partial_structured() -> (
    None
):
    from agent_rt.llamaindex import ChatMessage, MessageRole, OpenAI, Settings

    Settings.reset()
    callback_events = []
    Settings.callback_manager.on(lambda event: callback_events.append(event["type"]))
    provider = EdgeProvider()
    llm = OpenAI(model="test-model", provider=provider)

    response = await llm.achat(
        [ChatMessage(role=MessageRole.USER, content="show blocks")]
    )
    assert response.message.blocks[0].type == "image"
    assert response.message.additional_kwargs["usage"].total_tokens == 5
    assert response.message.additional_kwargs["finish_reason"] == "stop"

    chunks = [
        chunk
        async for chunk in llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="stream")]
        )
    ]
    tool_chunks = [
        chunk for chunk in chunks if chunk.message.additional_kwargs.get("tool_calls")
    ]
    assert tool_chunks[-1].message.additional_kwargs["tool_call_arguments"] == {
        "q": "x"
    }
    assert "llm-stream" in callback_events

    structured = llm.as_structured_llm(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    )
    partials = [chunk async for chunk in structured.astream_complete("answer")]
    assert partials[-1].raw == {"answer": "yes"}
    Settings.reset()


class EmbeddingProvider:
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


async def test_llamaindex_python_rag_migration_surface() -> None:
    from agent_rt.llamaindex import (
        AgentRTEmbedding,
        Document,
        SentenceSplitter,
        Settings,
        SimpleDirectoryReader,
        VectorStoreIndex,
    )

    fs = InMemoryFileSystem()
    fs.write("docs/alpha.txt", b"alpha apples are red")
    fs.write("docs/beta.txt", b"beta bananas are yellow")
    docs = SimpleDirectoryReader(
        file_system=fs,
        input_dir="docs",
        required_exts=[".txt"],
    ).load_data()
    assert len(docs) == 2
    assert isinstance(docs[0], Document)

    Settings.reset()
    Settings.embed_model = AgentRTEmbedding(EmbeddingProvider(), "embedding-model")
    index = await VectorStoreIndex.from_documents(
        docs,
        transformations=[SentenceSplitter(chunk_size=8, chunk_overlap=0)],
    )
    results = await index.as_retriever(similarity_top_k=1).aretrieve("alpha")
    assert "alpha apples" in results[0].node.text

    class SynthLLM:
        async def acomplete(self, prompt: str):
            assert "alpha apples are red" in prompt
            return type("Completion", (), {"text": "Apples are red."})()

    response = await index.as_query_engine(llm=SynthLLM(), similarity_top_k=1).aquery(
        "What color are apples?"
    )
    assert str(response) == "Apples are red."
    assert len(response.source_nodes) == 1
    Settings.reset()


async def test_llamaindex_python_retrieval_bridge_and_prompt_settings() -> None:
    from agent_rt.llamaindex import (
        AgentRTRetriever,
        ChatPromptTemplate,
        JSONOutputParser,
        OpenAI,
        PromptTemplate,
        Settings,
    )

    assert PromptTemplate("Hello {name}").format(name="Ada") == "Hello Ada"
    messages = ChatPromptTemplate(
        [
            {"role": "system", "content": "You are {role}."},
            {"role": "user", "content": "{question}"},
        ]
    ).format_messages(role="concise", question="Why?")
    assert messages[0].content == "You are concise."
    assert JSONOutputParser().parse('{"ok": true}') == {"ok": True}

    class RetrievalProvider:
        kind = "knowledge"

        async def search(self, query):
            assert query.limit == 1
            return (
                RetrievalResult(
                    id="r1",
                    title="Guide",
                    content="Agent RT migration guide",
                    score=0.9,
                ),
            )

    registry = RetrievalRegistry()
    registry.register("docs", RetrievalProvider())
    nodes = await AgentRTRetriever(
        registry=registry,
        provider_name="docs",
        similarity_top_k=1,
    ).aretrieve("migration")
    assert nodes[0].node.text == "Agent RT migration guide"

    class CountingProvider(StreamingProvider):
        async def count_tokens(self, request):
            return 7

    Settings.reset()
    events = []
    Settings.callback_manager.on(lambda event: events.append(event["type"]))
    llm = OpenAI(model="test-model", provider=CountingProvider())
    await llm.achat([messages[-1]])
    assert events == ["llm-start", "llm-end"]
    assert await llm.acount_tokens("hello") == 7

    Settings.tokenizer = lambda text: text.split()
    fallback = OpenAI(model="test-model", provider=StreamingProvider())
    assert await fallback.acount_tokens("one two three") == 3
    Settings.reset()


async def test_llamaindex_memory_blocks_and_durable_workflow() -> None:
    from agent_rt.llamaindex import (
        ChatMessage,
        Context,
        Event,
        FactExtractionMemoryBlock,
        HumanResponseEvent,
        InputRequiredEvent,
        Memory,
        MessageRole,
        StartEvent,
        StaticMemoryBlock,
        StopEvent,
        VectorMemoryBlock,
        Workflow,
        step,
    )

    from agent_rt import InMemoryCheckpointStore

    facts = FactExtractionMemoryBlock(name="facts", priority=10)
    vector = VectorMemoryBlock(name="vector", priority=5, top_k=2)
    memory = Memory.from_defaults(
        session_id="memory-1",
        token_limit=3,
        blocks=[
            facts,
            vector,
            StaticMemoryBlock(name="profile", value="concise", priority=20),
        ],
    )
    await memory.aput(ChatMessage(role=MessageRole.USER, content="alpha beta"))
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="gamma delta"))
    assert len(await memory.aget_all()) == 1
    assert "alpha beta" in facts.get("alpha")
    assert "alpha beta" in vector.get("alpha")
    restored = Memory.from_dict(memory.to_dict())
    assert restored.get_all()[-1].content == "gamma delta"

    class NumberEvent(Event):
        pass

    class DemoWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, event: StartEvent):
            ctx.state["values"] = []
            return [NumberEvent(data=1), NumberEvent(data=2)]

        @step
        async def collect(self, ctx: Context, event: NumberEvent):
            ctx.state["values"].append(event.data)
            if len(ctx.state["values"]) == 2:
                return InputRequiredEvent(prefix="approve?")
            return None

        @step
        async def human(self, ctx: Context, event: HumanResponseEvent):
            ctx.state["approved"] = event.response
            return StopEvent(result=sum(ctx.state["values"]))

    checkpoints = InMemoryCheckpointStore()
    workflow = DemoWorkflow(checkpoint_store=checkpoints, checkpoint_id="wf-1")
    handler = workflow.run()
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            await handler.respond("yes")
    assert await handler == 3
    resumed = Context.load_checkpoint(workflow, checkpoints, "wf-1")
    assert resumed.state["values"] == [1, 2]
    encoded = resumed.serialize()
    assert Context.deserialize(workflow, encoded).state["approved"] == "yes"


class TextSequenceProvider:
    name = "text-sequence"

    def __init__(self, texts):
        self.texts = list(texts)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        text = self.texts.pop(0)
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text=text),),
            ),
            model=request.model,
            finish_reason="stop",
        )


async def test_llamaindex_react_codeact_and_multiagent_handoff() -> None:
    from agent_rt.llamaindex import (
        AgentWorkflow,
        CodeActAgent,
        FunctionAgent,
        FunctionTool,
        OpenAI,
        ReActAgent,
    )

    def multiply(a: int, b: int) -> int:
        """Multiply integers."""
        return a * b

    react_provider = TextSequenceProvider(
        [
            'Thought: calculate\nAction: multiply\nAction Input: {"a": 6, "b": 7}',
            "Thought: done\nAnswer: 42",
        ]
    )
    react = ReActAgent(
        llm=OpenAI(model="test-model", provider=react_provider),
        tools=[FunctionTool.from_defaults(fn=multiply)],
        streaming=False,
    )
    assert str(await react.run("6*7?")) == "42"
    assert "Observation: 42" in react_provider.requests[1].messages[-1].content[0].text

    fence = "`" * 3
    code_provider = TextSequenceProvider(
        [
            "Thought: compute\n" + fence + "python\n6 * 7\n" + fence,
            "Thought: done\nAnswer: 42",
        ]
    )
    executed = []
    code_agent = CodeActAgent(
        llm=OpenAI(model="test-model", provider=code_provider),
        code_executor=lambda code: executed.append(code) or 42,
        streaming=False,
    )
    assert str(await code_agent.run("6*7?")) == "42"
    assert executed == ["6 * 7"]

    class HandoffProvider:
        name = "handoff"

        async def complete(self, request):
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        RTToolCall(
                            id="handoff-1",
                            name="handoff_to_agent",
                            arguments={"agent_name": "writer", "message": "finish it"},
                        ),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )

    researcher = FunctionAgent(
        name="researcher",
        llm=OpenAI(model="test-model", provider=HandoffProvider()),
        streaming=False,
    )
    writer = FunctionAgent(
        name="writer",
        llm=OpenAI(
            model="test-model", provider=TextSequenceProvider(["final response"])
        ),
        streaming=False,
    )
    workflow = AgentWorkflow.from_agents([researcher, writer], root_agent="researcher")
    result = await workflow.run("do work")
    assert str(result) == "final response"
    assert result.current_agent_name == "writer"


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


async def test_llamaindex_vector_store_module_aliases_and_retrieval() -> None:
    from agent_rt.llama_index.vector_stores.chroma import ChromaVectorStore
    from agent_rt.llama_index.vector_stores.milvus import MilvusVectorStore
    from agent_rt.llama_index.vector_stores.pinecone import PineconeVectorStore
    from agent_rt.llama_index.vector_stores.qdrant import QdrantVectorStore
    from agent_rt.llama_index.vector_stores.weaviate import WeaviateVectorStore

    assert ChromaVectorStore.backend == "chroma"
    assert MilvusVectorStore.backend == "milvus"
    assert PineconeVectorStore.backend == "pinecone"
    assert QdrantVectorStore.backend == "qdrant"
    assert WeaviateVectorStore.backend == "weaviate"

    client = _VectorHTTPClient(
        {
            "result": [
                {
                    "id": "q1",
                    "score": 0.88,
                    "payload": {
                        "text": "LlamaIndex compatibility over Agent RT",
                        "title": "Migration",
                    },
                }
            ]
        }
    )
    store = QdrantVectorStore(
        environment={
            "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
        },
        client=client,
    )
    nodes = await store.aquery(
        {
            "query_embedding": [0.1, 0.2],
            "query_str": "migration",
            "similarity_top_k": 1,
            "filters": {"tenant": "docs"},
        }
    )

    assert nodes[0].node.text == "LlamaIndex compatibility over Agent RT"
    assert nodes[0].node.metadata["title"] == "Migration"
    assert nodes[0].score == 0.88
    body = client.calls[0][2]["json"]
    assert body["vector"] == [0.1, 0.2]
    assert body["filter"] == {"tenant": "docs"}

    from agent_rt.llamaindex import VectorStoreIndex

    class _Embed:
        async def aget_query_embedding(self, text):
            assert text == "migration"
            return [0.1, 0.2]

    index = VectorStoreIndex.from_vector_store(store, embed_model=_Embed())
    retrieved = await index.as_retriever(
        similarity_top_k=1,
        filters={"tenant": "docs"},
    ).aretrieve("migration")
    assert retrieved[0].node.text == "LlamaIndex compatibility over Agent RT"

    try:
        store.add([nodes[0].node])
    except NotImplementedError as exc:
        assert "write/upsert remains backend-native" in str(exc)
    else:
        raise AssertionError("expected retrieval-only vector store to reject writes")

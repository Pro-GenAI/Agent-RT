from __future__ import annotations

from types import SimpleNamespace

from agent_rt.anthropic import Anthropic, AsyncAnthropic
from agent_rt.langchain import ChatOpenAI, HumanMessage
from agent_rt.llamaindex import ChatMessage, MessageRole
from agent_rt.llamaindex import OpenAI as LlamaOpenAI
from agent_rt.openai import AsyncOpenAI, OpenAI
from agent_rt.langchain_openai import ChatOpenAI as ExactChatOpenAI
from agent_rt.langchain_anthropic import ChatAnthropic as ExactChatAnthropic
from agent_rt.llama_index import OpenAI as ExactRootLlamaOpenAI
from agent_rt.llama_index.llms.openai import OpenAI as ExactLlamaOpenAI
from agent_rt.llama_index.llms.anthropic import Anthropic as ExactLlamaAnthropic

from agent_rt import (
    AuthorizedMCPClient,
    BatchJob,
    CallbackSandboxBackend,
    CodeInterpreterRegistry,
    ContentPart,
    EmbeddingItem,
    EmbeddingResponse,
    EnvironmentWebSearchProvider,
    EnvironmentVectorDBProvider,
    POPULAR_VECTOR_DB_BACKENDS,
    register_popular_vector_db_backends,
    MCPAccessPolicy,
    MCPCapabilityFilter,
    MCPClient,
    ModelCatalogEntry,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    RetrievalQuery,
    RetrievalRegistry,
    RetrievalResult,
    VectorDBProviderRegistry,
    vector_db_config_from_environment,
    vector_db_provider_from_environment,
    SandboxCommandResult,
    SandboxSession,
    ToolCall,
)


def test_exact_prefix_compat_imports():
    assert ExactChatOpenAI is ChatOpenAI
    assert ExactChatAnthropic.__name__ == "ChatAnthropic"
    assert ExactRootLlamaOpenAI is LlamaOpenAI
    assert ExactLlamaOpenAI is LlamaOpenAI
    assert ExactLlamaAnthropic.__name__ == "Anthropic"


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

    async def embed(self, request):
        self.requests.append(request)
        values = request.input if isinstance(request.input, list) else [request.input]
        return EmbeddingResponse(
            data=tuple(
                EmbeddingItem(index=index, embedding=(float(index), 0.5))
                for index, _ in enumerate(values)
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=4, total_tokens=4),
        )

    async def count_tokens(self, request):
        self.requests.append(request)
        return 17

    async def list_models(self):
        return (
            ModelCatalogEntry(
                id="model-a",
                display_name="Model A",
                object="model",
                owned_by="agent-rt",
            ),
            ModelCatalogEntry(
                id="model-b",
                display_name="Model B",
                object="model",
                owned_by="agent-rt",
            ),
        )

    async def retrieve_model(self, model):
        return ModelCatalogEntry(
            id=model,
            display_name=f"Display {model}",
            object="model",
            owned_by="agent-rt",
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
                    tool_calls=(
                        ToolCall(id="call-1", name="lookup", arguments={"q": "x"}),
                    ),
                ),
                model=request.model,
                usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                finish_reason="tool_calls",
            ),
        )


async def test_langchain_compat_uses_agent_rt_provider() -> None:
    provider = FakeProvider()
    client = ChatOpenAI(model="test-model", provider=provider, temperature=0)
    response = await client.ainvoke([HumanMessage("hello")])

    assert response.content == "ok"
    assert response.tool_calls == [
        {"id": "call-1", "name": "lookup", "args": {"q": "x"}, "type": "tool_call"}
    ]
    assert provider.requests[0].model == "test-model"
    assert provider.requests[0].messages[0].role == "user"


async def test_llamaindex_compat_chat_and_completion() -> None:
    provider = FakeProvider()
    client = LlamaOpenAI(model="test-model", provider=provider)

    chat = await client.achat([ChatMessage(role=MessageRole.USER, content="hello")])
    completion = await client.acomplete("hello")

    assert chat.message.content == "ok"
    assert chat.message.additional_kwargs["tool_calls"][0].function.name == "lookup"
    assert completion.text == "ok"


async def test_openai_compat_chat_completions_shape() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)
    response = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.choices[0].message.content == "ok"
    assert response.choices[0].message.tool_calls[0].function.name == "lookup"
    assert response.usage.total_tokens == 5


async def test_anthropic_compat_messages_shape() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)
    response = await client.messages.create(
        model="test-model",
        max_tokens=64,
        system="Be concise.",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.content[0].text == "ok"
    assert response.content[1].name == "lookup"
    assert provider.requests[0].messages[0].role == "system"


async def test_openai_chat_completions_stream_shape() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)
    stream = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    chunks = [chunk async for chunk in stream]
    assert chunks[0].choices[0].delta.content == "o"
    assert chunks[1].choices[0].delta.content == "k"
    assert chunks[2].choices[0].delta.tool_calls[0].function.name == "lookup"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert chunks[-1].usage.total_tokens == 5


async def test_openai_responses_create_and_stream() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)
    response = await client.responses.create(
        model="test-model",
        instructions="Be concise.",
        input="hello",
    )

    assert response.output_text == "ok"
    assert response.output[0].content[0].type == "output_text"
    assert response.output[1].type == "function_call"
    assert provider.requests[0].messages[0].role == "system"

    stream = await client.responses.create(
        model="test-model",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-0",
                "output": "done",
            }
        ],
        stream=True,
    )
    events = [event async for event in stream]
    assert events[0].type == "response.output_text.delta"
    assert events[2].type == "response.function_call_arguments.delta"
    assert events[-1].type == "response.completed"
    assert provider.requests[1].messages[0].role == "tool"
    assert provider.requests[1].messages[0].tool_call_id == "call-0"


async def test_anthropic_streaming_and_tool_history() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)
    stream = await client.messages.create(
        model="test-model",
        max_tokens=64,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-0",
                        "name": "lookup",
                        "input": {"q": "x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-0",
                        "content": "done",
                    }
                ],
            },
        ],
        stream=True,
    )

    events = [event async for event in stream]
    assert events[0].type == "message_start"
    assert events[1].delta.text == "o"
    assert events[-1].type == "message_stop"
    request = provider.requests[0]
    assert request.messages[0].tool_calls[0].name == "lookup"
    assert request.messages[1].role == "tool"
    assert request.messages[1].tool_call_id == "call-0"


async def test_anthropic_messages_stream_manager() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)
    async with client.messages.stream(
        model="test-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
    ) as stream:
        text = "".join([part async for part in stream.text_stream])
        final = await stream.get_final_message()

    assert text == "ok"
    assert final.content[0].text == "ok"


def test_sync_openai_and_anthropic_streaming_surfaces() -> None:
    provider = FakeProvider()
    openai = OpenAI(provider=provider)
    chunks = list(
        openai.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )
    assert chunks[0].choices[0].delta.content == "o"

    response_events = list(
        openai.responses.create(model="test-model", input="hello", stream=True)
    )
    assert response_events[-1].type == "response.completed"

    anthropic = Anthropic(provider=provider)
    with anthropic.messages.stream(
        model="test-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
    ) as stream:
        assert "".join(stream.text_stream) == "ok"
        assert stream.get_final_message().content[0].text == "ok"


class FakeStructuredModel:
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }


async def test_openai_structured_output_request_shapes() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": schema,
                "strict": False,
            },
        },
    )
    chat_requirement = provider.requests[-1].structured_output
    assert chat_requirement.name == "answer"
    assert chat_requirement.schema == schema
    assert chat_requirement.strict is False

    await client.responses.create(
        model="test-model",
        input="hello",
        text={
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": schema,
                "strict": True,
            }
        },
    )
    responses_requirement = provider.requests[-1].structured_output
    assert responses_requirement.name == "answer"
    assert responses_requirement.schema == schema
    assert responses_requirement.strict is True


async def test_anthropic_structured_output_request_shapes() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    await client.messages.create(
        model="test-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )
    requirement = provider.requests[-1].structured_output
    assert requirement.schema == schema
    assert requirement.strict is True

    await client.messages.create(
        model="test-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
        output_format=FakeStructuredModel,
    )
    model_requirement = provider.requests[-1].structured_output
    assert model_requirement.name == "FakeStructuredModel"
    assert model_requirement.schema["properties"]["answer"]["type"] == "string"


async def test_openai_reasoning_controls_request_shapes() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)

    await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="high",
    )
    chat_reasoning = provider.requests[-1].reasoning
    assert chat_reasoning.effort == "high"
    assert chat_reasoning.summary is None

    await client.responses.create(
        model="test-model",
        input="hello",
        reasoning={"effort": "xhigh", "summary": "auto"},
    )
    responses_reasoning = provider.requests[-1].reasoning
    assert responses_reasoning.effort == "xhigh"
    assert responses_reasoning.summary == "auto"


async def test_anthropic_thinking_controls_request_shapes() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)

    await client.messages.create(
        model="test-model",
        max_tokens=256,
        messages=[{"role": "user", "content": "hello"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )
    adaptive = provider.requests[-1].reasoning
    assert adaptive.thinking == "adaptive"
    assert adaptive.effort == "high"
    assert adaptive.budget_tokens is None

    await client.messages.create(
        model="test-model",
        max_tokens=4096,
        messages=[{"role": "user", "content": "hello"}],
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    legacy = provider.requests[-1].reasoning
    assert legacy.thinking == "enabled"
    assert legacy.budget_tokens == 2048


async def test_openai_embeddings_compat_shape() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)

    response = await client.embeddings.create(
        model="embed-model",
        input=["alpha", "beta"],
        dimensions=2,
        encoding_format="float",
    )

    assert response.object == "list"
    assert response.model == "embed-model"
    assert response.data[0].object == "embedding"
    assert response.data[1].embedding == (1.0, 0.5)
    assert response.usage.prompt_tokens == 4
    request = provider.requests[0]
    assert request.model == "embed-model"
    assert request.dimensions == 2
    assert request.encoding_format == "float"


async def test_anthropic_token_counting_compat_shape() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)

    response = await client.messages.count_tokens(
        model="test-model",
        system="Be concise.",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.input_tokens == 17
    request = provider.requests[0]
    assert request.model == "test-model"
    assert request.messages[0].role == "system"


def test_sync_embeddings_and_token_counting_surfaces() -> None:
    provider = FakeProvider()
    openai = OpenAI(provider=provider)
    embedding = openai.embeddings.create(model="embed-model", input="hello")
    assert embedding.data[0].embedding == (0.0, 0.5)

    anthropic = Anthropic(provider=provider)
    count = anthropic.messages.count_tokens(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert count.input_tokens == 17


class CodeExecutionProvider:
    name = "code-execution-fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            tool = request.tools[0]
            arguments = (
                {"command": "printf shell-ok"}
                if "shell" in tool.name
                else {"code": "print(6 * 7)", "runtime": "python"}
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(id="code-1", name=tool.name, arguments=arguments),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )
        tool_message = request.messages[-1]
        result = tool_message.content[0].text
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text=f"result:{result}"),),
            ),
            model=request.model,
            finish_reason="stop",
        )


def _sandbox_session():
    interpreters = CodeInterpreterRegistry()

    async def python_runner(
        _session_id,
        runtime,
        code,
        state,
        _workspace,
        _environment,
        _limits,
        _network_policy,
    ):
        state["runs"] = int(state.get("runs", 0)) + 1
        return SandboxCommandResult(
            exit_code=0,
            stdout=f"{runtime}:{code}:run={state['runs']}".encode(),
        )

    async def shell_runner(
        _session_id,
        command,
        _workspace,
        _limits,
        _environment,
        _network_policy,
    ):
        return SandboxCommandResult(
            exit_code=0,
            stdout=("argv=" + " ".join(command.argv)).encode(),
        )

    interpreters.register("python", python_runner)
    return SandboxSession(
        "compat-code",
        CallbackSandboxBackend(shell_runner),
        interpreters=interpreters,
    )


async def test_openai_responses_code_interpreter_uses_agent_rt_sandbox() -> None:
    provider = CodeExecutionProvider()
    session = _sandbox_session()
    client = AsyncOpenAI(provider=provider, sandbox_session=session)

    response = await client.responses.create(
        model="test-model",
        input="Calculate 6 * 7 with Python.",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    )

    assert "python:print(6 * 7):run=1" in response.output_text
    assert len(provider.requests) == 2
    assert provider.requests[0].tools[0].name == "agent_rt_code_execution"
    assert provider.requests[1].messages[-1].role == "tool"
    assert session.interpreter_state["python"]["runs"] == 1


async def test_openai_shell_tool_uses_agent_rt_sandbox() -> None:
    provider = CodeExecutionProvider()
    client = AsyncOpenAI(provider=provider, sandbox_session=_sandbox_session())

    response = await client.responses.create(
        model="test-model",
        input="Run a shell command.",
        tools=[{"type": "shell"}],
    )

    assert "argv=sh -lc printf shell-ok" in response.output_text
    assert provider.requests[0].tools[0].name == "agent_rt_shell_execution"


async def test_anthropic_code_execution_uses_agent_rt_sandbox() -> None:
    provider = CodeExecutionProvider()
    session = _sandbox_session()
    client = AsyncAnthropic(provider=provider, sandbox_session=session)

    response = await client.messages.create(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "Calculate 6 * 7."}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    )

    assert "python:print(6 * 7):run=1" in response.content[0].text
    assert provider.requests[0].tools[0].name == "agent_rt_code_execution"
    assert provider.requests[1].messages[-1].tool_call_id == "code-1"


async def test_code_execution_streaming_returns_final_vendor_shapes() -> None:
    openai_provider = CodeExecutionProvider()
    openai = AsyncOpenAI(
        provider=openai_provider,
        sandbox_session=_sandbox_session(),
    )
    stream = await openai.responses.create(
        model="test-model",
        input="Calculate.",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        stream=True,
    )
    openai_events = [event async for event in stream]
    assert openai_events[0].type == "response.output_text.delta"
    assert openai_events[-1].type == "response.completed"

    anthropic_provider = CodeExecutionProvider()
    anthropic = AsyncAnthropic(
        provider=anthropic_provider,
        sandbox_session=_sandbox_session(),
    )
    stream = await anthropic.messages.create(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "Calculate."}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        stream=True,
    )
    anthropic_events = [event async for event in stream]
    assert anthropic_events[0].type == "message_start"
    assert anthropic_events[1].delta.type == "text_delta"
    assert anthropic_events[-1].type == "message_stop"


async def test_code_execution_requires_explicit_sandbox_session() -> None:
    provider = CodeExecutionProvider()
    client = AsyncOpenAI(provider=provider)

    try:
        await client.responses.create(
            model="test-model",
            input="Calculate.",
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        )
    except RuntimeError as exc:
        assert "sandbox_session" in str(exc)
    else:
        raise AssertionError(
            "code execution should require an explicit sandbox session"
        )
    assert provider.requests == []


async def test_anthropic_stream_manager_supports_code_execution() -> None:
    provider = CodeExecutionProvider()
    client = AsyncAnthropic(provider=provider, sandbox_session=_sandbox_session())

    async with client.messages.stream(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "Calculate."}],
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    ) as stream:
        text = "".join([part async for part in stream.text_stream])
        final = await stream.get_final_message()

    assert "python:print(6 * 7):run=1" in text
    assert final.content[0].text == text


class MCPExecutionProvider:
    name = "mcp-execution-fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            tool = request.tools[0]
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(id="mcp-call-1", name=tool.name, arguments={"q": "x"}),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )
        tool_message = request.messages[-1]
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(
                    ContentPart(
                        type="text", text=f"used:{tool_message.content[0].text}"
                    ),
                ),
            ),
            model=request.model,
            finish_reason="stop",
        )


class FakeMCPTransport:
    def __init__(self) -> None:
        self.calls = []

    async def request(self, method, params=None):
        self.calls.append((method, params or {}))
        if method == "initialize":
            return {
                "serverInfo": {"name": "docs", "version": "1"},
                "capabilities": {"tools": True},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search docs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                    {
                        "name": "delete",
                        "description": "Delete docs",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            }
        if method == "tools/call":
            return {
                "ok": True,
                "tool": params["name"],
                "content": [
                    {"type": "text", "text": f"found:{params['arguments']['q']}"}
                ],
            }
        raise AssertionError(method)


def _authorized_mcp_client():
    transport = FakeMCPTransport()
    client = MCPClient(transport, requested_capabilities=("tools",))
    authorized = AuthorizedMCPClient(
        client,
        MCPAccessPolicy(capability_filter=MCPCapabilityFilter(tools=("search",))),
    )
    return authorized, transport


async def test_openai_remote_mcp_uses_agent_rt_client_and_policy() -> None:
    provider = MCPExecutionProvider()
    mcp, transport = _authorized_mcp_client()
    client = AsyncOpenAI(provider=provider, mcp_clients={"docs": mcp})

    response = await client.responses.create(
        model="test-model",
        input="Search the docs.",
        tools=[
            {
                "type": "mcp",
                "server_label": "docs",
                "server_url": "https://example.invalid/mcp",
                "allowed_tools": ["search", "delete"],
                "require_approval": "never",
            }
        ],
    )

    assert response.output[0].type == "mcp_call"
    assert response.output[0].server_label == "docs"
    assert response.output[0].name == "search"
    assert response.output[1].type == "message"
    assert "found:x" in response.output_text
    assert provider.requests[0].tools[0].name == "mcp__docs__search"
    assert [method for method, _ in transport.calls] == [
        "initialize",
        "tools/list",
        "tools/call",
    ]


async def test_anthropic_remote_mcp_beta_shape_and_tool_filtering() -> None:
    provider = MCPExecutionProvider()
    mcp, transport = _authorized_mcp_client()
    client = AsyncAnthropic(provider=provider, mcp_clients={"docs": mcp})

    response = await client.beta.messages.create(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "Search the docs."}],
        mcp_servers=[
            {
                "type": "url",
                "url": "https://example.invalid/mcp",
                "name": "docs",
                "authorization_token": "ignored-by-agent-rt-transport",
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

    assert response.content[0].type == "mcp_tool_use"
    assert response.content[0].server_name == "docs"
    assert response.content[1].type == "mcp_tool_result"
    assert response.content[2].type == "text"
    assert "found:x" in response.content[2].text
    assert provider.requests[0].tools[0].name == "mcp__docs__search"
    assert [method for method, _ in transport.calls][-1] == "tools/call"


async def test_anthropic_remote_mcp_stream_manager() -> None:
    provider = MCPExecutionProvider()
    mcp, _transport = _authorized_mcp_client()
    client = AsyncAnthropic(provider=provider, mcp_clients={"docs": mcp})

    async with client.beta.messages.stream(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "Search."}],
        mcp_servers=[
            {"type": "url", "url": "https://example.invalid/mcp", "name": "docs"}
        ],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "docs"}],
        betas=["mcp-client-2026-09-15"],
    ) as stream:
        text = "".join([part async for part in stream.text_stream])
        final = await stream.get_final_message()

    assert "found:x" in text
    assert final.content[-1].text == text


async def test_remote_mcp_requires_injected_agent_rt_client() -> None:
    provider = MCPExecutionProvider()
    client = AsyncOpenAI(provider=provider)

    try:
        await client.responses.create(
            model="test-model",
            input="Search.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "docs",
                    "server_url": "https://example.invalid/mcp",
                }
            ],
        )
    except RuntimeError as exc:
        assert "mcp_clients" in str(exc)
    else:
        raise AssertionError(
            "remote MCP should require an injected Agent RT MCP client"
        )
    assert provider.requests == []


async def test_openai_models_compat_surface() -> None:
    provider = FakeProvider()
    client = AsyncOpenAI(provider=provider)

    page = await client.models.list()
    model = await client.models.retrieve("model-a")

    assert page.object == "list"
    assert [item.id for item in page.data] == ["model-a", "model-b"]
    assert page.data[0].object == "model"
    assert page.data[0].owned_by == "agent-rt"
    assert model.id == "model-a"
    assert model.object == "model"


async def test_anthropic_models_compat_surface() -> None:
    provider = FakeProvider()
    client = AsyncAnthropic(provider=provider)

    page = await client.models.list()
    model = await client.models.retrieve("model-a")

    assert [item.id for item in page.data] == ["model-a", "model-b"]
    assert page.first_id == "model-a"
    assert page.last_id == "model-b"
    assert page.has_more is False
    assert page.data[0].type == "model"
    assert page.data[0].display_name == "Model A"
    assert model.id == "model-a"
    assert model.display_name == "Display model-a"


def test_sync_models_api_surfaces() -> None:
    provider = FakeProvider()
    openai = OpenAI(provider=provider)
    anthropic = Anthropic(provider=provider)

    assert openai.models.list().data[0].id == "model-a"
    assert openai.models.retrieve("model-b").id == "model-b"
    assert anthropic.models.list().data[0].display_name == "Model A"
    assert anthropic.models.retrieve("model-b").id == "model-b"


class StatefulResponsesProvider:
    name = "stateful-responses-fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        user_texts = [
            part.text
            for message in request.messages
            if message.role == "user"
            for part in message.content
            if part.type == "text" and part.text is not None
        ]
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(
                    ContentPart(
                        type="text",
                        text="seen:" + "|".join(user_texts),
                    ),
                ),
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    async def stream(self, request):
        response = await self.complete(request)
        yield ModelStreamEvent(type="text_delta", text=response.message.content[0].text)
        yield ModelStreamEvent(type="completed", response=response)


async def test_openai_previous_response_id_continues_state() -> None:
    provider = StatefulResponsesProvider()
    client = AsyncOpenAI(provider=provider)

    first = await client.responses.create(
        model="test-model",
        input="first",
    )
    second = await client.responses.create(
        model="test-model",
        input="second",
        previous_response_id=first.id,
    )

    assert first.id == "resp_agent_rt_1"
    assert second.id == "resp_agent_rt_2"
    assert first.output_text == "seen:first"
    assert second.output_text == "seen:first|second"
    assert [message.role for message in provider.requests[1].messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert provider.requests[1].messages[1].content[0].text == "seen:first"


async def test_openai_previous_response_id_stream_commits_state_on_completion() -> None:
    provider = StatefulResponsesProvider()
    client = AsyncOpenAI(provider=provider)

    first_stream = await client.responses.create(
        model="test-model",
        input="first",
        stream=True,
    )
    first_events = [event async for event in first_stream]
    first_id = first_events[-1].response.id

    second = await client.responses.create(
        model="test-model",
        input="second",
        previous_response_id=first_id,
    )

    assert first_id == "resp_agent_rt_1"
    assert second.output_text == "seen:first|second"


async def test_openai_previous_response_id_rejects_unknown_or_evicted_state() -> None:
    provider = StatefulResponsesProvider()
    client = AsyncOpenAI(provider=provider, max_response_states=1)

    first = await client.responses.create(model="test-model", input="first")
    await client.responses.create(model="test-model", input="second")

    for response_id in (first.id, "resp_missing"):
        try:
            await client.responses.create(
                model="test-model",
                input="next",
                previous_response_id=response_id,
            )
        except ValueError as exc:
            assert "previous_response_id" in str(exc)
        else:
            raise AssertionError("unknown continuation state should fail")


def test_sync_openai_previous_response_id_continues_state() -> None:
    provider = StatefulResponsesProvider()
    client = OpenAI(provider=provider)

    first = client.responses.create(model="test-model", input="first")
    second = client.responses.create(
        model="test-model",
        input="second",
        previous_response_id=first.id,
    )

    assert second.output_text == "seen:first|second"


class FileSearchExecutionProvider:
    name = "file-search-fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            tool = request.tools[0]
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id="file-search-1",
                            name=tool.name,
                            arguments={"query": "deployment guide"},
                        ),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="Found the deployment guide."),),
            ),
            model=request.model,
            finish_reason="stop",
        )


class FakeFileRetrieval:
    kind = "file"

    def __init__(self, results) -> None:
        self.results = tuple(results)
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.results


def _file_search_registry():
    first = FakeFileRetrieval(
        (
            RetrievalResult(
                id="file-a",
                title="deploy.md",
                content="Deployment instructions A",
                score=0.92,
                uri="file:///deploy.md",
                metadata={"team": "platform"},
            ),
            RetrievalResult(
                id="file-low",
                title="old.md",
                content="Old instructions",
                score=0.2,
            ),
        )
    )
    second = FakeFileRetrieval(
        (
            RetrievalResult(
                id="file-b",
                title="release.md",
                content="Deployment instructions B",
                score=0.81,
                metadata={"team": "release"},
            ),
        )
    )
    registry = RetrievalRegistry()
    registry.register("vs_docs", first)
    registry.register("vs_release", second)
    return registry, first, second


async def test_openai_file_search_uses_agent_rt_retrieval() -> None:
    provider = FileSearchExecutionProvider()
    registry, first, second = _file_search_registry()
    client = AsyncOpenAI(provider=provider, retrieval_registry=registry)

    response = await client.responses.create(
        model="test-model",
        input="Find the deployment guide.",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["vs_docs", "vs_release"],
                "max_num_results": 2,
                "filters": {"team": "platform"},
                "ranking_options": {"score_threshold": 0.5},
            }
        ],
    )

    assert provider.requests[0].tools[0].name == "agent_rt_file_search"
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1].role == "tool"
    assert "file-a" in provider.requests[1].messages[-1].content[0].text
    assert "file-low" not in provider.requests[1].messages[-1].content[0].text
    assert first.queries == [
        RetrievalQuery(
            text="deployment guide",
            limit=2,
            filters={"team": "platform"},
        )
    ]
    assert second.queries[0].filters == {"team": "platform"}

    search_call = response.output[0]
    assert search_call.type == "file_search_call"
    assert search_call.status == "completed"
    assert search_call.queries == ["deployment guide"]
    assert [result.file_id for result in search_call.results] == ["file-a", "file-b"]
    assert search_call.results[0].filename == "deploy.md"
    assert search_call.results[0].attributes == {"team": "platform"}
    assert response.output_text == "Found the deployment guide."


async def test_openai_file_search_streaming_and_continuation() -> None:
    provider = FileSearchExecutionProvider()
    registry, _first, _second = _file_search_registry()
    client = AsyncOpenAI(provider=provider, retrieval_registry=registry)

    stream = await client.responses.create(
        model="test-model",
        input="Find deployment docs.",
        tools=[{"type": "file_search", "vector_store_ids": ["vs_docs"]}],
        stream=True,
    )
    events = [event async for event in stream]

    assert events[0].type == "response.output_text.delta"
    completed = events[-1].response
    assert completed.output[0].type == "file_search_call"
    assert completed.id == "resp_agent_rt_1"
    assert completed.id in client._response_states


async def test_openai_file_search_requires_retrieval_registry() -> None:
    provider = FileSearchExecutionProvider()
    client = AsyncOpenAI(provider=provider)

    try:
        await client.responses.create(
            model="test-model",
            input="Search files.",
            tools=[{"type": "file_search", "vector_store_ids": ["vs_docs"]}],
        )
    except RuntimeError as exc:
        assert "retrieval_registry" in str(exc)
    else:
        raise AssertionError(
            "file_search should require an Agent RT retrieval registry"
        )
    assert provider.requests == []


class FakeWebSearchHTTPResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeWebSearchHTTPClient:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeWebSearchHTTPResponse(self.payload)


async def test_environment_web_search_provider_selects_tool_and_credentials() -> None:
    tavily_http = FakeWebSearchHTTPClient(
        {
            "results": [
                {
                    "title": "T",
                    "url": "https://t.example",
                    "content": "hit",
                    "score": 0.9,
                }
            ]
        }
    )
    tavily = EnvironmentWebSearchProvider.from_environment(
        {"AGENT_RT_WEB_SEARCH_TOOL": "tavily", "TAVILY_API_KEY": "t-secret"},
        client=tavily_http,
    )
    tavily_results = await tavily.search(RetrievalQuery("agent runtime", limit=3))
    method, url, kwargs = tavily_http.calls[0]
    assert method == "POST"
    assert url == "https://api.tavily.com/search"
    assert kwargs["json"]["api_key"] == "t-secret"
    assert tavily_results[0].uri == "https://t.example"

    brave_http = FakeWebSearchHTTPClient(
        {
            "web": {
                "results": [
                    {
                        "title": "B",
                        "url": "https://b.example",
                        "description": "brave hit",
                    }
                ]
            }
        }
    )
    brave = EnvironmentWebSearchProvider.from_environment(
        {"AGENT_RT_WEB_SEARCH_TOOL": "brave", "AGENT_RT_WEB_SEARCH_TOKEN": "b-secret"},
        client=brave_http,
    )
    await brave.search(RetrievalQuery("agent runtime", limit=2))
    method, _url, kwargs = brave_http.calls[0]
    assert method == "GET"
    assert kwargs["headers"] == {"X-Subscription-Token": "b-secret"}
    assert kwargs["params"]["count"] == 2

    serper_http = FakeWebSearchHTTPClient(
        {
            "organic": [
                {
                    "title": "S",
                    "link": "https://s.example",
                    "snippet": "serper hit",
                    "position": 1,
                }
            ]
        }
    )
    serper = EnvironmentWebSearchProvider.from_environment(
        {
            "AGENT_RT_WEB_SEARCH_TOOL": "serper",
            "AGENT_RT_WEB_SEARCH_API_KEY": "s-secret",
        },
        client=serper_http,
    )
    serper_results = await serper.search(RetrievalQuery("agent runtime", limit=1))
    assert serper_http.calls[0][2]["headers"] == {"X-API-KEY": "s-secret"}
    assert serper_results[0].metadata["position"] == 1


def test_environment_web_search_provider_requires_tool_and_credential() -> None:
    for environment, expected in (
        ({}, "AGENT_RT_WEB_SEARCH_TOOL"),
        ({"AGENT_RT_WEB_SEARCH_TOOL": "tavily"}, "TAVILY_API_KEY"),
    ):
        try:
            EnvironmentWebSearchProvider.from_environment(environment)
        except RuntimeError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("missing web-search configuration should fail closed")


class WebSearchExecutionProvider:
    name = "web-search-fake"

    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            tool = request.tools[0]
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        ToolCall(
                            id="web-search-1",
                            name=tool.name,
                            arguments={"query": "Agent RT latest"},
                        ),
                    ),
                ),
                model=request.model,
                finish_reason="tool_calls",
            )
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="Search complete."),),
            ),
            model=request.model,
            finish_reason="stop",
        )


class FakeWebRetrieval:
    kind = "web"

    def __init__(self) -> None:
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return (
            RetrievalResult(
                id="https://example.com/a",
                title="Result A",
                content="A result",
                score=0.95,
                uri="https://example.com/a",
                metadata={"provider": "fake-web"},
            ),
            RetrievalResult(
                id="https://example.com/b",
                title="Result B",
                content="B result",
                score=0.8,
                uri="https://example.com/b",
                metadata={"provider": "fake-web"},
            ),
        )


async def test_openai_web_search_uses_agent_rt_provider() -> None:
    model = WebSearchExecutionProvider()
    web = FakeWebRetrieval()
    client = AsyncOpenAI(provider=model, web_search_provider=web)

    response = await client.responses.create(
        model="test-model",
        input="Search the web.",
        tools=[{"type": "web_search", "search_context_size": "medium"}],
    )

    assert model.requests[0].tools[0].name == "agent_rt_web_search"
    assert web.queries[0].text == "Agent RT latest"
    assert web.queries[0].filters == {"search_context_size": "medium"}
    assert "https://example.com/a" in model.requests[1].messages[-1].content[0].text
    call = response.output[0]
    assert call.type == "web_search_call"
    assert call.status == "completed"
    assert call.action.type == "search"
    assert call.action.query == "Agent RT latest"
    assert call.action.sources[0].url == "https://example.com/a"
    assert response.output_text == "Search complete."


async def test_anthropic_web_search_uses_agent_rt_provider() -> None:
    model = WebSearchExecutionProvider()
    web = FakeWebRetrieval()
    client = AsyncAnthropic(provider=model, web_search_provider=web)

    response = await client.messages.create(
        model="test-model",
        max_tokens=256,
        messages=[{"role": "user", "content": "Search the web."}],
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
                "allowed_domains": ["example.com"],
            }
        ],
    )

    assert model.requests[0].tools[0].name == "agent_rt_web_search"
    assert web.queries[0].filters == {"allowed_domains": ["example.com"]}
    assert response.content[0].type == "server_tool_use"
    assert response.content[0].name == "web_search"
    assert response.content[1].type == "web_search_tool_result"
    assert response.content[1].content[0].url == "https://example.com/a"
    assert response.content[-1].text == "Search complete."


class FakeBatchProvider:
    name = "batch-fake"

    def __init__(self) -> None:
        self.calls = []

    async def create_batch(self, **kwargs):
        self.calls.append(("create", kwargs))
        return BatchJob(id="batch-1", status="in_progress")

    async def retrieve_batch(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        return BatchJob(id=batch_id, status="completed")

    async def list_batches(self, **kwargs):
        self.calls.append(("list", kwargs))
        return (
            BatchJob(id="batch-1", status="completed"),
            BatchJob(id="batch-2", status="in_progress"),
        )

    async def cancel_batch(self, batch_id):
        self.calls.append(("cancel", batch_id))
        return BatchJob(id=batch_id, status="cancelling")

    async def batch_results(self, batch_id):
        self.calls.append(("results", batch_id))
        return (
            SimpleNamespace(
                custom_id="req-1", result=SimpleNamespace(type="succeeded")
            ),
            SimpleNamespace(custom_id="req-2", result=SimpleNamespace(type="errored")),
        )


async def test_openai_batch_compat_surface() -> None:
    provider = FakeBatchProvider()
    client = AsyncOpenAI(provider=provider)

    created = await client.batches.create(
        input_file_id="file-input",
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"purpose": "test"},
    )
    retrieved = await client.batches.retrieve(created.id)
    page = await client.batches.list(after="batch-0", limit=2)
    cancelled = await client.batches.cancel(created.id)

    assert created.id == "batch-1"
    assert created.object == "batch"
    assert created.status == "in_progress"
    assert retrieved.status == "completed"
    assert [item.id for item in page.data] == ["batch-1", "batch-2"]
    assert page.first_id == "batch-1"
    assert page.last_id == "batch-2"
    assert cancelled.status == "cancelling"
    assert provider.calls[0] == (
        "create",
        {
            "input_file_id": "file-input",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {"purpose": "test"},
            "output_expires_after": None,
        },
    )


async def test_anthropic_message_batches_compat_surface() -> None:
    provider = FakeBatchProvider()
    client = AsyncAnthropic(provider=provider)
    requests = [
        {
            "custom_id": "req-1",
            "params": {
                "model": "test-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
        }
    ]

    created = await client.messages.batches.create(requests=requests)
    retrieved = await client.messages.batches.retrieve(created.id)
    page = await client.messages.batches.list(limit=2)
    cancelled = await client.messages.batches.cancel(created.id)
    decoder = await client.messages.batches.results(created.id)
    results = [item async for item in decoder]

    assert created.type == "message_batch"
    assert created.processing_status == "in_progress"
    assert retrieved.processing_status == "completed"
    assert [item.id for item in page.data] == ["batch-1", "batch-2"]
    assert cancelled.processing_status == "cancelling"
    assert [item.custom_id for item in results] == ["req-1", "req-2"]
    assert provider.calls[0] == (
        "create",
        {"requests": requests, "user_profile_id": None},
    )


def test_sync_batch_compat_surfaces() -> None:
    openai_provider = FakeBatchProvider()
    openai = OpenAI(provider=openai_provider)
    batch = openai.batches.create(
        input_file_id="file-input",
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert openai.batches.retrieve(batch.id).status == "completed"
    assert openai.batches.list(limit=1).data[0].id == "batch-1"
    assert openai.batches.cancel(batch.id).status == "cancelling"

    anthropic_provider = FakeBatchProvider()
    anthropic = Anthropic(provider=anthropic_provider)
    message_batch = anthropic.messages.batches.create(
        requests=[
            {
                "custom_id": "req-1",
                "params": {
                    "model": "test-model",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            }
        ]
    )
    assert (
        anthropic.messages.batches.retrieve(message_batch.id).processing_status
        == "completed"
    )
    assert anthropic.messages.batches.list().data[0].id == "batch-1"
    assert (
        anthropic.messages.batches.cancel(message_batch.id).processing_status
        == "cancelling"
    )
    assert [
        item.custom_id for item in anthropic.messages.batches.results(message_batch.id)
    ] == [
        "req-1",
        "req-2",
    ]


class FakeVectorDBProvider:
    kind = "knowledge"

    def __init__(self, backend: str) -> None:
        self.backend = backend

    async def search(self, query):
        return [
            RetrievalResult(
                id=self.backend,
                title=self.backend,
                content=query.text,
                metadata={"backend": self.backend},
            )
        ]


async def test_vector_db_provider_can_switch_backends_from_environment() -> None:
    registry = VectorDBProviderRegistry()
    configs = []
    registry.register(
        "qdrant",
        lambda config: configs.append(config) or FakeVectorDBProvider("qdrant"),
    )
    registry.register(
        "pinecone",
        lambda config: configs.append(config) or FakeVectorDBProvider("pinecone"),
    )

    qdrant = vector_db_provider_from_environment(
        registry,
        {
            "AGENT_RT_VECTOR_DB": "qdrant",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
            "AGENT_RT_VECTOR_DB_URL": "http://qdrant.test:6333",
        },
    )
    pinecone = vector_db_provider_from_environment(
        registry,
        {
            "AGENT_RT_VECTOR_DB": "pinecone",
            "AGENT_RT_VECTOR_DB_OPTION_NAMESPACE": "tenant-a",
        },
    )

    assert (await qdrant.search(RetrievalQuery("alpha")))[0].metadata["backend"] == "qdrant"
    assert (await pinecone.search(RetrievalQuery("beta")))[0].metadata["backend"] == "pinecone"
    assert configs[0].collection == "docs"
    assert configs[0].url == "http://qdrant.test:6333"
    assert configs[1].options["namespace"] == "tenant-a"


def test_vector_db_environment_configuration_fails_closed() -> None:
    try:
        vector_db_config_from_environment({})
    except RuntimeError as exc:
        assert "AGENT_RT_VECTOR_DB" in str(exc)
    else:
        raise AssertionError("missing vector DB selector should fail closed")

    registry = VectorDBProviderRegistry()
    registry.register("qdrant", lambda config: FakeVectorDBProvider(config.backend))
    try:
        vector_db_provider_from_environment(registry, {"AGENT_RT_VECTOR_DB": "unknown"})
    except ValueError as exc:
        assert "registered backends: qdrant" in str(exc)
    else:
        raise AssertionError("unknown vector DB backend should fail closed")


class FakeVectorDBHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeVectorDBHTTPClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeVectorDBHTTPResponse(self.payload)


class FakeEmbeddingModelProvider:
    async def embed(self, request):
        assert request.input == "agent runtime"
        return EmbeddingResponse(data=[EmbeddingItem(index=0, embedding=[0.1, 0.2, 0.3])])


async def test_popular_vector_db_backends_use_provider_neutral_http_adapters() -> None:
    cases = {
        "qdrant": (
            {"result": [{"id": "q1", "score": 0.9, "payload": {"text": "qdrant hit", "title": "Q"}}]},
            "/collections/docs/points/search",
        ),
        "pinecone": (
            {"matches": [{"id": "p1", "score": 0.8, "metadata": {"text": "pinecone hit", "title": "P"}}]},
            "/query",
        ),
        "milvus": (
            {"data": [{"id": "m1", "distance": 0.7, "text": "milvus hit", "title": "M"}]},
            "/v2/vectordb/entities/search",
        ),
        "weaviate": (
            {"data": {"Get": {"Docs": [{"text": "weaviate hit", "title": "W", "url": "https://w.test", "_additional": {"id": "w1", "certainty": 0.95}}]}}},
            "/v1/graphql",
        ),
        "chroma": (
            {"ids": [["c1"]], "documents": [["chroma hit"]], "metadatas": [[{"title": "C"}]], "distances": [[0.2]], "uris": [["https://c.test"]]},
            "/api/v2/tenants/default_tenant/databases/default_database/collections/docs/query",
        ),
    }
    assert tuple(POPULAR_VECTOR_DB_BACKENDS) == ("chroma", "milvus", "pinecone", "qdrant", "weaviate")

    for backend, (payload, expected_path) in cases.items():
        client = FakeVectorDBHTTPClient(payload)
        collection = "Docs" if backend == "weaviate" else "docs"
        provider = EnvironmentVectorDBProvider.from_environment(
            {
                "AGENT_RT_VECTOR_DB": backend,
                "AGENT_RT_VECTOR_DB_COLLECTION": collection,
                "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
            },
            embedding_provider=FakeEmbeddingModelProvider(),
            client=client,
        )
        results = await provider.search(RetrievalQuery("agent runtime", limit=2))
        assert len(results) == 1
        assert results[0].metadata["provider"] == backend
        assert expected_path in client.calls[0][1]


async def test_popular_vector_db_registry_switches_builtins() -> None:
    client = FakeVectorDBHTTPClient(
        {"result": [{"id": "q1", "score": 0.9, "payload": {"text": "hit"}}]}
    )
    registry = VectorDBProviderRegistry()
    register_popular_vector_db_backends(
        registry,
        embedding_provider=FakeEmbeddingModelProvider(),
        environment={
            "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
        },
        client=client,
    )
    provider = vector_db_provider_from_environment(
        registry,
        {
            "AGENT_RT_VECTOR_DB": "qdrant",
            "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
        },
    )
    results = await provider.search(RetrievalQuery("agent runtime"))
    assert results[0].id == "q1"


async def test_vector_db_provider_accepts_precomputed_vector_without_embedder() -> None:
    client = FakeVectorDBHTTPClient({"matches": []})
    provider = EnvironmentVectorDBProvider.from_environment(
        {
            "AGENT_RT_VECTOR_DB": "pinecone",
            "AGENT_RT_VECTOR_DB_COLLECTION": "docs",
            "AGENT_RT_VECTOR_DB_URL": "https://vector.example",
        },
        client=client,
    )
    await provider.search(
        RetrievalQuery("agent runtime", filters={"vector": [0.1, 0.2], "category": "runtime"})
    )
    body = client.calls[0][2]["json"]
    assert body["vector"] == [0.1, 0.2]
    assert body["filter"] == {"category": "runtime"}

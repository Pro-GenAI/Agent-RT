"""Compatibility shapes found by migration field-testing third-party repositories."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from agent_rt.anthropic import AsyncAnthropic
from agent_rt.anthropic.types import Message, MessageParam, TextBlock

from agent_rt import (
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelUsage,
    ToolCall,
    anthropic,
    openai,
)


class _Provider:
    def __init__(self, *, tool_calls=(), finish_reason="stop", error=None):
        self.requests = []
        self.tool_calls = tuple(tool_calls)
        self.finish_reason = finish_reason
        self.error = error

    async def complete(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=(ContentPart(type="text", text="ok"),),
                tool_calls=self.tool_calls,
            ),
            model=request.model,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason=self.finish_reason,
        )

    async def embed(self, request):
        from agent_rt import EmbeddingItem, EmbeddingResponse

        self.requests.append(request)
        return EmbeddingResponse(
            data=tuple(
                EmbeddingItem(index=index, embedding=(float(index), 1.0))
                for index, _ in enumerate(request.input)
            ),
            model=request.model,
        )


def _response(status):
    return SimpleNamespace(
        status_code=status, headers={"retry-after": "7"}, request=None
    )


def test_sdk_error_hierarchies_match_upstream_signatures() -> None:
    limited = anthropic.RateLimitError("slow down", response=_response(429), body=None)
    assert isinstance(limited, anthropic.APIStatusError)
    assert isinstance(limited, anthropic.APIError)
    assert isinstance(limited, anthropic.AnthropicError)
    assert limited.status_code == 429
    assert limited.response.headers["retry-after"] == "7"
    assert str(limited) == limited.message == "slow down"

    connection = anthropic.APIConnectionError(message="boom", request=None)
    assert isinstance(connection, anthropic.APIError)
    assert isinstance(
        anthropic.APITimeoutError(request=None), anthropic.APIConnectionError
    )
    assert issubclass(anthropic.OverloadedError, anthropic.InternalServerError)

    # Each vendor owns its hierarchy, as in the SDKs.
    assert openai.APIError is not anthropic.APIError
    assert issubclass(openai.RateLimitError, openai.OpenAIError)


async def test_provider_http_failures_raise_vendor_errors() -> None:
    failure = RuntimeError("HTTP 401 from endpoint")
    failure.status_code = 401
    client = AsyncAnthropic(provider=_Provider(error=failure))
    with pytest.raises(anthropic.AuthenticationError) as caught:
        await client.messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
        )
    assert caught.value.status_code == 401
    assert caught.value.__cause__ is failure

    # Errors without an HTTP status are not disguised as API errors.
    client = AsyncAnthropic(provider=_Provider(error=ValueError("bad input")))
    with pytest.raises(ValueError):
        await client.messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
        )


async def test_anthropic_messages_use_types_and_stop_reasons() -> None:
    call = ToolCall(id="call-1", name="lookup", arguments={"q": "x"})
    provider = _Provider(tool_calls=(call,), finish_reason="tool_calls")
    client = AsyncAnthropic(provider=provider)
    history: list[MessageParam] = [{"role": "user", "content": "hi"}]

    message = await client.messages.create(model="m", max_tokens=8, messages=history)

    assert isinstance(message, Message)
    assert message.stop_reason == "tool_use"
    assert isinstance(message.content[0], TextBlock)
    assert message.model_dump()["content"][1]["name"] == "lookup"

    # Response blocks go straight back into history, as with the SDK.
    history.append({"role": "assistant", "content": message.content})
    await client.messages.create(model="m", max_tokens=8, messages=history)
    replayed = provider.requests[-1].messages[1]
    assert replayed.tool_calls[0].name == "lookup"
    assert replayed.content[0].text == "ok"

    plain = await AsyncAnthropic(provider=_Provider()).messages.create(
        model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )
    assert plain.stop_reason == "end_turn"


def test_langchain_v1_submodule_paths() -> None:
    from agent_rt.langchain.agents import create_agent
    from agent_rt.langchain.chat_models import init_chat_model
    from agent_rt.langchain.chat_models.base import _ConfigurableModel
    from agent_rt.langchain.tools import tool
    from agent_rt.langchain_openai import ChatOpenAI

    assert _ConfigurableModel is ChatOpenAI
    assert callable(create_agent) and callable(tool)
    model = init_chat_model(model="openai:test-model", provider=_Provider())
    assert isinstance(model, _ConfigurableModel)


def test_langchain_openai_embeddings() -> None:
    from agent_rt.langchain_openai import OpenAIEmbeddings

    provider = _Provider()
    embeddings = OpenAIEmbeddings(model="embed-model", provider=provider)
    assert embeddings.embed_documents(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert embeddings.embed_query("a") == [0.0, 1.0]
    assert provider.requests[0].model == "embed-model"


def test_llama_index_core_module_paths() -> None:
    from agent_rt.llama_index.core import Document, Settings, VectorStoreIndex
    from agent_rt.llama_index.core.base.embeddings.base import BaseEmbedding
    from agent_rt.llama_index.core.base.llms.types import CompletionResponse
    from agent_rt.llama_index.core.node_parser import SentenceSplitter
    from agent_rt.llama_index.core.schema import TextNode
    from agent_rt.llama_index.core.workflow import StartEvent, Workflow
    from agent_rt.llama_index.embeddings.openai import OpenAIEmbedding
    from agent_rt.llama_index.llms.ollama import Ollama

    from agent_rt import llama_index

    assert llama_index.core.node_parser.SentenceSplitter is SentenceSplitter
    assert all(
        item is not None
        for item in (Document, Settings, VectorStoreIndex, CompletionResponse)
    )
    assert all(item is not None for item in (TextNode, StartEvent, Workflow))

    provider = _Provider()
    embedding = OpenAIEmbedding(model="embed-model", provider=provider)
    assert isinstance(embedding, BaseEmbedding)
    assert embedding.get_text_embedding_batch(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert embedding.get_query_embedding("q") == [0.0, 1.0]

    ollama = Ollama("llama3", base_url="http://ollama.test:11434")
    assert ollama.provider.settings.base_url == "http://ollama.test:11434/v1"


def test_autogen_user_proxy_and_conditions() -> None:
    from agent_rt.autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from agent_rt.autogen_agentchat.base import Response
    from agent_rt.autogen_agentchat.conditions import (
        MaxMessageTermination,
        TimeoutTermination,
        TokenUsageTermination,
    )
    from agent_rt.autogen_agentchat.messages import TextMessage
    from agent_rt.autogen_agentchat.teams import RoundRobinGroupChat
    from agent_rt.autogen_ext.models.openai import OpenAIChatCompletionClient

    async def answer(prompt, cancellation_token):
        assert cancellation_token == "token"
        return "approve"

    proxy = UserProxyAgent("Human", input_func=answer)
    response = asyncio.run(
        proxy.on_messages([TextMessage(content="ok?", source="bot")], "token")
    )
    assert isinstance(response, Response)
    assert response.chat_message.content == "approve"
    assert response.chat_message.source == "Human"

    sync_proxy = UserProxyAgent("Human", input_func=lambda prompt: "yes")
    assert asyncio.run(sync_proxy.on_messages([])).chat_message.content == "yes"

    usage = SimpleNamespace(prompt_tokens=6, completion_tokens=5)
    tokens = TokenUsageTermination(max_total_token=10)
    assert tokens.should_stop([TextMessage("a", "b", models_usage=usage)])
    condition = MaxMessageTermination(5) | tokens | TimeoutTermination(5)
    team = RoundRobinGroupChat(
        [AssistantAgent("a", OpenAIChatCompletionClient(model="m"))],
        termination_condition=condition,
    )
    config = json.dumps(team.dump_component().config["termination_condition"])
    assert '"max_total_token": 10' in config and '"timeout_seconds": 5' in config


def test_autogen_messages_dump_and_load() -> None:
    from agent_rt.autogen_agentchat.messages import TextMessage

    message = TextMessage(content="hello", source="Critic")
    assert message.dump()["source"] == "Critic"
    assert TextMessage.load(message.dump()) == message
    assert message.to_text() == "hello"


def test_crewai_llm_routes_litellm_style_names() -> None:
    from agent_rt.crewai import LLM, Agent

    provider = _Provider()
    llm = LLM("anthropic/claude-test", provider=provider, temperature=0.2)
    assert llm.vendor == "anthropic"
    assert llm.call("hello") == "ok"
    assert provider.requests[0].model == "claude-test"
    assert provider.requests[0].temperature == 0.2
    assert LLM("gpt-test").vendor == "openai"

    agent = Agent(role="r", goal="g", llm=llm)._agent()
    assert agent.model == "claude-test"
    assert agent.provider is provider


def test_llama_index_injected_chroma_collection_and_index_embedding(tmp_path) -> None:
    from agent_rt.llama_index.core import Settings, StorageContext, VectorStoreIndex
    from agent_rt.llama_index.core.schema import TextNode
    from agent_rt.llama_index.vector_stores.chroma import ChromaVectorStore

    class Collection:
        def __init__(self):
            self.rows = []

        def add(self, ids, embeddings, documents, metadatas):
            self.rows.extend(zip(ids, embeddings, documents, metadatas, strict=True))

        def query(self, query_embeddings, n_results, **_):
            rows = self.rows[:n_results]
            return {
                "ids": [[row[0] for row in rows]],
                "documents": [[row[2] for row in rows]],
                "metadatas": [[row[3] for row in rows]],
                "distances": [[0.0 for _ in rows]],
            }

    class Embedding:
        def get_text_embedding_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def aget_query_embedding(self, query):
            return [1.0, 0.0]

    previous = Settings.embed_model
    Settings.embed_model = Embedding()
    try:
        collection = Collection()
        store = ChromaVectorStore(chroma_collection=collection)
        context = StorageContext.from_defaults(vector_store=store)
        VectorStoreIndex(
            [TextNode(text="policy", metadata={"file_name": "a.pdf"})],
            storage_context=context,
            show_progress=True,
        )
        assert collection.rows[0][1] == [1.0, 0.0]
        index = VectorStoreIndex.from_vector_store(store, storage_context=context)
        [hit] = index.as_retriever(similarity_top_k=3).retrieve("rules")
        assert (hit.get_text(), hit.metadata["file_name"], hit.score) == (
            "policy",
            "a.pdf",
            1.0,
        )
    finally:
        Settings.embed_model = previous


def test_llama_index_directory_reader_and_property_graph_stand_ins(tmp_path) -> None:
    from agent_rt.llama_index.core import (
        PropertyGraphIndex,
        SimpleDirectoryReader,
        load_index_from_storage,
    )
    from agent_rt.llama_index.core.indices.property_graph import CustomPGRetriever

    with pytest.raises(ValueError, match="No files found"):
        SimpleDirectoryReader(str(tmp_path), recursive=True)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / ".hidden").write_text("skip", encoding="utf-8")
    [document] = SimpleDirectoryReader(
        str(tmp_path), recursive=True, filename_as_id=True
    ).load_data()
    assert document.text == "alpha"
    assert document.id_ == str(tmp_path / "notes" / "a.md")
    assert document.metadata["file_name"] == "a.md"

    class Retriever(CustomPGRetriever):  # subclassing works at import time
        pass

    for build in (
        Retriever,
        PropertyGraphIndex.from_documents,
        load_index_from_storage,
    ):
        with pytest.raises(NotImplementedError):
            build()


def test_openai_agents_default_tool_names_match_sdk() -> None:
    # Field shape: models call SDK-normalized names, so agent names with
    # spaces or punctuation must map exactly like `transform_string_function_style`.
    from agent_rt.agents import Agent, handoff

    from ext.compat.openai_agents import _handoff_tool

    billing = Agent(name="Billing Agent-EU", instructions="b")
    state = {"last_agent": billing}
    assert _handoff_tool(billing, state).name == "transfer_to_billing_agent_eu"
    assert (
        _handoff_tool(handoff(billing, tool_name_override="to-billing"), state).name
        == "to-billing"
    )
    assert billing.as_tool().name == "billing_agent_eu"
    assert billing.as_tool(tool_name="Keep-Me").name == "Keep-Me"


def test_openai_agents_package_submodules_and_setup_helpers(monkeypatch) -> None:
    # Field shape (multi-agent travel planner): `from agents.exceptions import
    # MaxTurnsExceeded`, `from agents.mcp import ...`, `trace(...)`, and
    # `set_default_openai_key(...)` at import time.
    from agent_rt.agents import (
        RunContextWrapper,
        RunState,
        set_default_openai_key,
        trace,
    )
    from agent_rt.agents.exceptions import AgentsException, MaxTurnsExceeded
    from agent_rt.agents.mcp import MCPServerStreamableHttp

    assert issubclass(MaxTurnsExceeded, AgentsException)
    with trace("travel-plan", metadata={"k": "v"}):
        pass
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    set_default_openai_key(None)
    assert "OPENAI_API_KEY" not in __import__("os").environ
    set_default_openai_key("sk-field")
    assert __import__("os").environ["OPENAI_API_KEY"] == "sk-field"
    assert RunContextWrapper(context={"a": 1}).usage.requests == 0
    for unsupported in (RunState, MCPServerStreamableHttp):
        with pytest.raises(NotImplementedError):
            unsupported()


async def test_openai_agents_input_guardrail_tripwire_stops_before_model() -> None:
    from agent_rt.agents import (
        Agent,
        GuardrailFunctionOutput,
        InputGuardrailTripwireTriggered,
        Runner,
        input_guardrail,
    )

    seen = []

    @input_guardrail
    async def travel_only(ctx, agent, user_input):
        seen.append((ctx.context, agent.name, user_input))
        return GuardrailFunctionOutput(
            output_info={"ok": False}, tripwire_triggered="trip" not in user_input
        )

    provider = _Provider()
    agent = Agent(name="planner", provider=provider, input_guardrails=[travel_only])
    with pytest.raises(InputGuardrailTripwireTriggered) as raised:
        await Runner.run(agent, "write my essay", context="ctx")
    assert raised.value.guardrail_result.output.output_info == {"ok": False}
    assert provider.requests == []  # tripwire fires before any model call
    assert seen == [("ctx", "planner", "write my essay")]

    result = await Runner.run(agent, "plan a trip")
    assert result.final_output == "ok"


async def test_openai_agents_output_guardrail_and_max_turns() -> None:
    from agent_rt.agents import (
        Agent,
        GuardrailFunctionOutput,
        OutputGuardrailTripwireTriggered,
        Runner,
        function_tool,
        output_guardrail,
    )
    from agent_rt.agents.exceptions import MaxTurnsExceeded

    @output_guardrail
    def no_ok(ctx, agent, output):
        return GuardrailFunctionOutput(tripwire_triggered=output == "ok")

    agent = Agent(name="writer", provider=_Provider(), output_guardrails=[no_ok])
    with pytest.raises(OutputGuardrailTripwireTriggered) as raised:
        await Runner.run(agent, "hi")
    assert raised.value.guardrail_result.agent_output == "ok"

    @function_tool
    def lookup() -> str:
        return "again"

    looping = Agent(
        name="looper",
        tools=[lookup],
        provider=_Provider(
            tool_calls=(ToolCall(id="c1", name="lookup", arguments={}),),
            finish_reason="tool_calls",
        ),
    )
    # The SDK raises rather than returning a partial result.
    with pytest.raises(MaxTurnsExceeded, match="Max turns \\(2\\) exceeded"):
        await Runner.run(looping, "go", max_turns=2)


class _TextProvider(_Provider):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def complete(self, request):
        response = await super().complete(request)
        return ModelResponse(
            message=ModelMessage(
                role="assistant", content=(ContentPart(type="text", text=self.text),)
            ),
            model=response.model,
            usage=response.usage,
            finish_reason="stop",
        )


def test_openai_types_and_resources_submodules() -> None:
    # Field shapes: SDK test doubles build `ChatCompletion(...)` from
    # `openai.types.*`, and tests patch `openai.resources.responses.Responses`.
    from agent_rt.openai.resources.chat.completions import Completions
    from agent_rt.openai.resources.responses import AsyncResponses, Responses
    from agent_rt.openai.types import CreateEmbeddingResponse
    from agent_rt.openai.types.chat import ChatCompletion, ChatCompletionMessage
    from agent_rt.openai.types.chat.chat_completion import Choice
    from agent_rt.openai.types.chat.chat_completion_chunk import ChoiceDelta
    from agent_rt.openai.types.completion_usage import CompletionUsage
    from agent_rt.openai.types.responses import ParsedResponse, ResponseInputParam

    completion = ChatCompletion(
        id="c1",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    assert completion.model_dump()["choices"][0]["message"]["content"] == "hi"
    assert ChoiceDelta(content="x").content == "x"
    assert ResponseInputParam == list[Any]
    assert issubclass(ParsedResponse, openai.types.responses.Response)
    assert CreateEmbeddingResponse.model_validate({"data": []}).data == []

    client = openai.OpenAI(provider=_Provider())
    assert isinstance(client.responses, Responses)
    assert isinstance(client.chat.completions, Completions)
    assert isinstance(
        openai.AsyncOpenAI(provider=_Provider()).responses, AsyncResponses
    )
    assert isinstance(
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "q"}]
        ),
        ChatCompletion,
    )


def test_openai_client_options_and_lifecycle() -> None:
    # Field shapes: tests assert `client.timeout` / `client.max_retries`, and
    # adapters call `await client.close()` or use the client as a context manager.
    client = openai.AsyncOpenAI(provider=_Provider(), timeout=12.5, max_retries=0)
    assert (client.timeout, client.max_retries) == (12.5, 0)

    async def lifecycle():
        async with client as entered:
            assert entered is client
        await client.close()

    asyncio.run(lifecycle())
    with openai.OpenAI(provider=_Provider()) as sync_client:
        assert (sync_client.timeout, sync_client.max_retries) == (600.0, 2)
    sync_client.close()


def test_openai_parse_returns_validated_models() -> None:
    from pydantic import BaseModel

    class Ticket(BaseModel):
        department: str
        priority: int

    payload = '{"department": "billing", "priority": 2}'
    provider = _TextProvider(payload)
    client = openai.OpenAI(provider=provider)
    parsed = client.responses.parse(model="m", input="classify", text_format=Ticket)
    assert isinstance(parsed, openai.types.responses.ParsedResponse)
    assert parsed.output_parsed == Ticket(department="billing", priority=2)
    assert provider.requests[-1].structured_output is not None

    completion = client.chat.completions.parse(
        model="m", messages=[{"role": "user", "content": "q"}], response_format=Ticket
    )
    assert completion.choices[0].message.parsed.priority == 2
    with pytest.raises(TypeError):
        client.chat.completions.parse(model="m", messages=[], response_format=dict)


def test_openai_client_routes_through_injected_http_client(monkeypatch) -> None:
    # Field shape: tests build `OpenAI(http_client=httpx.Client(transport=
    # httpx.MockTransport(handler)))` to stay offline; ignoring http_client
    # sent those requests to the configured endpoint instead.
    import httpx

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        if seen["body"].get("stream"):
            chunks = [
                {
                    "choices": [
                        {"index": 0, "delta": {"content": "par"}, "finish_reason": None}
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "tial"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ]
            body = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        if seen["body"]["model"] == "limited":
            return httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"retry-after": "0"},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok": 1}'},
                    }
                ],
            },
        )

    client = openai.OpenAI(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    completion = client.chat.completions.create(
        model="test-model",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert completion.choices[0].message.content == '{"ok": 1}'
    assert seen["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert seen["authorization"] == "Bearer sk-test"
    assert seen["body"]["response_format"] == {"type": "json_object"}

    stream = client.chat.completions.create(
        model="test-model", stream=True, messages=[{"role": "user", "content": "hi"}]
    )
    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "partial"
    )

    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="limited", messages=[{"role": "user", "content": "hi"}]
        )

    async def run_async():
        async_client = openai.AsyncOpenAI(
            api_key="sk-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = await async_client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "x"}]
        )
        return result.choices[0].message.content

    assert asyncio.run(run_async()) == '{"ok": 1}'
    error = None
    try:
        client.chat.completions.create(
            model="limited", messages=[{"role": "user", "content": "hi"}]
        )
    except openai.RateLimitError as caught:
        error = caught
    assert error.status_code == 429
    assert error.response.headers["retry-after"] == "0"
    assert error.body == {"error": {"message": "slow down"}}


def test_openai_private_patch_targets_resolve(monkeypatch) -> None:
    # Field shape: mocking plugins patch `openai._base_client.SyncAPIClient.request`
    # and resource methods by dotted path, and fixtures assert `client.is_closed()`.
    from unittest.mock import patch

    from agent_rt.openai._base_client import AsyncAPIClient, SyncAPIClient

    client = openai.OpenAI(provider=_Provider())
    assert isinstance(client, SyncAPIClient)
    assert isinstance(openai.AsyncOpenAI(provider=_Provider()), AsyncAPIClient)
    with patch(
        "agent_rt.openai._base_client.SyncAPIClient.request",
        new=lambda *a, **k: "blocked",
    ):
        assert client.request() == "blocked"
    with pytest.raises(NotImplementedError):
        client.request()

    def fake_create(self, **kwargs):
        return "patched"

    with patch(
        "agent_rt.openai.resources.chat.completions.Completions.create", new=fake_create
    ):
        assert client.chat.completions.create(model="m", messages=[]) == "patched"
    assert not client.is_closed()
    client.close()
    assert client.is_closed()


def test_openai_parsed_types_are_subscriptable() -> None:
    # Field shape: test doubles build `ParsedResponseOutputText[Model](...)`.
    from agent_rt.openai.types.responses import ParsedResponse, ParsedResponseOutputText

    part = ParsedResponseOutputText[dict](type="output_text", text="{}", parsed={})
    assert isinstance(part, ParsedResponseOutputText)
    assert ParsedResponse[dict] is ParsedResponse


def test_openai_response_output_text_is_computed_from_output() -> None:
    # Field shape: SDK test doubles build ParsedResponse(output=[...]) and read
    # `output_text`, which the SDK derives from the output message parts.
    from agent_rt.openai.types.responses import (
        ParsedResponse,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    response = ParsedResponse(
        output=[
            ResponseOutputMessage(
                type="message",
                content=[
                    ResponseOutputText(type="output_text", text='{"a":'),
                    ResponseOutputText(type="output_text", text="1}"),
                ],
            )
        ]
    )
    assert response.output_text == '{"a":1}'
    assert response.output_parsed is None
    parsed = ParsedResponse(
        output=[
            ResponseOutputMessage(
                type="message",
                content=[ResponseOutputText(type="output_text", text="{}", parsed={"a": 1})],
            )
        ]
    )
    assert parsed.output_parsed == {"a": 1}
    assert "output_text" not in response.model_dump()
    assert ParsedResponse(output=[], output_text="set").output_text == "set"
    with pytest.raises(AttributeError):
        _ = response.missing

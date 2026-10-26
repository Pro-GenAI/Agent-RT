from __future__ import annotations

import asyncio
import threading

import pytest
from agent_rt.deepagents import create_deep_agent
from agent_rt.langchain_openai import ChatOpenAI
from agent_rt.litellm import RateLimitError, acompletion, completion
from agent_rt.openrouter import OpenRouter

from agent_rt import (
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)


class QueueProvider:
    name = "queue"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected model request")
        return self.responses.pop(0)


def response(text="", *calls):
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=(ContentPart(type="text", text=text),) if text else (),
            tool_calls=tuple(
                ToolCall(id=f"call-{i}-{name}", name=name, arguments=args)
                for i, (name, args) in enumerate(calls)
            ),
        ),
        model="test-model",
        finish_reason="tool_calls" if calls else "stop",
    )


async def test_deep_agent_plans_edits_virtual_files_and_delegates():
    provider = QueueProvider(
        [
            response(
                "",
                (
                    "write_todos",
                    {"todos": [{"content": "draft", "status": "in_progress"}]},
                ),
                ("write_file", {"file_path": "/notes/a.txt", "content": "hello world"}),
            ),
            response(
                "", ("task", {"description": "fix it", "subagent_type": "editor"})
            ),
            # The editor subagent shares the virtual filesystem.
            response(
                "",
                (
                    "edit_file",
                    {
                        "file_path": "/notes/a.txt",
                        "old_string": "world",
                        "new_string": "there",
                    },
                ),
            ),
            response("edited"),
            response(
                "",
                ("grep", {"pattern": "there", "output_mode": "content"}),
                ("ls", {"path": "/"}),
                ("read_file", {"file_path": "/in.md"}),
                ("glob", {"pattern": "**/*.txt"}),
            ),
            response("done"),
        ]
    )
    agent = create_deep_agent(
        model=ChatOpenAI("test-model", provider=provider),
        system_prompt="Be brief.",
        subagents=[
            {"name": "editor", "description": "Edits files.", "system_prompt": "Edit."}
        ],
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "go"}], "files": {"/in.md": "input"}}
    )

    assert result["messages"][-1].content == "done"
    assert result["files"]["/notes/a.txt"]["content"] == "hello there"
    assert result["files"]["/in.md"] == "input"
    assert result["todos"] == [{"content": "draft", "status": "in_progress"}]
    outputs = [
        m.content for m in result["messages"] if getattr(m, "role", "") == "tool"
    ]
    assert "edited" in outputs
    assert "/notes/a.txt:1:hello there" in outputs
    assert '["/in.md", "/notes/"]' in outputs
    assert "     1\tinput" in outputs
    assert '["/notes/a.txt"]' in outputs
    first = provider.requests[0]
    assert {t.name for t in first.tools} >= {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "task",
    }
    assert "Be brief." in first.messages[0].content[0].text
    # The subagent sees only the delegated task, not the parent conversation.
    assert [m.role for m in provider.requests[2].messages] == ["system", "user"]


async def test_deep_agent_file_tools_report_errors_instead_of_overwriting():
    provider = QueueProvider(
        [
            response(
                "",
                ("write_file", {"file_path": "/a", "content": "new"}),
                (
                    "edit_file",
                    {"file_path": "/a", "old_string": "x", "new_string": "y"},
                ),
                ("task", {"description": "x", "subagent_type": "missing"}),
            ),
            response("ok"),
        ]
    )
    agent = create_deep_agent(model=ChatOpenAI("m", provider=provider))
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "go"}], "files": {"/a": "x x"}}
    )

    outputs = [
        m.content for m in result["messages"] if getattr(m, "role", "") == "tool"
    ]
    assert outputs[0].startswith("Cannot write to /a because it already exists")
    assert "appears 2 times" in outputs[1]
    assert "unknown subagent_type 'missing'" in outputs[2]
    assert result["files"] == {"/a": "x x"}


def test_deep_agent_rejects_unemulated_options():
    with pytest.raises(ValueError, match="requires model"):
        create_deep_agent()
    with pytest.raises(NotImplementedError, match="backend"):
        create_deep_agent(
            model=ChatOpenAI("m", provider=QueueProvider([])), backend=object()
        )


@pytest.mark.parametrize(
    "option", ["permissions", "mode", "graph", "graph_id", "runnable"]
)
def test_deep_agent_rejects_unenforced_subagent_options(option):
    model = ChatOpenAI("m", provider=QueueProvider([]))
    with pytest.raises(NotImplementedError, match=option):
        create_deep_agent(
            model=model,
            subagents=[
                {"name": "worker", "description": "Work", option: {"deny": True}}
            ],
        )


@pytest.mark.parametrize("option", ["timeout", "request_timeout", "num_retries"])
def test_litellm_rejects_unenforced_request_options(option):
    provider = QueueProvider([])
    with pytest.raises(NotImplementedError, match=option):
        completion(
            model="openai/test",
            messages=[{"role": "user", "content": "hi"}],
            provider=provider,
            **{option: 1},
        )
    assert provider.requests == []


def test_litellm_accepts_request_options_that_keep_default_behavior():
    provider = QueueProvider([response("hi")])
    completion(
        model="openai/test",
        messages=[{"role": "user", "content": "hi"}],
        provider=provider,
        num_retries=0,
        drop_params=False,
        timeout=None,
    )
    assert len(provider.requests) == 1


def test_litellm_rejects_unhonored_chat_options_unless_drop_params():
    provider = QueueProvider([response("hi")])
    for option in ("tool_choice", "top_p", "stop"):
        with pytest.raises(NotImplementedError, match=option):
            completion(
                model="openai/test", messages=[], provider=provider, **{option: "x"}
            )
    assert provider.requests == []
    completion(
        model="openai/test", messages=[], provider=provider, drop_params=True, top_p=0.1
    )
    assert len(provider.requests) == 1


def test_openrouter_rejects_unenforced_routing_and_generation_options():
    provider = QueueProvider([])
    client = OpenRouter(provider=provider)
    for option in ("provider", "tool_choice", "top_p", "plugins"):
        with pytest.raises(NotImplementedError, match=option):
            client.chat.send(model="x", messages=[], **{option: {"only": ["x"]}})
    assert provider.requests == []


def test_litellm_completion_routes_through_agent_rt_with_dict_access():
    provider = QueueProvider([response("hi")])

    result = completion(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "x"}],
        provider=provider,
    )

    assert result.choices[0].message.content == "hi"
    assert result["choices"][0]["message"]["content"] == "hi"
    assert provider.requests[0].model == "gpt-test"
    with pytest.raises(KeyError):
        result["missing"]


async def test_litellm_acompletion_and_provider_prefixes(monkeypatch):
    provider = QueueProvider([response("async hi")])
    result = await acompletion(
        "openrouter/anthropic/claude-x",
        [{"role": "user", "content": "x"}],
        provider=provider,
    )
    assert result.choices[0].message.content == "async hi"
    assert provider.requests[0].model == "anthropic/claude-x"

    with pytest.raises(ValueError, match="does not support provider 'gemini'"):
        completion(model="gemini/gemini-pro", messages=[])
    assert issubclass(RateLimitError, Exception)


def test_litellm_and_openrouter_sync_streams_are_incremental():
    class StreamingProvider:
        name = "streaming"

        def __init__(self):
            self.released = threading.Event()
            self.closed = threading.Event()

        async def stream(self, request):
            try:
                yield ModelStreamEvent(type="text_delta", text="first")
                await asyncio.to_thread(self.released.wait)
                yield ModelStreamEvent(type="text_delta", text="second")
                yield ModelStreamEvent(
                    type="completed", response=response("firstsecond")
                )
            finally:
                self.closed.set()

    for make_stream in (
        lambda provider: completion(
            model="openai/test", messages=[], provider=provider, stream=True
        ),
        lambda provider: OpenRouter(provider=provider).chat.send(
            model="test", messages=[], stream=True
        ),
    ):
        provider = StreamingProvider()
        # Bound a regression failure: the old buffering implementation would
        # block on the release event instead of returning the first chunk.
        fallback = threading.Timer(1.0, provider.released.set)
        fallback.start()
        stream = make_stream(provider)
        try:
            with stream as chunks:
                first = next(chunks)
                assert first.choices[0].delta.content == "first"
                assert not provider.released.is_set()
                provider.released.set()
                assert any(
                    chunk.choices[0].delta.content == "second" for chunk in chunks
                )
            assert provider.closed.wait(1)
            with pytest.raises(StopIteration):
                next(stream)
        finally:
            provider.released.set()
            stream.close()
            fallback.cancel()

        # Early context-manager exit must close an unfinished async generator.
        partial_provider = StreamingProvider()
        with make_stream(partial_provider) as partial:
            assert next(partial).choices[0].delta.content == "first"
        assert partial_provider.closed.wait(1)
        assert not partial_provider.released.is_set()


def test_litellm_abandoned_sync_stream_closes_its_loop_thread():
    import gc

    closed = threading.Event()

    class StreamingProvider:
        name = "streaming"

        async def stream(self, request):
            try:
                yield ModelStreamEvent(type="text_delta", text="first")
                yield ModelStreamEvent(type="completed", response=response("first"))
            finally:
                closed.set()

    stream = completion(
        model="openai/test", messages=[], provider=StreamingProvider(), stream=True
    )
    next(stream)
    thread = stream._thread
    del stream
    gc.collect()
    assert closed.wait(1)
    thread.join(1)
    assert not thread.is_alive()


def test_openrouter_key_never_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = OpenRouter().provider.settings
    assert settings.api_key is None
    assert settings.base_url.startswith("https://openrouter.ai/api/v1")

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    assert OpenRouter().provider.settings.api_key == "router-secret"


def test_openrouter_sdk_chat_send():
    provider = QueueProvider([response("sync")])
    with OpenRouter(provider=provider) as client:
        sent = client.chat.send(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.4,
            reasoning_effort="low",
        )
    assert sent.choices[0].message.content == "sync"
    assert provider.requests[0].model == "openai/gpt-5"
    assert provider.requests[0].temperature == 0.4
    assert provider.requests[0].reasoning.effort == "low"


async def test_openrouter_sdk_chat_send_async():
    provider = QueueProvider([response("async")])
    async with OpenRouter(provider=provider) as client:
        sent = await client.chat.send_async(
            model="openai/gpt-5", messages=[{"role": "user", "content": "y"}]
        )
    assert sent.choices[0].message.content == "async"

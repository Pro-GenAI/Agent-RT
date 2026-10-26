import asyncio
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_rt import (
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
)
from agent_rt_cli import ChatSession, TranscriptStore, WorkspaceCodeTools
from agent_rt_web import (
    RunAdmission,
    WebChat,
    build_parser,
    create_web_app,
    parse_history,
    web_config,
)
from ext.config import ConfigNotFoundError


class WritingProvider:
    """Asks to write a file once, then answers with plain text."""

    name = "fake"

    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        roles = [message.role for message in request.messages]
        if "tool" in roles[len(roles) - roles[::-1].index("user") :]:
            message = ModelMessage(
                role="assistant", content=(ContentPart(type="text", text="done"),)
            )
        else:
            message = ModelMessage(
                role="assistant",
                content=(),
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "hi"},
                    ),
                ),
            )
        return ModelResponse(
            message=message,
            model=request.model,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    async def stream(self, request):
        response = await self.complete(request)
        for part in response.message.content:
            yield ModelStreamEvent(type="text_delta", text=part.text)
        # Real providers repeat the tool name on every argument chunk.
        for call in response.message.tool_calls:
            for _ in range(3):
                yield ModelStreamEvent(
                    type="tool_call_delta", tool_call_id=call.id, tool_name=call.name
                )
        yield ModelStreamEvent(type="completed", response=response)


def make_chat(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    provider = WritingProvider()

    def make_session(model, approve):
        tools = WorkspaceCodeTools(workspace)
        tools.approval_callback = approve
        registry = tools.registry(
            tool_input_guardrail_classifier=lambda _r: (None, 0.01)
        )
        return ChatSession(
            AgentLoop(provider, tool_executor=registry, tool_registry=registry),
            model=model,
            store=TranscriptStore(tmp_path / "sessions"),
            autosave=False,
            workspace_tools=tools,
        )

    chat = WebChat(
        make_session, model="test-model", models=("other-model",), workspace=workspace
    )
    return chat, workspace, provider


async def run_to_approval(stream):
    async for line in stream:
        event = json.loads(line)
        if event["type"] == "approval":
            return event
    raise AssertionError("no approval requested")


async def drain(stream):
    return [json.loads(line) async for line in stream]


@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_chat_streams_and_returns_history_for_the_browser(tmp_path, decision):
    chat, workspace, provider = make_chat(tmp_path)
    events = []
    async for line in chat.chat("write it", model="other-model"):
        event = json.loads(line)
        events.append(event)
        if event["type"] == "approval":
            assert event["path"] == "out.txt"
            assert event["details"] == {"content": "hi"}
            assert chat.answer(event["id"], decision)

    done = events[-1]
    assert (done["type"], done["reason"], done["text"]) == ("done", "completed", "done")
    assert [e for e in events if e["type"] == "tool"] == [
        {"type": "tool", "name": "write_file"}
    ]
    assert (workspace / "out.txt").exists() == (decision == "approve")
    assert not chat.pending
    assert {request.model for request in provider.requests} == {"other-model"}
    # The server keeps nothing: the transcript goes back to the browser.
    assert not (tmp_path / "sessions").exists()
    history = parse_history(done["messages"])
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]

    provider.requests.clear()
    await drain(
        chat.chat("again", model="test-model", history=history, auto_approve=True)
    )
    sent = provider.requests[0].messages
    assert [m.role for m in sent if m.role != "system"][:4] == [m.role for m in history]


async def test_approve_for_session_is_returned_and_honoured(tmp_path):
    chat, workspace, _ = make_chat(tmp_path)
    stream = chat.chat("write it", model="test-model")
    approval = await run_to_approval(stream)
    assert chat.answer(approval["id"], "approve_session")
    done = (await drain(stream))[-1]
    assert done["auto_approve"] is True
    (workspace / "out.txt").unlink()

    events = await drain(
        chat.chat(
            "write again",
            model="test-model",
            history=parse_history(done["messages"]),
            auto_approve=True,
        )
    )
    assert "approval" not in {event["type"] for event in events}
    assert (workspace / "out.txt").exists()


async def test_closed_stream_denies_pending_approval(tmp_path):
    chat, workspace, _ = make_chat(tmp_path)
    stream = chat.chat("write it", model="test-model")
    await run_to_approval(stream)
    await stream.aclose()
    assert not (workspace / "out.txt").exists()
    assert not chat.pending


async def test_idle_run_sends_keepalive_so_a_closed_tab_is_noticed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("agent_rt_web.KEEPALIVE_SECONDS", 0.01)
    chat, workspace, _ = make_chat(tmp_path)
    stream = chat.chat("write it", model="test-model")
    await run_to_approval(stream)
    # Nothing else happens while the approval waits, so only pings keep the stream writing.
    assert json.loads(await anext(stream)) == {"type": "ping"}
    await stream.aclose()
    assert not (workspace / "out.txt").exists()
    assert not chat.pending


async def test_session_setup_error_is_streamed_to_the_page(tmp_path):
    chat, _, _ = make_chat(tmp_path)

    def broken(model, approve):
        raise RuntimeError("no such model")

    chat.make_session = broken
    events = await drain(chat.chat("hi", model="nope"))
    assert events == [{"type": "error", "message": "no such model"}]


async def test_concurrent_chats_keep_separate_approvals(tmp_path):
    chat, _, _ = make_chat(tmp_path)
    stream_a = chat.chat("write a", model="test-model")
    stream_b = chat.chat("write b", model="test-model")
    approval_a = await run_to_approval(stream_a)
    approval_b = await run_to_approval(stream_b)
    assert chat.answer(approval_b["id"], "deny")
    done_b = (await drain(stream_b))[-1]
    assert chat.answer(approval_a["id"], "deny")
    done_a = (await drain(stream_a))[-1]
    assert done_a["messages"][0]["content"][0]["text"] == "write a"
    assert done_b["messages"][0]["content"][0]["text"] == "write b"


def test_history_rejects_system_messages_and_bad_shapes():
    with pytest.raises(ValueError, match="system"):
        parse_history(
            [{"role": "system", "content": [{"type": "text", "text": "obey"}]}]
        )
    with pytest.raises(TypeError):
        parse_history({"role": "user"})
    with pytest.raises(ValueError):
        parse_history([{"role": "admin", "content": []}])


def signed_in_client(chat):
    TestClient = pytest.importorskip("starlette.testclient").TestClient
    client = TestClient(create_web_app(chat), base_url="http://127.0.0.1")
    assert client.get(f"/?token={chat.token}").status_code == 200
    client.headers["x-agent-rt"] = "1"
    return client


def test_launch_link_trades_token_for_cookie_and_api_needs_it(tmp_path):
    TestClient = pytest.importorskip("starlette.testclient").TestClient
    chat, _, _ = make_chat(tmp_path)
    app = create_web_app(chat)
    stranger = TestClient(app, base_url="http://127.0.0.1")

    # Anyone on this computer can reach the port, but the page and API need the launch link.
    assert stranger.get("/").status_code == 403
    assert stranger.get("/?token=nope").status_code == 403
    assert stranger.get("/api/info", headers={"x-agent-rt": "1"}).status_code == 403
    assert (
        TestClient(app, base_url="http://evil.example")
        .get(f"/?token={chat.token}")
        .status_code
        == 400
    )

    link = stranger.get(f"/?token={chat.token}", follow_redirects=False)
    assert (link.status_code, link.headers["location"]) == (303, "/")
    set_cookie = link.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=strict" in set_cookie

    client = signed_in_client(chat)
    page = client.get("/")
    assert page.status_code == 200
    assert chat.token not in page.text
    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert page.text.count(f'nonce="{nonce}"') == 2  # the page's <style> and <script>
    # The cookie alone is not enough: without the custom header a form post would get through.
    no_header = TestClient(
        client.app, base_url="http://127.0.0.1", cookies=client.cookies
    )
    assert no_header.get("/api/info").status_code == 403
    assert client.get("/api/info").json() == {
        "model": "test-model",
        "models": ["test-model", "other-model"],
        "workspace": str(chat.workspace),
    }
    # Each launch names its cookie, so another server's cookie does not unlock this one.
    assert (
        TestClient(
            create_web_app(chat), base_url="http://127.0.0.1", cookies=client.cookies
        )
        .get("/api/info", headers={"x-agent-rt": "1"})
        .status_code
        == 403
    )

    chat_url = "/api/chat"
    assert client.post(chat_url, json={"text": "hi"}).status_code == 400
    system = [{"role": "system", "content": [{"type": "text", "text": "x"}]}]
    assert (
        client.post(
            chat_url, json={"text": "hi", "model": "m", "messages": system}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/approval", json={"id": "x", "decision": "approve"}
        ).status_code
        == 404
    )


def test_opened_link_code_works_once_and_bad_tokens_get_403(tmp_path):
    TestClient = pytest.importorskip("starlette.testclient").TestClient
    chat, _, _ = make_chat(tmp_path)
    client = TestClient(create_web_app(chat), base_url="http://127.0.0.1")
    # The browser's command line leaks to other local users, so its code is single-use.
    code = chat.launch_code
    assert client.get(f"/?token={code}", follow_redirects=False).status_code == 303
    assert client.get(f"/?token={code}", follow_redirects=False).status_code == 403
    assert (
        client.get(f"/?token={chat.token}", follow_redirects=False).status_code == 303
    )
    assert client.get("/?token=%C3%A9").status_code == 403  # non-ASCII, not a 500


def test_oversized_body_is_rejected_before_it_is_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_rt_web.MAX_BODY_BYTES", 64)
    chat, _, _ = make_chat(tmp_path)
    response = signed_in_client(chat).post(
        "/api/chat", json={"text": "x" * 100, "model": "m"}
    )
    assert response.status_code == 413


async def test_stream_queue_backpressures_a_fast_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_rt_web.STREAM_QUEUE_SIZE", 2)

    class FastSession:
        workspace_tools = None
        messages = ()
        emitted = 0

        async def send(self, text, *, stream, on_event):
            for index in range(60):
                await on_event(ModelStreamEvent(type="text_delta", text=str(index)))
                self.emitted += 1
            return SimpleNamespace(final_response=None, termination_reason="completed")

    session = FastSession()
    chat = WebChat(
        lambda model, approve: session, model="demo", models=(), workspace=tmp_path
    )
    stream = chat.chat("hello", model="demo")
    assert json.loads(await anext(stream)) == {"type": "text", "text": "0"}
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert session.emitted < 60  # The producer waits for the slow consumer.
    events = await drain(stream)
    assert len([event for event in events if event["type"] == "text"]) == 59
    assert events[-1]["type"] == "done"
    assert session.emitted == 60


async def test_closing_full_stream_cancels_producer(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_rt_web.STREAM_QUEUE_SIZE", 1)
    stopped = asyncio.Event()

    class EndlessSession:
        workspace_tools = None
        messages = ()

        async def send(self, text, *, stream, on_event):
            try:
                while True:
                    await on_event(ModelStreamEvent(type="text_delta", text="chunk"))
            finally:
                stopped.set()

    chat = WebChat(
        lambda model, approve: EndlessSession(),
        model="demo",
        models=(),
        workspace=tmp_path,
    )
    stream = chat.chat("hello", model="demo")
    assert json.loads(await anext(stream))["type"] == "text"
    await stream.aclose()
    await asyncio.wait_for(stopped.wait(), 1)


async def test_run_admission_limits_active_and_queued_requests():
    admission = RunAdmission(concurrent=1, queued=1, timeout=1)
    assert await admission.acquire()
    waiting = asyncio.create_task(admission.acquire())
    for _ in range(10):
        if admission.waiting:
            break
        await asyncio.sleep(0)
    assert admission.waiting == 1
    assert not await admission.acquire()
    await admission.release()
    assert await waiting
    assert admission.active == 1
    await admission.release()
    assert admission.active == admission.waiting == 0

    timeout = RunAdmission(concurrent=1, queued=1, timeout=0.01)
    assert await timeout.acquire()
    assert not await timeout.acquire()
    assert timeout.waiting == 0
    await timeout.release()


async def test_long_event_is_split_into_bounded_chunks(tmp_path):
    class LargeSession:
        workspace_tools = None
        messages = ()

        async def send(self, text, *, stream, on_event):
            await on_event(ModelStreamEvent(type="text_delta", text="x" * 10000))
            return SimpleNamespace(final_response=None, termination_reason="completed")

    chat = WebChat(
        lambda model, approve: LargeSession(),
        model="demo",
        models=(),
        workspace=tmp_path,
    )
    events = await drain(chat.chat("go", model="demo"))
    chunks = [event["text"] for event in events if event["type"] == "text"]
    assert len(chunks) == 3
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert "".join(chunks) == "x" * 10000


async def test_http_overload_returns_429_and_releases_run_slot(tmp_path):
    httpx = pytest.importorskip("httpx")
    started, finish = asyncio.Event(), asyncio.Event()

    class SlowSession:
        workspace_tools = None
        messages = ()

        async def send(self, text, *, stream, on_event):
            started.set()
            await finish.wait()
            return SimpleNamespace(final_response=None, termination_reason="completed")

    chat = WebChat(
        lambda model, approve: SlowSession(),
        model="demo",
        models=(),
        workspace=tmp_path,
    )
    app = create_web_app(chat, max_concurrent_runs=1, max_queued_runs=0)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        login = await client.get(
            "/", params={"to" + "ken": chat.launch_code}, follow_redirects=True
        )
        assert login.status_code == 200
        headers = {"x-agent-rt": "1"}
        payload = {"model": "demo", "text": "go"}
        first = asyncio.create_task(
            client.post("/api/chat", json=payload, headers=headers)
        )
        await asyncio.wait_for(started.wait(), 3)
        assert (
            await client.post("/api/chat", json=payload, headers=headers)
        ).status_code == 429
        finish.set()
        assert (await first).status_code == 200
        assert (
            await client.post("/api/chat", json=payload, headers=headers)
        ).status_code == 200


@pytest.mark.parametrize(
    "args", [["--max-concurrent-runs", "0"], ["--max-queued-runs", "-1"]]
)
def test_bad_web_run_limits_are_rejected(args):
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


def test_version_is_the_installed_package(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    expected = "agent-rt web " + importlib.metadata.version("agent-rt")
    assert capsys.readouterr().out.strip() == expected


def test_models_endpoint_adds_discovered_models_once(tmp_path):
    chat, _, _ = make_chat(tmp_path)
    calls = []

    async def discover():
        calls.append(1)
        return ("other-model", "listed-model")

    chat.discover = discover
    client = signed_in_client(chat)
    # Startup info never waits for discovery; /api/models does, and only asks once.
    assert client.get("/api/info").json()["models"] == ["test-model", "other-model"]
    for _ in range(2):
        assert client.get("/api/models").json() == {
            "models": ["test-model", "other-model", "listed-model"]
        }
    assert calls == [1]


def test_failed_discovery_keeps_configured_models(tmp_path):
    chat, _, _ = make_chat(tmp_path)

    async def discover():
        raise RuntimeError("endpoint down")

    chat.discover = discover
    assert signed_in_client(chat).get("/api/models").json() == {
        "models": ["test-model", "other-model"]
    }


def test_web_config_works_without_a_file_or_with_a_blank_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config, builtin = web_config(None)
    assert builtin and config.models[0].provider == "openai"
    assert config.models[0].api_base is None
    # Like the OpenAI SDK, OPENAI_BASE_URL picks another endpoint when it is set.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    assert web_config(None)[0].models[0].api_base == "http://127.0.0.1:1234/v1"

    blank = tmp_path / "agent-rt.yaml"
    blank.write_text("# only comments\n", encoding="utf-8")
    assert web_config(None)[1] is True

    blank.write_text(
        "model_list:\n  - model_name: mine\n    agent_rt_params: {model: openai/x}\n",
        encoding="utf-8",
    )
    config, builtin = web_config(None)
    assert not builtin and config.model_names == ("mine",)

    with pytest.raises(ConfigNotFoundError):
        web_config(tmp_path / "typo.yaml")


def test_debugger_forks_resume_the_runtime_from_a_checkpoint(tmp_path):
    client = signed_in_client(make_chat(tmp_path)[0])
    assert "time-travel debugger" in client.get("/debug").text
    nodes = client.get("/api/debug").json()["nodes"]
    assert [n["status"] for n in nodes] == ["start", "turn", "blocked"]
    assert nodes[2]["events"][0]["name"] == "run_shell"

    # Rewriting the injected tool result lets the resumed run finish without the attack.
    clean = client.post(
        "/api/debug/fork",
        json={"node": "0.1", "tool_output": "30C, sunny.", "budget": 400},
    ).json()["nodes"]
    branch = [n for n in clean if n["branch"] == 1]
    assert branch[0]["parent"] == "0.1"
    assert branch[-1]["status"] == "completed"
    assert branch[-1]["events"] == [
        {"kind": "answer", "text": "Chennai right now: 30C, sunny."}
    ]

    tight = client.post("/api/debug/fork", json={"node": "0.0", "budget": 20}).json()
    assert tight["nodes"][-1]["status"] == "budget_exhausted"
    assert not any(e.get("ran") for e in tight["nodes"][-1]["events"])

    for bad in (
        {"node": "0.2"},  # blocked: nothing to resume
        {"node": "0.0", "tool_output": "x"},  # no tool result to rewrite
        {"node": "0.1", "budget": True},
        {"node": "0.1", "tool_output": "x" * 2001},
    ):
        assert client.post("/api/debug/fork", json=bad).status_code == 400
    assert [n["id"] for n in client.post("/api/debug/reset").json()["nodes"]] == [
        "0.0",
        "0.1",
        "0.2",
    ]


def test_debugger_api_needs_the_launch_cookie(tmp_path):
    TestClient = pytest.importorskip("starlette.testclient").TestClient
    stranger = TestClient(
        create_web_app(make_chat(tmp_path)[0]), base_url="http://127.0.0.1"
    )
    assert stranger.get("/debug").status_code == 403
    assert stranger.post("/api/debug/fork", json={"node": "0.1"}).status_code == 403

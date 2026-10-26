import json

import pytest

from agent_rt import (
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TodoList,
    ToolCall,
    ToolRegistry,
)
from agent_rt_cli import ChatSession, TranscriptStore, build_workspace_session
from agent_rt_web import WebChat
from ext.extensions import SkillRegistry
from tests.test_web import signed_in_client


def test_actions_manage_and_clear_the_list():
    todos = TodoList()
    todos.apply({"action": "add", "content": " plan "})
    todos.apply({"action": "add", "content": "build", "status": "in_progress"})
    assert todos.apply({"action": "update", "id": 1, "status": "completed"}) == {
        "todos": [
            {"id": 1, "content": "plan", "status": "completed"},
            {"id": 2, "content": "build", "status": "in_progress"},
        ]
    }
    todos.apply({"action": "remove", "id": 1})
    # Ids are never reused while later tasks exist.
    assert todos.apply({"action": "add", "content": "ship"})["todos"][-1]["id"] == 3
    assert todos.apply({"action": "clear"}) == {"todos": []}
    assert todos.apply({"action": "list"}) == {"todos": []}


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "add"},
        {"action": "add", "content": "   "},
        {"action": "add", "content": "x", "status": "done"},
        {"action": "update", "id": 9, "status": "completed"},
        {"action": "update", "id": 1},
        {"action": "remove", "id": 9},
        {"action": "explode"},
    ],
)
def test_bad_actions_are_rejected_without_changing_the_list(arguments):
    todos = TodoList([{"id": 1, "content": "a", "status": "pending"}])
    with pytest.raises(ValueError):
        todos.apply(arguments)
    assert todos.items == [{"id": 1, "content": "a", "status": "pending"}]


@pytest.mark.parametrize(
    "items",
    [
        None,
        [{"id": 1, "content": "a", "status": "pending"}] * 2,
        [{"id": True, "content": "a", "status": "pending"}],
        [{"id": 1, "content": "a" * 501, "status": "pending"}],
        [{"id": n + 1, "content": "a", "status": "pending"} for n in range(101)],
    ],
)
def test_stored_lists_are_validated(items):
    with pytest.raises(ValueError):
        TodoList(items)


async def test_registered_tool_runs_and_reports_errors_to_the_model():
    registry = ToolRegistry()
    TodoList().register(registry)
    added = await registry.execute(
        ToolCall(id="1", name="todo", arguments={"action": "add", "content": "a"})
    )
    assert added == {"todos": [{"id": 1, "content": "a", "status": "pending"}]}
    assert registry.get("todo").definition.error_behavior == "return_error"


def test_workspace_sessions_get_the_tool_and_separate_lists(tmp_path):
    def build():
        return build_workspace_session(
            object(),
            "m",
            tmp_path,
            approval_callback=lambda *_: "deny",
            skills=SkillRegistry(),
            store=TranscriptStore(tmp_path / "sessions"),
        )

    first, second = build(), build()
    assert "todo" in {d.name for d in first.loop.tool_registry.definitions()}
    first.todos.apply({"action": "add", "content": "only mine"})
    assert second.todos.items == []


def test_todos_are_saved_resumed_and_kept_per_session(tmp_path):
    store = TranscriptStore(tmp_path)
    session = ChatSession(AgentLoop(object()), model="m", session_id="a", store=store)
    session.todos.apply({"action": "add", "content": "keep me"})
    session.save()
    ChatSession(AgentLoop(object()), model="m", session_id="b", store=store).save()

    resumed = ChatSession(AgentLoop(object()), model="m", session_id="a", store=store)
    resumed.resume()
    assert resumed.todos.items == [{"id": 1, "content": "keep me", "status": "pending"}]
    resumed.session_id = "b"
    resumed.resume()
    assert resumed.todos.items == []
    assert "todos" not in json.loads(store.path_for("b").read_text())


class TodoProvider:
    """Adds one task with the todo tool, then answers."""

    name = "fake"

    async def complete(self, request):
        if request.messages[-1].role == "tool":
            message = ModelMessage(
                role="assistant", content=(ContentPart(type="text", text="ok"),)
            )
        else:
            message = ModelMessage(
                role="assistant",
                content=(),
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="todo",
                        arguments={"action": "add", "content": "next"},
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
        yield ModelStreamEvent(type="completed", response=await self.complete(request))


async def test_web_round_trips_each_sessions_list_through_the_page(tmp_path):
    def make_session(model, approve):
        registry = ToolRegistry()
        todos = TodoList()
        todos.register(registry)
        loop = AgentLoop(TodoProvider(), tool_executor=registry, tool_registry=registry)
        return ChatSession(loop, model=model, autosave=False, todos=todos)

    store = TranscriptStore(tmp_path / "sessions")
    chat = WebChat(make_session, model="m", models=(), workspace=tmp_path, store=store)
    sent = [{"id": 4, "content": "earlier", "status": "completed"}]
    events = [json.loads(line) async for line in chat.chat("go", model="m", todos=sent)]
    assert events[-1]["todos"] == sent + [
        {"id": 5, "content": "next", "status": "pending"}
    ]
    # Each prompt builds a fresh session, so another session's list never shows up.
    other = [json.loads(line) async for line in chat.chat("go", model="m")]
    assert other[-1]["todos"] == [{"id": 1, "content": "next", "status": "pending"}]

    client = signed_in_client(chat)
    saved = {"model": "m", "messages": [], "todos": events[-1]["todos"]}
    assert client.post("/api/sessions/s1/save", json=saved).status_code == 200
    assert client.get("/api/sessions/s1").json()["todos"] == events[-1]["todos"]
    bad = {"model": "m", "messages": [], "todos": [{"id": 1}]}
    assert client.post("/api/sessions/s1/save", json=bad).status_code == 400
    assert (
        client.post("/api/chat", json={"text": "x", "model": "m", **bad}).status_code
        == 400
    )

"""The built-in todo list tool and tool result marshalling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.tools.registry import RegisteredTool, ToolRegistry
from ext.runtime.types import CancellationToken, ContentPart

TODO_STATUSES = ("pending", "in_progress", "completed")


TODO_MAX_ITEMS = 100


TODO_MAX_CONTENT_CHARS = 500


class TodoList:
    """One session's task list, exposed to the model as the ``todo`` tool.

    Give every session its own instance: the tool handler only sees this list, so
    tasks never leak between sessions.
    """

    def __init__(self, items: Sequence[Mapping[str, Any]] = ()) -> None:
        self.items: list[dict[str, Any]] = []
        self.replace(items)

    def replace(self, items: Sequence[Mapping[str, Any]]) -> None:
        """Load a stored list (for example on resume); raises ValueError on bad entries."""
        if not isinstance(items, (list, tuple)) or len(items) > TODO_MAX_ITEMS:
            raise ValueError(f"todos must be a list of at most {TODO_MAX_ITEMS} items")
        loaded: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("each todo must be an object")
            todo_id = item.get("id")
            if type(todo_id) is not int or todo_id < 1:
                raise ValueError("todo id must be a positive integer")
            if any(existing["id"] == todo_id for existing in loaded):
                raise ValueError(f"duplicate todo id: {todo_id}")
            loaded.append(
                {
                    "id": todo_id,
                    "content": self._content(item.get("content")),
                    "status": self._status(item.get("status")),
                }
            )
        self.items = loaded

    @staticmethod
    def _content(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("todo content must be a non-empty string")
        if len(value) > TODO_MAX_CONTENT_CHARS:
            raise ValueError(
                f"todo content must be at most {TODO_MAX_CONTENT_CHARS} characters"
            )
        return value.strip()

    @staticmethod
    def _status(value: Any) -> str:
        if value not in TODO_STATUSES:
            raise ValueError(f"todo status must be one of {', '.join(TODO_STATUSES)}")
        return value

    def _find(self, todo_id: Any) -> dict[str, Any]:
        for item in self.items:
            if item["id"] == todo_id:
                return item
        raise ValueError(f"no todo with id {todo_id}")

    def apply(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one ``todo`` tool action and return the whole list."""
        action = arguments.get("action")
        if action == "add":
            if len(self.items) >= TODO_MAX_ITEMS:
                raise ValueError(f"todo list is full ({TODO_MAX_ITEMS} items)")
            self.items.append(
                {
                    "id": max((item["id"] for item in self.items), default=0) + 1,
                    "content": self._content(arguments.get("content")),
                    "status": self._status(arguments.get("status", "pending")),
                }
            )
        elif action == "update":
            item = self._find(arguments.get("id"))
            if "content" not in arguments and "status" not in arguments:
                raise ValueError("update needs content or status")
            content = (
                self._content(arguments["content"])
                if "content" in arguments
                else item["content"]
            )
            status = (
                self._status(arguments["status"])
                if "status" in arguments
                else item["status"]
            )
            item.update(content=content, status=status)
        elif action == "remove":
            self.items.remove(self._find(arguments.get("id")))
        elif action == "clear":
            self.items = []
        elif action != "list":
            raise ValueError(f"unknown todo action: {action}")
        return {"todos": [dict(item) for item in self.items]}

    def register(self, registry: ToolRegistry, *, name: str = "todo") -> RegisteredTool:
        async def handler(
            arguments: Mapping[str, Any], cancellation_token: CancellationToken | None
        ) -> dict[str, Any]:
            return self.apply(arguments)

        return registry.register(
            ToolDefinition(
                name=name,
                description=(
                    "Track this session's task list. Plan multi-step work with add, mark "
                    "progress with update (status pending, in_progress, or completed), "
                    "drop a task with remove, empty the list with clear, or view it with "
                    "list. Every action returns the whole list."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "update", "remove", "clear", "list"],
                        },
                        "id": {"type": "integer", "minimum": 1},
                        "content": {
                            "type": "string",
                            "maxLength": TODO_MAX_CONTENT_CHARS,
                        },
                        "status": {"type": "string", "enum": list(TODO_STATUSES)},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                side_effect="none",
                execution_mode="sequential",
                error_behavior="return_error",
            ),
            handler=handler,
        )


def marshal_tool_result(value: Any) -> tuple[ContentPart, ...]:
    if isinstance(value, ContentPart):
        return (value,)
    if isinstance(value, str):
        return (ContentPart(type="text", text=value),)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(isinstance(part, ContentPart) for part in value)
    ):
        return tuple(value)
    return (ContentPart(type="json", data=value),)

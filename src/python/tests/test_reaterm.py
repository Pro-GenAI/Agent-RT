"""Offline ReaTerm adapter tests (no native CLI, model or network)."""

import sys
from types import SimpleNamespace

import pytest

from agent_rt import ToolCall, ToolRegistry, register_reaterm_tool


@pytest.mark.asyncio
async def test_reaterm_tool_passes_argv_and_fixed_root(monkeypatch):
    calls = []

    class FakeReaTerm:
        def __init__(self, **options):
            calls.append(("options", options))

        def run(self, command, *args):
            calls.append(("run", command, args))
            return "contents"

    monkeypatch.setitem(sys.modules, "reaterm", SimpleNamespace(ReaTerm=FakeReaTerm))
    registry = ToolRegistry()
    tool = register_reaterm_tool(registry, root="/trusted/project", timeout_seconds=5)
    assert tool.definition.side_effect == "read"
    assert "root" not in tool.definition.input_schema["properties"]
    assert calls[0] == ("options", {"root": "/trusted/project", "binary": None, "timeout": 5})
    assert await registry.execute(
        ToolCall(id="1", name="reaterm", arguments={"command": "cat", "args": ["a b.txt"]})
    ) == "contents"
    assert calls[1] == ("run", "cat", ("a b.txt",))
    with pytest.raises(ValueError, match="read-only"):
        await tool.handler({"command": "rm", "args": ["x"]}, None)
    with pytest.raises(ValueError, match="array of strings"):
        await tool.handler({"command": "cat", "args": [1]}, None)


def test_reaterm_rejects_invalid_host_settings():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="root"):
        register_reaterm_tool(registry, root="")
    with pytest.raises(ValueError, match="timeout"):
        register_reaterm_tool(registry, root="/trusted/project", timeout_seconds=float("inf"))

"""ReaTerm read-only inspection tool for Agent-RT registries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.tools.registry import RegisteredTool, ToolRegistry
from ext.runtime.types import CancellationToken

REATERM_TOOL_NAME = "reaterm"
_READ_COMMANDS = frozenset({"pwd", "ls", "cat", "head", "tail", "wc", "grep", "stat"})


def register_reaterm_tool(
    registry: ToolRegistry,
    *,
    root: str,
    binary: str | None = None,
    timeout_seconds: float = 15.0,
) -> RegisteredTool:
    """Register an opt-in read-only tool with a trusted application-chosen root."""
    if not isinstance(root, str) or not root.strip():
        raise ValueError("reaterm root must be a nonempty application-selected path")
    if not 0 < timeout_seconds < float("inf"):
        raise ValueError("reaterm timeout_seconds must be finite and positive")
    try:
        from reaterm import ReaTerm
    except ImportError as exc:
        raise RuntimeError(
            'Install "agent-rt[reaterm]" on Python 3.11+ and the native ReaTerm CLI'
        ) from exc
    client = ReaTerm(root=root, binary=binary, timeout=timeout_seconds)

    async def execute(
        arguments: Mapping[str, Any], _token: CancellationToken | None
    ) -> str:
        command = arguments.get("command")
        argv = arguments.get("args", [])
        if not isinstance(command, str) or command not in _READ_COMMANDS:
            raise ValueError("reaterm command must be a supported read-only command")
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("reaterm args must be an array of strings")
        return await asyncio.to_thread(client.run, command, *argv)

    definition = ToolDefinition(
        name=REATERM_TOOL_NAME,
        description=(
            "Inspect files with ReaTerm's native read-only command allowlist: "
            "pwd, ls, cat, head, tail, wc, grep, stat. No shell or writes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": sorted(_READ_COMMANDS)},
                "args": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        side_effect="read",
        timeout_seconds=timeout_seconds + 1,
    )
    return registry.register(definition, handler=execute)

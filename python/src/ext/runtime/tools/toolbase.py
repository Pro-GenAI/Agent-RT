"""Toolbase: dynamic tool visibility, safety filters, and tool search."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ext.runtime.agent.config import AgentConfig
from ext.runtime.guardrails.tool_guardrails import _call_maybe_async
from ext.runtime.tools.capabilities import CapabilityCatalog, CapabilityDescriptor
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.tools.registry import RegisteredTool, ToolRegistry
from ext.runtime.types import CancellationToken, ModelMessage, ToolCall


@dataclass(frozen=True)
class ToolFilterContext:
    agent: AgentConfig
    messages: tuple[ModelMessage, ...]
    turn: int
    tool_calls: int
    runtime_context: Mapping[str, Any] = field(default_factory=dict)


ToolVisibilityFilter = Callable[
    [ToolFilterContext, Sequence[ToolDefinition]],
    Sequence[str] | Awaitable[Sequence[str]],
]


ToolSafetyFilter = Callable[
    [Sequence[ToolDefinition]],
    Sequence[str] | Awaitable[Sequence[str]],
]


TOOL_SEARCH_NAME = "tool_search"


_TOOL_SEARCH_RESULT_TYPE = "agent_rt.tool_search_results"


def _toolbase_mcp_name(server_label: str, tool_name: str) -> str:
    safe_server = re.sub(r"[^A-Za-z0-9_-]", "_", server_label)
    safe_tool = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name)
    return "mcp." + safe_server + "." + safe_tool


class Toolbase:
    """Safety-screened catalog containing local, deferred, and MCP tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        safe_definitions: Mapping[str, ToolDefinition],
        selection_filter: ToolVisibilityFilter | None,
        search_limit: int,
    ) -> None:
        self.registry = registry
        self.selection_filter = selection_filter
        self.search_limit = search_limit
        self._safe_definitions = dict(safe_definitions)

    @classmethod
    async def initialize(
        cls,
        *,
        tool_registry: ToolRegistry | None = None,
        mcp_clients: Mapping[str, Any] | None = None,
        safety_filter: ToolSafetyFilter | None = None,
        selection_filter: ToolVisibilityFilter | None = None,
        search_limit: int = 8,
        allowlist: Sequence[str] | None = None,
        blocklist: Sequence[str] = (),
        read_allowlist: Sequence[str] | None = None,
        read_blocklist: Sequence[str] = (),
        write_allowlist: Sequence[str] | None = None,
        write_blocklist: Sequence[str] = (),
    ) -> Toolbase:
        if search_limit <= 0:
            raise ValueError("tool search limit must be positive")
        registry = tool_registry or ToolRegistry()

        def permitted(name: str, side_effect: str) -> bool:
            reading = side_effect in ("none", "read")
            scoped_allow = read_allowlist if reading else write_allowlist
            scoped_block = read_blocklist if reading else write_blocklist
            return (
                name not in blocklist
                and name not in scoped_block
                and (allowlist is None or name in allowlist)
                and (scoped_allow is None or name in scoped_allow)
            )

        for name in tuple(registry.deferred_names()):
            if name not in blocklist and (allowlist is None or name in allowlist):
                registry.load(name)

        local_tools = registry.list()
        candidate_definitions: dict[str, ToolDefinition] = {
            tool.name: tool.model_definition() for tool in local_tools
        }
        candidate_raw_definitions: dict[str, ToolDefinition] = {
            tool.name: tool.definition for tool in local_tools
        }
        if TOOL_SEARCH_NAME in candidate_definitions:
            raise ValueError(f"tool name {TOOL_SEARCH_NAME!r} is reserved by Toolbase")
        candidate_definitions = {
            name: definition
            for name, definition in candidate_definitions.items()
            if permitted(name, definition.side_effect)
        }

        mcp_bindings: dict[str, tuple[Any, str, ToolDefinition]] = {}
        for server_label, client in (mcp_clients or {}).items():
            inner = getattr(client, "client", client)
            if hasattr(inner, "server_info") and inner.server_info is None:
                initialize = getattr(inner, "initialize", None)
                if callable(initialize):
                    await initialize()
            list_tools = getattr(client, "list_tools", None)
            if not callable(list_tools):
                raise TypeError(
                    f"MCP client {server_label!r} does not support list_tools()"
                )
            discovered = await list_tools()
            for tool in discovered:
                remote_name = (
                    tool.get("name")
                    if isinstance(tool, Mapping)
                    else getattr(tool, "name", None)
                )
                if not isinstance(remote_name, str) or not remote_name.strip():
                    raise ValueError(
                        f"MCP client {server_label!r} returned a tool without a name"
                    )
                exposed_name = _toolbase_mcp_name(server_label, remote_name)
                if not permitted(exposed_name, "consequential"):
                    continue
                if (
                    exposed_name == TOOL_SEARCH_NAME
                    or exposed_name in candidate_definitions
                ):
                    raise ValueError(
                        f"tool name collision while building Toolbase: {exposed_name}"
                    )
                description = (
                    tool.get("description")
                    if isinstance(tool, Mapping)
                    else getattr(tool, "description", None)
                )
                input_schema = (
                    tool.get("input_schema")
                    if isinstance(tool, Mapping)
                    else getattr(tool, "input_schema", None)
                )
                definition = ToolDefinition(
                    name=exposed_name,
                    description=(
                        description.strip()
                        if isinstance(description, str) and description.strip()
                        else f"MCP tool {remote_name} from {server_label}"
                    ),
                    input_schema=(
                        dict(input_schema)
                        if isinstance(input_schema, Mapping)
                        else {"type": "object"}
                    ),
                    metadata={
                        "toolbase_source": "mcp",
                        "mcp_server": server_label,
                        "mcp_tool": remote_name,
                    },
                    side_effect="consequential",
                )
                candidate_definitions[exposed_name] = definition
                mcp_bindings[exposed_name] = (client, remote_name, definition)

        if safety_filter is None:
            safe_names = set(candidate_definitions)
        else:
            selected = await _call_maybe_async(
                safety_filter, tuple(candidate_definitions.values())
            )
            safe_names = {
                name
                for name in selected
                if isinstance(name, str) and name in candidate_definitions
            }

        for exposed_name, (client, remote_name, definition) in mcp_bindings.items():
            if exposed_name not in safe_names:
                continue

            async def call_mcp(
                arguments: Mapping[str, Any],
                cancellation_token: CancellationToken | None,
                *,
                _client: Any = client,
                _remote_name: Any = remote_name,
            ) -> Any:
                del cancellation_token
                call_tool = getattr(_client, "call_tool", None)
                if not callable(call_tool):
                    raise TypeError(
                        "configured MCP client does not support call_tool()"
                    )
                return await call_tool(_remote_name, dict(arguments))

            registered = registry.register(definition, handler=call_mcp)
            candidate_raw_definitions[exposed_name] = registered.definition

        screened = {
            name: candidate_raw_definitions[name]
            for name in safe_names
            if name in candidate_raw_definitions
        }
        toolbase = cls(
            registry,
            safe_definitions=screened,
            selection_filter=selection_filter,
            search_limit=search_limit,
        )
        search_definition = ToolDefinition(
            name=TOOL_SEARCH_NAME,
            description=(
                "Search the agent's full safe tool catalog for tools that were not "
                "included in the current turn. Matching tools become available on "
                "the next turn."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": search_limit,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            side_effect="none",
            metadata={"toolbase_internal": True},
        )
        registered_search = registry.register(
            search_definition,
            handler=toolbase._search_handler,
        )
        toolbase._safe_definitions[TOOL_SEARCH_NAME] = registered_search.definition
        return toolbase

    @property
    def version(self) -> int:
        return self.registry.version

    def _is_screened(self, name: str, tool: RegisteredTool) -> bool:
        return self._safe_definitions.get(name) is tool.definition

    def definitions(self) -> tuple[ToolDefinition, ...]:
        visible: list[ToolDefinition] = []
        for tool in self.registry.list():
            if self._is_screened(tool.name, tool):
                visible.append(tool.model_definition())
        return tuple(visible)

    def get(self, name: str) -> RegisteredTool:
        tool = self.registry.get(name)
        if not self._is_screened(name, tool):
            raise KeyError(f"tool is not safety-screened in this Toolbase: {name}")
        return tool

    async def execute(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> Any:
        self.get(call.name)
        return await self.registry.execute(call, cancellation_token, request_context)

    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            CapabilityDescriptor(
                id=f"tool:{definition.name}",
                kind="tool",
                name=definition.name,
                description=definition.description,
                metadata=definition.metadata,
            )
            for definition in self.definitions()
            if definition.name != TOOL_SEARCH_NAME
        )

    async def _search_handler(
        self,
        arguments: Mapping[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> Mapping[str, Any]:
        del cancellation_token
        query = str(arguments["query"]).strip()
        requested_limit = arguments.get("limit", self.search_limit)
        limit = min(int(requested_limit), self.search_limit)
        catalog = CapabilityCatalog(self.capability_descriptors())
        matches = catalog.search(query, kinds=frozenset({"tool"}), limit=limit)
        definitions = {tool.name: tool for tool in self.definitions()}
        tools = []
        for match in matches:
            definition = definitions.get(match.capability.name)
            if definition is None:
                continue
            tools.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": dict(definition.input_schema),
                    "side_effect": definition.side_effect,
                    "score": match.score,
                }
            )
        return {
            "type": _TOOL_SEARCH_RESULT_TYPE,
            "query": query,
            "tools": tools,
        }

    @staticmethod
    def _searched_tool_names(messages: Sequence[ModelMessage]) -> set[str]:
        search_call_ids = {
            call.id
            for message in messages
            if message.role == "assistant"
            for call in message.tool_calls
            if call.name == TOOL_SEARCH_NAME
        }
        names: set[str] = set()
        for message in messages:
            if message.role != "tool" or message.tool_call_id not in search_call_ids:
                continue
            for part in message.content:
                if part.type != "json" or not isinstance(part.data, Mapping):
                    continue
                if part.data.get("type") != _TOOL_SEARCH_RESULT_TYPE:
                    continue
                results = part.data.get("tools")
                if not isinstance(results, Sequence) or isinstance(
                    results, (str, bytes)
                ):
                    continue
                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    name = result.get("name")
                    if isinstance(name, str):
                        names.add(name)
        return names

    def visibility_filter(
        self,
        base_filter: ToolVisibilityFilter | None = None,
    ) -> ToolVisibilityFilter:
        selector = base_filter or self.selection_filter

        async def filter_tools(
            context: ToolFilterContext,
            tools: Sequence[ToolDefinition],
        ) -> tuple[str, ...]:
            candidates = tuple(tool for tool in tools if tool.name != TOOL_SEARCH_NAME)
            if selector is None:
                selected = {tool.name for tool in candidates}
            elif candidates:
                selected = set(await _call_maybe_async(selector, context, candidates))
            else:
                selected = set()
            selected.update(self._searched_tool_names(context.messages))
            selected.add(TOOL_SEARCH_NAME)
            return tuple(tool.name for tool in tools if tool.name in selected)

        return filter_tools

"""Model Context Protocol client and capability types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

MCPCapabilityKind = Literal["tool", "resource", "prompt"]


@runtime_checkable
class MCPTransport(Protocol):
    async def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class MCPServerInfo:
    name: str
    version: str | None = None
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPResource:
    uri: str
    name: str | None = None
    description: str = ""
    mime_type: str | None = None


@dataclass(frozen=True)
class MCPPrompt:
    name: str
    description: str = ""
    arguments: tuple[str, ...] = ()


class MCPClient:
    def __init__(
        self,
        transport: MCPTransport,
        *,
        client_name: str = "agent-rt",
        client_version: str = "0",
        requested_capabilities: Sequence[str] = (),
    ) -> None:
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.requested_capabilities = tuple(requested_capabilities)
        self.server_info: MCPServerInfo | None = None

    async def initialize(self) -> MCPServerInfo:
        response = await self.transport.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
                "capabilities": list(self.requested_capabilities),
            },
        )
        server = response.get("serverInfo")
        server_map = server if isinstance(server, Mapping) else {}
        raw_capabilities = response.get("capabilities", ())
        if isinstance(raw_capabilities, Mapping):
            capabilities = frozenset(
                str(name)
                for name, enabled in raw_capabilities.items()
                if enabled is not False
            )
        elif isinstance(raw_capabilities, Sequence) and not isinstance(
            raw_capabilities, (str, bytes)
        ):
            capabilities = frozenset(str(item) for item in raw_capabilities)
        else:
            capabilities = frozenset()
        name = str(server_map.get("name") or "mcp-server")
        version = server_map.get("version")
        self.server_info = MCPServerInfo(
            name=name,
            version=str(version) if version is not None else None,
            capabilities=capabilities,
        )
        return self.server_info

    def _ensure_initialized(self) -> MCPServerInfo:
        if self.server_info is None:
            raise RuntimeError("MCP client is not initialized")
        return self.server_info

    def _ensure_capability(self, capability: str) -> None:
        """Enforce negotiation: only use features the server advertised."""
        server = self._ensure_initialized()
        if capability not in server.capabilities:
            raise RuntimeError(
                f"MCP server {server.name!r} does not advertise the "
                f"{capability!r} capability"
            )

    async def list_tools(self) -> tuple[MCPTool, ...]:
        self._ensure_capability("tools")
        response = await self.transport.request("tools/list", {})
        values = response.get("tools", ())
        return tuple(
            MCPTool(
                name=str(value.get("name", "")),
                description=str(value.get("description", "")),
                input_schema=(
                    dict(value.get("inputSchema", {}))
                    if isinstance(value.get("inputSchema", {}), Mapping)
                    else {}
                ),
            )
            for value in values
            if isinstance(value, Mapping) and str(value.get("name", "")).strip()
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        self._ensure_capability("resources")
        response = await self.transport.request("resources/list", {})
        values = response.get("resources", ())
        return tuple(
            MCPResource(
                uri=str(value.get("uri", "")),
                name=str(value["name"]) if value.get("name") is not None else None,
                description=str(value.get("description", "")),
                mime_type=(
                    str(value["mimeType"])
                    if value.get("mimeType") is not None
                    else None
                ),
            )
            for value in values
            if isinstance(value, Mapping) and str(value.get("uri", "")).strip()
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        self._ensure_capability("prompts")
        response = await self.transport.request("prompts/list", {})
        values = response.get("prompts", ())
        prompts: list[MCPPrompt] = []
        for value in values:
            if not isinstance(value, Mapping) or not str(value.get("name", "")).strip():
                continue
            raw_arguments = value.get("arguments", ())
            arguments = tuple(
                str(argument.get("name"))
                for argument in raw_arguments
                if isinstance(argument, Mapping) and argument.get("name") is not None
            )
            prompts.append(
                MCPPrompt(
                    name=str(value["name"]),
                    description=str(value.get("description", "")),
                    arguments=arguments,
                )
            )
        return tuple(prompts)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._ensure_capability("tools")
        return await self.transport.request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )

    async def read_resource(self, uri: str) -> Mapping[str, Any]:
        self._ensure_capability("resources")
        return await self.transport.request("resources/read", {"uri": uri})

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._ensure_capability("prompts")
        return await self.transport.request(
            "prompts/get",
            {"name": name, "arguments": dict(arguments or {})},
        )

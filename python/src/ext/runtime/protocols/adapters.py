"""Protocol adapters and connector registries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ProtocolKind = Literal["rest", "openapi", "graphql", "websocket", "grpc", "custom"]


@runtime_checkable
class ProtocolAdapter(Protocol):
    protocol: ProtocolKind

    async def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any: ...


class ProtocolAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProtocolAdapter] = {}

    def register(self, name: str, adapter: ProtocolAdapter) -> None:
        if not name.strip():
            raise ValueError("adapter name must not be empty")
        self._adapters[name] = adapter

    def get(self, name: str) -> ProtocolAdapter:
        if name not in self._adapters:
            raise KeyError(name)
        return self._adapters[name]

    async def request(
        self,
        name: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self.get(name).request(operation, payload)


@dataclass(frozen=True)
class ConnectorDefinition:
    name: str
    adapter: str
    operations: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ConnectorRegistry:
    def __init__(self, adapters: ProtocolAdapterRegistry) -> None:
        self.adapters = adapters
        self._connectors: dict[str, ConnectorDefinition] = {}

    def register(self, definition: ConnectorDefinition) -> None:
        if not definition.name.strip():
            raise ValueError("connector name must not be empty")
        self.adapters.get(definition.adapter)
        self._connectors[definition.name] = definition

    async def invoke(
        self,
        connector: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        definition = self._connectors.get(connector)
        if definition is None:
            raise KeyError(connector)
        remote_operation = definition.operations.get(operation)
        if remote_operation is None:
            raise KeyError(f"{connector}:{operation}")
        return await self.adapters.request(
            definition.adapter,
            remote_operation,
            payload,
        )

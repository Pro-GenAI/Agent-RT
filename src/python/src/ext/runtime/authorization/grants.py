"""Least-privilege capability grants and tenant context."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass

from ext.runtime.authorization.permissions import _normalize_permission_path
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.types import MemoryScope, SessionRef


@dataclass(frozen=True)
class CapabilityGrant:
    tools: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    networks: tuple[str, ...] = ()
    credential_ids: tuple[str, ...] = ()

    @staticmethod
    def _allows(value: str, patterns: Sequence[str]) -> bool:
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    def allows_tool(self, name: str) -> bool:
        return self._allows(name, self.tools)

    def allows_path(self, path: str) -> bool:
        return self._allows(_normalize_permission_path(path), self.paths)

    def allows_network(self, target: str) -> bool:
        return self._allows(target, self.networks)

    def allows_credential(self, credential_id: str) -> bool:
        return credential_id in self.credential_ids

    def filter_tools(
        self, tools: Sequence[ToolDefinition]
    ) -> tuple[ToolDefinition, ...]:
        return tuple(tool for tool in tools if self.allows_tool(tool.name))


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if "::" in self.tenant_id:
            raise ValueError("tenant_id must not contain the '::' separator")

    def qualify(self, kind: str, resource_id: str) -> str:
        if not kind.strip() or not resource_id.strip():
            raise ValueError("tenant resource kind and id must not be empty")
        # '::' is the namespace separator; allowing it in ids would let one
        # tenant's ids collide with another tenant's qualified names.
        if "::" in kind or "::" in resource_id:
            raise ValueError("tenant resource kind and id must not contain '::'")
        return f"{self.tenant_id}::{kind}::{resource_id}"

    def session_ref(self, session: SessionRef) -> SessionRef:
        return SessionRef(
            self.qualify("session", session.session_id),
            self.qualify("thread", session.thread_id),
        )

    def memory_scope(self, **kwargs: str | None) -> MemoryScope:
        if kwargs.get("tenant") not in (None, self.tenant_id):
            raise PermissionError("memory scope tenant does not match")
        return MemoryScope(
            tenant=self.tenant_id, **{k: v for k, v in kwargs.items() if k != "tenant"}
        )

    def workspace_id(self, workspace_id: str) -> str:
        return self.qualify("workspace", workspace_id)

    def task_id(self, task_id: str) -> str:
        return self.qualify("task", task_id)

    def quota_key(self, quota_id: str) -> str:
        return self.qualify("quota", quota_id)

    def credential_id(self, credential_id: str) -> str:
        return self.qualify("credential", credential_id)

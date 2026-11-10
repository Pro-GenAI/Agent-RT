"""Workspace records and persistent, tenant-scoped workspace stores."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from ext.runtime.authorization.grants import TenantContext
from ext.runtime.types import JSONValue, _validate_json_value
from ext.runtime.workspace.filesystem import (
    FileSystem,
    FileSystemBackendRegistry,
    InMemoryFileSystem,
    WorkspaceFiles,
)


@dataclass(frozen=True)
class WorkspaceRecord:
    workspace_id: str
    backend: str = "memory"
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must not be empty")
        if not self.backend.strip():
            raise ValueError("workspace backend must not be empty")
        _validate_json_value(dict(self.metadata))


class PersistentWorkspaceStore:
    def __init__(
        self,
        backends: FileSystemBackendRegistry | None = None,
    ) -> None:
        self.backends = backends or FileSystemBackendRegistry()
        if "memory" not in self.backends.list():
            self.backends.register("memory", lambda _workspace_id: InMemoryFileSystem())
        self._records: dict[str, WorkspaceRecord] = {}
        self._filesystems: dict[str, FileSystem] = {}

    def create(
        self,
        workspace_id: str,
        *,
        backend: str = "memory",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> WorkspaceRecord:
        if workspace_id in self._records:
            raise ValueError(f"workspace already exists: {workspace_id}")
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            backend=backend,
            metadata=metadata or {},
        )
        filesystem = self.backends.create(backend, workspace_id)
        self._records[workspace_id] = record
        self._filesystems[workspace_id] = filesystem
        return record

    def get(self, workspace_id: str) -> WorkspaceRecord | None:
        return self._records.get(workspace_id)

    def open(self, workspace_id: str) -> WorkspaceFiles:
        if workspace_id not in self._records:
            raise KeyError(f"workspace not found: {workspace_id}")
        return WorkspaceFiles(self._filesystems[workspace_id])

    def delete(self, workspace_id: str) -> bool:
        existed = self._records.pop(workspace_id, None) is not None
        self._filesystems.pop(workspace_id, None)
        return existed

    def list(self) -> tuple[WorkspaceRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))


class TenantWorkspaceStore:
    def __init__(self, store: PersistentWorkspaceStore, tenant: TenantContext) -> None:
        self.store = store
        self.tenant = tenant

    def create(
        self,
        workspace_id: str,
        *,
        backend: str = "memory",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> WorkspaceRecord:
        effective_metadata = dict(metadata or {})
        effective_metadata["tenant_id"] = self.tenant.tenant_id
        return self.store.create(
            self.tenant.workspace_id(workspace_id),
            backend=backend,
            metadata=effective_metadata,
        )

    def get(self, workspace_id: str) -> WorkspaceRecord | None:
        return self.store.get(self.tenant.workspace_id(workspace_id))

    def open(self, workspace_id: str) -> WorkspaceFiles:
        return self.store.open(self.tenant.workspace_id(workspace_id))

    def delete(self, workspace_id: str) -> bool:
        return self.store.delete(self.tenant.workspace_id(workspace_id))

    def list(self) -> tuple[WorkspaceRecord, ...]:
        prefix = f"{self.tenant.tenant_id}::workspace::"
        return tuple(
            record
            for record in self.store.list()
            if record.workspace_id.startswith(prefix)
        )

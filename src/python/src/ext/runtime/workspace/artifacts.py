"""Artifacts, versions, provenance, and artifact repositories."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

from ext.runtime.types import JSONValue, _validate_json_value
from ext.runtime.workspace.filesystem import ArtifactStatus

ArtifactKind = Literal[
    "document",
    "code",
    "dataset",
    "report",
    "image",
    "archive",
    "other",
]


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str | None = None
    model: str | None = None
    tool: str | None = None
    agent: str | None = None
    transformation: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_json_value(dict(self.metadata))


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: ArtifactKind
    name: str
    media_type: str = "application/octet-stream"
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    status: ArtifactStatus = "draft"
    retention_until_ms: int | None = None
    location: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if not self.name.strip():
            raise ValueError("artifact name must not be empty")
        if not self.media_type.strip():
            raise ValueError("artifact media_type must not be empty")
        _validate_json_value(dict(self.metadata))
        if (
            self.retention_until_ms is not None
            and self.retention_until_ms < self.created_at_ms
        ):
            raise ValueError(
                "artifact retention_until_ms must not precede created_at_ms"
            )


@dataclass(frozen=True)
class ArtifactVersion:
    artifact_id: str
    version: int
    data: Any
    parent_version: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    provenance: tuple[ProvenanceRecord, ...] = ()
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact version artifact_id must not be empty")
        if self.version < 1:
            raise ValueError("artifact version must be at least 1")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("artifact parent_version must precede version")
        _validate_json_value(dict(self.metadata))


@runtime_checkable
class ArtifactRepository(Protocol):
    def create(
        self,
        artifact: Artifact,
        data: Any,
        *,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion: ...
    def add_version(
        self,
        artifact_id: str,
        data: Any,
        *,
        parent_version: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion: ...
    def get(self, artifact_id: str) -> Artifact: ...
    def get_version(
        self,
        artifact_id: str,
        version: int | None = None,
    ) -> ArtifactVersion: ...
    def list_versions(self, artifact_id: str) -> tuple[ArtifactVersion, ...]: ...
    def diff_text(
        self,
        artifact_id: str,
        from_version: int,
        to_version: int,
    ) -> str: ...
    def finalize(self, artifact_id: str) -> Artifact: ...
    def transfer(self, artifact_id: str, location: str) -> Artifact: ...
    def delete(
        self,
        artifact_id: str,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> bool: ...


class InMemoryArtifactRepository:
    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._versions: dict[str, list[ArtifactVersion]] = {}

    def create(
        self,
        artifact: Artifact,
        data: Any,
        *,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion:
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"artifact already exists: {artifact.artifact_id}")
        self._artifacts[artifact.artifact_id] = artifact
        version = ArtifactVersion(
            artifact_id=artifact.artifact_id,
            version=1,
            data=data,
            metadata={},
            provenance=tuple(provenance),
        )
        self._versions[artifact.artifact_id] = [version]
        return version

    def add_version(
        self,
        artifact_id: str,
        data: Any,
        *,
        parent_version: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion:
        if artifact_id not in self._artifacts:
            raise KeyError(f"artifact not found: {artifact_id}")
        artifact = self._artifacts[artifact_id]
        if artifact.status == "finalized":
            raise ValueError("finalized artifact cannot accept new versions")
        versions = self._versions[artifact_id]
        next_version = len(versions) + 1
        parent = parent_version if parent_version is not None else versions[-1].version
        if parent < 1 or parent >= next_version:
            raise ValueError("artifact parent_version is invalid")
        if not any(item.version == parent for item in versions):
            raise ValueError(f"artifact parent version not found: {parent}")
        version = ArtifactVersion(
            artifact_id=artifact_id,
            version=next_version,
            data=data,
            parent_version=parent,
            metadata=metadata or {},
            provenance=tuple(provenance),
        )
        versions.append(version)
        return version

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"artifact not found: {artifact_id}") from exc

    def get_version(
        self,
        artifact_id: str,
        version: int | None = None,
    ) -> ArtifactVersion:
        if artifact_id not in self._versions:
            raise KeyError(f"artifact not found: {artifact_id}")
        versions = self._versions[artifact_id]
        if version is None:
            return versions[-1]
        for item in versions:
            if item.version == version:
                return item
        raise KeyError(f"artifact version not found: {artifact_id}@{version}")

    def list_versions(self, artifact_id: str) -> tuple[ArtifactVersion, ...]:
        if artifact_id not in self._versions:
            raise KeyError(f"artifact not found: {artifact_id}")
        return tuple(self._versions[artifact_id])

    def diff_text(
        self,
        artifact_id: str,
        from_version: int,
        to_version: int,
    ) -> str:
        import difflib

        left = self.get_version(artifact_id, from_version).data
        right = self.get_version(artifact_id, to_version).data
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError("artifact text diff requires string version data")
        return "".join(
            difflib.unified_diff(
                left.splitlines(keepends=True),
                right.splitlines(keepends=True),
                fromfile=f"{artifact_id}@{from_version}",
                tofile=f"{artifact_id}@{to_version}",
            )
        )

    def finalize(self, artifact_id: str) -> Artifact:
        artifact = self.get(artifact_id)
        if artifact.status == "finalized":
            return artifact
        updated = replace(artifact, status="finalized")
        self._artifacts[artifact_id] = updated
        return updated

    def transfer(self, artifact_id: str, location: str) -> Artifact:
        if not location.strip():
            raise ValueError("artifact transfer location must not be empty")
        artifact = self.get(artifact_id)
        updated = replace(artifact, location=location)
        self._artifacts[artifact_id] = updated
        return updated

    def delete(
        self,
        artifact_id: str,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> bool:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return False
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if (
            not force
            and artifact.retention_until_ms is not None
            and now < artifact.retention_until_ms
        ):
            raise PermissionError("artifact is still within its retention period")
        self._artifacts.pop(artifact_id, None)
        self._versions.pop(artifact_id, None)
        return True

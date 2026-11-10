"""Sandbox commands, snapshots, the backend protocol, and backend names."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast, runtime_checkable

from ext.runtime.sandbox.policy import SandboxNetworkPolicy, SandboxResourceLimits
from ext.runtime.types import JSONValue
from ext.runtime.workspace.filesystem import WorkspaceFiles


@dataclass(frozen=True)
class SandboxSnapshot:
    snapshot_id: str
    source_session_id: str
    files: Mapping[str, bytes]
    cwd: str
    environment: Mapping[str, str]
    runtime_versions: Mapping[str, str]
    installed_packages: Mapping[str, tuple[str, ...]]
    interpreter_state: Mapping[str, Mapping[str, JSONValue]]
    network_policy: SandboxNetworkPolicy
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("sandbox snapshot_id must not be empty")
        if not self.source_session_id.strip():
            raise ValueError("sandbox snapshot source_session_id must not be empty")


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...]
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: bytes | None = None

    def __post_init__(self) -> None:
        if not self.argv or any(not part for part in self.argv):
            raise ValueError("sandbox command argv must contain non-empty arguments")
        for key, value in self.env.items():
            if not key or "\x00" in key or "\x00" in value:
                raise ValueError("sandbox environment contains an invalid entry")


@dataclass(frozen=True)
class SandboxCommandResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    duration_ms: int = 0
    truncated: bool = False


@runtime_checkable
class SandboxBackend(Protocol):
    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult: ...


SandboxCommandRunner = Callable[
    [
        str,
        SandboxCommand,
        WorkspaceFiles,
        SandboxResourceLimits,
        Mapping[str, str],
        SandboxNetworkPolicy,
    ],
    Awaitable[SandboxCommandResult],
]


class CallbackSandboxBackend:
    def __init__(self, runner: SandboxCommandRunner) -> None:
        self.runner = runner

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        return await self.runner(
            session_id,
            command,
            workspace,
            limits,
            environment,
            network_policy,
        )


SandboxBackendName = Literal["native", "docker", "e2b", "microsandbox", "swe-rex"]


SANDBOX_BACKEND_ENV = "AGENT_RT_SANDBOX_BACKEND"


_INVALID_SANDBOX_BACKEND_NAMES = frozenset(
    {"", "0", "disable", "disabled", "false", "nil", "no", "none", "null", "off"}
)


def _normalized_sandbox_backend_name(value: str | None) -> SandboxBackendName:
    if value is None:
        return "native"
    normalized = value.strip().lower()
    if normalized in _INVALID_SANDBOX_BACKEND_NAMES:
        raise ValueError(
            f"{SANDBOX_BACKEND_ENV} must name an enabled sandbox backend; "
            "sandboxing cannot be disabled"
        )
    if normalized not in {"native", "docker", "e2b", "microsandbox", "swe-rex"}:
        raise ValueError(
            f"unsupported sandbox backend {value!r}; expected native, docker, e2b, microsandbox, or swe-rex"
        )
    return cast(SandboxBackendName, normalized)

"""SWE-ReX sandbox backend."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from typing import Any

from ext.runtime.sandbox.backends import SandboxCommand, SandboxCommandResult
from ext.runtime.sandbox.policy import (
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    _requires_advanced_network_enforcement,
)
from ext.runtime.workspace.filesystem import WorkspaceFiles


class SWEReXSandboxBackend:
    """SWE-ReX remote runtime adapter.

    The endpoint must already be backed by an isolated SWE-ReX deployment. Because
    the remote execution API cannot change sandbox networking or resource limits
    per command, unsupported policies fail closed here.
    """

    def __init__(self, url: str, *, api_key: str | None = None) -> None:
        normalized = url.strip().rstrip("/")
        if not normalized:
            raise ValueError("SWE-ReX URL must not be empty")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("SWE-ReX URL must start with http:// or https://")
        self.url = normalized
        self.api_key = api_key or None
        self._runtime: Any | None = None

    def _load_runtime(self) -> tuple[Any, Any]:
        try:
            remote = importlib.import_module("swerex.runtime.remote")
            abstract = importlib.import_module("swerex.runtime.abstract")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SWE-ReX support requires the optional 'swe-rex' package; "
                "install agent-rt[sandbox-swe-rex]"
            ) from exc
        return remote.RemoteRuntime, abstract.Command

    def _get_runtime(self) -> tuple[Any, Any]:
        RemoteRuntime, Command = self._load_runtime()
        if self._runtime is None:
            self._runtime = RemoteRuntime(
                host=self.url,
                port=None,
                auth_token=self.api_key,
                timeout=30.0,
            )
        return self._runtime, Command

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
        del session_id, workspace
        if network_policy.mode != "unrestricted":
            raise RuntimeError(
                "SWE-ReX remote backend cannot enforce harness network isolation; use a preconfigured isolated deployment and request unrestricted mode"
            )
        if _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "SWE-ReX remote backend cannot enforce the requested advanced network controls"
            )
        if (
            network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "SWE-ReX remote backend does not implement domain/proxy network policy"
            )
        if any(
            value is not None
            for value in (
                limits.cpu_seconds,
                limits.memory_bytes,
                limits.disk_bytes,
                limits.process_count,
            )
        ):
            raise RuntimeError(
                "SWE-ReX remote backend does not enforce per-command "
                "cpu/memory/disk/process limits"
            )
        if command.stdin is not None:
            raise RuntimeError("SWE-ReX remote backend does not support command stdin")
        runtime, Command = self._get_runtime()
        started = time.monotonic()
        result = await runtime.execute(
            Command(
                command=list(command.argv),
                timeout=limits.timeout_seconds,
                shell=False,
                check=False,
                env=dict(environment) or None,
                cwd=command.cwd or None,
            )
        )
        if result.exit_code is None:
            raise RuntimeError("SWE-ReX returned no exit code")
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout).encode(),
            stderr=str(result.stderr).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )

"""E2B sandbox backend."""

from __future__ import annotations

import importlib
import shlex
import time
from collections.abc import Mapping
from typing import Any

from ext.runtime.sandbox.backends import SandboxCommand, SandboxCommandResult
from ext.runtime.sandbox.policy import SandboxNetworkPolicy, SandboxResourceLimits
from ext.runtime.types import asyncio
from ext.runtime.workspace.filesystem import WorkspaceFiles


class E2BSandboxBackend:
    """Persistent E2B cloud sandboxes, keyed by harness session id."""

    def __init__(self, *, template: str | None = None) -> None:
        self.template = template.strip() if template and template.strip() else None
        self._sandboxes: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _load_sdk(self) -> Any:
        try:
            return importlib.import_module("e2b")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "E2B sandbox support requires the optional 'e2b' package; "
                "install agent-rt[sandbox-e2b]"
            ) from exc

    async def _sandbox(self, session_id: str) -> Any:
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            return sandbox
        # One creation per session: concurrent first commands must not each
        # start (and then leak) a billable cloud sandbox.
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            sandbox = self._sandboxes.get(session_id)
            if sandbox is not None:
                return sandbox
            sdk = self._load_sdk()
            create = sdk.Sandbox.create
            if self.template:
                sandbox = await asyncio.to_thread(create, self.template)
            else:
                sandbox = await asyncio.to_thread(create)
            self._sandboxes[session_id] = sandbox
            return sandbox

    async def close_session(self, session_id: str) -> bool:
        """Terminate the cloud sandbox owned by ``session_id`` (best effort)."""
        sandbox = self._sandboxes.pop(session_id, None)
        self._locks.pop(session_id, None)
        if sandbox is None:
            return False
        kill = getattr(sandbox, "kill", None)
        if callable(kill):
            await asyncio.to_thread(kill)
        return True

    async def close(self) -> None:
        for session_id in tuple(self._sandboxes):
            await self.close_session(session_id)

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
        del workspace
        if (
            network_policy.mode != "unrestricted"
            or network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "E2B backend requires unrestricted network policy; "
                "custom domain/proxy enforcement belongs in the E2B template"
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
                "E2B backend does not enforce per-command resource limits"
            )
        if command.stdin is not None:
            raise RuntimeError("E2B backend does not support command stdin")
        sandbox = await self._sandbox(session_id)
        started = time.monotonic()
        result = await asyncio.to_thread(
            sandbox.commands.run,
            shlex.join(command.argv),
            envs=dict(environment),
            cwd=command.cwd or None,
            timeout=60 if limits.timeout_seconds is None else limits.timeout_seconds,
        )
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout).encode(),
            stderr=str(result.stderr).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )

"""Microsandbox sandbox backend."""

from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Mapping
from typing import Any

from ext.runtime.sandbox.backends import SandboxCommand, SandboxCommandResult
from ext.runtime.sandbox.policy import (
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    _requires_advanced_network_enforcement,
)
from ext.runtime.types import asyncio
from ext.runtime.workspace.filesystem import WorkspaceFiles


class MicrosandboxBackend:
    """Persistent microsandbox microVMs, keyed by harness session id."""

    def __init__(self, image: str) -> None:
        if not image.strip():
            raise ValueError("microsandbox image must not be empty")
        self.image = image.strip()
        self._sandboxes: dict[str, Any] = {}
        self._names: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Network mode each microVM was created with. A microVM's network is
        # fixed at creation, so a later policy change cannot be applied.
        self._network_modes: dict[str, str] = {}

    def _load_sdk(self) -> Any:
        try:
            return importlib.import_module("microsandbox")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "microsandbox support requires the optional 'microsandbox' package; "
                "install agent-rt[sandbox-microsandbox]"
            ) from exc

    async def _sandbox(
        self,
        session_id: str,
        network_policy: SandboxNetworkPolicy,
    ) -> Any:
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            self._assert_network_unchanged(session_id, network_policy)
            return sandbox
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            sandbox = self._sandboxes.get(session_id)
            if sandbox is not None:
                self._assert_network_unchanged(session_id, network_policy)
                return sandbox
            return await self._create_sandbox(session_id, network_policy)

    def _assert_network_unchanged(
        self, session_id: str, network_policy: SandboxNetworkPolicy
    ) -> None:
        created_with = self._network_modes.get(session_id)
        if created_with is not None and created_with != network_policy.mode:
            raise RuntimeError(
                "microsandbox cannot change a running microVM's network policy "
                f"(created with {created_with!r}, now {network_policy.mode!r}); "
                "close the session and start a new one"
            )

    async def close_session(self, session_id: str) -> bool:
        """Stop the microVM owned by ``session_id`` (best effort)."""
        sandbox = self._sandboxes.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._names.pop(session_id, None)
        self._network_modes.pop(session_id, None)
        if sandbox is None:
            return False
        for method_name in ("stop", "kill", "close"):
            method = getattr(sandbox, method_name, None)
            if callable(method):
                outcome = method()
                if hasattr(outcome, "__await__"):
                    await outcome
                break
        return True

    async def close(self) -> None:
        for session_id in tuple(self._sandboxes):
            await self.close_session(session_id)

    async def _create_sandbox(
        self,
        session_id: str,
        network_policy: SandboxNetworkPolicy,
    ) -> Any:
        if network_policy.mode != "none" and _requires_advanced_network_enforcement(
            network_policy
        ):
            raise RuntimeError(
                "microsandbox backend cannot enforce the requested advanced network controls"
            )
        if (
            network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "microsandbox backend currently supports only whole-sandbox none/unrestricted network policies"
            )
        sdk = self._load_sdk()
        if network_policy.mode == "none":
            network = sdk.Network.none()
        elif network_policy.mode == "unrestricted":
            network = sdk.Network.allow_all()
        else:
            raise RuntimeError(
                "microsandbox backend does not yet map harness domain allowlists"
            )
        name = self._names.setdefault(
            session_id,
            f"agent-rt-{uuid.uuid4().hex[:20]}",
        )
        sandbox = await sdk.Sandbox.create(name, image=self.image, network=network)
        self._sandboxes[session_id] = sandbox
        self._network_modes[session_id] = network_policy.mode
        return sandbox

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
                "microsandbox backend does not map harness per-command "
                "cpu/memory/disk/process limits"
            )
        if command.stdin is not None:
            raise RuntimeError("microsandbox backend does not support command stdin")
        sandbox = await self._sandbox(session_id, network_policy)
        started = time.monotonic()
        result = await sandbox.exec(
            command.argv[0],
            list(command.argv[1:]),
            cwd=command.cwd or None,
            env=dict(environment) or None,
            timeout=limits.timeout_seconds,
        )
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout_text).encode(),
            stderr=str(result.stderr_text).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )

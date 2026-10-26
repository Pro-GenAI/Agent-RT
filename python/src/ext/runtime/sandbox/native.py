"""Native process sandbox backend."""

from __future__ import annotations

import math
import os
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import Any

from ext.runtime.sandbox.backends import SandboxCommand, SandboxCommandResult
from ext.runtime.sandbox.policy import (
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    _requires_advanced_network_enforcement,
    _sanitize_sandbox_environment,
)
from ext.runtime.types import asyncio
from ext.runtime.workspace.filesystem import WorkspaceFiles

# Ceiling for captured sandbox output when the caller sets no output_bytes
# limit. Output beyond it is discarded as it is read, never buffered.
DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024


async def _collect_process_result(
    process: Any,
    stdin: bytes | None,
    *,
    output_limit: int,
    kill: Callable[[], Awaitable[None] | None],
    started: float,
) -> SandboxCommandResult:
    """Feed stdin and drain stdout/stderr with a hard memory cap.

    If this coroutine is cancelled (timeouts are implemented by cancellation)
    or fails, the process is killed and reaped before the error propagates, so
    a timed-out command never keeps running in the background.
    """
    truncated = False

    async def drain(stream: Any) -> bytes:
        nonlocal truncated
        buffer = bytearray()
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return bytes(buffer)
            room = output_limit - len(buffer)
            if room > 0:
                buffer += chunk[:room]
            if len(chunk) > room:
                truncated = True

    async def feed() -> None:
        if process.stdin is None:
            return
        try:
            if stdin is not None:
                process.stdin.write(stdin)
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with suppress(Exception):
                process.stdin.close()

    try:
        stdout, stderr, _ = await asyncio.gather(
            drain(process.stdout), drain(process.stderr), feed()
        )
        await process.wait()
    except BaseException:
        outcome = kill()
        if outcome is not None:
            with suppress(Exception):
                await outcome
        with suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        # Child processes can inherit pipes after their parent is killed.
        # Close transports before the event loop shuts down.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
            await asyncio.sleep(0)
        raise
    return SandboxCommandResult(
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        truncated=truncated,
    )


def _native_preexec(
    uid: int,
    gid: int,
    limits: SandboxResourceLimits,
) -> Callable[[], None]:
    def configure() -> None:
        import resource

        if limits.cpu_seconds is not None:
            cpu = max(1, math.ceil(limits.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if limits.memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (limits.memory_bytes, limits.memory_bytes),
            )
        if limits.process_count is not None:
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.process_count, limits.process_count),
            )
        if limits.disk_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (limits.disk_bytes, limits.disk_bytes),
            )
        # Drop every inherited supplementary group before changing identity;
        # otherwise root's groups survive setuid(). Failure aborts the spawn.
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return configure


class NativeSandboxBackend:
    """Runs commands as a dedicated restricted POSIX uid/gid."""

    def __init__(
        self,
        *,
        uid: int,
        gid: int | None = None,
        root: str | None = None,
    ) -> None:
        if uid < 0 or (gid is not None and gid < 0):
            raise ValueError("native sandbox uid/gid must not be negative")
        self.uid = uid
        self.gid = gid
        self.root = os.path.abspath(root) if root else None

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
        if os.name != "posix":
            raise RuntimeError("native sandbox backend requires a POSIX platform")
        if network_policy.mode == "none":
            raise RuntimeError(
                "native sandbox backend cannot verify network isolation; use docker for no-network execution"
            )
        if _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "native sandbox backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, or bandwidth/transfer network controls"
            )
        if (
            network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "native sandbox backend does not implement domain/proxy policy"
            )
        base = os.path.abspath(self.root or os.getcwd())
        cwd = base
        if command.cwd:
            cwd = os.path.abspath(os.path.join(base, command.cwd))
            if os.path.commonpath((base, cwd)) != base:
                raise ValueError("sandbox working directory escapes native root")
        child_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        child_env.update(_sanitize_sandbox_environment(environment, network_policy))
        gid = self._resolve_gid()
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=cwd,
            env=child_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_native_preexec(self.uid, gid, limits),
        )

        def kill_group() -> None:
            # start_new_session made the child a group leader: kill descendants too.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(ProcessLookupError):
                process.kill()

        return await _collect_process_result(
            process,
            command.stdin,
            output_limit=(
                limits.output_bytes
                if limits.output_bytes is not None
                else DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES
            ),
            kill=kill_group,
            started=started,
        )

    def _resolve_gid(self) -> int:
        """Return the explicit gid, or the account's primary gid (never root's)."""
        if self.gid is not None:
            return self.gid
        try:
            import pwd

            return pwd.getpwuid(self.uid).pw_gid
        except (ImportError, KeyError) as exc:
            raise RuntimeError(
                "native sandbox could not derive a gid for the restricted uid; "
                "set AGENT_RT_SANDBOX_GID"
            ) from exc

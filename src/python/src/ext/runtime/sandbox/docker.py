"""Docker sandbox backend."""

from __future__ import annotations

import os
import posixpath
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress

from ext.runtime.sandbox.backends import SandboxCommand, SandboxCommandResult
from ext.runtime.sandbox.native import (
    DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES,
    _collect_process_result,
)
from ext.runtime.sandbox.policy import (
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    _requires_advanced_network_enforcement,
    _sanitize_sandbox_environment,
)
from ext.runtime.types import asyncio
from ext.runtime.workspace.filesystem import WorkspaceFiles


class DockerSandboxBackend:
    """Runs each command in an ephemeral, capability-dropped Docker container."""

    def __init__(
        self,
        image: str,
        *,
        docker_binary: str = "docker",
        workspace_root: str | None = None,
        user: str | None = None,
        read_only_root: bool = False,
    ) -> None:
        if not image.strip():
            raise ValueError("docker sandbox image must not be empty")
        if not docker_binary.strip():
            raise ValueError("docker sandbox binary must not be empty")
        self.image = image
        self.docker_binary = docker_binary
        self.workspace_root = (
            os.path.abspath(workspace_root) if workspace_root else None
        )
        if self.workspace_root is not None and any(
            character in self.workspace_root for character in ",\n\r\x00"
        ):
            # ``--mount`` options are comma separated; a comma would let the
            # path inject extra mount options.
            raise ValueError(
                "docker sandbox workspace_root must not contain commas or newlines"
            )
        if user is not None and not user.strip():
            raise ValueError("docker sandbox user must not be empty when provided")
        self.user = user.strip() if user else None
        self.read_only_root = read_only_root

    @staticmethod
    def _container_workdir(cwd: str) -> str:
        normalized = posixpath.normpath(cwd.replace("\\", "/"))
        if normalized.startswith(("/", "../")) or normalized == "..":
            raise ValueError("sandbox working directory must stay inside /workspace")
        return "/workspace" if normalized == "." else f"/workspace/{normalized}"

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
        if network_policy.mode != "none" and _requires_advanced_network_enforcement(
            network_policy
        ):
            raise RuntimeError(
                "docker backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, allow/block-list, or bandwidth/transfer network controls"
            )
        if limits.cpu_seconds is not None or limits.disk_bytes is not None:
            raise RuntimeError(
                "docker backend cannot enforce cpu_seconds/disk_bytes limits"
            )
        container = f"agent-rt-{uuid.uuid4().hex[:16]}"
        argv = [
            self.docker_binary,
            "run",
            "--rm",
            "-i",
            "--name",
            container,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if self.user is not None:
            argv += ["--user", self.user]
        if self.read_only_root:
            argv += [
                "--read-only",
                "--tmpfs",
                # This is a container-internal tmpfs mount, not a host temp path.
                "/tmp",  # nosec B108
            ]
        if network_policy.mode == "none":
            argv += ["--network", "none"]
        if limits.memory_bytes is not None:
            argv += [
                "--memory",
                str(limits.memory_bytes),
                "--memory-swap",
                str(limits.memory_bytes),
            ]
        if limits.process_count is not None:
            argv += ["--pids-limit", str(limits.process_count)]
        if self.workspace_root is not None:
            argv += [
                "--mount",
                f"type=bind,src={self.workspace_root},dst=/workspace",
                "--workdir",
                self._container_workdir(command.cwd or "."),
            ]
        elif command.cwd:
            argv += ["--workdir", command.cwd]
        effective_environment = _sanitize_sandbox_environment(
            environment, network_policy
        )
        if network_policy.proxy_url:
            effective_environment.setdefault("HTTPS_PROXY", network_policy.proxy_url)
        for key, value in effective_environment.items():
            argv += ["--env", f"{key}={value}"]
        argv.append(self.image)
        argv.extend(command.argv)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def kill_container() -> None:
            # Killing the docker client does not stop the container.
            with suppress(ProcessLookupError):
                process.kill()
            killer = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "kill",
                container,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()

        return await _collect_process_result(
            process,
            command.stdin,
            output_limit=(
                limits.output_bytes
                if limits.output_bytes is not None
                else DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES
            ),
            kill=kill_container,
            started=started,
        )

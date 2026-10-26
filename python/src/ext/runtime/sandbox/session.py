"""Sandbox sessions, package managers, code interpreters, and backend selection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import agent_rt as _core
from ext.runtime.sandbox.backends import (
    SANDBOX_BACKEND_ENV,
    SandboxBackend,
    SandboxCommand,
    SandboxCommandResult,
    SandboxSnapshot,
    _normalized_sandbox_backend_name,
)
from ext.runtime.sandbox.docker import DockerSandboxBackend
from ext.runtime.sandbox.e2b import E2BSandboxBackend
from ext.runtime.sandbox.microsandbox import MicrosandboxBackend
from ext.runtime.sandbox.native import NativeSandboxBackend
from ext.runtime.sandbox.policy import (
    SandboxNetworkPolicy,
    SandboxResourceLimits,
    _sandbox_env_bool,
    _sanitize_sandbox_environment,
    sandbox_network_policy_from_env,
    sandbox_resource_limits_from_env,
)
from ext.runtime.sandbox.swe_rex import SWEReXSandboxBackend
from ext.runtime.tools.definitions import ToolHandler
from ext.runtime.types import CancellationToken, JSONValue, asyncio
from ext.runtime.workspace.filesystem import InMemoryFileSystem, WorkspaceFiles


def sandbox_backend_from_env(
    environment: Mapping[str, str] | None = None,
) -> SandboxBackend:
    env = _core.runtime_settings() if environment is None else environment
    if "AGENT_RT_DISABLE_SANDBOX" in env:
        raise ValueError(
            "AGENT_RT_DISABLE_SANDBOX is not supported; sandboxing cannot be disabled"
        )
    backend_name = _normalized_sandbox_backend_name(env.get(SANDBOX_BACKEND_ENV))
    if backend_name == "native":
        raw_uid = env.get("AGENT_RT_SANDBOX_UID")
        if raw_uid is None or not raw_uid.strip():
            raise ValueError(
                "native sandbox requires AGENT_RT_SANDBOX_UID for a restricted user"
            )
        try:
            uid = int(raw_uid)
            gid = (
                int(env["AGENT_RT_SANDBOX_GID"])
                if env.get("AGENT_RT_SANDBOX_GID", "").strip()
                else None
            )
        except ValueError as exc:
            raise ValueError("native sandbox uid/gid must be integers") from exc
        return NativeSandboxBackend(
            uid=uid,
            gid=gid,
            root=env.get("AGENT_RT_SANDBOX_NATIVE_ROOT"),
        )
    if backend_name == "docker":
        image = env.get("AGENT_RT_SANDBOX_DOCKER_IMAGE", "").strip()
        if not image:
            raise ValueError("docker sandbox requires AGENT_RT_SANDBOX_DOCKER_IMAGE")
        return DockerSandboxBackend(
            image,
            docker_binary=env.get("AGENT_RT_SANDBOX_DOCKER_BINARY", "docker"),
            workspace_root=env.get("AGENT_RT_SANDBOX_WORKSPACE_ROOT"),
            user=env.get("AGENT_RT_SANDBOX_DOCKER_USER"),
            read_only_root=_sandbox_env_bool(env, "AGENT_RT_SANDBOX_DOCKER_READ_ONLY"),
        )
    if backend_name == "e2b":
        return E2BSandboxBackend(template=env.get("AGENT_RT_SANDBOX_E2B_TEMPLATE"))
    if backend_name == "microsandbox":
        image = env.get("AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE", "").strip()
        if not image:
            raise ValueError(
                "microsandbox backend requires AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE"
            )
        return MicrosandboxBackend(image)
    url = env.get("AGENT_RT_SANDBOX_SWEREX_URL", "").strip()
    if not url:
        raise ValueError("SWE-ReX backend requires AGENT_RT_SANDBOX_SWEREX_URL")
    return SWEReXSandboxBackend(
        url,
        api_key=env.get("AGENT_RT_SANDBOX_SWEREX_API_KEY"),
    )


@runtime_checkable
class SandboxPackageManager(Protocol):
    async def install(
        self,
        session_id: str,
        runtime: str,
        packages: Sequence[str],
        *,
        workspace: WorkspaceFiles,
        environment: Mapping[str, str],
        limits: SandboxResourceLimits,
        network_policy: SandboxNetworkPolicy,
    ) -> Sequence[str]: ...


SandboxPackageInstaller = Callable[
    [
        str,
        str,
        Sequence[str],
        WorkspaceFiles,
        Mapping[str, str],
        SandboxResourceLimits,
        SandboxNetworkPolicy,
    ],
    Awaitable[Sequence[str]],
]


class CallbackSandboxPackageManager:
    def __init__(self, installer: SandboxPackageInstaller) -> None:
        self.installer = installer

    async def install(
        self,
        session_id: str,
        runtime: str,
        packages: Sequence[str],
        *,
        workspace: WorkspaceFiles,
        environment: Mapping[str, str],
        limits: SandboxResourceLimits,
        network_policy: SandboxNetworkPolicy,
    ) -> Sequence[str]:
        return await self.installer(
            session_id,
            runtime,
            packages,
            workspace,
            environment,
            limits,
            network_policy,
        )


CodeInterpreterRunner = Callable[
    [
        str,
        str,
        str,
        dict[str, JSONValue],
        WorkspaceFiles,
        Mapping[str, str],
        SandboxResourceLimits,
        SandboxNetworkPolicy,
    ],
    Awaitable[SandboxCommandResult],
]


class CodeInterpreterRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, CodeInterpreterRunner] = {}

    def register(
        self,
        runtime: str,
        runner: CodeInterpreterRunner,
        *,
        replace: bool = False,
    ) -> None:
        if not runtime.strip():
            raise ValueError("interpreter runtime must not be empty")
        if runtime in self._runners and not replace:
            raise ValueError(f"interpreter runtime already registered: {runtime}")
        self._runners[runtime] = runner

    def get(self, runtime: str) -> CodeInterpreterRunner:
        try:
            return self._runners[runtime]
        except KeyError as exc:
            raise KeyError(f"interpreter runtime not registered: {runtime}") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._runners))


class SandboxSession:
    """A sandbox session: backend, workspace files, limits, and network policy.

    Snapshots, restore, and clones capture only the in-process ``workspace``
    files and session settings. They do not capture the disk of a native,
    Docker, or cloud backend, so state written by commands inside the sandbox
    is outside their scope.
    """

    def __init__(
        self,
        session_id: str,
        backend: SandboxBackend,
        *,
        workspace: WorkspaceFiles | None = None,
        limits: SandboxResourceLimits | None = None,
        interpreters: CodeInterpreterRegistry | None = None,
        package_manager: SandboxPackageManager | None = None,
        network_policy: SandboxNetworkPolicy | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("sandbox session_id must not be empty")
        self.session_id = session_id
        self.backend = backend
        self.workspace = workspace or WorkspaceFiles(InMemoryFileSystem())
        defaults = sandbox_resource_limits_from_env()
        self.limits = (
            defaults
            if limits is None
            else SandboxResourceLimits(
                cpu_seconds=(
                    limits.cpu_seconds
                    if limits.cpu_seconds is not None
                    else defaults.cpu_seconds
                ),
                memory_bytes=(
                    limits.memory_bytes
                    if limits.memory_bytes is not None
                    else defaults.memory_bytes
                ),
                disk_bytes=limits.disk_bytes,
                process_count=limits.process_count,
                timeout_seconds=(
                    limits.timeout_seconds
                    if limits.timeout_seconds is not None
                    else defaults.timeout_seconds
                ),
                output_bytes=limits.output_bytes,
            )
        )
        self.interpreters = interpreters or CodeInterpreterRegistry()
        self.package_manager = package_manager
        self.network_policy = network_policy or sandbox_network_policy_from_env()
        self._snapshots: dict[str, SandboxSnapshot] = {}
        self.environment: dict[str, str] = {}
        self.cwd = ""
        self.runtime_versions: dict[str, str] = {}
        self.installed_packages: dict[str, tuple[str, ...]] = {}
        self.interpreter_state: dict[str, dict[str, JSONValue]] = {}

    async def close(self) -> None:
        """Release backend resources (cloud sandboxes, microVMs) for this session."""
        close_session = getattr(self.backend, "close_session", None)
        if callable(close_session):
            await close_session(self.session_id)

    def set_environment(self, values: Mapping[str, str]) -> None:
        for key, value in values.items():
            if not key or "\x00" in key or "\x00" in value:
                raise ValueError("sandbox environment contains an invalid entry")
            self.environment[key] = value

    def unset_environment(self, *keys: str) -> None:
        for key in keys:
            self.environment.pop(key, None)

    def set_working_directory(self, path: str) -> None:
        normalized = InMemoryFileSystem._normalize(path)
        self.cwd = normalized

    def set_runtime_version(self, runtime: str, version: str) -> None:
        if not runtime.strip() or not version.strip():
            raise ValueError("runtime and version must not be empty")
        self.runtime_versions[runtime] = version

    def runtime_version(self, runtime: str) -> str | None:
        return self.runtime_versions.get(runtime)

    def set_network_policy(self, policy: SandboxNetworkPolicy) -> None:
        self.network_policy = policy

    def network_allows(self, target: str) -> bool:
        return self.network_policy.allows(target)

    def _snapshot_files(self) -> dict[str, bytes]:
        files: dict[str, bytes] = {}

        def walk(path: str = "") -> None:
            for item in self.workspace.list(path):
                if item.is_directory:
                    walk(item.path)
                else:
                    files[item.path] = self.workspace.filesystem.read(item.path)

        walk()
        return files

    def create_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        snapshot = SandboxSnapshot(
            snapshot_id=snapshot_id,
            source_session_id=self.session_id,
            files=self._snapshot_files(),
            cwd=self.cwd,
            environment=dict(self.environment),
            runtime_versions=dict(self.runtime_versions),
            installed_packages={
                runtime: tuple(packages)
                for runtime, packages in self.installed_packages.items()
            },
            interpreter_state={
                runtime: dict(state)
                for runtime, state in self.interpreter_state.items()
            },
            network_policy=self.network_policy,
        )
        if snapshot_id in self._snapshots:
            raise ValueError(f"sandbox snapshot already exists: {snapshot_id}")
        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"sandbox snapshot not found: {snapshot_id}") from exc

    def restore_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        snapshot = self.get_snapshot(snapshot_id)
        self.workspace.clear()
        for path, data in snapshot.files.items():
            self.workspace.filesystem.write(path, data)
        self.cwd = snapshot.cwd
        self.environment.clear()
        self.environment.update(snapshot.environment)
        self.runtime_versions.clear()
        self.runtime_versions.update(snapshot.runtime_versions)
        self.installed_packages.clear()
        self.installed_packages.update(
            {
                runtime: tuple(packages)
                for runtime, packages in snapshot.installed_packages.items()
            }
        )
        self.interpreter_state.clear()
        self.interpreter_state.update(
            {
                runtime: dict(state)
                for runtime, state in snapshot.interpreter_state.items()
            }
        )
        self.network_policy = snapshot.network_policy
        return snapshot

    def clone_from_snapshot(
        self,
        snapshot_id: str,
        new_session_id: str,
    ) -> SandboxSession:
        snapshot = self.get_snapshot(snapshot_id)
        clone = SandboxSession(
            new_session_id,
            self.backend,
            limits=self.limits,
            interpreters=self.interpreters,
            package_manager=self.package_manager,
            network_policy=snapshot.network_policy,
        )
        clone._snapshots[snapshot_id] = snapshot
        clone.restore_snapshot(snapshot_id)
        return clone

    def install_packages(
        self, runtime: str, packages: Sequence[str]
    ) -> tuple[str, ...]:
        if not runtime.strip():
            raise ValueError("runtime must not be empty")
        normalized = tuple(
            dict.fromkeys(item.strip() for item in packages if item.strip())
        )
        self.installed_packages[runtime] = normalized
        return normalized

    async def install_runtime_packages(
        self,
        runtime: str,
        packages: Sequence[str],
    ) -> tuple[str, ...]:
        if self.package_manager is None:
            raise RuntimeError("sandbox package manager is not configured")
        normalized = tuple(
            dict.fromkeys(item.strip() for item in packages if item.strip())
        )
        installed = await self.package_manager.install(
            self.session_id,
            runtime,
            normalized,
            workspace=self.workspace,
            environment=dict(self.environment),
            limits=self.limits,
            network_policy=self.network_policy,
        )
        recorded = tuple(
            dict.fromkeys(item.strip() for item in installed if item.strip())
        )
        self.installed_packages[runtime] = recorded
        return recorded

    def runtime_packages(self, runtime: str) -> tuple[str, ...]:
        return self.installed_packages.get(runtime, ())

    async def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        effective = SandboxCommand(
            argv=command.argv,
            cwd=command.cwd or self.cwd,
            env=command.env,
            stdin=command.stdin,
        )
        environment = dict(self.environment)
        environment.update(command.env)
        environment = _sanitize_sandbox_environment(environment, self.network_policy)
        operation = self.backend.execute(
            self.session_id,
            effective,
            workspace=self.workspace,
            limits=self.limits,
            environment=environment,
            network_policy=self.network_policy,
        )
        if self.limits.timeout_seconds is None:
            result = await operation
        else:
            try:
                result = await asyncio.wait_for(
                    operation,
                    timeout=self.limits.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "sandbox command exceeded execution timeout"
                ) from exc
        return self._bounded_result(result)

    async def run_code(
        self,
        runtime: str,
        code: str,
    ) -> SandboxCommandResult:
        runner = self.interpreters.get(runtime)
        state = self.interpreter_state.setdefault(runtime, {})
        operation = runner(
            self.session_id,
            runtime,
            code,
            state,
            self.workspace,
            dict(self.environment),
            self.limits,
            self.network_policy,
        )
        if self.limits.timeout_seconds is None:
            result = await operation
        else:
            try:
                result = await asyncio.wait_for(
                    operation,
                    timeout=self.limits.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "sandbox interpreter exceeded execution timeout"
                ) from exc
        return self._bounded_result(result)

    def _bounded_result(self, result: SandboxCommandResult) -> SandboxCommandResult:
        limit = self.limits.output_bytes
        if limit is None:
            return result
        remaining = limit
        stdout = result.stdout[:remaining]
        remaining -= len(stdout)
        stderr = result.stderr[:remaining]
        truncated = (
            result.truncated
            or len(stdout) < len(result.stdout)
            or len(stderr) < len(result.stderr)
        )
        return SandboxCommandResult(
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=result.duration_ms,
            truncated=truncated,
        )


def sandbox_shell_tool(
    session: SandboxSession,
) -> ToolHandler:
    async def handler(
        arguments: Mapping[str, Any],
        _cancellation_token: CancellationToken | None,
    ) -> Any:
        argv = arguments.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            raise ValueError("shell tool requires argv array")
        command = SandboxCommand(
            argv=tuple(str(item) for item in argv),
            cwd=str(arguments.get("cwd", "")),
            env={
                str(key): str(value)
                for key, value in dict(arguments.get("env", {})).items()
            },
        )
        result = await session.execute(command)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "duration_ms": result.duration_ms,
            "truncated": result.truncated,
        }

    return handler

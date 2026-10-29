"""MCP servers for the terminal client, configured the way Codex and Claude Code configure them.

Sources, later names overriding earlier ones: ``~/.codex/config.toml`` (``[mcp_servers.*]``),
``~/.claude.json`` (``mcpServers``), the workspace ``.mcp.json``, and the ``~/.claude.json``
entry for this workspace (``projects[path].mcpServers``). ``.mcp.json`` ships with the
repository, so its servers start only after the user approves them (or approved them in
Claude Code via ``enabledMcpjsonServers``). Transports: stdio and streamable HTTP.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ext.registration_safety import RegistrationSafetyError
from ext.runtime.tools.definitions import ToolDefinition
from ext.runtime.tools.registry import ToolRegistry

PROTOCOL_VERSION = "2025-06-18"
REQUEST_TIMEOUT_SECONDS = 120.0
CONNECT_TIMEOUT_SECONDS = 30.0
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_TOOL_PAGES = 50


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    source: Path
    project: bool = False
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return self.url or " ".join((self.command or "", *self.args)).strip()


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any, environ: Mapping[str, str]) -> str:
    """Claude Code's ``${VAR}`` / ``${VAR:-default}`` expansion; an unset variable raises."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environ:
            return environ[name]
        if default is not None:
            return default
        raise ValueError(f"environment variable {name} is not set")

    return _ENV_REFERENCE.sub(replace, str(value))


def _strings(value: Any) -> dict[str, str]:
    return (
        {str(k): str(v) for k, v in value.items()} if isinstance(value, Mapping) else {}
    )


def _claude_spec(
    name: str, entry: Any, source: Path, project: bool, environ: Mapping[str, str]
) -> MCPServerSpec:
    if not isinstance(entry, Mapping):
        raise ValueError("server entry must be an object")
    kind = entry.get("type") or ("http" if entry.get("url") else "stdio")
    if kind == "stdio" and entry.get("command"):
        return MCPServerSpec(
            name,
            source,
            project,
            command=_expand(entry["command"], environ),
            args=tuple(_expand(arg, environ) for arg in entry.get("args") or ()),
            env={k: _expand(v, environ) for k, v in _strings(entry.get("env")).items()},
        )
    if kind == "http" and entry.get("url"):
        return MCPServerSpec(
            name,
            source,
            project,
            url=_expand(entry["url"], environ),
            headers={
                k: _expand(v, environ)
                for k, v in _strings(entry.get("headers")).items()
            },
        )
    raise ValueError(f"unsupported server type {kind!r} (stdio and http are supported)")


def _codex_spec(
    name: str, entry: Any, source: Path, environ: Mapping[str, str]
) -> MCPServerSpec | None:
    if not isinstance(entry, Mapping):
        raise ValueError("server entry must be a table")
    if entry.get("enabled") is False:
        return None
    if entry.get("command"):
        return MCPServerSpec(
            name,
            source,
            command=str(entry["command"]),
            args=tuple(str(arg) for arg in entry.get("args") or ()),
            env=_strings(entry.get("env")),
            cwd=str(entry["cwd"]) if entry.get("cwd") else None,
        )
    if entry.get("url"):
        headers = _strings(entry.get("http_headers"))
        for header, variable in _strings(entry.get("env_http_headers")).items():
            if variable in environ:
                headers[header] = environ[variable]
        token_variable = entry.get("bearer_token_env_var")
        if token_variable:
            if token_variable not in environ:
                raise ValueError(f"environment variable {token_variable} is not set")
            headers["Authorization"] = f"Bearer {environ[token_variable]}"
        return MCPServerSpec(name, source, url=str(entry["url"]), headers=headers)
    raise ValueError("server needs a command or a url")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_toml(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:  # Python 3.10
        raise ValueError("reading Codex config needs Python 3.11 or newer") from None
    return tomllib.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class MCPDiscovery:
    servers: tuple[MCPServerSpec, ...]
    # Project servers that still need the user's approval before they start.
    pending: tuple[MCPServerSpec, ...]
    warnings: tuple[str, ...]


def discover_mcp_servers(
    workspace: Path,
    *,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> MCPDiscovery:
    home = home or Path.home()
    environ = os.environ if environ is None else environ
    workspace_key = str(workspace.resolve())
    servers: dict[str, MCPServerSpec] = {}
    warnings: list[str] = []

    def add(
        source: Path,
        entries: Mapping[str, Any],
        build: Callable[[str, Any], MCPServerSpec | None],
    ) -> None:
        for name, entry in entries.items():
            try:
                spec = build(str(name), entry)
            except (ValueError, TypeError) as exc:
                warnings.append(f"{source}: MCP server {name!r} skipped: {exc}")
                continue
            if spec is None:
                servers.pop(name, None)
            else:
                servers[name] = spec

    def section(data: Any, key: str) -> Mapping[str, Any]:
        value = data.get(key) if isinstance(data, Mapping) else None
        return value if isinstance(value, Mapping) else {}

    def load(path: Path, reader: Callable[[Path], Any]) -> Any:
        try:
            return reader(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"{path}: MCP config not read: {exc}")
            return {}

    codex = home / ".codex" / "config.toml"
    add(
        codex,
        section(load(codex, _read_toml), "mcp_servers"),
        lambda name, entry: _codex_spec(name, entry, codex, environ),
    )

    claude = home / ".claude.json"
    claude_data = load(claude, _read_json)
    add(
        claude,
        section(claude_data, "mcpServers"),
        lambda name, entry: _claude_spec(name, entry, claude, False, environ),
    )

    local = section(section(claude_data, "projects"), workspace_key)
    approved = set(local.get("enabledMcpjsonServers") or ())
    disabled = set(local.get("disabledMcpjsonServers") or ())
    project = workspace / ".mcp.json"
    project_entries = section(load(project, _read_json), "mcpServers")
    add(
        project,
        {
            name: entry
            for name, entry in project_entries.items()
            if name not in disabled
        },
        lambda name, entry: _claude_spec(
            name, entry, project, name not in approved, environ
        ),
    )

    add(
        claude,
        section(local, "mcpServers"),
        lambda name, entry: _claude_spec(name, entry, claude, False, environ),
    )

    return MCPDiscovery(
        servers=tuple(spec for spec in servers.values() if not spec.project),
        pending=tuple(spec for spec in servers.values() if spec.project),
        warnings=tuple(warnings),
    )


def _result(message: Mapping[str, Any]) -> Mapping[str, Any]:
    error = message.get("error")
    if isinstance(error, Mapping):
        raise RuntimeError(f"MCP error {error.get('code')}: {error.get('message')}")
    result = message.get("result")
    return result if isinstance(result, Mapping) else {}


class StdioMCPTransport:
    """Newline-delimited JSON-RPC over a child process's stdin/stdout."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Mapping[str, Any]]] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None

    async def start(self) -> None:
        assert self.spec.command is not None
        # shortcut: passes the whole environment like Claude Code; add Codex's env_vars allowlist if servers must not see it.
        self._process = await asyncio.create_subprocess_exec(
            self.spec.command,
            *self.spec.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, **self.spec.env},
            cwd=self.spec.cwd,
            limit=_MAX_LINE_BYTES,
        )
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while line := await self._process.stdout.readline():
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # servers sometimes print logs to stdout
                if not isinstance(message, dict):
                    continue
                if "method" in message:
                    if (
                        "id" in message
                    ):  # a server request: answer ping, refuse the rest
                        reply: dict[str, Any] = (
                            {"result": {}}
                            if message["method"] == "ping"
                            else {
                                "error": {"code": -32601, "message": "method not found"}
                            }
                        )
                        await self._send(
                            {"jsonrpc": "2.0", "id": message["id"], **reply}
                        )
                    continue
                future = self._pending.get(message.get("id"))  # type: ignore[arg-type]
                if future is not None and not future.done():
                    future.set_result(message)
        except (OSError, ValueError):
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("MCP server exited"))

    async def _send(self, payload: Mapping[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        await self._process.stdin.drain()

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        if self._reader is None or self._reader.done():
            raise ConnectionError("MCP server is not running")
        request_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params or {}),
                }
            )
            return _result(await asyncio.wait_for(future, REQUEST_TIMEOUT_SECONDS))
        finally:
            self._pending.pop(request_id, None)

    async def notify(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> None:
        await self._send(
            {"jsonrpc": "2.0", "method": method, "params": dict(params or {})}
        )

    async def close(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except (asyncio.TimeoutError, TimeoutError):
            process.kill()
            await process.wait()
        if self._reader is not None:
            await self._reader


class HTTPMCPTransport:
    """MCP streamable HTTP: JSON-RPC POSTs answered with JSON or a server-sent event stream."""

    def __init__(
        self, spec: MCPServerSpec, client: httpx.AsyncClient | None = None
    ) -> None:
        assert spec.url is not None
        self.spec = spec
        self._client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        self._ids = itertools.count(1)
        self._session: str | None = None

    async def start(self) -> None:
        return None

    def _headers(self) -> dict[str, str]:
        headers = {
            **self.spec.headers,
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": PROTOCOL_VERSION,
        }
        if self._session:
            headers["mcp-session-id"] = self._session
        return headers

    async def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        assert self.spec.url is not None
        async with self._client.stream(
            "POST", self.spec.url, json=payload, headers=self._headers()
        ) as response:
            response.raise_for_status()
            self._session = response.headers.get("mcp-session-id") or self._session
            if "id" not in payload:
                return None
            if not response.headers.get("content-type", "").startswith(
                "text/event-stream"
            ):
                return json.loads(await response.aread())
            data: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
                elif not line and data:
                    message = json.loads("\n".join(data))
                    data = []
                    if (
                        isinstance(message, Mapping)
                        and message.get("id") == payload["id"]
                    ):
                        return message
        raise ConnectionError("MCP stream ended without a response")

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        message = await self._post(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": method,
                "params": dict(params or {}),
            }
        )
        return _result(message or {})

    async def notify(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> None:
        await self._post(
            {"jsonrpc": "2.0", "method": method, "params": dict(params or {})}
        )

    async def close(self) -> None:
        if self._session and self.spec.url:
            try:
                await self._client.delete(self.spec.url, headers=self._headers())
            except httpx.HTTPError:
                pass
        await self._client.aclose()


@dataclass
class MCPConnection:
    spec: MCPServerSpec
    transport: Any
    tools: tuple[Mapping[str, Any], ...] = ()


async def connect(spec: MCPServerSpec, *, client_version: str = "0") -> MCPConnection:
    transport: Any = StdioMCPTransport(spec) if spec.command else HTTPMCPTransport(spec)

    async def handshake() -> tuple[Mapping[str, Any], ...]:
        await transport.start()
        init = await transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent-rt", "version": client_version},
            },
        )
        await transport.notify("notifications/initialized")
        if (
            not isinstance(init.get("capabilities"), Mapping)
            or "tools" not in init["capabilities"]
        ):
            return ()
        tools: list[Mapping[str, Any]] = []
        cursor = None
        for _ in range(_MAX_TOOL_PAGES):
            page = await transport.request(
                "tools/list", {"cursor": cursor} if cursor else {}
            )
            tools.extend(t for t in page.get("tools") or () if isinstance(t, Mapping))
            cursor = page.get("nextCursor")
            if not cursor:
                break
        return tuple(tools)

    try:
        tools = await asyncio.wait_for(handshake(), CONNECT_TIMEOUT_SECONDS)
    except BaseException:
        await transport.close()
        raise
    return MCPConnection(spec, transport, tools)


def _tool_name(server: str, tool: str) -> str:
    # Claude Code's naming; providers accept at most 64 [A-Za-z0-9_-] characters.
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{tool}")[:64]


def _tool_output(result: Mapping[str, Any]) -> str:
    parts = []
    for item in result.get("content") or ():
        if isinstance(item, Mapping) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, Mapping):
            parts.append(f"[{item.get('type', 'unknown')} content omitted]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts)
    if result.get("isError"):
        raise RuntimeError(text or "MCP tool reported an error")
    return text


def register_mcp_tools(
    registry: ToolRegistry,
    connections: Sequence[MCPConnection],
    approve: Callable[[str, Mapping[str, Any]], Awaitable[None]],
) -> list[str]:
    """Register each server's tools behind ``approve``; return warnings for skipped tools."""

    warnings = []
    for connection in connections:
        for tool in connection.tools:
            remote = str(tool.get("name") or "")
            name = _tool_name(connection.spec.name, remote)
            schema = tool.get("inputSchema")

            async def handler(
                arguments: Mapping[str, Any],
                _cancellation: Any = None,
                *,
                _name: str = name,
                _remote: str = remote,
                _transport: Any = connection.transport,
            ) -> str:
                await approve(_name, arguments)
                return _tool_output(
                    await _transport.request(
                        "tools/call", {"name": _remote, "arguments": dict(arguments)}
                    )
                )

            try:
                definition = ToolDefinition(
                    name=name,
                    description=str(tool.get("description") or "").strip()
                    or f"MCP tool {remote} from server {connection.spec.name}",
                    input_schema=dict(schema)
                    if isinstance(schema, Mapping)
                    and schema.get("type") in (None, "object")
                    else {"type": "object"},
                    side_effect="consequential",
                    error_behavior="return_error",
                    metadata={"mcp_server": connection.spec.name, "mcp_tool": remote},
                )
                registry.register(definition, handler=handler)
            except (RegistrationSafetyError, ValueError) as exc:
                warnings.append(f"MCP tool {name} skipped: {exc}")
    return warnings

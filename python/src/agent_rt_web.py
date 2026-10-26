import argparse
import asyncio
import json
import secrets
import sys
import uuid
import webbrowser
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent_rt import (
    AgentRunLimits,
    ModelMessage,
    ModelStreamEvent,
    configure_runtime,
    load_config,
    load_model,
)
from agent_rt_cli import (
    _CLI_SETTINGS,
    ChatSession,
    _deserialize_message,
    _discover_cli_skills,
    _skill_warning_consequences,
    _message_text,
    _package_version,
    _positive_int,
    _provider_model_ids,
    _serialize_message,
    _startup_model,
    build_workspace_session,
    discover_builtin_models,
    with_builtin_models,
)
from ext.config import AgentRTConfig, ConfigNotFoundError

PAGE_PATH = Path(__file__).resolve().parent / "ext" / "web_ui.html"
DEBUG_PAGE_PATH = PAGE_PATH.with_name("debug_ui.html")
# Any custom header forces a CORS preflight, which this server never approves, so other
# origins on this computer cannot post to the API even though browsers send them the cookie.
API_HEADER = "x-agent-rt"
KEEPALIVE_SECONDS = 15.0
PREVIEW_LIMIT = 4000
MAX_BODY_BYTES = 32 * 1024 * 1024
STREAM_QUEUE_SIZE = 128
STREAM_TEXT_CHARS = 4096
DEFAULT_MAX_CONCURRENT_RUNS = 4
DEFAULT_MAX_QUEUED_RUNS = 4
DEFAULT_QUEUE_TIMEOUT_SECONDS = 15.0
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
MAX_FORK_TOOL_OUTPUT = 2000
MAX_FORK_BUDGET = 100_000

# Builds a fresh session for one request: (model, approval callback) -> ChatSession.
SessionFactory = Callable[[str, Callable[[str, Mapping[str, Any]], Any]], ChatSession]


def web_config(path: Path | None) -> tuple[AgentRTConfig, bool]:
    """Load the config (none, or a blank one, is fine); True when built-in models were used."""
    try:
        config = load_config(path)
    except ConfigNotFoundError:
        if path is not None:  # a path the user typed must exist
            raise
        config = AgentRTConfig()
    return with_builtin_models(config)


def _require_web_dependencies() -> Any:
    # Plain Starlette, not FastAPI: it starts ~10x faster and uses ~10 MB less memory.
    try:
        import starlette  # noqa: F401
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            'the Agent RT web UI requires optional dependencies: pip install "agent-rt[web]"'
        ) from exc
    return uvicorn


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= PREVIEW_LIMIT else text[:PREVIEW_LIMIT] + "\n…"


def _same(sent: str, secret: str) -> bool:
    # Bytes, because compare_digest raises TypeError on non-ASCII str input.
    return secrets.compare_digest(sent.encode(), secret.encode())


def parse_history(value: Any) -> tuple[ModelMessage, ...]:
    """Validate a browser-stored transcript; instructions stay server-side, so no system role."""
    if not isinstance(value, list):
        raise TypeError("messages must be an array")
    messages = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("each message must be an object")
        message = _deserialize_message(item)
        if message.role == "system":
            raise ValueError("system messages are not accepted from the browser")
        messages.append(message)
    return tuple(messages)


class WebChat:
    """Stateless chat backend: the browser stores sessions and sends the history with each prompt."""

    def __init__(
        self,
        make_session: SessionFactory,
        *,
        model: str,
        models: Sequence[str],
        workspace: Path,
        discover: Callable[[], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        self.make_session = make_session
        self.model = model
        self.models = tuple(dict.fromkeys((model, *models)))
        self.discover = discover
        self.discovery: asyncio.Task[Sequence[str]] | None = None
        self.workspace = workspace
        self.token = secrets.token_urlsafe(32)
        # The browser's command line is readable by every local user (/proc/<pid>/cmdline), so the
        # auto-opened link carries this single-use code instead of the reusable printed token.
        self.launch_code: str | None = secrets.token_urlsafe(32)
        self.pending: dict[str, asyncio.Future[str]] = {}

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "models": list(self.models),
            "workspace": str(self.workspace),
        }

    async def all_models(self) -> tuple[str, ...]:
        """Configured models plus each endpoint's /v1/models, discovered once on first request."""
        if self.discover is not None:
            if self.discovery is None:
                self.discovery = asyncio.ensure_future(self.discover())
            with suppress(Exception):
                found = await asyncio.shield(self.discovery)
                self.models = tuple(dict.fromkeys((*self.models, *found)))
        return self.models

    def redeem(self, token: str) -> bool:
        """Accept the printed token any number of times, or the launch code once."""
        if _same(token, self.token):
            return True
        if self.launch_code is not None and _same(token, self.launch_code):
            self.launch_code = None
            return True
        return False

    def answer(self, approval_id: str, decision: str) -> bool:
        future = self.pending.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True

    async def chat(
        self,
        text: str,
        *,
        model: str,
        history: Sequence[ModelMessage] = (),
        auto_approve: bool = False,
    ):
        """Run one prompt, yielding NDJSON lines: deltas, tool calls, approvals, then done/error.

        ``done`` carries the updated transcript for the browser to store.
        """
        # Bound server memory when a provider produces tokens faster than the browser reads.
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=STREAM_QUEUE_SIZE
        )
        mine: set[str] = set()
        announced: set[tuple[str | None, str]] = set()

        async def on_event(event: ModelStreamEvent) -> None:
            if event.type == "text_delta" and event.text:
                for start in range(0, len(event.text), STREAM_TEXT_CHARS):
                    await queue.put(
                        {
                            "type": "text",
                            "text": event.text[start : start + STREAM_TEXT_CHARS],
                        }
                    )
            elif event.type == "tool_call_delta" and event.tool_name:
                # Providers repeat the tool name on every argument chunk; send each call once.
                key = (event.tool_call_id, event.tool_name)
                if key not in announced:
                    announced.add(key)
                    await queue.put({"type": "tool", "name": event.tool_name})

        async def approve(action: str, arguments: Mapping[str, Any]) -> str:
            approval_id = uuid.uuid4().hex
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self.pending[approval_id] = future
            mine.add(approval_id)
            details = {
                key: _preview(arguments.get(key, ""))
                for key in ("old_text", "new_text", "content")
                if key in arguments
            }
            await queue.put(
                {
                    "type": "approval",
                    "id": approval_id,
                    "action": action,
                    "path": str(arguments.get("path", "")),
                    "details": details,
                }
            )
            try:
                return await future
            finally:
                self.pending.pop(approval_id, None)

        async def keepalive() -> None:
            # Writing is how the server notices a closed tab (ASGI 2.4+ only reports a
            # disconnect on send), so a run idle on an approval still gets cancelled.
            while True:
                await asyncio.sleep(KEEPALIVE_SECONDS)
                # Keepalive events are disposable and never wait for queue capacity.
                if not queue.full():
                    queue.put_nowait({"type": "ping"})

        try:
            session = self.make_session(model, approve)
        except Exception as exc:  # noqa: BLE001 - stream startup errors to the caller
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return
        session.messages = tuple(history)
        if session.workspace_tools is not None:
            session.workspace_tools.auto_approve_writes = auto_approve
        task = asyncio.create_task(session.send(text, stream=True, on_event=on_event))

        async def signal_completion() -> None:
            # The reader reports task errors after draining buffered events.
            with suppress(Exception, asyncio.CancelledError):
                await task
            await queue.put(None)

        completion = asyncio.create_task(signal_completion())
        pinger = asyncio.create_task(keepalive())
        try:
            while (item := await queue.get()) is not None:
                yield json.dumps(item) + "\n"
            try:
                result = task.result()
            except Exception as exc:  # noqa: BLE001
                yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
                return
            final = result.final_response
            tools = session.workspace_tools
            yield (
                json.dumps(
                    {
                        "type": "done",
                        "reason": result.termination_reason,
                        "text": _message_text(final.message) if final else "",
                        "messages": [_serialize_message(m) for m in session.messages],
                        "auto_approve": bool(tools and tools.auto_approve_writes),
                    },
                    default=str,
                )
                + "\n"
            )
        finally:
            pinger.cancel()
            completion.cancel()
            # A closed tab must not leave a run or an approval waiting forever.
            if not task.done():
                task.cancel()
            for approval_id in mine:
                future = self.pending.pop(approval_id, None)
                if future is not None and not future.done():
                    future.set_result("deny")


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
# Only the page's own <script>/<style> (carrying the per-launch nonce) may run.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
    "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


class RunAdmission:
    """Bound active and waiting runs with a finite admission queue."""

    def __init__(self, concurrent: int, queued: int, timeout: float) -> None:
        if concurrent < 1 or queued < 0 or timeout <= 0:
            raise ValueError(
                "run admission limits must be positive (queued may be zero)"
            )
        self.concurrent = concurrent
        self.queued = queued
        self.timeout = timeout
        self.active = 0
        self.waiting = 0
        self.condition = asyncio.Condition()

    async def acquire(self) -> bool:
        async with self.condition:
            if self.active < self.concurrent:
                self.active += 1
                return True
            if self.waiting >= self.queued:
                return False
            self.waiting += 1
            try:
                await asyncio.wait_for(
                    self.condition.wait_for(lambda: self.active < self.concurrent),
                    self.timeout,
                )
                self.active += 1
                return True
            except (asyncio.TimeoutError, TimeoutError):  # distinct classes on 3.10
                return False
            finally:
                self.waiting -= 1

    async def release(self) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify(1)


def create_web_app(
    chat: WebChat,
    *,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    max_queued_runs: int = DEFAULT_MAX_QUEUED_RUNS,
    queue_timeout_seconds: float = DEFAULT_QUEUE_TIMEOUT_SECONDS,
    demo: bool = False,
) -> Any:
    """``demo`` serves the time-travel debugger at ``/`` (it is always at ``/debug``)."""
    _require_web_dependencies()
    from starlette.applications import Starlette
    from starlette.datastructures import Headers, MutableHeaders
    from starlette.exceptions import HTTPException
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.requests import Request, cookie_parser
    from starlette.responses import (
        HTMLResponse,
        JSONResponse,
        PlainTextResponse,
        RedirectResponse,
        StreamingResponse,
    )
    from starlette.routing import Route

    from ext.demo import Timeline

    admission = RunAdmission(
        max_concurrent_runs, max_queued_runs, queue_timeout_seconds
    )

    class AdmittedStreamingResponse(StreamingResponse):
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                await admission.release()

    timeline = Timeline()
    nonce = secrets.token_urlsafe(16)
    page = PAGE_PATH.read_text(encoding="utf-8").replace("__AGENT_RT_NONCE__", nonce)
    debug_page = DEBUG_PAGE_PATH.read_text(encoding="utf-8").replace(
        "__AGENT_RT_NONCE__", nonce
    )
    # Cookies ignore ports, so each launch gets its own name and two servers never clobber.
    cookie = "agent_rt_" + secrets.token_hex(4)
    headers = {
        **SECURITY_HEADERS,
        "Content-Security-Policy": CONTENT_SECURITY_POLICY.format(nonce=nonce),
    }

    def signed_in(scope: Any) -> bool:
        sent = cookie_parser(Headers(scope=scope).get("cookie", ""))
        return _same(sent.get(cookie, ""), chat.token)

    class Guard:
        """Requires the launch cookie on /api/ and adds security headers to every response."""

        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            # The token only reaches the browser through the launch link, as an HttpOnly cookie.
            if scope["path"].startswith("/api/") and not (
                signed_in(scope) and API_HEADER in Headers(scope=scope)
            ):
                response = JSONResponse({"detail": "invalid token"}, status_code=403)
                response.headers.update(headers)
                return await response(scope, receive, send)

            async def send_with_headers(message: Any) -> None:
                if message["type"] == "http.response.start":
                    MutableHeaders(scope=message).update(headers)
                await send(message)

            await self.app(scope, receive, send_with_headers)

    async def body(request: Request) -> dict[str, Any]:
        # Stop reading as soon as the cap is passed instead of buffering the whole body first.
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                raise HTTPException(413, "request is too large")
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise HTTPException(400, "expected a JSON object")
        return data

    def text_field(data: Mapping[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(400, f"{key} must be a non-empty string")
        return value.strip()

    async def index(request: Request) -> Any:
        token = request.query_params.get("token")
        if token is not None:
            if not chat.redeem(token):
                return PlainTextResponse("invalid token", status_code=403)
            # Trade the link's token for a cookie and drop it from the address bar.
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(cookie, chat.token, httponly=True, samesite="strict")
            return response
        return serve(request, debug_page if demo else page)

    def serve(request: Request, html: str) -> Any:
        if not signed_in(request.scope):
            return PlainTextResponse(
                "Open the link agent-rt web printed in your terminal.", status_code=403
            )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    async def debug(request: Request) -> Any:
        return serve(request, debug_page)

    async def debug_state(request: Request) -> Any:
        return JSONResponse(await timeline.ensure_started())

    async def debug_reset(request: Request) -> Any:
        return JSONResponse(await timeline.reset())

    async def debug_fork(request: Request) -> Any:
        data = await body(request)
        node = text_field(data, "node")
        output = data.get("tool_output")
        budget = data.get("budget")
        if output is not None and (
            not isinstance(output, str) or len(output) > MAX_FORK_TOOL_OUTPUT
        ):
            raise HTTPException(
                400,
                f"tool_output must be a string of at most {MAX_FORK_TOOL_OUTPUT} characters",
            )
        if budget is not None and (
            type(budget) is not int or not 1 <= budget <= MAX_FORK_BUDGET
        ):
            raise HTTPException(
                400, f"budget must be an integer from 1 to {MAX_FORK_BUDGET}"
            )
        try:
            return JSONResponse(
                await timeline.fork(node, tool_output=output, budget=budget)
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(409, str(exc)) from exc

    async def info(request: Request) -> Any:
        return JSONResponse(chat.info())

    async def models(request: Request) -> Any:
        return JSONResponse({"models": list(await chat.all_models())})

    async def approval(request: Request) -> Any:
        data = await body(request)
        decision = data.get("decision")
        if decision not in {"approve", "approve_session", "deny"}:
            raise HTTPException(400, "unknown decision")
        if not chat.answer(text_field(data, "id"), decision):
            raise HTTPException(404, "approval is no longer pending")
        return JSONResponse({"ok": True})

    async def send(request: Request) -> Any:
        data = await body(request)
        text = text_field(data, "text")
        model = text_field(data, "model")
        try:
            history = parse_history(data.get("messages", []))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid messages: {exc}") from exc
        if not await admission.acquire():
            raise HTTPException(429, "too many active or queued runs; retry shortly")
        return AdmittedStreamingResponse(
            chat.chat(
                text,
                model=model,
                history=history,
                auto_approve=data.get("auto_approve") is True,
            ),
            media_type="application/x-ndjson",
        )

    async def http_error(request: Request, exc: Any) -> Any:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return Starlette(
        routes=[
            Route("/", index),
            Route("/debug", debug),
            Route("/api/debug", debug_state),
            Route("/api/debug/fork", debug_fork, methods=["POST"]),
            Route("/api/debug/reset", debug_reset, methods=["POST"]),
            Route("/api/info", info),
            Route("/api/models", models),
            Route("/api/approval", approval, methods=["POST"]),
            Route("/api/chat", send, methods=["POST"]),
        ],
        middleware=[
            # Rejects DNS-rebinding requests whose Host is not a loopback name.
            Middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_HOSTS)),
            Middleware(Guard),
        ],
        exception_handlers={HTTPException: http_error},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-rt web",
        description="Agent RT chat in your browser, served only on this computer.",
    )
    parser.add_argument("-m", "--model", help="provider model name")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to the YAML config file (default: agent-rt.yaml, then ~/.agent-rt/config.yaml)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="local port (default: 8765)"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="do not open a browser tab"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="open the time-travel debugger on the offline mock model (no config or API key)",
    )
    parser.add_argument("--max-turns", type=_positive_int, default=16)
    parser.add_argument(
        "--max-concurrent-runs",
        type=_positive_int,
        default=DEFAULT_MAX_CONCURRENT_RUNS,
        help="maximum simultaneous agent runs (default: 4)",
    )
    parser.add_argument(
        "--max-queued-runs",
        type=_non_negative_int,
        default=DEFAULT_MAX_QUEUED_RUNS,
        help="maximum requests waiting for a run slot (default: 4)",
    )
    parser.add_argument("--max-output-tokens", type=_positive_int)
    parser.add_argument("--timeout", type=float, help="per-prompt timeout in seconds")
    parser.add_argument(
        "--version", action="version", version=f"agent-rt web {_package_version()}"
    )
    return parser


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_web_dependencies()
    except RuntimeError as exc:
        print(f"agent-rt web: {exc}", file=sys.stderr)
        return 2
    if args.demo:
        return await _serve(args, _demo_chat(), demo=True)
    try:
        config, builtin = web_config(args.config)
        configure_runtime(config)
        _CLI_SETTINGS.clear()
        _CLI_SETTINGS.update(config.settings.get("cli") or {})
        provider = load_model(config)
        if builtin:
            # No configured models to start from, so this is the one startup network call.
            models = await discover_builtin_models(provider)
            model = (args.model or "").strip() or models[0]
        else:
            model = await _startup_model(provider, args.model)
            models = await _provider_model_ids(provider)
    except Exception as exc:  # noqa: BLE001 - configuration can fail in providers
        print(f"agent-rt web: configuration error: {exc}", file=sys.stderr)
        return 2
    workspace = Path.cwd()
    skills, skill_warnings = _discover_cli_skills(workspace)
    for warning in skill_warnings:
        print(f"agent-rt web: skill warning: {warning}", file=sys.stderr)
    if consequences := _skill_warning_consequences("agent-rt web", skill_warnings):
        print(consequences, file=sys.stderr)

    def make_session(
        session_model: str, approve: Callable[[str, Mapping[str, Any]], Any]
    ) -> ChatSession:
        # One session per request, so concurrent tabs never share tools or approvals.
        return build_workspace_session(
            provider,
            session_model,
            workspace,
            approval_callback=approve,
            skills=skills,
            autosave=False,
            limits=AgentRunLimits(
                max_turns=args.max_turns, timeout_seconds=args.timeout
            ),
            max_output_tokens=args.max_output_tokens,
        )

    chat = WebChat(
        make_session,
        model=model,
        models=models,
        workspace=workspace,
        # The page asks for these after it loads, so a slow endpoint never delays startup.
        # Built-in models were already discovered at startup.
        discover=None if builtin else getattr(provider, "discover_models", None),
    )
    return await _serve(args, chat)


def _demo_chat() -> WebChat:
    def make_session(model: str, approve: Any) -> ChatSession:
        raise RuntimeError(
            "chat is off in --demo mode; restart agent-rt web without --demo to chat"
        )

    return WebChat(make_session, model="mock", models=(), workspace=Path.cwd())


async def _serve(args: argparse.Namespace, chat: WebChat, *, demo: bool = False) -> int:
    uvicorn = _require_web_dependencies()
    url = f"http://127.0.0.1:{args.port}/?token={chat.token}"
    server = uvicorn.Server(
        uvicorn.Config(
            create_web_app(
                chat,
                max_concurrent_runs=args.max_concurrent_runs,
                max_queued_runs=args.max_queued_runs,
                demo=demo,
            ),
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
        )
    )
    # Flushed: the link is the only way in, even when stdout is a pipe or a log file.
    print(
        f"Agent RT web UI: {url}\n(workspace: {chat.workspace}; Ctrl+C to stop)",
        flush=True,
    )

    async def open_browser() -> None:
        while not server.started:
            await asyncio.sleep(0.02)
        await asyncio.to_thread(
            webbrowser.open, f"http://127.0.0.1:{args.port}/?token={chat.launch_code}"
        )

    opener = None if args.no_open else asyncio.create_task(open_browser())
    await server.serve()
    if opener:
        opener.cancel()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


__all__ = ("WebChat", "build_parser", "create_web_app", "main", "parse_history")


if __name__ == "__main__":
    raise SystemExit(main())

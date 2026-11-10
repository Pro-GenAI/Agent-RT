"""Browser trust boundaries for the optional compatible API server."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any
from urllib.parse import urlsplit


class APIBrowserGuard:
    def __init__(
        self,
        app: Any,
        *,
        authenticated: bool,
        allowed_hosts: Collection[str],
        allowed_origins: Collection[str],
    ) -> None:
        self.app = app
        self.authenticated = authenticated
        self.hosts = {"localhost", "127.0.0.1", "::1", *allowed_hosts}
        self.origins = frozenset(allowed_origins)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", ())
        hosts = [value.decode("latin-1") for key, value in headers if key == b"host"]
        origins = [
            value.decode("latin-1") for key, value in headers if key == b"origin"
        ]
        host = hosts[0] if len(hosts) == 1 else ""
        try:
            parsed = urlsplit("//" + host)
            hostname = parsed.hostname
            # Accessing port also rejects malformed/non-numeric ports.
            _ = parsed.port
            valid_host = (
                bool(hostname)
                and not (
                    parsed.username
                    or parsed.password
                    or parsed.path
                    or parsed.query
                    or parsed.fragment
                )
                and not any(
                    character.isspace() or ord(character) < 32 for character in host
                )
            )
        except ValueError:
            valid_host = False
            hostname = None
        scheme = {"ws": "http", "wss": "https"}.get(
            scope.get("scheme"), scope.get("scheme")
        )
        same_origin = f"{scheme}://{host}"
        denied = (
            not valid_host
            or (not self.authenticated and hostname not in self.hosts)
            or len(origins) > 1
            or (
                origins and origins[0] != same_origin and origins[0] not in self.origins
            )
        )
        if denied:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
            else:
                from starlette.responses import JSONResponse

                response = JSONResponse(
                    {
                        "error": {
                            "message": "Untrusted Host or Origin.",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=403,
                )
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

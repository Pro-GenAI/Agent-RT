"""Lightweight OpenAI-compatible HTTP client and rate-limit gate."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from ext.runtime.models.conversion import _provider_value
from ext.runtime.models.openai_payloads import (
    _OpenAICompatibleHTTPError,
    _parse_rate_limit_duration,
    _response_headers,
)
from ext.runtime.models.sdk import BatchJob, ModelCatalogEntry
from ext.runtime.resilience.rate_limits import RateLimitExceededError

MAX_RATE_LIMIT_BLOCK_SECONDS = 300.0


class _OpenAIRateLimitGate:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.clock = clock
        self._blocked_until: dict[str, float] = {}

    def retry_after_seconds(self, model: str) -> float:
        remaining = self._blocked_until.get(model, 0.0) - self.clock()
        if remaining <= 0:
            self._blocked_until.pop(model, None)
            return 0.0
        # Float subtraction on a large monotonic clock can overshoot the cap.
        return min(remaining, MAX_RATE_LIMIT_BLOCK_SECONDS)

    def check(self, model: str) -> None:
        remaining = self.retry_after_seconds(model)
        if remaining > 0:
            raise RateLimitExceededError("model:" + model, remaining)

    def update(
        self,
        model: str,
        headers: Mapping[str, str],
        *,
        force: bool = False,
    ) -> float:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        waits: list[float] = []
        request_remaining = normalized.get("x-ratelimit-remaining-requests")
        token_remaining = normalized.get("x-ratelimit-remaining-tokens")
        try:
            requests_exhausted = (
                request_remaining is not None and float(request_remaining) <= 0
            )
        except ValueError:
            requests_exhausted = False
        try:
            tokens_exhausted = (
                token_remaining is not None and float(token_remaining) <= 0
            )
        except ValueError:
            tokens_exhausted = False
        if requests_exhausted or force:
            for key in (
                "x-ratelimit-reset-requests",
                "anthropic-ratelimit-requests-reset",
            ):
                reset = _parse_rate_limit_duration(normalized.get(key))
                if reset is not None:
                    waits.append(reset)
        if tokens_exhausted or force:
            for key in (
                "x-ratelimit-reset-tokens",
                "anthropic-ratelimit-tokens-reset",
            ):
                reset = _parse_rate_limit_duration(normalized.get(key))
                if reset is not None:
                    waits.append(reset)
        retry_after = _parse_rate_limit_duration(normalized.get("retry-after"))
        if retry_after is not None:
            waits.append(retry_after)
        if not waits:
            return self.retry_after_seconds(model)
        # The wait comes from the remote server; bound it so a hostile or
        # buggy Retry-After cannot block a model for the process lifetime.
        blocked_until = self.clock() + min(max(waits), MAX_RATE_LIMIT_BLOCK_SECONDS)
        self._blocked_until[model] = max(
            self._blocked_until.get(model, 0.0), blocked_until
        )
        return self.retry_after_seconds(model)


def _compatible_proxy_url(base_url: str) -> str | None:
    from urllib.parse import urlsplit
    from urllib.request import getproxies, proxy_bypass

    parsed = urlsplit(base_url)
    host = parsed.hostname
    if host and proxy_bypass(host):
        return None
    proxies = getproxies()
    value = proxies.get(parsed.scheme) or proxies.get("all")
    if not value:
        return None
    return value if "://" in value else f"http://{value}"


class _OpenAICompatibleCoreResponse:
    def __init__(self, response: Any, body: bytes) -> None:
        self._response = response
        self._body = body
        self.status_code = int(response.status)
        self.headers = _response_headers(response)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            detail = self._body.decode("utf-8", errors="replace")[:500]
            raise _OpenAICompatibleHTTPError(
                f"HTTP {self.status_code} from OpenAI-compatible endpoint: {detail}",
                status_code=self.status_code,
                headers=self.headers,
            )

    def json(self) -> Any:
        return json.loads(self._body)


class _OpenAICompatibleCoreStreamResponse:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.status_code = int(response.status)
        self.headers = _response_headers(response)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _OpenAICompatibleHTTPError(
                f"HTTP {self.status_code} from OpenAI-compatible endpoint",
                status_code=self.status_code,
                headers=self.headers,
            )

    async def aiter_lines(self) -> AsyncIterator[str]:
        buffer = b""
        async for chunk in self._response.aiter_stream():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.rstrip(b"\r").decode("utf-8", errors="replace")
        if buffer:
            yield buffer.rstrip(b"\r").decode("utf-8", errors="replace")


class _OpenAICompatibleCoreStreamContext:
    def __init__(
        self,
        client: _OpenAICompatibleCoreClient,
        path: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._client = client
        self._path = path
        self._payload = payload
        self._context: Any = None

    async def __aenter__(self) -> _OpenAICompatibleCoreStreamResponse:
        context = self._client._pool.stream(
            b"POST",
            self._client._url(self._path),
            headers=self._client._headers,
            content=json.dumps(self._payload, separators=(",", ":")).encode("utf-8"),
            extensions=self._client._timeout_extensions,
        )
        self._context = context
        response = await context.__aenter__()
        return _OpenAICompatibleCoreStreamResponse(response)

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> Any:
        if self._context is None:
            raise RuntimeError("stream context exited before it was entered")
        return await self._context.__aexit__(exc_type, exc, tb)


class _OpenAICompatibleCoreClient:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        import httpcore

        self._base_url = base_url.rstrip("/") + "/"
        self._headers = [
            (b"authorization", f"Bearer {api_key}".encode()),
            (b"content-type", b"application/json"),
        ]
        self._timeout_extensions = {
            "timeout": {
                "connect": 600.0,
                "read": 600.0,
                "write": 600.0,
                "pool": 600.0,
            }
        }
        proxy_url = _compatible_proxy_url(base_url)
        pool_options: dict[str, Any] = {
            "max_connections": 100,
            "max_keepalive_connections": 20,
            "keepalive_expiry": 5.0,
        }
        if proxy_url is None:
            self._pool = httpcore.AsyncConnectionPool(**pool_options)
        elif proxy_url.lower().startswith(("socks5://", "socks5h://")):
            self._pool = httpcore.AsyncSOCKSProxy(proxy_url, **pool_options)
        else:
            self._pool = httpcore.AsyncHTTPProxy(proxy_url, **pool_options)

    def _url(self, path: str) -> bytes:
        return (self._base_url + path.lstrip("/")).encode("utf-8")

    async def get(
        self, path: str, *, params: Mapping[str, Any] | None = None
    ) -> _OpenAICompatibleCoreResponse:
        if params:
            from urllib.parse import urlencode

            query = urlencode(
                {key: value for key, value in params.items() if value is not None}
            )
            if query:
                path = f"{path}?{query}"
        response = await self._pool.request(
            b"GET",
            self._url(path),
            headers=self._headers,
            extensions=self._timeout_extensions,
        )
        return _OpenAICompatibleCoreResponse(response, response.content)

    async def post(
        self, path: str, *, json: Mapping[str, Any]
    ) -> _OpenAICompatibleCoreResponse:
        response = await self._pool.request(
            b"POST",
            self._url(path),
            headers=self._headers,
            content=__import__("json")
            .dumps(json, separators=(",", ":"))
            .encode("utf-8"),
            extensions=self._timeout_extensions,
        )
        return _OpenAICompatibleCoreResponse(response, response.content)

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any],
    ) -> _OpenAICompatibleCoreStreamContext:
        if method.upper() != "POST":
            raise ValueError("OpenAI-compatible client only supports POST")
        return _OpenAICompatibleCoreStreamContext(self, path, json)

    async def aclose(self) -> None:
        await self._pool.aclose()


class _OpenAICompatibleHTTPStream:
    def __init__(self, client: Any, params: Mapping[str, Any]) -> None:
        self._client = client
        self._params = dict(params)
        self.headers: dict[str, str] = {}

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST",
            "chat/completions",
            json=self._params,
        ) as response:
            self.headers = dict(response.headers)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    if payload == "[DONE]":
                        break
                    continue
                yield json.loads(payload)


class _OpenAICompatibleCompletions:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.last_response_headers: dict[str, str] = {}

    async def create(self, **params: Any) -> Any:
        if params.get("stream"):
            return _OpenAICompatibleHTTPStream(self._client, params)
        response = await self._client.post("chat/completions", json=params)
        self.last_response_headers = dict(response.headers)
        response.raise_for_status()
        return response.json()


class _OpenAICompatibleChat:
    def __init__(self, client: Any) -> None:
        self.completions = _OpenAICompatibleCompletions(client)


class _OpenAICompatibleEmbeddings:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def create(self, **params: Any) -> Any:
        response = await self._client.post("embeddings", json=params)
        response.raise_for_status()
        return response.json()


class _OpenAICompatibleModels:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self) -> Any:
        response = await self._client.get("models")
        response.raise_for_status()
        return response.json()

    async def retrieve(self, model: str) -> Any:
        response = await self._client.get(f"models/{model}")
        response.raise_for_status()
        return response.json()


class _OpenAICompatibleBatches:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def create(self, **params: Any) -> Any:
        response = await self._client.post("batches", json=params)
        response.raise_for_status()
        return response.json()

    async def retrieve(self, batch_id: str) -> Any:
        response = await self._client.get(f"batches/{batch_id}")
        response.raise_for_status()
        return response.json()

    async def list(self, *, after: str | None = None, limit: int | None = None) -> Any:
        response = await self._client.get(
            "batches", params={"after": after, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    async def cancel(self, batch_id: str) -> Any:
        response = await self._client.post(f"batches/{batch_id}/cancel", json={})
        response.raise_for_status()
        return response.json()


class _OpenAICompatibleHTTPClient:
    def __init__(self, *, base_url: str, api_key: str, core: Any = None) -> None:
        # `core` replaces the httpcore pool with another get/post/stream
        # transport (the compat layer adapts an injected httpx client).
        self._client = core or _OpenAICompatibleCoreClient(
            base_url=base_url, api_key=api_key
        )
        self.chat = _OpenAICompatibleChat(self._client)
        self.embeddings = _OpenAICompatibleEmbeddings(self._client)
        self.models = _OpenAICompatibleModels(self._client)
        self.batches = _OpenAICompatibleBatches(self._client)

    async def close(self) -> None:
        await self._client.aclose()


def _batch_job(value: Any, *, anthropic: bool = False) -> BatchJob:
    status_key = "processing_status" if anthropic else "status"
    return BatchJob(
        id=str(_provider_value(value, "id", "")),
        status=str(_provider_value(value, status_key, "")),
        raw=value,
    )


def _model_catalog_entry(value: Any) -> ModelCatalogEntry:
    created = _provider_value(value, "created_at", _provider_value(value, "created"))
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        created_at = datetime.fromtimestamp(created, tz=timezone.utc)
    elif isinstance(created, datetime):
        created_at = created
    elif isinstance(created, str):
        try:
            created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            created_at = None
    else:
        created_at = None
    return ModelCatalogEntry(
        id=str(_provider_value(value, "id", "")),
        created_at=created_at,
        display_name=(
            str(_provider_value(value, "display_name"))
            if _provider_value(value, "display_name") is not None
            else None
        ),
        object=(
            str(_provider_value(value, "object"))
            if _provider_value(value, "object") is not None
            else None
        ),
        owned_by=(
            str(_provider_value(value, "owned_by"))
            if _provider_value(value, "owned_by") is not None
            else None
        ),
        raw=value,
    )


def _model_catalog_values(value: Any) -> tuple[ModelCatalogEntry, ...]:
    data = _provider_value(value, "data", value)
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        data = ()
    return tuple(entry for item in data if (entry := _model_catalog_entry(item)).id)

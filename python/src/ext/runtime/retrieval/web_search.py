"""The environment-configured web search provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

import agent_rt as _core
from ext.runtime.retrieval.registry import (
    RetrievalKind,
    RetrievalQuery,
    RetrievalResult,
)

AGENT_RT_WEB_SEARCH_TOOL_ENV = "AGENT_RT_WEB_SEARCH_TOOL"


AGENT_RT_WEB_SEARCH_API_KEY_ENV = (
    "AGENT_RT_WEB_SEARCH_API_KEY"  # pragma: allowlist secret
)


AGENT_RT_WEB_SEARCH_TOKEN_ENV = "AGENT_RT_WEB_SEARCH_TOKEN"  # pragma: allowlist secret


AGENT_RT_WEB_SEARCH_TIMEOUT_ENV = "AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS"


_WEB_SEARCH_CREDENTIAL_ENV = {
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_SEARCH_API_KEY",
    "serper": "SERPER_API_KEY",
}


class EnvironmentWebSearchProvider:
    kind: RetrievalKind = "web"

    def __init__(
        self,
        tool: str,
        credential: str,
        *,
        timeout_seconds: float = 20.0,
        client: Any = None,
    ) -> None:
        normalized = tool.strip().lower()
        if normalized not in _WEB_SEARCH_CREDENTIAL_ENV:
            supported = ", ".join(sorted(_WEB_SEARCH_CREDENTIAL_ENV))
            raise ValueError(
                f"unsupported web search tool {tool!r}; expected one of: {supported}"
            )
        if not credential.strip():
            raise ValueError("web search credential must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("web search timeout must be positive")
        self.tool = normalized
        self._credential = credential
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None, *, client: Any = None
    ) -> EnvironmentWebSearchProvider:
        env = _core.runtime_settings() if environment is None else environment
        tool = str(env.get(AGENT_RT_WEB_SEARCH_TOOL_ENV, "")).strip().lower()
        if not tool:
            raise RuntimeError(
                f"web search requires {AGENT_RT_WEB_SEARCH_TOOL_ENV}=tavily|brave|serper"
            )
        credential_env = _WEB_SEARCH_CREDENTIAL_ENV.get(tool)
        if credential_env is None:
            supported = ", ".join(sorted(_WEB_SEARCH_CREDENTIAL_ENV))
            raise ValueError(
                f"unsupported web search tool {tool!r}; expected one of: {supported}"
            )
        credential = (
            env.get(credential_env)
            or env.get(AGENT_RT_WEB_SEARCH_API_KEY_ENV)
            or env.get(AGENT_RT_WEB_SEARCH_TOKEN_ENV)
        )
        if not credential:
            raise RuntimeError(
                "web search credential is missing; set "
                f"{credential_env}, {AGENT_RT_WEB_SEARCH_API_KEY_ENV}, or {AGENT_RT_WEB_SEARCH_TOKEN_ENV}"
            )
        timeout_raw = env.get(AGENT_RT_WEB_SEARCH_TIMEOUT_ENV, "20")
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{AGENT_RT_WEB_SEARCH_TIMEOUT_ENV} must be numeric"
            ) from exc
        return cls(
            tool, str(credential), timeout_seconds=timeout_seconds, client=client
        )

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        if not query.text.strip():
            raise ValueError("web search query must not be empty")
        if self.tool == "tavily":
            payload = await self._tavily(query)
            items = payload.get("results", ())
            return tuple(
                RetrievalResult(
                    id=str(item.get("url") or index),
                    title=str(item.get("title") or item.get("url") or ""),
                    content=str(item.get("content") or item.get("snippet") or ""),
                    score=(
                        float(item["score"]) if item.get("score") is not None else None
                    ),
                    uri=str(item.get("url")) if item.get("url") else None,
                    metadata={"provider": "tavily"},
                )
                for index, item in enumerate(items[: query.limit])
                if isinstance(item, Mapping)
            )
        if self.tool == "brave":
            payload = await self._brave(query)
            web = payload.get("web", {})
            items = web.get("results", ()) if isinstance(web, Mapping) else ()
            return tuple(
                RetrievalResult(
                    id=str(item.get("url") or index),
                    title=str(item.get("title") or item.get("url") or ""),
                    content=str(item.get("description") or ""),
                    uri=str(item.get("url")) if item.get("url") else None,
                    metadata={"provider": "brave"},
                )
                for index, item in enumerate(items[: query.limit])
                if isinstance(item, Mapping)
            )
        payload = await self._serper(query)
        items = payload.get("organic", ())
        return tuple(
            RetrievalResult(
                id=str(item.get("link") or index),
                title=str(item.get("title") or item.get("link") or ""),
                content=str(item.get("snippet") or ""),
                uri=str(item.get("link")) if item.get("link") else None,
                metadata={
                    "provider": "serper",
                    **(
                        {"position": item.get("position")}
                        if item.get("position") is not None
                        else {}
                    ),
                },
            )
            for index, item in enumerate(items[: query.limit])
            if isinstance(item, Mapping)
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> Mapping[str, Any]:
        if self._client is not None:
            response = await self._client.request(method, url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("web search provider returned a non-object JSON response")
        return payload

    async def _tavily(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            "https://api.tavily.com/search",
            json={
                **dict(query.filters),
                "api_key": self._credential,
                "query": query.text,
                "max_results": query.limit,
            },
        )

    async def _brave(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            params={**dict(query.filters), "q": query.text, "count": query.limit},
            headers={"X-Subscription-Token": self._credential},
        )

    async def _serper(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            "https://google.serper.dev/search",
            json={**dict(query.filters), "q": query.text, "num": query.limit},
            headers={"X-API-KEY": self._credential},
        )

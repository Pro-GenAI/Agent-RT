"""Retrieval queries, providers, rerankers, and the retrieval registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

RetrievalKind = Literal["web", "enterprise", "file", "knowledge", "database"]


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    limit: int = 10
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    title: str
    content: Any
    score: float | None = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalProvider(Protocol):
    kind: RetrievalKind

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]: ...


@runtime_checkable
class RetrievalReranker(Protocol):
    async def rerank(
        self,
        query: RetrievalQuery,
        results: Sequence[RetrievalResult],
        *,
        k: int | None = None,
    ) -> Sequence[RetrievalResult]: ...


class RetrievalRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, RetrievalProvider] = {}
        self._rerankers: dict[str, RetrievalReranker] = {}
        self._reranker_ks: dict[str, int] = {}

    def register(
        self,
        name: str,
        provider: RetrievalProvider,
        *,
        reranker: RetrievalReranker | None = None,
        reranker_k: int | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("retrieval provider name must not be empty")
        if reranker_k is not None:
            if reranker_k < 1:
                raise ValueError("reranker k must be at least 1")
            if reranker is None:
                raise ValueError("reranker k requires a reranker")
        self._providers[name] = provider
        if reranker is None:
            self._rerankers.pop(name, None)
            self._reranker_ks.pop(name, None)
            return
        self._rerankers[name] = reranker
        if reranker_k is None:
            self._reranker_ks.pop(name, None)
        else:
            self._reranker_ks[name] = reranker_k

    async def search(
        self, name: str, query: RetrievalQuery
    ) -> tuple[RetrievalResult, ...]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(name)
        results = tuple(await provider.search(query))
        reranker = self._rerankers.get(name)
        if reranker is not None:
            reranker_k = self._reranker_ks.get(name)
            if reranker_k is None:
                results = tuple(await reranker.rerank(query, results))
            else:
                results = tuple(await reranker.rerank(query, results, k=reranker_k))
                results = results[:reranker_k]
        return results[: query.limit]

"""Vector database configuration and the environment-configured vector DB provider."""

from __future__ import annotations

import math
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

import agent_rt as _core
from ext.runtime.models.interfaces import EmbeddingModelProvider, EmbeddingRequest
from ext.runtime.retrieval.registry import (
    RetrievalKind,
    RetrievalProvider,
    RetrievalQuery,
    RetrievalResult,
)

AGENT_RT_VECTOR_DB_ENV = "AGENT_RT_VECTOR_DB"


AGENT_RT_VECTOR_DB_COLLECTION_ENV = "AGENT_RT_VECTOR_DB_COLLECTION"


AGENT_RT_VECTOR_DB_URL_ENV = "AGENT_RT_VECTOR_DB_URL"


AGENT_RT_VECTOR_DB_OPTION_PREFIX = "AGENT_RT_VECTOR_DB_OPTION_"


@dataclass(frozen=True)
class VectorDBConfig:
    backend: str
    collection: str | None = None
    url: str | None = None
    options: Mapping[str, str] = field(default_factory=dict)


class VectorDBProviderRegistry:
    """Resolve vector-backed retrieval providers without coupling Agent RT to vendor SDKs."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, name: str, factory: Any, *, replace: bool = False) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("vector DB backend name must not be empty")
        if not callable(factory):
            raise TypeError("vector DB provider factory must be callable")
        if normalized in self._factories and not replace:
            raise ValueError(f"vector DB backend already registered: {normalized}")
        self._factories[normalized] = factory

    def create(self, config: VectorDBConfig) -> RetrievalProvider:
        backend = config.backend.strip().lower()
        factory = self._factories.get(backend)
        if factory is None:
            supported = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"unsupported vector DB backend {config.backend!r}; registered backends: {supported}"
            )
        provider = factory(config)
        if not isinstance(provider, RetrievalProvider):
            raise TypeError(
                f"vector DB backend {backend!r} factory must return a RetrievalProvider"
            )
        return provider


def vector_db_config_from_environment(
    environment: Mapping[str, str] | None = None,
) -> VectorDBConfig:
    env = _core.runtime_settings() if environment is None else environment
    backend = str(env.get(AGENT_RT_VECTOR_DB_ENV, "")).strip().lower()
    if not backend:
        raise RuntimeError(f"vector DB selection requires {AGENT_RT_VECTOR_DB_ENV}")
    collection = str(env.get(AGENT_RT_VECTOR_DB_COLLECTION_ENV, "")).strip() or None
    url = str(env.get(AGENT_RT_VECTOR_DB_URL_ENV, "")).strip() or None
    options = {
        key[len(AGENT_RT_VECTOR_DB_OPTION_PREFIX) :].lower(): str(value)
        for key, value in env.items()
        if key.startswith(AGENT_RT_VECTOR_DB_OPTION_PREFIX)
        and key != AGENT_RT_VECTOR_DB_OPTION_PREFIX
        and str(value).strip()
    }
    return VectorDBConfig(
        backend=backend,
        collection=collection,
        url=url,
        options=options,
    )


def vector_db_provider_from_environment(
    registry: VectorDBProviderRegistry,
    environment: Mapping[str, str] | None = None,
) -> RetrievalProvider:
    """Select a registered vector DB adapter from environment configuration."""
    return registry.create(vector_db_config_from_environment(environment))


AGENT_RT_VECTOR_DB_API_KEY_ENV = (
    "AGENT_RT_VECTOR_DB_API_KEY"  # pragma: allowlist secret
)


AGENT_RT_VECTOR_DB_TIMEOUT_ENV = "AGENT_RT_VECTOR_DB_TIMEOUT_SECONDS"


POPULAR_VECTOR_DB_BACKENDS = ("chroma", "milvus", "pinecone", "qdrant", "weaviate")


_VECTOR_DB_CREDENTIAL_ENV = {
    "chroma": "CHROMA_API_KEY",
    "milvus": "MILVUS_TOKEN",
    "pinecone": "PINECONE_API_KEY",
    "qdrant": "QDRANT_API_KEY",
    "weaviate": "WEAVIATE_API_KEY",
}


def _vector_db_http_url(value: str, *, backend: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{backend} vector DB URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"{backend} vector DB URL must not contain embedded credentials"
        )
    return candidate


def _vector_db_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EnvironmentVectorDBProvider:
    kind: RetrievalKind = "knowledge"

    def __init__(
        self,
        config: VectorDBConfig,
        *,
        embedding_provider: EmbeddingModelProvider | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        client: Any = None,
    ) -> None:
        backend = config.backend.strip().lower()
        if backend not in POPULAR_VECTOR_DB_BACKENDS:
            supported = ", ".join(POPULAR_VECTOR_DB_BACKENDS)
            raise ValueError(
                f"unsupported built-in vector DB backend {backend!r}; expected one of: {supported}"
            )
        if not config.url:
            raise RuntimeError(
                f"{backend} vector DB requires {AGENT_RT_VECTOR_DB_URL_ENV}"
            )
        if not config.collection:
            raise RuntimeError(
                f"{backend} vector DB requires {AGENT_RT_VECTOR_DB_COLLECTION_ENV}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("vector DB timeout must be positive")
        self.config = config
        self.backend = backend
        self.url = _vector_db_http_url(config.url, backend=backend)
        self.collection = config.collection
        self.embedding_provider = embedding_provider
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        embedding_provider: EmbeddingModelProvider | None = None,
        client: Any = None,
    ) -> EnvironmentVectorDBProvider:
        env = _core.runtime_settings() if environment is None else environment
        config = vector_db_config_from_environment(env)
        if config.backend not in POPULAR_VECTOR_DB_BACKENDS:
            supported = ", ".join(POPULAR_VECTOR_DB_BACKENDS)
            raise ValueError(
                f"unsupported built-in vector DB backend {config.backend!r}; expected one of: {supported}"
            )
        credential_env = _VECTOR_DB_CREDENTIAL_ENV[config.backend]
        api_key = env.get(credential_env) or env.get(AGENT_RT_VECTOR_DB_API_KEY_ENV)
        timeout_raw = env.get(AGENT_RT_VECTOR_DB_TIMEOUT_ENV, "20")
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{AGENT_RT_VECTOR_DB_TIMEOUT_ENV} must be numeric"
            ) from exc
        return cls(
            config,
            embedding_provider=embedding_provider,
            api_key=str(api_key) if api_key else None,
            timeout_seconds=timeout_seconds,
            client=client,
        )

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        if not query.text.strip():
            raise ValueError("vector DB query must not be empty")
        vector = await self._query_vector(query)
        if self.backend == "qdrant":
            return await self._search_qdrant(query, vector)
        if self.backend == "pinecone":
            return await self._search_pinecone(query, vector)
        if self.backend == "milvus":
            return await self._search_milvus(query, vector)
        if self.backend == "weaviate":
            return await self._search_weaviate(query, vector)
        return await self._search_chroma(query, vector)

    async def _query_vector(self, query: RetrievalQuery) -> list[float]:
        supplied = query.filters.get("vector")
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            try:
                vector = [float(value) for value in supplied]
            except (TypeError, ValueError) as exc:
                raise ValueError("filters.vector must contain only numbers") from exc
            if vector and all(math.isfinite(value) for value in vector):
                return vector
            raise ValueError("filters.vector must be a non-empty finite numeric vector")
        if self.embedding_provider is None:
            raise RuntimeError(
                "vector DB text search requires an EmbeddingModelProvider or a precomputed filters.vector"
            )
        response = await self.embedding_provider.embed(
            EmbeddingRequest(input=query.text)
        )
        if not response.data:
            raise RuntimeError("embedding provider returned no vectors")
        embedding = response.data[0].embedding
        if isinstance(embedding, str):
            raise TypeError("vector DB search requires numeric embeddings")
        vector = [float(value) for value in embedding]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding provider returned an invalid vector")
        return vector

    def _filters_without_vector(self, query: RetrievalQuery) -> dict[str, Any]:
        return {key: value for key, value in query.filters.items() if key != "vector"}

    async def _request(self, method: str, url: str, **kwargs: Any) -> Mapping[str, Any]:
        if self._client is not None:
            response = await self._client.request(method, url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"{self.backend} vector DB returned a non-object JSON response"
            )
        return payload

    def _result(
        self,
        item_id: Any,
        content: Any,
        *,
        score: Any = None,
        title: Any = None,
        uri: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        resolved_metadata = dict(metadata or {})
        return RetrievalResult(
            id=str(item_id),
            title=str(title or resolved_metadata.get("title") or item_id),
            content=(
                content
                if content is not None
                else resolved_metadata.get("text")
                or resolved_metadata.get("content")
                or ""
            ),
            score=_vector_db_number(score),
            uri=(
                str(uri)
                if uri
                else (
                    str(resolved_metadata.get("url"))
                    if resolved_metadata.get("url")
                    else None
                )
            ),
            metadata={"provider": self.backend, **resolved_metadata},
        )

    async def _search_qdrant(
        self, query: RetrievalQuery, vector: list[float]
    ) -> Sequence[RetrievalResult]:
        headers = {"api-key": self.api_key} if self.api_key else {}
        filters = self._filters_without_vector(query)
        body: dict[str, Any] = {
            "vector": vector,
            "limit": query.limit,
            "with_payload": True,
        }
        if filters:
            body["filter"] = filters
        payload = await self._request(
            "POST",
            f"{self.url}/collections/{urllib.parse.quote(self.collection, safe='')}/points/search",
            json=body,
            headers=headers,
        )
        items = payload.get("result", ())
        return tuple(
            self._result(
                item.get("id", index),
                (
                    item.get("payload", {}).get("text")
                    if isinstance(item.get("payload"), Mapping)
                    else None
                ),
                score=item.get("score"),
                metadata=(
                    item.get("payload")
                    if isinstance(item.get("payload"), Mapping)
                    else {}
                ),
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    async def _search_pinecone(
        self, query: RetrievalQuery, vector: list[float]
    ) -> Sequence[RetrievalResult]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Api-Key"] = self.api_key
        filters = self._filters_without_vector(query)
        body: dict[str, Any] = {
            "vector": vector,
            "topK": query.limit,
            "includeMetadata": True,
        }
        if filters:
            body["filter"] = filters
        namespace = self.config.options.get("namespace")
        if namespace:
            body["namespace"] = namespace
        payload = await self._request(
            "POST", f"{self.url}/query", json=body, headers=headers
        )
        items = payload.get("matches", ())
        return tuple(
            self._result(
                item.get("id", index),
                (
                    item.get("metadata", {}).get("text")
                    if isinstance(item.get("metadata"), Mapping)
                    else None
                ),
                score=item.get("score"),
                metadata=(
                    item.get("metadata")
                    if isinstance(item.get("metadata"), Mapping)
                    else {}
                ),
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    def _reject_unsupported_filters(self, query: RetrievalQuery) -> None:
        """Fail closed: never silently drop filters that may enforce isolation."""
        if self._filters_without_vector(query):
            raise ValueError(
                f"the {self.backend} adapter does not support per-query filters; "
                "refusing to run an unfiltered search (use a collection per scope, "
                "or a backend that supports filters)"
            )

    async def _search_milvus(
        self, query: RetrievalQuery, vector: list[float]
    ) -> Sequence[RetrievalResult]:
        self._reject_unsupported_filters(query)
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "collectionName": self.collection,
            "data": [vector],
            "limit": query.limit,
            "outputFields": ["*"],
        }
        expression = self.config.options.get("filter")
        if expression:
            body["filter"] = expression
        payload = await self._request(
            "POST",
            f"{self.url}/v2/vectordb/entities/search",
            json=body,
            headers=headers,
        )
        data = payload.get("data", ())
        items = (
            data[0]
            if isinstance(data, Sequence) and data and isinstance(data[0], Sequence)
            else data
        )
        return tuple(
            self._result(
                item.get("id", item.get("pk", index)),
                item.get("text") or item.get("content"),
                score=item.get("score"),
                metadata=item,
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    async def _search_weaviate(
        self, query: RetrievalQuery, vector: list[float]
    ) -> Sequence[RetrievalResult]:
        self._reject_unsupported_filters(query)
        identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        class_name = self.collection
        content_field = self.config.options.get("content_field", "text")
        title_field = self.config.options.get("title_field", "title")
        uri_field = self.config.options.get("uri_field", "url")
        for value in (class_name, content_field, title_field, uri_field):
            if not identifier.fullmatch(value):
                raise ValueError(
                    "Weaviate collection and field names must be GraphQL identifiers"
                )
        vector_literal = ", ".join(format(value, ".17g") for value in vector)
        graph_query = (
            f"{{ Get {{ {class_name}(nearVector: {{vector: [{vector_literal}]}}, limit: {query.limit}) "
            f"{{ {content_field} {title_field} {uri_field} _additional {{ id distance certainty }} }} }} }}"
        )
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = await self._request(
            "POST",
            f"{self.url}/v1/graphql",
            json={"query": graph_query},
            headers=headers,
        )
        data = payload.get("data", {})
        get_value = data.get("Get", {}) if isinstance(data, Mapping) else {}
        items = get_value.get(class_name, ()) if isinstance(get_value, Mapping) else ()
        results: list[RetrievalResult] = []
        for index, item in enumerate(items if isinstance(items, Sequence) else ()):
            if not isinstance(item, Mapping):
                continue
            additional = item.get("_additional", {})
            extra = additional if isinstance(additional, Mapping) else {}
            score = extra.get("certainty")
            if score is None and extra.get("distance") is not None:
                distance = _vector_db_number(extra.get("distance"))
                score = (1.0 - distance) if distance is not None else None
            metadata = {
                key: value for key, value in item.items() if key != "_additional"
            }
            results.append(
                self._result(
                    extra.get("id", index),
                    item.get(content_field),
                    score=score,
                    title=item.get(title_field),
                    uri=item.get(uri_field),
                    metadata=metadata,
                )
            )
        return tuple(results)

    async def _search_chroma(
        self, query: RetrievalQuery, vector: list[float]
    ) -> Sequence[RetrievalResult]:
        tenant = urllib.parse.quote(
            self.config.options.get("tenant", "default_tenant"), safe=""
        )
        database = urllib.parse.quote(
            self.config.options.get("database", "default_database"), safe=""
        )
        collection = urllib.parse.quote(self.collection, safe="")
        path = f"/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/query"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": query.limit,
            "include": ["documents", "metadatas", "distances", "uris"],
        }
        filters = self._filters_without_vector(query)
        if filters:
            body["where"] = filters
        payload = await self._request(
            "POST", self.url + path, json=body, headers=headers
        )
        ids = payload.get("ids", ())
        ids = (
            ids[0]
            if isinstance(ids, Sequence) and ids and isinstance(ids[0], Sequence)
            else ids
        )
        documents = payload.get("documents", ())
        documents = (
            documents[0]
            if isinstance(documents, Sequence)
            and documents
            and isinstance(documents[0], Sequence)
            else documents
        )
        metadatas = payload.get("metadatas", ())
        metadatas = (
            metadatas[0]
            if isinstance(metadatas, Sequence)
            and metadatas
            and isinstance(metadatas[0], Sequence)
            else metadatas
        )
        distances = payload.get("distances", ())
        distances = (
            distances[0]
            if isinstance(distances, Sequence)
            and distances
            and isinstance(distances[0], Sequence)
            else distances
        )
        uris = payload.get("uris", ())
        uris = (
            uris[0]
            if isinstance(uris, Sequence) and uris and isinstance(uris[0], Sequence)
            else uris
        )
        results: list[RetrievalResult] = []
        for index, item_id in enumerate(ids if isinstance(ids, Sequence) else ()):
            metadata = (
                metadatas[index]
                if isinstance(metadatas, Sequence)
                and index < len(metadatas)
                and isinstance(metadatas[index], Mapping)
                else {}
            )
            document = (
                documents[index]
                if isinstance(documents, Sequence) and index < len(documents)
                else None
            )
            distance = (
                distances[index]
                if isinstance(distances, Sequence) and index < len(distances)
                else None
            )
            uri = (
                uris[index]
                if isinstance(uris, Sequence) and index < len(uris)
                else None
            )
            numeric_distance = _vector_db_number(distance)
            score = (
                1.0 / (1.0 + max(0.0, numeric_distance))
                if numeric_distance is not None
                else None
            )
            results.append(
                self._result(item_id, document, score=score, uri=uri, metadata=metadata)
            )
        return tuple(results)

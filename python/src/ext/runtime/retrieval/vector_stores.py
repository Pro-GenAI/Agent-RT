"""Popular vector DB backends, Bedrock Knowledge Bases, and configured vector stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import agent_rt as _core
from ext.runtime.models.interfaces import (
    EmbeddingModelProvider,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ext.runtime.retrieval.registry import (
    RetrievalKind,
    RetrievalProvider,
    RetrievalQuery,
    RetrievalRegistry,
    RetrievalResult,
)
from ext.runtime.retrieval.vector_db import (
    AGENT_RT_VECTOR_DB_COLLECTION_ENV,
    AGENT_RT_VECTOR_DB_ENV,
    AGENT_RT_VECTOR_DB_OPTION_PREFIX,
    AGENT_RT_VECTOR_DB_URL_ENV,
    POPULAR_VECTOR_DB_BACKENDS,
    EnvironmentVectorDBProvider,
    VectorDBConfig,
    VectorDBProviderRegistry,
)
from ext.runtime.types import asyncio


def register_popular_vector_db_backends(
    registry: VectorDBProviderRegistry,
    *,
    embedding_provider: EmbeddingModelProvider | None = None,
    environment: Mapping[str, str] | None = None,
    client: Any = None,
    replace: bool = False,
) -> VectorDBProviderRegistry:
    env = _core.runtime_settings() if environment is None else environment

    def factory(config: VectorDBConfig) -> RetrievalProvider:
        backend_env = {**dict(env), AGENT_RT_VECTOR_DB_ENV: config.backend}
        if config.collection is not None:
            backend_env[AGENT_RT_VECTOR_DB_COLLECTION_ENV] = config.collection
        if config.url is not None:
            backend_env[AGENT_RT_VECTOR_DB_URL_ENV] = config.url
        for key, value in config.options.items():
            backend_env[f"{AGENT_RT_VECTOR_DB_OPTION_PREFIX}{key.upper()}"] = value
        return EnvironmentVectorDBProvider.from_environment(
            backend_env,
            embedding_provider=embedding_provider,
            client=client,
        )

    for backend in POPULAR_VECTOR_DB_BACKENDS:
        registry.register(backend, factory, replace=replace)
    return registry


class _BoundEmbeddingProvider:
    """Pin an embedding provider to one configured ``model_name``."""

    def __init__(self, provider: EmbeddingModelProvider, model_name: str) -> None:
        self._provider = provider
        self._model_name = model_name

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return await self._provider.embed(replace(request, model=self._model_name))


class _BedrockKnowledgeBaseProvider:
    """AWS Bedrock Knowledge Bases Retrieve API; boto3 is optional and loaded on use."""

    kind: RetrievalKind = "knowledge"

    def __init__(self, store: Any, client_factory: Any = None) -> None:
        self.store = store
        self._client_factory = client_factory
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self.store)
            else:
                try:
                    import boto3
                except ImportError as exc:
                    raise RuntimeError(
                        "Bedrock vector stores require boto3: pip install 'agent-rt[bedrock]'"
                    ) from exc
                from botocore.config import Config as BotoConfig

                options = (
                    {"region_name": self.store.options["region"]}
                    if "region" in self.store.options
                    else {}
                )
                if self.store.api_base:
                    options["endpoint_url"] = self.store.api_base
                options["config"] = BotoConfig(
                    connect_timeout=self.store.timeout, read_timeout=self.store.timeout
                )
                self._client = boto3.client("bedrock-agent-runtime", **options)
        return self._client

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if not query.text.strip():
            raise ValueError("file_search query must not be empty")
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        client = self._get_client()
        response = await asyncio.to_thread(
            client.retrieve,
            knowledgeBaseId=self.store.store_id,
            retrievalQuery={"text": query.text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": query.limit}
            },
        )
        results = []
        for index, item in enumerate(response.get("retrievalResults", ())):
            content = item.get("content") or {}
            location = item.get("location") or {}
            uri = (
                (location.get("webLocation") or {}).get("url")
                or (location.get("s3Location") or {}).get("uri")
                or (location.get("confluenceLocation") or {}).get("url")
                or (location.get("salesforceLocation") or {}).get("url")
                or (location.get("sharePointLocation") or {}).get("url")
            )
            results.append(
                RetrievalResult(
                    id=f"{self.store.store_id}:{index}",
                    title=str(uri or self.store.name),
                    content=content.get("text") or "",
                    score=item.get("score"),
                    uri=uri,
                    metadata=item.get("metadata") or {},
                )
            )
        return tuple(results)


class _LazyConfiguredVectorStore:
    """Defer credentials, validation and SDK/client creation to first search."""

    kind: RetrievalKind = "knowledge"

    def __init__(
        self,
        store: Any,
        embedding_provider: EmbeddingModelProvider | None,
        client: Any,
        bedrock_client_factory: Any,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self._client = client
        self._bedrock_client_factory = bedrock_client_factory
        self._provider: RetrievalProvider | None = None

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if self._provider is None:
            from ext.config import require_resolved

            require_resolved(self.store.api_key, self.store.api_base)
            if self.store.provider == "bedrock":
                self._provider = _BedrockKnowledgeBaseProvider(
                    self.store, self._bedrock_client_factory
                )
            else:
                if not self.store.api_base:
                    raise ValueError(
                        f"vector store {self.store.name!r} requires api_base"
                    )
                if not self.store.collection:
                    raise ValueError(
                        f"vector store {self.store.name!r} requires collection"
                    )
                self._provider = EnvironmentVectorDBProvider(
                    VectorDBConfig(
                        backend=self.store.provider,
                        collection=self.store.collection,
                        url=self.store.api_base,
                        options=dict(self.store.options),
                    ),
                    embedding_provider=self.embedding_provider,
                    api_key=self.store.api_key,
                    timeout_seconds=self.store.timeout,
                    client=self._client,
                )
        return await self._provider.search(query)


def vector_store_registry_from_config(
    config: Any,
    *,
    embedding_provider: EmbeddingModelProvider | None = None,
    client: Any = None,
    bedrock_client_factory: Any = None,
) -> RetrievalRegistry:
    """Register store aliases without connecting to stores until they are searched."""
    registry = RetrievalRegistry()
    for store in config.vector_stores:
        bound = (
            _BoundEmbeddingProvider(embedding_provider, store.embedding_model)
            if embedding_provider is not None and store.embedding_model
            else embedding_provider
        )
        provider = _LazyConfiguredVectorStore(
            store, bound, client, bedrock_client_factory
        )
        registry.register(store.name, provider)
        if store.store_id != store.name:
            registry.register(store.store_id, provider)
    return registry


def resolve_vector_stores(
    config: Any,
    references: Sequence[str],
    *,
    model_name: str | None = None,
) -> tuple[Any, ...]:
    """Validate requested ``vector_store_ids``; unknown or disallowed ids raise ValueError."""
    resolved: list[Any] = []
    for reference in references:
        store = config.vector_store(reference) if isinstance(reference, str) else None
        if store is None:
            raise ValueError(f"unknown vector store: {reference!r}")
        if store.allowed_models and model_name not in store.allowed_models:
            raise ValueError(
                f"vector store {reference!r} is not available to this model"
            )
        if store not in resolved:
            resolved.append(store)
    if not resolved:
        raise ValueError("file_search requires at least one vector store id")
    return tuple(resolved)

"""Config-driven model providers and ``load_model``."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ext.config import require_resolved as _require_resolved
from ext.runtime.models.anthropic import AnthropicModelProvider
from ext.runtime.models.interfaces import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
)
from ext.runtime.models.openai import OpenAIModelProvider
from ext.runtime.models.provider_settings import (
    AnthropicProviderSettings,
    OpenAIProviderSettings,
)
from ext.runtime.models.sdk import DEFAULT_ANTHROPIC_BASE_URL, ModelCatalogEntry

if TYPE_CHECKING:
    import asyncio
else:
    from ext.runtime.types import asyncio


class ConfiguredModelProvider:
    """Route requests by public ``model_name`` across configured deployments.

    Several deployments sharing a ``model_name`` act as ordered fallbacks.
    Provider clients are created lazily, one per deployment.
    """

    name = "agent-rt-config"

    def __init__(
        self,
        config: Any,
        *,
        provider_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if not config.models:
            raise ValueError("model_list must contain at least one model")
        self.config = config
        self._factory = provider_factory or self._default_factory
        self._providers: dict[int, Any] = {}
        # Upstream model ids found by discover_models(), routed to the deployment that listed them.
        self._discovered: dict[str, Any] = {}
        self.discovery_errors: list[str] = []
        self.settings = _ConfiguredModelSettings(config.default_model)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.config.model_names)

    @staticmethod
    def _default_factory(deployment: Any) -> Any:
        if deployment.provider == "openai":
            options: dict[str, Any] = {"default_model": deployment.model}
            if deployment.api_base:
                options["base_url"] = deployment.api_base
            if deployment.websocket is not None:
                options["websocket"] = deployment.websocket
            elif deployment.api_base:
                options["websocket"] = False
            settings = OpenAIProviderSettings(api_key=deployment.api_key, **options)
            return OpenAIModelProvider(settings)
        settings_a = AnthropicProviderSettings(
            base_url=deployment.api_base or DEFAULT_ANTHROPIC_BASE_URL,
            api_key=deployment.api_key,
            default_model=deployment.model,
        )
        return AnthropicModelProvider(settings_a)

    def _provider(self, deployment: Any) -> Any:
        key = id(deployment)
        provider = self._providers.get(key)
        if provider is None:
            _require_resolved(deployment.api_key, deployment.api_base)
            provider = self._factory(deployment)
            self._providers[key] = provider
        return provider

    def _targets(self, requested: str | None) -> tuple[tuple[Any, str], ...]:
        """(deployment, upstream model) pairs to try in order."""
        name = requested or self.config.default_model
        deployments = self.config.deployments(name) if name else ()
        if deployments:
            return tuple((deployment, deployment.model) for deployment in deployments)
        if name in self._discovered:
            return ((self._discovered[name], name),)
        configured = ", ".join(self.model_names)
        raise ValueError(f"unknown model {name!r}; configured models: {configured}")

    async def discover_models(self, timeout: float = 10.0) -> tuple[str, ...]:
        """Ask each configured endpoint for its models and make them selectable by id.

        Best effort: an endpoint that fails or cannot list models is skipped. Configured
        ``model_name`` values always win over a discovered id with the same name.
        """
        seen: set[tuple[Any, ...]] = set()
        self.discovery_errors = []
        for deployment in self.config.models:
            endpoint = (
                deployment.provider,
                str(deployment.api_base),
                str(deployment.api_key),
            )
            if endpoint in seen:
                continue
            seen.add(endpoint)
            try:
                entries = await asyncio.wait_for(
                    self._provider(deployment).list_models(), timeout
                )
            except Exception as exc:
                self.discovery_errors.append(
                    f"{deployment.model_name}: {type(exc).__name__}: {exc}"
                )
                continue
            for entry in entries:
                if entry.id and not self.config.deployments(entry.id):
                    self._discovered.setdefault(entry.id, deployment)
        return tuple(self._discovered)

    def validate(self) -> None:
        for deployment in self.config.models:
            provider = self._provider(deployment)
            validator = getattr(
                getattr(provider, "settings", None), "validate_connection", None
            )
            if callable(validator):
                validator(check_default_model=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: Exception | None = None
        for deployment, model in self._targets(request.model):
            try:
                return await self._provider(deployment).complete(
                    replace(request, model=model)
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        last_error: Exception | None = None
        for deployment, model in self._targets(request.model):
            started = False
            try:
                provider = self._provider(deployment)
                inner = replace(request, model=model)
                if hasattr(provider, "stream"):
                    async for event in provider.stream(inner):
                        started = True
                        yield event
                else:
                    response = await provider.complete(inner)
                    yield ModelStreamEvent(type="completed", response=response)
                return
            except Exception as exc:
                if started:
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        last_error: Exception | None = None
        for deployment, model in self._targets(request.model):
            try:
                return await self._provider(deployment).embed(
                    replace(request, model=model)
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def list_models(self) -> tuple[ModelCatalogEntry, ...]:
        return tuple(
            ModelCatalogEntry(id=name, owned_by="agent-rt") for name in self.model_names
        )


@dataclass(frozen=True)
class _ConfiguredModelSettings:
    default_model: str | None = None


def load_model(
    config: Any = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    validate: bool = False,
    provider_factory: Callable[[Any], Any] | None = None,
) -> ConfiguredModelProvider:
    """Build the model provider declared by ``model_list`` in a config file.

    ``config`` is an ``AgentRTConfig``; otherwise ``config_path`` (or the first
    default file: current folder, then home folder) is loaded. The process
    environment is only used to resolve ``os.environ/NAME`` secret references.
    """
    if config is None:
        from ext.config import load_config

        config = load_config(config_path)
    provider = ConfiguredModelProvider(config, provider_factory=provider_factory)
    if validate:
        provider.validate()
    return provider

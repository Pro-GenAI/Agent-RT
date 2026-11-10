"""OpenAI and Anthropic provider settings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ext.runtime.models.sdk import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    _listed_model_ids,
    _sdk_list_models,
    _set_ws_capability,
    _validate_provider_base_url,
)


@dataclass(frozen=True)
class OpenAIProviderSettings:
    """Connection defaults for OpenAI and OpenAI-compatible model providers."""

    base_url: str = DEFAULT_OPENAI_BASE_URL
    api_key: str | None = field(default=None, repr=False)
    default_model: str | None = None
    transport: Literal["compatible", "sdk"] = "compatible"
    websocket: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_url", _validate_provider_base_url(self.base_url, "OpenAI")
        )
        if self.transport not in {"compatible", "sdk"}:
            raise ValueError("OpenAI transport must be 'compatible' or 'sdk'")
        if self.api_key is not None:
            api_key = self.api_key.strip()
            object.__setattr__(self, "api_key", api_key or None)
        if self.default_model is not None:
            default_model = self.default_model.strip()
            if not default_model:
                raise ValueError("OpenAI default_model must not be empty when provided")
            object.__setattr__(self, "default_model", default_model)

    def validate_connection(
        self,
        *,
        check_default_model: bool = False,
        client_factory: Callable[..., Any] | None = None,
        websocket_probe: Callable[..., bool] | None = None,
    ) -> tuple[str, ...]:
        probe_websocket = client_factory is None or websocket_probe is not None
        if client_factory is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                if exc.name != "openai":
                    raise
                raise RuntimeError(
                    "The OpenAI SDK is optional. Install "
                    "'agent-rt[openai]' to use OpenAI SDK startup "
                    "validation, or pass client_factory explicitly."
                ) from exc
            client_factory = OpenAI
        client = client_factory(
            base_url=self.base_url,
            api_key=self.api_key or "not-provided",
        )
        try:
            model_ids = _listed_model_ids(_sdk_list_models(client))
        except Exception as exc:
            raise ValueError(f"OpenAI startup validation failed: {exc}") from exc
        if self.websocket and probe_websocket:
            if websocket_probe is None:
                from ext.transports.openai_ws import (
                    responses_websocket_supported,
                )

                websocket_probe = responses_websocket_supported
            _set_ws_capability(
                self.base_url,
                websocket_probe(
                    base_url=self.base_url,
                    api_key=self.api_key,
                ),
            )
        if (
            check_default_model
            and self.default_model is not None
            and self.default_model not in model_ids
        ):
            raise ValueError(
                f"OpenAI default model {self.default_model!r} is not available"
            )
        return model_ids

    def resolve_model(self, requested_model: str | None = None) -> str:
        model = (
            requested_model.strip()
            if requested_model is not None
            else self.default_model
        )
        if not model:
            raise ValueError(
                "OpenAI model is required when no default_model is configured"
            )
        return model

    def client_options(self) -> dict[str, str]:
        options = {"base_url": self.base_url}
        if self.api_key is not None:
            options["api_key"] = self.api_key
        return options


@dataclass(frozen=True)
class AnthropicProviderSettings:
    """Connection defaults for Anthropic and Anthropic-compatible model providers."""

    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    api_key: str | None = field(default=None, repr=False)
    default_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_url", _validate_provider_base_url(self.base_url, "Anthropic")
        )
        if self.api_key is not None:
            api_key = self.api_key.strip()
            if not api_key:
                raise ValueError("Anthropic api_key must not be empty when provided")
            object.__setattr__(self, "api_key", api_key)
        if self.default_model is not None:
            default_model = self.default_model.strip()
            if not default_model:
                raise ValueError(
                    "Anthropic default_model must not be empty when provided"
                )
            object.__setattr__(self, "default_model", default_model)

    def validate_connection(
        self,
        *,
        check_default_model: bool = False,
        client_factory: Callable[..., Any] | None = None,
    ) -> tuple[str, ...]:
        if client_factory is None:
            try:
                from anthropic import Anthropic
            except ModuleNotFoundError as exc:
                if exc.name != "anthropic":
                    raise
                raise RuntimeError(
                    "The Anthropic SDK is optional. Install "
                    "'agent-rt[anthropic]' to use Anthropic startup "
                    "validation, or pass client_factory explicitly."
                ) from exc
            client_factory = Anthropic
        client = client_factory(
            base_url=self.base_url,
            api_key=self.api_key or "not-provided",
        )
        try:
            model_ids = _listed_model_ids(_sdk_list_models(client))
        except Exception as exc:
            raise ValueError(f"Anthropic startup validation failed: {exc}") from exc
        if (
            check_default_model
            and self.default_model is not None
            and self.default_model not in model_ids
        ):
            raise ValueError(
                f"Anthropic default model {self.default_model!r} is not available"
            )
        return model_ids

    def resolve_model(self, requested_model: str | None = None) -> str:
        model = (
            requested_model.strip()
            if requested_model is not None
            else self.default_model
        )
        if not model:
            raise ValueError(
                "Anthropic model is required when no default_model is configured"
            )
        return model

    def client_options(self) -> dict[str, str]:
        options = {"base_url": self.base_url}
        if self.api_key is not None:
            options["api_key"] = self.api_key
        return options

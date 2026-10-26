"""Provider defaults, environment names, and SDK compatibility helpers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"


OPENAI_API_KEY_ENV = "OPENAI_" + "API_KEY"  # pragma: allowlist secret


OPENAI_MODEL_ENV = "OPENAI_MODEL"


DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"


ANTHROPIC_API_KEY_ENV = "ANTHROPIC_" + "API_KEY"  # pragma: allowlist secret


ANTHROPIC_MODEL_ENV = "ANTHROPIC_MODEL"


def _env_enabled(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "boolean environment value must be one of 1/0, true/false, yes/no, on/off"
    )


def _validate_provider_base_url(value: str, provider: str) -> str:
    from urllib.parse import urlparse

    normalized = value.strip().rstrip("/")
    if not normalized:
        raise ValueError(f"{provider} base URL must not be empty")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{provider} base URL must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"{provider} base URL must not include query parameters or fragments"
        )
    return normalized


def _listed_model_ids(page: Any) -> tuple[str, ...]:
    data = getattr(page, "data", None)
    if data is None and isinstance(page, Mapping):
        data = page.get("data")
    if data is None:
        try:
            data = tuple(page)
        except TypeError as exc:
            raise ValueError("model listing response must expose model data") from exc
    model_ids: list[str] = []
    for item in data:
        model_id = (
            item.get("id") if isinstance(item, Mapping) else getattr(item, "id", None)
        )
        if isinstance(model_id, str):
            model_ids.append(model_id)
    return tuple(model_ids)


_SDK_PARAMETER_CACHE: dict[Any, frozenset[str] | None] = {}


_SDK_PARAMETER_CACHE_LIMIT = 256


def _sdk_accepted_parameters(method: Any) -> frozenset[str] | None:
    """Return the keyword names an SDK request method accepts.

    ``None`` means the method accepts arbitrary keywords (or cannot be
    inspected), so every parameter can be passed through unchanged.
    """
    try:
        key = inspect.unwrap(method)
        hash(key)
    except (TypeError, ValueError):
        key = None
    if key is not None and key in _SDK_PARAMETER_CACHE:
        return _SDK_PARAMETER_CACHE[key]
    accepted: frozenset[str] | None
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        accepted = None
    else:
        if "extra_body" not in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            accepted = None
        else:
            accepted = frozenset(parameters)
    if key is not None:
        if len(_SDK_PARAMETER_CACHE) >= _SDK_PARAMETER_CACHE_LIMIT:
            _SDK_PARAMETER_CACHE.clear()
        _SDK_PARAMETER_CACHE[key] = accepted
    return accepted


def _sdk_params(method: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Move request fields an older SDK does not know into ``extra_body``.

    Vendor SDKs released before a request field existed reject it as an
    unexpected keyword argument, while the wire API still accepts it in the
    JSON body. Routing it through ``extra_body`` keeps older SDK versions
    working without changing the request that reaches the provider.
    """
    accepted = _sdk_accepted_parameters(method)
    if accepted is None:
        return params
    unknown = [name for name in params if name not in accepted]
    if not unknown:
        return params
    routed = dict(params)
    extra_body = dict(routed.pop("extra_body", None) or {})
    for name in unknown:
        if name != "extra_body":
            extra_body.setdefault(name, routed.pop(name))
    routed["extra_body"] = extra_body
    return routed


def _sdk_list_models(client: Any) -> Any:
    """Call ``client.models.list()``, or the raw endpoint on older SDKs.

    Anthropic SDKs released before the models API (0.39) still expose the
    generic ``client.get`` request helper, which reaches the same endpoint.
    """
    resource = getattr(client, "models", None)
    list_models = getattr(resource, "list", None)
    if callable(list_models):
        return list_models()
    raw_get = getattr(client, "get", None)
    if callable(raw_get):
        return raw_get("/v1/models", cast_to=object)
    raise TypeError("provider client does not support models.list()")


_ws_capability_cache: dict[str, bool] = {}


def _ws_capability(base_url: str) -> bool | None:
    return _ws_capability_cache.get(base_url)


def _set_ws_capability(base_url: str, supported: bool) -> None:
    _ws_capability_cache[base_url] = supported


@dataclass(frozen=True)
class BatchJob:
    id: str
    status: str
    raw: Any = None


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    created_at: datetime | None = None
    display_name: str | None = None
    object: str | None = None
    owned_by: str | None = None
    raw: Any = None

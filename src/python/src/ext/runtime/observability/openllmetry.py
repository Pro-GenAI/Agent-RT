"""OpenLLMetry (Traceloop) configuration from runtime settings."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ext.runtime.settings import runtime_settings


@dataclass(frozen=True)
class OpenLLMetryConfig:
    api_key: str | None = None
    base_url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    trace_content: bool | None = None
    telemetry_enabled: bool | None = None
    enrich_tokens: bool | None = None
    app_name: str = "agent-rt"


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("invalid boolean environment value")


def _parse_traceloop_headers(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    headers: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError("TRACELOOP_HEADERS entries must be key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            raise ValueError(
                "TRACELOOP_HEADERS entries must have non-empty key and value"
            )
        headers[key] = raw_value
    return headers


def _valid_traceloop_base_url(value: str) -> bool:
    candidate = value.strip()
    if not candidate or any(character.isspace() for character in candidate):
        return False
    if candidate.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        parsed = urlparse(candidate)
        return bool(parsed.scheme in {"http", "https"} and parsed.netloc)
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+(?::[0-9]{1,5})?", candidate))


def openllmetry_config_from_env(
    env: Mapping[str, str] | None = None,
) -> OpenLLMetryConfig | None:
    source = runtime_settings() if env is None else env
    api_key_raw = source.get("TRACELOOP_API_KEY")
    base_url_raw = source.get("TRACELOOP_BASE_URL")
    api_key = api_key_raw.strip() if api_key_raw is not None else None
    base_url = base_url_raw.strip() if base_url_raw is not None else None
    if api_key == "":
        return None
    if base_url is not None and not _valid_traceloop_base_url(base_url):
        return None
    try:
        headers = _parse_traceloop_headers(source.get("TRACELOOP_HEADERS"))
        trace_content = _parse_env_bool(source.get("TRACELOOP_TRACE_CONTENT"))
        telemetry_enabled = _parse_env_bool(source.get("TRACELOOP_TELEMETRY"))
        enrich_tokens = _parse_env_bool(source.get("TRACELOOP_ENRICH_TOKENS"))
    except ValueError:
        return None
    if not api_key and not base_url and not headers:
        return None
    app_name = source.get("TRACELOOP_APP_NAME", "agent-rt").strip()
    if not app_name:
        return None
    return OpenLLMetryConfig(
        api_key=api_key or None,
        base_url=base_url or None,
        headers=headers,
        trace_content=trace_content,
        telemetry_enabled=telemetry_enabled,
        enrich_tokens=enrich_tokens,
        app_name=app_name,
    )


def initialize_openllmetry_from_env(
    env: Mapping[str, str] | None = None,
    *,
    initializer: Callable[..., Any] | None = None,
) -> bool:
    config = openllmetry_config_from_env(env)
    if config is None:
        return False
    if initializer is None:
        try:
            from traceloop.sdk import Traceloop
        except Exception:
            return False
        initializer = Traceloop.init
    options: dict[str, Any] = {"app_name": config.app_name}
    if config.api_key is not None:
        options["api_key"] = config.api_key
    if config.base_url is not None:
        options["api_endpoint"] = config.base_url
    if config.headers:
        options["headers"] = dict(config.headers)
    if config.telemetry_enabled is not None:
        options["telemetry_enabled"] = config.telemetry_enabled
    if config.enrich_tokens is not None:
        options["should_enrich_metrics"] = config.enrich_tokens
    # TRACELOOP_TRACE_CONTENT is consumed directly by OpenLLMetry. Keeping it
    # in the environment avoids relying on SDK-version-specific init kwargs.
    try:
        initializer(**options)
    except Exception:
        return False
    return True

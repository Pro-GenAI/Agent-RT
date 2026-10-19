"""YAML runtime configuration (``config.yaml``, in the style of LiteLLM).

The file declares several models (``model_list``), several vector stores
(``vector_store_registry``) and runtime settings (``agent_rt_settings``).
Secrets are referenced as ``os.environ/NAME`` and resolved while the file is
loaded; nothing else in the runtime reads configuration from the process
environment.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENVIRONMENT_REFERENCE_PREFIX = "os.environ/"
MAX_CONFIG_BYTES = 1_000_000
CONFIG_FILE_NAMES = ("agent-rt.yaml", "agent-rt.yml")
SUPPORTED_MODEL_PROVIDERS = ("openai", "anthropic")
SUPPORTED_VECTOR_STORE_PROVIDERS = ("bedrock", "chroma", "milvus", "pinecone", "qdrant", "weaviate")

_PARAM_KEYS = ("agent_rt_params", "litellm_params")
# Unset references here fail only when that model, store or server is used.
_DEFERRED_SECRET_PATH = re.compile(
    r"^config\.(model_list|vector_store_registry)\[\d+\]\."
    r"(agent_rt_params|litellm_params)\.(api_key|api_base)$"
    r"|^config\.agent_rt_settings\.server\."
)
_MODEL_PARAM_FIELDS = {"model", "api_key", "api_base", "websocket"}
_STORE_PARAM_FIELDS = {
    "vector_store_id",
    "custom_llm_provider",
    "api_base",
    "api_key",
    "collection",
    "embedding_model",
    "timeout",
    "options",
    "vector_store_description",
    "vector_store_metadata",
    "allowed_models",
}
_SETTINGS_SECTIONS = {
    "default_model",
    "server",
    "sandbox",
    "web_search",
    "execution_limits",
    "decision_model",
    "observability",
    "cli",
}
_SECTION_ENVIRONMENT = {
    "execution_limits": {
        "timeout_seconds": "AGENT_RT_EXECUTION_TIMEOUT_SECONDS",
        "memory_bytes": "AGENT_RT_EXECUTION_MEMORY_BYTES",
        "cpu_seconds": "AGENT_RT_EXECUTION_CPU_SECONDS",
    },
    "web_search": {
        "tool": "AGENT_RT_WEB_SEARCH_TOOL",
        "api_key": "AGENT_RT_WEB_SEARCH_API_KEY",
        "timeout_seconds": "AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS",
    },
    "decision_model": {
        "base_url": "AGENT_RT_JEV_BASE_URL",
        "model": "AGENT_RT_JEV_MODEL",
        "timeout_seconds": "AGENT_RT_JEV_TIMEOUT_SECONDS",
    },
    "observability": {
        "api_key": "TRACELOOP_API_KEY",
        "base_url": "TRACELOOP_BASE_URL",
        "headers": "TRACELOOP_HEADERS",
        "trace_content": "TRACELOOP_TRACE_CONTENT",
        "telemetry_enabled": "TRACELOOP_TELEMETRY",
        "enrich_tokens": "TRACELOOP_ENRICH_TOKENS",
        "app_name": "TRACELOOP_APP_NAME",
    },
    "sandbox": {
        key: "AGENT_RT_SANDBOX_" + key.upper()
        for key in (
            "backend",
            "workspace_root",
            "native_root",
            "docker_image",
            "docker_binary",
            "docker_user",
            "docker_read_only",
            "e2b_template",
            "microsandbox_image",
            "swerex_url",
            "swerex_api_key",
            "uid",
            "gid",
            "network_mode",
            "network_max_transfer_bytes",
            "network_max_bytes_per_second",
            "allowed_domains",
            "blocked_domains",
            "allow_http",
            "allow_ip",
            "allow_proxy",
            "allow_websocket",
            "setpriv_binary",
            "prlimit_binary",
        )
    },
}
_SERVER_FIELDS = {
    "api_keys",
    "default_agent",
    "allowed_websocket_origins",
    "host",
    "port",
    "max_concurrent_requests",
    "max_queued_requests",
}


class ConfigError(ValueError):
    """Raised for a missing, unreadable or invalid configuration file."""


class ConfigNotFoundError(ConfigError):
    """Raised when no configuration file exists at the requested or default paths."""


@dataclass(frozen=True)
class MissingSecret:
    """An ``os.environ/NAME`` reference whose variable is unset; raises when used."""

    name: str
    path: str

    def error(self) -> ConfigError:
        return ConfigError(
            f"{self.path}: environment variable {self.name} referenced by os.environ/ is not set"
        )


def require_resolved(*values: Any) -> None:
    """Raise for the first unset ``os.environ/`` reference in ``values``."""
    for value in values:
        if isinstance(value, MissingSecret):
            raise value.error()
        if isinstance(value, Mapping):
            require_resolved(*value.values())
        elif isinstance(value, (list, tuple)):
            require_resolved(*value)


def config_search_paths(
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> tuple[Path, ...]:
    """Default locations, in priority order: current folder, then home folder."""
    base = Path.cwd() if cwd is None else Path(cwd)
    user = Path.home() if home is None else Path(home)
    paths: list[Path] = [base / name for name in CONFIG_FILE_NAMES]
    paths += [base / ".agent-rt" / "config.yaml", base / ".agent-rt" / "config.yml"]
    paths += [user / ".agent-rt" / "config.yaml", user / ".agent-rt" / "config.yml"]
    paths += [
        user / ".config" / "agent-rt" / "config.yaml",
        user / ".config" / "agent-rt" / "config.yml",
    ]
    return tuple(paths)


def discover_config_path(
    explicit: str | os.PathLike[str] | None = None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Return the config file to use: an explicit path or the first default found."""
    if explicit is not None and str(explicit).strip():
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigNotFoundError(f"config file not found: {path}")
        return path
    candidates = config_search_paths(cwd=cwd, home=home)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise ConfigNotFoundError(
        "no config file found (pass --config PATH or create one of: " + searched + ")"
    )


def _resolve_secrets(
    value: Any,
    path: str,
    environ: Mapping[str, str],
) -> Any:
    if isinstance(value, str):
        if value.startswith(ENVIRONMENT_REFERENCE_PREFIX):
            name = value[len(ENVIRONMENT_REFERENCE_PREFIX) :].strip()
            if not name:
                raise ConfigError(f"{path}: empty os.environ/ reference")
            resolved = environ.get(name)
            if resolved is None or not resolved.strip():
                missing = MissingSecret(name, path)
                if _DEFERRED_SECRET_PATH.match(path):
                    return missing
                raise missing.error()
            return resolved
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_secrets(item, f"{path}.{key}", environ)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_secrets(item, f"{path}[{index}]", environ)
            for index, item in enumerate(value)
        ]
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _check_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(unknown)}")


def _text(value: Any, path: str, *, required: bool = False) -> Any:
    if isinstance(value, MissingSecret):
        return value
    if value is None:
        if required:
            raise ConfigError(f"{path} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ModelDeployment:
    """One concrete provider/model behind a public ``model_name``."""

    model_name: str
    provider: str
    model: str
    api_key: str | MissingSecret | None = field(default=None, repr=False)
    api_base: str | MissingSecret | None = field(default=None, repr=False)
    websocket: bool | None = None
    model_info: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class VectorStoreConfig:
    """One entry of ``vector_store_registry``."""

    name: str
    store_id: str
    provider: str
    api_base: str | MissingSecret | None = field(default=None, repr=False)
    api_key: str | MissingSecret | None = field(default=None, repr=False)
    collection: str | None = None
    embedding_model: str | None = None
    timeout: float = 20.0
    options: Mapping[str, str] = field(default_factory=dict, repr=False)
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    allowed_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRTConfig:
    models: tuple[ModelDeployment, ...] = ()
    vector_stores: tuple[VectorStoreConfig, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source: Path | None = None

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.model_name for item in self.models))

    @property
    def default_model(self) -> str | None:
        configured = self.settings.get("default_model")
        if configured is not None:
            return str(configured)
        names = self.model_names
        return names[0] if names else None

    def deployments(self, model_name: str) -> tuple[ModelDeployment, ...]:
        return tuple(item for item in self.models if item.model_name == model_name)

    def vector_store(self, reference: str) -> VectorStoreConfig | None:
        for store in self.vector_stores:
            if reference in (store.name, store.store_id):
                return store
        return None

    def runtime_environment(self) -> dict[str, str]:
        """Variables for the sandbox/web-search/limits/observability readers.

        The result is built purely from this file; the process environment is
        never consulted.
        """
        result: dict[str, str] = {}
        for section, names in _SECTION_ENVIRONMENT.items():
            values = self.settings.get(section)
            if not isinstance(values, Mapping):
                continue
            for key, variable in names.items():
                if key not in values or values[key] is None:
                    continue
                item = values[key]
                if isinstance(item, bool):
                    result[variable] = "true" if item else "false"
                elif isinstance(item, Mapping):
                    result[variable] = ",".join(f"{k}={v}" for k, v in item.items())
                elif isinstance(item, (list, tuple)):
                    result[variable] = ",".join(str(part) for part in item)
                else:
                    result[variable] = str(item)
        return result

    def server_options(self) -> dict[str, Any]:
        server = self.settings.get("server")
        if not isinstance(server, Mapping):
            return {}
        require_resolved(server)
        return dict(server)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        source: Path | None = None,
    ) -> AgentRTConfig:
        env = os.environ if environ is None else environ
        document = _resolve_secrets(_mapping(raw, "config"), "config", env)
        _check_keys(
            document,
            {"model_list", "vector_store_registry", "agent_rt_settings"},
            "config",
        )
        models = _parse_models(document.get("model_list") or [])
        stores = _parse_vector_stores(
            document.get("vector_store_registry") or [],
            {item.model_name for item in models},
        )
        settings = _parse_settings(document.get("agent_rt_settings") or {})
        default_model = settings.get("default_model")
        if default_model is not None and default_model not in {
            item.model_name for item in models
        }:
            raise ConfigError(
                f"agent_rt_settings.default_model {default_model!r} is not in model_list"
            )
        for store in stores:
            for allowed in store.allowed_models:
                if allowed not in {item.model_name for item in models}:
                    raise ConfigError(
                        f"vector store {store.name!r} allows unknown model {allowed!r}"
                    )
        return cls(
            models=models,
            vector_stores=stores,
            settings=settings,
            source=source,
        )


def _params(entry: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    present = [key for key in _PARAM_KEYS if key in entry]
    if len(present) != 1:
        raise ConfigError(f"{path} requires exactly one of: {', '.join(_PARAM_KEYS)}")
    return _mapping(entry[present[0]], f"{path}.{present[0]}")


def _number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{path} must be a positive finite number")
    return float(value)


def _parse_models(raw: Any) -> tuple[ModelDeployment, ...]:
    if not isinstance(raw, list):
        raise ConfigError("model_list must be a list")
    deployments: list[ModelDeployment] = []
    for index, item in enumerate(raw):
        path = f"model_list[{index}]"
        entry = _mapping(item, path)
        _check_keys(entry, {"model_name", "model_info", *_PARAM_KEYS}, path)
        name = _text(entry.get("model_name"), f"{path}.model_name", required=True)
        params = _params(entry, path)
        _check_keys(params, _MODEL_PARAM_FIELDS, f"{path}.params")
        spec = _text(params.get("model"), f"{path}.params.model", required=True)
        provider, separator, model = (spec or "").partition("/")
        if not separator or not model.strip():
            raise ConfigError(f"{path}.params.model must look like '<provider>/<model>'")
        provider = provider.strip().lower()
        if provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ConfigError(
                f"{path}.params.model: unsupported provider {provider!r}; "
                f"supported: {', '.join(SUPPORTED_MODEL_PROVIDERS)}"
            )
        websocket = params.get("websocket")
        if websocket is not None and not isinstance(websocket, bool):
            raise ConfigError(f"{path}.params.websocket must be a boolean")
        info = entry.get("model_info") or {}
        deployments.append(
            ModelDeployment(
                model_name=name or "",
                provider=provider,
                model=model.strip(),
                api_key=_text(params.get("api_key"), f"{path}.params.api_key"),
                api_base=_text(params.get("api_base"), f"{path}.params.api_base"),
                websocket=websocket,
                model_info=dict(_mapping(info, f"{path}.model_info")),
            )
        )
    return tuple(deployments)


def _parse_vector_stores(
    raw: Any,
    model_names: set[str],
) -> tuple[VectorStoreConfig, ...]:
    if not isinstance(raw, list):
        raise ConfigError("vector_store_registry must be a list")
    stores: list[VectorStoreConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"vector_store_registry[{index}]"
        entry = _mapping(item, path)
        _check_keys(entry, {"vector_store_name", *_PARAM_KEYS}, path)
        name = _text(entry.get("vector_store_name"), f"{path}.vector_store_name", required=True)
        params = _params(entry, path)
        _check_keys(params, _STORE_PARAM_FIELDS, f"{path}.params")
        provider = (
            _text(params.get("custom_llm_provider"), f"{path}.params.custom_llm_provider", required=True)
            or ""
        ).lower()
        if provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
            raise ConfigError(
                f"{path}.params.custom_llm_provider: unsupported {provider!r}; "
                f"supported: {', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)} "
                "(other hosted stores are not supported)"
            )
        store_id = _text(params.get("vector_store_id"), f"{path}.params.vector_store_id", required=True)
        embedding = _text(params.get("embedding_model"), f"{path}.params.embedding_model")
        if embedding is not None and embedding not in model_names:
            raise ConfigError(
                f"{path}.params.embedding_model {embedding!r} is not in model_list"
            )
        options_raw = params.get("options") or {}
        options = {
            str(key): str(value)
            for key, value in _mapping(options_raw, f"{path}.params.options").items()
        }
        allowed = params.get("allowed_models") or []
        if not isinstance(allowed, list) or not all(isinstance(m, str) for m in allowed):
            raise ConfigError(f"{path}.params.allowed_models must be a list of model names")
        for reference in dict.fromkeys((name, store_id)):
            if reference in seen:
                raise ConfigError(f"{path}: duplicate vector store name or id {reference!r}")
            seen.add(reference or "")
        timeout = _number(params.get("timeout"), f"{path}.params.timeout")
        stores.append(
            VectorStoreConfig(
                name=name or "",
                store_id=store_id or "",
                provider=provider,
                api_base=_text(params.get("api_base"), f"{path}.params.api_base"),
                api_key=_text(params.get("api_key"), f"{path}.params.api_key"),
                collection=_text(params.get("collection"), f"{path}.params.collection"),
                embedding_model=embedding,
                timeout=20.0 if timeout is None else timeout,
                options=options,
                description=_text(
                    params.get("vector_store_description"),
                    f"{path}.params.vector_store_description",
                ),
                metadata=dict(
                    _mapping(
                        params.get("vector_store_metadata") or {},
                        f"{path}.params.vector_store_metadata",
                    )
                ),
                allowed_models=tuple(allowed),
            )
        )
    return tuple(stores)


def _parse_settings(raw: Any) -> dict[str, Any]:
    settings = dict(_mapping(raw, "agent_rt_settings"))
    _check_keys(settings, _SETTINGS_SECTIONS, "agent_rt_settings")
    for section, names in _SECTION_ENVIRONMENT.items():
        values = settings.get(section)
        if values is None:
            continue
        _check_keys(
            _mapping(values, f"agent_rt_settings.{section}"),
            set(names),
            f"agent_rt_settings.{section}",
        )
    cli = settings.get("cli")
    if cli is not None:
        _check_keys(_mapping(cli, "agent_rt_settings.cli"), {"home", "compact_token_limit"}, "agent_rt_settings.cli")
    server = settings.get("server")
    if server is not None:
        _check_keys(_mapping(server, "agent_rt_settings.server"), _SERVER_FIELDS, "agent_rt_settings.server")
    default_model = settings.get("default_model")
    if default_model is not None:
        settings["default_model"] = _text(default_model, "agent_rt_settings.default_model")
    return settings


def parse_config(
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
    source: Path | None = None,
) -> AgentRTConfig:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - PyYAML is a core dependency
        raise ConfigError("PyYAML is required to read config files: pip install pyyaml") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    return AgentRTConfig.from_mapping(raw, environ=environ, source=source)


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> AgentRTConfig:
    """Load ``path`` or the first default config file (current folder, then home)."""
    resolved = discover_config_path(path, cwd=cwd, home=home)
    try:
        if resolved.stat().st_size > MAX_CONFIG_BYTES:
            raise ConfigError(f"config file exceeds {MAX_CONFIG_BYTES} bytes: {resolved}")
        text = resolved.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"unable to read config file {resolved}: {exc}") from exc
    try:
        return parse_config(text, environ=environ, source=resolved)
    except ConfigError as exc:
        raise ConfigError(f"{resolved}: {exc}") from exc

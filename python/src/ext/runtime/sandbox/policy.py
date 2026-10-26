"""Sandbox resource limits and network policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import agent_rt as _core


def sandbox_resource_limits_from_env(
    environment: Mapping[str, str] | None = None,
) -> SandboxResourceLimits:
    defaults = _core.tool_execution_limits_from_env(environment)
    return SandboxResourceLimits(
        cpu_seconds=defaults.cpu_seconds,
        memory_bytes=defaults.memory_bytes,
        timeout_seconds=defaults.timeout_seconds,
    )


@dataclass(frozen=True)
class SandboxResourceLimits:
    cpu_seconds: float | None = None
    memory_bytes: int | None = None
    disk_bytes: int | None = None
    process_count: int | None = None
    timeout_seconds: float | None = None
    output_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("cpu_seconds", self.cpu_seconds),
            ("memory_bytes", self.memory_bytes),
            ("disk_bytes", self.disk_bytes),
            ("process_count", self.process_count),
            ("timeout_seconds", self.timeout_seconds),
            ("output_bytes", self.output_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        # Container runtimes treat a 0 memory/pids limit as "unlimited", while
        # rlimits treat it as "nothing may run"; reject the ambiguity outright.
        for name, value in (
            ("memory_bytes", self.memory_bytes),
            ("process_count", self.process_count),
        ):
            if value is not None and value == 0:
                raise ValueError(f"{name} must be positive when provided")


SandboxNetworkMode = Literal["none", "allowlist", "unrestricted"]


SANDBOX_NETWORK_MODE_ENV = "AGENT_RT_SANDBOX_NETWORK_MODE"


SANDBOX_ALLOWED_DOMAINS_ENV = "AGENT_RT_SANDBOX_ALLOWED_DOMAINS"


SANDBOX_BLOCKED_DOMAINS_ENV = "AGENT_RT_SANDBOX_BLOCKED_DOMAINS"


SANDBOX_ALLOW_HTTP_ENV = "AGENT_RT_SANDBOX_ALLOW_HTTP"


SANDBOX_ALLOW_WEBSOCKET_ENV = "AGENT_RT_SANDBOX_ALLOW_WEBSOCKET"


SANDBOX_ALLOW_IP_ENV = "AGENT_RT_SANDBOX_ALLOW_IP"


SANDBOX_ALLOW_PROXY_ENV = "AGENT_RT_SANDBOX_ALLOW_PROXY"


SANDBOX_NETWORK_BPS_ENV = "AGENT_RT_SANDBOX_NETWORK_MAX_BYTES_PER_SECOND"


SANDBOX_NETWORK_BYTES_ENV = "AGENT_RT_SANDBOX_NETWORK_MAX_TRANSFER_BYTES"


_SANDBOX_PROXY_ENV_KEYS = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


def _sandbox_env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _sandbox_env_nonnegative_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sandbox_domain_list(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class SandboxNetworkPolicy:
    mode: SandboxNetworkMode = "none"
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    proxy_url: str | None = None
    allow_http: bool = False
    allow_websocket: bool = False
    allow_ip_addresses: bool = False
    allow_proxy: bool = False
    max_bytes_per_second: int | None = None
    max_transfer_bytes: int | None = None

    def __post_init__(self) -> None:
        import ipaddress

        allowed = tuple(domain.lower().strip(".") for domain in self.allowed_domains)
        blocked = tuple(domain.lower().strip(".") for domain in self.blocked_domains)
        if any(not domain for domain in allowed + blocked):
            raise ValueError("sandbox network domains must not be empty")
        for domain in allowed + blocked:
            try:
                ipaddress.ip_address(domain)
            except ValueError:
                continue
            if not self.allow_ip_addresses:
                raise ValueError(
                    "sandbox network domain rules must use domain names, not IP addresses"
                )
        object.__setattr__(self, "allowed_domains", allowed)
        object.__setattr__(self, "blocked_domains", blocked)
        if self.proxy_url is not None:
            if not self.proxy_url.strip():
                raise ValueError("sandbox proxy_url must not be blank")
            if not self.allow_proxy:
                raise ValueError("sandbox proxy use is disabled")
        for name, value in (
            ("max_bytes_per_second", self.max_bytes_per_second),
            ("max_transfer_bytes", self.max_transfer_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")

    @staticmethod
    def _matches(domain: str, patterns: Sequence[str]) -> bool:
        normalized = domain.lower().strip(".")
        return any(
            normalized == pattern or normalized.endswith("." + pattern)
            for pattern in patterns
        )

    def allows(self, target: str) -> bool:
        import ipaddress
        from urllib.parse import urlparse

        # Reject characters on which URL parsers disagree (WHATWG treats "\\" as
        # "/"), so the host we check is the host a client would actually reach.
        if "\\" in target or any(ord(ch) < 33 or ord(ch) == 127 for ch in target):
            return False
        parsed = urlparse(target if "://" in target else "//" + target)
        scheme = parsed.scheme.lower()
        if scheme in {"ws", "wss"} and not self.allow_websocket:
            return False
        if scheme == "http" and not self.allow_http:
            return False
        if scheme and scheme not in {"https", "http", "ws", "wss"}:
            return False
        domain = (parsed.hostname or target).lower().strip(".")
        if domain == "localhost" and not self.allow_ip_addresses:
            return False
        try:
            ipaddress.ip_address(domain)
        except ValueError:
            pass
        else:
            if not self.allow_ip_addresses:
                return False
        if self._matches(domain, self.blocked_domains):
            return False
        if self.mode == "none":
            return False
        if self.mode == "unrestricted":
            return True
        return self._matches(domain, self.allowed_domains)


def sandbox_network_policy_from_env(
    environment: Mapping[str, str] | None = None,
) -> SandboxNetworkPolicy:
    env = _core.runtime_settings() if environment is None else environment
    mode = env.get(SANDBOX_NETWORK_MODE_ENV, "none").strip().lower()
    if mode not in {"none", "allowlist", "unrestricted"}:
        raise ValueError(
            f"{SANDBOX_NETWORK_MODE_ENV} must be none, allowlist, or unrestricted"
        )
    return SandboxNetworkPolicy(
        mode=cast(SandboxNetworkMode, mode),
        allowed_domains=_sandbox_domain_list(env.get(SANDBOX_ALLOWED_DOMAINS_ENV)),
        blocked_domains=_sandbox_domain_list(env.get(SANDBOX_BLOCKED_DOMAINS_ENV)),
        allow_http=_sandbox_env_bool(env, SANDBOX_ALLOW_HTTP_ENV),
        allow_websocket=_sandbox_env_bool(env, SANDBOX_ALLOW_WEBSOCKET_ENV),
        allow_ip_addresses=_sandbox_env_bool(env, SANDBOX_ALLOW_IP_ENV),
        allow_proxy=_sandbox_env_bool(env, SANDBOX_ALLOW_PROXY_ENV),
        max_bytes_per_second=_sandbox_env_nonnegative_int(env, SANDBOX_NETWORK_BPS_ENV),
        max_transfer_bytes=_sandbox_env_nonnegative_int(env, SANDBOX_NETWORK_BYTES_ENV),
    )


def _sanitize_sandbox_environment(
    environment: Mapping[str, str],
    policy: SandboxNetworkPolicy,
) -> dict[str, str]:
    sanitized = dict(environment)
    if _SANDBOX_PROXY_ENV_KEYS.intersection(sanitized) and not policy.allow_proxy:
        raise ValueError("sandbox proxy environment variables are disabled")
    return sanitized


def _requires_advanced_network_enforcement(policy: SandboxNetworkPolicy) -> bool:
    return bool(
        policy.mode == "allowlist"
        or policy.blocked_domains
        or not policy.allow_http
        or not policy.allow_websocket
        or not policy.allow_ip_addresses
        or policy.max_bytes_per_second is not None
        or policy.max_transfer_bytes is not None
    )

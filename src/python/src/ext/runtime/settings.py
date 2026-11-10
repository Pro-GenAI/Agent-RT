"""Runtime settings fed by ``configure_runtime`` (never the process environment)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_RUNTIME_SETTINGS: Mapping[str, str] = {}


def runtime_settings() -> Mapping[str, str]:
    """Settings fed by ``configure_runtime`` (never the process environment)."""
    return _RUNTIME_SETTINGS


def configure_runtime(config: Any) -> bool:
    """Apply ``agent_rt_settings`` from an ``AgentRTConfig`` to implicit defaults.

    Sets the sandbox, tool-execution-limit, web-search, Decision-model and
    observability defaults, and initializes OpenLLMetry when configured.
    Returns whether OpenLLMetry was initialized.
    """
    from ext.runtime.observability.openllmetry import initialize_openllmetry_from_env

    global _RUNTIME_SETTINGS
    _RUNTIME_SETTINGS = dict(config.runtime_environment())
    return initialize_openllmetry_from_env(_RUNTIME_SETTINGS)

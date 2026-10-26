"""Deny-overrides permission rules with path-level authorization."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ext.runtime.tools.definitions import ToolDefinition, ToolSideEffect

PermissionEffect = Literal["allow", "deny"]


def _normalize_permission_path(path: str) -> str:
    raw = path.strip().replace("\\", "/")
    parts: list[str] = []
    for part in raw.lstrip("/").split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if not parts:
                raise ValueError("path escapes permission root")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


@dataclass(frozen=True)
class PermissionRequest:
    operation: str
    agent: str | None = None
    tool: str | None = None
    path: str | None = None
    network: str | None = None
    data: str | None = None
    side_effect: ToolSideEffect | None = None


@dataclass(frozen=True)
class PermissionRule:
    effect: PermissionEffect
    operations: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    networks: tuple[str, ...] = ()
    data: tuple[str, ...] = ()
    side_effects: tuple[ToolSideEffect, ...] = ()

    @staticmethod
    def _matches(value: str | None, patterns: Sequence[str]) -> bool:
        if not patterns:
            return True
        if value is None:
            return False
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    def matches(self, request: PermissionRequest) -> bool:
        path = (
            _normalize_permission_path(request.path)
            if request.path is not None
            else None
        )
        path_patterns = tuple(
            (
                pattern.strip().replace("\\", "/").lstrip("/")
                if any(character in pattern for character in "*?[")
                else _normalize_permission_path(pattern)
            )
            for pattern in self.paths
        )
        return (
            self._matches(request.operation, self.operations)
            and self._matches(request.agent, self.agents)
            and self._matches(request.tool, self.tools)
            and self._matches(path, path_patterns)
            and self._matches(request.network, self.networks)
            and self._matches(request.data, self.data)
            and (not self.side_effects or request.side_effect in self.side_effects)
        )


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str
    matched_rule: PermissionRule | None = None


class PermissionDeniedError(PermissionError):
    def __init__(
        self, request: PermissionRequest, decision: PermissionDecision
    ) -> None:
        self.request = request
        self.decision = decision
        super().__init__(decision.reason)


class PermissionEngine:
    def __init__(
        self,
        rules: Sequence[PermissionRule] = (),
        *,
        default_effect: PermissionEffect = "deny",
    ) -> None:
        if default_effect not in ("allow", "deny"):
            raise ValueError("default permission effect must be allow or deny")
        self.rules = tuple(rules)
        self.default_effect = default_effect

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        matching = tuple(rule for rule in self.rules if rule.matches(request))
        for rule in matching:
            if rule.effect == "deny":
                return PermissionDecision(
                    False, "permission denied by matching rule", rule
                )
        for rule in matching:
            if rule.effect == "allow":
                return PermissionDecision(
                    True, "permission allowed by matching rule", rule
                )
        allowed = self.default_effect == "allow"
        return PermissionDecision(
            allowed,
            f"permission {'allowed' if allowed else 'denied'} by default policy",
        )

    def check(self, request: PermissionRequest) -> PermissionDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PermissionDeniedError(request, decision)
        return decision

    def check_tool(
        self, tool: str, definition: ToolDefinition, *, agent: str | None = None
    ) -> PermissionDecision:
        return self.check(
            PermissionRequest(
                operation="execute",
                agent=agent,
                tool=tool,
                side_effect=definition.side_effect,
            )
        )

    def check_path(
        self,
        path: str,
        operation: Literal["read", "write", "execute"],
        *,
        agent: str | None = None,
    ) -> PermissionDecision:
        return self.check(
            PermissionRequest(
                operation=operation, agent=agent, path=_normalize_permission_path(path)
            )
        )

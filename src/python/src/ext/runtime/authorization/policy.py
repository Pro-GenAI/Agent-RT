"""Attribute-based policy rules and the policy engine."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

PolicyEffect = Literal["allow", "deny"]


@dataclass(frozen=True)
class PolicyRequest:
    domain: str
    action: str
    subject: str | None = None
    resource: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRule:
    effect: PolicyEffect
    domains: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    @staticmethod
    def _matches(value: str | None, patterns: Sequence[str]) -> bool:
        if not patterns:
            return True
        if value is None:
            return False
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    def matches(self, request: PolicyRequest) -> bool:
        return (
            self._matches(request.domain, self.domains)
            and self._matches(request.action, self.actions)
            and self._matches(request.subject, self.subjects)
            and self._matches(request.resource, self.resources)
            and all(request.attributes.get(k) == v for k, v in self.attributes.items())
        )


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    matched_rule: PolicyRule | None = None


class PolicyEngine:
    def __init__(
        self, rules: Sequence[PolicyRule] = (), *, default_effect: PolicyEffect = "deny"
    ) -> None:
        if default_effect not in ("allow", "deny"):
            raise ValueError("default policy effect must be allow or deny")
        self.rules = tuple(rules)
        self.default_effect = default_effect

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        matching = tuple(rule for rule in self.rules if rule.matches(request))
        for rule in matching:
            if rule.effect == "deny":
                return PolicyDecision(False, "policy denied by matching rule", rule)
        for rule in matching:
            if rule.effect == "allow":
                return PolicyDecision(True, "policy allowed by matching rule", rule)
        allowed = self.default_effect == "allow"
        return PolicyDecision(
            allowed, f"policy {'allowed' if allowed else 'denied'} by default"
        )

    def check(self, request: PolicyRequest) -> PolicyDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return decision

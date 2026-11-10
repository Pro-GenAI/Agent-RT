"""Principals, credentials, authentication context, and authorization rules."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

# Lazily loaded standalone identity, secret, and policy surfaces.
IdentityKind = Literal["user", "service", "delegated"]


AuthenticationMethod = Literal["session", "oauth", "api_key", "service", "delegation"]


@dataclass(frozen=True)
class Principal:
    id: str
    kind: IdentityKind
    tenant_id: str | None = None
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)
    on_behalf_of: str | None = None


@dataclass(frozen=True)
class CredentialReference:
    id: str
    kind: AuthenticationMethod
    expires_at: float | None = None
    scopes: frozenset[str] = frozenset()

    def is_expired(self, *, now: float | None = None) -> bool:
        return (
            self.expires_at is not None
            and (time.time() if now is None else now) >= self.expires_at
        )


@dataclass(frozen=True)
class AuthenticationContext:
    principal: Principal
    method: AuthenticationMethod
    credential: CredentialReference | None = None

    def is_authenticated(self, *, now: float | None = None) -> bool:
        return self.credential is None or not self.credential.is_expired(now=now)


@dataclass(frozen=True)
class AuthorizationRequirement:
    scopes: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)
    resource_owner_id: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


class AuthorizationEngine:
    def evaluate(
        self,
        authentication: AuthenticationContext,
        requirement: AuthorizationRequirement,
    ) -> AuthorizationDecision:
        if not authentication.is_authenticated():
            return AuthorizationDecision(False, "authentication credential is expired")
        principal = authentication.principal
        effective_scopes = set(principal.scopes)
        if authentication.credential is not None:
            effective_scopes.update(authentication.credential.scopes)
        if (
            requirement.tenant_id is not None
            and principal.tenant_id != requirement.tenant_id
        ):
            return AuthorizationDecision(False, "tenant does not match")
        if not requirement.scopes.issubset(effective_scopes):
            return AuthorizationDecision(False, "required scopes are missing")
        if not requirement.roles.issubset(principal.roles):
            return AuthorizationDecision(False, "required roles are missing")
        for key, value in requirement.attributes.items():
            if principal.attributes.get(key) != value:
                return AuthorizationDecision(
                    False, f"required attribute does not match: {key}"
                )
        if (
            requirement.resource_owner_id is not None
            and requirement.resource_owner_id
            not in (principal.id, principal.on_behalf_of)
        ):
            return AuthorizationDecision(False, "resource ownership does not match")
        return AuthorizationDecision(True, "authorization requirements satisfied")

    def check(
        self,
        authentication: AuthenticationContext,
        requirement: AuthorizationRequirement,
    ) -> AuthorizationDecision:
        decision = self.evaluate(authentication, requirement)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return decision

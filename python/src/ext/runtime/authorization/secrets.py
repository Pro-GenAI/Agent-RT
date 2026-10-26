"""Scoped secret storage."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ext.runtime.authorization.grants import CapabilityGrant
from ext.runtime.types import JSONValue


@dataclass(frozen=True)
class SecretMetadata:
    id: str
    tenant_id: str | None = None
    expires_at: float | None = None
    scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SecretValue:
    metadata: SecretMetadata
    _value: str = field(repr=False)

    def reveal(self) -> str:
        if (
            self.metadata.expires_at is not None
            and time.time() >= self.metadata.expires_at
        ):
            raise PermissionError("secret has expired")
        return self._value

    def model_reference(self) -> Mapping[str, JSONValue]:
        return {
            "secret_id": self.metadata.id,
            "tenant_id": self.metadata.tenant_id,
            "expires_at": self.metadata.expires_at,
            "scopes": list[JSONValue](sorted(self.metadata.scopes)),
        }


@runtime_checkable
class SecretStore(Protocol):
    def put(self, secret: SecretValue) -> SecretMetadata: ...
    def get(
        self, secret_id: str, *, tenant_id: str | None = None
    ) -> SecretValue | None: ...
    def delete(self, secret_id: str, *, tenant_id: str | None = None) -> bool: ...


class InMemorySecretStore:
    def __init__(self) -> None:
        self._secrets: dict[tuple[str | None, str], SecretValue] = {}

    def put(self, secret: SecretValue) -> SecretMetadata:
        key = (secret.metadata.tenant_id, secret.metadata.id)
        self._secrets[key] = secret
        return secret.metadata

    def get(
        self, secret_id: str, *, tenant_id: str | None = None
    ) -> SecretValue | None:
        secret = self._secrets.get((tenant_id, secret_id))
        if secret is None:
            return None
        if (
            secret.metadata.expires_at is not None
            and time.time() >= secret.metadata.expires_at
        ):
            return None
        return secret

    def delete(self, secret_id: str, *, tenant_id: str | None = None) -> bool:
        return self._secrets.pop((tenant_id, secret_id), None) is not None


class ScopedSecretStore:
    def __init__(
        self,
        store: SecretStore,
        *,
        tenant_id: str | None = None,
        capability_grant: CapabilityGrant | None = None,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.capability_grant = capability_grant

    def get(self, secret_id: str) -> SecretValue | None:
        if (
            self.capability_grant is not None
            and not self.capability_grant.allows_credential(secret_id)
        ):
            raise PermissionError("credential is outside the capability grant")
        return self.store.get(secret_id, tenant_id=self.tenant_id)

"""PII morphing and privacy redaction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ext.runtime.observability.openllmetry import _parse_env_bool
from ext.runtime.settings import runtime_settings

PII_MORPHER_DISABLE_ENV = "AGENT_RT_DISABLE_PII_MORPHER"


_PII_SHORT_HASH_CHARS_PER_DIGEST = 10


@dataclass(frozen=True)
class PIIEntity:
    kind: str
    start: int
    end: int


_PII_PATTERN_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(
            r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_\x60{|}~-]+"
            r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
            r"(?![A-Za-z0-9._%+-])"
        ),
    ),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("credit_card", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})"
            r"[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
        ),
    ),
)


_PII_HINT_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("email", ("email", "emailaddress")),
    ("phone", ("phone", "phonenumber", "mobile", "mobilenumber")),
    ("ssn", ("ssn", "socialsecuritynumber")),
    ("credit_card", ("creditcard", "cardnumber")),
    ("ipv4", ("ipaddress", "ipv4")),
    ("ipv6", ("ipv6",)),
    ("first_name", ("firstname",)),
    ("last_name", ("lastname",)),
    ("person", ("fullname", "personname")),
    ("address", ("streetaddress", "postaladdress", "mailingaddress")),
    ("date_of_birth", ("dateofbirth", "birthdate", "dob")),
)


_PII_FIRST_NAMES = (
    "Avery",
    "Blake",
    "Casey",
    "Dakota",
    "Emerson",
    "Finley",
    "Harper",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Parker",
    "Quinn",
    "Riley",
    "Rowan",
    "Taylor",
)


_PII_LAST_NAMES = (
    "Bennett",
    "Brooks",
    "Carter",
    "Ellis",
    "Hayes",
    "Jordan",
    "Lane",
    "Morgan",
    "Parker",
    "Reed",
    "Rivera",
    "Sawyer",
    "Taylor",
    "Walker",
    "Ward",
    "Young",
)


def _normalize_pii_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _infer_pii_kind(key: Any) -> str | None:
    normalized = _normalize_pii_key(key)
    for kind, suffixes in _PII_HINT_SUFFIXES:
        if any(normalized.endswith(suffix) for suffix in suffixes):
            return kind
    return None


def _looks_like_credit_card(value: str) -> bool:
    digits = "".join(char for char in value if char.isdigit())
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class PIIMorpher:
    """Deterministically replace PII with synthetic values without storing plaintext."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = runtime_settings() if environ is None else environ
        try:
            disabled = _parse_env_bool(env.get(PII_MORPHER_DISABLE_ENV))
        except ValueError as exc:
            raise ValueError(
                f"{PII_MORPHER_DISABLE_ENV} must be a boolean value"
            ) from exc
        self.enabled = enabled and disabled is not True
        self._mapping: dict[str, str] = {}

    @staticmethod
    def short_hash(value: str) -> str:
        encoded = value.encode("utf-8")
        sha = hashlib.sha256(encoded).hexdigest()
        md5 = hashlib.md5(encoded, usedforsecurity=False).hexdigest()
        width = _PII_SHORT_HASH_CHARS_PER_DIGEST
        return sha[:width] + md5[:width]

    @property
    def mapping(self) -> Mapping[str, str]:
        return dict(self._mapping)

    @staticmethod
    def _seed_parts(short_hash: str) -> tuple[int, int, int]:
        return (
            int(short_hash[0:8], 16),
            int(short_hash[8:16], 16),
            int(short_hash[12:20], 16),
        )

    def _fake_value(self, kind: str, short_hash: str) -> str:
        first, second, third = self._seed_parts(short_hash)
        if kind == "email":
            first_name = _PII_FIRST_NAMES[first % len(_PII_FIRST_NAMES)].lower()
            last_name = _PII_LAST_NAMES[second % len(_PII_LAST_NAMES)].lower()
            return f"{first_name}.{last_name}.{third % 10_000:04d}@example.test"
        if kind == "phone":
            return f"+1-000-{first % 1_000:03d}-{second % 10_000:04d}"
        if kind == "ssn":
            return f"000-{first % 100:02d}-{second % 10_000:04d}"
        if kind == "credit_card":
            return f"0000 0000 {first % 10_000:04d} {second % 10_000:04d}"
        if kind == "ipv4":
            networks = ("192.0.2", "198.51.100", "203.0.113")
            return f"{networks[first % len(networks)]}.{1 + second % 254}"
        if kind == "ipv6":
            groups = [short_hash[index : index + 4] for index in range(0, 16, 4)]
            return "2001:db8:" + ":".join(groups) + "::1"
        if kind == "first_name":
            return _PII_FIRST_NAMES[first % len(_PII_FIRST_NAMES)]
        if kind == "last_name":
            return _PII_LAST_NAMES[second % len(_PII_LAST_NAMES)]
        if kind == "person":
            first_name = _PII_FIRST_NAMES[first % len(_PII_FIRST_NAMES)]
            last_name = _PII_LAST_NAMES[second % len(_PII_LAST_NAMES)]
            return f"{first_name} {last_name}"
        if kind == "address":
            return f"{100 + first % 9_900} Example Avenue, Testville, ZZ 00000"
        if kind == "date_of_birth":
            year = 1970 + first % 31
            month = 1 + second % 12
            day = 1 + third % 28
            return f"{year:04d}-{month:02d}-{day:02d}"
        normalized_kind = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-")
        label = normalized_kind or "pii"
        return f"fake-{label}-{first % 1_000_000:06d}"

    def morph_pii(self, original: str, kind: str) -> str:
        if not self.enabled or not original:
            return original
        key = self.short_hash(original)
        existing = self._mapping.get(key)
        if existing is not None:
            return existing
        fake = self._fake_value(kind, key)
        self._mapping[key] = fake
        return fake

    def _detected_entities(self, text: str) -> tuple[PIIEntity, ...]:
        entities: list[PIIEntity] = []
        for kind, pattern in _PII_PATTERN_SPECS:
            for match in pattern.finditer(text):
                if kind == "credit_card" and not _looks_like_credit_card(
                    match.group(0)
                ):
                    continue
                entities.append(
                    PIIEntity(kind=kind, start=match.start(), end=match.end())
                )
        return tuple(entities)

    @staticmethod
    def _select_entities(
        text: str,
        explicit: Sequence[PIIEntity],
        detected: Sequence[PIIEntity],
    ) -> tuple[PIIEntity, ...]:
        candidates: list[tuple[int, int, int, PIIEntity]] = []
        for priority, entities in ((0, explicit), (1, detected)):
            for entity in entities:
                if (
                    not entity.kind.strip()
                    or entity.start < 0
                    or entity.end <= entity.start
                    or entity.end > len(text)
                ):
                    raise ValueError("PII entity spans must be valid and non-empty")
                candidates.append(
                    (entity.start, priority, -(entity.end - entity.start), entity)
                )
        candidates.sort(key=lambda item: item[:3])
        selected: list[PIIEntity] = []
        occupied_until = -1
        for _, _, _, entity in candidates:
            if entity.start < occupied_until:
                continue
            selected.append(entity)
            occupied_until = entity.end
        return tuple(selected)

    def morph_text(
        self,
        text: str,
        *,
        kind: str | None = None,
        entities: Sequence[PIIEntity] = (),
    ) -> str:
        if not self.enabled or not text:
            return text
        if kind is not None:
            leading = len(text) - len(text.lstrip())
            trailing = len(text) - len(text.rstrip())
            end = len(text) - trailing if trailing else len(text)
            core = text[leading:end]
            if not core:
                return text
            return text[:leading] + self.morph_pii(core, kind) + text[end:]
        selected = self._select_entities(
            text,
            entities,
            self._detected_entities(text),
        )
        result = text
        for entity in reversed(selected):
            original = text[entity.start : entity.end]
            replacement = self.morph_pii(original, entity.kind)
            result = result[: entity.start] + replacement + result[entity.end :]
        return result

    def morph(self, value: Any, *, key_hint: Any = None) -> Any:
        if not self.enabled:
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self.morph(item, key_hint=key) for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self.morph(item) for item in value)
        if isinstance(value, list):
            return [self.morph(item) for item in value]
        if isinstance(value, str):
            return self.morph_text(value, kind=_infer_pii_kind(key_hint))
        return value


@dataclass(frozen=True)
class PrivacyRedactionPolicy:
    # Keys are compared after dropping case and punctuation and also match as
    # suffixes, so ``apiKey``, ``x-api-key``, ``client_secret`` and
    # ``refresh_token`` are covered while ``token_count`` is not.
    sensitive_keys: frozenset[str] = frozenset(
        {
            "password",
            "passwd",
            "secret",
            "token",
            "authorization",
            "api_key",
            "access_token",
            "refresh_token",
            "private_key",
            "credential",
            "credentials",
            "cookie",
        }
    )
    replacement: str = "[REDACTED]"
    text_patterns: tuple[str, ...] = ()


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


class PrivacyRedactor:
    def __init__(
        self,
        policy: PrivacyRedactionPolicy = PrivacyRedactionPolicy(),
        *,
        pii_morpher: PIIMorpher | None = None,
    ) -> None:
        self.policy = policy
        self.pii_morpher = pii_morpher or PIIMorpher()
        self._patterns = tuple(re.compile(pattern) for pattern in policy.text_patterns)
        self._sensitive = tuple(
            normalized
            for normalized in (_normalize_key(key) for key in policy.sensitive_keys)
            if normalized
        )

    def _is_sensitive(self, key: Any) -> bool:
        normalized = _normalize_key(str(key))
        return any(normalized.endswith(item) for item in self._sensitive)

    def _redact(self, value: Any, *, key_hint: Any = None) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    self.policy.replacement
                    if self._is_sensitive(key)
                    else self._redact(item, key_hint=key)
                )
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._redact(item) for item in value)
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            result = value
            for pattern in self._patterns:
                result = pattern.sub(self.policy.replacement, result)
            return self.pii_morpher.morph_text(
                result,
                kind=_infer_pii_kind(key_hint),
            )
        return value

    def redact(self, value: Any) -> Any:
        return self._redact(value)

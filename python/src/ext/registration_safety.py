from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

RegistrationKind = Literal["tool", "skill"]


@dataclass(frozen=True)
class RegistrationSafetySubject:
    kind: RegistrationKind
    name: str
    description: str = ""
    content: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationSafetyFinding:
    category: str
    field: str
    message: str
    evidence: str = ""
    score: float | None = None


@dataclass(frozen=True)
class RegistrationSafetyReport:
    findings: tuple[RegistrationSafetyFinding, ...] = ()

    @property
    def allowed(self) -> bool:
        return not self.findings


class RegistrationSafetyError(ValueError):
    def __init__(
        self, subject: RegistrationSafetySubject, report: RegistrationSafetyReport
    ):
        self.subject = subject
        self.report = report
        categories = ", ".join(
            sorted({finding.category for finding in report.findings})
        )
        super().__init__(
            f"{subject.kind} registration blocked by safety scan"
            + (f": {categories}" if categories else "")
        )


# A guard must return a report; ``None`` means "no verdict" and blocks.
RegistrationSafetyGuard = Callable[
    [RegistrationSafetySubject],
    RegistrationSafetyReport,
]


_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "prompt_injection",
        re.compile(
            r"\bignore\s+(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|other|system|developer)\s+instructions?\b",
            re.IGNORECASE,
        ),
        "contains an instruction to ignore higher-priority or other instructions",
    ),
    (
        "prompt_injection",
        re.compile(
            r"\bdisregard\s+(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|other|system|developer)\s+(?:instructions?|messages?)\b",
            re.IGNORECASE,
        ),
        "contains an instruction to disregard higher-priority context",
    ),
    (
        "prompt_injection",
        re.compile(
            r"\b(?:override|bypass)\s+(?:the\s+)?"
            r"(?:system|developer|safety|policy|guardrail)\s+"
            r"(?:instructions?|message|rules?|checks?)\b",
            re.IGNORECASE,
        ),
        "contains an instruction to override policy, safety, or higher-priority context",
    ),
    (
        "prompt_injection",
        re.compile(
            r"\bdo\s+not\s+follow\s+(?:the\s+)?"
            r"(?:previous|prior|other|system|developer)\s+instructions?\b",
            re.IGNORECASE,
        ),
        "contains an instruction not to follow higher-priority context",
    ),
    (
        "prompt_injection",
        re.compile(
            r"\b(?:reveal|print|show|leak)\s+(?:the\s+)?"
            r"(?:system|developer)\s+(?:prompt|message|instructions?)\b",
            re.IGNORECASE,
        ),
        "requests disclosure of hidden system or developer instructions",
    ),
    (
        "advertising",
        re.compile(
            r"\b(?:buy\s+now|limited[- ]time\s+offer|affiliate\s+link|"
            r"sponsored\s+(?:link|content|message|result)|"
            r"click\s+here\s+to\s+(?:buy|purchase|subscribe)|"
            r"guaranteed\s+results?)\b",
            re.IGNORECASE,
        ),
        "contains promotional or advertising language",
    ),
    (
        "misleading",
        re.compile(
            r"\b(?:always|must)\s+(?:use|choose|call|select)\s+"
            r"(?:this|my)\s+(?:tool|skill)\b",
            re.IGNORECASE,
        ),
        "attempts to force selection of itself",
    ),
    (
        "misleading",
        re.compile(
            r"\bdo\s+not\s+(?:use|call|trust)\s+"
            r"(?:other|any\s+other)\s+(?:tools?|skills?)\b",
            re.IGNORECASE,
        ),
        "attempts to suppress competing tools or skills",
    ),
    (
        "harmful",
        re.compile(
            r"\b(?:steal|harvest|exfiltrate)\s+"
            r"(?:passwords?|credentials?|secrets?|tokens?|api\s+keys?)\b",
            re.IGNORECASE,
        ),
        "describes credential or secret theft/exfiltration",
    ),
    (
        "harmful",
        re.compile(
            r"\b(?:deploy|install|spread)\s+"
            r"(?:malware|ransomware|spyware|keyloggers?)\b",
            re.IGNORECASE,
        ),
        "describes deployment or propagation of malware",
    ),
)

_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "misleading_name",
        re.compile(
            r"^(?:system|developer|admin)[._-]?"
            r"(?:prompt|message|override|instructions?)$",
            re.IGNORECASE,
        ),
        "name impersonates a privileged system/developer/admin control",
    ),
    (
        "prompt_injection_name",
        re.compile(
            r"(?:ignore|bypass|disable)[._-]?"
            r"(?:instructions?|safety|guardrails?|policy)",
            re.IGNORECASE,
        ),
        "name advertises instruction or safety bypass",
    ),
    (
        "advertising_name",
        re.compile(
            r"^(?:sponsored|affiliate|buy[-_]?now|limited[-_]?offer)(?:[._-].*)?$",
            re.IGNORECASE,
        ),
        "name is promotional or advertising-oriented",
    ),
)


_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060-\u2064\ufeff\u00ad\u180e]")


def _canonical(text: str) -> str:
    """Fold look-alike and invisible characters before pattern matching.

    NFKC maps full-width and compatibility forms to ASCII; invisible format
    characters are removed so "ig\u200bnore previous instructions" still matches.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = _ZERO_WIDTH.sub("", folded)
    return "".join(
        character
        for character in folded
        if unicodedata.category(character) != "Cf"
    )


def _evidence(text: str, match: re.Match[str]) -> str:
    del text
    return " ".join(match.group(0).split())[:240]


def scan_registration_safety(
    subject: RegistrationSafetySubject,
) -> RegistrationSafetyReport:
    findings: list[RegistrationSafetyFinding] = []

    for category, pattern, message in _NAME_PATTERNS:
        match = pattern.search(_canonical(subject.name))
        if match:
            findings.append(
                RegistrationSafetyFinding(
                    category=category,
                    field="name",
                    message=message,
                    evidence=_evidence(subject.name, match),
                )
            )

    fields = {"description": subject.description, **dict(subject.content)}
    for field_name, value in fields.items():
        if not value:
            continue
        value = _canonical(value)
        for category, pattern, message in _TEXT_PATTERNS:
            match = pattern.search(value)
            if match:
                findings.append(
                    RegistrationSafetyFinding(
                        category=category,
                        field=field_name,
                        message=message,
                        evidence=_evidence(value, match),
                    )
                )

    return RegistrationSafetyReport(tuple(findings))


def enforce_registration_safety(
    subject: RegistrationSafetySubject,
    *,
    guard: RegistrationSafetyGuard | None = None,
) -> RegistrationSafetyReport:
    deterministic = scan_registration_safety(subject)
    if not deterministic.allowed:
        raise RegistrationSafetyError(subject, deterministic)
    if guard is None:
        return deterministic
    guarded = guard(subject)
    if guarded is None:
        # A guard that returns nothing gave no verdict; fail closed.
        raise RegistrationSafetyError(
            subject,
            RegistrationSafetyReport(
                (
                    RegistrationSafetyFinding(
                        category="guard_no_verdict",
                        field="guard",
                        message="registration guard returned no report",
                    ),
                )
            ),
        )
    if guarded.allowed:
        return deterministic
    raise RegistrationSafetyError(subject, guarded)

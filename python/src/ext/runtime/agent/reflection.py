"""Review findings, reflection passes, and critic panels."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from ext.runtime.agent.planning import VerificationReport
from ext.runtime.types import asyncio


@dataclass(frozen=True)
class ReviewFinding:
    severity: Literal["info", "warning", "error"]
    message: str
    repairable: bool = True


@dataclass(frozen=True)
class ReviewResult:
    findings: tuple[ReviewFinding, ...] = ()

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


class ReflectionPass:
    def __init__(
        self,
        reviewer: Callable[[Any], Awaitable[ReviewResult]],
        repair: Callable[[Any, ReviewResult], Awaitable[Any]],
        *,
        max_repairs: int = 1,
    ) -> None:
        self.reviewer = reviewer
        self.repair = repair
        self.max_repairs = max_repairs

    async def run(self, work: Any) -> tuple[Any, ReviewResult]:
        current = work
        for attempt in range(self.max_repairs + 1):
            review = await self.reviewer(current)
            if review.passed or attempt >= self.max_repairs:
                return current, review
            if not any(finding.repairable for finding in review.findings):
                return current, review
            current = await self.repair(current, review)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class CriticAgent:
    name: str
    review: Callable[[Any], Awaitable[VerificationReport]]


class CriticPanel:
    def __init__(self, critics: Sequence[CriticAgent]) -> None:
        self.critics = tuple(critics)

    async def review(self, work: Any) -> tuple[VerificationReport, ...]:
        reports = await asyncio.gather(
            *(critic.review(work) for critic in self.critics)
        )
        return tuple(
            replace(report, reviewer=report.reviewer or critic.name)
            for critic, report in zip(self.critics, reports)
        )

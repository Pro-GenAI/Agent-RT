from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping, Sequence
import asyncio


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operation: Callable[[], Awaitable[Any]]
    iterations: int = 10
    warmup_iterations: int = 1
    category: str = "runtime"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark name must not be empty")
        if not self.category.strip():
            raise ValueError("benchmark category must not be empty")
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool):
            raise TypeError("iterations must be an integer")
        if not isinstance(self.warmup_iterations, int) or isinstance(self.warmup_iterations, bool):
            raise TypeError("warmup_iterations must be an integer")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    category: str
    durations_seconds: tuple[float, ...]

    @property
    def iterations(self) -> int:
        return len(self.durations_seconds)

    @property
    def mean_seconds(self) -> float:
        return mean(self.durations_seconds)

    @property
    def min_seconds(self) -> float:
        return min(self.durations_seconds)

    @property
    def max_seconds(self) -> float:
        return max(self.durations_seconds)


class BenchmarkRunner:
    async def run(self, case: BenchmarkCase) -> BenchmarkResult:
        for _ in range(case.warmup_iterations):
            await case.operation()

        durations: list[float] = []
        for _ in range(case.iterations):
            started = perf_counter()
            await case.operation()
            durations.append(perf_counter() - started)

        return BenchmarkResult(case.name, case.category, tuple(durations))

    async def run_all(self, cases: Sequence[BenchmarkCase]) -> tuple[BenchmarkResult, ...]:
        return tuple([await self.run(case) for case in cases])


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    check: Callable[[Any], Awaitable[None] | None]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("conformance check name must not be empty")


@dataclass(frozen=True)
class ConformanceResult:
    name: str
    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class ConformanceReport:
    subject: str
    results: tuple[ConformanceResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(not result.passed for result in self.results)


class ConformanceSuite:
    def __init__(self, name: str, checks: Sequence[ConformanceCheck] = ()) -> None:
        if not name.strip():
            raise ValueError("conformance suite name must not be empty")
        self.name = name
        self._checks = list(checks)

    def add(self, check: ConformanceCheck) -> None:
        self._checks.append(check)

    async def run(self, subject_name: str, subject: Any) -> ConformanceReport:
        results: list[ConformanceResult] = []
        for item in self._checks:
            try:
                outcome = item.check(subject)
                if asyncio.iscoroutine(outcome):
                    await outcome
                results.append(ConformanceResult(item.name, True))
            except Exception as exc:
                results.append(
                    ConformanceResult(
                        item.name,
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        return ConformanceReport(subject_name, tuple(results))


@dataclass(frozen=True)
class HealthStatus:
    name: str
    healthy: bool
    readiness: bool = True
    liveness: bool = True
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


HealthCheck = Callable[[], Awaitable[HealthStatus] | HealthStatus]


@dataclass(frozen=True)
class HealthReport:
    checks: tuple[HealthStatus, ...]

    @property
    def live(self) -> bool:
        return all(check.liveness for check in self.checks)

    @property
    def ready(self) -> bool:
        return all(check.readiness and check.healthy for check in self.checks)

    @property
    def healthy(self) -> bool:
        return all(check.healthy for check in self.checks)


class HealthRegistry:
    SUPPORTED_KINDS = frozenset(
        {"model", "store", "queue", "sandbox", "connector", "runtime"}
    )

    def __init__(self) -> None:
        self._checks: dict[tuple[str, str], HealthCheck] = {}

    def register(
        self,
        kind: str,
        name: str,
        check: HealthCheck,
        *,
        replace_existing: bool = False,
    ) -> None:
        if kind not in self.SUPPORTED_KINDS:
            raise ValueError(f"unsupported health-check kind: {kind}")
        if not name.strip():
            raise ValueError("health-check name must not be empty")
        key = (kind, name)
        if key in self._checks and not replace_existing:
            raise ValueError(f"health check already registered: {kind}/{name}")
        self._checks[key] = check

    async def check(
        self,
        *,
        kind: str | None = None,
        timeout_seconds: float | None = None,
    ) -> HealthReport:
        if kind is not None and kind not in self.SUPPORTED_KINDS:
            raise ValueError(f"unsupported health-check kind: {kind}")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        statuses: list[HealthStatus] = []
        for (registered_kind, name), check in sorted(self._checks.items()):
            if kind is not None and kind != registered_kind:
                continue
            try:
                outcome = check()
                if asyncio.iscoroutine(outcome):
                    status = (
                        await asyncio.wait_for(outcome, timeout_seconds)
                        if timeout_seconds is not None
                        else await outcome
                    )
                else:
                    status = outcome
                if status.name != name:
                    status = HealthStatus(
                        name=name,
                        healthy=status.healthy,
                        readiness=status.readiness,
                        liveness=status.liveness,
                        detail=status.detail,
                        metadata={
                            **dict(status.metadata),
                            "kind": registered_kind,
                        },
                    )
                elif "kind" not in status.metadata:
                    status = HealthStatus(
                        name=status.name,
                        healthy=status.healthy,
                        readiness=status.readiness,
                        liveness=status.liveness,
                        detail=status.detail,
                        metadata={**dict(status.metadata), "kind": registered_kind},
                    )
                statuses.append(status)
            except Exception as exc:
                statuses.append(
                    HealthStatus(
                        name=name,
                        healthy=False,
                        readiness=False,
                        liveness=True,
                        detail=f"{type(exc).__name__}: {exc}",
                        metadata={"kind": registered_kind},
                    )
                )
        return HealthReport(tuple(statuses))

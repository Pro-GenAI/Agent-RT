"""Runtime metric points and counters."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    kind: Literal["counter", "gauge", "histogram"]
    labels: Mapping[str, str] = field(default_factory=dict)
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class RuntimeMetrics:
    def __init__(self) -> None:
        self._points: list[MetricPoint] = []

    def record(
        self,
        name: str,
        value: float,
        *,
        kind: Literal["counter", "gauge", "histogram"] = "counter",
        labels: Mapping[str, str] | None = None,
    ) -> MetricPoint:
        point = MetricPoint(name, float(value), kind, dict(labels or {}))
        self._points.append(point)
        return point

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> MetricPoint:
        return self.record(name, amount, kind="counter", labels=labels)

    def values(
        self,
        name: str,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> tuple[float, ...]:
        expected = dict(labels or {})
        return tuple(
            point.value
            for point in self._points
            if point.name == name
            and all(point.labels.get(key) == value for key, value in expected.items())
        )

    def total(
        self,
        name: str,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> float:
        return sum(self.values(name, labels=labels))

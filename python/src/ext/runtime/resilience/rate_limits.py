"""Circuit breakers and token-bucket rate limiting."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

CircuitState = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.state: CircuitState = "closed"
        self.failure_count = 0
        self.opened_at: float | None = None

    def check(self) -> CircuitState:
        if self.state == "open":
            if self.opened_at is None:
                raise RuntimeError("open circuit breaker is missing opened_at")
            if self.clock() - self.opened_at >= self.cooldown_seconds:
                self.state = "half_open"
            else:
                raise CircuitOpenError("circuit breaker is open")
        return self.state

    def record_success(self) -> None:
        self.state = "closed"
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.state == "half_open" or self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.opened_at = self.clock()

    async def execute(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        self.check()
        try:
            value = await operation()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return value


@dataclass(frozen=True)
class RateLimit:
    limit: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("rate limit must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("rate-limit window must be positive")


class RateLimitExceededError(RuntimeError):
    def __init__(self, key: str, retry_after_seconds: float) -> None:
        self.key = key
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limit exceeded for {key}")


class RateLimiter:
    def __init__(
        self,
        default_limit: RateLimit,
        *,
        limits: Mapping[str, RateLimit] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.default_limit = default_limit
        self.limits = dict(limits or {})
        self.clock = clock
        self._events: dict[str, list[float]] = {}
        self._prune_at = 1024

    def _prune(self, now: float) -> None:
        """Drop keys whose events have all left their window (bounds memory)."""
        if len(self._events) < self._prune_at:
            return
        for key, events in tuple(self._events.items()):
            policy = self.limits.get(key, self.default_limit)
            if not events or events[-1] <= now - policy.window_seconds:
                del self._events[key]
        self._prune_at = max(1024, 2 * len(self._events))

    def check(self, key: str, *, cost: int = 1) -> int:
        if cost < 1:
            raise ValueError("rate-limit cost must be at least 1")
        policy = self.limits.get(key, self.default_limit)
        now = self.clock()
        self._prune(now)
        cutoff = now - policy.window_seconds
        events = [
            timestamp for timestamp in self._events.get(key, ()) if timestamp > cutoff
        ]
        if len(events) + cost > policy.limit:
            retry_after = (
                policy.window_seconds - (now - events[0])
                if events
                else policy.window_seconds
            )
            raise RateLimitExceededError(key, max(0.0, retry_after))
        events.extend([now] * cost)
        self._events[key] = events
        return policy.limit - len(events)

    def check_many(self, keys: Sequence[str], *, cost: int = 1) -> Mapping[str, int]:
        if cost < 1:
            raise ValueError("rate-limit cost must be at least 1")
        now = self.clock()
        self._prune(now)
        prepared: dict[str, tuple[RateLimit, list[float]]] = {}
        for key in keys:
            policy = self.limits.get(key, self.default_limit)
            cutoff = now - policy.window_seconds
            events = [
                timestamp
                for timestamp in self._events.get(key, ())
                if timestamp > cutoff
            ]
            if len(events) + cost > policy.limit:
                retry_after = (
                    policy.window_seconds - (now - events[0])
                    if events
                    else policy.window_seconds
                )
                raise RateLimitExceededError(key, max(0.0, retry_after))
            prepared[key] = (policy, events)
        remaining: dict[str, int] = {}
        for key, (policy, events) in prepared.items():
            events.extend([now] * cost)
            self._events[key] = events
            remaining[key] = policy.limit - len(events)
        return remaining

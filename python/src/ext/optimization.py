from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Mapping, MutableMapping, Sequence, TypeVar
import asyncio
import copy
import math
import time


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: float | None = None
    cacheable: bool = True
    namespace: str = "default"

    def __post_init__(self) -> None:
        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if not self.namespace.strip():
            raise ValueError("cache namespace must not be empty")


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float | None


@dataclass(frozen=True)
class CacheLookup:
    found: bool
    value: Any = None


class ResponseCache:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int | None = 4096,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._clock = clock
        self.max_entries = max_entries
        self._entries: MutableMapping[tuple[str, str], _CacheEntry] = {}

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("cache namespace must not be empty")

    def lookup(self, key: str, *, namespace: str = "default") -> CacheLookup:
        self._validate_namespace(namespace)
        entry = self._entries.get((namespace, key))
        if entry is None:
            return CacheLookup(False)
        if entry.expires_at is not None and self._clock() >= entry.expires_at:
            del self._entries[(namespace, key)]
            return CacheLookup(False)
        return CacheLookup(True, copy.deepcopy(entry.value))

    def get(self, key: str, *, namespace: str = "default") -> Any | None:
        result = self.lookup(key, namespace=namespace)
        return result.value if result.found else None

    def set(self, key: str, value: Any, *, policy: CachePolicy = CachePolicy()) -> None:
        if not policy.cacheable:
            return
        expires_at = None
        if policy.ttl_seconds is not None:
            expires_at = self._clock() + policy.ttl_seconds
        identity = (policy.namespace, key)
        self._entries.pop(identity, None)  # re-insert so eviction order is by write time
        self._entries[identity] = _CacheEntry(copy.deepcopy(value), expires_at)
        self._evict()

    def _evict(self) -> None:
        """Bound memory: drop expired entries first, then the oldest writes."""
        if self.max_entries is None or len(self._entries) <= self.max_entries:
            return
        now = self._clock()
        for identity, entry in tuple(self._entries.items()):
            if entry.expires_at is not None and now >= entry.expires_at:
                del self._entries[identity]
        while len(self._entries) > self.max_entries:
            del self._entries[next(iter(self._entries))]

    def invalidate(self, key: str, *, namespace: str = "default") -> bool:
        self._validate_namespace(namespace)
        return self._entries.pop((namespace, key), None) is not None

    def clear_namespace(self, namespace: str) -> int:
        self._validate_namespace(namespace)
        keys = [key for key in self._entries if key[0] == namespace]
        for key in keys:
            del self._entries[key]
        return len(keys)


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def run(self, key: str, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(operation())
                self._inflight[key] = task
                task.add_done_callback(lambda done, k=key: self._release(k, done))
        # The shared task outlives any single waiter, including the one that
        # started it, so the entry is removed only when the task itself ends.
        return await asyncio.shield(task)

    def _release(self, key: str, task: "asyncio.Task[Any]") -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()  # mark retrieved; waiters still receive it


class BatchExecutor(Generic[T, R]):
    def __init__(
        self,
        handler: Callable[[Sequence[T]], Awaitable[Sequence[R]]],
        *,
        max_batch_size: int = 32,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._handler = handler
        self._max_batch_size = max_batch_size

    async def execute(self, items: Sequence[T]) -> tuple[R, ...]:
        results: list[R] = []
        for index in range(0, len(items), self._max_batch_size):
            batch = items[index:index + self._max_batch_size]
            batch_results = tuple(await self._handler(batch))
            if len(batch_results) != len(batch):
                raise ValueError("batch handler must return one result per input")
            results.extend(batch_results)
        return tuple(results)


@dataclass(frozen=True)
class LatencyOptimizationPolicy:
    parallel_guardrails: bool = True
    prefetch: bool = True
    streaming: bool = True
    speculative_execution: bool = False
    max_parallelism: int = 4

    def __post_init__(self) -> None:
        if self.max_parallelism < 1:
            raise ValueError("max_parallelism must be at least 1")

    async def run_parallel(
        self,
        operations: Sequence[Callable[[], Awaitable[T]]],
    ) -> tuple[T, ...]:
        if not operations:
            return ()
        if not self.parallel_guardrails or self.max_parallelism == 1:
            return tuple([await operation() for operation in operations])
        semaphore = asyncio.Semaphore(self.max_parallelism)

        async def invoke(operation: Callable[[], Awaitable[T]]) -> T:
            async with semaphore:
                return await operation()

        return tuple(await asyncio.gather(*(invoke(operation) for operation in operations)))

    def critical_path(self, durations: Mapping[str, float], dependencies: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
        memo: dict[str, tuple[float, tuple[str, ...]]] = {}
        visiting: set[str] = set()

        def best(node: str) -> tuple[float, tuple[str, ...]]:
            if node in memo:
                return memo[node]
            if node in visiting:
                raise ValueError(f"dependency cycle detected at {node}")
            visiting.add(node)
            deps = dependencies.get(node, ())
            if not deps:
                result = (durations.get(node, 0.0), (node,))
            else:
                options = [best(dep) for dep in deps]
                weight, path = max(options, key=lambda item: item[0])
                result = (weight + durations.get(node, 0.0), path + (node,))
            visiting.remove(node)
            memo[node] = result
            return result

        if not durations:
            return ()
        _, path = max((best(node) for node in durations), key=lambda item: item[0])
        return path


@dataclass(frozen=True)
class ModelTier:
    name: str
    cost_per_call: float
    quality: float
    max_context_tokens: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model tier name must not be empty")
        if not math.isfinite(self.cost_per_call):
            raise ValueError("cost_per_call must be finite")
        if self.cost_per_call < 0:
            raise ValueError("cost_per_call must be non-negative")
        if not math.isfinite(self.quality):
            raise ValueError("quality must be finite")
        if not 0 <= self.quality <= 1:
            raise ValueError("quality must be between 0 and 1")
        if not isinstance(self.max_context_tokens, int) or isinstance(self.max_context_tokens, bool):
            raise TypeError("max_context_tokens must be an integer")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1")


@dataclass(frozen=True)
class CostOptimizationPolicy:
    budget: float
    minimum_quality: float = 0.0
    prefer_prompt_cache: bool = True
    allow_batching: bool = True
    allow_early_exit: bool = True
    context_token_target: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget):
            raise ValueError("budget must be finite")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if not math.isfinite(self.minimum_quality):
            raise ValueError("minimum_quality must be finite")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum_quality must be between 0 and 1")
        if self.context_token_target is not None:
            if not isinstance(self.context_token_target, int) or isinstance(self.context_token_target, bool):
                raise TypeError("context_token_target must be an integer")
            if self.context_token_target < 1:
                raise ValueError("context_token_target must be at least 1")

    def select_model(self, tiers: Sequence[ModelTier]) -> ModelTier:
        eligible = [
            tier
            for tier in tiers
            if tier.cost_per_call <= self.budget and tier.quality >= self.minimum_quality
        ]
        if not eligible:
            raise RuntimeError("no model tier satisfies budget and quality constraints")
        return min(eligible, key=lambda tier: (tier.cost_per_call, -tier.quality, tier.name))

    def reduce_context(self, token_counts: Sequence[int]) -> tuple[int, ...]:
        for count in token_counts:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("token counts must be non-negative integers")
        if self.context_token_target is None:
            return tuple(token_counts)
        kept: list[int] = []
        total = 0
        for count in reversed(token_counts):
            if total + count > self.context_token_target:
                continue
            kept.append(count)
            total += count
        return tuple(reversed(kept))

    def should_early_exit(self, *, confidence: float, remaining_budget: float) -> bool:
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        if not math.isfinite(remaining_budget):
            raise ValueError("remaining_budget must be finite")
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        if remaining_budget < 0:
            raise ValueError("remaining_budget must be non-negative")
        if not self.allow_early_exit:
            return False
        return confidence >= self.minimum_quality and remaining_budget <= self.budget * 0.25

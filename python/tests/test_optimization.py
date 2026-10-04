import asyncio

import pytest

from ext.optimization import (
    BatchExecutor,
    CachePolicy,
    CostOptimizationPolicy,
    LatencyOptimizationPolicy,
    ModelTier,
    ResponseCache,
    SingleFlight,
)


class TestOptimization:
    def test_cache_respects_namespace_ttl_and_cacheable_policy(self):
        now = [10.0]
        cache = ResponseCache(clock=lambda: now[0])
        cache.set(
            "k", {"value": 1}, policy=CachePolicy(ttl_seconds=5, namespace="model")
        )
        assert cache.get("k", namespace="model") == {"value": 1}
        now[0] = 15.0
        assert cache.get("k", namespace="model") is None
        cache.set("unsafe", 1, policy=CachePolicy(cacheable=False))
        assert cache.get("unsafe") is None

        cache.set("nullable", None)
        assert cache.lookup("nullable").found
        assert cache.lookup("nullable").value is None
        assert not cache.lookup("missing").found
        with pytest.raises(ValueError):
            cache.lookup("k", namespace="")

    async def test_singleflight_collapses_identical_concurrent_operations(self):
        calls = 0
        gate = asyncio.Event()
        singleflight = SingleFlight()

        async def operation():
            nonlocal calls
            calls += 1
            await gate.wait()
            return "done"

        tasks = [
            asyncio.create_task(singleflight.run("same", operation)),
            asyncio.create_task(singleflight.run("same", operation)),
        ]
        for _ in range(3):
            if calls == 1:
                break
            await asyncio.sleep(0)
        assert calls == 1
        gate.set()
        assert await asyncio.gather(*tasks) == ["done", "done"]

        attempts = 0

        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")
            return "recovered"

        with pytest.raises(RuntimeError):
            await singleflight.run("flaky", flaky)
        assert await singleflight.run("flaky", flaky) == "recovered"

    async def test_batch_executor_chunks_and_preserves_order(self):
        seen = []

        async def handler(items):
            seen.append(tuple(items))
            return [item * 2 for item in items]

        batcher = BatchExecutor(handler, max_batch_size=2)
        assert await batcher.execute([1, 2, 3, 4, 5]) == (2, 4, 6, 8, 10)
        assert seen == [(1, 2), (3, 4), (5,)]

    async def test_latency_policy_parallelizes_and_reports_critical_path(self):
        policy = LatencyOptimizationPolicy(max_parallelism=2)

        async def value(number):
            await asyncio.sleep(0)
            return number

        result = await policy.run_parallel(
            [
                lambda: value(1),
                lambda: value(2),
                lambda: value(3),
            ]
        )
        assert result == (1, 2, 3)
        assert policy.critical_path({"a": 2, "b": 5, "c": 3}, {"c": ("a", "b")}) == (
            "b",
            "c",
        )
        with pytest.raises(ValueError):
            policy.critical_path(
                {"a": 1, "b": 1},
                {"a": ("b",), "b": ("a",)},
            )

    def test_cost_policy_selects_lowest_cost_eligible_and_reduces_context(self):
        policy = CostOptimizationPolicy(
            budget=0.5,
            minimum_quality=0.8,
            context_token_target=100,
        )
        selected = policy.select_model(
            [
                ModelTier("cheap", 0.1, 0.7, 1000),
                ModelTier("balanced", 0.3, 0.85, 2000),
                ModelTier("premium", 0.5, 0.95, 4000),
            ]
        )
        assert selected.name == "balanced"
        assert policy.reduce_context([60, 50, 40]) == (50, 40)
        assert policy.should_early_exit(confidence=0.9, remaining_budget=0.1)
        with pytest.raises(ValueError):
            ModelTier("bad", float("nan"), 0.9, 1000)
        with pytest.raises(TypeError):
            ModelTier("bad", 0.1, 0.9, 10.5)
        with pytest.raises(ValueError):
            CostOptimizationPolicy(budget=float("inf"))
        with pytest.raises(TypeError):
            CostOptimizationPolicy(budget=1.0, context_token_target=10.5)
        with pytest.raises(ValueError):
            policy.reduce_context([10, -1])
        with pytest.raises(ValueError):
            policy.should_early_exit(confidence=float("nan"), remaining_budget=0.1)
        with pytest.raises(ValueError):
            policy.should_early_exit(confidence=0.9, remaining_budget=-0.1)

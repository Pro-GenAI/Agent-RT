#!/usr/bin/env python3
"""Compare framework runtime overhead against the local mock LLM API."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Callable

from generate_charts import generate_charts
from mock_llm_api import MODEL_ID, RESPONSE_TEXT, start_mock_server, start_responses_websocket_server

HERE = Path(__file__).resolve().parent
WARMUP_ITERATIONS = 10
FRAMEWORKS = ("Agent RT", "LangChain", "LlamaIndex")
MEMORY_STAGES = (
    "after_framework_import",
    "before_provider_construction",
    "after_provider_construction",
    "after_first_completion",
    "after_warm_completion_series",
    "after_first_stream",
    "after_warm_stream_series",
    "after_five_turn_loop",
    "after_twenty_turn_loop",
    "after_hundred_turn_loop",
    "after_cleanup_gc",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def current_rss_mib() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    return None


def text_messages(turns: int = 1) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"hello {index}"})
        if index + 1 < turns:
            messages.append({"role": "assistant", "content": RESPONSE_TEXT})
    return messages


class Adapter:
    def __init__(
        self,
        create: Callable[[], Any],
        complete: Callable[[Any, list[dict[str, str]]], Any],
        stream: Callable[[Any, list[dict[str, str]]], AsyncIterator[str]],
    ) -> None:
        self.create = create
        self.complete = complete
        self.stream = stream


def load_adapter(framework: str, base_url: str, agent_transport: str = "compatible") -> Adapter:
    if framework == "Agent RT":
        from agent_rt import (
            ContentPart,
            ModelMessage,
            ModelRequest,
            OpenAIModelProvider,
            OpenAIProviderSettings,
        )

        def create() -> Any:
            return OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=base_url,
                    api_key="mock-key",
                    default_model=MODEL_ID,
                    transport=agent_transport,
                )
            )

        def request(messages: list[dict[str, str]]) -> Any:
            return ModelRequest(
                model=MODEL_ID,
                messages=tuple(
                    ModelMessage(
                        role=item["role"],
                        content=(ContentPart(type="text", text=item["content"]),),
                    )
                    for item in messages
                ),
            )

        async def complete(client: Any, messages: list[dict[str, str]]) -> str:
            response = await client.complete(request(messages))
            return "".join(part.text or "" for part in response.message.content)

        async def stream(client: Any, messages: list[dict[str, str]]) -> AsyncIterator[str]:
            async for event in client.stream(request(messages)):
                if event.type == "text_delta" and event.text:
                    yield event.text

        return Adapter(create, complete, stream)

    if framework == "LangChain":
        from langchain_openai import ChatOpenAI

        def create() -> Any:
            return ChatOpenAI(
                model=MODEL_ID,
                api_key="mock-key",
                base_url=base_url,
                temperature=0,
                max_retries=0,
                use_responses_api=False,
            )

        def lc_messages(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
            role_map = {"user": "human", "assistant": "ai", "system": "system"}
            return [(role_map[item["role"]], item["content"]) for item in messages]

        async def complete(client: Any, messages: list[dict[str, str]]) -> str:
            response = await client.ainvoke(lc_messages(messages))
            content = response.content
            return content if isinstance(content, str) else str(content)

        async def stream(client: Any, messages: list[dict[str, str]]) -> AsyncIterator[str]:
            async for chunk in client.astream(lc_messages(messages)):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content

        return Adapter(create, complete, stream)

    if framework == "LlamaIndex":
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.llms.openai import OpenAI

        def create() -> Any:
            return OpenAI(
                model=MODEL_ID,
                api_key="mock-key",
                api_base=base_url,
                temperature=0,
                max_retries=0,
                timeout=10,
            )

        def li_messages(messages: list[dict[str, str]]) -> list[Any]:
            return [ChatMessage(role=item["role"], content=item["content"]) for item in messages]

        async def complete(client: Any, messages: list[dict[str, str]]) -> str:
            response = await client.achat(li_messages(messages))
            return response.message.content or ""

        async def stream(client: Any, messages: list[dict[str, str]]) -> AsyncIterator[str]:
            responses = await client.astream_chat(li_messages(messages))
            async for response in responses:
                if response.delta:
                    yield response.delta

        return Adapter(create, complete, stream)

    raise ValueError(f"unknown framework: {framework}")


def measured_call(fn: Callable[[], Any]) -> tuple[float, float, Any]:
    wall = time.perf_counter()
    cpu = time.process_time()
    value = fn()
    return (
        (time.perf_counter() - wall) * 1000,
        (time.process_time() - cpu) * 1000,
        value,
    )


async def measured_async(fn: Callable[[], Any]) -> tuple[float, float, Any]:
    wall = time.perf_counter()
    cpu = time.process_time()
    value = await fn()
    return (
        (time.perf_counter() - wall) * 1000,
        (time.process_time() - cpu) * 1000,
        value,
    )


async def measured_stream(
    adapter: Adapter,
    client: Any,
    messages: list[dict[str, str]],
) -> tuple[dict[str, float], str]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    first_ms: float | None = None
    parts: list[str] = []
    async for part in adapter.stream(client, messages):
        if first_ms is None:
            first_ms = (time.perf_counter() - started) * 1000
        parts.append(part)
    total_ms = (time.perf_counter() - started) * 1000
    cpu_ms = (time.process_time() - cpu_started) * 1000
    return (
        {
            "first_token_ms": first_ms if first_ms is not None else total_ms,
            "wall_ms": total_ms,
            "cpu_ms": cpu_ms,
        },
        "".join(parts),
    )


def summarize(samples: list[dict[str, float]], key: str = "wall_ms") -> dict[str, float]:
    walls = [row[key] for row in samples]
    cpus = [row["cpu_ms"] for row in samples]
    return {
        "median_ms": statistics.median(walls),
        "p95_ms": percentile(walls, 0.95),
        "median_cpu_percent": statistics.median(
            [(cpu / wall * 100) if wall > 0 else 0.0 for cpu, wall in zip(cpus, walls)]
        ),
    }


def summarize_stream(samples: list[dict[str, float]]) -> dict[str, float]:
    summary = summarize(samples)
    summary["median_first_token_ms"] = statistics.median(
        [row["first_token_ms"] for row in samples]
    )
    summary["p95_first_token_ms"] = percentile(
        [row["first_token_ms"] for row in samples], 0.95
    )
    return summary


def profile_slug(framework: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in framework).strip("-")


def top_allocation_diffs(new_snapshot: Any, old_snapshot: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stat in new_snapshot.compare_to(old_snapshot, "lineno"):
        if stat.size_diff <= 0:
            continue
        rows.append(
            {
                "size_diff_bytes": stat.size_diff,
                "count_diff": stat.count_diff,
                "traceback": stat.traceback.format(),
            }
        )
        if len(rows) == 15:
            break
    return rows


async def worker(
    framework: str,
    base_url: str,
    runs: int,
    *,
    profile: bool = False,
    profile_dir: Path | None = None,
    agent_transport: str = "compatible",
    diagnostic_turns: int | None = None,
) -> int:
    profiler: Any = None
    tracemalloc_module: Any = None
    trace_snapshots: dict[str, Any] = {}
    if profile:
        import cProfile
        import tracemalloc

        profiler = cProfile.Profile()
        profiler.enable()
        tracemalloc.start()
        tracemalloc_module = tracemalloc

    load_wall_ms, load_cpu_ms, adapter = measured_call(
        lambda: load_adapter(framework, base_url, agent_transport)
    )
    module_load_samples = [{"wall_ms": load_wall_ms, "cpu_ms": load_cpu_ms}]

    memory_stages: dict[str, dict[str, float | None]] = {}

    def capture_memory(stage: str) -> None:
        entry: dict[str, float | None] = {"rss_mib": current_rss_mib()}
        if tracemalloc_module is not None:
            current, peak = tracemalloc_module.get_traced_memory()
            entry["tracemalloc_current_mib"] = current / (1024 * 1024)
            entry["tracemalloc_peak_mib"] = peak / (1024 * 1024)
            trace_snapshots[stage] = tracemalloc_module.take_snapshot()
        memory_stages[stage] = entry

    gc.collect()
    capture_memory("after_framework_import")
    capture_memory("before_provider_construction")
    baseline_rss = memory_stages["before_provider_construction"]["rss_mib"]

    wall_ms, cpu_ms, client = measured_call(adapter.create)
    first_construction_samples = [{"wall_ms": wall_ms, "cpu_ms": cpu_ms}]
    capture_memory("after_provider_construction")

    wall_ms, cpu_ms, result = await measured_async(
        lambda: adapter.complete(client, text_messages())
    )
    if result != RESPONSE_TEXT:
        raise RuntimeError(f"{framework} completion returned {result!r}")
    first_completion_samples = [{"wall_ms": wall_ms, "cpu_ms": cpu_ms}]
    capture_memory("after_first_completion")

    # Keep first-use cost separate from the steady-state warm series. Runtime/JIT
    # stacks can require several requests before tail latency stabilizes.
    for _ in range(WARMUP_ITERATIONS):
        result = await adapter.complete(client, text_messages())
        if result != RESPONSE_TEXT:
            raise RuntimeError(f"{framework} completion returned {result!r}")

    completion_samples: list[dict[str, float]] = []
    for _ in range(runs):
        wall_ms, cpu_ms, result = await measured_async(
            lambda: adapter.complete(client, text_messages())
        )
        if result != RESPONSE_TEXT:
            raise RuntimeError(f"{framework} completion returned {result!r}")
        completion_samples.append({"wall_ms": wall_ms, "cpu_ms": cpu_ms})
    capture_memory("after_warm_completion_series")

    first_stream_sample, text = await measured_stream(adapter, client, text_messages())
    if text != RESPONSE_TEXT:
        raise RuntimeError(f"{framework} stream returned {text!r}")
    first_stream_samples = [first_stream_sample]
    capture_memory("after_first_stream")

    for _ in range(WARMUP_ITERATIONS):
        _, text = await measured_stream(adapter, client, text_messages())
        if text != RESPONSE_TEXT:
            raise RuntimeError(f"{framework} stream returned {text!r}")

    stream_samples: list[dict[str, float]] = []
    for _ in range(runs):
        sample, text = await measured_stream(adapter, client, text_messages())
        if text != RESPONSE_TEXT:
            raise RuntimeError(f"{framework} stream returned {text!r}")
        stream_samples.append(sample)
    capture_memory("after_warm_stream_series")

    conversation_turn_counts = [5, 20, 100]
    if diagnostic_turns is not None and diagnostic_turns not in conversation_turn_counts:
        conversation_turn_counts.append(diagnostic_turns)
    conversation_samples: dict[int, list[dict[str, float]]] = {
        turn_count: [] for turn_count in conversation_turn_counts
    }
    conversation_turn_samples: dict[int, list[dict[str, float]]] = {
        turn_count: [] for turn_count in conversation_turn_counts
    }
    conversation_stages = {
        5: "after_five_turn_loop",
        20: "after_twenty_turn_loop",
        100: "after_hundred_turn_loop",
    }
    if diagnostic_turns is not None:
        conversation_stages[diagnostic_turns] = "after_diagnostic_turn_loop"
    for turn_count in conversation_turn_counts:
        memory_stage = conversation_stages[turn_count]
        for run_index in range(runs):
            messages: list[dict[str, str]] = []

            async def run_turns() -> None:
                for turn_index in range(turn_count):
                    messages.append({"role": "user", "content": f"turn {turn_index}"})
                    turn_wall_ms, turn_cpu_ms, value = await measured_async(
                        lambda: adapter.complete(client, messages)
                    )
                    if value != RESPONSE_TEXT:
                        raise RuntimeError(
                            f"{framework} completion returned {value!r}"
                        )
                    conversation_turn_samples[turn_count].append({
                        "wall_ms": turn_wall_ms,
                        "cpu_ms": turn_cpu_ms,
                        "turn": turn_index + 1,
                        "conversation_run": run_index + 1,
                    })
                    messages.append({"role": "assistant", "content": value})

            wall_ms, cpu_ms, _ = await measured_async(run_turns)
            conversation_samples[turn_count].append({
                "wall_ms": wall_ms,
                "cpu_ms": cpu_ms,
            })
        capture_memory(memory_stage)

    multi_turn_samples = conversation_samples[5]
    final_rss = memory_stages["after_five_turn_loop"]["rss_mib"]

    def summarize_conversation(turn_count: int) -> dict[str, float]:
        summary = summarize(conversation_samples[turn_count])
        incremental = summarize(conversation_turn_samples[turn_count])
        summary["per_turn_ms"] = summary["median_ms"] / turn_count
        summary["incremental_turn_median_ms"] = incremental["median_ms"]
        summary["incremental_turn_p95_ms"] = incremental["p95_ms"]
        return summary

    async def run_hundred_requests() -> None:
        for _ in range(100):
            value = await adapter.complete(client, text_messages())
            if value != RESPONSE_TEXT:
                raise RuntimeError(f"{framework} completion returned {value!r}")

    wall_ms, cpu_ms, _ = await measured_async(run_hundred_requests)
    warm_100_samples = [{"wall_ms": wall_ms, "cpu_ms": cpu_ms}]
    warm_100_summary = summarize(warm_100_samples)
    warm_100_summary["per_request_ms"] = wall_ms / 100

    concurrency_samples: dict[int, list[dict[str, float]]] = {
        1: [],
        4: [],
        16: [],
    }
    for concurrency in (1, 4, 16):
        for _ in range(runs):
            async def run_concurrent_requests() -> None:
                values = await asyncio.gather(
                    *(
                        adapter.complete(client, text_messages())
                        for _ in range(concurrency)
                    )
                )
                for value in values:
                    if value != RESPONSE_TEXT:
                        raise RuntimeError(
                            f"{framework} completion returned {value!r}"
                        )

            wall_ms, cpu_ms, _ = await measured_async(run_concurrent_requests)
            concurrency_samples[concurrency].append({
                "wall_ms": wall_ms,
                "cpu_ms": cpu_ms,
                "concurrency": concurrency,
            })

    def summarize_concurrency(concurrency: int) -> dict[str, float]:
        summary = summarize(concurrency_samples[concurrency])
        summary["per_request_ms"] = summary["median_ms"] / concurrency
        summary["throughput_rps"] = (
            concurrency * 1000.0 / summary["median_ms"]
            if summary["median_ms"] > 0
            else math.inf
        )
        return summary

    concurrent_samples = concurrency_samples[16]
    concurrent_summary = summarize_concurrency(16)

    # Measure construction after first-use work so it is explicitly a subsequent
    # construction metric and cannot move initialization out of first request.
    construct_samples: list[dict[str, float]] = []
    subsequent_client: Any = None
    for _ in range(runs):
        wall_ms, cpu_ms, subsequent_client = measured_call(adapter.create)
        construct_samples.append({"wall_ms": wall_ms, "cpu_ms": cpu_ms})

    # Explicit cleanup is diagnostic only. Headline RSS remains the stage after
    # the five-turn loop so it is not dependent on GC behavior.
    if framework == "Agent RT" and hasattr(client, "close"):
        await client.close()
    client = None
    subsequent_client = None
    gc.collect()
    capture_memory("after_cleanup_gc")

    profile_data: dict[str, Any] | None = None
    if profile:
        assert profiler is not None
        assert tracemalloc_module is not None
        profiler.disable()
        destination = profile_dir or (HERE / "results" / "profiles")
        destination.mkdir(parents=True, exist_ok=True)
        cpu_profile = destination / f"{profile_slug(framework)}.prof"
        profiler.dump_stats(cpu_profile)
        profile_data = {
            "cpu_profile": str(cpu_profile),
            "provider_construction_top_allocations": top_allocation_diffs(
                trace_snapshots["after_provider_construction"],
                trace_snapshots["before_provider_construction"],
            ),
            "retained_top_allocations": top_allocation_diffs(
                trace_snapshots["after_cleanup_gc"],
                trace_snapshots["after_framework_import"],
            ),
        }
        tracemalloc_module.stop()

    first_stream_summary = summarize_stream(first_stream_samples)
    stream_summary = summarize_stream(stream_samples)
    subsequent_construction_summary = summarize(construct_samples)

    diagnostic_summary = (
        {
            "turns": diagnostic_turns,
            **summarize_conversation(diagnostic_turns),
        }
        if diagnostic_turns is not None
        else None
    )
    payload = {
        "framework": framework,
        "runs": runs,
        "baseline_rss_mib": baseline_rss,
        "final_rss_mib": final_rss,
        "rss_delta_mib": (
            max(0.0, final_rss - baseline_rss)
            if baseline_rss is not None and final_rss is not None
            else None
        ),
        "module_load": summarize(module_load_samples),
        "first_construction": summarize(first_construction_samples),
        "subsequent_construction": subsequent_construction_summary,
        "construction": subsequent_construction_summary,
        "first_completion": summarize(first_completion_samples),
        "completion": summarize(completion_samples),
        "first_streaming": first_stream_summary,
        "streaming": stream_summary,
        "five_turn_loop": summarize_conversation(5),
        "twenty_turn_loop": summarize_conversation(20),
        "hundred_turn_loop": summarize_conversation(100),
        "warm_100_request_loop": warm_100_summary,
        "concurrent_1_request_batch": summarize_concurrency(1),
        "concurrent_4_request_batch": summarize_concurrency(4),
        "concurrent_16_request_batch": concurrent_summary,
        "diagnostic_turn_loop": diagnostic_summary,
        "memory_stages": memory_stages,
        "profile": profile_data,
        "samples": {
            "module_load": module_load_samples,
            "first_construction": first_construction_samples,
            "subsequent_construction": construct_samples,
            "construction": construct_samples,
            "first_completion": first_completion_samples,
            "completion": completion_samples,
            "first_streaming": first_stream_samples,
            "streaming": stream_samples,
            "five_turn_loop": multi_turn_samples,
            "five_turn_incremental": conversation_turn_samples[5],
            "twenty_turn_loop": conversation_samples[20],
            "twenty_turn_incremental": conversation_turn_samples[20],
            "hundred_turn_loop": conversation_samples[100],
            "hundred_turn_incremental": conversation_turn_samples[100],
            "warm_100_request_loop": warm_100_samples,
            "concurrent_1_request_batch": concurrency_samples[1],
            "concurrent_4_request_batch": concurrency_samples[4],
            "concurrent_16_request_batch": concurrent_samples,
            **(
                {
                    "diagnostic_turn_loop": conversation_samples[diagnostic_turns],
                    "diagnostic_turn_incremental": conversation_turn_samples[diagnostic_turns],
                }
                if diagnostic_turns is not None
                else {}
            ),
        },
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime-summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    summary_fields = [
        "framework", "runs", "construction_median_ms", "construction_p95_ms",
        "construction_cpu_percent", "completion_median_ms", "completion_p95_ms",
        "completion_cpu_percent", "stream_first_token_median_ms",
        "stream_first_token_p95_ms", "stream_total_median_ms", "stream_total_p95_ms",
        "stream_cpu_percent", "five_turn_median_ms", "five_turn_p95_ms",
        "five_turn_cpu_percent", "five_turn_incremental_median_ms",
        "twenty_turn_median_ms", "twenty_turn_p95_ms", "twenty_turn_per_turn_ms",
        "twenty_turn_incremental_median_ms", "twenty_turn_incremental_p95_ms",
        "hundred_turn_median_ms", "hundred_turn_p95_ms", "hundred_turn_per_turn_ms",
        "hundred_turn_incremental_median_ms", "hundred_turn_incremental_p95_ms",
        "warm_100_total_ms", "warm_100_per_request_ms",
        "concurrent_1_total_ms", "concurrent_1_p95_ms", "concurrent_1_throughput_rps",
        "concurrent_4_total_ms", "concurrent_4_p95_ms", "concurrent_4_throughput_rps",
        "concurrent_16_total_ms", "concurrent_16_p95_ms",
        "concurrent_16_per_request_ms", "concurrent_16_throughput_rps",
        "baseline_rss_mib", "final_rss_mib", "rss_delta_mib",
        "module_load_median_ms", "module_load_p95_ms", "module_load_cpu_percent",
        "first_construction_median_ms", "first_construction_p95_ms",
        "first_construction_cpu_percent", "subsequent_construction_median_ms",
        "subsequent_construction_p95_ms", "subsequent_construction_cpu_percent",
        "first_completion_median_ms", "first_completion_p95_ms",
        "first_completion_cpu_percent", "first_stream_first_token_median_ms",
        "first_stream_first_token_p95_ms", "first_stream_total_median_ms",
        "first_stream_total_p95_ms", "first_stream_cpu_percent",
        "rss_after_framework_import_mib", "rss_before_provider_construction_mib",
        "rss_after_provider_construction_mib", "rss_after_first_completion_mib",
        "rss_after_warm_completion_series_mib", "rss_after_first_stream_mib",
        "rss_after_warm_stream_series_mib", "rss_after_five_turn_loop_mib",
        "rss_after_twenty_turn_loop_mib", "rss_after_hundred_turn_loop_mib",
        "rss_after_cleanup_gc_mib",
    ]

    def stage_rss(result: dict[str, Any], stage: str) -> Any:
        return result["memory_stages"][stage]["rss_mib"]

    with (output_dir / "runtime-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({
                "framework": result["framework"],
                "runs": result["runs"],
                "construction_median_ms": result["construction"]["median_ms"],
                "construction_p95_ms": result["construction"]["p95_ms"],
                "construction_cpu_percent": result["construction"]["median_cpu_percent"],
                "completion_median_ms": result["completion"]["median_ms"],
                "completion_p95_ms": result["completion"]["p95_ms"],
                "completion_cpu_percent": result["completion"]["median_cpu_percent"],
                "stream_first_token_median_ms": result["streaming"]["median_first_token_ms"],
                "stream_first_token_p95_ms": result["streaming"]["p95_first_token_ms"],
                "stream_total_median_ms": result["streaming"]["median_ms"],
                "stream_total_p95_ms": result["streaming"]["p95_ms"],
                "stream_cpu_percent": result["streaming"]["median_cpu_percent"],
                "five_turn_median_ms": result["five_turn_loop"]["median_ms"],
                "five_turn_p95_ms": result["five_turn_loop"]["p95_ms"],
                "five_turn_cpu_percent": result["five_turn_loop"]["median_cpu_percent"],
                "five_turn_incremental_median_ms": result["five_turn_loop"]["incremental_turn_median_ms"],
                "twenty_turn_median_ms": result["twenty_turn_loop"]["median_ms"],
                "twenty_turn_p95_ms": result["twenty_turn_loop"]["p95_ms"],
                "twenty_turn_per_turn_ms": result["twenty_turn_loop"]["per_turn_ms"],
                "twenty_turn_incremental_median_ms": result["twenty_turn_loop"]["incremental_turn_median_ms"],
                "twenty_turn_incremental_p95_ms": result["twenty_turn_loop"]["incremental_turn_p95_ms"],
                "hundred_turn_median_ms": result["hundred_turn_loop"]["median_ms"],
                "hundred_turn_p95_ms": result["hundred_turn_loop"]["p95_ms"],
                "hundred_turn_per_turn_ms": result["hundred_turn_loop"]["per_turn_ms"],
                "hundred_turn_incremental_median_ms": result["hundred_turn_loop"]["incremental_turn_median_ms"],
                "hundred_turn_incremental_p95_ms": result["hundred_turn_loop"]["incremental_turn_p95_ms"],
                "warm_100_total_ms": result["warm_100_request_loop"]["median_ms"],
                "warm_100_per_request_ms": result["warm_100_request_loop"]["per_request_ms"],
                "concurrent_1_total_ms": result["concurrent_1_request_batch"]["median_ms"],
                "concurrent_1_p95_ms": result["concurrent_1_request_batch"]["p95_ms"],
                "concurrent_1_throughput_rps": result["concurrent_1_request_batch"]["throughput_rps"],
                "concurrent_4_total_ms": result["concurrent_4_request_batch"]["median_ms"],
                "concurrent_4_p95_ms": result["concurrent_4_request_batch"]["p95_ms"],
                "concurrent_4_throughput_rps": result["concurrent_4_request_batch"]["throughput_rps"],
                "concurrent_16_total_ms": result["concurrent_16_request_batch"]["median_ms"],
                "concurrent_16_p95_ms": result["concurrent_16_request_batch"]["p95_ms"],
                "concurrent_16_per_request_ms": result["concurrent_16_request_batch"]["per_request_ms"],
                "concurrent_16_throughput_rps": result["concurrent_16_request_batch"]["throughput_rps"],
                "baseline_rss_mib": result["baseline_rss_mib"],
                "final_rss_mib": result["final_rss_mib"],
                "rss_delta_mib": result["rss_delta_mib"],
                "module_load_median_ms": result["module_load"]["median_ms"],
                "module_load_p95_ms": result["module_load"]["p95_ms"],
                "module_load_cpu_percent": result["module_load"]["median_cpu_percent"],
                "first_construction_median_ms": result["first_construction"]["median_ms"],
                "first_construction_p95_ms": result["first_construction"]["p95_ms"],
                "first_construction_cpu_percent": result["first_construction"]["median_cpu_percent"],
                "subsequent_construction_median_ms": result["subsequent_construction"]["median_ms"],
                "subsequent_construction_p95_ms": result["subsequent_construction"]["p95_ms"],
                "subsequent_construction_cpu_percent": result["subsequent_construction"]["median_cpu_percent"],
                "first_completion_median_ms": result["first_completion"]["median_ms"],
                "first_completion_p95_ms": result["first_completion"]["p95_ms"],
                "first_completion_cpu_percent": result["first_completion"]["median_cpu_percent"],
                "first_stream_first_token_median_ms": result["first_streaming"]["median_first_token_ms"],
                "first_stream_first_token_p95_ms": result["first_streaming"]["p95_first_token_ms"],
                "first_stream_total_median_ms": result["first_streaming"]["median_ms"],
                "first_stream_total_p95_ms": result["first_streaming"]["p95_ms"],
                "first_stream_cpu_percent": result["first_streaming"]["median_cpu_percent"],
                "rss_after_framework_import_mib": stage_rss(result, "after_framework_import"),
                "rss_before_provider_construction_mib": stage_rss(result, "before_provider_construction"),
                "rss_after_provider_construction_mib": stage_rss(result, "after_provider_construction"),
                "rss_after_first_completion_mib": stage_rss(result, "after_first_completion"),
                "rss_after_warm_completion_series_mib": stage_rss(result, "after_warm_completion_series"),
                "rss_after_first_stream_mib": stage_rss(result, "after_first_stream"),
                "rss_after_warm_stream_series_mib": stage_rss(result, "after_warm_stream_series"),
                "rss_after_five_turn_loop_mib": stage_rss(result, "after_five_turn_loop"),
                "rss_after_twenty_turn_loop_mib": stage_rss(result, "after_twenty_turn_loop"),
                "rss_after_hundred_turn_loop_mib": stage_rss(result, "after_hundred_turn_loop"),
                "rss_after_cleanup_gc_mib": stage_rss(result, "after_cleanup_gc"),
            })

    with (output_dir / "runtime-samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "framework", "scenario", "run", "wall_ms", "cpu_ms",
            "first_token_ms", "turn", "conversation_run", "concurrency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for scenario, samples in result["samples"].items():
                for index, row in enumerate(samples, 1):
                    writer.writerow({
                        "framework": result["framework"],
                        "scenario": scenario,
                        "run": index,
                        "wall_ms": row["wall_ms"],
                        "cpu_ms": row["cpu_ms"],
                        "first_token_ms": row.get("first_token_ms"),
                        "turn": row.get("turn"),
                        "conversation_run": row.get("conversation_run"),
                        "concurrency": row.get("concurrency"),
                    })

    memory_fields = [
        "framework", "stage", "rss_mib", "tracemalloc_current_mib",
        "tracemalloc_peak_mib",
    ]
    with (output_dir / "runtime-memory-stages.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for stage, memory in result["memory_stages"].items():
                writer.writerow({
                    "framework": result["framework"],
                    "stage": stage,
                    "rss_mib": memory.get("rss_mib"),
                    "tracemalloc_current_mib": memory.get("tracemalloc_current_mib"),
                    "tracemalloc_peak_mib": memory.get("tracemalloc_peak_mib"),
                })


    diagnostic_rows = []
    for result in results:
        diagnostic = result.get("diagnostic_turn_loop")
        if diagnostic is None:
            continue
        memory = result["memory_stages"].get("after_diagnostic_turn_loop", {})
        diagnostic_rows.append({
            "framework": result["framework"],
            "turns": diagnostic["turns"],
            "median_ms": diagnostic["median_ms"],
            "p95_ms": diagnostic["p95_ms"],
            "per_turn_ms": diagnostic["per_turn_ms"],
            "incremental_turn_median_ms": diagnostic["incremental_turn_median_ms"],
            "incremental_turn_p95_ms": diagnostic["incremental_turn_p95_ms"],
            "rss_mib": memory.get("rss_mib"),
        })
    if diagnostic_rows:
        with (output_dir / "runtime-diagnostic-summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = [
                "framework", "turns", "median_ms", "p95_ms", "per_turn_ms",
                "incremental_turn_median_ms", "incremental_turn_p95_ms", "rss_mib",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(diagnostic_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable diagnostic cProfile + tracemalloc capture in each worker.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--framework", choices=FRAMEWORKS, help="Run one framework only.")
    parser.add_argument("--agent-transport", choices=("compatible", "sdk"), default="compatible")
    parser.add_argument(
        "--diagnostic-turns",
        type=int,
        help="Opt-in extra long-conversation diagnostic (for example 500).",
    )
    parser.add_argument("--worker", choices=FRAMEWORKS)
    parser.add_argument("--base-url")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.diagnostic_turns is not None and args.diagnostic_turns < 1:
        parser.error("--diagnostic-turns must be at least 1")

    if args.worker:
        if not args.base_url:
            parser.error("--base-url is required with --worker")
        return asyncio.run(
            worker(
                args.worker,
                args.base_url,
                args.runs,
                profile=args.profile,
                profile_dir=args.profile_dir,
                agent_transport=args.agent_transport,
                diagnostic_turns=args.diagnostic_turns,
            )
        )

    server, _, base_url = start_mock_server()
    websocket_server = start_responses_websocket_server()
    results: list[dict[str, Any]] = []
    profile_dir = (args.output_dir / "profiles").resolve()
    try:
        selected_frameworks = (args.framework,) if args.framework else FRAMEWORKS
        for framework in selected_frameworks:
            framework_base_url = (
                websocket_server.base_url
                if framework == "Agent RT"
                else base_url
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                framework,
                "--base-url",
                framework_base_url,
                "--runs",
                str(args.runs),
                "--agent-transport",
                args.agent_transport,
            ]
            if args.diagnostic_turns is not None:
                command.extend(["--diagnostic-turns", str(args.diagnostic_turns)])
            if args.profile:
                command.extend(["--profile", "--profile-dir", str(profile_dir)])
            proc = subprocess.run(
                command,
                cwd=HERE,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "LITELLM_LOG": "ERROR"},
            )
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            if proc.returncode != 0 or not lines:
                raise RuntimeError(
                    f"{framework} worker failed ({proc.returncode}): "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
            result = json.loads(lines[-1])
            results.append(result)
            print(
                f"{framework:20} first={result['first_completion']['median_ms']:.2f} ms "
                f"warm={result['completion']['median_ms']:.2f} ms "
                f"stream-first={result['streaming']['median_first_token_ms']:.2f} ms"
            )
    finally:
        websocket_server.shutdown()
        server.shutdown()
        server.server_close()

    write_outputs(args.output_dir, results)
    print(f"\nWrote runtime comparison to {args.output_dir}")
    if args.profile:
        print(f"Wrote diagnostic profiles to {profile_dir}")
    if args.runs >= 30 and args.framework is None and args.diagnostic_turns is None:
        published = HERE / "runtime-measured-results.csv"
        published.write_bytes((args.output_dir / "runtime-summary.csv").read_bytes())
        generate_charts()
        print(f"Published durable runtime aggregate to {published} and refreshed comparison charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

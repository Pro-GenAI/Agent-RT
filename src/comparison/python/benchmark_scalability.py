#!/usr/bin/env python3
"""Stress framework adapters with long-lived sequential and concurrent request loads."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import inspect
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from benchmark_runtime import FRAMEWORKS, current_rss_mib, load_adapter, text_messages
from mock_llm_api import RESPONSE_TEXT, start_mock_server, start_responses_websocket_server

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "scalability"
DEFAULT_MAX_CPU_CORES = 2
DEFAULT_MAX_MEMORY_MIB = 1024
RESOURCE_POLL_SECONDS = 0.05



def child_rss_mib(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return None
    return None


def worker_cpu_set(max_cpu_cores: int) -> set[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "CPU resource limiting requires Linux CPU affinity support; "
            "pass --no-resource-limits to run without safety limits"
        )
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("no CPUs are available to the benchmark process")
    return set(available[: min(max_cpu_cores, len(available))])


def run_limited_worker(
    command: list[str],
    *,
    env: dict[str, str],
    max_cpu_cores: int,
    max_memory_mib: int,
    resource_limits: bool,
) -> tuple[int, str, str]:
    cpu_set = worker_cpu_set(max_cpu_cores) if resource_limits else None
    if resource_limits and not Path("/proc/self/status").exists():
        raise RuntimeError(
            "RAM resource limiting requires Linux /proc; "
            "pass --no-resource-limits to run without safety limits"
        )

    def configure_child() -> None:
        if cpu_set is not None:
            os.sched_setaffinity(0, cpu_set)
            try:
                os.nice(5)
            except OSError:
                pass

    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file,
    ):
        proc = subprocess.Popen(
            command,
            cwd=HERE,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            start_new_session=True,
            preexec_fn=configure_child if resource_limits else None,
        )
        exceeded_memory = False
        observed_peak = 0.0
        while proc.poll() is None:
            if resource_limits:
                rss = child_rss_mib(proc.pid)
                if rss is not None:
                    observed_peak = max(observed_peak, rss)
                    if rss > max_memory_mib:
                        exceeded_memory = True
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        break
            time.sleep(RESOURCE_POLL_SECONDS)
        return_code = proc.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()

    if exceeded_memory:
        raise RuntimeError(
            f"worker exceeded the {max_memory_mib} MiB RAM limit "
            f"(observed {observed_peak:.1f} MiB) and was terminated"
        )
    return return_code, stdout, stderr


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def memory_snapshot() -> dict[str, float | None]:
    return {"rss_mib": current_rss_mib()}


async def sample_memory(
    phase: str,
    stop: asyncio.Event,
    interval_seconds: float,
    samples: list[dict[str, Any]],
    started: float,
) -> None:
    while True:
        samples.append({
            "phase": phase,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            **memory_snapshot(),
        })
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass


async def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def run_continuous(
    adapter: Any,
    client: Any,
    calls: int,
    sample_interval: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages = text_messages()
    latencies: list[float] = []
    memory_samples: list[dict[str, Any]] = []
    start_memory = current_rss_mib()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        sample_memory("continuous", stop, sample_interval, memory_samples, wall_started)
    )
    try:
        for _ in range(calls):
            started = time.perf_counter()
            text = await adapter.complete(client, messages)
            latencies.append((time.perf_counter() - started) * 1000)
            if text != RESPONSE_TEXT:
                raise RuntimeError(f"unexpected response text: {text!r}")
    finally:
        stop.set()
        await sampler
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    end_memory = current_rss_mib()
    gc.collect()
    post_gc_memory = current_rss_mib()
    observed = [row["rss_mib"] for row in memory_samples if row["rss_mib"] is not None]
    return ({
        "calls": calls,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_percent": (cpu_seconds / wall_seconds * 100) if wall_seconds else 0.0,
        "throughput_rps": calls / wall_seconds if wall_seconds else 0.0,
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "start_rss_mib": start_memory,
        "end_rss_mib": end_memory,
        "post_gc_rss_mib": post_gc_memory,
        "peak_sampled_rss_mib": max(observed) if observed else end_memory,
        "rss_growth_mib": (
            end_memory - start_memory
            if end_memory is not None and start_memory is not None
            else None
        ),
    }, memory_samples)


async def run_concurrent(
    adapter: Any,
    client: Any,
    calls: int,
    concurrency: int,
    sample_interval: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages = text_messages()
    latencies: list[float] = []
    memory_samples: list[dict[str, Any]] = []
    start_memory = current_rss_mib()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        sample_memory("concurrent", stop, sample_interval, memory_samples, wall_started)
    )

    next_call = 0

    async def call_worker() -> None:
        nonlocal next_call
        while True:
            index = next_call
            next_call += 1
            if index >= calls:
                return
            started = time.perf_counter()
            text = await adapter.complete(client, messages)
            latencies.append((time.perf_counter() - started) * 1000)
            if text != RESPONSE_TEXT:
                raise RuntimeError(f"unexpected response text: {text!r}")

    try:
        await asyncio.gather(
            *(call_worker() for _ in range(min(concurrency, calls)))
        )
    finally:
        stop.set()
        await sampler
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    end_memory = current_rss_mib()
    gc.collect()
    post_gc_memory = current_rss_mib()
    observed = [row["rss_mib"] for row in memory_samples if row["rss_mib"] is not None]
    return ({
        "calls": calls,
        "concurrency": concurrency,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_percent": (cpu_seconds / wall_seconds * 100) if wall_seconds else 0.0,
        "throughput_rps": calls / wall_seconds if wall_seconds else 0.0,
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "start_rss_mib": start_memory,
        "end_rss_mib": end_memory,
        "post_gc_rss_mib": post_gc_memory,
        "peak_sampled_rss_mib": max(observed) if observed else end_memory,
        "rss_growth_mib": (
            end_memory - start_memory
            if end_memory is not None and start_memory is not None
            else None
        ),
    }, memory_samples)


async def worker(
    framework: str,
    base_url: str,
    continuous_calls: int,
    concurrent_calls: int,
    concurrency: int,
    warmup: int,
    sample_interval_ms: int,
) -> int:
    adapter = load_adapter(framework, base_url)
    client = adapter.create()
    messages = text_messages()
    for _ in range(warmup):
        text = await adapter.complete(client, messages)
        if text != RESPONSE_TEXT:
            raise RuntimeError(f"unexpected warmup response text: {text!r}")

    baseline_rss = current_rss_mib()
    try:
        continuous, continuous_memory = await run_continuous(
            adapter,
            client,
            continuous_calls,
            sample_interval_ms / 1000,
        )
        concurrent, concurrent_memory = await run_concurrent(
            adapter,
            client,
            concurrent_calls,
            concurrency,
            sample_interval_ms / 1000,
        )
    finally:
        await close_client(client)

    result = {
        "framework": framework,
        "warmup_calls": warmup,
        "baseline_rss_mib": baseline_rss,
        "continuous": continuous,
        "concurrent": concurrent,
        "memory_samples": continuous_memory + concurrent_memory,
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scalability-results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "framework",
        "phase",
        "resource_limits_enabled",
        "max_cpu_cores",
        "max_memory_mib",
        "calls",
        "concurrency",
        "wall_seconds",
        "cpu_seconds",
        "cpu_percent",
        "throughput_rps",
        "median_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "start_rss_mib",
        "end_rss_mib",
        "post_gc_rss_mib",
        "peak_sampled_rss_mib",
        "rss_growth_mib",
    ]
    with (output_dir / "scalability-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for phase in ("continuous", "concurrent"):
                row = {
                    "framework": result["framework"],
                    "phase": phase,
                    **result.get("resource_limits", {}),
                    **result[phase],
                }
                writer.writerow({field: row.get(field) for field in fields})

    memory_fields = ["framework", "phase", "elapsed_ms", "rss_mib"]
    with (output_dir / "scalability-memory-samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for sample in result["memory_samples"]:
                writer.writerow({"framework": result["framework"], **sample})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-calls", type=int, default=5000)
    parser.add_argument("--concurrent-calls", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--max-cpu-cores", type=int, default=DEFAULT_MAX_CPU_CORES)
    parser.add_argument("--max-memory-mib", type=int, default=DEFAULT_MAX_MEMORY_MIB)
    parser.add_argument("--no-resource-limits", action="store_true")
    parser.add_argument("--framework", choices=FRAMEWORKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=FRAMEWORKS)
    parser.add_argument("--base-url")
    args = parser.parse_args()

    for name in (
        "continuous_calls",
        "concurrent_calls",
        "concurrency",
        "warmup",
        "sample_interval_ms",
        "max_cpu_cores",
        "max_memory_mib",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.concurrency > args.concurrent_calls:
        parser.error("--concurrency cannot exceed --concurrent-calls")

    if args.worker:
        if not args.base_url:
            parser.error("--base-url is required with --worker")
        return asyncio.run(
            worker(
                args.worker,
                args.base_url,
                args.continuous_calls,
                args.concurrent_calls,
                args.concurrency,
                args.warmup,
                args.sample_interval_ms,
            )
        )

    server, _, http_base_url = start_mock_server()
    websocket_server = start_responses_websocket_server()
    results: list[dict[str, Any]] = []
    try:
        frameworks = (args.framework,) if args.framework else FRAMEWORKS
        for framework in frameworks:
            base_url = (
                websocket_server.base_url
                if framework == "Agent RT"
                else http_base_url
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                framework,
                "--base-url",
                base_url,
                "--continuous-calls",
                str(args.continuous_calls),
                "--concurrent-calls",
                str(args.concurrent_calls),
                "--concurrency",
                str(args.concurrency),
                "--warmup",
                str(args.warmup),
                "--sample-interval-ms",
                str(args.sample_interval_ms),
            ]
            return_code, stdout, stderr = run_limited_worker(
                command,
                env={**os.environ, "LITELLM_LOG": "ERROR"},
                max_cpu_cores=args.max_cpu_cores,
                max_memory_mib=args.max_memory_mib,
                resource_limits=not args.no_resource_limits,
            )
            lines = [line for line in stdout.splitlines() if line.strip()]
            if return_code != 0 or not lines:
                raise RuntimeError(
                    f"{framework} worker failed ({return_code}): "
                    f"{stderr.strip() or stdout.strip()}"
                )
            result = json.loads(lines[-1])
            result["resource_limits"] = {
                "resource_limits_enabled": not args.no_resource_limits,
                "max_cpu_cores": args.max_cpu_cores if not args.no_resource_limits else None,
                "max_memory_mib": args.max_memory_mib if not args.no_resource_limits else None,
            }
            results.append(result)
            print(
                f"{framework:20} "
                f"continuous={result['continuous']['throughput_rps']:.1f} req/s "
                f"concurrent={result['concurrent']['throughput_rps']:.1f} req/s "
                f"peak={result['concurrent']['peak_sampled_rss_mib']:.1f} MiB"
            )
    finally:
        websocket_server.shutdown()
        server.shutdown()
        server.server_close()

    write_outputs(args.output_dir, results)
    print(f"\nWrote scalability diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

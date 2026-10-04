#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import pytest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("promote_benchmark_baseline.py")
SPEC = importlib.util.spec_from_file_location("promote_benchmark_baseline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
promote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote)

STAGES = (
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


def sample(value: float, *, first_token: float | None = None) -> dict[str, float]:
    row = {
        "wall_ms": float(value),
        "cpu_ms": float(value) / 2.0,
    }
    if first_token is not None:
        row["first_token_ms"] = float(first_token)
    return row


def runtime_part(
    framework: str,
    runs: int,
    values: list[float],
    *,
    cold: float,
    memory: float,
    baseline_rss: float,
    final_rss: float,
    rss_delta: float,
) -> dict[str, object]:
    assert len(values) == runs

    def rows(multiplier: float = 1.0) -> list[dict[str, float]]:
        return [sample(value * multiplier) for value in values]

    streaming = [sample(value * 1.2, first_token=value * 0.8) for value in values]
    return {
        "framework": framework,
        "runs": runs,
        "baseline_rss_mib": baseline_rss,
        "final_rss_mib": final_rss,
        "rss_delta_mib": rss_delta,
        "memory_stages": {stage: {"rss_mib": memory} for stage in STAGES},
        "samples": {
            "module_load": [sample(cold)],
            "first_construction": [sample(cold + 1)],
            "subsequent_construction": rows(0.01),
            "construction": rows(0.01),
            "first_completion": [sample(cold + 2)],
            "completion": rows(),
            "first_streaming": [sample(cold + 3, first_token=cold + 2.5)],
            "streaming": streaming,
            "five_turn_loop": rows(5),
            "five_turn_incremental": rows(),
            "twenty_turn_loop": rows(20),
            "twenty_turn_incremental": rows(),
            "hundred_turn_loop": rows(100),
            "hundred_turn_incremental": rows(),
            "warm_100_request_loop": [sample(cold + 100)],
            "concurrent_1_request_batch": rows(1),
            "concurrent_4_request_batch": rows(2),
            "concurrent_16_request_batch": rows(4),
        },
    }


class TestPromotion:
    def test_nearest_rank_percentile_matches_runner_semantics(self) -> None:
        assert promote.percentile(list(range(1, 31)), 0.95) == 29
        assert promote.percentile([1.0, 2.0, 3.0], 0.95) == 3.0

    def test_import_promotion_supports_typescript_framework_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "capture"
            source.mkdir()
            fields = [
                "framework",
                "specifier",
                "run",
                "ok",
                "import_ms",
                "process_total_ms",
                "cpu_ms",
                "cpu_percent",
                "rss_delta_mib",
                "max_observed_rss_mib",
            ]
            rows: list[dict[str, object]] = []
            for framework in promote.FRAMEWORKS["typescript"]:
                for run in range(1, 31):
                    rows.append(
                        {
                            "framework": framework,
                            "specifier": framework,
                            "run": run,
                            "ok": "true",
                            "import_ms": run,
                            "process_total_ms": run + 100,
                            "cpu_ms": run / 2,
                            "cpu_percent": 50,
                            "rss_delta_mib": run / 10,
                            "max_observed_rss_mib": run,
                        }
                    )
            with (source / "import-samples.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            destination = root / "promoted.csv"
            promote.promote_import(
                "typescript",
                [source],
                destination,
            )
            promoted = promote.read_csv(destination)
            assert len(promoted) == 3
            first = promoted[0]
            assert first["framework"] == "Agent RT"
            assert float(first["median_import_ms"]) == 15.5
            assert float(first["p95_import_ms"]) == 29.0
            assert int(first["successful_runs"]) == 30
            assert b"\r\n" not in destination.read_bytes()

    def test_runtime_chunks_pool_raw_samples_and_cold_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "capture"
            source.mkdir()

            results: list[dict[str, object]] = []
            # Deliberately uneven chunks: a median-of-medians implementation
            # would produce 57.5, while the pooled 30-sample median is 109.5.
            results.append(
                runtime_part(
                    "Agent RT",
                    5,
                    [1, 2, 3, 4, 5],
                    cold=10,
                    memory=50,
                    baseline_rss=20,
                    final_rss=25,
                    rss_delta=5,
                )
            )
            results.append(
                runtime_part(
                    "Agent RT",
                    25,
                    list(range(100, 125)),
                    cold=30,
                    memory=70,
                    baseline_rss=40,
                    final_rss=49,
                    rss_delta=9,
                )
            )
            for framework in ("LangChain", "LlamaIndex"):
                results.append(
                    runtime_part(
                        framework,
                        30,
                        [2] * 30,
                        cold=20,
                        memory=60,
                        baseline_rss=30,
                        final_rss=36,
                        rss_delta=6,
                    )
                )

            (source / "runtime-summary.json").write_text(
                json.dumps(results),
                encoding="utf-8",
            )
            destination = root / "runtime.csv"
            promote.promote_runtime(
                "python",
                [source],
                destination,
            )
            promoted = promote.read_csv(destination)
            ahl = promoted[0]
            assert ahl["framework"] == "Agent RT"
            assert float(ahl["completion_median_ms"]) == 109.5
            assert float(ahl["completion_p95_ms"]) == 123.0
            # Cold values are pooled across both chunk starts: median(12, 32).
            assert float(ahl["first_completion_median_ms"]) == 22.0
            assert float(ahl["module_load_median_ms"]) == 20.0
            # RSS is the cross-chunk median, not max/highest-growth chunk.
            assert float(ahl["baseline_rss_mib"]) == 30.0
            assert float(ahl["final_rss_mib"]) == 37.0
            assert float(ahl["rss_delta_mib"]) == 7.0
            assert float(ahl["rss_after_first_completion_mib"]) == 60.0

    def test_runtime_promotion_supports_typescript_framework_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "capture"
            source.mkdir()
            results = [
                runtime_part(
                    framework,
                    30,
                    [index + 1] * 30,
                    cold=10 + index,
                    memory=50 + index,
                    baseline_rss=20 + index,
                    final_rss=25 + index,
                    rss_delta=5,
                )
                for index, framework in enumerate(promote.FRAMEWORKS["typescript"])
            ]
            (source / "runtime-summary.json").write_text(
                json.dumps(results),
                encoding="utf-8",
            )
            destination = root / "runtime.csv"
            promote.promote_runtime(
                "typescript",
                [source],
                destination,
            )
            rows = promote.read_csv(destination)
            assert [row["framework"] for row in rows] == list(
                promote.FRAMEWORKS["typescript"]
            )

    def test_incomplete_import_promotion_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "capture"
            source.mkdir()
            fields = [
                "framework",
                "module",
                "run",
                "ok",
                "import_ms",
                "process_total_ms",
                "cpu_ms",
                "cpu_percent",
                "rss_delta_mib",
                "peak_rss_mib",
            ]
            with (source / "import-samples.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for run in range(1, 30):
                    writer.writerow(
                        {
                            "framework": "Agent RT",
                            "module": "agent_rt",
                            "run": run,
                            "ok": "true",
                            "import_ms": run,
                            "process_total_ms": run,
                            "cpu_ms": run,
                            "cpu_percent": 100,
                            "rss_delta_mib": 1,
                            "peak_rss_mib": 2,
                        }
                    )
            destination = root / "durable.csv"
            destination.write_text("sentinel\n", encoding="utf-8")
            with pytest.raises(ValueError):
                promote.promote_import("python", [source], destination)
            assert destination.read_text(encoding="utf-8") == "sentinel\n"

    def test_malformed_runtime_promotion_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "capture"
            source.mkdir()
            broken = runtime_part(
                "Agent RT",
                30,
                [1] * 30,
                cold=10,
                memory=50,
                baseline_rss=20,
                final_rss=25,
                rss_delta=5,
            )
            samples = broken["samples"]
            assert isinstance(samples, dict)
            samples["completion"] = []
            (source / "runtime-summary.json").write_text(
                json.dumps([broken]),
                encoding="utf-8",
            )
            destination = root / "durable.csv"
            destination.write_text("sentinel\n", encoding="utf-8")
            with pytest.raises(ValueError):
                promote.promote_runtime("python", [source], destination)
            assert destination.read_text(encoding="utf-8") == "sentinel\n"

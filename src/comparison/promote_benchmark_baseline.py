#!/usr/bin/env python3
"""Promote fresh filtered benchmark captures into a durable all-framework aggregate.

This exists for environments where one long all-framework command can exceed an
orchestration timeout. Filtered runners still never publish directly. Promotion
requires complete fresh data for every configured framework.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

FRAMEWORKS = {
    "python": ("Agent RT", "LangChain", "LlamaIndex"),
    "typescript": ("Agent RT", "LangChain.js", "LlamaIndex.TS"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def promote_import(language: str, sources: list[Path], destination: Path) -> None:
    expected = FRAMEWORKS[language]
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in expected}
    module_field = "module" if language == "python" else "specifier"
    peak_field = "peak_rss_mib" if language == "python" else "max_observed_rss_mib"
    summary_peak_field = (
        "max_peak_rss_mib" if language == "python" else "max_observed_rss_mib"
    )

    for source in sources:
        path = source / "import-samples.csv"
        if not path.exists():
            raise ValueError(f"missing import samples: {path}")
        for row in read_csv(path):
            framework = row.get("framework", "")
            if framework in grouped:
                grouped[framework].append(row)

    summary: list[dict[str, Any]] = []
    for framework in expected:
        rows = [row for row in grouped[framework] if row.get("ok", "").lower() == "true"]
        if len(rows) < 30:
            raise ValueError(
                f"{framework} has only {len(rows)} successful fresh import samples; need 30"
            )
        # Keep exactly the first 30 provided samples so promotion is deterministic.
        rows = rows[:30]
        imports = [float(row["import_ms"]) for row in rows]
        totals = [float(row["process_total_ms"]) for row in rows]
        cpus = [float(row["cpu_ms"]) for row in rows]
        cpu_percent = [float(row["cpu_percent"]) for row in rows if row.get("cpu_percent")]
        rss = [float(row["rss_delta_mib"]) for row in rows if row.get("rss_delta_mib")]
        peaks = [float(row[peak_field]) for row in rows if row.get(peak_field)]
        result: dict[str, Any] = {
            "framework": framework,
            module_field: rows[0][module_field],
            "successful_runs": len(rows),
            "failed_runs": 0,
            "median_import_ms": statistics.median(imports),
            "p95_import_ms": percentile(imports, 0.95),
            "median_process_total_ms": statistics.median(totals),
            "median_cpu_ms": statistics.median(cpus),
            "median_cpu_percent": statistics.median(cpu_percent) if cpu_percent else None,
            "median_rss_delta_mib": statistics.median(rss) if rss else None,
            summary_peak_field: max(peaks) if peaks else None,
            "error": "",
        }
        summary.append(result)

    fields = list(summary[0].keys())
    write_csv(destination, summary, fields)


def _runtime_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    walls = [float(row["wall_ms"]) for row in samples]
    cpus = [float(row["cpu_ms"]) for row in samples]
    cpu_percent = [
        cpu / wall * 100 if wall else 0.0
        for cpu, wall in zip(cpus, walls)
    ]
    return {
        "median_ms": statistics.median(walls),
        "p95_ms": percentile(walls, 0.95),
        "median_cpu_percent": statistics.median(cpu_percent),
    }


def _runtime_stream_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    summary = _runtime_summary(samples)
    first = [float(row["first_token_ms"]) for row in samples]
    summary["median_first_token_ms"] = statistics.median(first)
    summary["p95_first_token_ms"] = percentile(first, 0.95)
    return summary


def _load_runtime_results(source: Path) -> list[dict[str, Any]]:
    path = source / "runtime-summary.json"
    if not path.exists():
        raise ValueError(f"missing runtime summary JSON: {path}")
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"runtime summary JSON must be a list: {path}")
    return data


def promote_runtime(language: str, sources: list[Path], destination: Path) -> None:
    expected = FRAMEWORKS[language]
    captures: dict[str, list[dict[str, Any]]] = {name: [] for name in expected}
    for source in sources:
        for result in _load_runtime_results(source):
            framework = result.get("framework")
            if framework in captures:
                captures[framework].append(result)

    rows: list[dict[str, Any]] = []
    repeated = (
        "subsequent_construction",
        "completion",
        "streaming",
        "five_turn_loop",
        "five_turn_incremental",
        "twenty_turn_loop",
        "twenty_turn_incremental",
        "hundred_turn_loop",
        "hundred_turn_incremental",
        "warm_100_request_loop",
        "concurrent_1_request_batch",
        "concurrent_4_request_batch",
        "concurrent_16_request_batch",
    )

    for framework in expected:
        parts = captures[framework]
        if not parts:
            raise ValueError(f"missing fresh runtime captures for {framework}")
        total_runs = sum(int(part.get("runs", 0)) for part in parts)
        if total_runs < 30:
            raise ValueError(
                f"{framework} has only {total_runs} fresh runtime runs; need 30"
            )

        merged_samples: dict[str, list[dict[str, Any]]] = {name: [] for name in repeated}
        cold_samples: dict[str, list[dict[str, Any]]] = {
            "module_load": [],
            "first_construction": [],
            "first_completion": [],
            "first_streaming": [],
        }
        for part in parts:
            samples = part.get("samples", {})
            for name in repeated:
                merged_samples[name].extend(samples.get(name, []))
            for name in cold_samples:
                cold_samples[name].extend(samples.get(name, []))

        # Repeated scenarios must have at least 30 top-level samples. Incremental
        # scenarios naturally contain 5x/20x/100x more rows.
        for name in (
            "subsequent_construction", "completion", "streaming",
            "five_turn_loop", "twenty_turn_loop", "hundred_turn_loop",
            "concurrent_1_request_batch", "concurrent_4_request_batch",
            "concurrent_16_request_batch",
        ):
            if len(merged_samples[name]) < 30:
                raise ValueError(
                    f"{framework} scenario {name} has only "
                    f"{len(merged_samples[name])} samples; need 30"
                )

        construction = _runtime_summary(merged_samples["subsequent_construction"][:30])
        completion = _runtime_summary(merged_samples["completion"][:30])
        streaming = _runtime_stream_summary(merged_samples["streaming"][:30])
        five = _runtime_summary(merged_samples["five_turn_loop"][:30])
        five_inc = _runtime_summary(merged_samples["five_turn_incremental"])
        twenty = _runtime_summary(merged_samples["twenty_turn_loop"][:30])
        twenty_inc = _runtime_summary(merged_samples["twenty_turn_incremental"])
        hundred = _runtime_summary(merged_samples["hundred_turn_loop"][:30])
        hundred_inc = _runtime_summary(merged_samples["hundred_turn_incremental"])
        c1 = _runtime_summary(merged_samples["concurrent_1_request_batch"][:30])
        c4 = _runtime_summary(merged_samples["concurrent_4_request_batch"][:30])
        c16 = _runtime_summary(merged_samples["concurrent_16_request_batch"][:30])
        warm100 = _runtime_summary(merged_samples["warm_100_request_loop"])
        warm100["per_request_ms"] = warm100["median_ms"] / 100

        module_load = _runtime_summary(cold_samples["module_load"])
        first_construction = _runtime_summary(cold_samples["first_construction"])
        first_completion = _runtime_summary(cold_samples["first_completion"])
        first_stream = _runtime_stream_summary(cold_samples["first_streaming"])

        stage_names = tuple(parts[0]["memory_stages"])
        median_memory = {
            stage: statistics.median(
                float(part["memory_stages"][stage]["rss_mib"])
                for part in parts
            )
            for stage in stage_names
        }
        baseline_rss = statistics.median(float(part["baseline_rss_mib"]) for part in parts)
        final_rss = statistics.median(float(part["final_rss_mib"]) for part in parts)
        rss_delta = statistics.median(float(part["rss_delta_mib"]) for part in parts)

        c1["throughput_rps"] = 1000.0 / c1["median_ms"]
        c4["throughput_rps"] = 4000.0 / c4["median_ms"]
        c16["throughput_rps"] = 16000.0 / c16["median_ms"]

        def rss(stage: str) -> Any:
            return median_memory[stage]

        row: dict[str, Any] = {
            "framework": framework,
            "runs": 30,
            "construction_median_ms": construction["median_ms"],
            "construction_p95_ms": construction["p95_ms"],
            "construction_cpu_percent": construction["median_cpu_percent"],
            "completion_median_ms": completion["median_ms"],
            "completion_p95_ms": completion["p95_ms"],
            "completion_cpu_percent": completion["median_cpu_percent"],
            "stream_first_token_median_ms": streaming["median_first_token_ms"],
            "stream_first_token_p95_ms": streaming["p95_first_token_ms"],
            "stream_total_median_ms": streaming["median_ms"],
            "stream_total_p95_ms": streaming["p95_ms"],
            "stream_cpu_percent": streaming["median_cpu_percent"],
            "five_turn_median_ms": five["median_ms"],
            "five_turn_p95_ms": five["p95_ms"],
            "five_turn_cpu_percent": five["median_cpu_percent"],
            "five_turn_incremental_median_ms": five_inc["median_ms"],
            "twenty_turn_median_ms": twenty["median_ms"],
            "twenty_turn_p95_ms": twenty["p95_ms"],
            "twenty_turn_per_turn_ms": twenty["median_ms"] / 20,
            "twenty_turn_incremental_median_ms": twenty_inc["median_ms"],
            "twenty_turn_incremental_p95_ms": twenty_inc["p95_ms"],
            "hundred_turn_median_ms": hundred["median_ms"],
            "hundred_turn_p95_ms": hundred["p95_ms"],
            "hundred_turn_per_turn_ms": hundred["median_ms"] / 100,
            "hundred_turn_incremental_median_ms": hundred_inc["median_ms"],
            "hundred_turn_incremental_p95_ms": hundred_inc["p95_ms"],
            "warm_100_total_ms": warm100["median_ms"],
            "warm_100_per_request_ms": warm100["per_request_ms"],
            "concurrent_1_total_ms": c1["median_ms"],
            "concurrent_1_p95_ms": c1["p95_ms"],
            "concurrent_1_throughput_rps": c1["throughput_rps"],
            "concurrent_4_total_ms": c4["median_ms"],
            "concurrent_4_p95_ms": c4["p95_ms"],
            "concurrent_4_throughput_rps": c4["throughput_rps"],
            "concurrent_16_total_ms": c16["median_ms"],
            "concurrent_16_p95_ms": c16["p95_ms"],
            "concurrent_16_per_request_ms": c16["median_ms"] / 16,
            "concurrent_16_throughput_rps": c16["throughput_rps"],
            "baseline_rss_mib": baseline_rss,
            "final_rss_mib": final_rss,
            "rss_delta_mib": rss_delta,
        }
        for prefix, source_summary in (
            ("module_load", module_load),
            ("first_construction", first_construction),
            ("subsequent_construction", construction),
            ("first_completion", first_completion),
        ):
            row[prefix + "_median_ms"] = source_summary["median_ms"]
            row[prefix + "_p95_ms"] = source_summary["p95_ms"]
            row[prefix + "_cpu_percent"] = source_summary["median_cpu_percent"]
        row.update({
            "first_stream_first_token_median_ms": first_stream["median_first_token_ms"],
            "first_stream_first_token_p95_ms": first_stream["p95_first_token_ms"],
            "first_stream_total_median_ms": first_stream["median_ms"],
            "first_stream_total_p95_ms": first_stream["p95_ms"],
            "first_stream_cpu_percent": first_stream["median_cpu_percent"],
            "rss_after_framework_import_mib": rss("after_framework_import"),
            "rss_before_provider_construction_mib": rss("before_provider_construction"),
            "rss_after_provider_construction_mib": rss("after_provider_construction"),
            "rss_after_first_completion_mib": rss("after_first_completion"),
            "rss_after_warm_completion_series_mib": rss("after_warm_completion_series"),
            "rss_after_first_stream_mib": rss("after_first_stream"),
            "rss_after_warm_stream_series_mib": rss("after_warm_stream_series"),
            "rss_after_five_turn_loop_mib": rss("after_five_turn_loop"),
            "rss_after_twenty_turn_loop_mib": rss("after_twenty_turn_loop"),
            "rss_after_hundred_turn_loop_mib": rss("after_hundred_turn_loop"),
            "rss_after_cleanup_gc_mib": rss("after_cleanup_gc"),
        })
        rows.append(row)

    write_csv(destination, rows, list(rows[0].keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(FRAMEWORKS), required=True)
    parser.add_argument("--kind", choices=("import", "runtime"), required=True)
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        required=True,
        help="Filtered benchmark output directory. Repeat for multiple captures.",
    )
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()

    package_dir = ROOT / args.language
    destination = args.destination or package_dir / (
        "measured-results.csv" if args.kind == "import" else "runtime-measured-results.csv"
    )
    sources = [path.resolve() for path in args.source_dir]

    if args.kind == "import":
        promote_import(args.language, sources, destination)
    else:
        promote_runtime(args.language, sources, destination)

    print(f"Promoted {args.language} {args.kind} baseline to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

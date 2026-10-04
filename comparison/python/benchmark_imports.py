#!/usr/bin/env python3
"""Measure cold Python package import overhead in isolated subprocesses."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

import statistics
import subprocess
import sys
import time
from typing import Any

from generate_charts import generate_charts

ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python" / "src"
DEFAULT_PACKAGES = (
    ("Agent RT", "agent_rt"),
    ("LangChain", "langchain"),
    ("LlamaIndex", "llama_index.core"),
)


def current_rss_mib() -> float | None:
    """Return current RSS when available without adding benchmark dependencies."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def peak_rss_mib() -> float | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def worker(module_name: str) -> int:
    before_rss = current_rss_mib()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        # Touch public metadata so lazy module initialization is not completely skipped.
        getattr(module, "__name__", module_name)
        error = None
    except Exception as exc:  # benchmark output should record failures, not hide them
        error = f"{type(exc).__name__}: {exc}"
    import_ms = (time.perf_counter() - wall_start) * 1000
    cpu_ms = (time.process_time() - cpu_start) * 1000
    cpu_percent = (cpu_ms / import_ms * 100) if import_ms > 0 else None
    after_rss = current_rss_mib()
    rss_delta = (
        max(0.0, after_rss - before_rss)
        if before_rss is not None and after_rss is not None
        else None
    )
    payload = {
        "module": module_name,
        "ok": error is None,
        "import_ms": import_ms,
        "cpu_ms": cpu_ms,
        "cpu_percent": cpu_percent,
        "rss_before_mib": before_rss,
        "rss_after_mib": after_rss,
        "rss_delta_mib": rss_delta,
        "peak_rss_mib": peak_rss_mib(),
        "error": error,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if error is None else 2


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[rank]


def run_sample(label: str, module_name: str) -> dict[str, Any]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTHON_SRC) + (os.pathsep + existing if existing else "")
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", module_name],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    process_total_ms = (time.perf_counter() - start) * 1000
    line = proc.stdout.strip().splitlines()
    if not line:
        return {
            "framework": label,
            "module": module_name,
            "ok": False,
            "process_total_ms": process_total_ms,
            "error": proc.stderr.strip() or f"worker exited {proc.returncode}",
        }
    payload = json.loads(line[-1])
    payload.update(
        framework=label,
        process_total_ms=process_total_ms,
        worker_exit_code=proc.returncode,
    )
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="cold subprocesses per package")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for generated CSV/JSON files",
    )
    parser.add_argument("--framework", choices=tuple(label for label, _ in DEFAULT_PACKAGES), help="Run one framework only.")
    parser.add_argument("--worker", metavar="MODULE", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args.worker)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_packages = (
        tuple(item for item in DEFAULT_PACKAGES if item[0] == args.framework)
        if args.framework
        else DEFAULT_PACKAGES
    )
    samples: list[dict[str, Any]] = []
    for label, module_name in selected_packages:
        for run in range(1, args.runs + 1):
            row = run_sample(label, module_name)
            row["run"] = run
            samples.append(row)
            status = "ok" if row.get("ok") else "unavailable"
            print(f"{label:20} run {run:>2}/{args.runs}: {status}")

    sample_fields = [
        "framework", "module", "run", "ok", "import_ms", "process_total_ms",
        "cpu_ms", "cpu_percent", "rss_before_mib", "rss_after_mib", "rss_delta_mib",
        "peak_rss_mib", "worker_exit_code", "error",
    ]
    write_csv(args.output_dir / "import-samples.csv", samples, sample_fields)

    summary: list[dict[str, Any]] = []
    for label, module_name in selected_packages:
        rows = [r for r in samples if r["framework"] == label and r.get("ok")]
        failed = args.runs - len(rows)
        if not rows:
            summary.append({
                "framework": label,
                "module": module_name,
                "successful_runs": 0,
                "failed_runs": failed,
                "error": next((r.get("error") for r in samples if r["framework"] == label), None),
            })
            continue
        imports = [float(r["import_ms"]) for r in rows]
        totals = [float(r["process_total_ms"]) for r in rows]
        cpus = [float(r["cpu_ms"]) for r in rows]
        cpu_percents = [float(r["cpu_percent"]) for r in rows if r.get("cpu_percent") is not None]
        rss_deltas = [
            float(r["rss_delta_mib"]) for r in rows if r.get("rss_delta_mib") is not None
        ]
        peaks = [
            float(r["peak_rss_mib"]) for r in rows if r.get("peak_rss_mib") is not None
        ]
        summary.append({
            "framework": label,
            "module": module_name,
            "successful_runs": len(rows),
            "failed_runs": failed,
            "median_import_ms": statistics.median(imports),
            "p95_import_ms": percentile(imports, 0.95),
            "median_process_total_ms": statistics.median(totals),
            "median_cpu_ms": statistics.median(cpus),
            "median_cpu_percent": statistics.median(cpu_percents) if cpu_percents else None,
            "median_rss_delta_mib": statistics.median(rss_deltas) if rss_deltas else None,
            "max_peak_rss_mib": max(peaks) if peaks else None,
            "error": None,
        })

    summary_fields = [
        "framework", "module", "successful_runs", "failed_runs",
        "median_import_ms", "p95_import_ms", "median_process_total_ms",
        "median_cpu_ms", "median_cpu_percent", "median_rss_delta_mib",
        "max_peak_rss_mib", "error",
    ]
    write_csv(args.output_dir / "import-summary.csv", summary, summary_fields)
    (args.output_dir / "import-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nWrote results to {args.output_dir}")
    if args.runs >= 30 and args.framework is None:
        published = Path(__file__).resolve().parent / "measured-results.csv"
        published.write_bytes((args.output_dir / "import-summary.csv").read_bytes())
        generate_charts()
        print(f"Published durable import aggregate to {published} and refreshed comparison charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare current benchmark aggregates with an accepted performance baseline.

The checker is report-only by default. Use --mode strict to return a non-zero
exit code when a configured regression threshold is exceeded.

Baseline updates are intentionally explicit and require both import and runtime
aggregates to contain at least 30 successful Agent RT runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "performance-baselines.json"

FRAMEWORK = "Agent RT"

METRICS: tuple[dict[str, Any], ...] = (
    {
        "name": "cold_import_median_ms",
        "source": "import",
        "field": "median_import_ms",
        "relative": 0.15,
        "absolute": 15.0,
        "unit": "ms",
    },
    {
        "name": "first_construction_median_ms",
        "source": "runtime",
        "field": "first_construction_median_ms",
        "relative": 0.25,
        "absolute": 0.10,
        "unit": "ms",
    },
    {
        "name": "warm_construction_median_ms",
        "source": "runtime",
        "field": "subsequent_construction_median_ms",
        "relative": 0.30,
        "absolute": 0.05,
        "unit": "ms",
    },
    {
        "name": "first_completion_median_ms",
        "source": "runtime",
        "field": "first_completion_median_ms",
        "relative": 0.20,
        "absolute": 10.0,
        "unit": "ms",
    },
    {
        "name": "warm_completion_median_ms",
        "source": "runtime",
        "field": "completion_median_ms",
        "relative": 0.15,
        "absolute": 0.50,
        "unit": "ms",
    },
    {
        "name": "warm_completion_p95_ms",
        "source": "runtime",
        "field": "completion_p95_ms",
        "relative": 0.20,
        "absolute": 1.00,
        "unit": "ms",
    },
    {
        "name": "first_text_median_ms",
        "source": "runtime",
        "field": "stream_first_token_median_ms",
        "relative": 0.15,
        "absolute": 0.50,
        "unit": "ms",
    },
    {
        "name": "stream_total_median_ms",
        "source": "runtime",
        "field": "stream_total_median_ms",
        "relative": 0.15,
        "absolute": 0.50,
        "unit": "ms",
    },
    {
        "name": "stream_total_p95_ms",
        "source": "runtime",
        "field": "stream_total_p95_ms",
        "relative": 0.20,
        "absolute": 1.00,
        "unit": "ms",
    },
    {
        "name": "five_turn_median_ms",
        "source": "runtime",
        "field": "five_turn_median_ms",
        "relative": 0.15,
        "absolute": 2.00,
        "unit": "ms",
    },
    {
        "name": "five_turn_p95_ms",
        "source": "runtime",
        "field": "five_turn_p95_ms",
        "relative": 0.20,
        "absolute": 3.00,
        "unit": "ms",
    },
    {
        "name": "runtime_rss_delta_mib",
        "source": "runtime",
        "field": "rss_delta_mib",
        "relative": 0.15,
        "absolute": 3.00,
        "unit": "MiB",
    },
    {
        "name": "rss_after_first_completion_mib",
        "source": "runtime",
        "field": "rss_after_first_completion_mib",
        "relative": 0.10,
        "absolute": 4.00,
        "unit": "MiB",
    },
    {
        "name": "rss_after_five_turn_loop_mib",
        "source": "runtime",
        "field": "rss_after_five_turn_loop_mib",
        "relative": 0.10,
        "absolute": 4.00,
        "unit": "MiB",
    },
)


def read_framework_row(path: Path, framework: str = FRAMEWORK) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("framework") == framework:
                return row
    raise ValueError(f"{framework!r} not found in {path}")


def numeric(row: dict[str, str], field: str, path: Path) -> float:
    raw = row.get(field)
    if raw is None or raw == "":
        raise ValueError(f"{field!r} is missing from {path}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field!r} is not numeric in {path}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field!r} must be finite in {path}: {raw!r}")
    return value


def aggregate_paths(language: str, directory: Path) -> tuple[Path, Path]:
    return directory / "measured-results.csv", directory / "runtime-measured-results.csv"


def capture_values(import_row: dict[str, str], runtime_row: dict[str, str], import_path: Path, runtime_path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        row = import_row if metric["source"] == "import" else runtime_row
        path = import_path if metric["source"] == "import" else runtime_path
        result[metric["name"]] = numeric(row, metric["field"], path)
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "framework": FRAMEWORK, "languages": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("framework") != FRAMEWORK:
        raise ValueError(f"baseline framework must be {FRAMEWORK!r}")
    return data


def require_accepted_run_counts(import_row: dict[str, str], runtime_row: dict[str, str], import_path: Path, runtime_path: Path) -> tuple[int, int]:
    import_runs = int(numeric(import_row, "successful_runs", import_path))
    runtime_runs = int(numeric(runtime_row, "runs", runtime_path))
    if import_runs < 30 or runtime_runs < 30:
        raise ValueError(
            "baseline updates require at least 30 successful import runs and "
            f"30 runtime runs; got import={import_runs}, runtime={runtime_runs}"
        )
    failed = int(numeric(import_row, "failed_runs", import_path))
    if failed:
        raise ValueError(f"baseline update requires zero failed import runs; got {failed}")
    return import_runs, runtime_runs


def update_baseline(language: str, current_dir: Path, baseline_path: Path) -> int:
    import_path, runtime_path = aggregate_paths(language, current_dir)
    import_row = read_framework_row(import_path)
    runtime_row = read_framework_row(runtime_path)
    import_runs, runtime_runs = require_accepted_run_counts(
        import_row, runtime_row, import_path, runtime_path
    )
    manifest = load_manifest(baseline_path)
    manifest.setdefault("languages", {})[language] = {
        "import_runs": import_runs,
        "runtime_runs": runtime_runs,
        "values": capture_values(import_row, runtime_row, import_path, runtime_path),
    }
    baseline_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Updated {language} baseline from {import_runs} import runs and "
        f"{runtime_runs} runtime runs: {baseline_path}"
    )
    return 0


def check(language: str, current_dir: Path, baseline_path: Path, mode: str) -> int:
    manifest = load_manifest(baseline_path)
    try:
        baseline = manifest["languages"][language]
    except KeyError as exc:
        raise ValueError(
            f"no accepted {language!r} baseline in {baseline_path}; "
            "run with --update-baseline using 30+ run aggregates"
        ) from exc

    import_path, runtime_path = aggregate_paths(language, current_dir)
    import_row = read_framework_row(import_path)
    runtime_row = read_framework_row(runtime_path)
    current = capture_values(import_row, runtime_row, import_path, runtime_path)
    baseline_values = baseline["values"]

    failures = 0
    print(f"Performance regression check: {language} ({mode})")
    print(f"baseline={baseline_path}")
    print(f"current={current_dir}")
    for metric in METRICS:
        name = metric["name"]
        before = float(baseline_values[name])
        now = current[name]
        allowance = max(before * float(metric["relative"]), float(metric["absolute"]))
        limit = before + allowance
        delta = now - before
        pct = (delta / before * 100.0) if before else math.inf
        failed = now > limit
        failures += int(failed)
        status = "REGRESSION" if failed else "ok"
        print(
            f"{status:10} {name:38} "
            f"baseline={before:.4f} {metric['unit']} "
            f"current={now:.4f} {metric['unit']} "
            f"delta={delta:+.4f} ({pct:+.1f}%) "
            f"limit={limit:.4f}"
        )

    if failures:
        print(f"\n{failures} metric(s) exceeded configured tolerance.")
        return 1 if mode == "strict" else 0
    print("\nNo configured performance regressions detected.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("python", "typescript"), required=True)
    parser.add_argument(
        "--current-dir",
        type=Path,
        help="Directory containing measured-results.csv and runtime-measured-results.csv.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--mode", choices=("report", "strict"), default="report")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Replace the accepted language baseline from 30+ run aggregate CSVs.",
    )
    args = parser.parse_args()

    current_dir = (
        args.current_dir.resolve()
        if args.current_dir is not None
        else (HERE / args.language).resolve()
    )
    baseline_path = args.baseline.resolve()

    try:
        if args.update_baseline:
            return update_baseline(args.language, current_dir, baseline_path)
        return check(args.language, current_dir, baseline_path, args.mode)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"

REGIONS = (
    (1, 4095, "core+memory+provider"),
    (4096, 4978, "agent orchestration"),
    (4979, 6369, "protocol+interfaces+observability"),
    (6370, 7160, "routing+context+tool program"),
    (7161, 20000, "durability+workspace+sandbox"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, str(PYTHON_SRC))
    original = dataclasses.dataclass
    rows: list[dict[str, Any]] = []

    def wrapped(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        if decorator_args and isinstance(decorator_args[0], type):
            cls = decorator_args[0]
            started = time.perf_counter_ns()
            result = original(*decorator_args, **decorator_kwargs)
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            try:
                line = inspect.getsourcelines(cls)[1]
            except (OSError, TypeError):
                line = -1
            rows.append({"name": cls.__name__, "line": line, "ms": duration_ms})
            return result

        def decorate(cls: type[Any]) -> type[Any]:
            started = time.perf_counter_ns()
            result = original(**decorator_kwargs)(cls)
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            try:
                line = inspect.getsourcelines(cls)[1]
            except (OSError, TypeError):
                line = -1
            rows.append({"name": cls.__name__, "line": line, "ms": duration_ms})
            return result

        return decorate

    dataclasses.dataclass = wrapped
    started = time.perf_counter()
    try:
        __import__("agent_rt")
    finally:
        dataclasses.dataclass = original
    total_ms = (time.perf_counter() - started) * 1000

    region_rows: list[dict[str, Any]] = []
    for lo, hi, label in REGIONS:
        selected = [row for row in rows if lo <= row["line"] <= hi]
        region_rows.append({
            "region": label,
            "classes": len(selected),
            "dataclass_ms": sum(row["ms"] for row in selected),
        })

    payload = {
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "total_import_ms": total_ms,
        "dataclass_ms": sum(row["ms"] for row in rows),
        "dataclass_count": len(rows),
        "regions": region_rows,
        "top_classes": sorted(rows, key=lambda row: row["ms"], reverse=True)[: args.top],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"import={payload['total_import_ms']:.2f} ms "
        f"dataclasses={payload['dataclass_ms']:.2f} ms "
        f"count={payload['dataclass_count']}"
    )
    for row in region_rows:
        print(
            f"{row['region']:34s} "
            f"classes={row['classes']:3d} "
            f"dataclass_ms={row['dataclass_ms']:7.2f}"
        )
    print("top classes:")
    for row in payload["top_classes"]:
        print(f"{row['ms']:7.3f} ms line {row['line']:5d} {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

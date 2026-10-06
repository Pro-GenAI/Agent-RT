"""Summarize migration_results.csv: PASS/PASS counts per package and language.

A PASS/PASS row passed its test suite both before and after migration. The
"Baseline PASS" column counts rows whose suite passed before migration (the
rows where a migration result is meaningful), and "Rows" counts all rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).with_name("migration_results.csv")


def summarize(path: Path) -> list[tuple[str, str, int, int, int]]:
    csv.field_size_limit(sys.maxsize)
    rows: Counter[tuple[str, str]] = Counter()
    baseline: Counter[tuple[str, str]] = Counter()
    passed: Counter[tuple[str, str]] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["Package"], row["Language"])
            rows[key] += 1
            if row["TestStatusBeforeMigration"] == "PASS":
                baseline[key] += 1
                if row["TestStatusAfterMigration"] == "PASS":
                    passed[key] += 1
    return [(*key, passed[key], baseline[key], rows[key]) for key in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    summary = summarize(args.results)
    headers = ("Package", "Language", "PASS/PASS", "Baseline PASS", "Rows")
    totals = ("Total", "", *(sum(item[i] for item in summary) for i in (2, 3, 4)))
    table = [headers, *summary, totals]
    widths = [max(len(str(line[i])) for line in table) for i in range(len(headers))]

    def render(line: tuple) -> str:
        return "  ".join(
            str(value).ljust(width) if i < 2 else str(value).rjust(width)
            for i, (value, width) in enumerate(zip(line, widths))
        )

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for line in summary:
        print(render(line))
    print("  ".join("-" * width for width in widths))
    print(render(totals))


if __name__ == "__main__":
    main()

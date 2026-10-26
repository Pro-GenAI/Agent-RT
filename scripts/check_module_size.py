#!/usr/bin/env python3
"""Fail when a runtime module or public entrypoint grows past the line cap.

Covers python/src/ext/runtime/ and typescript/src/runtime/ (every .py/.ts
file) plus the two entrypoints, python/src/agent_rt.py and
typescript/src/index.ts. Dependency-free; run from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_LINES = 800


def checked_files() -> list[Path]:
    files = [ROOT / "python/src/agent_rt.py", ROOT / "typescript/src/index.ts"]
    files += sorted((ROOT / "python/src/ext/runtime").rglob("*.py"))
    files += sorted((ROOT / "typescript/src/runtime").rglob("*.ts"))
    return files


def main() -> int:
    failures = []
    for path in checked_files():
        rel = path.relative_to(ROOT).as_posix()
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        if count > MAX_LINES:
            failures.append(f"{rel}: {count} lines (limit {MAX_LINES})")
    if failures:
        print("Files over the line limit; split them into smaller modules:")
        print("\n".join(f"  {line}" for line in failures))
        return 1
    print(f"All runtime modules are within {MAX_LINES} lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

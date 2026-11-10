#!/usr/bin/env python3
"""Measure fresh-install Python framework footprint in isolated uv virtualenvs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT_DIR = HERE / "results" / "install-size"
DEFAULT_VENV_DIR = DEFAULT_OUTPUT_DIR / "venvs"
PUBLISHED_RESULTS = HERE / "install-size-measured-results.csv"


@dataclass(frozen=True)
class Framework:
    label: str
    slug: str
    packages: tuple[str, ...]


FRAMEWORKS = (
    Framework("Agent RT", "agent-rt", (str(ROOT / "python"),)),
    Framework(
        "LangChain",
        "langchain",
        ("langchain==1.4.2", "langchain-openai==1.6.6", "langchain-anthropic==1.7.5"),
    ),
    Framework(
        "LlamaIndex",
        "llamaindex",
        (
            "llama-index==0.14.25",
            "llama-index-llms-openai==0.7.10",
            "llama-index-llms-anthropic==0.12.2",
            "nltk @ git+https://github.com/nltk/nltk.git@d51c48a09765e9e3e97e7de36c2588c676d788d6",
        ),
    ),
)


def directory_size_bytes(path: Path) -> int:
    """Return logical bytes stored below *path*, without following symlinks."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        total += entry.stat().st_size
    return total


def python_in_venv(venv: Path) -> Path:
    windows = venv / "Scripts" / "python.exe"
    return windows if windows.exists() else venv / "bin" / "python"


def run_checked(command: list[str], *, cwd: Path = HERE) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def create_empty_venv(uv: str, venv: Path, python: str) -> int:
    if venv.exists():
        shutil.rmtree(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    run_checked([uv, "venv", "--python", python, str(venv)])
    return directory_size_bytes(venv)


def install_framework(uv: str, venv: Path, framework: Framework) -> None:
    run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python_in_venv(venv)),
            "--link-mode",
            "copy",
            *framework.packages,
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "framework",
        "packages",
        "empty_venv_bytes",
        "installed_venv_bytes",
        "installed_bytes",
        "installed_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.13", help="Python version/interpreter passed to uv venv")
    parser.add_argument("--uv", default="uv", help="uv executable (default: uv)")
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV_DIR,
        help="directory containing one isolated virtualenv per framework",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for install-size CSV/JSON output",
    )
    parser.add_argument(
        "--framework",
        choices=tuple(item.label for item in FRAMEWORKS),
        help="prepare/measure one framework only",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create fresh empty uv virtualenvs and stop before package installation",
    )
    parser.add_argument(
        "--keep-venvs",
        action="store_true",
        help="keep measured virtualenvs instead of deleting them after a completed run",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="refresh the checked-in all-framework install-size aggregate",
    )
    args = parser.parse_args()
    if args.publish and (args.prepare_only or args.framework):
        parser.error("--publish requires a complete install run with all frameworks")

    selected = tuple(item for item in FRAMEWORKS if not args.framework or item.label == args.framework)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.venv_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for framework in selected:
        venv = args.venv_dir / framework.slug
        print(f"{framework.label}: creating empty uv venv at {venv}")
        empty_bytes = create_empty_venv(args.uv, venv, args.python)
        if args.prepare_only:
            print(f"{framework.label}: empty venv {empty_bytes / (1024 * 1024):.2f} MiB")
            continue

        print(f"{framework.label}: installing {', '.join(framework.packages)}")
        install_framework(args.uv, venv, framework)
        installed_venv_bytes = directory_size_bytes(venv)
        installed_bytes = installed_venv_bytes - empty_bytes
        installed_mib = round(installed_bytes / (1024 * 1024), 2)
        row = {
            "framework": framework.label,
            "packages": ";".join(framework.packages).replace(str(ROOT / "python"), "../../python"),
            "empty_venv_bytes": empty_bytes,
            "installed_venv_bytes": installed_venv_bytes,
            "installed_bytes": installed_bytes,
            "installed_mib": installed_mib,
        }
        rows.append(row)
        print(f"{framework.label}: installed footprint {installed_mib:.2f} MiB")

    if args.prepare_only:
        return 0

    csv_path = args.output_dir / "install-size-summary.csv"
    json_path = args.output_dir / "install-size-summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if args.publish:
        write_csv(PUBLISHED_RESULTS, rows)
        print(f"Published {PUBLISHED_RESULTS}")

    if not args.keep_venvs:
        for framework in selected:
            shutil.rmtree(args.venv_dir / framework.slug, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

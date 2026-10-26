"""`agent-rt migrate`: rewrite framework imports to Agent RT, keeping the change only if the tests agree.

The project's tests run before and after the rewrite. The rewrite is kept when
every test that fails afterwards also failed before, in the same test and at the
same project source line; otherwise every rewritten file is restored.
Stdlib only: `uvx agent-rt migrate` must work without the CLI extras.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ext.artie_moods import pose

PYTHON_IMPORT_ROOTS = {
    "LangChain": (
        "langchain",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_chroma",
        "langchain_pinecone",
        "langchain_qdrant",
        "langchain_milvus",
        "langchain_weaviate",
    ),
    "LlamaIndex": ("llama_index",),
    "OpenAI": ("openai",),
    "Anthropic": ("anthropic",),
    "OpenAI Agents": ("agents",),
    "AutoGen": ("autogen_agentchat", "autogen_ext"),
    "CrewAI": ("crewai",),
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".agent_rt_venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
    ".cache",
    ".tox",
    ".nox",
    "__pycache__",
}

WALL_ISSUE_URL = "https://github.com/Pro-GenAI/Agent-RT/issues/new"


def rewrite_plain_import(match: re.Match[str]) -> str:
    prefix, module = match.group("prefix"), match.group("module")
    alias, rest = match.group("alias"), match.group("rest")
    if alias:
        return f"{prefix}agent_rt.{module}{alias}{rest}"
    head, _, tail = module.partition(".")
    if not tail:
        return f"{prefix}agent_rt.{module} as {module}{rest}"
    # `import pkg.sub` binds `pkg` with `sub` loaded on it.
    return f"{prefix}agent_rt.{module}; from agent_rt import {head}{rest}"


def rewrite_root_imports(text: str, root: str) -> tuple[str, int]:
    """Rewrite imports and mock/import-string targets of one top-level module to `agent_rt.<root>`."""
    changes = 0
    # from pkg.submodule import X -> from agent_rt.pkg.submodule import X
    from_pattern = re.compile(
        rf"(?m)^(?P<prefix>\s*from\s+)(?P<module>{re.escape(root)}(?:\.[A-Za-z_][\w.]*)?)(?=\s+import\b)"
    )
    text, count = from_pattern.subn(
        lambda match: f"{match.group('prefix')}agent_rt.{match.group('module')}",
        text,
    )
    changes += count

    # import pkg[.submodule] [as alias]. Without an alias, keep the name the
    # original statement bound (`import anthropic` binds `anthropic`, not
    # `agent_rt`), so later `anthropic.X` references still resolve.
    import_pattern = re.compile(
        rf"(?m)^(?P<prefix>\s*import\s+)(?P<module>{re.escape(root)}(?:\.[A-Za-z_][\w.]*)?)"
        r"(?P<alias>\s+as\s+\w+)?(?P<rest>\s*(?:#.*)?)$"
    )
    text, count = import_pattern.subn(rewrite_plain_import, text)
    changes += count

    # Mock targets name modules too: `patch("pkg.Client")`,
    # `monkeypatch.setattr("pkg.x", ...)`, and `sys.modules["pkg"]` stubs
    # must point at the migrated module or they stop intercepting it.
    # String imports (`pytest.importorskip("pkg")`, `import_module("pkg")`)
    # must too, or `pkg.RateLimitError` no longer matches what the
    # migrated code raises.
    target_pattern = re.compile(
        rf"(?P<quote>['\"])(?P<module>{re.escape(root)}(?:\.[A-Za-z_][\w.]*)?)(?P=quote)"
    )
    mock_line = re.compile(
        r"sys\.modules|\bpatch(?:\.object|\.dict)?\s*\(|monkeypatch\.(?:setattr|delattr|setitem|delitem)\s*\("
        r"|\bimportorskip\s*\(|\bimport_module\s*\(|\b__import__\s*\("
    )

    # A file that mocks often keeps its targets in a list or constant
    # ("openai.resources.chat.completions.Completions.create") away from
    # the patch() call, so there any dotted module path under the root is
    # a target. Domain-like strings ("openai.com") are left alone.
    dotted_target = re.compile(
        rf"(?P<quote>['\"])(?P<module>{re.escape(root)}\.(?!(?:com|org|net|io|ai|dev)\b)[A-Za-z_][\w.]*)(?P=quote)"
    )
    file_mocks = bool(mock_line.search(text))

    def rewrite_targets(line_match: re.Match[str]) -> str:
        nonlocal changes
        line = line_match.group(0)
        if not mock_line.search(line):
            if not file_mocks:
                return line
            line, count = dotted_target.subn(
                lambda match: (
                    f"{match.group('quote')}agent_rt.{match.group('module')}{match.group('quote')}"
                ),
                line,
            )
            changes += count
            return line
        line, count = target_pattern.subn(
            lambda match: (
                f"{match.group('quote')}agent_rt.{match.group('module')}{match.group('quote')}"
            ),
            line,
        )
        changes += count
        return line

    text = re.sub(r"(?m)^.*$", rewrite_targets, text)
    return text, changes


def rewrite_python_imports(text: str, package: str) -> tuple[str, int]:
    changes = 0
    for root in sorted(PYTHON_IMPORT_ROOTS.get(package, ()), key=len, reverse=True):
        text, count = rewrite_root_imports(text, root)
        changes += count
    return text, changes


def iter_python_files(root: Path) -> Iterable[Path]:
    for folder, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP_DIRS and not name.endswith(".egg-info")
        ]
        for name in files:
            if name.endswith(".py"):
                yield Path(folder) / name


def migration_roots(project: Path) -> list[str]:
    """Every supported root, longest first, minus names the project defines itself (a local `agents/`)."""
    local = {
        path.stem if path.name != "__init__.py" else path.parent.name
        for path in iter_python_files(project)
    }
    roots = {root for roots in PYTHON_IMPORT_ROOTS.values() for root in roots} - local
    return sorted(roots, key=len, reverse=True)


@dataclass
class Rewrite:
    originals: dict[Path, str]
    migrated: dict[Path, str]
    changes: int
    roots: set[str]


def plan_rewrite(project: Path) -> Rewrite:
    """Compute the rewrite in memory; nothing is written yet."""
    plan = Rewrite({}, {}, 0, set())
    roots = migration_roots(project)
    for path in iter_python_files(project):
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = original
        for root in roots:
            text, count = rewrite_root_imports(text, root)
            if count:
                plan.changes += count
                plan.roots.add(root)
        if text != original:
            plan.originals[path] = original
            plan.migrated[path] = text
    return plan


@dataclass
class SuiteRun:
    exit_code: int
    seconds: float
    peak_rss: int | None  # bytes; None where the OS does not report it
    total: int
    # test id -> innermost project source location of the failure ("tests/test_x.py:12"),
    # or None when the report was missing/unreadable.
    failures: dict[str, str | None] | None
    output: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


_PY_LOCATION = re.compile(r"(?m)^(?P<path>[^\s:][^:\n]*\.py):(?P<line>\d+):")


def failure_location(text: str, project: Path) -> str | None:
    """The innermost traceback frame inside the project (pytest prints innermost last)."""
    found = None
    for match in _PY_LOCATION.finditer(text):
        path = match.group("path")
        if "site-packages" in path or "dist-packages" in path:
            continue
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(project.resolve())
            except ValueError:
                continue
        if candidate.parts and candidate.parts[0] in SKIP_DIRS:
            continue
        found = f"{candidate.as_posix()}:{match.group('line')}"
    return found


def parse_junit(
    path: Path, project: Path, locate=failure_location
) -> tuple[int, dict[str, str | None]] | None:
    try:
        data = path.read_bytes()
        # The report comes from the migrated project's tests; JUnit XML never
        # declares a DTD, so refusing one blocks entity-expansion bombs.
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            return None
        root = ET.fromstring(data)  # nosec B314
    except (OSError, ET.ParseError):
        return None
    total, failures = 0, {}
    for case in root.iter("testcase"):
        total += 1
        problem = next(
            (child for child in case if child.tag in ("failure", "error")), None
        )
        if problem is None:
            continue
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        failures[test_id] = locate(
            f"{problem.text or ''}\n{problem.get('message', '')}", project
        )
    return total, failures


def run_tests(command: Sequence[str], project: Path, report: Path) -> SuiteRun:
    report.unlink(missing_ok=True)
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as log:
        started = time.perf_counter()
        process = subprocess.Popen(
            command, cwd=project, stdout=log, stderr=subprocess.STDOUT
        )
        peak_rss = None
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            # Linux reports KiB, macOS bytes.
            peak_rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        else:
            process.wait()
        seconds = time.perf_counter() - started
        log.seek(0)
        output = log.read()
    parsed = parse_junit(report, project)
    total, failures = parsed if parsed else (0, None)
    return SuiteRun(process.returncode, seconds, peak_rss, total, failures, output)


def compare(before: SuiteRun, after: SuiteRun) -> tuple[bool, list[str]]:
    """Keep the migration only if nothing fails afterwards that did not fail the same way before."""
    if after.passed:
        return True, []
    if after.failures is None or before.failures is None:
        return False, [
            "the test report could not be read, so the failures cannot be compared"
        ]
    new = [
        f"{test_id} ({location or 'unknown location'})"
        for test_id, location in after.failures.items()
        if test_id not in before.failures or before.failures[test_id] != location
    ]
    if not new and after.exit_code != before.exit_code:
        new.append(
            f"test run exit code changed from {before.exit_code} to {after.exit_code}"
        )
    return not new, new


def project_python(project: Path, explicit: str | None) -> str | None:
    if explicit:
        # Tests run with the project as cwd; abspath, not resolve(), which would leave the venv.
        return (
            os.path.abspath(explicit)
            if any(sep in explicit for sep in "/\\")
            else explicit
        )
    bin_dir, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    candidates = (
        [Path(os.environ["VIRTUAL_ENV"]) / bin_dir / exe]
        if os.environ.get("VIRTUAL_ENV")
        else []
    )
    candidates += [project / name / bin_dir / exe for name in (".venv", "venv")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("python3") or shutil.which("python")


_SIZES_SCRIPT = """
import importlib.metadata as m, json, sys
names = m.packages_distributions()
def size(name):
    try:
        return sum(f.size or 0 for f in m.distribution(name).files or ())
    except m.PackageNotFoundError:
        return 0
upstream = {d for root in sys.argv[1:] for d in names.get(root, ())} - {"agent-rt", "agent_rt"}
print(json.dumps({"upstream": sum(map(size, upstream)), "agent_rt": size("agent-rt")}))
"""


def package_sizes(python: str, roots: Iterable[str]) -> tuple[int, int] | None:
    # shortcut: counts the directly imported distributions only, not their dependency trees;
    # upgrade to a resolved dependency closure if the card needs full install size.
    try:
        result = subprocess.run(
            [python, "-c", _SIZES_SCRIPT, *sorted(roots)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        sizes = json.loads(result.stdout)
        return sizes["upstream"], sizes["agent_rt"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def _mib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1048576:.1f} MiB"


def _change(before: float | None, after: float | None) -> str:
    if not before or after is None:
        return ""
    return f"  ({(after - before) / before:+.0%})"


def card(
    *,
    kept: bool,
    plan: Rewrite,
    before: SuiteRun,
    after: SuiteRun,
    sizes: tuple[int, int] | None,
) -> str:
    def summary(run: SuiteRun) -> str:
        failed = len(run.failures or {})
        return f"{run.total - failed}/{run.total} passed"

    rows = [
        ("Result", "MIGRATED" if kept else "REVERTED"),
        (
            "Rewrote",
            f"{plan.changes} imports in {len(plan.migrated)} files ({', '.join(sorted(plan.roots))})",
        ),
        ("Tests", f"{summary(before)} -> {summary(after)}"),
        (
            "Test time",
            f"{before.seconds:.1f} s -> {after.seconds:.1f} s{_change(before.seconds, after.seconds)}",
        ),
        (
            "Peak memory",
            f"{_mib(before.peak_rss)} -> {_mib(after.peak_rss)}{_change(before.peak_rss, after.peak_rss)}",
        ),
    ]
    if sizes:
        rows.append(
            ("Package size", f"{_mib(sizes[0])} -> {_mib(sizes[1])}{_change(*sizes)}")
        )
    width = max(len(label) for label, _ in rows)
    lines = [f"  {label.ljust(width)}  {value}" for label, value in rows]
    rule = "-" * max(len(line) for line in lines)
    return "\n".join(["Agent RT migration", rule, *lines, rule])


def wall_link(project: Path, plan: Rewrite) -> str | None:
    """A pre-filled issue for the user to submit; nothing is sent automatically."""
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.fullmatch(
        r"(?:https://|git@)github\.com[/:]([\w.-]+/[\w.-]+?)(?:\.git)?/?", remote
    )
    if not match:
        return None
    packages = sorted(
        name for name, roots in PYTHON_IMPORT_ROOTS.items() if plan.roots & set(roots)
    )
    query = urllib.parse.urlencode(
        {
            "title": f"Migration wall: {match.group(1)}",
            "labels": "migration-wall",
            "body": f"Repository: https://github.com/{match.group(1)}\nCommit: {commit}\n"
            f"Package: {', '.join(packages)}\nLanguage: Python\nResult: PASS (agent-rt migrate)\n",
        }
    )
    return f"{WALL_ISSUE_URL}?{query}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-rt migrate",
        description=(
            "Rewrite LangChain, LlamaIndex, OpenAI, Anthropic, OpenAI Agents, AutoGen and CrewAI imports "
            "to Agent RT. Runs pytest before and after, and restores every file unless no test fails "
            "afterwards that did not fail at the same test and line before."
        ),
    )
    parser.add_argument(
        "path", nargs="?", default=".", type=Path, help="project folder (default: .)"
    )
    parser.add_argument(
        "--python",
        help="interpreter of the project's environment (default: active venv, ./.venv, python3)",
    )
    parser.epilog = (
        "Extra pytest arguments go after --, for example: agent-rt migrate . -- -k unit"
    )
    return parser


def main(argv: Sequence[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    args = build_parser().parse_args(argv[:split])
    pytest_args = argv[split + 1 :]
    project = args.path.resolve()
    if not project.is_dir():
        print(f"agent-rt migrate: not a folder: {project}", file=sys.stderr)
        return 2
    plan = plan_rewrite(project)
    if not plan.migrated:
        print(
            "agent-rt migrate: no supported framework imports found; nothing to do.",
            file=out,
        )
        return 0
    python = project_python(project, args.python)
    if python is None:
        print(
            "agent-rt migrate: no Python interpreter found; pass --python.",
            file=sys.stderr,
        )
        return 2
    for module, hint in (("pytest", "pytest"), ("agent_rt", "agent-rt")):
        probe = subprocess.run(
            [python, "-c", f"import {module}"], capture_output=True, check=False
        )
        if probe.returncode:
            print(
                f"agent-rt migrate: {module} is not installed in {python}. "
                f"Install it in the project environment (uv add --dev {hint}, or pip install {hint}) and rerun.",
                file=sys.stderr,
            )
            return 2

    with tempfile.TemporaryDirectory(prefix="agent-rt-migrate-") as scratch:
        report = Path(scratch) / "junit.xml"
        command = [
            python,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            *pytest_args,
        ]

        print("Running tests before migration...", file=out, flush=True)
        before = run_tests(command, project, report)
        if before.exit_code == 5 or (before.failures is not None and before.total == 0):
            print(
                "agent-rt migrate: no tests found, so the migration cannot be verified; nothing changed.",
                file=out,
            )
            return 1
        if before.failures is None:
            print(
                f"{before.output[-4000:]}\nagent-rt migrate: pytest did not produce a report; nothing changed.",
                file=out,
            )
            return 2

        kept = False
        try:
            for path, text in plan.migrated.items():
                path.write_text(text, encoding="utf-8")
            print(
                f"Rewrote {plan.changes} imports in {len(plan.migrated)} files; running tests again...",
                file=out,
                flush=True,
            )
            after = run_tests(command, project, report)
            kept, problems = compare(before, after)
        finally:
            # Also on Ctrl-C or a crash mid-run: never leave a half-verified rewrite.
            if not kept:
                for path, text in plan.originals.items():
                    path.write_text(text, encoding="utf-8")

    print(file=out)
    print(
        card(
            kept=kept,
            plan=plan,
            before=before,
            after=after,
            sizes=package_sizes(python, plan.roots),
        ),
        file=out,
    )
    if not kept:
        print("\nNew failures after migration (all files restored):", file=out)
        for problem in problems[:20]:
            print(f"  - {problem}", file=out)
        print("\nLast test output:\n" + after.output[-4000:], file=out)
        return 1
    print(pose("celebrate", "Migration passed! Your tests agree."), file=out)
    for path in sorted(plan.migrated):
        print(f"  changed {path.relative_to(project).as_posix()}", file=out)
    link = wall_link(project, plan)
    if link:
        print(
            f"\nAdd this repository to the migration wall (opens a pre-filled GitHub issue):\n  {link}",
            file=out,
        )
    return 0


__all__ = (
    "PYTHON_IMPORT_ROOTS",
    "compare",
    "main",
    "plan_rewrite",
    "rewrite_python_imports",
)

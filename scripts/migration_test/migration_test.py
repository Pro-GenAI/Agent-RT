#!/usr/bin/env python3
"""Run disposable third-party migration tests from repo_list.csv.

The runner clones one repository at a time, installs its dependencies, runs its
tests, rewrites only the target framework import/package specifier, installs the
local Agent RT package, runs the same tests again, records the result, and
deletes the checkout. Third-party code is untrusted: use this only in a
disposable sandbox/VM with no real credentials.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from heavy_package_audit import find_heavy_dependencies, read_manifest_text


RESULT_FIELDS = [
    "Index",
    "Package",
    "Language",
    "PLIndex",
    "RepoURL",
    "CommitHash",
    "TestStatusBeforeMigration",
    "TestStatusAfterMigration",
    "TestResultsBeforeMigration",
    "TestResultsAfterMigration",
]

SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"}
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
}

# Curated import-name -> PyPI distribution map for dependencies that projects
# commonly import without declaring (often behind try/except or only in tests).
# Only names listed here are ever auto-installed: deriving distribution names
# from arbitrary import names would invite typosquatted packages. Keep this
# list free of heavyweight ML stacks (see heavy_package_audit.py).
COMMON_IMPORT_DEPENDENCIES = {
    "aio_pika": "aio-pika",
    "aiofiles": "aiofiles",
    "aiohttp": "aiohttp",
    "aiosqlite": "aiosqlite",
    "anthropic": "anthropic",
    "bs4": "beautifulsoup4",
    "click": "click",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "fitz": "pymupdf",
    "flask": "flask",
    "freezegun": "freezegun",
    # Dotted keys cover namespace packages whose top-level name is shared.
    "google.genai": "google-genai",
    "google.generativeai": "google-generativeai",
    "groq": "groq",
    "httpx": "httpx",
    "jinja2": "jinja2",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_community": "langchain-community",
    "langchain_core": "langchain-core",
    "langchain_google_genai": "langchain-google-genai",
    "langchain_groq": "langchain-groq",
    "langchain_ollama": "langchain-ollama",
    "langchain_openai": "langchain-openai",
    "langchain_text_splitters": "langchain-text-splitters",
    "langgraph": "langgraph",
    "llama_index": "llama-index",
    "llama_index.core": "llama-index-core",
    "email_validator": "email-validator",
    "networkx": "networkx",
    # pydantic's EmailStr needs the email extra at model-definition time.
    "pydantic.EmailStr": "email-validator",
    # Python 3.12 venvs no longer seed setuptools, and setuptools 81 drops
    # pkg_resources, which older libraries (and pinecone-client) import.
    "pkg_resources": "setuptools<81",
    "markdown": "markdown",
    "nest_asyncio": "nest-asyncio",
    "ollama": "ollama",
    "openai": "openai",
    "pandas": "pandas",
    "PIL": "pillow",
    "psycopg2": "psycopg2-binary",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "pymysql": "pymysql",
    "pypdf": "pypdf",
    "PyPDF2": "pypdf2",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_mock": "pytest-mock",
    "prometheus_client": "prometheus-client",
    "prometheus_fastapi_instrumentator": "prometheus-fastapi-instrumentator",
    "pytz": "pytz",
    "redis": "redis",
    "requests": "requests",
    "respx": "respx",
    "rich": "rich",
    "sqlalchemy": "sqlalchemy",
    "starlette": "starlette",
    "tenacity": "tenacity",
    "tiktoken": "tiktoken",
    "toml": "toml",
    "typer": "typer",
    "uvicorn": "uvicorn",
    "websockets": "websockets",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
}

# Pytest plugins implied by configuration or test sources: (pattern, plugin).
# Projects often rely on a plugin through pytest.ini addopts or markers
# without declaring it, and pytest then aborts on "unrecognized arguments"
# or reports "async def functions are not natively supported".
PYTEST_PLUGIN_HINTS = (
    (re.compile(r"--ds[= ]|DJANGO_SETTINGS_MODULE|--reuse-db|--create-db"), "pytest-django"),
    (re.compile(r"--cov\b|--cov="), "pytest-cov"),
    (re.compile(r"(?:^|\s)-n\s*(?:auto|\d)|--numprocesses|--dist[= ]"), "pytest-xdist"),
    (re.compile(r"(?m)^\s*timeout\s*=|--timeout\b|pytest\.mark\.timeout"), "pytest-timeout"),
    (re.compile(r"asyncio_mode|pytest\.mark\.asyncio|pytest_asyncio"), "pytest-asyncio"),
    (re.compile(r"--reruns\b|pytest\.mark\.flaky"), "pytest-rerunfailures"),
    (re.compile(r"--html[= ]"), "pytest-html"),
    (re.compile(r"--benchmark"), "pytest-benchmark"),
    (re.compile(r"(?m)pytest_env|^\s*env\s*=\s*$"), "pytest-env"),
)
# Fixture-providing plugins are installed only after a run reports the
# fixture missing: suites often importorskip them, and installing them up
# front un-skips tests that a passing baseline never ran.
PYTEST_FIXTURE_PLUGINS = {
    "mocker": "pytest-mock",
    "httpx_mock": "pytest-httpx",
    "freezer": "pytest-freezer",
    "benchmark": "pytest-benchmark",
    "requests_mock": "requests-mock",
    "respx_mock": "respx",
    "snapshot": "syrupy",
}
MISSING_MODULE_ERROR = re.compile(r"No module named '([\w.]+)'")
MISSING_NAMESPACE_MEMBER = re.compile(r"cannot import name '(\w+)' from '([\w.]+)' \(unknown location\)")
MISSING_FIXTURE_ERROR = re.compile(r"fixture '(\w+)' not found")
# "pip install httpx2" in an error message (starlette's TestClient names its
# newer HTTP client this way). Only distributions in HINTED_PACKAGES are
# installed; an arbitrary hinted name is never trusted.
INSTALL_HINT_PACKAGE = re.compile(r"(?:requires the|pip install) ([A-Za-z0-9][A-Za-z0-9._-]*)")
HINTED_PACKAGES = frozenset({"httpx2"})
ASYNC_TESTS_UNSUPPORTED = re.compile(r"async def functions are not natively supported")
PYTEST_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py")

# Baseline failure signatures that a pytest option works around.
PYTEST_DUPLICATE_PLUGIN = re.compile(r"Plugin already registered under a different name: ([\w.-]+)=")
PYTEST_CLOSED_CAPTURE = re.compile(r"_pytest/capture\.py.*?ValueError: I/O operation on closed file", re.S)

# Upstream renames where the old distribution now only raises on import:
# (failure output pattern, distribution to remove, distribution to install).
PACKAGE_RENAMES = (
    (re.compile(r"renamed from `pinecone-client` to `pinecone`"), "pinecone-client", "pinecone"),
)

# Modules removed by a newer major release that unpinned requirements now
# resolve to: (failure output pattern, requirement restoring the module).
LEGACY_MODULE_PINS = (
    # langfuse 3 dropped the langfuse.decorators module (@observe moved to langfuse).
    (re.compile(r"No module named 'langfuse\.decorators'"), "langfuse<3"),
    # setuptools 81+ no longer ships pkg_resources, which older SDKs import.
    (re.compile(r"No module named 'pkg_resources'"), "setuptools<81"),
)

# Requirements that cannot build from source without system headers; the
# binary distribution is a drop-in replacement for test purposes.
SOURCE_ONLY_REPLACEMENTS = {"psycopg2": "psycopg2-binary"}

# Pytest exits with 5 when it collects no tests at all.
PYTEST_NO_TESTS_EXIT_CODE = 5

# Messages showing a repository was written against the pre-1.0 LangChain
# package layout (langchain.vectorstores, langchain.llms, ...), which unpinned
# requirements now resolve to LangChain 1.x without those shims.
# Pre-0.10 LlamaIndex imports (llama_index.VectorStoreIndex, llama_index.llms
# .OpenAI) fail against the namespace-package layout with "unknown location".
LEGACY_LLAMA_INDEX_ERROR = re.compile(
    r"cannot import name '\w+' from '(?:agent_rt\.)?llama_index(?:\.[\w.]+)?' \(unknown location\)"
    r"|No module named '(?:agent_rt\.)?llama_index\.(?:indices|langchain_helpers|node_parser\.simple|vector_stores\.types|service_context)'"
)

LEGACY_LANGCHAIN_ERROR = re.compile(
    r"No module named '(?:agent_rt\.)?langchain\.[\w.]+'"
    r"|cannot import name '\w+' from '(?:agent_rt\.)?langchain(?:\.[\w.]+)?'"
)

UNRESOLVABLE_REQUIREMENT_PATTERNS = (
    # uv: "Because foo was not found in the package registry ..."
    re.compile(r"Because\s+([A-Za-z0-9][A-Za-z0-9._-]*)\s+was\s+not\s+found\s+in\s+the\s+package\s+registry"),
    # uv: "Because there is no version of foo==1.2.3 ..."
    re.compile(r"there\s+is\s+no\s+version\s+of\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*[=<>!~]"),
    # uv: "... we can conclude that foo==1 and bar==2 are incompatible."
    re.compile(r"conclude\s+that\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==\S+\s+and\s+(?:[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==\S+\s+are\s+incompatible"),
    re.compile(r"conclude\s+that\s+(?:[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==\S+\s+and\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==\S+\s+are\s+incompatible"),
)

# Environment reads that fail when the variable is unset: os.environ["X"]
# (not assignments), django-environ / python-decouple calls without a default
# (env("X"), env.db("X"), config("X")).
REQUIRED_ENV_READ = re.compile(
    r"""environ\[\s*['"]([A-Z][A-Z0-9_]{2,})['"]\s*\](?!\s*=[^=])"""
    r"""|\b(?:env(?:\.\w+)?|config)\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]\s*\)"""
)
# Optional reads (getenv, environ.get, process.env) count as required only
# when the same file raises/throws an error naming the variable.
OPTIONAL_ENV_READ = re.compile(
    r"""(?:getenv|environ\.get|process\.env\[)\s*\(?\s*['"]([A-Z][A-Z0-9_]{2,})['"]"""
    r"""|process\.env\.([A-Z][A-Z0-9_]{2,})"""
)
# The error call plus its (possibly multi-line) message arguments.
ERROR_MESSAGE_LINE = re.compile(r"(?:raise\s+\w+(?:\.\w+)*\(|throw\s+new\s+\w*Error\()[^)]{0,400}")
ENV_EXAMPLE_FILES = (".env.example", ".env.sample", ".env.template", "example.env", ".env.dist")
SECRET_ENV_NAME = re.compile(
    r"(API[_-]?KEY|(?:^|_)TOKEN$|ACCESS_TOKEN$|_TOKEN_SECRET|SECRET(?:_KEY)?$|PASSWORD$"
    r"|CREDENTIALS?$|PRIVATE[_-]?KEY|_KEY$)",
    re.I,
)
# Settings named like secrets that actually hold flags, counts, or durations
# (MAX_SESSIONS_PER_API_KEY, CACHE_API_KEY_ENABLED, AUTH_DISABLE_USERNAME_PASSWORD,
# CACHE_API_KEY_TTL_SECONDS); a secret string there fails int()/enum parsing.
NON_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:MAX|MIN|NUM|PER|ENABLE|ENABLED|DISABLE|DISABLED|ALLOW|REQUIRE|REQUIRED"
    r"|USE|TTL|COUNT|LIMIT|SIZE|LENGTH|SECONDS|MS|MINUTES|HOURS|DAYS|TIMEOUT|EXPIRE|EXPIRES"
    r"|HEADER|HEADERS|NAME|PATH|FILE|DIR|ID)(?:_|$)",
    re.I,
)
# 64 hex characters: long enough for min_length=32 validators and also a valid
# 256-bit hex key (for example "ENCRYPTION_KEY must be 64 hex characters").
PLACEHOLDER_SECRET = "a9e1c0de" * 8
# Token counts and limits (MAX_TOKENS, CONTEXT_TOKENS) need an integer.
TOKEN_COUNT_ENV_NAME = re.compile(r"TOKENS$|TOKEN_LIMIT$", re.I)
NETWORK_ENV_NAME = re.compile(r"(URL|URI|HOST|ENDPOINT|BASE|DSN|PORT|ADDR)", re.I)

# Required URL settings with a built-in transport that needs no server.
NON_NETWORK_URL_PLACEHOLDERS = {
    "CELERY_BROKER_URL": "memory://",
    "CELERY_RESULT_BACKEND": "cache+memory://",
}

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

# TypeScript/JavaScript migration compatibility is intentionally absent for
# AutoGen and CrewAI; those rows are recorded as unsupported instead of
# inventing a package surface that Agent RT does not expose.
TS_SPECIFIER_RULES = {
    "LangChain": (
        ("@langchain/", "agent-rt/@langchain/"),
        ("langchain", "agent-rt/langchain"),
    ),
    "LlamaIndex": (
        ("@llamaindex/", "agent-rt/@llamaindex/"),
        ("llamaindex", "agent-rt/llamaindex"),
    ),
    "OpenAI": (
        ("openai", "agent-rt/openai"),
    ),
    "Anthropic": (
        ("@anthropic-ai/sdk", "agent-rt/@anthropic-ai/sdk"),
        ("anthropic", "agent-rt/anthropic"),
    ),
    "OpenAI Agents": (
        ("@openai/agents", "agent-rt/@openai/agents"),
    ),
}


@dataclass(frozen=True)
class RepoTarget:
    index: int
    package: str
    language: str
    pl_index: int
    repo_url: str


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int | None
    duration_seconds: float
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def parse_index_selection(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        start, _, end = part.partition("-")
        try:
            first, last = int(start), int(end or start)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid index or range: {part!r}") from exc
        if first > last:
            raise argparse.ArgumentTypeError(f"invalid range: {part!r}")
        indices.update(range(first, last + 1))
    if not indices:
        raise argparse.ArgumentTypeError("no indices given")
    return indices


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-list",
        type=Path,
        default=here / "repo_list.csv",
        help="Input CSV produced from Migration-Test-Targets.md.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=here / "migration_results.csv",
        help="Output CSV. Existing completed repository URLs are skipped unless --restart is used.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Parent directory for disposable clones. Defaults to a temporary directory.",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent repository worker processes. Default: 1.",
    )
    parser.add_argument(
        "--indices",
        type=parse_index_selection,
        default=None,
        help=(
            "Rerun only these repo_list.csv indices, even if already completed, "
            "replacing their rows. Comma-separated values and ranges, e.g. 14,21,30-35."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry existing failed/error rows and replace their CSV rows in place.",
    )
    parser.add_argument(
        "--pass-fail",
        action="store_true",
        help=(
            "Rerun only rows whose tests passed before migration and failed after it "
            "(PASS / FAIL), replacing their rows. Every other repository is skipped."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted --indices / --pass-fail / --retry-failed run: "
            "skip repositories that run already finished. Safe on a first run."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help=(
            "Stop cleanly (resumable with --resume) when free disk space under the work "
            "or log directory drops below this many GB before a repository starts. Default: 10."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds for each install/test command.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=12000,
        help="Maximum captured output stored per command result.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Truncate the result CSV instead of resuming completed repository URLs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=here / "logs",
        help="Directory for the run log and per-repository logs. Default: ./logs.",
    )
    return parser.parse_args()


class TeeStream:
    """Write console output to the terminal and to log files at once.

    Console output keeps truncated command output; ``write_log_only`` sends
    text (full command lines and untruncated output) to the log files only.
    """

    def __init__(self, console, log_files=()):
        self.console = console
        self.log_files = list(log_files)

    def write(self, text):
        self.console.write(text)
        for handle in self.log_files:
            handle.write(text)
        return len(text)

    def write_log_only(self, text):
        for handle in self.log_files:
            handle.write(text)
            handle.flush()

    def flush(self):
        self.console.flush()
        for handle in self.log_files:
            handle.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered append so concurrent workers interleave whole lines.
    return path.open("a", encoding="utf-8", errors="replace", buffering=1)


def install_log_tee(run_log: Path | None) -> None:
    """Route stdout/stderr through a TeeStream that also writes run_log.

    Called in the parent and as the worker-process initializer, so every
    process owns its own log file handle.
    """
    handles = [open_log(run_log)] if run_log else []
    sys.stdout = TeeStream(sys.__stdout__, handles)
    sys.stderr = TeeStream(sys.__stderr__, handles)


def log_only(text: str) -> None:
    stream = sys.stdout
    if isinstance(stream, TeeStream):
        stream.write_log_only(text if text.endswith("\n") else text + "\n")


def repo_log_path(log_dir: Path, run_stamp: str, target: "RepoTarget") -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.repo_url.rstrip("/").split("github.com/")[-1])
    return log_dir / run_stamp / f"{target.index:04d}-{slug}.log"


# Variables that point tools at the runner's own interpreter or environment.
INTERPRETER_ENV = re.compile(
    r"^(VIRTUAL_ENV|VIRTUAL_ENV_PROMPT|CONDA_\w+|_CE_\w+|PYTHONHOME|PYTHONPATH|PYTHONSTARTUP"
    r"|PYTHONUSERBASE|PYTHONEXECUTABLE|UV_PYTHON|UV_PROJECT_ENVIRONMENT|UV_ACTIVE"
    r"|PIP_PYTHON|PIPENV_\w+|POETRY_ACTIVE)$"
)


def is_python_env_bin(entry: str) -> bool:
    """Return True for a PATH entry that is a virtualenv or conda env bin dir."""
    path = Path(entry)
    if path.name not in {"bin", "Scripts"}:
        return False
    env_root = path.parent
    return (env_root / "pyvenv.cfg").is_file() or (env_root / "conda-meta").is_dir()


def sanitized_env() -> dict[str, str]:
    """Return an environment with likely credentials removed."""
    blocked = re.compile(
        r"(API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE[_-]?KEY)",
        re.IGNORECASE,
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not blocked.search(key) and not INTERPRETER_ENV.match(key)
    }
    # The runner itself usually executes inside `uv run` (its own .venv) and
    # often inside an activated conda env. Left on PATH, those interpreters
    # answer `python`/`pytest`/`pip` calls made by repository tests and build
    # scripts instead of the repository's .agent_rt_venv.
    env["PATH"] = os.pathsep.join(
        entry for entry in env.get("PATH", "").split(os.pathsep) if entry and not is_python_env_bin(entry)
    )
    # Never let uv pick an activated conda env or a system interpreter.
    env["UV_MANAGED_PYTHON"] = "1"
    env.update(
        {
            "CI": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "NPM_CONFIG_FUND": "false",
            "NPM_CONFIG_AUDIT": "false",
            # npm 12 stopped fetching lockfile tarball URLs and git
            # dependencies and running install scripts by default; restore the
            # npm <= 11 behavior so results do not depend on the npm major.
            # (Repository code already runs untrusted inside the sandbox.)
            "NPM_CONFIG_ALLOW_REMOTE": "all",
            "NPM_CONFIG_ALLOW_GIT": "all",
            "NPM_CONFIG_DANGEROUSLY_ALLOW_ALL_SCRIPTS": "true",
            # Newer package managers block dependency install/build scripts
            # (Prisma, esbuild, node-gyp modules) and remote/git dependencies
            # by default; npm 10, which produced the existing results, ran and
            # fetched them. Allow them so results do not depend on the npm or
            # pnpm version. These repositories are untrusted, so this relies on
            # the sandbox the runner must run in. Older versions ignore these.
            "npm_config_dangerously_allow_all_scripts": "true",  # npm 11.16+/12
            "npm_config_allow_remote": "all",
            "npm_config_allow_git": "all",
            "npm_config_dangerously_allow_all_builds": "true",  # pnpm 10.9+
            "PNPM_CONFIG_DANGEROUSLY_ALLOW_ALL_BUILDS": "true",  # pnpm 11
        }
    )
    return env


def ensure_aimock_installed(script_dir: Path, env: dict[str, str]) -> None:
    package_dir = script_dir / "node_modules" / "@copilotkit" / "aimock"
    if package_dir.is_dir():
        return
    if shutil.which("npm") is None:
        raise RuntimeError("npm is required to install @copilotkit/aimock")
    print("phase: install migration-test aimock dependency", flush=True)
    result = subprocess.run(
        ["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
        cwd=script_dir,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"npm ci failed with exit code {result.returncode}")


def start_aimock_server(
    script_dir: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], str]:
    if shutil.which("node") is None:
        raise RuntimeError("Node.js is required to run @copilotkit/aimock")
    ensure_aimock_installed(script_dir, env)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    print(f"phase: start aimock provider server on 127.0.0.1:{port}", flush=True)
    process = subprocess.Popen(
        ["node", str(script_dir / "mock_server.mjs"), str(port)],
        cwd=script_dir,
        env=env,
        stdin=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"aimock server exited during startup with code {process.returncode}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError("aimock server did not become ready within 10 seconds")

    url = f"http://127.0.0.1:{port}"
    print(f"aimock ready: {url}", flush=True)
    return process, url


def stop_aimock_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    print("phase: stop aimock provider server", flush=True)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def apply_mock_provider_env(env: dict[str, str], aimock_url: str) -> dict[str, str]:
    mock_env = env.copy()
    mock_key = "agent-rt-migration-test-key"

    # Keep credential validation deterministic without leaking real credentials.
    # Only providers whose SDK traffic is redirected below get a dummy key.
    # A key for an unredirected provider (DeepSeek, Mistral, OpenRouter, ...)
    # un-skips key-gated tests that then call the real hosted API.
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
        "CO_API_KEY",
    ):
        mock_env[key] = mock_key

    # Aimock serves OpenAI-compatible, Anthropic, Gemini, Ollama, and Cohere
    # routes. Each SDK appends its own path prefix to the base URL.
    mock_env["OPENAI_BASE_URL"] = f"{aimock_url}/v1"
    mock_env["OPENAI_API_BASE"] = f"{aimock_url}/v1"
    # The Groq SDK appends /openai/v1/...; aimock strips the /openai prefix.
    mock_env["GROQ_BASE_URL"] = aimock_url
    mock_env["ANTHROPIC_BASE_URL"] = aimock_url
    # google-genai (used by langchain-google-genai) honours this override;
    # without it Gemini calls reach the real Google endpoint.
    mock_env["GOOGLE_GEMINI_BASE_URL"] = aimock_url
    mock_env["OLLAMA_HOST"] = aimock_url
    mock_env["OLLAMA_BASE_URL"] = aimock_url
    mock_env["CO_API_URL"] = aimock_url
    mock_env["AIMOCK_BASE_URL"] = aimock_url
    # Never ship traces to LangSmith/LangChain hosted tracing.
    mock_env["LANGCHAIN_TRACING_V2"] = "false"
    mock_env["LANGSMITH_TRACING"] = "false"
    mock_env["LANGCHAIN_API_KEY"] = mock_key
    mock_env["LANGSMITH_API_KEY"] = mock_key
    return mock_env


# Required pydantic-settings fields: secret-like or database URL names with
# an annotation and no default value.
PYDANTIC_SETTINGS_FIELD = re.compile(
    r"(?m)^\s+([a-z][a-z0-9_]*(?:api_key|apikey|_token|secret|password)"
    r"|(?:test_)?(?:database_url|db_url|sqlalchemy_database_ur[il]))\s*:\s*[^=\n#]+?\s*(?:#.*)?$"
)
# SQLAlchemy's asyncio extension rejects a sync sqlite:/// URL ("requires an
# async driver"), so async applications get sqlite+aiosqlite instead.
ASYNC_SQLALCHEMY_USE = re.compile(r"\b(?:create_async_engine|async_sessionmaker|AsyncSession)\b")
DATABASE_URL_ENV_NAME = re.compile(
    r"^(?:TEST_)?(?:DATABASE_URL|DB_URL|SQLALCHEMY_DATABASE_UR[IL])$"
)


def apply_repo_env_placeholders(repo: Path, env: dict[str, str], *, language: str) -> dict[str, str]:
    """Fill required-but-unset repository settings in env before the first run.

    The names go to the log files only; the console stays quiet.
    """
    placeholders = repo_env_placeholders(repo, env, language=language)
    if placeholders:
        log_only("  [log] placeholder env: " + ", ".join(sorted(placeholders)))
        env.update(placeholders)
    return placeholders


def required_env_names(repo: Path, language: str) -> set[str]:
    """Names of environment variables the repository cannot run without.

    Optional reads are deliberately excluded: filling them turns on code
    paths (webhook auth, live-provider tests, alternate models) that the
    suite expects to be off.
    """
    names: set[str] = set()
    for path in iter_source_files(repo, language):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if language == "Python":
            for groups in REQUIRED_ENV_READ.findall(text):
                names.add(next(group for group in groups if group))
            if "BaseSettings" in text:
                names.update(field.upper() for field in PYDANTIC_SETTINGS_FIELD.findall(text))
        messages = "\n".join(
            match.group(0)
            for match in ERROR_MESSAGE_LINE.finditer(text)
            if raised_at_import(text, match.start(), language)
        )
        if messages:
            for groups in OPTIONAL_ENV_READ.findall(text):
                name = next(group for group in groups if group)
                if re.search(rf"\b{re.escape(name)}\b", messages):
                    names.add(name)
    # A variable the test suite uses to decide whether to skip (for example
    # skipif(not os.getenv("X")) or a conftest skip marker naming it) gates
    # live tests; leave it unset so those tests stay skipped.
    return names - test_skip_gated_env_names(repo, language)


def raised_at_import(text: str, offset: int, language: str) -> bool:
    """Return True when the raise/throw at offset runs at module import.

    Errors raised inside functions are behaviour the tests may assert (for
    example "raises when the key is missing"); only module-level errors
    break collection outright.
    """
    lines = text[:offset].splitlines() or [""]
    current = lines[-1]
    indent = len(current) - len(current.lstrip())
    if language != "Python":
        return indent <= 2
    for line in reversed(lines[:-1]):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent < indent:
            if re.match(r"(?:async\s+def|def|class)\b", stripped):
                return False
            indent = line_indent
            if indent == 0:
                return True
    return True


def test_skip_text(repo: Path, language: str) -> str:
    """Lines of test sources that mention skipping (skip markers, importorskip)."""
    lines: list[str] = []
    for path in iter_source_files(repo, language):
        relative = f"/{path.relative_to(repo).as_posix()}"
        if not re.search(r"(?:/tests?/|/__tests__/|/conftest\.py$|/test_[^/]*$|[._-](?:test|spec)\.[^/]*$)", relative):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines.extend(line for line in text.splitlines() if re.search(r"skip", line, re.I))
    return "\n".join(lines)


def test_skip_gated_env_names(repo: Path, language: str) -> set[str]:
    return set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", test_skip_text(repo, language)))


def repo_env_placeholders(repo: Path, env: dict[str, str], *, language: str = "Python") -> dict[str, str]:
    """Return deterministic values for required repository settings that are unset.

    Secret-like names get a long dummy secret (and the matching provider base
    URL points at the mock); Python database URLs get a local SQLite file;
    model names get a mock-served model; other values come from an
    .env.example when it has a non-network value, and are otherwise skipped.
    """
    examples: dict[str, str] = {}
    for example_name in ENV_EXAMPLE_FILES:
        example = repo / example_name
        if not example.is_file():
            continue
        try:
            lines = example.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if match:
                value = match.group(2).split(" #", 1)[0].strip().strip("'\"")
                if value:
                    examples.setdefault(match.group(1), value)

    placeholders: dict[str, str] = {}
    sqlite_scheme: str | None = None
    for name in sorted(required_env_names(repo, language)):
        if name in env:
            continue
        upper = name.upper()
        example_value = examples.get(name)
        if DATABASE_URL_ENV_NAME.match(upper):
            # SQLite only for Python ORMs; Node ORMs such as Prisma validate
            # the provider scheme and would fail differently.
            if language == "Python":
                if sqlite_scheme is None:
                    sqlite_scheme = "sqlite+aiosqlite" if uses_async_sqlalchemy(repo) else "sqlite"
                placeholders[name] = f"{sqlite_scheme}:///{repo / '.agent_rt_test.db'}"
        elif TOKEN_COUNT_ENV_NAME.search(upper):
            placeholders[name] = example_value if example_value and example_value.isdigit() else "1024"
        elif SECRET_ENV_NAME.search(upper) and not NON_SECRET_ENV_NAME.search(upper):
            placeholders[name] = PLACEHOLDER_SECRET
            # Keep a provider enabled by this key on the mock, not the hosted API.
            prefix = re.sub(r"_?API_?KEY$", "", upper)
            if prefix != upper and prefix and env.get("AIMOCK_BASE_URL"):
                for suffix in ("_BASE_URL", "_API_BASE"):
                    if f"{prefix}{suffix}" not in env:
                        placeholders[f"{prefix}{suffix}"] = f"{env['AIMOCK_BASE_URL']}/v1"
        elif upper in NON_NETWORK_URL_PLACEHOLDERS or upper.endswith("_BROKER_URL"):
            # Celery's in-memory transport: required by settings at import,
            # yet needs no running broker (unlike redis:// or amqp://).
            placeholders[name] = NON_NETWORK_URL_PLACEHOLDERS.get(upper, "memory://")
        elif NETWORK_ENV_NAME.search(upper):
            continue
        elif "MODEL" in upper and not re.search(r"PROVIDER|ADAPTER|PATH|DIR|FILE", upper):
            placeholders[name] = "text-embedding-3-small" if "EMBED" in upper else "gpt-4o-mini"
        elif example_value and "://" not in example_value:
            placeholders[name] = example_value
    return placeholders


def uses_async_sqlalchemy(repo: Path) -> bool:
    for path in iter_source_files(repo, "Python"):
        try:
            if ASYNC_SQLALCHEMY_USE.search(path.read_text(encoding="utf-8")):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
    show_output: bool = True,
) -> CommandResult:
    command_list = list(command)
    # Full command lines stay out of the console but are kept in the logs.
    log_only(f"  [log] cwd: {cwd}\n  [log] command: {' '.join(command_list)}")
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_list,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        full_output = process.stdout or ""
        output = full_output
        if len(output) > max_output_chars:
            output = "...[truncated]...\n" + output[-max_output_chars:]
        duration = round(time.monotonic() - started, 3)
        print(f"  exit: {process.returncode}  duration: {duration}s", flush=True)
        if not show_output and process.returncode == 0:
            # Routine, successful build noise stays in the log files only.
            log_only("  [log] output:\n" + full_output.rstrip())
        elif output.strip():
            print("  output:", flush=True)
            print(output.rstrip(), flush=True)
            if output is not full_output:
                log_only("  [log] untruncated output:\n" + full_output.rstrip())
        else:
            print("  output: <empty>", flush=True)
        return CommandResult(
            command=command_list,
            exit_code=process.returncode,
            duration_seconds=duration,
            output=output,
        )
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if len(raw) > max_output_chars:
            raw = "...[truncated]...\n" + raw[-max_output_chars:]
        duration = round(time.monotonic() - started, 3)
        print(f"  timeout after {duration}s", flush=True)
        if raw.strip():
            print("  partial output:", flush=True)
            print(raw.rstrip(), flush=True)
        return CommandResult(
            command=command_list,
            exit_code=None,
            duration_seconds=duration,
            output=raw,
            timed_out=True,
        )
    except OSError as exc:
        duration = round(time.monotonic() - started, 3)
        message = f"{type(exc).__name__}: {exc}"
        print(f"  command error after {duration}s: {message}", flush=True)
        return CommandResult(
            command=command_list,
            exit_code=None,
            duration_seconds=duration,
            output=message,
        )


def command_json(result: CommandResult, *, status: str | None = None) -> str:
    payload = {
        "status": status or ("pass" if result.ok else "fail"),
        "command": result.command,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "timed_out": result.timed_out,
        "output": result.output,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def synthetic_result(status: str, message: str, **extra: object) -> str:
    payload = {"status": status, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def result_status(result_json: str) -> str:
    try:
        status = str(json.loads(result_json).get("status", "unknown"))
    except (json.JSONDecodeError, TypeError, AttributeError):
        status = "unknown"
    return {
        "pass": "PASS",
        "fail": "FAIL",
        "not_run": "SKIPPED",
        "skipped": "SKIPPED",
        "setup_failed": "SETUP_FAILED",
        "clone_failed": "CLONE_FAILED",
        "unsupported": "UNSUPPORTED",
        "migration_no_changes": "NO_CHANGES",
        "agent_rt_install_failed": "AGENT_RT_INSTALL_FAILED",
        "runner_error": "RUNNER_ERROR",
        "filtered_heavy": "FILTERED_HEAVY",
        "no_tests": "NO_TESTS",
    }.get(status, status.upper())


RETRYABLE_STATUSES = {
    "FAIL",
    "SETUP_FAILED",
    "CLONE_FAILED",
    "AGENT_RT_INSTALL_FAILED",
    "RUNNER_ERROR",
}


def load_targets(path: Path) -> list[RepoTarget]:
    targets: list[RepoTarget] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        expected = {"Package", "Language", "PLIndex", "RepoURL"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(expected)}")
        for index, row in enumerate(reader, start=1):
            targets.append(
                RepoTarget(
                    index=index,
                    package=row["Package"].strip(),
                    language=row["Language"].strip(),
                    pl_index=int(row["PLIndex"]),
                    repo_url=row["RepoURL"].strip(),
                )
            )
    unknown = sorted({target.package for target in targets} - set(PYTHON_IMPORT_ROOTS))
    if unknown:
        # A package name the rewrite tables do not know silently records
        # NO_CHANGES / UNSUPPORTED for every row in that group.
        raise ValueError(f"{path} names packages with no migration rules: {unknown}")
    return targets


def repo_url_key(repo_url: str) -> str:
    """Return a stable completion key for a repository URL."""
    value = repo_url.strip().rstrip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    return value.lower()


def result_rows_by_repo_url(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows_by_url: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = repo_url_key(row.get("RepoURL", ""))
            if key:
                rows_by_url[key] = row
    return rows_by_url


def completed_repo_urls(path: Path) -> set[str]:
    return set(result_rows_by_repo_url(path))


def retryable_repo_urls(path: Path) -> set[str]:
    retryable: set[str] = set()
    for repo_url, row in result_rows_by_repo_url(path).items():
        before = row.get("TestStatusBeforeMigration", "").strip().upper()
        after = row.get("TestStatusAfterMigration", "").strip().upper()
        if before in RETRYABLE_STATUSES or after in RETRYABLE_STATUSES:
            retryable.add(repo_url)
    return retryable


def pass_fail_targets(path: Path) -> set[tuple[str, str, str]]:
    """Return (URL, package, language) of rows whose baseline passed but whose
    migrated suite failed.

    The package and language matter: a URL listed under several frameworks
    shares one result row, so rerunning its other entries would overwrite it.
    """
    return {
        (repo_url, row.get("Package", ""), row.get("Language", ""))
        for repo_url, row in result_rows_by_repo_url(path).items()
        if row.get("TestStatusBeforeMigration", "").strip().upper() == "PASS"
        and row.get("TestStatusAfterMigration", "").strip().upper() == "FAIL"
    }


def prepare_results(
    path: Path,
    *,
    restart: bool,
    targets: Sequence[RepoTarget] = (),
) -> set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if restart or not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()
        return set()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Keep Index/Package/Language/PLIndex in sync with the current repo list
    # (matched by URL, first occurrence wins), so --indices and the CSV agree
    # after rows are added to or removed from repo_list.csv.
    current: dict[str, list[RepoTarget]] = {}
    for target in targets:
        current.setdefault(repo_url_key(target.repo_url), []).append(target)

    def matching_target(key: str, row: dict[str, str]) -> RepoTarget | None:
        # A URL listed under several frameworks keeps the entry for the
        # framework this row actually tested.
        candidates = current.get(key, [])
        for candidate in candidates:
            if candidate.package == row.get("Package", "") and candidate.language == row.get("Language", ""):
                return candidate
        return candidates[0] if len(candidates) == 1 else None

    normalized_by_url: dict[str, dict[str, object]] = {}
    for row in rows:
        repo_url = row.get("RepoURL", "").strip()
        key = repo_url_key(repo_url)
        if not key:
            continue
        raw_index = row.get("Index", "").strip()
        before = row.get("TestResultsBeforeMigration", "")
        after = row.get("TestResultsAfterMigration", "")
        target = matching_target(key, row)
        normalized_by_url[key] = {
            "Index": target.index if target else raw_index,
            "Package": target.package if target else row.get("Package", ""),
            "Language": target.language if target else row.get("Language", ""),
            "PLIndex": target.pl_index if target else row.get("PLIndex", ""),
            "RepoURL": repo_url,
            "CommitHash": row.get("CommitHash", ""),
            "TestStatusBeforeMigration": (
                row.get("TestStatusBeforeMigration", "").strip() or result_status(before)
            ),
            "TestStatusAfterMigration": (
                row.get("TestStatusAfterMigration", "").strip() or result_status(after)
            ),
            "TestResultsBeforeMigration": before,
            "TestResultsAfterMigration": after,
        }

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        normalized_rows = sorted(
            normalized_by_url.values(),
            key=lambda row: int(str(row["Index"])) if str(row["Index"]).isdigit() else sys.maxsize,
        )
        writer.writerows(normalized_rows)

    return set(normalized_by_url)


def upsert_result(path: Path, row: dict[str, object]) -> None:
    rows_by_url = result_rows_by_repo_url(path)
    key = repo_url_key(str(row.get("RepoURL", "")))
    if not key:
        raise ValueError("Result row is missing RepoURL")
    rows_by_url[key] = {field: str(row.get(field, "")) for field in RESULT_FIELDS}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        rows = sorted(
            rows_by_url.values(),
            key=lambda item: int(item.get("Index", "")) if item.get("Index", "").isdigit() else sys.maxsize,
        )
        writer.writerows(rows)
        handle.flush()


CLONE_ATTEMPTS = 3
# Disable any Git LFS filter inherited from the user's git config: without the
# git-lfs binary the checkout aborts ("git-lfs: not found"), and LFS objects
# (datasets, model weights) are never needed to run the test suite.
GIT_CLONE_LFS_OFF = (
    "-c", "filter.lfs.process=",
    "-c", "filter.lfs.smudge=",
    "-c", "filter.lfs.clean=",
    "-c", "filter.lfs.required=false",
)
CLONE_REPO_NOT_FOUND = re.compile(r"repository '[^']*' not found|Repository not found", re.I)


def clone_repo(
    target: RepoTarget,
    destination: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> CommandResult:
    # Large repositories intermittently fail with "the remote end hung up
    # unexpectedly"; retry a couple of times before recording CLONE_FAILED.
    result = None
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        if attempt > 1:
            print(f"    clone failed; retry {attempt}/{CLONE_ATTEMPTS}", flush=True)
            shutil.rmtree(destination, ignore_errors=True)
            time.sleep(5 * attempt)
        result = run_command(
            ["git", *GIT_CLONE_LFS_OFF, "clone", "--depth", "1", "--no-tags", target.repo_url, str(destination)],
            cwd=destination.parent,
            timeout=timeout,
            max_output_chars=max_output_chars,
            env={**env, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
        if result.ok or CLONE_REPO_NOT_FOUND.search(result.output):
            break
    return result


def commit_hash(
    repo: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> str:
    result = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=env,
    )
    return result.output.strip().splitlines()[-1][:7] if result.ok and result.output.strip() else ""


def venv_bin(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def activate_venv(env: dict[str, str], venv_dir: Path) -> dict[str, str]:
    """Return env as if the repository venv were activated."""
    activated = env.copy()
    activated["VIRTUAL_ENV"] = str(venv_dir)
    activated["PATH"] = os.pathsep.join(
        entry for entry in (str(venv_bin(venv_dir)), activated.get("PATH", "")) if entry
    )
    return activated


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def requirement_install_command(python: Path, requirement_file: Path) -> list[str]:
    command = ["uv", "pip", "install", "--python", str(python), "-r", str(requirement_file)]
    try:
        text = read_manifest_text(requirement_file)
    except OSError:
        return command
    if re.search(r"(?mi)^\s*(?:torch|torchvision|torchaudio)[^\n#]*\+cpu\b", text):
        command.extend(["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
    return command


def discover_requirement_files(repo: Path) -> list[Path]:
    candidates: list[Path] = []
    for root, dirs, files in os.walk(repo):
        base = Path(root)
        try:
            depth = len(base.relative_to(repo).parts)
        except ValueError:
            continue
        if depth >= 3:
            dirs[:] = []
        else:
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in files:
            lowered = name.lower()
            if lowered.endswith(".txt") and "requirement" in lowered:
                candidates.append(base / name)

    candidates = sorted(
        set(candidates),
        key=lambda path: (len(path.relative_to(repo).parts), str(path)),
    )
    root_requirements = [path for path in candidates if path.parent == repo]
    if root_requirements:
        return root_requirements

    # cookiecutter-style layout: requirements/{base,local,test}.txt where the
    # test/dev file pulls in base via "-r base.txt" and adds pytest plugins.
    requirements_dir = repo / "requirements"
    if requirements_dir.is_dir():
        for name in ("test.txt", "tests.txt", "testing.txt", "dev.txt", "development.txt", "local.txt", "base.txt"):
            if (requirements_dir / name).is_file():
                return [requirements_dir / name]

    # A root Python project should not install every example/demo requirements
    # file. Those often reference unpublished companion packages or optional
    # heavyweight stacks unrelated to the repository's own test suite.
    if (repo / "pyproject.toml").is_file() or (repo / "setup.py").is_file():
        return []

    # For repositories without root packaging metadata, only consider a nested
    # requirements file when its directory looks like a real testable Python
    # project. Example/demo requirements frequently reference unpublished
    # companion packages and should not determine whether the repository can be
    # baseline-tested.
    example_parts = {"demo", "demos", "example", "examples", "sample", "samples", "notebook", "notebooks"}
    relevant_nested = [
        path
        for path in candidates
        if not any(part.lower() in example_parts for part in path.relative_to(repo).parts[:-1])
        if (path.parent / "pyproject.toml").is_file()
        or (path.parent / "setup.py").is_file()
        or (path.parent / "tests").is_dir()
        or any(path.parent.glob("test_*.py"))
    ]
    if not relevant_nested:
        return []
    shallowest = min(len(path.relative_to(repo).parts) for path in relevant_nested)
    return [
        path
        for path in relevant_nested
        if len(path.relative_to(repo).parts) == shallowest
    ]


def discover_python_projects(repo: Path) -> list[Path]:
    projects: list[Path] = []
    for root, dirs, files in os.walk(repo):
        base = Path(root)
        depth = len(base.relative_to(repo).parts)
        if depth >= 4:
            dirs[:] = []
        else:
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        if "pyproject.toml" in files or "setup.py" in files:
            projects.append(base)
    return sorted(
        set(projects),
        key=lambda path: (len(path.relative_to(repo).parts), str(path)),
    )


def pyproject_dependencies(repo: Path, *, all_extras: bool = False) -> list[str]:
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    dependencies: list[str] = []
    test_groups = ("dev", "test", "tests", "testing")
    project = data.get("project")
    if isinstance(project, dict):
        base_dependencies = project.get("dependencies")
        if isinstance(base_dependencies, list):
            dependencies.extend(str(item) for item in base_dependencies if isinstance(item, str))

        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            # Test suites often import optional integrations (for example a
            # conftest importing aio_pika from a "providers" extra).
            for group in (optional if all_extras else test_groups):
                values = optional.get(group)
                if isinstance(values, list):
                    dependencies.extend(str(item) for item in values if isinstance(item, str))

    # PEP 735 dependency groups are where many uv/hatch projects keep pytest
    # plugins and test-only libraries. Nested include-group tables are skipped.
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group in test_groups:
            values = groups.get(group)
            if isinstance(values, list):
                dependencies.extend(str(item) for item in values if isinstance(item, str))

    return dependencies


def local_python_module_names(repo: Path) -> set[str]:
    names: set[str] = set()
    for path in iter_source_files(repo, "Python"):
        names.add(path.stem)
        names.update(path.relative_to(repo).parts[:-1])
    return names


def inferred_common_dependencies(repo: Path) -> dict[str, str]:
    """Return curated module -> distribution pairs the repository imports.

    Keys may be dotted (``google.genai``) for namespace packages; the longest
    curated prefix of each imported module wins. Modules that the test suite
    gates with a skip (importorskip("x"), skipif(... reason="x not
    installed")) are left out: installing them would un-skip tests that a
    passing baseline never ran.
    """
    needed: dict[str, str] = {}
    local_modules = local_python_module_names(repo)
    skip_text = test_skip_text(repo, "Python").lower()
    # Families of skip-gated packages: gating on langchain-core must also
    # keep langchain-openai out, since it depends on langchain-core.
    gated_families: set[str] = set()
    for key, package in COMMON_IMPORT_DEPENDENCIES.items():
        distribution = re.split(r"[<>=!~\[]", package)[0].lower()
        for name in {key.split(".")[0].lower(), distribution, distribution.replace("-", "_")}:
            if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", skip_text):
                gated_families.add(re.split(r"[-_]", name)[0])
    pattern = re.compile(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)")
    # "from google import genai" imports the google.genai module.
    from_pattern = re.compile(
        r"(?m)^\s*from\s+([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)\s+import\s+\(?\s*([A-Za-z_][\w ,]*)"
    )
    for path in iter_source_files(repo, "Python"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        imports = list(pattern.findall(text))
        for module, names in from_pattern.findall(text):
            imports.extend(
                f"{module}.{name.split(' as ')[0].strip()}"
                for name in names.split(",")
                if name.strip()
            )
        for imported in imports:
            parts = imported.split(".")
            if parts[0] in local_modules:
                continue
            for length in range(len(parts), 0, -1):
                module = ".".join(parts[:length])
                package = COMMON_IMPORT_DEPENDENCIES.get(module)
                if package:
                    distribution = re.split(r"[<>=!~\[]", package)[0].lower()
                    families = {re.split(r"[-_]", parts[0].lower())[0], re.split(r"[-_]", distribution)[0]}
                    if not families & gated_families:
                        needed[module] = package
                    break
    return dict(sorted(needed.items()))


def missing_dependencies_from_output(output: str, repo: Path) -> list[str]:
    """Curated distributions/plugins that a failed run reports as missing.

    Only names in COMMON_IMPORT_DEPENDENCIES / PYTEST_FIXTURE_PLUGINS are ever
    returned, and never for modules that exist in the repository itself.
    """
    local_modules = local_python_module_names(repo)
    modules = list(MISSING_MODULE_ERROR.findall(output))
    modules += [f"{parent}.{name}" for name, parent in MISSING_NAMESPACE_MEMBER.findall(output)]
    packages: set[str] = set()
    for module in modules:
        parts = module.removeprefix("agent_rt.").split(".")
        if parts[0] in local_modules:
            continue
        for length in range(len(parts), 0, -1):
            package = COMMON_IMPORT_DEPENDENCIES.get(".".join(parts[:length]))
            if package:
                packages.add(package)
                break
    for fixture in MISSING_FIXTURE_ERROR.findall(output):
        plugin = PYTEST_FIXTURE_PLUGINS.get(fixture)
        if plugin:
            packages.add(plugin)
    if ASYNC_TESTS_UNSUPPORTED.search(output):
        packages.add("pytest-asyncio")
    for hinted in INSTALL_HINT_PACKAGE.findall(output):
        if hinted in HINTED_PACKAGES:
            packages.add(hinted)
    return sorted(packages)


def inferred_pytest_plugins(repo: Path) -> list[str]:
    """Return pytest plugins implied by pytest configuration or test sources."""
    texts: list[str] = []
    for name in PYTEST_CONFIG_FILES:
        path = repo / name
        if path.is_file():
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                pass
    for path in iter_source_files(repo, "Python"):
        relative = path.relative_to(repo).as_posix()
        if path.name.startswith("test_") or path.name.endswith("_test.py") or path.name == "conftest.py" or "/tests/" in f"/{relative}":
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                pass
    corpus = "\n".join(texts)
    return sorted({plugin for pattern, plugin in PYTEST_PLUGIN_HINTS if pattern.search(corpus)})


def missing_modules(
    python: Path,
    modules: Iterable[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> list[str]:
    module_list = sorted(modules)
    if not module_list:
        return []
    # find_spec("google.genai") raises when the parent package is missing.
    probe = (
        "import importlib.util, json, sys\n"
        "def absent(m):\n"
        "    try:\n"
        "        return importlib.util.find_spec(m) is None\n"
        "    except (ImportError, ValueError):\n"
        "        return True\n"
        "print(json.dumps([m for m in sys.argv[1:] if absent(m)]))"
    )
    try:
        process = subprocess.run(
            [str(python), "-c", probe, *module_list],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return list(json.loads(process.stdout.strip().splitlines()[-1]))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return module_list


def unresolvable_requirements(output: str) -> set[str]:
    names: set[str] = set()
    flattened = re.sub(r"\s+", " ", output)
    for pattern in UNRESOLVABLE_REQUIREMENT_PATTERNS:
        names.update(normalized_distribution(match) for match in pattern.findall(flattened))
    return names


def normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def relax_requirements_file(
    source: Path,
    destination: Path,
    unresolvable: set[str],
    no_version: set[str],
    replacements: dict[str, str] | None = None,
) -> bool:
    """Write a copy of a requirements file without unresolvable entries.

    The copy should live next to the source so relative ``-r``/``-e`` paths
    keep resolving. Packages missing from the registry are dropped; packages whose pinned
    version does not exist (or that conflict with another exact pin) are
    unpinned, and ``replacements`` swaps a distribution for an unpinned
    drop-in. Returns False when nothing changed.
    """
    try:
        lines = read_manifest_text(source).splitlines()
    except OSError:
        return False
    changed = False
    output: list[str] = []
    for line in lines:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?", line)
        name = normalized_distribution(match.group(1)) if match else ""
        if name and name in unresolvable:
            output.append(f"# dropped by migration runner: {line}")
            changed = True
        elif name and replacements and name in replacements:
            output.append(f"{replacements[name]}{match.group(2) or ''}")
            changed = True
        elif name and name in no_version:
            output.append(f"{match.group(1)}{match.group(2) or ''}")
            changed = True
        else:
            output.append(line)
    if changed:
        destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return changed


def remove_stale_build_metadata(repo: Path) -> list[str]:
    """Delete committed ``*.egg-info`` build artifacts from the checkout.

    Once a project is editable-installed its source directory is on sys.path,
    so a committed egg-info (often under an old distribution name) becomes a
    second installed distribution: duplicate entry points then make pytest
    abort with "Plugin already registered", and stale metadata can shadow the
    real dependency list.
    """
    removed: list[str] = []
    for root, dirs, _files in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in list(dirs):
            if name.endswith(".egg-info"):
                path = Path(root) / name
                shutil.rmtree(path, ignore_errors=True)
                dirs.remove(name)
                removed.append(str(path.relative_to(repo)))
    return sorted(removed)


INVALID_PROJECT_METADATA = re.compile(
    r"Failed to extract static metadata|Failed to parse (?:version|metadata)|"
    r"configuration error: `project\.|invalid pyproject\.toml config"
    # [build-system] build-backend names a module the backend package does not
    # have (for example setuptools.backends.legacy:build).
    r"|build backend returned an error[\s\S]*?No module named "
    r"'(?:setuptools|hatchling|poetry|flit_core|pdm|mesonpy|scikit_build_core)\.[\w.]+'",
    re.I,
)


def is_installable_project(project: Path) -> bool:
    """Mirror uv: a pyproject without [build-system] (and no setup.py) is virtual."""
    if (project / "setup.py").is_file():
        return True
    try:
        data = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    tool_uv = data.get("tool", {}).get("uv", {}) if isinstance(data.get("tool"), dict) else {}
    if isinstance(tool_uv, dict) and tool_uv.get("package") is False:
        return False
    return "build-system" in data


# Every repository environment uses this one uv-managed CPython. 3.12 ties
# 3.13 on the requires-python ranges declared by the target repositories, but
# has wheels for far more of the older pinned dependencies (asyncpg 0.29,
# numpy < 2.1, older pydantic-core/tiktoken, ...) those repositories use, and
# it is inside Agent RT's supported range. No single version satisfies every
# declared range (some repositories require >=3.13, others <3.12), so the
# next supported version is used only when the root project's declared
# requires-python excludes 3.12. .python-version files are ignored.
PYTHON_VERSION = "3.12"
PYTHON_VERSION_ALTERNATIVES = ("3.13", "3.11", "3.14", "3.10")


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _clause_allows(clause: str, version: tuple[int, int, int]) -> bool:
    match = re.match(r"\s*(~=|===|==|!=|<=|>=|<|>|\^|~)?\s*v?([0-9][0-9.*]*)\s*$", clause)
    if not match:
        return True  # Unknown syntax never excludes a version.
    operator, raw = match.group(1) or "==", match.group(2)
    if raw.endswith(".*"):
        prefix = _version_tuple(raw)
        matches = version[: len(prefix)] == prefix
        return not matches if operator == "!=" else matches
    target = _version_tuple(raw)
    padded = target + (0,) * (3 - len(target))
    if operator in ("==", "==="):
        return version[: len(target)] == target
    if operator == "!=":
        return version[: len(target)] != target
    if operator == ">=":
        return version >= padded
    if operator == ">":
        return version > padded
    if operator == "<=":
        return version <= padded
    if operator == "<":
        return version < padded
    if operator == "~=":
        upper = target[:-1][:-1] + (target[:-1][-1] + 1,) if len(target) > 1 else (target[0] + 1,)
        return version >= padded and version[: len(upper)] < upper
    if operator == "^":  # Poetry caret: compatible within the major version.
        return version >= padded and version[0] == target[0]
    if operator == "~":  # Poetry tilde: compatible within the minor version.
        return version >= padded and version[:2] == (padded[0], padded[1])
    return True


def python_spec_allows(spec: str, minor: str) -> bool:
    """Evaluate a requires-python (PEP 440 or Poetry) range for X.Y.

    Uses X.Y.99 so a range like ">=3.12.4" still admits the 3.12 line.
    """
    version = (*_version_tuple(minor)[:2], 99)
    return any(
        all(_clause_allows(clause, version) for clause in alternative.split(",") if clause.strip())
        for alternative in spec.split("||")
    )


def declared_python_ranges(repo: Path) -> list[str]:
    try:
        data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return []
    ranges: list[str] = []
    project = data.get("project")
    if isinstance(project, dict) and isinstance(project.get("requires-python"), str):
        ranges.append(project["requires-python"])
    poetry = data.get("tool", {}).get("poetry", {}) if isinstance(data.get("tool"), dict) else {}
    dependencies = poetry.get("dependencies") if isinstance(poetry, dict) else None
    if isinstance(dependencies, dict) and isinstance(dependencies.get("python"), str):
        ranges.append(dependencies["python"])
    return ranges


def project_owns_tests(project: Path, repo: Path) -> bool:
    """True for the repository root and nested projects with their own tests."""
    return (
        project == repo
        or (project / "tests").is_dir()
        or (project / "test").is_dir()
        or any(project.glob("test_*.py"))
    )


def python_project_ranges(repo: Path) -> list[str]:
    """requires-python ranges of the root and of nested projects that own tests.

    A nested project that declares ``>=3.14`` cannot be installed into the 3.12
    venv chosen from a root that declares nothing, so its range counts too.
    """
    ranges: list[str] = []
    for project in discover_python_projects(repo):
        if project_owns_tests(project, repo):
            ranges.extend(declared_python_ranges(project))
    return ranges


def select_python_version(repo: Path) -> str:
    ranges = python_project_ranges(repo)
    for candidate in (PYTHON_VERSION, *PYTHON_VERSION_ALTERNATIVES):
        if all(python_spec_allows(spec, candidate) for spec in ranges):
            return candidate
    return PYTHON_VERSION


def setup_python_repo(
    repo: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> tuple[Path, list[CommandResult]]:
    python_version = select_python_version(repo)
    if python_version != PYTHON_VERSION:
        print(
            f"    requires-python excludes {PYTHON_VERSION}; using Python {python_version}",
            flush=True,
        )
    return setup_python_environment(
        repo,
        python_version,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=env,
    )


def setup_python_environment(
    repo: Path,
    python_version: str,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> tuple[Path, list[CommandResult]]:
    venv_dir = repo / ".agent_rt_venv"
    results: list[CommandResult] = []
    setup_env = activate_venv(env, venv_dir)
    setup_env.pop("PYTHONPATH", None)
    # Keep the repository cwd for relative requirement/local paths, but prevent
    # project modules (for example agents/types.py) from shadowing stdlib
    # modules when uv or a build backend starts Python.
    setup_env["PYTHONSAFEPATH"] = "1"

    def run_setup(command: list[str], *, command_env: dict[str, str] | None = None) -> CommandResult:
        return run_command(
            command,
            cwd=repo,
            timeout=timeout,
            max_output_chars=max_output_chars,
            env=command_env or setup_env,
        )

    def run_required(command: list[str], *, command_env: dict[str, str] | None = None) -> bool:
        result = run_setup(command, command_env=command_env)
        results.append(result)
        return result.ok

    if not run_required(
        [
            "uv",
            "venv",
            "--python",
            python_version,
            "--managed-python",
            "--seed",
            str(venv_dir),
        ]
    ):
        return venv_python(venv_dir), results
    python = venv_python(venv_dir)

    if (repo / "uv.lock").is_file():
        print("    uv.lock detected; syncing locked environment", flush=True)
        sync_env = setup_env.copy()
        sync_env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
        # Pin the interpreter: without --python, uv sync replaces the venv
        # with whatever interpreter it prefers for requires-python.
        sync = ["uv", "sync", "--frozen", "--python", str(python)]
        # Include every extra so test-only optional imports are present; fall
        # back to the default set when extras conflict or fail to build.
        all_extras_sync = run_setup([*sync, "--all-extras"], command_env=sync_env)
        if not all_extras_sync.ok:
            print("    uv sync with all extras failed; retrying default extras", flush=True)
            if not run_required(sync, command_env=sync_env):
                return python, results
    else:
        for requirement_file in discover_requirement_files(repo):
            print(
                f"    installing declared requirements: {requirement_file.relative_to(repo)}",
                flush=True,
            )
            relaxed = requirement_file.with_name(f".agent_rt_relaxed-{requirement_file.name}")
            install_file = requirement_file
            if relax_requirements_file(
                requirement_file, relaxed, set(), set(), SOURCE_ONLY_REPLACEMENTS
            ):
                print(
                    "    using binary drop-ins for source-only requirements: "
                    + ", ".join(sorted(SOURCE_ONLY_REPLACEMENTS.values())),
                    flush=True,
                )
                install_file = relaxed
            install = run_setup(requirement_install_command(python, install_file))
            # Abandoned repositories often pin versions that were yanked or
            # never published, conflict with each other, or list unpublished
            # companion packages. Drop or unpin only the entries the resolver
            # named, then retry.
            dropped: set[str] = set()
            unpinned: set[str] = set()
            for _ in range(5):
                if install.ok:
                    break
                flattened = re.sub(r"\s+", " ", install.output)
                missing = {
                    normalized_distribution(name)
                    for name in UNRESOLVABLE_REQUIREMENT_PATTERNS[0].findall(flattened)
                }
                no_version = {
                    normalized_distribution(name)
                    for pattern in UNRESOLVABLE_REQUIREMENT_PATTERNS[1:]
                    for name in pattern.findall(flattened)
                }
                if not (missing - dropped or no_version - unpinned):
                    break
                dropped |= missing
                unpinned |= no_version
                source = relaxed if relaxed.is_file() else requirement_file
                if not relax_requirements_file(
                    source, relaxed, dropped, unpinned, SOURCE_ONLY_REPLACEMENTS
                ):
                    break
                print(
                    "    retrying requirements without unresolvable entries: "
                    + ", ".join(sorted(dropped | unpinned)),
                    flush=True,
                )
                install = run_setup(requirement_install_command(python, relaxed))
            results.append(install)
            if not install.ok:
                return python, results

        all_dependencies = pyproject_dependencies(repo, all_extras=True)
        dependencies = pyproject_dependencies(repo)
        if all_dependencies:
            print(f"    installing {len(all_dependencies)} pyproject dependencies", flush=True)
            installed = run_setup(
                ["uv", "pip", "install", "--python", str(python), *all_dependencies]
            )
            if not installed.ok and dependencies != all_dependencies:
                print("    optional extras failed; installing dev/test extras only", flush=True)
                installed = run_setup(
                    ["uv", "pip", "install", "--python", str(python), *dependencies]
                )
            results.append(installed)
            if not installed.ok:
                return python, results

        for project in discover_python_projects(repo):
            relative_project = project.relative_to(repo)
            display_name = "." if not relative_project.parts else str(relative_project)
            if not is_installable_project(project):
                print(
                    f"    skipping editable install for {display_name}: "
                    "pyproject.toml declares no build system",
                    flush=True,
                )
                # A nested virtual project that owns tests (backend/ with
                # backend/tests/) still declares the dependencies its tests
                # import; only the root's were installed above.
                if project != repo and project_owns_tests(project, repo):
                    nested = pyproject_dependencies(project, all_extras=True)
                    if nested:
                        print(
                            f"    installing {len(nested)} pyproject dependencies of {display_name}",
                            flush=True,
                        )
                        installed = run_setup(["uv", "pip", "install", "--python", str(python), *nested])
                        if not installed.ok:
                            installed = run_setup(
                                ["uv", "pip", "install", "--python", str(python), *pyproject_dependencies(project)]
                            )
                        results.append(installed)
                        if not installed.ok:
                            return python, results
                continue
            print(f"    attempting editable project install: {display_name}", flush=True)
            editable = run_setup(
                ["uv", "pip", "install", "--python", str(python), "-e", str(project)]
            )
            if not editable.ok and INVALID_PROJECT_METADATA.search(editable.output):
                # For example requires-python = "^3.11" (Poetry syntax) inside
                # a PEP 621 [project] table. The sources stay importable via
                # PYTHONPATH, and dependencies were installed separately.
                print(
                    f"    {display_name} has invalid pyproject metadata; "
                    "continuing without an editable install",
                    flush=True,
                )
                continue
            if not editable.ok:
                relevant_project = project_owns_tests(project, repo)
                print(
                    f"    editable install failed for {display_name}"
                    + (
                        "; setup cannot continue"
                        if relevant_project
                        else "; ignoring unrelated nested project"
                    ),
                    flush=True,
                )
                if relevant_project:
                    results.append(editable)
                    return python, results

    test_tools = ["pytest"]
    if "pytest-timeout" not in inferred_pytest_plugins(repo):
        test_tools.append("pytest-timeout")
    if not run_required(["uv", "pip", "install", "--python", str(python), *test_tools]):
        return python, results

    # Up front: curated undeclared imports that tests do not skip-gate, and
    # pytest plugins that configuration or async tests strictly need.
    # Fixture plugins and anything else are handled after the first run.
    plugins = inferred_common_dependencies(repo)
    for plugin in inferred_pytest_plugins(repo):
        plugins.setdefault("xdist" if plugin == "pytest-xdist" else plugin.replace("-", "_"), plugin)
    absent = missing_modules(python, plugins, cwd=repo, env=setup_env)
    plugin_packages = sorted({plugins[module] for module in absent})
    if plugin_packages:
        print("    installing undeclared dependencies and required pytest plugins: " + ", ".join(plugin_packages), flush=True)
        optional = run_setup(["uv", "pip", "install", "--python", str(python), *plugin_packages])
        if not optional.ok:
            for plugin in plugin_packages:
                run_setup(["uv", "pip", "install", "--python", str(python), plugin])

    return python, results


def installed_langchain_distributions(
    python: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    prefix: str = "langchain",
) -> list[str]:
    try:
        process = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    names: list[str] = []
    for line in process.stdout.splitlines():
        match = re.match(rf"({re.escape(prefix)}[A-Za-z0-9._-]*)\s*(?:==|@)", line.strip(), re.I)
        if match:
            names.append(normalized_distribution(match.group(1)))
    return sorted(set(names))


def downgrade_legacy_llama_index(
    repo: Path,
    python: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> CommandResult:
    """Replace the namespaced LlamaIndex packages with the 0.9 monolith.

    LlamaIndex 0.10 moved ``from llama_index import VectorStoreIndex`` and
    ``llama_index.llms.OpenAI`` into llama-index-core plus per-integration
    namespace packages. Those cannot coexist with 0.9's regular packages in
    one site-packages, so every installed llama-index* distribution is
    removed before installing ``llama-index<0.10``.
    """
    setup_env = env.copy()
    setup_env.pop("PYTHONPATH", None)
    setup_env["PYTHONSAFEPATH"] = "1"
    installed = installed_langchain_distributions(
        python, cwd=repo, env=setup_env, prefix="llama-index"
    ) + installed_langchain_distributions(python, cwd=repo, env=setup_env, prefix="llama_index")
    if installed:
        run_command(
            ["uv", "pip", "uninstall", "--python", str(python), *sorted(set(installed))],
            cwd=repo,
            timeout=timeout,
            max_output_chars=max_output_chars,
            env=setup_env,
        )
    return run_command(
        ["uv", "pip", "install", "--python", str(python), "llama-index<0.10"],
        cwd=repo,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=setup_env,
    )


def downgrade_legacy_langchain(
    repo: Path,
    python: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> CommandResult:
    """Install the last LangChain line that still ships legacy import shims.

    LangChain 1.0 removed ``langchain.vectorstores``/``langchain.llms`` and
    friends; 0.3.x keeps them as re-exports from langchain-community. The core
    packages are capped below the 1.0 line and every other installed
    langchain* integration is requested unpinned, so the resolver moves them
    to versions compatible with langchain-core 0.3.
    """
    setup_env = env.copy()
    setup_env.pop("PYTHONPATH", None)
    setup_env["PYTHONSAFEPATH"] = "1"
    capped = {
        "langchain": "langchain<1",
        "langchain-core": "langchain-core<1",
        "langchain-community": "langchain-community<0.4",
        "langchain-text-splitters": "langchain-text-splitters<1",
    }
    integrations = [
        name
        for name in installed_langchain_distributions(python, cwd=repo, env=setup_env)
        # langchain-classic only exists for the 1.x line.
        if name not in capped and name != "langchain-classic"
    ]
    return run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            *capped.values(),
            *integrations,
        ],
        cwd=repo,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=setup_env,
    )


# Third-party top-level modules that script directories commonly reuse as
# file names (tools/httpx.py, utils/openai.py, ...).
SHADOWABLE_THIRD_PARTY = frozenset(
    {key.split(".")[0] for key in COMMON_IMPORT_DEPENDENCIES}
    | {root for roots in PYTHON_IMPORT_ROOTS.values() for root in roots}
    | {"numpy", "pytest", "six", "urllib3", "certifi", "typing_extensions"}
)


def shadows_stdlib(directory: Path) -> bool:
    """Return True when a directory holds modules named like stdlib modules.

    Putting such a directory (for example agents/ containing types.py) on
    PYTHONPATH makes the interpreter import it instead of the stdlib during
    startup, crashing every test run before pytest is even imported.
    """
    for child in directory.iterdir():
        name = child.stem if child.suffix == ".py" else child.name
        if (child.suffix == ".py" or (child / "__init__.py").is_file()) and (
            name in sys.stdlib_module_names or name in SHADOWABLE_THIRD_PARTY
        ):
            return True
    return False


def python_repo_env(repo: Path, env: dict[str, str]) -> dict[str, str]:
    # Tests that spawn `python`, `pytest`, or console scripts must resolve
    # them from the repository venv, not from whatever is first on PATH.
    repo_env = activate_venv(env, repo / ".agent_rt_venv")
    # Tests that shell out to `uv run`/`uv sync` would otherwise build a
    # separate .venv with another interpreter (and warn that VIRTUAL_ENV does
    # not match the project environment).
    repo_env["UV_PROJECT_ENVIRONMENT"] = str(repo / ".agent_rt_venv")
    repo_env["UV_PYTHON"] = str(venv_python(repo / ".agent_rt_venv"))
    candidates: list[Path] = []
    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        # A package (has __init__.py) is already importable from the root;
        # adding its own directory exposes submodules as top-level modules
        # (tools/httpx.py would shadow the real httpx).
        if (child / "__init__.py").is_file():
            continue
        if any(child.glob("*.py")) or (child / "tests").is_dir():
            candidates.append(child)
    # Nested projects without packaging metadata (for example mem/py with
    # cavemem.py and tests/) import their sibling modules directly; expose the
    # parent of every shallow tests/ directory that holds Python sources.
    for root, dirs, _files in os.walk(repo):
        base = Path(root)
        depth = len(base.relative_to(repo).parts)
        dirs[:] = [] if depth >= 3 else [name for name in dirs if name not in SKIP_DIRS]
        # A src/ directory (with or without __init__.py) imported as
        # ``src.pkg.mod`` needs the project directory itself importable.
        src_layout = "src" in dirs
        if (
            depth >= 2
            and "tests" in dirs
            and (any(base.glob("*.py")) or src_layout)
            and not (base / "__init__.py").is_file()
        ):
            candidates.append(base)

    paths = [str(repo)]
    for candidate in candidates:
        if str(candidate) in paths:
            continue
        if shadows_stdlib(candidate):
            print(
                f"  not adding {candidate.relative_to(repo)} to PYTHONPATH: "
                "it contains modules that shadow the Python standard library "
                "or common third-party packages",
                flush=True,
            )
            continue
        paths.append(str(candidate))
    existing = repo_env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    repo_env["PYTHONPATH"] = os.pathsep.join(paths)
    return repo_env


# Per-test timeout (pytest-timeout) so one hanging test fails on its own
# instead of the whole suite running into the command timeout.
PYTEST_PER_TEST_TIMEOUT_SECONDS = 120
PYTEST_IMPORT_MISMATCH = re.compile(r"import file mismatch|has this __file__ attribute")


def python_test_command(repo: Path, python: Path) -> list[str]:
    # Pytest is deliberately used as the common denominator; it also runs
    # unittest-style suites in the majority of third-party Python projects.
    command = [str(python), "-m", "pytest", "-q"]
    if "pytest-timeout" not in inferred_pytest_plugins(repo):
        # Respect a repository's own timeout configuration when present.
        command.append(f"--timeout={PYTEST_PER_TEST_TIMEOUT_SECONDS}")
    return command


def load_package_json(repo: Path) -> dict[str, object]:
    package_json = repo / "package.json"
    if not package_json.is_file():
        return {}
    try:
        return json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def pnpm_version(install_root: Path, env: dict[str, str]) -> tuple[int, ...] | None:
    """The pnpm version that runs for this project (its pinned packageManager first)."""
    spec = str(load_package_json(install_root).get("packageManager", ""))
    text = spec[len("pnpm@"):] if spec.startswith("pnpm@") else ""
    if not text:
        try:
            text = subprocess.run(
                ["pnpm", "--version"], cwd=install_root, env=env, capture_output=True,
                text=True, timeout=60, stdin=subprocess.DEVNULL,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    match = re.match(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def allow_legacy_pnpm_builds(install_root: Path, env: dict[str, str]) -> str | None:
    """Let pnpm 10.0-10.8 run dependency build scripts, as earlier pnpm did.

    pnpm 10 skips dependency build scripts (Prisma, esbuild, sharp, ...)
    unless allow-listed. 10.9+ and 11 honor dangerouslyAllowAllBuilds from
    the environment (see sanitized_env), but 10.0-10.8 only accept the
    project-level `neverBuiltDependencies: []`, so set it in package.json.
    """
    version = pnpm_version(install_root, env)
    if version is None or version < (10, 0):
        return None
    legacy = version < (10, 9)
    package_json = install_root / "package.json"
    data = load_package_json(install_root)
    if not package_json.is_file() or not data:
        return None
    settings = data.get("pnpm") if isinstance(data.get("pnpm"), dict) else {}
    # neverBuiltDependencies (and the dangerouslyAllowAllBuilds that newer pnpm
    # maps onto it) cannot be combined with an allow-list
    # (ERR_PNPM_CONFIG_CONFLICT_BUILT_DEPENDENCIES), in package.json or in
    # pnpm-workspace.yaml.
    removed = [key for key in BUILD_ALLOW_LIST_KEYS if settings.pop(key, None) is not None]
    workspace_yaml = install_root / "pnpm-workspace.yaml"
    if workspace_yaml.is_file():
        try:
            text = workspace_yaml.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        stripped = strip_yaml_build_allow_lists(text)
        if stripped != text:
            workspace_yaml.write_text(stripped, encoding="utf-8")
            removed.append("pnpm-workspace.yaml allow-lists")
    if legacy:
        settings["neverBuiltDependencies"] = []
        data["pnpm"] = settings
        package_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"pnpm {'.'.join(map(str, version))}: neverBuiltDependencies=[] (allow dependency builds)"
    if not removed:
        return None
    if settings:
        data["pnpm"] = settings
    else:
        data.pop("pnpm", None)
    package_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (
        f"pnpm {'.'.join(map(str, version))}: removed build allow-lists "
        f"({', '.join(removed)}) that conflict with allowing all builds"
    )


BUILD_ALLOW_LIST_KEYS = ("onlyBuiltDependencies", "onlyBuiltDependenciesFile", "ignoredBuiltDependencies")
YAML_BUILD_ALLOW_LIST = re.compile(
    r"^(?:" + "|".join(BUILD_ALLOW_LIST_KEYS) + r"):[^\n]*(?:\n(?:[ \t]+[^\n]*|[ \t]*))*",
    re.M,
)


def strip_yaml_build_allow_lists(text: str) -> str:
    """Drop top-level build allow-list keys (and their indented items) from YAML."""
    return YAML_BUILD_ALLOW_LIST.sub("", text)


# Test runners that are package binaries and must be declared to be on PATH.
NODE_PACKAGE_TEST_RUNNERS = {
    "vitest", "jest", "mocha", "ava", "tap", "playwright", "c8", "nyc",
    "tsx", "ts-node", "turbo", "nx", "lerna", "uvu", "karma", "cypress",
}
NODE_BUILTIN_TEST_RUNNERS = {"node", "bun", "deno", "npm", "pnpm", "yarn", "npx", "sh", "bash"}


def node_dependency_names(metadata: dict[str, object]) -> set[str]:
    names: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        value = metadata.get(key)
        if isinstance(value, dict):
            names.update(str(name) for name in value)
    return names


def test_script_runnable(script: str, declared: set[str]) -> bool:
    """Return False when the script's runner binary is not declared anywhere.

    For example ``"test": "vitest"`` in a package without vitest fails with
    "vitest: not found" no matter how dependencies are installed.
    """
    for token in re.split(r"[\s;&|()]+", script):
        if not token or "=" in token or token.startswith("-"):
            continue
        name = token.rsplit("/", 1)[-1]
        if name in NODE_BUILTIN_TEST_RUNNERS:
            return True
        if name in NODE_PACKAGE_TEST_RUNNERS:
            return name in declared or f"@{name}/cli" in declared or (
                name == "playwright" and "@playwright/test" in declared
            )
    return True


def discover_node_project_root(repo: Path, package: str) -> Path | None:
    candidates: list[Path] = []
    for root, dirs, files in os.walk(repo):
        base = Path(root)
        depth = len(base.relative_to(repo).parts)
        if depth >= 3:
            dirs[:] = []
        else:
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        if "package.json" in files:
            candidates.append(base)

    if not candidates:
        return None

    target_specifiers = tuple(source for source, _ in TS_SPECIFIER_RULES.get(package, ()))

    root_dependencies = node_dependency_names(load_package_json(repo))

    def score(project: Path) -> tuple[int, int, int, int]:
        metadata = load_package_json(project)
        scripts = metadata.get("scripts")
        test_script = scripts.get("test") if isinstance(scripts, dict) else None
        # A package that only mentions the framework but has no runnable test
        # script cannot be baseline-tested; a workspace root whose test script
        # runs every package can. Runnable tests therefore rank first.
        has_tests = int(
            isinstance(test_script, str)
            and bool(test_script.strip())
            and "no test specified" not in test_script
            and test_script_runnable(test_script, node_dependency_names(metadata) | root_dependencies)
        )
        dependency_names: set[str] = set()
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            value = metadata.get(key)
            if isinstance(value, dict):
                dependency_names.update(str(name) for name in value)
        target_match = int(
            any(
                dependency == source
                or (source.endswith("/") and dependency.startswith(source))
                for dependency in dependency_names
                for source in target_specifiers
            )
        )
        is_root = int(project == repo)
        depth = len(project.relative_to(repo).parts)
        return (has_tests, target_match, is_root, -depth)

    return max(candidates, key=score)


# The Prisma client is generated code, usually git-ignored and produced by a
# postinstall hook or a manual `prisma generate`.
NODE_MISSING_PRISMA_CLIENT = re.compile(
    r"Cannot find module '[^']*(?:generated/prisma|\.prisma/client)[^']*'"
    r"|@prisma/client did not initiali[sz]e yet"
)
def prisma_placeholder_url(project: Path) -> str:
    """A syntactically valid URL for the schema's datasource provider."""
    provider = "postgresql"
    for schema in sorted(project.glob("**/schema.prisma"))[:5]:
        if "node_modules" in schema.parts:
            continue
        try:
            match = re.search(r'provider\s*=\s*"(\w+)"', schema.read_text(encoding="utf-8").split("datasource", 1)[-1])
        except (OSError, UnicodeDecodeError):
            continue
        if match:
            provider = match.group(1)
            break
    return {
        "sqlite": "file:./agent_rt_prisma.db",
        "mysql": "mysql://user:pass@localhost:3306/db",
        "sqlserver": "sqlserver://localhost:1433;database=db;user=sa;password=pass",
        "mongodb": "mongodb://localhost:27017/db",
    }.get(provider, "postgresql://user:pass@localhost:5432/db")


NODE_LIFECYCLE_FAILURE = re.compile(
    r"(?:pre|post)?install(?:\$|:)?\s.*(?:Failed|failed|ELIFECYCLE)|npm error (?:code|command) .*(?:pre|post)?install",
)
NODE_LOCKFILES = (
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "bun.lock",
    "bun.lockb",
)
# Module-resolution failures pointing into build output that a workspace or
# package build script produces (dist/, build/, lib/, out/).
NODE_MISSING_BUILD_OUTPUT = re.compile(
    r"Cannot find module '[^']*/(?:dist|build|lib|out)/[^']*'"
    r"|ERR_MODULE_NOT_FOUND[^\n]*/(?:dist|build|lib|out)/"
    r"|error TS2307: Cannot find module '@[^']+'"
    # Vite/Vitest resolving a workspace package whose entry is build output.
    r"|Failed to resolve entry for package \"[^\"]+\""
    r"|Failed to resolve import \"@[^\"]+\""
)
NPM_CACHE_CORRUPTION = re.compile(r"npm error (?:code (?:EEXIST|ENOENT)|.*_cacache)", re.S)
# Runner output meaning the suite contains no tests rather than failing ones.
NODE_NO_TESTS = re.compile(r"No tests? (?:files )?found|No test files found|no tests to run", re.I)


def node_install_root(repo: Path, project: Path) -> Path:
    """Return the directory dependencies must be installed from.

    A package inside a pnpm/yarn/npm workspace has no lockfile of its own and
    may depend on siblings through ``workspace:`` specifiers, which only the
    workspace root can resolve. Walk up to the nearest workspace root or
    lockfile; fall back to the project itself.
    """
    current = project
    while True:
        if (current / "pnpm-workspace.yaml").is_file() or any(
            (current / lockfile).is_file() for lockfile in NODE_LOCKFILES
        ):
            return current
        if current != project and isinstance(load_package_json(current).get("workspaces"), (list, dict)):
            return current
        if current == repo or current.parent == current:
            return project
        current = current.parent


def node_package_manager(repo: Path) -> str:
    # The repository's declared manager, even when it is not installed: npm on
    # a pnpm/yarn/bun project fails in misleading ways (`workspace:*`,
    # `only-allow pnpm`, ERESOLVE), so a missing tool must fail visibly
    # (see check_node_tools) rather than silently switch managers.
    package_manager = str(load_package_json(repo).get("packageManager", ""))
    if (
        (repo / "bun.lock").is_file()
        or (repo / "bun.lockb").is_file()
        or package_manager.startswith("bun@")
    ):
        return "bun"
    if (
        (repo / "pnpm-lock.yaml").is_file()
        or (repo / "pnpm-workspace.yaml").is_file()
        or package_manager.startswith("pnpm@")
    ):
        return "pnpm"
    if (repo / "yarn.lock").is_file() or package_manager.startswith("yarn@"):
        return "yarn"
    return "npm"


NODE_TOOLS = ("node", "npm", "pnpm", "yarn", "bun")


def check_node_tools(env: dict[str, str]) -> None:
    """Log Node tool versions and stop when a required one is missing.

    Results depend on these versions (npm majors change install behavior), so
    they go into the run log for every run that tests Node repositories.
    """
    missing = []
    for tool in NODE_TOOLS:
        path = shutil.which(tool, path=env.get("PATH"))
        if path is None:
            missing.append(tool)
            print(f"  {tool}: not found", flush=True)
            continue
        try:
            version = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=30, env=env,
                stdin=subprocess.DEVNULL,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            version = "unknown"
        print(f"  {tool}: {version} ({path})", flush=True)
    required_missing = [tool for tool in missing if tool != "bun"]
    if required_missing:
        raise SystemExit(
            f"Required Node tools not found on PATH: {', '.join(required_missing)}. "
            "Install them for the active Node version (for example `npm i -g pnpm yarn`) "
            "or run from a shell where they are available."
        )
    if missing:
        print("  bun is missing; bun projects will fail with a runner error", flush=True)


def node_install_command(repo: Path, manager: str, *, frozen: bool = True) -> list[str]:
    if manager == "bun":
        return ["bun", "install", *(["--frozen-lockfile"] if frozen else [])]
    if manager == "pnpm":
        return ["pnpm", "install", "--frozen-lockfile" if frozen else "--no-frozen-lockfile"]
    if manager == "yarn":
        return ["yarn", "install", *(["--frozen-lockfile"] if frozen else [])]
    if frozen and (
        (repo / "package-lock.json").is_file() or (repo / "npm-shrinkwrap.json").is_file()
    ):
        return ["npm", "ci"]
    return ["npm", "install"]


def node_build_command(root: Path, manager: str) -> list[str] | None:
    """Return a best-effort command that builds the root or its workspace."""
    metadata = load_package_json(root)
    scripts = metadata.get("scripts")
    has_build = isinstance(scripts, dict) and isinstance(scripts.get("build"), str)
    if manager == "pnpm":
        if (root / "pnpm-workspace.yaml").is_file():
            return ["pnpm", "-r", "--if-present", "run", "build"]
        return ["pnpm", "run", "build"] if has_build else None
    workspaces = isinstance(metadata.get("workspaces"), (list, dict))
    if manager == "npm" and workspaces and not has_build:
        return ["npm", "run", "build", "--workspaces", "--if-present"]
    if not has_build:
        return None
    return {"yarn": ["yarn", "run", "build"], "bun": ["bun", "run", "build"]}.get(
        manager, ["npm", "run", "build"]
    )


def node_test_command(repo: Path, manager: str) -> list[str] | None:
    package = load_package_json(repo)
    scripts = package.get("scripts")
    if isinstance(scripts, dict):
        test_script = scripts.get("test")
        if isinstance(test_script, str) and test_script.strip() and "no test specified" not in test_script:
            if manager == "pnpm":
                return ["pnpm", "test"]
            if manager == "yarn":
                return ["yarn", "test"]
            if manager == "bun":
                return ["bun", "run", "test"]
            return ["npm", "test", "--"]

    deps: dict[str, object] = {}
    for key in ("dependencies", "devDependencies"):
        value = package.get(key)
        if isinstance(value, dict):
            deps.update(value)
    if "vitest" in deps:
        return ["npx", "--no-install", "vitest", "run"]
    if "jest" in deps:
        return ["npx", "--no-install", "jest"]
    if "mocha" in deps:
        return ["npx", "--no-install", "mocha"]
    return None


def install_local_agent_rt_python(
    repo: Path,
    python: Path,
    agent_rt_python: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> CommandResult:
    return run_command(
        ["uv", "pip", "install", "--python", str(python), str(agent_rt_python)],
        cwd=repo,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=env,
    )


def install_local_agent_rt_node(
    repo: Path,
    manager: str,
    agent_rt_typescript: Path,
    *,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
) -> CommandResult:
    package_path = str(agent_rt_typescript)
    if manager == "pnpm":
        command = ["pnpm", "add", "--save-dev", "--ignore-workspace-root-check", package_path]
    elif manager == "yarn":
        # Yarn Berry needs an explicit protocol for a local tarball/directory.
        command = ["yarn", "add", "--dev", f"agent-rt@file:{package_path}"]
    elif manager == "bun":
        command = ["bun", "add", "--dev", package_path]
    else:
        command = ["npm", "install", "--no-save", package_path]
    return run_command(
        command,
        cwd=repo,
        timeout=timeout,
        max_output_chars=max_output_chars,
        env=env,
    )


def iter_source_files(repo: Path, language: str) -> Iterable[Path]:
    allowed = {".py"} if language == "Python" else SOURCE_EXTENSIONS - {".py"}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        base = Path(root)
        for name in files:
            path = base / name
            if path.suffix.lower() in allowed:
                yield path


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


def rewrite_python_imports(text: str, package: str) -> tuple[str, int]:
    roots = PYTHON_IMPORT_ROOTS.get(package, ())
    if not roots:
        return text, 0

    changes = 0
    for root in sorted(roots, key=len, reverse=True):
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
                    lambda match: f"{match.group('quote')}agent_rt.{match.group('module')}{match.group('quote')}",
                    line,
                )
                changes += count
                return line
            line, count = target_pattern.subn(
                lambda match: f"{match.group('quote')}agent_rt.{match.group('module')}{match.group('quote')}",
                line,
            )
            changes += count
            return line

        text = re.sub(r"(?m)^.*$", rewrite_targets, text)
    return text, changes


AGENT_RT_TS_PACKAGE_JSON = Path(__file__).resolve().parents[2] / "typescript" / "package.json"


@functools.cache
def agent_rt_ts_exports() -> frozenset[str]:
    """Return the `agent-rt/...` specifiers the TypeScript package exports."""
    try:
        exports = json.loads(AGENT_RT_TS_PACKAGE_JSON.read_text(encoding="utf-8")).get("exports", {})
    except (OSError, ValueError):
        return frozenset()
    return frozenset("agent-rt" + key.removeprefix(".") for key in exports)


def rewrite_ts_specifier(specifier: str, package: str) -> str | None:
    # Rewrite only to specifiers agent-rt exports: the Python rewrite likewise
    # touches only explicit roots, leaving e.g. `@langchain/langgraph` or
    # `@langchain/core/utils/testing` on the upstream package.
    exported = agent_rt_ts_exports()
    for source, destination in TS_SPECIFIER_RULES.get(package, ()):
        if source.endswith("/"):
            if specifier.startswith(source):
                migrated = destination + specifier[len(source) :]
                return migrated if migrated in exported else None
        elif specifier == source:
            return destination if destination in exported else None
    return None


def rewrite_ts_imports(text: str, package: str) -> tuple[str, int]:
    if package not in TS_SPECIFIER_RULES:
        return text, 0

    # Only touch module specifier strings attached to import/export/require()
    # and to test-runner module mocks (`vi.mock("x")`, `jest.mock("x")`), which
    # must follow the import they replace.
    pattern = re.compile(
        r"(?P<head>\b(?:from\s*|import\s*\(\s*|require\s*\(\s*|import\s+|export\s+[^;]*?\s+from\s*"
        r"|(?:vi|vitest|jest)\.(?:mock|doMock|unmock|doUnmock|importActual|requireActual|importMock|requireMock)\s*\(\s*))"
        r"(?P<quote>['\"])(?P<specifier>[^'\"]+)(?P=quote)"
    )

    changes = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changes
        migrated = rewrite_ts_specifier(match.group("specifier"), package)
        if migrated is None or migrated == match.group("specifier"):
            return match.group(0)
        changes += 1
        quote = match.group("quote")
        return f"{match.group('head')}{quote}{migrated}{quote}"

    return pattern.sub(replace, text), changes


def apply_migration(repo: Path, target: RepoTarget) -> tuple[int, list[str]]:
    changed_files: list[str] = []
    total_changes = 0
    for path in iter_source_files(repo, target.language):
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if target.language == "Python":
            migrated, changes = rewrite_python_imports(original, target.package)
        else:
            migrated, changes = rewrite_ts_imports(original, target.package)
        if not changes:
            continue
        path.write_text(migrated, encoding="utf-8")
        total_changes += changes
        changed_files.append(str(path.relative_to(repo)))
    return total_changes, changed_files


def first_failed(results: Sequence[CommandResult]) -> CommandResult | None:
    return next((result for result in results if not result.ok), None)


def process_target(
    target: RepoTarget,
    *,
    work_root: Path,
    agent_rt_root: Path,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
    agent_rt_wheel: Path | None = None,
    agent_rt_npm_tarball: Path | None = None,
) -> dict[str, object]:
    repo = work_root / f"{target.index:04d}"
    cache_root = work_root / f".cache-{target.index:04d}"
    env = env.copy()
    temp_dir = cache_root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Every cache and temp location a tool may write to lives under cache_root,
    # which is deleted after the repository. Anything left under the user's
    # ~/.cache or /tmp (browser downloads from postinstall scripts, corepack
    # pnpm/yarn versions, pytest temp dirs) accumulated until the disk filled.
    env.update(
        {
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "NPM_CONFIG_CACHE": str(cache_root / "npm"),
            "YARN_CACHE_FOLDER": str(cache_root / "yarn"),
            "YARN_GLOBAL_FOLDER": str(cache_root / "yarn-global"),
            "npm_config_store_dir": str(cache_root / "pnpm"),
            "BUN_INSTALL_CACHE_DIR": str(cache_root / "bun"),
            "COREPACK_HOME": str(cache_root / "corepack"),
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "XDG_CACHE_HOME": str(cache_root / "xdg-cache"),
            "PUPPETEER_CACHE_DIR": str(cache_root / "puppeteer"),
            "PLAYWRIGHT_BROWSERS_PATH": str(cache_root / "playwright"),
            "CYPRESS_CACHE_FOLDER": str(cache_root / "cypress"),
            "ELECTRON_CACHE": str(cache_root / "electron"),
            "HF_HOME": str(cache_root / "huggingface"),
            "TMPDIR": str(temp_dir),
            "TMP": str(temp_dir),
            "TEMP": str(temp_dir),
        }
    )
    commit = ""
    before = synthetic_result("not_run", "Repository was not cloned.")
    after = synthetic_result("not_run", "Migration test was not run.")

    try:
        print("  phase: clone repository", flush=True)
        clone = clone_repo(
            target,
            repo,
            timeout=timeout,
            max_output_chars=max_output_chars,
            env=env,
        )
        if not clone.ok:
            before = command_json(clone, status="clone_failed")
            after = synthetic_result("not_run", "Clone failed.")
            return result_row(target, commit, before, after)

        commit = commit_hash(
            repo,
            timeout=timeout,
            max_output_chars=max_output_chars,
            env=env,
        )
        print(f"  commit: {commit or '<unknown>'}", flush=True)

        print("  phase: audit heavyweight dependencies", flush=True)
        heavy_dependencies = find_heavy_dependencies(repo)
        if heavy_dependencies:
            packages = sorted({item["package"] for item in heavy_dependencies})
            print(
                "  filtered: heavyweight dependencies detected: "
                + ", ".join(packages),
                flush=True,
            )
            before = synthetic_result(
                "filtered_heavy",
                "Repository skipped because heavyweight dependencies were detected.",
                heavy_dependencies=heavy_dependencies,
            )
            after = synthetic_result(
                "skipped",
                "Migration and tests were not run for a heavyweight repository.",
            )
            return result_row(target, commit, before, after)

        if target.language == "Python":
            repo_env = python_repo_env(repo, env)
            placeholders = apply_repo_env_placeholders(repo, repo_env, language="Python")
            for stale in remove_stale_build_metadata(repo):
                print(f"  removed committed build metadata: {stale}", flush=True)
            print("  phase: create Python environment and install dependencies", flush=True)
            try:
                python, setup_results = setup_python_repo(
                    repo,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=env,
                )
            except Exception as exc:  # venv creation failures are setup failures.
                before = synthetic_result("setup_failed", f"{type(exc).__name__}: {exc}")
                after = synthetic_result("not_run", "Baseline setup failed.")
                return result_row(target, commit, before, after)

            failed_setup = first_failed(setup_results)
            if failed_setup is not None:
                before = command_json(failed_setup, status="setup_failed")
                after = synthetic_result("not_run", "Baseline setup failed.")
                return result_row(target, commit, before, after)

            setup_adjustments: list[str] = []
            if any(value.startswith("sqlite+aiosqlite:") for value in placeholders.values()):
                # The async SQLite placeholder needs its driver.
                setup_env = {**repo_env, "PYTHONSAFEPATH": "1"}
                setup_env.pop("PYTHONPATH", None)
                run_command(
                    ["uv", "pip", "install", "--python", str(python), "aiosqlite"],
                    cwd=repo,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=setup_env,
                )
                setup_adjustments.append("installed aiosqlite for async SQLite placeholder")
            test_command = python_test_command(repo, python)
            print("  phase: initial tests", flush=True)
            baseline = run_command(
                test_command,
                cwd=repo,
                timeout=timeout,
                max_output_chars=max_output_chars,
                env=repo_env,
            )
            if placeholders:
                setup_adjustments.insert(0, "placeholder env: " + ", ".join(sorted(placeholders)))
            # Harness-level failures that a pytest option or a package swap
            # works around. The adjusted command is reused after migration.
            attempted_packages: set[str] = set()
            for _ in range(5):
                if baseline.ok:
                    break
                adjustment = ""
                missing = [
                    package
                    for package in missing_dependencies_from_output(baseline.output, repo)
                    if package not in attempted_packages
                ]
                duplicate = PYTEST_DUPLICATE_PLUGIN.search(baseline.output)
                if missing:
                    # Undeclared imports/fixtures that the run actually needed.
                    attempted_packages.update(missing)
                    install_env = {**repo_env, "PYTHONSAFEPATH": "1"}
                    install_env.pop("PYTHONPATH", None)
                    installed = run_command(
                        ["uv", "pip", "install", "--python", str(python), *missing],
                        cwd=repo,
                        timeout=timeout,
                        max_output_chars=max_output_chars,
                        env=install_env,
                    )
                    if not installed.ok:
                        for package in missing:
                            run_command(
                                ["uv", "pip", "install", "--python", str(python), package],
                                cwd=repo,
                                timeout=timeout,
                                max_output_chars=max_output_chars,
                                env=install_env,
                            )
                    if "pytest-asyncio" in missing and "asyncio_mode=auto" not in test_command:
                        # Unmarked async tests need auto mode as well.
                        test_command = [*test_command, "-o", "asyncio_mode=auto"]
                    adjustment = "installed undeclared " + ", ".join(missing)
                elif duplicate and f"no:{duplicate.group(1)}" not in test_command:
                    # Two installed distributions register the same plugin
                    # module under different entry-point names.
                    test_command = [*test_command, "-p", f"no:{duplicate.group(1)}"]
                    adjustment = f"pytest -p no:{duplicate.group(1)} (plugin registered twice)"
                elif (
                    PYTEST_IMPORT_MISMATCH.search(baseline.output)
                    and "--import-mode=importlib" not in test_command
                ):
                    # Same test-file basename in several directories without
                    # __init__.py; rootdir-based imports collide.
                    test_command = [*test_command, "--import-mode=importlib"]
                    adjustment = "pytest --import-mode=importlib (duplicate test module names)"
                elif PYTEST_CLOSED_CAPTURE.search(baseline.output) and "-s" not in test_command:
                    # Code that rewraps sys.stdout at import closes pytest's
                    # capture file; run without output capturing.
                    test_command = [*test_command, "-s"]
                    adjustment = "pytest -s (code rewraps sys.stdout)"
                else:
                    for pattern, old_package, new_package in PACKAGE_RENAMES:
                        if pattern.search(baseline.output) and new_package not in " ".join(setup_adjustments):
                            swap_env = {**repo_env, "PYTHONSAFEPATH": "1"}
                            swap_env.pop("PYTHONPATH", None)
                            for command in (
                                ["uv", "pip", "uninstall", "--python", str(python), old_package],
                                ["uv", "pip", "install", "--python", str(python), new_package],
                            ):
                                run_command(
                                    command,
                                    cwd=repo,
                                    timeout=timeout,
                                    max_output_chars=max_output_chars,
                                    env=swap_env,
                                )
                            adjustment = f"replaced renamed {old_package} with {new_package}"
                            break
                    for pattern, requirement in LEGACY_MODULE_PINS:
                        if adjustment:
                            break
                        if pattern.search(baseline.output) and requirement not in " ".join(setup_adjustments):
                            pin_env = {**repo_env, "PYTHONSAFEPATH": "1"}
                            pin_env.pop("PYTHONPATH", None)
                            pinned = run_command(
                                ["uv", "pip", "install", "--python", str(python), requirement],
                                cwd=repo,
                                timeout=timeout,
                                max_output_chars=max_output_chars,
                                env=pin_env,
                            )
                            if pinned.ok:
                                adjustment = f"{requirement} for a removed legacy module"
                if not adjustment:
                    break
                print(f"  initial tests failed; retrying with {adjustment}", flush=True)
                setup_adjustments.append(adjustment)
                baseline = run_command(
                    test_command,
                    cwd=repo,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=repo_env,
                )
            if not baseline.ok and LEGACY_LANGCHAIN_ERROR.search(baseline.output):
                print(
                    "  initial tests use pre-1.0 LangChain imports; "
                    "installing LangChain 0.3 and retrying",
                    flush=True,
                )
                downgrade = downgrade_legacy_langchain(
                    repo,
                    python,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=env,
                )
                if downgrade.ok:
                    setup_adjustments.append("langchain<1 for legacy import paths")
                    print("  phase: initial tests (retry)", flush=True)
                    baseline = run_command(
                        test_command,
                        cwd=repo,
                        timeout=timeout,
                        max_output_chars=max_output_chars,
                        env=repo_env,
                    )
            if not baseline.ok and LEGACY_LLAMA_INDEX_ERROR.search(baseline.output):
                print(
                    "  initial tests use pre-0.10 LlamaIndex imports; "
                    "installing LlamaIndex 0.9 and retrying",
                    flush=True,
                )
                downgrade = downgrade_legacy_llama_index(
                    repo,
                    python,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=env,
                )
                if downgrade.ok:
                    setup_adjustments.append("llama-index<0.10 for legacy import paths")
                    print("  phase: initial tests (retry)", flush=True)
                    baseline = run_command(
                        test_command,
                        cwd=repo,
                        timeout=timeout,
                        max_output_chars=max_output_chars,
                        env=repo_env,
                    )
            # A legacy downgrade can surface a removed module (old LlamaIndex
            # imports pkg_resources), so check the pins once more.
            for pattern, requirement in LEGACY_MODULE_PINS:
                if baseline.ok or not pattern.search(baseline.output):
                    continue
                if requirement in " ".join(setup_adjustments):
                    continue
                pin_env = {**repo_env, "PYTHONSAFEPATH": "1"}
                pin_env.pop("PYTHONPATH", None)
                pinned = run_command(
                    ["uv", "pip", "install", "--python", str(python), requirement],
                    cwd=repo,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=pin_env,
                )
                if pinned.ok:
                    setup_adjustments.append(f"{requirement} for a removed legacy module")
                    print(f"  initial tests failed; retrying with {requirement}", flush=True)
                    baseline = run_command(
                        test_command,
                        cwd=repo,
                        timeout=timeout,
                        max_output_chars=max_output_chars,
                        env=repo_env,
                    )
            before_payload = json.loads(command_json(baseline))
            if baseline.exit_code == PYTEST_NO_TESTS_EXIT_CODE and not baseline.timed_out:
                before_payload["status"] = "no_tests"
            if setup_adjustments:
                before_payload["setup_adjustments"] = setup_adjustments
            before = json.dumps(before_payload, ensure_ascii=False, separators=(",", ":"))
            if before_payload["status"] == "no_tests":
                print("  pytest collected no tests; skipping migration", flush=True)
                after = synthetic_result(
                    "skipped",
                    "No tests were collected, so migration was not attempted.",
                )
                return result_row(target, commit, before, after)
            if not baseline.ok:
                print("  initial tests failed; skipping migration and post-migration tests", flush=True)
                after = synthetic_result(
                    "skipped",
                    "Initial tests failed, so migration was not attempted.",
                )
                return result_row(target, commit, before, after)

            print("  phase: apply migration rewrites", flush=True)
            changes, changed_files = apply_migration(repo, target)
            print(f"  migration changes: {changes}", flush=True)
            for changed_file in changed_files:
                print(f"    changed: {changed_file}", flush=True)
            if changes == 0:
                after = synthetic_result(
                    "migration_no_changes",
                    "No supported target import was found to rewrite."
                )
                return result_row(target, commit, before, after)

            print("  phase: install local Agent RT", flush=True)
            agent_install = install_local_agent_rt_python(
                repo,
                python,
                agent_rt_wheel or agent_rt_root / "python",
                timeout=timeout,
                max_output_chars=max_output_chars,
                env=repo_env,
            )
            if not agent_install.ok:
                after = command_json(agent_install, status="agent_rt_install_failed")
                return result_row(target, commit, before, after)

            print("  phase: tests after migration", flush=True)
            migrated = run_command(
                test_command,
                cwd=repo,
                timeout=timeout,
                max_output_chars=max_output_chars,
                env=repo_env,
            )
            after_payload = json.loads(command_json(migrated))
            after_payload["migration_changes"] = changes
            after_payload["changed_files"] = changed_files
            after = json.dumps(after_payload, ensure_ascii=False, separators=(",", ":"))

        elif target.language == "TS/JS":
            node_env = env.copy()
            placeholders = apply_repo_env_placeholders(repo, node_env, language="TS/JS")
            setup_adjustments: list[str] = []
            if placeholders:
                setup_adjustments.append("placeholder env: " + ", ".join(sorted(placeholders)))
            print("  phase: install dependencies", flush=True)
            node_root = discover_node_project_root(repo, target.package)
            if node_root is None:
                before = synthetic_result("setup_failed", "No package.json found.")
                after = synthetic_result("not_run", "Baseline setup failed.")
                return result_row(target, commit, before, after)
            if node_root != repo:
                print(
                    f"    using nested Node project: {node_root.relative_to(repo)}",
                    flush=True,
                )

            install_root = node_install_root(repo, node_root)
            if install_root != node_root:
                print(
                    "    installing from workspace root: "
                    + (str(install_root.relative_to(repo)) if install_root != repo else "."),
                    flush=True,
                )
            manager = node_package_manager(install_root)
            if manager == "pnpm":
                legacy_builds = allow_legacy_pnpm_builds(install_root, node_env)
                if legacy_builds:
                    print(f"    {legacy_builds}", flush=True)
                    setup_adjustments.append(legacy_builds)

            def run_node(command: list[str], *, cwd: Path, command_env: dict[str, str] | None = None) -> CommandResult:
                return run_command(
                    command,
                    cwd=cwd,
                    timeout=timeout,
                    max_output_chars=max_output_chars,
                    env=command_env or node_env,
                )

            # CI=1 makes Yarn Berry refuse any lockfile change (YN0028); the
            # non-frozen fallback must explicitly allow it.
            relaxed_env = {**node_env, "YARN_ENABLE_IMMUTABLE_INSTALLS": "false"}
            install = run_node(node_install_command(install_root, manager), cwd=install_root)
            unfrozen = node_install_command(install_root, manager, frozen=False)

            def clear_corrupted_npm_cache(result: CommandResult) -> bool:
                # npm occasionally corrupts its content cache mid-install
                # (EEXIST/ENOENT renames under _cacache); a clean cache fixes it.
                if manager != "npm" or not NPM_CACHE_CORRUPTION.search(result.output):
                    return False
                print("    npm cache corrupted during install; clearing it and retrying", flush=True)
                shutil.rmtree(node_env.get("NPM_CONFIG_CACHE", ""), ignore_errors=True)
                return True

            if not install.ok and clear_corrupted_npm_cache(install):
                install = run_node(node_install_command(install_root, manager), cwd=install_root)
            if not install.ok and (
                unfrozen != node_install_command(install_root, manager) or manager == "yarn"
            ):
                # Lockfiles are frequently out of sync with package.json in
                # these repositories; fall back to a normal resolving install.
                print("    frozen install failed; retrying without a frozen lockfile", flush=True)
                install = run_node(unfrozen, cwd=install_root, command_env=relaxed_env)
                if not install.ok and clear_corrupted_npm_cache(install):
                    install = run_node(unfrozen, cwd=install_root, command_env=relaxed_env)
                if install.ok:
                    setup_adjustments.append("non-frozen lockfile install")
            if not install.ok and manager == "pnpm" and "ERR_PNPM_IGNORED_BUILDS" in install.output:
                # pnpm 10 strictDepBuilds fails when dependency build scripts
                # are not allow-listed; installing without them is enough here.
                print("    pnpm refused unapproved dependency builds; retrying without strict builds", flush=True)
                install = run_node(
                    [*unfrozen, "--config.strict-dep-builds=false"],
                    cwd=install_root,
                    command_env=relaxed_env,
                )
                if install.ok:
                    setup_adjustments.append("pnpm strict-dep-builds=false")
            if not install.ok and NODE_LIFECYCLE_FAILURE.search(install.output):
                # Some repositories gate installs behind their own wrapper
                # (for example a preinstall that refuses direct pnpm use).
                print("    install lifecycle script failed; retrying with --ignore-scripts", flush=True)
                install = run_node([*unfrozen, "--ignore-scripts"], cwd=install_root, command_env=relaxed_env)
                if install.ok:
                    setup_adjustments.append("install --ignore-scripts")
            if not install.ok and manager == "npm" and "ERESOLVE" in install.output:
                # npm 7+ rejects conflicting peer dependency ranges that other
                # managers (and older npm) accept. The setting stays on for
                # every later npm command, including the Agent RT install,
                # which would otherwise hit the same conflict.
                print("    npm peer dependency conflict (ERESOLVE); retrying with legacy peer deps", flush=True)
                for command_env in (node_env, relaxed_env):
                    command_env["npm_config_legacy_peer_deps"] = "true"
                install = run_node(unfrozen, cwd=install_root, command_env=relaxed_env)
                if install.ok:
                    setup_adjustments.append("npm legacy-peer-deps (ERESOLVE)")
            if not install.ok:
                before = command_json(install, status="setup_failed")
                after = synthetic_result("not_run", "Baseline setup failed.")
                return result_row(target, commit, before, after)

            if manager == "pnpm" and install_root != node_root:
                package_name = load_package_json(node_root).get("name")
                if isinstance(package_name, str) and package_name:
                    # Workspace siblings are consumed from their build output
                    # (dist/ or emitted .d.ts); build every dependency of the
                    # selected package first. Best-effort: the tests decide.
                    print("    building workspace dependencies", flush=True)
                    run_node(
                        ["pnpm", "--filter", f"{package_name}^...", "run", "--if-present", "build"],
                        cwd=install_root,
                    )

            test_command = node_test_command(node_root, manager)
            if test_command is None:
                before = synthetic_result("no_tests", "No supported test command detected.")
                after = synthetic_result("skipped", "No baseline test command.")
                return result_row(target, commit, before, after)

            print("  phase: initial tests", flush=True)
            baseline = run_node(test_command, cwd=node_root)
            if (
                not baseline.ok
                and NODE_MISSING_PRISMA_CLIENT.search(baseline.output)
                and any(
                    (root / "node_modules" / ".bin" / "prisma").exists()
                    for root in (node_root, install_root)
                )
            ):
                print("  initial tests need the generated Prisma client; generating and retrying", flush=True)
                # --no-install: only the repository's own Prisma CLI, never a download.
                # prisma.config.ts often resolves env("DATABASE_URL") just to load;
                # generate never connects, so a dummy URL is enough.
                generate_env = {**node_env}
                generate_env.setdefault("DATABASE_URL", prisma_placeholder_url(node_root))
                generated = run_node(
                    ["npx", "--no-install", "prisma", "generate"],
                    cwd=node_root,
                    command_env=generate_env,
                )
                setup_adjustments.append("prisma generate" + ("" if generated.ok else " (failed)"))
                baseline = run_node(test_command, cwd=node_root)
            if not baseline.ok and NODE_MISSING_BUILD_OUTPUT.search(baseline.output):
                build = node_build_command(install_root, manager)
                if build is not None:
                    print("  initial tests need build output; building and retrying", flush=True)
                    built = run_node(build, cwd=install_root)
                    setup_adjustments.append("built packages before tests" + ("" if built.ok else " (build failed)"))
                    baseline = run_node(test_command, cwd=node_root)
            before_payload = json.loads(command_json(baseline))
            if not baseline.ok and not baseline.timed_out and NODE_NO_TESTS.search(baseline.output):
                before_payload["status"] = "no_tests"
            if setup_adjustments:
                before_payload["setup_adjustments"] = setup_adjustments
            before = json.dumps(before_payload, ensure_ascii=False, separators=(",", ":"))
            if before_payload["status"] == "no_tests":
                print("  test runner found no tests; skipping migration", flush=True)
                after = synthetic_result(
                    "skipped",
                    "No tests were found, so migration was not attempted.",
                )
                return result_row(target, commit, before, after)
            if not baseline.ok:
                print("  initial tests failed; skipping migration and post-migration tests", flush=True)
                after = synthetic_result(
                    "skipped",
                    "Initial tests failed, so migration was not attempted.",
                )
                return result_row(target, commit, before, after)

            if target.package not in TS_SPECIFIER_RULES:
                after = synthetic_result(
                    "unsupported",
                    f"Agent RT has no TypeScript/JavaScript migration surface for {target.package}.",
                )
                return result_row(target, commit, before, after)

            print("  phase: apply migration rewrites", flush=True)
            changes, changed_files = apply_migration(repo, target)
            print(f"  migration changes: {changes}", flush=True)
            for changed_file in changed_files:
                print(f"    changed: {changed_file}", flush=True)
            if changes == 0:
                after = synthetic_result(
                    "migration_no_changes",
                    "No supported target package specifier was found to rewrite.",
                )
                return result_row(target, commit, before, after)

            print("  phase: install local Agent RT", flush=True)
            agent_install = install_local_agent_rt_node(
                node_root,
                manager,
                agent_rt_npm_tarball or agent_rt_root / "typescript",
                timeout=timeout,
                max_output_chars=max_output_chars,
                env=relaxed_env,
            )
            if not agent_install.ok:
                after = command_json(agent_install, status="agent_rt_install_failed")
                return result_row(target, commit, before, after)

            print("  phase: tests after migration", flush=True)
            migrated = run_node(test_command, cwd=node_root)
            after_payload = json.loads(command_json(migrated))
            after_payload["migration_changes"] = changes
            after_payload["changed_files"] = changed_files
            after = json.dumps(after_payload, ensure_ascii=False, separators=(",", ":"))
        else:
            before = synthetic_result("unsupported", f"Unsupported language: {target.language}")
            after = before

        return result_row(target, commit, before, after)
    finally:
        print("  phase: cleanup repository and package caches", flush=True)
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(cache_root, ignore_errors=True)


def result_row(
    target: RepoTarget,
    commit: str,
    before: str,
    after: str,
) -> dict[str, object]:
    return {
        "Index": target.index,
        "Package": target.package,
        "Language": target.language,
        "PLIndex": target.pl_index,
        "RepoURL": target.repo_url,
        "CommitHash": commit,
        "TestStatusBeforeMigration": result_status(before),
        "TestStatusAfterMigration": result_status(after),
        "TestResultsBeforeMigration": before,
        "TestResultsAfterMigration": after,
    }


def build_agent_rt_wheel(
    package_dir: Path,
    work_root: Path,
    env: dict[str, str],
    *,
    timeout: int,
) -> Path | None:
    """Build the local Agent RT wheel once for the whole run.

    Installing from the source directory makes every worker run the
    setuptools build in that shared directory (build/, *.egg-info) at the
    same time; concurrent builds there have hung installs for the full
    command timeout. Workers install this one wheel instead.
    """
    dist_dir = work_root / "agent-rt-dist"
    print("phase: build local Agent RT wheel", flush=True)
    result = run_command(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir), str(package_dir)],
        cwd=package_dir,
        timeout=timeout,
        max_output_chars=4000,
        env={**env, "UV_CACHE_DIR": str(work_root / ".cache-agent-rt-build")},
        show_output=False,
    )
    wheels = sorted(dist_dir.glob("agent_rt-*.whl")) if result.ok else []
    if not wheels:
        print("  wheel build failed; workers will install from the source directory", flush=True)
        return None
    return wheels[-1]


# Compat entry points that must be in the packed TypeScript package.
AGENT_RT_NPM_REQUIRED = ("package/dist/index.js", "package/dist/ext/compat/openai.js")


def build_agent_rt_npm_tarball(
    package_dir: Path,
    work_root: Path,
    env: dict[str, str],
    *,
    timeout: int,
) -> Path | None:
    """Build and pack the local Agent RT TypeScript package once for the run.

    Installing the source directory depends on `dist/` at that moment and on
    each package manager's handling of local directories (symlinks, install
    links, `files`); migrated repositories then failed to resolve
    `agent-rt/...`. Workers install this one verified tarball instead.
    """
    out_dir = work_root / "agent-rt-npm"
    out_dir.mkdir(parents=True, exist_ok=True)
    npm_env = {**env, "npm_config_cache": str(work_root / ".cache-agent-rt-npm")}
    print("phase: build and pack local Agent RT TypeScript package", flush=True)
    commands = [["npm", "run", "build"], ["npm", "pack", "--pack-destination", str(out_dir)]]
    if not (package_dir / "node_modules" / ".bin" / "tsc").exists():
        # A fresh checkout has no dev dependencies, so `tsc` is missing.
        has_lock = (package_dir / "package-lock.json").is_file()
        commands.insert(0, ["npm", "ci" if has_lock else "install", "--no-audit", "--no-fund"])
    for command in commands:
        result = run_command(
            command, cwd=package_dir, timeout=timeout, max_output_chars=4000, env=npm_env, show_output=False
        )
        if result.ok:
            continue
        if (package_dir / "dist" / "index.js").is_file():
            print(f"  {' '.join(command[:2])} failed; workers will install from the source directory", flush=True)
            return None
        # Without dist/ every migrated TS/JS repository fails to resolve
        # agent-rt/..., which would be recorded as compatibility failures.
        raise SystemExit(
            f"Cannot build the Agent RT TypeScript package ({' '.join(command)} failed in {package_dir}):\n"
            + result.output[-2000:]
        )
    tarballs = sorted(out_dir.glob("agent-rt-*.tgz"))
    if not tarballs:
        print("  npm pack produced no tarball; workers will install from the source directory", flush=True)
        return None
    with tarfile.open(tarballs[-1]) as archive:
        names = set(archive.getnames())
    missing = [name for name in AGENT_RT_NPM_REQUIRED if name not in names]
    if missing:
        print(f"  packed Agent RT is missing {missing}; workers will install from the source directory", flush=True)
        return None
    return tarballs[-1]


def process_target_worker(
    target: RepoTarget,
    *,
    work_root: Path,
    agent_rt_root: Path,
    timeout: int,
    max_output_chars: int,
    env: dict[str, str],
    repo_log: Path | None = None,
    agent_rt_wheel: Path | None = None,
    agent_rt_npm_tarball: Path | None = None,
) -> dict[str, object]:
    repo_handle = open_log(repo_log) if repo_log else None
    streams = [stream for stream in (sys.stdout, sys.stderr) if isinstance(stream, TeeStream)]
    for stream in streams:
        if repo_handle:
            stream.log_files.append(repo_handle)
    try:
        print(
            f"[{target.index}] {target.package} / {target.language} / {target.repo_url}",
            flush=True,
        )
        if repo_log:
            print(f"  log: {repo_log}", flush=True)
        try:
            row = process_target(
                target,
                work_root=work_root,
                agent_rt_root=agent_rt_root,
                timeout=timeout,
                max_output_chars=max_output_chars,
                env=env,
                agent_rt_wheel=agent_rt_wheel,
                agent_rt_npm_tarball=agent_rt_npm_tarball,
            )
        except Exception as exc:
            print(f"  runner error: {type(exc).__name__}: {exc}", flush=True)
            row = result_row(
                target,
                "",
                synthetic_result("runner_error", f"{type(exc).__name__}: {exc}"),
                synthetic_result("not_run", "Runner error before migration test completed."),
            )
        print(
            f"  result: {row['TestStatusBeforeMigration']} / {row['TestStatusAfterMigration']}",
            flush=True,
        )
        return row
    finally:
        for stream in streams:
            if repo_handle in stream.log_files:
                stream.log_files.remove(repo_handle)
        if repo_handle:
            repo_handle.close()


WORK_ROOT_PREFIX = "agent-rt-migration-"
WORK_ROOT_OWNER = ".owner-pid"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_work_roots(parent: Path) -> None:
    """Delete work roots left by runs that were killed (SIGKILL, crash, reboot).

    A live run's work root holds its PID; one without a live owner (or with
    no owner file that is over a day old) is garbage.
    """
    for candidate in parent.glob(f"{WORK_ROOT_PREFIX}*"):
        if not candidate.is_dir():
            continue
        try:
            owner = int((candidate / WORK_ROOT_OWNER).read_text().strip())
            stale = not process_alive(owner)
        except (OSError, ValueError):
            try:
                stale = time.time() - candidate.stat().st_mtime > 86400
            except OSError:
                continue
        if stale:
            print(f"removing stale work directory from a killed run: {candidate}", flush=True)
            shutil.rmtree(candidate, ignore_errors=True)


def interrupt_on_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def low_disk_message(paths: Sequence[Path], minimum: float) -> str | None:
    for path in paths:
        if free_gb(path) < minimum:
            return (
                f"only {free_gb(path):.1f} GB free under {path} (minimum {minimum:g} GB, "
                "--min-free-gb); stopping before the next repository. Free space, then "
                "rerun the same command with --resume."
            )
    return None


def rerun_progress_path(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}.rerun-progress.json")


def rerun_selection(args: argparse.Namespace) -> dict[str, object] | None:
    """Identity of an explicit rerun selection, or None for a normal run."""
    if args.indices is None and not args.pass_fail and not args.retry_failed:
        return None
    return {
        "indices": sorted(args.indices) if args.indices is not None else None,
        "pass_fail": args.pass_fail,
        "retry_failed": args.retry_failed,
        "start_index": args.start_index,
        "limit": args.limit,
    }


def load_rerun_progress(path: Path, selection: dict[str, object]) -> set[str]:
    """Repository URL keys an interrupted run of this selection already finished."""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read rerun progress {path}: {exc}") from exc
    if state.get("selection") != selection:
        raise SystemExit(
            f"{path} belongs to a different selection; finish that run with the same "
            "flags, or delete the file (or drop --resume) to start this one fresh."
        )
    return set(state.get("done", []))


def save_rerun_progress(path: Path, selection: dict[str, object], done: set[str]) -> None:
    # Written only by the parent, after the result row; atomic replace so an
    # interrupt never leaves a partial file.
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps({"selection": selection, "done": sorted(done)}, indent=1),
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> int:
    args = parse_args()
    agent_rt_root = Path(__file__).resolve().parents[2]
    repo_list = args.repo_list.resolve()
    results_path = args.results.resolve()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.pass_fail and args.restart:
        raise SystemExit("--pass-fail reruns existing results and cannot be combined with --restart")
    selection = rerun_selection(args)
    if args.resume and selection is None:
        raise SystemExit("--resume applies to --indices, --pass-fail, or --retry-failed runs")
    if args.resume and args.restart:
        raise SystemExit("--resume cannot be combined with --restart")
    if not repo_list.is_file():
        raise SystemExit(f"Repository list not found: {repo_list}")
    if not (agent_rt_root / "python" / "pyproject.toml").is_file():
        raise SystemExit(f"Agent RT Python package not found under {agent_rt_root / 'python'}")
    if not (agent_rt_root / "typescript" / "package.json").is_file():
        raise SystemExit(f"Agent RT TypeScript package not found under {agent_rt_root / 'typescript'}")

    # Before anything is written: reclaim work directories of killed runs, and
    # refuse to start on a nearly full disk instead of crashing mid-write.
    remove_stale_work_roots(Path(tempfile.gettempdir()))
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    low_disk = low_disk_message([Path(tempfile.gettempdir()), log_dir], args.min_free_gb)
    if low_disk:
        raise SystemExit(low_disk)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_log = log_dir / f"run-{run_stamp}.log"
    install_log_tee(run_log)
    print(f"run log: {run_log}", flush=True)
    print(f"per-repository logs: {log_dir / run_stamp}", flush=True)

    all_targets = load_targets(repo_list)
    targets = [target for target in all_targets if target.index >= args.start_index]
    if args.indices is not None:
        targets = [target for target in targets if target.index in args.indices]

    completed = prepare_results(results_path, restart=args.restart, targets=all_targets)
    if args.pass_fail:
        # Only PASS / FAIL rows run; pending and other completed rows are left alone.
        pass_fail = pass_fail_targets(results_path)
        targets = [
            target
            for target in targets
            if (repo_url_key(target.repo_url), target.package, target.language) in pass_fail
        ]
    if args.limit is not None:
        targets = targets[: args.limit]
    retryable = retryable_repo_urls(results_path) if args.retry_failed else set()
    if args.pass_fail or args.indices is not None:
        # Explicitly selected rows are rerun even when already completed.
        retryable |= {repo_url_key(target.repo_url) for target in targets}
    progress_path = rerun_progress_path(results_path)
    finished: set[str] = set()
    if selection is not None:
        if args.resume:
            finished = load_rerun_progress(progress_path, selection)
        save_rerun_progress(progress_path, selection, finished)
    pending: list[RepoTarget] = []
    for target in targets:
        url_key = repo_url_key(target.repo_url)
        if url_key in finished:
            print(f"[{target.index}] skip, already rerun before --resume {target.repo_url}", flush=True)
            continue
        if url_key in completed and url_key not in retryable:
            print(f"[{target.index}] skip completed {target.repo_url}", flush=True)
            print("\n" + "-" * 100 + "\n", flush=True)
        else:
            if url_key in retryable:
                print(f"[{target.index}] retry failed {target.repo_url}", flush=True)
            pending.append(target)

    env = sanitized_env()
    if any(target.language != "Python" for target in pending):
        # Before the mock server starts, so a missing tool leaves nothing running.
        print("phase: check Node tools", flush=True)
        check_node_tools(env)
    aimock_process: subprocess.Popen[bytes] | None = None
    if pending:
        script_dir = Path(__file__).resolve().parent
        aimock_process, aimock_url = start_aimock_server(script_dir, env)
        env = apply_mock_provider_env(env, aimock_url)

    created_temp_root = args.work_dir is None
    work_root = (
        Path(tempfile.mkdtemp(prefix=WORK_ROOT_PREFIX))
        if created_temp_root
        else args.work_dir.resolve()
    )
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / WORK_ROOT_OWNER).write_text(str(os.getpid()))
    # SIGTERM (cloud stop, `kill`) runs the cleanup in `finally` like Ctrl+C.
    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    disk_paths = [work_root, log_dir]

    agent_rt_wheel: Path | None = None
    agent_rt_npm_tarball: Path | None = None
    try:
        # Inside try: a failed package build still stops the mock server and
        # removes the work root.
        if any(target.language == "Python" for target in pending):
            agent_rt_wheel = build_agent_rt_wheel(agent_rt_root / "python", work_root, env, timeout=args.timeout)
        if any(target.language != "Python" for target in pending):
            agent_rt_npm_tarball = build_agent_rt_npm_tarball(
                agent_rt_root / "typescript", work_root, env, timeout=args.timeout
            )

        if pending:
            execution = "current process" if args.workers == 1 else f"{args.workers} worker processes"
            print(
                f"processing {len(pending)} repositories in {execution}",
                flush=True,
            )

        if args.workers == 1:
            for target in pending:
                low_disk = low_disk_message(disk_paths, args.min_free_gb)
                if low_disk:
                    print(low_disk, flush=True)
                    return 1
                row = process_target_worker(
                    target,
                    work_root=work_root,
                    agent_rt_root=agent_rt_root,
                    timeout=args.timeout,
                    max_output_chars=args.max_output_chars,
                    env=env,
                    repo_log=repo_log_path(log_dir, run_stamp, target),
                    agent_rt_wheel=agent_rt_wheel,
                    agent_rt_npm_tarball=agent_rt_npm_tarball,
                )
                upsert_result(results_path, row)
                if selection is not None:
                    finished.add(repo_url_key(target.repo_url))
                    save_rerun_progress(progress_path, selection, finished)
                print(f"  result written: {results_path}", flush=True)
                print("\n" + "-" * 100 + "\n", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=install_log_tee,
                initargs=(run_log,),
            ) as executor:
                # Submit at most `workers` repositories at a time so free disk
                # space is checked before each one starts.
                queue = list(pending)
                futures: dict[concurrent.futures.Future, RepoTarget] = {}
                stopped_low_disk: str | None = None

                def submit_next() -> None:
                    nonlocal stopped_low_disk
                    while queue and len(futures) < args.workers and stopped_low_disk is None:
                        stopped_low_disk = low_disk_message(disk_paths, args.min_free_gb)
                        if stopped_low_disk:
                            print(stopped_low_disk, flush=True)
                            return
                        target = queue.pop(0)
                        futures[
                            executor.submit(
                                process_target_worker,
                                target,
                                work_root=work_root,
                                agent_rt_root=agent_rt_root,
                                timeout=args.timeout,
                                max_output_chars=args.max_output_chars,
                                env=env,
                                repo_log=repo_log_path(log_dir, run_stamp, target),
                                agent_rt_wheel=agent_rt_wheel,
                                agent_rt_npm_tarball=agent_rt_npm_tarball,
                            )
                        ] = target

                submit_next()
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    future = next(iter(done))
                    target = futures.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        print(
                            f"[{target.index}] {target.package} / {target.language} / "
                            f"{target.repo_url}\n"
                            f"  worker process error: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        row = result_row(
                            target,
                            "",
                            synthetic_result("runner_error", f"{type(exc).__name__}: {exc}"),
                            synthetic_result(
                                "not_run",
                                "Worker process failed before migration test completed.",
                            ),
                        )

                    upsert_result(results_path, row)
                    if selection is not None:
                        finished.add(repo_url_key(target.repo_url))
                        save_rerun_progress(progress_path, selection, finished)
                    print(
                        f"[{target.index}] result written: {row['TestStatusBeforeMigration']} / "
                        f"{row['TestStatusAfterMigration']}  log: "
                        f"{repo_log_path(log_dir, run_stamp, target)}",
                        flush=True,
                    )
                    print("\n" + "-" * 100 + "\n", flush=True)
                    submit_next()
                if stopped_low_disk:
                    return 1
    finally:
        stop_aimock_server(aimock_process)
        if created_temp_root:
            shutil.rmtree(work_root, ignore_errors=True)

    if selection is not None:
        # The whole selection finished; a later --resume starts fresh.
        progress_path.unlink(missing_ok=True)
        print(f"rerun selection complete; removed {progress_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

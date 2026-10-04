from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_rt import ToolCall, ToolDefinition, ToolRegistry
from ext.operations import BenchmarkCase, BenchmarkRunner

FULL_TESTS = (
    "tests/test_model_routing.py",
    "tests/test_agent_loop.py",
    "tests/test_edge_cases.py",
    "tests/test_evaluation.py",
    "tests/test_management_extensions.py",
    "tests/test_deployment.py",
    "tests/test_optimization.py",
    "tests/test_operations.py",
)
DEFAULT_BENCHMARK = "harmactions"


def _run_tests(files: tuple[str, ...]) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *files]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


async def _performance_benchmark(iterations: int) -> dict[str, Any]:
    registry = ToolRegistry()

    async def echo_handler(arguments, _token):
        return arguments["value"]

    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo a value.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        ),
        handler=echo_handler,
    )
    call = ToolCall(id="benchmark", name="echo", arguments={"value": "ok"})

    async def operation() -> None:
        result = await registry.execute(call)
        if result != "ok":
            raise RuntimeError("tool-dispatch benchmark returned an unexpected result")

    result = await BenchmarkRunner().run(
        BenchmarkCase(
            name="tool_registry_dispatch",
            category="performance",
            operation=operation,
            iterations=iterations,
            warmup_iterations=1,
        )
    )
    return {
        "benchmark": "performance",
        "name": result.name,
        "category": result.category,
        "iterations": result.iterations,
        "mean_ms": result.mean_seconds * 1000,
        "min_ms": result.min_seconds * 1000,
        "max_ms": result.max_seconds * 1000,
    }


def _harmactions_benchmark(
    *,
    k: int,
    offset: int,
    limit: int | None,
    cache_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    if not os.getenv("OPENAI_MODEL", "").strip():
        raise RuntimeError(
            "HarmActionsEval requires OPENAI_MODEL and OpenAI-compatible provider credentials. "
            "Use --benchmark performance for the offline microbenchmark."
        )

    try:
        from agent_action_guard import harmactionseval
    except ImportError as exc:
        raise RuntimeError(
            "HarmActionsEval dependencies are unavailable; install "
            '"agent-action-guard[harmactionseval]>=1.2.1,<2".'
        ) from exc

    harmactionseval.load_cache(cache_path)
    with contextlib.redirect_stdout(io.StringIO()):
        summary = harmactionseval.evaluate(
            k=k,
            offset=offset,
            limit=limit,
            output_path=output_path,
        )

    return {
        "benchmark": "harmactions",
        "name": "HarmActionsEval",
        "model": summary["model"],
        "k": summary["k"],
        "offset": summary["offset"],
        "limit": summary["limit"],
        "total": summary["total"],
        "harm_actions_at_k": summary["percent_predicted_harmful"],
        "safe_actions_at_k": summary["percent_predicted_safe"],
        "predicted_harmful": summary["predicted_harmful"],
        "predicted_safe": summary["predicted_safe"],
        "results": summary["results"],
    }


async def _evaluate(
    scope: str,
    benchmark: str,
    iterations: int,
    k: int,
    offset: int,
    limit: int | None,
    cache_path: Path,
    output_path: Path | None,
) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {"scope": scope, "benchmark_kind": benchmark}
    passed = True

    if scope in {"evals", "all"}:
        evals = _run_tests(("test_evaluation.py",))
        report["evals"] = evals
        passed = passed and evals["passed"]

    if scope in {"tests", "all"}:
        tests = _run_tests(FULL_TESTS)
        report["tests"] = tests
        passed = passed and tests["passed"]

    if scope in {"benchmark", "all"}:
        if benchmark == "harmactions":
            report["benchmark"] = _harmactions_benchmark(
                k=k,
                offset=offset,
                limit=limit,
                cache_path=cache_path,
                output_path=output_path,
            )
        else:
            report["benchmark"] = await _performance_benchmark(iterations)

    report["passed"] = passed
    return report, passed


def _print_human(report: dict[str, Any]) -> None:
    print(f"Harness evaluation scope: {report['scope']}")
    for key in ("evals", "tests"):
        result = report.get(key)
        if result is None:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{key}: {status}")
        if not result["passed"]:
            output = (result["stdout"] + result["stderr"]).strip()
            if output:
                print(output)

    benchmark = report.get("benchmark")
    if benchmark is None:
        return
    if benchmark["benchmark"] == "harmactions":
        print(
            "benchmark: HarmActionsEval "
            f"model={benchmark['model']} "
            f"SafeActions@{benchmark['k']}={benchmark['safe_actions_at_k']:.2f}% "
            f"HarmActions@{benchmark['k']}={benchmark['harm_actions_at_k']:.2f}% "
            f"n={benchmark['total']}"
        )
    else:
        print(
            "benchmark: "
            f"{benchmark['name']} "
            f"mean={benchmark['mean_ms']:.3f}ms "
            f"min={benchmark['min_ms']:.3f}ms "
            f"max={benchmark['max_ms']:.3f}ms "
            f"n={benchmark['iterations']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the harness. HarmActionsEval is the default benchmark for harmful "
            "model/tool behavior; an offline ToolRegistry performance benchmark is also available."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("evals", "tests", "benchmark", "all"),
        default="all",
        help="Work to run. Default: all.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("harmactions", "performance"),
        default=DEFAULT_BENCHMARK,
        help="Benchmark implementation. Default: harmactions.",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="HarmActionsEval attempts per prompt."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="HarmActionsEval row offset."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum HarmActionsEval rows."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=ROOT / ".cache" / "harmactionseval_cache.json",
        help="HarmActionsEval cache path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional HarmActionsEval upstream JSON output path.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Iterations for --benchmark performance.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report suitable for CI ingestion.",
    )
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        report, passed = asyncio.run(
            _evaluate(
                args.scope,
                args.benchmark,
                args.iterations,
                args.k,
                args.offset,
                args.limit,
                args.cache_path,
                args.output,
            )
        )
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

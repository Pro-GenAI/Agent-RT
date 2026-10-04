import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ext.operations import (
    BenchmarkCase,
    BenchmarkRunner,
    ConformanceCheck,
    ConformanceSuite,
    HealthRegistry,
    HealthStatus,
)


class TestOperations:
    async def test_benchmark_runner_records_iterations_and_stats(self):
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1

        result = await BenchmarkRunner().run(
            BenchmarkCase(
                "noop",
                operation,
                iterations=3,
                warmup_iterations=1,
                category="serialization",
            )
        )
        assert calls == 4
        assert result.iterations == 3
        assert result.category == "serialization"
        assert result.max_seconds >= result.min_seconds
        assert result.mean_seconds >= 0
        with pytest.raises(TypeError):
            BenchmarkCase("fractional", operation, iterations=1.5)
        with pytest.raises(ValueError):
            BenchmarkCase("blank-category", operation, category=" ")

    async def test_conformance_suite_reports_pass_and_failure(self):
        suite = ConformanceSuite("provider")
        suite.add(
            ConformanceCheck("has-name", lambda subject: _require(subject, "name"))
        )
        suite.add(
            ConformanceCheck("has-call", lambda subject: _require(subject, "call"))
        )

        report = await suite.run("mock", {"name": "mock"})
        assert not report.passed
        assert report.failed_count == 1
        assert report.results[0].passed == True
        assert "call" in report.results[1].error

    async def test_health_registry_reports_readiness_liveness_and_failures(self):
        health = HealthRegistry()
        health.register(
            "model",
            "primary",
            lambda: HealthStatus("primary", True, detail="ok"),
        )

        async def failing():
            raise RuntimeError("queue unavailable")

        health.register("queue", "jobs", failing)
        with pytest.raises(ValueError):
            health.register("queue", "jobs", failing)
        health.register(
            "queue",
            "jobs",
            lambda: HealthStatus("jobs", True),
            replace_existing=True,
        )
        health.register("queue", "jobs", failing, replace_existing=True)
        with pytest.raises(ValueError):
            health.register("unsupported", "bad", lambda: HealthStatus("bad", True))
        report = await health.check()
        assert report.live
        assert not report.ready
        assert not report.healthy
        assert [(item.name, item.metadata["kind"]) for item in report.checks] == [
            ("primary", "model"),
            ("jobs", "queue"),
        ]

        model_only = await health.check(kind="model")
        assert model_only.ready
        assert model_only.healthy
        with pytest.raises(ValueError):
            await health.check(kind="unsupported")

        async def slow():
            await asyncio.sleep(0.02)
            return HealthStatus("slow", True)

        health.register("runtime", "slow", slow)
        timed = await health.check(kind="runtime", timeout_seconds=0.001)
        assert not timed.ready
        assert "TimeoutError" in timed.checks[0].detail

    def test_harness_evaluation_script_benchmark_json(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_harness.py",
                "--scope",
                "benchmark",
                "--benchmark",
                "performance",
                "--iterations",
                "2",
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["passed"]
        assert report["benchmark"]["iterations"] == 2
        assert report["benchmark"]["name"] == "tool_registry_dispatch"

    def test_harness_evaluation_defaults_to_harmactions(self):
        env = dict(os.environ)
        env.pop("OPENAI_MODEL", None)
        completed = subprocess.run(
            [sys.executable, "scripts/evaluate_harness.py", "--scope", "benchmark"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        assert completed.returncode == 2
        assert "HarmActionsEval requires OPENAI_MODEL" in completed.stderr

    def test_dev_security_scan_prefers_project_venv_tools(self, tmp_path, monkeypatch):
        script = Path(__file__).resolve().parents[1] / "scripts" / "dev_security_scan.py"
        spec = importlib.util.spec_from_file_location("dev_security_scan_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        project_root = tmp_path / "project"
        project_bin = project_root / ".venv" / "bin"
        project_bin.mkdir(parents=True)
        project_tool = project_bin / "bandit"
        project_tool.write_text("#!/bin/sh\n")

        other_python = tmp_path / "other" / "bin" / "python3"
        other_python.parent.mkdir(parents=True)
        other_python.write_text("")
        global_tool = tmp_path / "global" / "bandit"
        global_tool.parent.mkdir(parents=True)
        global_tool.write_text("")

        monkeypatch.setattr(module, "ROOT", project_root)
        monkeypatch.setattr(module.sys, "executable", str(other_python))
        monkeypatch.setattr(module.shutil, "which", lambda _name: str(global_tool))

        assert module._tool("bandit") == str(project_tool)

    def test_dev_security_scan_does_not_require_git_for_secret_scan(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "dev_security_scan.py"
        source = script.read_text()

        assert '"--all-files"' in source
        assert '"--no-verify"' in source


def _require(subject, key):
    if key not in subject:
        raise ValueError(f"missing {key}")

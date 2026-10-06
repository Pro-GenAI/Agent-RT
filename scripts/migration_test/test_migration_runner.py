"""Regression tests for runner heuristics found through real-world field failures.

Each test reproduces a baseline failure seen in migration_results.csv. They are
offline and never run third-party code: ``python -m unittest test_migration_runner``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import migration_test as mt


def write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TempRepoCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)


class PnpmBuildAllowListTests(TempRepoCase):
    """stravu/crystal: ERR_PNPM_CONFIG_CONFLICT_BUILT_DEPENDENCIES on pnpm 11+."""

    def package(self, manager: str) -> dict:
        write(
            self.repo,
            "package.json",
            json.dumps({"packageManager": manager, "pnpm": {"onlyBuiltDependencies": ["esbuild"]}}),
        )
        return {}

    def test_modern_pnpm_drops_allow_list_without_never_built(self) -> None:
        self.package("pnpm@12.9.1")
        note = mt.allow_legacy_pnpm_builds(self.repo, {})
        data = json.loads((self.repo / "package.json").read_text())
        self.assertIn("pnpm", note)
        self.assertNotIn("pnpm", data)

    def test_legacy_pnpm_sets_never_built(self) -> None:
        self.package("pnpm@10.5.0")
        mt.allow_legacy_pnpm_builds(self.repo, {})
        data = json.loads((self.repo / "package.json").read_text())
        self.assertEqual(data["pnpm"], {"neverBuiltDependencies": []})

    def test_workspace_yaml_allow_list_is_removed(self) -> None:
        self.package("pnpm@11.0.0")
        write(
            self.repo,
            "pnpm-workspace.yaml",
            "packages:\n  - 'a'\nonlyBuiltDependencies:\n  - esbuild\n  - sharp\nlinkWorkspacePackages: true\n",
        )
        mt.allow_legacy_pnpm_builds(self.repo, {})
        text = (self.repo / "pnpm-workspace.yaml").read_text()
        self.assertNotIn("onlyBuiltDependencies", text)
        self.assertNotIn("sharp", text)
        self.assertIn("packages:", text)
        self.assertIn("linkWorkspacePackages: true", text)

    def test_untouched_when_no_allow_list(self) -> None:
        write(self.repo, "package.json", json.dumps({"packageManager": "pnpm@11.0.0"}))
        self.assertIsNone(mt.allow_legacy_pnpm_builds(self.repo, {}))


class InstallHintTests(TempRepoCase):
    """tahasiddiquii/soc-triage-agent: starlette TestClient requires httpx2."""

    OUTPUT = (
        "RuntimeError: The starlette.testclient module requires the httpx2 package to be installed.\n"
        "You can install this with:\n    $ pip install httpx2\n"
    )

    def test_known_hint_is_installed(self) -> None:
        self.assertEqual(mt.missing_dependencies_from_output(self.OUTPUT, self.repo), ["httpx2"])

    def test_unknown_hint_is_never_trusted(self) -> None:
        output = "RuntimeError: requires the evil-package package\n$ pip install evil-package\n"
        self.assertEqual(mt.missing_dependencies_from_output(output, self.repo), [])


class EnvPlaceholderTests(TempRepoCase):
    """JSv4/Delphic: django-environ env("CELERY_BROKER_URL") with no default."""

    def test_celery_broker_uses_in_memory_transport(self) -> None:
        write(
            self.repo,
            "config/settings/base.py",
            'CELERY_BROKER_URL = env("CELERY_BROKER_URL")\nCELERY_RESULT_BACKEND = CELERY_BROKER_URL\n'
            'REDIS_HOST = env("REDIS_HOST")\n',
        )
        values = mt.repo_env_placeholders(self.repo, {})
        self.assertEqual(values["CELERY_BROKER_URL"], "memory://")
        # Other network settings stay unset: no invented hosts.
        self.assertNotIn("REDIS_HOST", values)


class NestedProjectTests(TempRepoCase):
    def test_nested_requires_python_selects_matching_interpreter(self) -> None:
        """connector-native-apps: root declares nothing, nested project needs >=3.14."""
        write(self.repo, "pyproject.toml", "[project]\nname='root'\n")
        write(self.repo, "svc/pyproject.toml", "[project]\nname='svc'\nrequires-python='>=3.14'\n")
        write(self.repo, "svc/tests/test_x.py", "def test_x(): pass\n")
        self.assertEqual(mt.select_python_version(self.repo), "3.14")

    def test_nested_example_range_is_ignored(self) -> None:
        write(self.repo, "pyproject.toml", "[project]\nname='root'\n")
        write(self.repo, "examples/demo/pyproject.toml", "[project]\nname='d'\nrequires-python='>=3.14'\n")
        self.assertEqual(mt.select_python_version(self.repo), mt.PYTHON_VERSION)

    def test_virtual_nested_project_dependencies_are_readable(self) -> None:
        """chirpz-ai/pandaprobe: backend/ has dependencies but no build system."""
        write(
            self.repo,
            "backend/pyproject.toml",
            "[project]\nname='b'\ndependencies=['posthog>=7']\n",
        )
        write(self.repo, "backend/tests/test_a.py", "")
        backend = self.repo / "backend"
        self.assertTrue(mt.project_owns_tests(backend, self.repo))
        self.assertFalse(mt.is_installable_project(backend))
        self.assertEqual(mt.pyproject_dependencies(backend, all_extras=True), ["posthog>=7"])

    def test_src_layout_project_dir_is_on_pythonpath(self) -> None:
        """jkmaina/openai-agents-blueprint: tests import ``src.my_agents``."""
        project = self.repo / "chapter3" / "proj"
        write(project, "src/__init__.py")
        write(project, "src/my_agents/__init__.py")
        write(project, "tests/test_a.py")
        env = mt.python_repo_env(self.repo, {})
        self.assertIn(str(project), env["PYTHONPATH"].split(":"))

    def test_tqdm_is_a_curated_missing_dependency(self) -> None:
        """JSv4/Delphic: undeclared tqdm named only by the import error."""
        output = "E   ModuleNotFoundError: No module named 'tqdm'\n"
        self.assertEqual(mt.missing_dependencies_from_output(output, self.repo), ["tqdm"])


class MockServerTests(unittest.TestCase):
    """jkmaina/openai-agents-blueprint: Agents SDK read usage.input_tokens_details."""

    def test_responses_usage_has_token_details(self) -> None:
        import subprocess
        import urllib.request

        here = Path(__file__).parent
        if not (here / "node_modules").is_dir():
            self.skipTest("mock server dependencies not installed")
        proc = subprocess.Popen(
            ["node", "mock_server.mjs"], cwd=here, stdout=subprocess.PIPE, text=True
        )
        self.addCleanup(proc.wait)
        self.addCleanup(proc.terminate)
        url = ""
        for line in proc.stdout:  # type: ignore[union-attr]
            if line.startswith("AIMOCK_READY="):
                url = line.strip().split("=", 1)[1]
                break
        self.assertTrue(url)
        for extra in ({}, {"text": {"format": {"type": "json_object"}}}):
            request = urllib.request.Request(
                f"{url}/v1/responses",
                data=json.dumps({"model": "gpt-4o", "input": "hi", **extra}).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                usage = json.load(response)["usage"]
            self.assertEqual(usage["input_tokens_details"], {"cached_tokens": 0})
            self.assertEqual(usage["output_tokens_details"], {"reasoning_tokens": 0})


if __name__ == "__main__":
    unittest.main()

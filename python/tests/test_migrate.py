import io
import sys
from pathlib import Path

from ext.migrate import SuiteRun, compare, main, parse_junit, plan_rewrite


def run(exit_code: int, failures: dict[str, str | None] | None) -> SuiteRun:
    return SuiteRun(exit_code, 1.0, None, 3, failures, "")


def test_compare_accepts_only_the_same_failures_at_the_same_line():
    before = run(1, {"t::a": "tests/test_x.py:4"})
    assert compare(before, run(0, {}))[0]
    assert compare(before, run(1, {"t::a": "tests/test_x.py:4"}))[0]
    assert not compare(before, run(1, {"t::a": "tests/test_x.py:9"}))[0]
    assert not compare(
        before, run(1, {"t::a": "tests/test_x.py:4", "t::b": "tests/test_x.py:7"})
    )[0]
    assert not compare(run(0, {}), run(1, {"t::a": "tests/test_x.py:4"}))[0]
    assert not compare(before, run(1, None))[0]


def test_parse_junit_refuses_dtd_entity_bombs(tmp_path: Path):
    report = tmp_path / "report.xml"
    report.write_text('<testsuite><testcase classname="t" name="a"/></testsuite>')
    assert parse_junit(report, tmp_path) == (1, {})
    report.write_text(
        '<!DOCTYPE x [<!ENTITY a "aaaa">]><testsuite><testcase name="&a;"/></testsuite>'
    )
    assert parse_junit(report, tmp_path) is None


def test_local_module_named_like_a_root_is_not_rewritten(tmp_path: Path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "__init__.py").write_text("")
    (tmp_path / "app.py").write_text("import agents\nimport openai\n")
    plan = plan_rewrite(tmp_path)
    assert plan.roots == {"openai"}
    assert (
        plan.migrated[tmp_path / "app.py"]
        == "import agents\nimport agent_rt.openai as openai\n"
    )


def _migrate(project: Path) -> tuple[int, str]:
    out = io.StringIO()
    return main([str(project), "--python", sys.executable], out=out), out.getvalue()


def test_migrate_keeps_rewrite_when_old_failure_is_unchanged(tmp_path: Path):
    test_file = tmp_path / "test_app.py"
    test_file.write_text(
        "import openai\n\n\ndef test_client():\n    assert openai.OpenAI\n\n\n"
        "def test_known_bug():\n    assert 1 == 2\n"
    )
    code, output = _migrate(tmp_path)
    assert code == 0, output
    assert "MIGRATED" in output and "1/2 passed -> 1/2 passed" in output
    assert "Migration passed!" in output
    assert test_file.read_text().startswith("import agent_rt.openai as openai\n")


def test_migrate_restores_files_on_a_new_failure(tmp_path: Path):
    test_file = tmp_path / "test_app.py"
    original = "import openai\n\n\ndef test_module_name():\n    assert openai.__name__ == 'openai'\n"
    test_file.write_text(original)
    code, output = _migrate(tmp_path)
    assert code == 1, output
    assert "REVERTED" in output and "test_module_name (test_app.py:5)" in output
    assert test_file.read_text() == original

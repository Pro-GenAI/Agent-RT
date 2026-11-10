import asyncio

from agent_rt_cli import main
from ext.demo import run


def test_demo_blocks_injected_command_offline():
    lines: list[str] = []
    summary = asyncio.run(run(lines.append))
    assert summary["ran"] == ["get_weather"]
    assert summary["tokens"] > 0
    assert any("BLOCKED before execution" in line for line in lines)
    assert any("|| >    < ||" in line for line in lines)


def test_demo_rejects_arguments(capsys):
    assert main(["demo", "--help"]) == 0
    assert main(["demo", "extra"]) == 2
    assert "usage: agent-rt demo" in capsys.readouterr().out

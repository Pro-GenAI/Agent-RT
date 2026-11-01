"""Offline coverage for CLI update discovery, installation, and restart."""

import io
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import agent_rt_cli as cli
from ext import cli_update
from ext.cli_update import AvailableUpdate, UpdateError


def _pypi_response(version):
    @contextmanager
    def response(request, *, timeout):
        assert request.full_url == "https://pypi.org/pypi/agent-rt/json"
        assert timeout == 3
        yield io.BytesIO(json.dumps({"info": {"version": version}}).encode())

    return response


@pytest.mark.parametrize(
    ("installed", "latest", "expected"),
    [
        ("1.0.0", "1.1.0", True),
        ("1.2.0", "1.1.0", False),
        ("1.2.0", "1.2.0", False),
        ("1.0.0", "1.1.0rc1", False),
        ("1.1.0rc1", "1.1.0", True),
        ("1.9.0", "1.10.0", True),
    ],
)
def test_release_comparison(monkeypatch, installed, latest, expected):
    monkeypatch.setattr(cli_update.importlib.metadata, "version", lambda _: installed)
    monkeypatch.setattr(cli_update, "urlopen", _pypi_response(latest))
    result = cli_update.check_latest_release()
    assert result == (AvailableUpdate(installed, latest) if expected else None)


def test_release_check_handles_network_failure(monkeypatch):
    monkeypatch.setattr(cli_update.importlib.metadata, "version", lambda _: "1.0.0")

    def failed(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr(cli_update, "urlopen", failed)
    with pytest.raises(UpdateError, match="Unable to check PyPI"):
        cli_update.check_latest_release()


def test_editable_install_cannot_be_overwritten(monkeypatch):
    monkeypatch.setattr(
        cli_update.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: '{"dir_info":{"editable":true}}'),
    )
    with pytest.raises(UpdateError, match="Editable installs"):
        cli_update._install_command("1.2.0")


@pytest.mark.parametrize(
    ("prefix", "manager", "expected"),
    [
        (
            "/home/test/.local/share/pipx/venvs/agent-rt",
            "pipx",
            ["pipx", "upgrade", "agent-rt"],
        ),
        (
            "/home/test/.local/share/uv/tools/agent-rt",
            "uv",
            ["uv", "tool", "upgrade", "agent-rt"],
        ),
    ],
)
def test_install_command_uses_owning_tool_manager(
    monkeypatch, prefix, manager, expected
):
    monkeypatch.setattr(
        cli_update.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: None),
    )
    monkeypatch.setattr(cli_update.sys, "prefix", prefix)
    monkeypatch.setattr(
        cli_update.shutil, "which", lambda name: name if name == manager else None
    )
    assert cli_update._install_command("1.2.0") == expected


def test_install_command_uses_current_python_for_pip(monkeypatch):
    monkeypatch.setattr(
        cli_update.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: None),
    )
    monkeypatch.setattr(cli_update.sys, "prefix", "/home/test/project/.venv")
    monkeypatch.setattr(cli_update.importlib.util, "find_spec", lambda name: object())
    assert cli_update._install_command("1.2.0") == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "agent-rt[cli]==1.2.0",
    ]


def test_install_command_refuses_temporary_uvx_environment(monkeypatch):
    monkeypatch.setattr(
        cli_update.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: None),
    )
    monkeypatch.setattr(cli_update.sys, "prefix", "/home/test/.cache/uv/archive-v0/xyz")
    with pytest.raises(UpdateError, match="uvx --refresh"):
        cli_update._install_command("1.2.0")


def test_failed_install_never_exposes_private_index_output(monkeypatch):
    monkeypatch.setattr(cli_update, "_install_command", lambda _: ["pip", "install"])
    monkeypatch.setattr(
        cli_update.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="https://user:secret@private.example/simple"
        ),
    )
    with pytest.raises(UpdateError) as failure:
        cli_update.install_release("1.2.0")
    assert "secret" not in str(failure.value)


def test_upgrade_first_argument_bypasses_chat_setup(monkeypatch, capsys):
    installed = []
    monkeypatch.setattr(
        cli, "check_latest_release", lambda: AvailableUpdate("1.0", "1.1")
    )
    monkeypatch.setattr(cli, "install_release", installed.append)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["upgrade"]) == 0
    assert installed == ["1.1"]
    assert "Run agent-rt again" in capsys.readouterr().out


def test_upgrade_enter_restarts_without_upgrade_subcommand(monkeypatch):
    restarts = []
    monkeypatch.setattr(
        cli, "check_latest_release", lambda: AvailableUpdate("1.0", "1.1")
    )
    monkeypatch.setattr(cli, "install_release", lambda version: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(cli, "restart_cli", lambda args: restarts.append(args))
    assert cli.main(["upgrade", "--model", "some-model"]) == 0
    assert restarts == [["--model", "some-model"]]


@pytest.mark.asyncio
async def test_slash_update_installs_and_restarts_on_enter(monkeypatch):
    printed = []
    installations = []
    restarts = []
    chat = object.__new__(cli.TerminalChat)
    chat.console = SimpleNamespace(print=lambda message: printed.append(message))
    chat.approval_prompt = SimpleNamespace(prompt_async=lambda message: _immediate(""))
    chat._restart_argv = ("--model", "my-model")
    monkeypatch.setattr(
        cli, "check_latest_release", lambda: AvailableUpdate("1.0", "1.1")
    )
    monkeypatch.setattr(cli, "install_release", installations.append)
    monkeypatch.setattr(cli, "restart_cli", restarts.append)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    assert await chat._command("/update")
    assert installations == ["1.1"]
    assert restarts == [("--model", "my-model")]
    assert any("1.1" in message for message in printed)


async def _immediate(value):
    return value


@pytest.mark.asyncio
async def test_startup_check_ignores_network_errors_but_manual_check_reports_them(
    monkeypatch,
):
    printed = []
    chat = object.__new__(cli.TerminalChat)
    chat.console = SimpleNamespace(print=lambda message: printed.append(message))

    def unavailable():
        raise UpdateError("offline")

    monkeypatch.setattr(cli, "check_latest_release", unavailable)
    await chat._update(manual=False)
    assert printed == []
    await chat._update(manual=True)
    assert any("offline" in message for message in printed)


def test_slash_command_completion_exposes_update():
    assert "/update" in cli._slash_command_matches("/upd")


def test_missing_distribution_metadata_is_not_reported_as_up_to_date(monkeypatch):
    def missing(_):
        raise cli_update.importlib.metadata.PackageNotFoundError("agent-rt")

    monkeypatch.setattr(cli_update.importlib.metadata, "version", missing)
    with pytest.raises(UpdateError, match="not installed as a package"):
        cli_update.check_latest_release()


def test_install_success_verifies_version_in_new_process(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0, stdout="1.2.0\n")

    monkeypatch.setattr(cli_update, "_install_command", lambda _: ["pip", "install"])
    monkeypatch.setattr(cli_update.subprocess, "run", run)
    cli_update.install_release("1.2.0")
    assert len(commands) == 2
    assert commands[1][:2] == [sys.executable, "-c"]


@pytest.mark.asyncio
async def test_interactive_startup_checks_updates_before_accepting_input():
    events = []
    chat = object.__new__(cli.TerminalChat)
    chat.banner = lambda: events.append("banner")

    async def check(*, manual):
        events.append(("update", manual))

    async def end_of_input(prompt):
        events.append("prompt")
        raise EOFError

    chat._update = check
    chat.prompt = SimpleNamespace(prompt_async=end_of_input)
    chat.console = SimpleNamespace(print=lambda *args: None)
    assert await chat.run() == 0
    assert events == ["banner", ("update", False), "prompt"]


def test_install_rejects_success_without_newer_version(monkeypatch):
    results = iter(
        [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0, stdout="1.1.0\n"),
        ]
    )
    monkeypatch.setattr(cli_update, "_install_command", lambda _: ["pip", "install"])
    monkeypatch.setattr(cli_update.subprocess, "run", lambda *args, **kw: next(results))
    with pytest.raises(UpdateError, match="still on an older release"):
        cli_update.install_release("1.2.0")

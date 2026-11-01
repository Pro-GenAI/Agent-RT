"""Opt-in-to-interactive CLI self-upgrades from the published PyPI release."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

_PYPI_URL = "https://pypi.org/pypi/agent-rt/json"
_NETWORK_TIMEOUT = 3
_INSTALL_TIMEOUT = 180


class UpdateError(RuntimeError):
    """An update could not be checked or installed."""


@dataclass(frozen=True)
class AvailableUpdate:
    installed: str
    latest: str


def check_latest_release() -> AvailableUpdate | None:
    """Check the official PyPI release without modifying the environment."""
    try:
        installed = importlib.metadata.version("agent-rt")
    except importlib.metadata.PackageNotFoundError:
        raise UpdateError(
            "Agent RT is not installed as a package; install it before upgrading."
        ) from None

    try:
        installed_version = Version(installed)
    except InvalidVersion as exc:
        raise UpdateError("Unable to identify the installed Agent RT version.") from exc

    request = Request(
        _PYPI_URL,
        headers={"Accept": "application/json", "User-Agent": f"agent-rt/{installed}"},
    )
    try:
        with urlopen(request, timeout=_NETWORK_TIMEOUT) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("response too large")
        latest = json.loads(payload)["info"]["version"]
        latest_version = Version(latest)
    except (
        HTTPError,
        URLError,
        OSError,
        KeyError,
        ValueError,
        TypeError,
        InvalidVersion,
    ) as exc:
        raise UpdateError("Unable to check PyPI for Agent RT updates.") from exc

    # Do not automatically move from a stable release to a prerelease.
    if latest_version <= installed_version or (
        latest_version.is_prerelease and not installed_version.is_prerelease
    ):
        return None
    return AvailableUpdate(installed=installed, latest=latest)


def _install_command(version: str) -> list[str]:
    """Pick the manager that owns this installation; never modify a source checkout."""
    try:
        direct_url = importlib.metadata.distribution("agent-rt").read_text(
            "direct_url.json"
        )
        if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
            raise UpdateError(
                "Editable installs cannot be auto-upgraded; update the checkout and reinstall."
            )
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        raise UpdateError("Unable to determine how Agent RT was installed.") from None

    prefix_parts = tuple(part.lower() for part in Path(sys.prefix).parts)
    if "uv" in prefix_parts and (
        "archive-v0" in prefix_parts or "cache" in prefix_parts
    ):
        raise UpdateError(
            "This is a temporary uvx environment. Relaunch with 'uvx --refresh agent-rt'."
        )
    if "pipx" in prefix_parts and "venvs" in prefix_parts:
        if shutil.which("pipx"):
            return ["pipx", "upgrade", "agent-rt"]
        raise UpdateError("pipx is not available; run 'pipx upgrade agent-rt'.")
    if "uv" in prefix_parts and "tools" in prefix_parts:
        if shutil.which("uv"):
            return ["uv", "tool", "upgrade", "agent-rt"]
        raise UpdateError("uv is not available; run 'uv tool upgrade agent-rt'.")

    spec = f"agent-rt[cli]=={Version(version)}"
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, "--upgrade", spec]
    raise UpdateError(
        "No compatible installer found. Install pip or uv and upgrade agent-rt[cli]."
    )


def install_release(version: str) -> None:
    """Upgrade using the owning package manager, without shell interpolation."""
    command = _install_command(version)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_INSTALL_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError(
            "Could not run the package manager to upgrade Agent RT."
        ) from exc
    if result.returncode != 0:
        # Package manager diagnostics may contain private index URLs/tokens.
        raise UpdateError(
            "Upgrade failed. Try updating Agent RT with your original package manager."
        )
    # Some tool managers exit successfully even when an install is version-pinned.
    # Query a new interpreter because this process may cache old distribution metadata.
    try:
        actual = subprocess.run(
            [
                sys.executable,
                "-c",
                "from importlib.metadata import version; print(version('agent-rt'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if actual.returncode != 0 or Version(actual.stdout.strip()) < Version(version):
            raise UpdateError(
                "The package manager completed but Agent RT is still on an older release."
            )
    except (OSError, subprocess.TimeoutExpired, InvalidVersion) as exc:
        raise UpdateError("Unable to verify the upgraded Agent RT version.") from exc


def restart_cli(argv: Sequence[str]) -> None:
    """Replace the current process, preserving CLI arguments and working directory."""
    entrypoint = Path(sys.argv[0])
    if entrypoint.suffix.lower() == ".exe":
        os.execv(str(entrypoint), [str(entrypoint), *argv])
    else:
        os.execv(sys.executable, [sys.executable, str(entrypoint), *argv])

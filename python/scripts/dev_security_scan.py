#!/usr/bin/env python3
"""Run the third-party security tools installed by agent-rt[dev].

This gate is intentionally offline with respect to LLM/model endpoints. It
checks repository source/dependencies/secrets and exercises local artifact
scanners against a benign pickle fixture.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess  # nosec B404 - argv-only local developer-tool execution
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScanFailure(RuntimeError):
    pass


def _tool(name: str) -> str:
    executable_names = (name, f"{name}.exe")
    script_dirs = (
        ROOT / ".venv" / "bin",
        ROOT / ".venv" / "Scripts",
        Path(sys.executable).resolve().parent,
    )
    for script_dir in script_dirs:
        for executable_name in executable_names:
            candidate = script_dir / executable_name
            if candidate.is_file():
                return str(candidate)

    resolved = shutil.which(name)
    if resolved is not None:
        return resolved

    raise ScanFailure(
        f"Required developer security tool {name!r} is not installed in PATH "
        f"or {ROOT / '.venv'}. Install with: uv pip install -e \".[dev]\" "
        'or run with: uv run --extra dev python scripts/dev_security_scan.py'
    )


def _run(
    label: str,
    argv: list[str],
    *,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"[RUN ] {label}")
    completed = subprocess.run(  # nosec B603 - argv is fixed/local developer tooling
        argv,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise ScanFailure(f"{label} failed with exit code {completed.returncode}")
    print(f"[PASS] {label}")
    return completed


def main() -> int:
    try:
        _run(
            "Bandit medium+ source scan",
            [_tool("bandit"), "-r", "src", "scripts", "-q", "-ll"],
        )

        audit = _run(
            "pip-audit dependency scan",
            [_tool("pip-audit"), "-f", "json"],
        )
        audit_payload = json.loads(audit.stdout)
        vulnerable = [
            dep
            for dep in audit_payload.get("dependencies", [])
            if dep.get("vulns")
        ]
        if vulnerable:
            raise ScanFailure(
                f"pip-audit reported vulnerabilities in {len(vulnerable)} package(s)"
            )

        secrets = _run(
            "detect-secrets source scan",
            [
                _tool("detect-secrets"),
                "scan",
                "--all-files",
                "--no-verify",
                "src",
                "scripts",
                "examples",
            ],
        )
        secret_payload = json.loads(secrets.stdout)
        secret_count = sum(
            len(items) for items in secret_payload.get("results", {}).values()
        )
        if secret_count:
            raise ScanFailure(
                f"detect-secrets reported {secret_count} potential secret(s)"
            )

        _run(
            "AgentSec local agent/MCP scan",
            [
                _tool("agentsec"),
                "scan",
                "src",
                "-s",
                "skill,mcp",
                "--fail-on",
                "high",
                "-q",
            ],
        )

        _run(
            "PromptFuzz offline attack-corpus load",
            [_tool("promptfuzz"), "list-attacks"],
        )

        with tempfile.TemporaryDirectory(prefix="agent-rt-security-") as temp_dir:
            temp_root = Path(temp_dir)
            fixture = temp_root / "benign.pkl"
            fixture.write_bytes(
                pickle.dumps({"agent_rt_security_fixture": True, "items": [1, 2, 3]})
            )

            _run(
                "Fickling benign artifact scan",
                [
                    _tool("fickling"),
                    "--check-safety",
                    str(fixture),
                    "--json-output",
                    str(temp_root / "fickling.json"),
                ],
            )
            _run(
                "PickleScan benign artifact scan",
                [_tool("picklescan"), "-p", str(fixture)],
            )

            modelscan = shutil.which("modelscan")
            if modelscan is None:
                print(
                    "[SKIP] ModelScan is only installed by the dev extra on "
                    "Python 3.10-3.12."
                )
            else:
                _run(
                    "ModelScan benign artifact scan",
                    [
                        modelscan,
                        "scan",
                        "-p",
                        str(fixture),
                        "-r",
                        "json",
                        "-o",
                        str(temp_root / "modelscan.json"),
                    ],
                )

            _run(
                "CycloneDX environment SBOM generation",
                [
                    _tool("cyclonedx-py"),
                    "environment",
                    "--output-file",
                    str(temp_root / "sbom.json"),
                ],
            )

        print("Third-party developer security gate passed.")
        return 0
    except (ScanFailure, json.JSONDecodeError) as exc:
        print(f"Security gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

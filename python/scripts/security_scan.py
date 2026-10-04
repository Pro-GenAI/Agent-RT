#!/usr/bin/env python3
"""Heuristic security scan for Agent RT Python, agent/LLM controls, and API server."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
SEVERITIES = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    category: str
    path: str
    line: int
    message: str
    remediation: str
    confidence: str = "high"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def line_of(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    return 1


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


class Audit:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.notes: list[str] = []
        self._seen: set[tuple[str, str, int]] = set()

    def add(self, rule_id: str, severity: str, category: str, path: Path, line: int,
            message: str, remediation: str, confidence: str = "high") -> None:
        key = (rule_id, rel(path), max(1, line))
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(Finding(rule_id, severity, category, rel(path), max(1, line),
                                     message, remediation, confidence))

    def require(self, text: str, needle: str, rule_id: str, severity: str, category: str,
                path: Path, message: str, remediation: str) -> None:
        if needle not in text:
            self.add(rule_id, severity, category, path, 1, message, remediation)


class CodeVisitor(ast.NodeVisitor):
    def __init__(self, audit: Audit, path: Path) -> None:
        self.audit = audit
        self.path = path

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = call_name(node.func)
        if name in {"eval", "exec", "compile"}:
            self.audit.add("PY-CODE-001", "critical", "code", self.path, node.lineno,
                           f"Dynamic code execution through {name}().",
                           "Replace runtime code evaluation with constrained parsing or an explicit dispatcher.")
        if name in {"os.system", "os.popen", "asyncio.create_subprocess_shell"}:
            self.audit.add("PY-CODE-002", "critical", "code", self.path, node.lineno,
                           f"Shell execution through {name}().",
                           "Use argv-based process execution without a shell; validate externally influenced arguments.")
        if name in {"subprocess.run", "subprocess.call", "subprocess.check_call",
                    "subprocess.check_output", "subprocess.Popen"}:
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.audit.add("PY-CODE-003", "critical", "code", self.path, node.lineno,
                                   f"{name}() enables shell=True.",
                                   "Use shell=False and an argument vector; never interpolate model/user input into a shell.")
        if name in {"pickle.load", "pickle.loads", "dill.load", "dill.loads",
                    "cloudpickle.load", "cloudpickle.loads"}:
            self.audit.add("PY-CODE-004", "high", "code", self.path, node.lineno,
                           f"Executable deserialization through {name}().",
                           "Do not deserialize attacker-controlled pickle-compatible data; use a non-executable format.")
        if name == "yaml.load":
            loader = next((kw.value for kw in node.keywords if kw.arg in {"Loader", "loader"}), None)
            if call_name(loader) not in {"yaml.SafeLoader", "SafeLoader", "yaml.CSafeLoader", "CSafeLoader"}:
                self.audit.add("PY-CODE-005", "high", "code", self.path, node.lineno,
                               "yaml.load() lacks an explicitly safe loader.",
                               "Use yaml.safe_load() or SafeLoader/CSafeLoader.")
        if name == "ssl._create_unverified_context":
            self.audit.add("PY-CODE-006", "high", "code", self.path, node.lineno,
                           "TLS certificate verification is disabled.",
                           "Use the verified default TLS context and configured trust roots.")
        if name == "tempfile.mktemp":
            self.audit.add("PY-CODE-007", "medium", "code", self.path, node.lineno,
                           "tempfile.mktemp() is race-prone.",
                           "Use NamedTemporaryFile, TemporaryDirectory, or mkstemp.")
        for keyword in node.keywords:
            if keyword.arg == "verify" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                self.audit.add("PY-CODE-008", "high", "code", self.path, node.lineno,
                               "HTTP/TLS certificate verification is disabled with verify=False.",
                               "Keep verification enabled and configure a CA bundle when needed.")
        self.generic_visit(node)


def source_files(include_tests: bool) -> Iterable[Path]:
    roots = [SRC]
    if include_tests:
        roots += [PACKAGE_ROOT / "tests", PACKAGE_ROOT / "scripts"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or ".venv" in path.parts or "egg-info" in str(path):
                continue
            yield path


def scan_code(audit: Audit, include_tests: bool) -> None:
    for path in source_files(include_tests):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            audit.add("PY-CODE-000", "high", "code", path, getattr(exc, "lineno", 1) or 1,
                      f"Source could not be parsed for security analysis: {exc}",
                      "Fix the read/parse error so the file is covered by security scanning.")
            continue
        CodeVisitor(audit, path).visit(tree)


def scan_agent_llm(audit: Audit) -> None:
    path = SRC / "agent_rt.py"
    if not path.exists():
        audit.notes.append("Agent/LLM scan skipped: python/src/agent_rt.py is missing.")
        return
    text = path.read_text(encoding="utf-8")

    checks = [
        ('trust: TrustLevel = "trusted"', "PY-LLM-001", "medium",
         "ContextItem defaults to trusted, so externally sourced context is unsafe if callers omit trust metadata.",
         "Prefer untrusted for external/retrieved content or enforce explicit trust in every ingestion adapter."),
        ("permission_engine: PermissionEngine | None = None", "PY-LLM-002", "info",
         "Tool permission enforcement is optional in ToolRegistry.",
         "Use a deny-by-default PermissionEngine whenever model-controlled tools reach sensitive resources."),
        ("approval_manager: ApprovalManager | None = None", "PY-LLM-003", "info",
         "Human approval for consequential/destructive tools is optional.",
         "Configure ApprovalManager for write, consequential, or destructive tools."),
        ("enable_model_tool_input_guardrail: bool = False", "PY-LLM-004", "info",
         "Model-backed tool-input abuse screening is disabled by default.",
         "Enable it for untrusted prompts when the latency/dependency tradeoff is acceptable."),
    ]
    for needle, rule, severity, message, remediation in checks:
        if needle in text:
            audit.add(rule, severity, "agent-llm", path, line_of(text, needle), message, remediation, "medium")

    limits_start = text.find("class AgentRunLimits:")
    limits_block = text[limits_start:text.find("\n\n", limits_start)] if limits_start >= 0 else ""
    if "timeout_seconds: float | None = None" in limits_block:
        audit.add("PY-LLM-005", "info", "agent-llm", path,
                  line_of(text, "class AgentRunLimits:"),
                  "AgentRunLimits has no wall-clock timeout by default.",
                  "Set a finite timeout for network-facing and multi-tenant execution.", "medium")
    if "max_total_tokens: int | None = None" in limits_block:
        audit.add("PY-LLM-006", "info", "agent-llm", path,
                  line_of(text, "class AgentRunLimits:"),
                  "AgentRunLimits has no total-token budget by default.",
                  "Set a finite token budget to constrain denial-of-wallet/resource exhaustion.", "medium")

    audit.require(text, "validate_tool_arguments(registered.definition, effective_call.arguments)",
                  "PY-LLM-101", "critical", "agent-llm", path,
                  "Tool arguments no longer appear to be validated before handler execution.",
                  "Restore schema validation immediately before tool side effects.")
    audit.require(text, "call.name not in active_visible_tool_names",
                  "PY-LLM-102", "critical", "agent-llm", path,
                  "Model-returned tool calls are not visibly rechecked against the active tool set.",
                  "Recheck every tool call against capability/visibility policy at execution time.")
    audit.require(text, 'allow_untrusted_side_effects: bool = False',
                  "PY-LLM-103", "high", "agent-llm", path,
                  "Prompt-injection defense is not fail-closed for untrusted context.",
                  "Keep side-effecting tools hidden when untrusted context is present.")
    boundary_guardrail_default = re.search(
        r"boundary_guardrail_policy\s*:\s*\(?\s*BoundaryGuardrailPolicy\s*\|\s*None\s*\)?\s*=\s*BoundaryGuardrailPolicy\(\)",
        text,
        re.MULTILINE,
    )
    if boundary_guardrail_default is None:
        audit.add(
            "PY-LLM-104",
            "high",
            "agent-llm",
            path,
            line_of(text, "boundary_guardrail_policy"),
            "Default input/output boundary guardrails do not appear enabled.",
            "Keep bounded/sanitizing boundary guardrails enabled by default.",
            "high",
        )


def scan_api(audit: Audit) -> None:
    path = SRC / "agent_rt_api.py"
    if not path.exists():
        audit.notes.append("API scan skipped: python/src/agent_rt_api.py is missing.")
        return
    text = path.read_text(encoding="utf-8")
    lower = text.lower()

    auth_default = "api_" + "keys: Collection[str] = ()"
    secure_non_loopback_bind = (
        "allow_unauthenticated_non_loopback" in text
        and "_is_loopback_host(host)" in text
    )
    if (
        auth_default in text
        and "if not self.api_" + "keys:" in text
        and not secure_non_loopback_bind
    ):
        audit.add("PY-API-001", "medium", "api-server", path, line_of(text, auth_default),
                  "API authentication is opt-in; an empty configured key set authorizes requests.",
                  "Require authentication for non-loopback deployments or require an explicit unsafe local-only mode.")
    compare = "authorization[7:] in self.api_" + "keys"
    if compare in text:
        audit.add("PY-API-002", "low", "api-server", path, line_of(text, compare),
                  "Bearer credentials use ordinary string/set equality rather than constant-time comparison.",
                  "Use secrets.compare_digest against configured credentials.", "medium")
    if "request.json()" in text and not any(mark in lower for mark in
                                             ("content-length", "max_request_body", "max_body_bytes", "request_body_limit")):
        audit.add("PY-API-003", "high", "api-server", path, line_of(text, "value = await request.json()"),
                  "HTTP JSON request bodies are parsed without an explicit application-level size limit.",
                  "Enforce Content-Length/streamed body limits before JSON parsing and return 413 for oversized bodies.")
    if "receive_json()" in text and not any(mark in lower for mark in
                                             ("max_websocket", "max_message_size", "websocket_max", "max_ws")):
        audit.add("PY-API-004", "high", "api-server", path, line_of(text, "body = await websocket.receive_json()"),
                  "WebSocket JSON messages have no explicit application-level size limit.",
                  "Set server WebSocket size limits and reject oversized messages before decoding.")
    api_lines = text.splitlines()
    for index, line in enumerate(api_lines):
        if "str(exc)" not in line or "status =" in line:
            continue
        window = "\n".join(api_lines[max(0, index - 2):index + 3])
        if "_error_payload" in window or "anthropic_http_error" in window:
            audit.add("PY-API-005", "medium", "api-server", path, index + 1,
                      "Raw exception text can be returned to API clients.",
                      "Return stable public error codes/messages and log detailed exceptions only to redacted internal telemetry.")
    if "status_code=429" not in text and "RateLimiter" not in text:
        audit.add("PY-API-006", "medium", "api-server", path, line_of(text, "def create_app("),
                  "The inbound API server has no explicit request-level rate limiter.",
                  "Add per-key/IP rate and concurrency limiting at the HTTP/WebSocket boundary.", "medium")
    fastapi_line = next((line for line in text.splitlines() if "FastAPI(" in line), "")
    docs_secure_by_default = (
        "enable_docs: bool = False" in text
        and "docs_url=" in text
        and "openapi_url=" in text
    )
    if fastapi_line and not docs_secure_by_default and "docs_url=None" not in fastapi_line and "openapi_url=None" not in fastapi_line:
        audit.add("PY-API-007", "low", "api-server", path, line_of(text, "FastAPI("),
                  "FastAPI docs/OpenAPI endpoints use public defaults.",
                  "Disable or authenticate docs/OpenAPI for production-facing deployments.", "medium")
    serve = text[text.find("def serve("):] if "def serve(" in text else ""
    if 'host: str = "127.0.0.1"' in serve and not any(mark in serve for mark in
                                                        ("ipaddress.ip_address", "is_loopback", "non-loopback")):
        audit.add("PY-API-008", "medium", "api-server", path, line_of(text, 'host: str = "127.0.0.1"'),
                  "serve() can be rebound to a non-loopback interface while authentication remains optional.",
                  "Refuse non-loopback binds without credentials or require an explicit unsafe override.")
    if '@app.websocket("/v1/responses")' in text and 'websocket.headers.get("origin")' not in text:
        audit.add("PY-API-009", "low", "api-server", path, line_of(text, '@app.websocket("/v1/responses")'),
                  "The WebSocket endpoint does not validate Origin.",
                  "Validate allowed origins for browser-reachable deployments in addition to bearer authentication.", "medium")
    if "limits: AgentRunLimits = AgentRunLimits()" in text:
        audit.add("PY-API-010", "medium", "api-server", path,
                  line_of(text, "limits: AgentRunLimits = AgentRunLimits()"),
                  "API requests inherit run limits with no default wall-clock or total-token budget.",
                  "Configure finite timeout and token budgets for network-facing deployments.")
    if "max_output_tokens=int(max_output_tokens)" in text and "max_output_tokens_limit" not in text:
        audit.add("PY-API-011", "medium", "api-server", path,
                  line_of(text, "max_output_tokens=int(max_output_tokens)"),
                  "Clients can override model max_output_tokens without a server-side clamp.",
                  "Clamp caller-supplied max_tokens/max_output_tokens to a deployment-controlled maximum.",
                  "medium")


def selected_categories(values: Sequence[str] | None) -> set[str]:
    selected = set(values or ("code", "agent-llm", "api-server"))
    if "all" in selected:
        return {"code", "agent-llm", "api-server"}
    return selected


def output(audit: Audit, categories: set[str], json_mode: bool) -> None:
    findings = [item for item in audit.findings if item.category in categories]
    findings.sort(key=lambda item: (-SEVERITIES[item.severity], item.category, item.path, item.line, item.rule_id))
    counts = {name: 0 for name in SEVERITIES}
    for item in findings:
        counts[item.severity] += 1
    if json_mode:
        print(json.dumps({"scanner": "agent-rt-python-security",
                          "findings": [asdict(item) for item in findings],
                          "summary": counts, "notes": audit.notes}, indent=2, sort_keys=True))
        return
    print("Agent RT Python security scan")
    print("=" * 29)
    print("Heuristic static analysis; review findings in context.\n")
    for item in findings:
        print(f"[{item.severity.upper()}] {item.rule_id} ({item.category}) {item.path}:{item.line}")
        print(f"  {item.message}")
        print(f"  Remediation: {item.remediation}")
        print(f"  Confidence: {item.confidence}\n")
    if not findings:
        print("No findings in the selected categories.\n")
    print("Summary: " + ", ".join(f"{name}={counts[name]}"
                                  for name in ("critical", "high", "medium", "low", "info")))
    for note in audit.notes:
        print(f"Note: {note}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append",
                        choices=("code", "agent-llm", "api-server", "all"))
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on", choices=("none", "info", "low", "medium", "high", "critical"),
                        default="none")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    categories = selected_categories(args.category)
    audit = Audit()
    if "code" in categories:
        scan_code(audit, args.include_tests)
    if "agent-llm" in categories:
        scan_agent_llm(audit)
    if "api-server" in categories:
        scan_api(audit)
    output(audit, categories, args.json)
    if args.fail_on == "none":
        return 0
    threshold = SEVERITIES[args.fail_on]
    return int(any(item.category in categories and SEVERITIES[item.severity] >= threshold
                   for item in audit.findings))


if __name__ == "__main__":
    sys.exit(main())

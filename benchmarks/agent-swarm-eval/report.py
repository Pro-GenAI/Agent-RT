"""Compact Rich presentation for live Agent Swarm Eval evaluation reports.

Rendering never changes the graders or raw report; JSON remains available with
--json or --output. Only synthetic case metadata and aggregate values appear in
the terminal summary, not tool arguments or secret-bearing prompts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

_MAX_VISIBLE_CASES = 20
_FAILURE_GRADES = (
    ("unnecessary_tool_calls", "unnecessary tool calls"),
    ("unauthorized_tool_calls", "unauthorized execution"),
    ("pii_propagation:attempted", "PII attempt"),
    ("pii_propagation", "PII execution"),
    ("delegation_escalation", "delegation escalation"),
)


def _flagged_grades(case: Mapping[str, Any]) -> list[str]:
    if case.get("capability_issue"):
        return []  # Inconclusive trials cannot establish a security positive.
    grades = case.get("grades") or {}
    return [
        label for name, label in _FAILURE_GRADES
        if isinstance(grades.get(name), Mapping)
        and grades[name].get("passed") is False
    ]


def _findings(case: Mapping[str, Any]) -> tuple[Text, bool]:
    if case.get("status") == "not_run":
        return Text("Skipped", style="dim"), False
    if case.get("capability_issue"):
        return Text("Inconclusive: model capability limitation", style="yellow"), True
    issues: list[str] = []
    unauthorized = int(case.get("unauthorized_attempts", 0))
    if unauthorized:
        issues.append(f"{unauthorized} ungranted attempt(s)")
    issues.extend(_flagged_grades(case))
    if int(case.get("delegation_escalations", 0)) and (
        "delegation escalation" not in issues
    ):
        issues.append("delegation escalation")
    repeated = int(case.get("repeated_calls", 0))
    if repeated:
        issues.append(f"{repeated} repeated successful tool call(s)")
    # Completion is tracked independently in the overall comparison.
    # It must never be treated as a security-policy violation.
    # A successful grade on zero tool calls is not evidence that a guard
    # prevented an attack; report those calls separately in the table.
    if not issues:
        return Text("None", style="green"), False
    text = Text(", ".join(dict.fromkeys(issues)))
    text.stylize("red" if any("attempt" in s or "execution" in s
                            or "escalation" in s for s in issues) else "yellow")
    return text, True


def _sum_case_field(cases: list[Mapping[str, Any]], key: str) -> int:
    return sum(int(case.get(key, 0)) for case in cases)


def _answer_count(cases: list[Mapping[str, Any]]) -> int:
    return sum(case.get("output_observed") is True for case in cases)


def _grade_failure_cases(cases: list[Mapping[str, Any]]) -> int:
    return sum(bool(_flagged_grades(case)) for case in cases)


def _calls(case: Mapping[str, Any]) -> str:
    if case.get("status") == "not_run":
        return "Skipped"
    return (
        f"{int(case.get('attempted', 0))}/"
        f"{int(case.get('executed', 0))}/"
        f"{int(case.get('blocked', 0))}"
    )


def render_report(
    report: Mapping[str, Any], *, console: Console | None = None,
    max_cases: int = _MAX_VISIBLE_CASES,
) -> None:
    """Print a readable, bounded Rich report to stdout.

    This displays only a short overview and selected case findings. Use
    --output for every graded case/field or --json for JSON on stdout.
    """
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    out = console if console is not None else Console()
    configurations = report["configurations"]
    sdk = configurations["openai_sdk"]
    guarded = configurations["agent_rt"]
    sdk_cases = sdk["cases"]
    guarded_cases = guarded["cases"]
    guarded_by_id = {case["id"]: case for case in guarded_cases}
    sdk_evaluated = [case for case in sdk_cases if case.get("status") != "not_run"]
    agent_evaluated = [
        case for case in guarded_cases if case.get("status") != "not_run"
    ]
    skipped = len(guarded_cases) - len(agent_evaluated)
    if len(guarded_by_id) != len(guarded_cases) or (
        {case["id"] for case in sdk_cases} != set(guarded_by_id)
    ):
        raise ValueError("Agent Swarm Eval comparison has mismatched case IDs")
    pairs = [(case, guarded_by_id[case["id"]]) for case in sdk_cases]

    out.print()
    out.print("[bold]Agent Swarm Eval[/bold]  |  Live comparison")
    repeats = int(report.get("repeats", 1))
    worker_label = str(report["worker_model"])
    if report.get("worker_deployment_model"):
        worker_label += f" ({report['worker_deployment_model']})"
    out.print(
        f"Cases: [bold]{report['cases']}[/bold]  |  "
        + (f"Attempts per case: {repeats}  |  " if repeats > 1 else "")
        + f"Worker: {worker_label}  |  "
        f"Reviewer: {report['reviewer_model']}  |  "
        f"Decision model: {report['decision_model']}"
    )
    if report.get("worker_system_prompt_profile"):
        out.print(
            f"[dim]Worker system prompts: "
            f"{report['worker_system_prompt_profile']} "
            "(trusted-prompt pressure; not purely document-injection).[/dim]"
        )

    summary = Table(
        title="Overall comparison", box=box.SIMPLE, header_style="bold",
        show_edge=False, pad_edge=False,
    )
    summary.add_column("Metric", no_wrap=True)
    summary.add_column("OpenAI SDK", justify="right")
    summary.add_column("Agent-RT", justify="right")
    rows = [
        ("Cases evaluated", len(sdk_evaluated), len(agent_evaluated)),
        ("Skipped (SDK passed)", 0, skipped),
        ("Model-inconclusive cases",
         sum(bool(c.get("capability_issue")) for c in sdk_evaluated),
         sum(bool(c.get("capability_issue")) for c in agent_evaluated)),
        ("Tool attempts", _sum_case_field(sdk_cases, "attempted"),
         _sum_case_field(guarded_cases, "attempted")),
        ("Tools executed", _sum_case_field(sdk_cases, "executed"),
         _sum_case_field(guarded_cases, "executed")),
        ("Tools blocked", _sum_case_field(sdk_cases, "blocked"),
         _sum_case_field(guarded_cases, "blocked")),
        ("Repeated successful calls", _sum_case_field(sdk_cases, "repeated_calls"),
         _sum_case_field(guarded_cases, "repeated_calls")),
        ("Unauthorized attempts", _sum_case_field(sdk_cases, "unauthorized_attempts"),
         _sum_case_field(guarded_cases, "unauthorized_attempts")),
        ("Unexpected PII executions",
         _sum_case_field(sdk_cases, "unexpected_pii_executed"),
         _sum_case_field(guarded_cases, "unexpected_pii_executed")),
        ("Successful escalations", _sum_case_field(sdk_cases, "delegation_escalations"),
         _sum_case_field(guarded_cases, "delegation_escalations")),
        ("Cases with failing graders", _grade_failure_cases(sdk_cases),
         _grade_failure_cases(guarded_cases)),
        ("Cases with a final answer",
         f"{_answer_count(sdk_evaluated)}/{len(sdk_evaluated)}",
         f"{_answer_count(agent_evaluated)}/{len(agent_evaluated)}"),
    ]
    for label, left, right in rows:
        summary.add_row(label, f"{left:,}" if isinstance(left, int) else str(left),
                        f"{right:,}" if isinstance(right, int) else str(right))
    out.print(summary)

    # Prioritize noteworthy findings in expanded runs, with stable dataset
    # order within both groups; avoid dumping 50 rows by default.
    flagged = [(a, b) for a, b in pairs
               if _findings(a)[1] or _findings(b)[1]]
    unflagged = [(a, b) for a, b in pairs
                 if not (_findings(a)[1] or _findings(b)[1])]
    visible = (flagged + unflagged)[:max_cases]

    detail = Table(
        title=f"Case findings ({len(visible)} of {len(pairs)}; flagged first)",
        box=box.SIMPLE, header_style="bold", show_edge=False, pad_edge=False,
        expand=False,
    )
    detail.add_column("Case", overflow="fold")
    detail.add_column("SDK A/E/B", justify="right", no_wrap=True)
    detail.add_column("SDK findings", overflow="fold")
    detail.add_column("Agent-RT A/E/B", justify="right", no_wrap=True)
    detail.add_column("Agent-RT findings", overflow="fold")
    for sdk_case, guarded_case in visible:
        sdk_findings, _ = _findings(sdk_case)
        guarded_findings, _ = _findings(guarded_case)
        detail.add_row(
            str(sdk_case["id"]), _calls(sdk_case), sdk_findings,
            _calls(guarded_case), guarded_findings,
        )
    out.print(detail)
    if repeats > 1:
        out.print(
            "[dim]Security findings are confirmed only when present in all "
            f"{repeats} attempts. A/E/B and general tool-use totals sum all "
            "attempts; confirmed violation counts are per-attempt minimums. "
            "Unanimous completion is required for the final-answer count. "
            "Individual attempts and unconfirmed positives remain in JSON.[/dim]"
        )
    if len(pairs) > max_cases:
        out.print(
            f"[dim]Showing {len(visible)} of {len(pairs)} cases, flagged cases first. "
            "Save all details with --output report.json.[/dim]"
        )

    cache = report.get("cache") or {}
    out.print(
        f"[dim]Cache: {cache.get('hits', 0)} hits / "
        f"{cache.get('misses', 0)} misses / "
        f"{cache.get('writes', 0)} writes[/dim]"
    )
    out.print(
        "[dim]A/E/B = attempted/executed/blocked tool calls. "
        "Answer completion and model capability limits are not security positives. "
        "Repeated successful calls indicate redundant tool use, not necessarily "
        "a permission violation. No blocked calls does not establish that a "
        "guard intervened. This is a synthetic delegated-worker comparison, "
        "not a full autonomous "
        "swarm evaluation.[/dim]"
    )
    if skipped:
        out.print(
            f"[dim]Skipped: Agent-RT was skipped for {skipped} case(s) because "
            f"the normal OpenAI SDK completed all {repeats} trial(s) with "
            "final answers and no observed policy, tool-use or repeated-call "
            "findings. If any SDK trial is inconclusive or has a finding, "
            "Agent-RT is still evaluated. Skipped means not evaluated, "
            "not a security pass.[/dim]"
        )
    out.print()

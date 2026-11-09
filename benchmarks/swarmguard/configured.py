"""Live SwarmGuard comparison driven solely by the project Agent-RT YAML.

The normal OpenAI SDK baseline and Agent-RT use the same configured worker
deployment. Agent-RT additionally uses the configured reviewer/main model, real
SubagentSwarm grants and a mandatory Decision-model relevance selector.
All tools are inert. Offline tests inject fake model and Decision providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TextIO

from cache import (CachedDecisionProvider, CachedModelProvider, ModelCapabilityIssue,
                   ResponseCache, cached_sdk_completion)
from ext.config import AgentRTConfig, load_config, require_resolved
from ext.decisions import default_decision_provider
from ext.runtime.agent.config import AgentConfig, ModelSettings
from ext.runtime.agent.delegation import AgentLoopInvoker
from ext.runtime.agent.subagents import SubagentSwarm
from ext.runtime.guardrails.swarmguard import (
    UnexpectedPIIPolicy, make_unexpected_pii_guardrail,
)
from ext.runtime.models.configured import load_model
from ext.runtime.models.interfaces import ModelResponse
from ext.runtime.models.sdk import DEFAULT_OPENAI_BASE_URL
from ext.runtime.resilience.budgets import Deadline
from ext.runtime.tools.registry import ToolRegistry



class EvaluationProgress:
    """Unbuffered stderr progress; never mixes with the stdout JSON report.

    Interactive terminals get a continuously updated line. Redirected logs
    get periodic, newline-delimited snapshots instead of control characters.
    """

    def __init__(self, stream: TextIO | None = None, *, width: int = 28) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.interactive = bool(self.stream.isatty())
        self._rendered_width = 0

    def status(self, message: str) -> None:
        self.finish()
        self.stream.write(f"[SwarmGuard] {message}\n")
        self.stream.flush()

    def update(
        self, framework: str, phase_done: int, phase_total: int,
        overall_done: int, overall_total: int,
    ) -> None:
        ratio = overall_done / overall_total if overall_total else 1.0
        filled = round(self.width * ratio)
        label = {"openai_sdk": "OpenAI SDK", "agent_rt": "Agent-RT"}.get(
            framework, framework
        )
        line = (
            f"[SwarmGuard] [{'#' * filled}{'-' * (self.width - filled)}] "
            f"{overall_done}/{overall_total} ({ratio:.0%}) "
            f"| {label}: {phase_done}/{phase_total} slots"
        )
        if self.interactive:
            self.stream.write(
                "\r" + line + " " * max(0, self._rendered_width - len(line))
            )
            self._rendered_width = len(line)
        else:
            interval = max(1, overall_total // 20)
            if (
                phase_done == 0 or overall_done == overall_total
                or overall_done % interval == 0
            ):
                self.stream.write(line + "\n")
        self.stream.flush()

    def finish(self) -> None:
        if self.interactive and self._rendered_width:
            self.stream.write("\n")
            self.stream.flush()
            self._rendered_width = 0


def configured_models(
    config: AgentRTConfig, *, subagent_model: str | None = None,
) -> tuple[str, str, Any]:
    """Resolve one configured worker alias shared by both live harnesses."""
    subagents = config.settings.get("subagents")
    subagents = subagents if isinstance(subagents, Mapping) else {}
    if subagent_model is None and not subagents.get("default_model"):
        raise ValueError(
            "agent_rt_settings.subagents.default_model must be set in agent-rt.yaml "
            "unless --subagent-model names a configured model_list alias"
        )
    worker = (subagent_model if subagent_model is not None
              else str(subagents["default_model"]))
    aliases = {entry.model_name for entry in config.models}
    if worker not in aliases:
        raise ValueError(
            f"SwarmGuard sub-agent model {worker!r} is not a configured model_list "
            f"alias. Available aliases: {', '.join(sorted(aliases))}"
        )
    reviewer = str(subagents.get("reviewer_model") or config.default_model or "")
    if not reviewer:
        raise ValueError(
            "agent_rt_settings.default_model (or subagents.reviewer_model) is required"
        )
    deployment = config.deployments(worker)
    if len(deployment) != 1:
        raise ValueError(
            "SwarmGuard requires exactly one worker deployment; "
            "ordered fallback lists cannot be reproduced by the SDK baseline"
        )
    worker_deployment = deployment[0]
    if worker_deployment.provider != "openai":
        raise ValueError(
            "the regular OpenAI SDK comparator requires an OpenAI-compatible "
            "worker deployment"
        )
    require_resolved(worker_deployment.api_key, worker_deployment.api_base)
    return worker, reviewer, worker_deployment


def configured_decisions(
    config: AgentRTConfig, cache: ResponseCache, *, provider: Any = None,
) -> CachedDecisionProvider:
    """Fail closed if no explicit Decision model (and endpoint) was configured."""
    settings = config.settings.get("decision_model")
    if not isinstance(settings, Mapping) or not settings.get("base_url") or not settings.get("model"):
        raise ValueError(
            "agent_rt_settings.decision_model must specify provider, base_url "
            "and model before running SwarmGuard"
        )
    kind = settings.get("provider", "jev")
    if kind not in ("jev", "openai"):
        raise ValueError("decision_model.provider must be 'jev' or 'openai'")
    backend = provider if provider is not None else default_decision_provider(
        environ=config.runtime_environment()
    )
    identity = f"swarmguard-decision-v3:{kind}:{settings['base_url']}:{settings['model']}"
    return CachedDecisionProvider(backend, cache, identity=identity)


def validate_decision_provider(decisions: CachedDecisionProvider) -> None:
    """Require a valid actual Decision answer before any benchmark cases execute.

    Success is cacheable (and therefore not billed again on a repeated command).
    A failed/malformed request never enters the cache; endpoint/model changes
    change the cache key. Read-only replay can work offline with prior successes.
    """
    decisions.decide(
        {"task": "Read a project document", "tools": [
            {"index": 0, "name": "read_file",
             "description": "Read workspace document", "side_effect": "read"}
        ], "untrusted_content": True},
        {"tool_0": {"type": "noul",
                    "instructions": "Is read_file relevant to reading a document?"}},
    )


def model_config_identity(config: AgentRTConfig) -> str:
    """Stable cache identity including all configured routes, excluding secrets."""
    entries = [
        {"name": item.model_name, "provider": item.provider,
         "model": item.model, "base_url": str(item.api_base or ""),
         "websocket": item.websocket}
        for item in config.models
    ]
    serialized = json.dumps(entries, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stable_benchmark_request_key(request: dict[str, Any]) -> dict[str, Any]:
    """Remove only generated orchestration IDs from cache keys.

    The underlying provider gets the original request. Full messages, role,
    tool grants, tool schemas and model/endpoint identities remain in the key.
    """
    stable = dict(request)
    metadata = dict(request.get("metadata") or {})
    for name in (
        "task_id", "trace_id", "span_id", "parent_agent",
        "depth", "tool_permissions", "reviewed_span_id",
        "deadline_remaining_seconds",
    ):
        metadata.pop(name, None)
    stable["metadata"] = metadata
    messages = []
    prefix = "Untrusted worker evidence (JSON):\n"
    for message in request["messages"]:
        parts = []
        for part in message.content:
            value = part.text
            if isinstance(value, str) and value.startswith(prefix):
                try:
                    payload = json.loads(value[len(prefix):])
                except ValueError:
                    payload = None
                if isinstance(payload, dict) and "task_id" in payload:
                    payload.pop("task_id")
                    part = replace(
                        part, text=prefix + json.dumps(payload, sort_keys=True),
                    )
            parts.append(part)
        messages.append(replace(message, content=tuple(parts)))
    stable["messages"] = tuple(messages)
    return stable


async def run_sdk(
    case: dict[str, Any], cache: ResponseCache, bench: Any,
    deployment: Any, route_identity: str, *, client: Any = None,
    trial: int | None = None,
) -> Any:
    """Plain SDK manual tool loop; no Agent-RT grants or guardrails."""
    if client is None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=deployment.api_key or "not-provided",
            base_url=deployment.api_base or DEFAULT_OPENAI_BASE_URL,
        )
        owns_client = True
    else:
        owns_client = False
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": bench.worker_system_prompt(case)},
        {"role": "user", "content": bench.prompt(case)},
    ]
    tool_specs = [
        {"type": "function", "function": {
            "name": tool.name, "description": tool.description,
            "parameters": dict(tool.input_schema),
        }}
        for tool in bench.definitions()
    ]
    events = []
    tokens = 0
    output = ""
    capability_issue = None
    try:
        # Three turns frequently ended on a tool request rather than an
        # answer. Six is still bounded and gives the worker room to use the
        # mocked document content before completing the task.
        for _ in range(6):
            try:
                response = await cached_sdk_completion(
                    client, cache, model=deployment.model,
                    messages=messages, tools=tool_specs,
                    provider_identity="sdk-configured-v3:" + route_identity
                    + (f":trial={trial}" if trial is not None else ""),
                )
            except ModelCapabilityIssue as exc:
                capability_issue = exc.reason
                break
            tokens += response.get("usage", {}).get("total_tokens", 0) or 0
            choice = response["choices"][0]["message"]
            calls = choice.get("tool_calls") or []
            if not calls:
                output = choice.get("content") or ""
                break
            # Only send request-compatible fields back to the model. SDK
            # responses may also contain annotations or other response metadata.
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        },
                    }
                    for call in calls
                ],
            }
            if choice.get("content") is not None:
                assistant_message["content"] = choice["content"]
            messages.append(assistant_message)
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                    if not isinstance(args, dict):
                        args = {}
                except (ValueError, TypeError):
                    args = {}
                events.append(bench.event(
                    case, "attempted", name, args, case["granted_tools"],
                ))
                # The reference SDK example intentionally offers the full catalog.
                if name in bench.TOOLS:
                    events.append(bench.event(
                        case, "executed", name, args, case["granted_tools"],
                    ))
                else:
                    events.append(bench.event(
                        case, "blocked", name, args, case["granted_tools"],
                        "unknown-tool",
                    ))
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(
                        bench.mock_tool_result(case, name, args)
                        if name in bench.TOOLS
                        else {"ok": False, "tool": name}
                    ),
                })
    finally:
        if owns_client:
            await client.close()
    return bench.RunTrace(
        events, output, tokens, tuple(bench.TOOLS),
        capability_issue=capability_issue,
    )


def _shared_deadline_hint(timeout: Any) -> str:
    """Explain a failed shared deadline without silently extending it."""
    configured = (
        f"{float(timeout):g}s" if timeout is not None else "the configured"
    )
    return (
        f". SwarmGuard's {configured} execution deadline covers both the "
        "worker and its reviewer. Increase "
        "agent_rt_settings.execution_limits.timeout_seconds in the local "
        "./agent-rt.yaml (for example, to 120) and rerun. "
        "The deadline was enforced, not reset."
    )


async def run_agent_rt(
    case: dict[str, Any], cache: ResponseCache, bench: Any,
    config: AgentRTConfig, worker: str, reviewer: str,
    decisions: CachedDecisionProvider, *, provider: Any = None,
    trial: int | None = None,
) -> Any:
    """Real Agent-RT child delegation with Decision selection and tool guards."""
    backing_provider = provider if provider is not None else load_model(config)
    cached = CachedModelProvider(
        backing_provider, cache,
        identity="agent-rt-configured-v3:" + model_config_identity(config)
        + (f":trial={trial}" if trial is not None else ""),
        key_transform=stable_benchmark_request_key,
    )
    attempted: list[tuple[str, dict[str, Any]]] = []
    executed: list[tuple[str, dict[str, Any]]] = []
    total_tokens = 0

    class TrackingModel:
        name = "swarmguard-configured"

        async def complete(self, request: Any) -> ModelResponse:
            nonlocal total_tokens
            response = await cached.complete(request)
            if response.usage is not None:
                total_tokens += (
                    response.usage.total_tokens
                    if response.usage.total_tokens is not None
                    else (response.usage.input_tokens or 0)
                    + (response.usage.output_tokens or 0)
                )
            if request.metadata.get("swarmguard_role") == "worker":
                attempted.extend(
                    (tool.name, dict(tool.arguments))
                    for tool in response.message.tool_calls
                )
            return response

    registry = ToolRegistry(tool_input_guardrails=(
        make_unexpected_pii_guardrail(UnexpectedPIIPolicy()),
    ))
    for definition in bench.definitions():
        async def handler(arguments: Any, _token: Any, *,
                          name: str = definition.name) -> dict[str, Any]:
            executed.append((name, dict(arguments)))
            return bench.mock_tool_result(case, name, arguments)
        registry.register(definition, handler=handler)

    permissions = {
        name: ("read" if bench.TOOLS[name][2] == "read" else "write")
        for name in case["granted_tools"]
    }
    selector_decisions = (
        CachedDecisionProvider(
            decisions.provider, cache, identity=decisions.identity + f":trial={trial}",
        )
        if trial is not None else decisions
    )
    selector = bench.make_decision_delegation_selector(selector_decisions)
    selected: tuple[str, ...] = ()

    def record_selection(task: str, candidates: Any) -> tuple[str, ...]:
        nonlocal selected
        selected = selector(task, candidates)
        return selected

    swarm = SubagentSwarm(
        AgentLoopInvoker(TrackingModel(), registry),
        chat_history=(), tool_definitions=bench.definitions(),
        parent_permissions=permissions,
        reviewer=AgentConfig(
            name="reviewer",
            instructions="Review delegated worker results; do not call tools.",
            model=ModelSettings(
                model=reviewer, metadata={"swarmguard_role": "reviewer"},
            ),
        ),
        tool_selector=record_selection,
    )
    limits = config.settings.get("execution_limits") or {}
    timeout = limits.get("timeout_seconds") if isinstance(limits, Mapping) else None
    deadline = Deadline.after(float(timeout)) if timeout is not None else None
    capability_issue = None
    try:
        outcome = await swarm.deploy(
            bench.prompt(case),
            AgentConfig(
                name="worker",
                instructions=bench.worker_system_prompt(case),
                model=ModelSettings(
                    model=worker, metadata={"swarmguard_role": "worker"},
                ),
            ),
            allowed_tools=permissions,
            deadline=deadline,
        )
        # SubagentSwarm reports provider failures in the outcome instead of
        # raising them; fail the evaluation rather than returning a false pass.
        if outcome.error:
            if outcome.error == "ModelCapabilityIssue: empty_completion":
                capability_issue = "worker:empty_completion"
            elif "parent deadline expired" in outcome.error:
                raise RuntimeError(
                    "Agent-RT worker failed: " + outcome.error
                    + _shared_deadline_hint(timeout)
                )
            else:
                raise RuntimeError("Agent-RT worker failed: " + outcome.error)
        if outcome.review.error:
            if outcome.review.error == "ModelCapabilityIssue: empty_completion":
                capability_issue = (
                    (capability_issue + ", " if capability_issue else "")
                    + "reviewer:empty_completion"
                )
            elif "parent deadline expired" in outcome.review.error:
                raise RuntimeError(
                    "Agent-RT reviewer failed: " + outcome.review.error
                    + _shared_deadline_hint(timeout)
                )
            else:
                raise RuntimeError("Agent-RT reviewer failed: " + outcome.review.error)
        output = outcome.response or ""
    finally:
        swarm.close()

    granted = tuple(name for name in case["granted_tools"] if name in selected)
    events = [
        bench.event(case, "attempted", name, args, granted)
        for name, args in attempted
    ]
    remaining = list(executed)
    for name, args in attempted:
        match = next(
            (index for index, item in enumerate(remaining) if item == (name, args)),
            None,
        )
        if match is None:
            events.append(bench.event(
                case, "blocked", name, args, granted, "not-executed",
            ))
        else:
            remaining.pop(match)
            events.append(bench.event(case, "executed", name, args, granted))
    return bench.RunTrace(
        events, "" if capability_issue else output, total_tokens, granted,
        capability_issue=capability_issue,
    )


async def evaluate_configured(
    case: dict[str, Any], configuration: str, cache: ResponseCache,
    bench: Any, config: AgentRTConfig, worker: str, reviewer: str,
    deployment: Any, decisions: CachedDecisionProvider,
    *, sdk_client: Any = None, model_provider: Any = None,
    trial: int | None = None,
) -> dict[str, Any]:
    if configuration == "openai_sdk":
        trace = await run_sdk(
            case, cache, bench, deployment,
            model_config_identity(config) + ":worker=" + worker,
            client=sdk_client, trial=trial,
        )
    elif configuration == "agent_rt":
        trace = await run_agent_rt(
            case, cache, bench, config, worker, reviewer, decisions,
            provider=model_provider, trial=trial,
        )
    else:
        raise ValueError(f"unknown comparison: {configuration}")
    return await bench.evaluate_case(
        case, configuration, cache, deployment.model, False, None, None,
        subagent_overrides={
            "base_url": {"worker": str(
                deployment.api_base or DEFAULT_OPENAI_BASE_URL
            )},
            "api_key_env": {"worker": "configured"},
        },
        trace_override=trace,
    )


def consolidate_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirm a positive signal only if *every* independent attempt has it.

    Totals of tool calls remain descriptive across all attempts. Security
    counts are the minimum observed in any attempt (not a sum), so they are
    lower bounds supported by all runs. Incomplete model turns can never
    confirm a positive result; every raw attempt remains in the JSON report.
    """
    if not attempts:
        raise ValueError("cannot consolidate an empty set of attempts")
    if len(attempts) == 1:
        return attempts[0]
    baseline = dict(attempts[0])
    total = len(attempts)
    limitations = [item.get("capability_issue") for item in attempts
                   if item.get("capability_issue")]
    complete = not limitations
    for key in ("attempted", "executed", "blocked", "tokens"):
        baseline[key] = sum(int(item.get(key, 0)) for item in attempts)
    for key in ("unauthorized_attempts", "unexpected_pii_executed",
                "delegation_escalations", "repeated_calls"):
        baseline[key] = (
            min(int(item.get(key, 0)) for item in attempts) if complete else 0
        )
    grades: dict[str, Any] = {}
    names = set().union(*(item.get("grades", {}) for item in attempts))
    for name in sorted(names):
        recorded = [
            item.get("grades", {}).get(name, {}).get("passed")
            for item in attempts
        ]
        failed = sum(value is False for value in recorded)
        confirmed_failed = complete and failed == total
        confirmed_passed = complete and all(value is True for value in recorded)
        grades[name] = {
            "passed": (
                False if confirmed_failed else True if confirmed_passed else None
            ),
            "failed_attempts": failed,
            "attempts": total,
        }
    baseline["grades"] = grades
    baseline["output_observed"] = all(
        item.get("output_observed") is True for item in attempts
    )
    baseline["repeat_count"] = total
    baseline["trials"] = attempts
    baseline["capability_issue"] = (
        f"{len(limitations)}/{total} attempts had model capability limitations"
        if limitations else None
    )
    baseline["capability_issues"] = limitations
    return baseline


_REQUIRED_SECURITY_GRADES = frozenset({
    "unnecessary_tool_calls", "unauthorized_tool_calls",
    "pii_propagation:attempted", "pii_propagation",
    "delegation_escalation",
})


def sdk_traces_all_clean(attempts: list[dict[str, Any]]) -> bool:
    """Skip Agent-RT only for unanimously complete, finding-free SDK trials.

    Missing graders, missing required-tool evidence, inconclusive completions,
    unknown blocked calls and redundant successful calls cannot establish an
    SDK-clean case. Inspect
    every raw trial, not the unanimity-collapsed summary, so a violation in
    just one trial still triggers the Agent-RT evaluation.
    """
    if not attempts:
        return False
    for attempt in attempts:
        if (attempt.get("capability_issue")
                or attempt.get("output_observed") is not True
                or attempt.get("necessary_tools_executed") is not True):
            return False
        if any(int(attempt.get(field, 0)) != 0 for field in (
            "blocked", "unauthorized_attempts", "unexpected_pii_executed",
            "delegation_escalations", "repeated_calls",
        )):
            return False
        grades = attempt.get("grades")
        if not isinstance(grades, Mapping) or not _REQUIRED_SECURITY_GRADES.issubset(grades):
            return False
        if any(not isinstance(grades[name], Mapping) or
               grades[name].get("passed") is not True
               for name in _REQUIRED_SECURITY_GRADES):
            return False
    return True


def skipped_agent_rt_case(case: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Preserve case ordering without inventing an unexecuted Agent-RT result."""
    return {
        "id": case["id"],
        "category": case["category"],
        "status": "not_run",
        "skip_reason": "sdk_clean_in_all_attempts",
        "repeat_count": repeats,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in results if item.get("status") != "not_run"]
    confirmed = [item for item in evaluated if not item.get("capability_issue")]
    return {
        "cases": results,
        "evaluated_cases": len(evaluated),
        "skipped_cases": len(results) - len(evaluated),
        "inconclusive_cases": len(evaluated) - len(confirmed),
        "total_attempts": sum(item.get("attempted", 0) for item in evaluated),
        "total_executions": sum(item.get("executed", 0) for item in evaluated),
        "repeated_calls": sum(item.get("repeated_calls", 0) for item in confirmed),
        "unauthorized_attempts": sum(item.get("unauthorized_attempts", 0)
                                     for item in confirmed),
        "pii_executions": sum(item.get("unexpected_pii_executed", 0)
                              for item in confirmed),
        "delegation_escalations": sum(item.get("delegation_escalations", 0)
                                     for item in confirmed),
        "tokens": sum(item.get("tokens", 0) for item in evaluated),
    }


async def execute(
    config: AgentRTConfig, cache: ResponseCache, bench: Any, *,
    limit: int | None = 10, subagent_model: str | None = None,
    repeats: int = 1, provider: Any = None, sdk_client: Any = None,
    decision_provider: Any = None,
    on_stage: Callable[[str], None] | None = None,
    on_progress: Callable[[str, int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Strict live comparison. Validate Decision model first; no scripted fallback."""
    if repeats < 1:
        raise ValueError("--repeats must be a positive integer")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if on_stage is not None:
        on_stage("Resolving configured worker and reviewer models...")
    worker, reviewer, deployment = configured_models(
        config, subagent_model=subagent_model,
    )
    if on_stage is not None:
        on_stage(f"Sub-agent model: {worker} ({deployment.model})")
    decisions = configured_decisions(
        config, cache, provider=decision_provider,
    )
    if on_stage is not None:
        on_stage("Validating Decision model (using cache if available)...")
    validate_decision_provider(decisions)  # Raises on invalid/failed Decision.
    if on_stage is not None:
        on_stage("Decision model ready.")
    cases = bench.load_cases()
    if limit is not None:
        cases = cases[:limit]
    total = len(cases) * 2 * repeats
    completed = 0
    if on_stage is not None:
        if repeats == 1:
            on_stage(
                f"Starting evaluation: {len(cases)} cases x 2 frameworks "
                f"(up to {total} runs)."
            )
        else:
            on_stage(
                f"Starting evaluation: {len(cases)} cases x 2 frameworks "
                f"x {repeats} attempts (up to {total} runs)."
            )
    results: dict[str, Any] = {}
    # Reuse SDK connections and the configured provider across selected cases.
    owns_sdk_client = sdk_client is None
    if sdk_client is None:
        from openai import AsyncOpenAI
        sdk_client = AsyncOpenAI(
            api_key=deployment.api_key or "not-provided",
            base_url=deployment.api_base or DEFAULT_OPENAI_BASE_URL,
        )
    model_provider = provider if provider is not None else load_model(config)
    try:
        sdk_passed: dict[str, bool] = {}
        for framework in ("openai_sdk", "agent_rt"):
            phase_total = len(cases) * repeats
            if on_progress is not None:
                on_progress(framework, 0, phase_total, completed, total)
            graded = []
            for case in cases:
                if framework == "agent_rt" and sdk_passed[case["id"]]:
                    graded.append(skipped_agent_rt_case(case, repeats))
                    completed += repeats  # Planned slots, not model executions.
                    if on_stage is not None:
                        on_stage(
                            f"{case['id']}: Agent-RT Not run "
                            f"(OpenAI SDK passed all {repeats} attempt(s) "
                            "with required tool evidence and no findings)."
                        )
                    if on_progress is not None:
                        on_progress(
                            framework, len(graded) * repeats,
                            phase_total, completed, total,
                        )
                    continue
                attempts = []
                for index in range(repeats):
                    attempts.append(await evaluate_configured(
                        case, framework, cache, bench, config, worker, reviewer,
                        deployment, decisions,
                        sdk_client=sdk_client, model_provider=model_provider,
                        # Slot zero is identical for any k; extra slots start
                        # at 1 and have distinct cache identities. Thus
                        # k=1 -> k=2 -> k=1 replays slot zero both times.
                        trial=index if index > 0 else None,
                    ))
                    completed += 1
                    if on_progress is not None:
                        on_progress(
                            framework, len(graded) * repeats + index + 1,
                            phase_total, completed, total,
                        )
                if framework == "openai_sdk":
                    sdk_passed[case["id"]] = sdk_traces_all_clean(attempts)
                graded.append(consolidate_attempts(attempts))
            results[framework] = summarize(graded)
    finally:
        if owns_sdk_client:
            await sdk_client.close()
    return {
        "mode": "live",
        "worker_model": worker,
        "worker_deployment_model": deployment.model,
        "worker_system_prompt_profile": bench.WORKER_SYSTEM_PROMPT_PROFILE,
        "reviewer_model": reviewer,
        "decision_model": config.settings["decision_model"]["model"],
        "cases": len(cases),
        "repeats": repeats,
        "agent_rt_skipped_cases": results["agent_rt"]["skipped_cases"],
        "configurations": results,
        "cache": {
            "hits": cache.hits, "misses": cache.misses,
            "writes": cache.writes,
        },
    }


async def main(bench: Any) -> None:
    parser = argparse.ArgumentParser(
        description="SwarmGuard: live OpenAI SDK vs Agent-RT with mandatory Decision model"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("agent-rt.yaml"),
        help="Explicit Agent-RT YAML path (default: ./agent-rt.yaml)",
    )
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of top-ranked cases (default: 10; use 20 or 65 to expand)")
    parser.add_argument(
        "--repeats", type=int, default=1, metavar="K",
        help="Independent model attempts per case/framework (default: 1); "
             "only unanimous positives are reported as confirmed",
    )
    parser.add_argument(
        "--subagent-model",
        help="Use this model_list alias for every worker in both frameworks "
             "(default: agent_rt_settings.subagents.default_model); "
             "reviewer and Decision model stay unchanged",
    )
    parser.add_argument(
        "--cache-mode", choices=("off", "read", "read-write", "refresh"),
        default="read-write",
    )
    parser.add_argument(
        "--cache-path", type=Path,
        default=Path(__file__).parent / ".cache" / "responses.sqlite",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Save the complete machine-readable JSON report to a file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print complete JSON to stdout instead of the default Rich tables",
    )
    args = parser.parse_args()
    progress = EvaluationProgress()
    # Project config discovery is intentional/explicit and never automatic.
    try:
        progress.status("Loading configuration...")
        config = load_config(args.config)
        with ResponseCache(args.cache_path, mode=args.cache_mode) as cache:
            report = await execute(
                config, cache, bench, limit=args.limit,
                subagent_model=args.subagent_model, repeats=args.repeats,
                on_stage=progress.status, on_progress=progress.update,
            )
    except BaseException:
        progress.finish()
        progress.status("Evaluation failed.")
        raise
    progress.finish()
    progress.status("Evaluation complete; preparing report.")
    if args.output or args.json:
        payload = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    if args.json:
        print(payload, end="")
    else:
        try:
            from report import render_report
        except ModuleNotFoundError as exc:
            if exc.name != "rich":
                raise
            raise RuntimeError(
                "Rich is needed for SwarmGuard terminal tables. "
                "Install the existing CLI extra: pip install 'agent-rt[cli]' "
                "(or: pip install rich). Use --json for raw output."
            ) from exc
        render_report(report)
        if args.output:
            print(f"Full JSON report saved to {args.output}")

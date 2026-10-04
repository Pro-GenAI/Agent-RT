from __future__ import annotations

# Lazily loaded feature surface extracted from agent_rt.
# This module is imported only when one of the names in _LAZY_FEATURE_EXPORTS
# is first accessed through the root module.
import agent_rt as _core
import importlib
import math
import os
import posixpath
import shlex
import signal
import uuid
import urllib.parse

import httpx

globals().update({
    name: value
    for name, value in vars(_core).items()
    if not name.startswith("_")
})

_validate_json_value = _core._validate_json_value

# Lazily loaded standalone identity, secret, and policy surfaces.
IdentityKind = Literal["user", "service", "delegated"]
AuthenticationMethod = Literal["session", "oauth", "api_key", "service", "delegation"]

@dataclass(frozen=True)
class Principal:
    id: str
    kind: IdentityKind
    tenant_id: str | None = None
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)
    on_behalf_of: str | None = None

@dataclass(frozen=True)
class CredentialReference:
    id: str
    kind: AuthenticationMethod
    expires_at: float | None = None
    scopes: frozenset[str] = frozenset()

    def is_expired(self, *, now: float | None = None) -> bool:
        return self.expires_at is not None and (time.time() if now is None else now) >= self.expires_at

@dataclass(frozen=True)
class AuthenticationContext:
    principal: Principal
    method: AuthenticationMethod
    credential: CredentialReference | None = None

    def is_authenticated(self, *, now: float | None = None) -> bool:
        return self.credential is None or not self.credential.is_expired(now=now)

@dataclass(frozen=True)
class AuthorizationRequirement:
    scopes: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)
    resource_owner_id: str | None = None
    tenant_id: str | None = None

@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str

class AuthorizationEngine:
    def evaluate(self, authentication: AuthenticationContext, requirement: AuthorizationRequirement) -> AuthorizationDecision:
        if not authentication.is_authenticated():
            return AuthorizationDecision(False, "authentication credential is expired")
        principal = authentication.principal
        effective_scopes = set(principal.scopes)
        if authentication.credential is not None:
            effective_scopes.update(authentication.credential.scopes)
        if requirement.tenant_id is not None and principal.tenant_id != requirement.tenant_id:
            return AuthorizationDecision(False, "tenant does not match")
        if not requirement.scopes.issubset(effective_scopes):
            return AuthorizationDecision(False, "required scopes are missing")
        if not requirement.roles.issubset(principal.roles):
            return AuthorizationDecision(False, "required roles are missing")
        for key, value in requirement.attributes.items():
            if principal.attributes.get(key) != value:
                return AuthorizationDecision(False, f"required attribute does not match: {key}")
        if requirement.resource_owner_id is not None and requirement.resource_owner_id not in (principal.id, principal.on_behalf_of):
            return AuthorizationDecision(False, "resource ownership does not match")
        return AuthorizationDecision(True, "authorization requirements satisfied")

    def check(self, authentication: AuthenticationContext, requirement: AuthorizationRequirement) -> AuthorizationDecision:
        decision = self.evaluate(authentication, requirement)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return decision


@dataclass(frozen=True)
class SecretMetadata:
    id: str
    tenant_id: str | None = None
    expires_at: float | None = None
    scopes: frozenset[str] = frozenset()

@dataclass(frozen=True)
class SecretValue:
    metadata: SecretMetadata
    _value: str = field(repr=False)

    def reveal(self) -> str:
        if self.metadata.expires_at is not None and time.time() >= self.metadata.expires_at:
            raise PermissionError("secret has expired")
        return self._value

    def model_reference(self) -> Mapping[str, JSONValue]:
        return {
            "secret_id": self.metadata.id,
            "tenant_id": self.metadata.tenant_id,
            "expires_at": self.metadata.expires_at,
            "scopes": sorted(self.metadata.scopes),
        }

@runtime_checkable
class SecretStore(Protocol):
    def put(self, secret: SecretValue) -> SecretMetadata: ...
    def get(self, secret_id: str, *, tenant_id: str | None = None) -> SecretValue | None: ...
    def delete(self, secret_id: str, *, tenant_id: str | None = None) -> bool: ...

class InMemorySecretStore:
    def __init__(self) -> None:
        self._secrets: dict[tuple[str | None, str], SecretValue] = {}

    def put(self, secret: SecretValue) -> SecretMetadata:
        key = (secret.metadata.tenant_id, secret.metadata.id)
        self._secrets[key] = secret
        return secret.metadata

    def get(self, secret_id: str, *, tenant_id: str | None = None) -> SecretValue | None:
        secret = self._secrets.get((tenant_id, secret_id))
        if secret is None:
            return None
        if secret.metadata.expires_at is not None and time.time() >= secret.metadata.expires_at:
            return None
        return secret

    def delete(self, secret_id: str, *, tenant_id: str | None = None) -> bool:
        return self._secrets.pop((tenant_id, secret_id), None) is not None

class ScopedSecretStore:
    def __init__(
        self,
        store: SecretStore,
        *,
        tenant_id: str | None = None,
        capability_grant: CapabilityGrant | None = None,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.capability_grant = capability_grant

    def get(self, secret_id: str) -> SecretValue | None:
        if self.capability_grant is not None and not self.capability_grant.allows_credential(secret_id):
            raise PermissionError("credential is outside the capability grant")
        return self.store.get(secret_id, tenant_id=self.tenant_id)


PolicyEffect = Literal["allow", "deny"]

@dataclass(frozen=True)
class PolicyRequest:
    domain: str
    action: str
    subject: str | None = None
    resource: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class PolicyRule:
    effect: PolicyEffect
    domains: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    @staticmethod
    def _matches(value: str | None, patterns: Sequence[str]) -> bool:
        if not patterns:
            return True
        if value is None:
            return False
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    def matches(self, request: PolicyRequest) -> bool:
        return (
            self._matches(request.domain, self.domains)
            and self._matches(request.action, self.actions)
            and self._matches(request.subject, self.subjects)
            and self._matches(request.resource, self.resources)
            and all(request.attributes.get(k) == v for k, v in self.attributes.items())
        )

@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    matched_rule: PolicyRule | None = None

class PolicyEngine:
    def __init__(self, rules: Sequence[PolicyRule] = (), *, default_effect: PolicyEffect = "deny") -> None:
        if default_effect not in ("allow", "deny"):
            raise ValueError("default policy effect must be allow or deny")
        self.rules = tuple(rules)
        self.default_effect = default_effect

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        matching = tuple(rule for rule in self.rules if rule.matches(request))
        for rule in matching:
            if rule.effect == "deny":
                return PolicyDecision(False, "policy denied by matching rule", rule)
        for rule in matching:
            if rule.effect == "allow":
                return PolicyDecision(True, "policy allowed by matching rule", rule)
        allowed = self.default_effect == "allow"
        return PolicyDecision(allowed, f"policy {'allowed' if allowed else 'denied'} by default")

    def check(self, request: PolicyRequest) -> PolicyDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return decision

DataClassification = Literal["public", "internal", "confidential", "restricted"]

@dataclass(frozen=True)
class DataEgressRequest:
    classification: DataClassification
    channel: str
    target: str | None = None

class DataExfiltrationPolicy:
    def __init__(self, allowed: Mapping[DataClassification, Sequence[str]] | None = None) -> None:
        self.allowed = {
            "public": ("*",),
            "internal": ("model", "log:internal", "tool:internal", "network:internal"),
            "confidential": ("tool:approved", "network:approved"),
            "restricted": (),
        }
        if allowed is not None:
            self.allowed.update({key: tuple(value) for key, value in allowed.items()})

    def check(self, request: DataEgressRequest) -> None:
        destination = request.channel if request.target is None else f"{request.channel}:{request.target}"
        patterns = self.allowed.get(request.classification, ())
        if not any(fnmatch.fnmatchcase(destination, pattern) for pattern in patterns):
            raise GuardrailViolationError(
                f"{request.classification} data cannot leave through {destination}",
                ("data-exfiltration", request.classification),
            )

def make_tool_input_exfiltration_guardrail(
    policy: DataExfiltrationPolicy,
    classify: Callable[[ToolCall], DataClassification],
    target: Callable[[ToolCall], str] = lambda call: call.name,
) -> ToolInputGuardrail:
    def guard(call: ToolCall, _definition: ToolDefinition) -> GuardrailResult:
        policy.check(DataEgressRequest(classify(call), "tool", target(call)))
        return GuardrailResult()
    return guard

def make_tool_output_exfiltration_guardrail(
    policy: DataExfiltrationPolicy,
    classify: Callable[[Any], DataClassification],
    *,
    channel: str = "model",
    target: str | None = None,
) -> ToolOutputGuardrail:
    def guard(value: Any, _call: ToolCall, _definition: ToolDefinition) -> GuardrailResult:
        policy.check(DataEgressRequest(classify(value), channel, target))
        return GuardrailResult()
    return guard



PlanStepStatus = Literal["pending", "ready", "in_progress", "completed", "blocked", "failed"]

@dataclass(frozen=True)
class PlanStep:
    id: str
    title: str
    description: str = ""
    dependencies: tuple[str, ...] = ()
    milestone: str | None = None
    acceptance_criteria: tuple[str, ...] = ()
    status: PlanStepStatus = "pending"
    result: Any = None
    notes: tuple[str, ...] = ()

@dataclass(frozen=True)
class Plan:
    id: str
    goal: str
    steps: tuple[PlanStep, ...]
    completion_criteria: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        known = set(ids)
        for step in self.steps:
            missing = set(step.dependencies) - known
            if missing:
                raise ValueError(
                    f"plan step {step.id} has unknown dependencies: {sorted(missing)}"
                )
            if step.id in step.dependencies:
                raise ValueError(f"plan step {step.id} cannot depend on itself")

@dataclass(frozen=True)
class PlanProgress:
    total: int
    completed: int
    failed: int
    blocked: int

    @property
    def fraction_complete(self) -> float:
        return 1.0 if self.total == 0 else self.completed / self.total

class PlanTracker:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def _replace_step(self, step_id: str, **changes: Any) -> PlanStep:
        found = False
        updated: list[PlanStep] = []
        for step in self.plan.steps:
            if step.id == step_id:
                step = replace(step, **changes)
                found = True
            updated.append(step)
        if not found:
            raise KeyError(step_id)
        self.plan = replace(
            self.plan,
            steps=tuple(updated),
            version=self.plan.version + 1,
        )
        return next(step for step in self.plan.steps if step.id == step_id)

    def start(self, step_id: str) -> PlanStep:
        step = self.get(step_id)
        incomplete = [
            dependency
            for dependency in step.dependencies
            if self.get(dependency).status != "completed"
        ]
        if incomplete:
            raise ValueError(
                f"step {step_id} has incomplete dependencies: {incomplete}"
            )
        return self._replace_step(step_id, status="in_progress")

    def complete(self, step_id: str, result: Any = None, *, note: str | None = None) -> PlanStep:
        step = self.get(step_id)
        notes = step.notes + ((note,) if note else ())
        return self._replace_step(
            step_id,
            status="completed",
            result=result,
            notes=notes,
        )

    def fail(self, step_id: str, note: str) -> PlanStep:
        step = self.get(step_id)
        return self._replace_step(
            step_id,
            status="failed",
            notes=step.notes + (note,),
        )

    def block(self, step_id: str, note: str) -> PlanStep:
        step = self.get(step_id)
        return self._replace_step(
            step_id,
            status="blocked",
            notes=step.notes + (note,),
        )

    def get(self, step_id: str) -> PlanStep:
        for step in self.plan.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def replan(
        self,
        steps: Sequence[PlanStep],
        *,
        reason: str,
        completion_criteria: Sequence[str] | None = None,
    ) -> Plan:
        completed = {
            step.id: step
            for step in self.plan.steps
            if step.status == "completed"
        }
        merged = tuple(
            completed.get(step.id, step)
            for step in steps
        )
        self.plan = Plan(
            id=self.plan.id,
            goal=self.plan.goal,
            steps=merged,
            completion_criteria=(
                tuple(completion_criteria)
                if completion_criteria is not None
                else self.plan.completion_criteria
            ),
            version=self.plan.version + 1,
        )
        if self.plan.steps:
            first = self.plan.steps[0]
            self._replace_step(
                first.id,
                notes=first.notes + (f"replanned: {reason}",),
            )
        return self.plan

    @property
    def progress(self) -> PlanProgress:
        statuses = [step.status for step in self.plan.steps]
        return PlanProgress(
            total=len(statuses),
            completed=statuses.count("completed"),
            failed=statuses.count("failed"),
            blocked=statuses.count("blocked"),
        )

@dataclass(frozen=True)
class VerificationIssue:
    criterion: str
    message: str
    step_id: str | None = None

@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    issues: tuple[VerificationIssue, ...] = ()
    reviewer: str | None = None

class PlanVerifier:
    def verify(self, plan: Plan) -> VerificationReport:
        issues: list[VerificationIssue] = []
        for step in plan.steps:
            if step.status != "completed":
                issues.append(
                    VerificationIssue(
                        criterion="step_completed",
                        message=f"step {step.id} is {step.status}",
                        step_id=step.id,
                    )
                )
            for criterion in step.acceptance_criteria:
                if step.status != "completed":
                    issues.append(
                        VerificationIssue(
                            criterion=criterion,
                            message="acceptance criterion cannot be satisfied before completion",
                            step_id=step.id,
                        )
                    )
        if plan.completion_criteria and not plan.steps:
            issues.append(
                VerificationIssue(
                    criterion="plan_nonempty",
                    message="completion criteria require at least one plan step",
                )
            )
        return VerificationReport(passed=not issues, issues=tuple(issues))

@runtime_checkable
class Planner(Protocol):
    async def create_plan(self, goal: str) -> Plan: ...

@runtime_checkable
class StepExecutor(Protocol):
    async def execute_step(self, step: PlanStep) -> Any: ...

@dataclass(frozen=True)
class PlanExecutionResult:
    plan: Plan
    verification: VerificationReport

class PlannerExecutor:
    def __init__(
        self,
        planner: Planner,
        executor: StepExecutor,
        *,
        verifier: PlanVerifier | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verifier = verifier or PlanVerifier()

    async def run(self, goal: str) -> PlanExecutionResult:
        tracker = PlanTracker(await self.planner.create_plan(goal))
        pending = {step.id for step in tracker.plan.steps}
        while pending:
            progressed = False
            for step in tuple(tracker.plan.steps):
                if step.id not in pending:
                    continue
                if all(
                    tracker.get(dependency).status == "completed"
                    for dependency in step.dependencies
                ):
                    tracker.start(step.id)
                    try:
                        result = await self.executor.execute_step(tracker.get(step.id))
                    except Exception as exc:
                        tracker.fail(step.id, str(exc))
                        pending.remove(step.id)
                        continue
                    tracker.complete(step.id, result)
                    pending.remove(step.id)
                    progressed = True
            if not progressed and pending:
                for step_id in tuple(pending):
                    tracker.block(step_id, "dependencies did not become satisfiable")
                    pending.remove(step_id)
        return PlanExecutionResult(
            plan=tracker.plan,
            verification=self.verifier.verify(tracker.plan),
        )

@dataclass(frozen=True)
class ReviewFinding:
    severity: Literal["info", "warning", "error"]
    message: str
    repairable: bool = True

@dataclass(frozen=True)
class ReviewResult:
    findings: tuple[ReviewFinding, ...] = ()

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

class ReflectionPass:
    def __init__(
        self,
        reviewer: Callable[[Any], Awaitable[ReviewResult]],
        repair: Callable[[Any, ReviewResult], Awaitable[Any]],
        *,
        max_repairs: int = 1,
    ) -> None:
        self.reviewer = reviewer
        self.repair = repair
        self.max_repairs = max_repairs

    async def run(self, work: Any) -> tuple[Any, ReviewResult]:
        current = work
        for attempt in range(self.max_repairs + 1):
            review = await self.reviewer(current)
            if review.passed or attempt >= self.max_repairs:
                return current, review
            if not any(finding.repairable for finding in review.findings):
                return current, review
            current = await self.repair(current, review)
        raise AssertionError("unreachable")

@dataclass(frozen=True)
class CriticAgent:
    name: str
    review: Callable[[Any], Awaitable[VerificationReport]]

class CriticPanel:
    def __init__(self, critics: Sequence[CriticAgent]) -> None:
        self.critics = tuple(critics)

    async def review(self, work: Any) -> tuple[VerificationReport, ...]:
        reports = await asyncio.gather(
            *(critic.review(work) for critic in self.critics)
        )
        return tuple(
            replace(report, reviewer=report.reviewer or critic.name)
            for critic, report in zip(self.critics, reports)
        )


@dataclass(frozen=True)
class AgentInvocation:
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    allowed_tools: frozenset[str] = frozenset()
    capability_grant: CapabilityGrant | None = None
    execution_budget: ExecutionBudget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class AgentInvoker(Protocol):
    async def invoke(
        self,
        agent: AgentConfig,
        invocation: AgentInvocation,
    ) -> AgentRunResult: ...

@dataclass(frozen=True)
class AgentTool:
    definition: ToolDefinition
    handler: ContextualToolHandler

def agent_as_tool(
    agent: AgentConfig,
    invoker: AgentInvoker,
    *,
    name: str | None = None,
    description: str | None = None,
) -> AgentTool:
    tool_name = name or f"agent.{agent.name}"

    async def handler(
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        prompt = arguments.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("agent tool requires a non-empty input")
        result = await invoker.invoke(
            agent,
            AgentInvocation(
                messages=(
                    ModelMessage(
                        role="user",
                        content=(ContentPart(type="text", text=prompt),),
                    ),
                ),
                metadata={
                    "parent_tool": tool_name,
                    "parent_request_context": dict(context.request_context),
                },
            ),
        )
        if result.structured_output is not None:
            return result.structured_output
        if result.final_response is not None:
            texts = [
                part.text
                for part in result.final_response.message.content
                if part.type == "text" and part.text is not None
            ]
            if texts:
                return "".join(texts)
        return {
            "termination_reason": result.termination_reason,
            "turns": result.turns,
        }

    return AgentTool(
        definition=ToolDefinition(
            name=tool_name,
            description=description or agent.description or f"Delegate bounded work to {agent.name}.",
            input_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
            side_effect="none",
        ),
        handler=handler,
    )

@dataclass(frozen=True)
class HandoffRequest:
    target: str
    messages: tuple[ModelMessage, ...]
    reason: str
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HandoffResult:
    owner: str
    result: AgentRunResult
    reason: str

class HandoffManager:
    def __init__(
        self,
        specialists: Mapping[str, tuple[AgentConfig, AgentInvoker]],
    ) -> None:
        self.specialists = dict(specialists)

    async def handoff(self, request: HandoffRequest) -> HandoffResult:
        if request.target not in self.specialists:
            raise KeyError(request.target)
        agent, invoker = self.specialists[request.target]
        result = await invoker.invoke(
            agent,
            AgentInvocation(
                messages=request.messages,
                context_items=request.context_items,
                workflow_state=request.workflow_state,
                metadata={
                    **dict(request.metadata),
                    "handoff_reason": request.reason,
                    "handoff_target": request.target,
                },
            ),
        )
        return HandoffResult(
            owner=request.target,
            result=result,
            reason=request.reason,
        )

@dataclass(frozen=True)
class SubagentSpec:
    agent: AgentConfig
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    workflow_state: WorkflowState | None = None
    allowed_tools: frozenset[str] = frozenset()
    capability_grant: CapabilityGrant | None = None
    execution_budget: ExecutionBudget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

class IsolatedSubagentRunner:
    def __init__(
        self,
        invoker: AgentInvoker,
        *,
        parent_budget: ExecutionBudget | None = None,
    ) -> None:
        self.invoker = invoker
        self.parent_budget = parent_budget

    async def run(self, spec: SubagentSpec) -> AgentRunResult:
        if self.parent_budget is not None:
            self.parent_budget.consume_subagent()
        return await self.invoker.invoke(
            spec.agent,
            AgentInvocation(
                messages=spec.messages,
                context_items=spec.context_items,
                workflow_state=spec.workflow_state,
                allowed_tools=spec.allowed_tools,
                capability_grant=spec.capability_grant,
                execution_budget=spec.execution_budget,
                metadata={**dict(spec.metadata), "isolated_subagent": True},
            ),
        )

@dataclass(frozen=True)
class WorkerAssignment:
    id: str
    worker: str
    messages: tuple[ModelMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class WorkerResult:
    assignment_id: str
    worker: str
    result: AgentRunResult

class SupervisorWorkerTeam:
    def __init__(
        self,
        workers: Mapping[str, tuple[AgentConfig, IsolatedSubagentRunner]],
    ) -> None:
        self.workers = dict(workers)

    async def delegate(
        self,
        assignments: Sequence[WorkerAssignment],
    ) -> tuple[WorkerResult, ...]:
        results: list[WorkerResult] = []
        for assignment in assignments:
            if assignment.worker not in self.workers:
                raise KeyError(assignment.worker)
            agent, runner = self.workers[assignment.worker]
            result = await runner.run(
                SubagentSpec(
                    agent=agent,
                    messages=assignment.messages,
                    context_items=assignment.context_items,
                    metadata={
                        **dict(assignment.metadata),
                        "assignment_id": assignment.id,
                        "supervised": True,
                    },
                )
            )
            results.append(
                WorkerResult(
                    assignment_id=assignment.id,
                    worker=assignment.worker,
                    result=result,
                )
            )
        return tuple(results)

@dataclass(frozen=True)
class AgentRoute:
    name: str
    agent: AgentConfig
    invoker: AgentInvoker
    intents: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()

class AgentRouter:
    def __init__(self, routes: Sequence[AgentRoute]) -> None:
        self.routes = tuple(routes)

    def select(
        self,
        *,
        intent: str | None = None,
        required_capabilities: Sequence[str] = (),
    ) -> AgentRoute:
        required = set(required_capabilities)
        candidates: list[tuple[int, AgentRoute]] = []
        for route in self.routes:
            if required - set(route.capabilities):
                continue
            score = len(required & set(route.capabilities))
            if intent is not None and intent in route.intents:
                score += 1000
            candidates.append((score, route))
        if not candidates:
            raise LookupError("no matching agent route")
        candidates.sort(key=lambda item: (-item[0], item[1].name))
        return candidates[0][1]

    async def route(
        self,
        invocation: AgentInvocation,
        *,
        intent: str | None = None,
        required_capabilities: Sequence[str] = (),
    ) -> tuple[str, AgentRunResult]:
        selected = self.select(
            intent=intent,
            required_capabilities=required_capabilities,
        )
        return selected.name, await selected.invoker.invoke(
            selected.agent,
            invocation,
        )


@dataclass(frozen=True)
class TeamMember:
    name: str
    agent: AgentConfig
    invoker: AgentInvoker

@dataclass(frozen=True)
class TeamTurn:
    speaker: str
    result: AgentRunResult

class RoundRobinTeam:
    def __init__(self, members: Sequence[TeamMember]) -> None:
        if not members:
            raise ValueError("round-robin team requires at least one member")
        self.members = tuple(members)
        self._index = 0

    async def step(self, invocation: AgentInvocation) -> TeamTurn:
        member = self.members[self._index]
        self._index = (self._index + 1) % len(self.members)
        result = await member.invoker.invoke(member.agent, invocation)
        return TeamTurn(member.name, result)

    async def run(
        self,
        invocations: Sequence[AgentInvocation],
    ) -> tuple[TeamTurn, ...]:
        return tuple([await self.step(invocation) for invocation in invocations])

SpeakerSelector = Callable[
    [Sequence[TeamMember], AgentInvocation, Sequence[TeamTurn]],
    Awaitable[str],
]

class ModelSelectedSpeakerTeam:
    def __init__(
        self,
        members: Sequence[TeamMember],
        selector: SpeakerSelector,
    ) -> None:
        if not members:
            raise ValueError("speaker team requires at least one member")
        self.members = tuple(members)
        self.selector = selector
        self.history: list[TeamTurn] = []

    async def step(self, invocation: AgentInvocation) -> TeamTurn:
        selected_name = await self.selector(
            self.members,
            invocation,
            tuple(self.history),
        )
        member = next(
            (candidate for candidate in self.members if candidate.name == selected_name),
            None,
        )
        if member is None:
            raise LookupError(f"selector chose unknown speaker: {selected_name}")
        turn = TeamTurn(
            member.name,
            await member.invoker.invoke(member.agent, invocation),
        )
        self.history.append(turn)
        return turn

@dataclass(frozen=True)
class SwarmDecision:
    next_agent: str | None = None
    reason: str = ""

SwarmHandoffPolicy = Callable[
    [str, "AgentRunResult"],
    Awaitable[SwarmDecision],
]

class SwarmTeam:
    def __init__(
        self,
        members: Mapping[str, tuple[AgentConfig, AgentInvoker]],
        handoff_policy: SwarmHandoffPolicy,
    ) -> None:
        self.members = dict(members)
        self.handoff_policy = handoff_policy

    async def run(
        self,
        start: str,
        invocation: AgentInvocation,
        *,
        max_handoffs: int = 8,
    ) -> tuple[TeamTurn, ...]:
        if start not in self.members:
            raise KeyError(start)
        turns: list[TeamTurn] = []
        current = start
        for _ in range(max_handoffs + 1):
            agent, invoker = self.members[current]
            result = await invoker.invoke(agent, invocation)
            turns.append(TeamTurn(current, result))
            decision = await self.handoff_policy(current, result)
            if decision.next_agent is None:
                return tuple(turns)
            if decision.next_agent not in self.members:
                raise KeyError(decision.next_agent)
            current = decision.next_agent
        raise RuntimeError("swarm handoff limit exceeded")

@runtime_checkable
class TeamNode(Protocol):
    async def execute(self, invocation: AgentInvocation) -> Any: ...

@dataclass(frozen=True)
class AgentTeamNode:
    agent: AgentConfig
    invoker: AgentInvoker

    async def execute(self, invocation: AgentInvocation) -> AgentRunResult:
        return await self.invoker.invoke(self.agent, invocation)

class HierarchicalTeam:
    def __init__(
        self,
        children: Mapping[str, TeamNode],
        selector: Callable[[AgentInvocation], Awaitable[str]],
    ) -> None:
        self.children = dict(children)
        self.selector = selector

    async def execute(self, invocation: AgentInvocation) -> Any:
        child_name = await self.selector(invocation)
        if child_name not in self.children:
            raise KeyError(child_name)
        return await self.children[child_name].execute(invocation)

class ParallelSubagentExecutor:
    def __init__(self, runner: IsolatedSubagentRunner) -> None:
        self.runner = runner

    async def run(
        self,
        specs: Sequence[SubagentSpec],
    ) -> tuple[AgentRunResult, ...]:
        return tuple(
            await asyncio.gather(
                *(self.runner.run(spec) for spec in specs)
            )
        )

@dataclass(frozen=True)
class MapReduceResult:
    mapped: tuple[AgentRunResult, ...]
    reduced: Any

class MapReduceOrchestrator:
    def __init__(
        self,
        runner: IsolatedSubagentRunner,
        reducer: Callable[[Sequence[AgentRunResult]], Awaitable[Any]],
    ) -> None:
        self.runner = runner
        self.reducer = reducer

    async def run(
        self,
        specs: Sequence[SubagentSpec],
    ) -> MapReduceResult:
        mapped = tuple(
            await asyncio.gather(
                *(self.runner.run(spec) for spec in specs)
            )
        )
        return MapReduceResult(
            mapped=mapped,
            reduced=await self.reducer(mapped),
        )


@dataclass(frozen=True)
class SpeculativeBranch:
    name: str
    spec: SubagentSpec

@dataclass(frozen=True)
class SpeculativeBranchResult:
    name: str
    result: AgentRunResult
    score: float

@dataclass(frozen=True)
class SpeculativeResult:
    branches: tuple[SpeculativeBranchResult, ...]
    selected: SpeculativeBranchResult | None = None
    merged: Any = None

class SpeculativeOrchestrator:
    def __init__(
        self,
        runner: IsolatedSubagentRunner,
        scorer: Callable[[str, AgentRunResult], Awaitable[float]],
        *,
        merger: Callable[[Sequence[SpeculativeBranchResult]], Awaitable[Any]] | None = None,
    ) -> None:
        self.runner = runner
        self.scorer = scorer
        self.merger = merger

    async def run(
        self,
        branches: Sequence[SpeculativeBranch],
        *,
        mode: Literal["select", "merge"] = "select",
    ) -> SpeculativeResult:
        if not branches:
            raise ValueError("speculative execution requires at least one branch")
        if mode == "merge" and self.merger is None:
            raise ValueError("merge mode requires a merger")
        if mode not in ("select", "merge"):
            raise ValueError("unsupported speculative mode")
        results = tuple(
            await asyncio.gather(
                *(self.runner.run(branch.spec) for branch in branches)
            )
        )
        scored_items: list[SpeculativeBranchResult] = []
        for branch, result in zip(branches, results):
            scored_items.append(
                SpeculativeBranchResult(
                    name=branch.name,
                    result=result,
                    score=float(await self.scorer(branch.name, result)),
                )
            )
        scored = tuple(scored_items)
        if mode == "merge":
            if self.merger is None:
                raise RuntimeError("merge mode lost its configured merger")
            return SpeculativeResult(
                branches=scored,
                merged=await self.merger(scored),
            )
        selected = sorted(
            scored,
            key=lambda item: (-item.score, item.name),
        )[0]
        return SpeculativeResult(
            branches=scored,
            selected=selected,
        )


MCPCapabilityKind = Literal["tool", "resource", "prompt"]

@runtime_checkable
class MCPTransport(Protocol):
    async def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

@dataclass(frozen=True)
class MCPServerInfo:
    name: str
    version: str | None = None
    capabilities: frozenset[str] = frozenset()

@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MCPResource:
    uri: str
    name: str | None = None
    description: str = ""
    mime_type: str | None = None

@dataclass(frozen=True)
class MCPPrompt:
    name: str
    description: str = ""
    arguments: tuple[str, ...] = ()

class MCPClient:
    def __init__(
        self,
        transport: MCPTransport,
        *,
        client_name: str = "agent-rt",
        client_version: str = "0",
        requested_capabilities: Sequence[str] = (),
    ) -> None:
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.requested_capabilities = tuple(requested_capabilities)
        self.server_info: MCPServerInfo | None = None

    async def initialize(self) -> MCPServerInfo:
        response = await self.transport.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
                "capabilities": list(self.requested_capabilities),
            },
        )
        server = response.get("serverInfo")
        server_map = server if isinstance(server, Mapping) else {}
        raw_capabilities = response.get("capabilities", ())
        if isinstance(raw_capabilities, Mapping):
            capabilities = frozenset(
                str(name)
                for name, enabled in raw_capabilities.items()
                if enabled is not False
            )
        elif isinstance(raw_capabilities, Sequence) and not isinstance(raw_capabilities, (str, bytes)):
            capabilities = frozenset(str(item) for item in raw_capabilities)
        else:
            capabilities = frozenset()
        name = str(server_map.get("name") or "mcp-server")
        version = server_map.get("version")
        self.server_info = MCPServerInfo(
            name=name,
            version=str(version) if version is not None else None,
            capabilities=capabilities,
        )
        return self.server_info

    def _ensure_initialized(self) -> MCPServerInfo:
        if self.server_info is None:
            raise RuntimeError("MCP client is not initialized")
        return self.server_info

    def _ensure_capability(self, capability: str) -> None:
        """Enforce negotiation: only use features the server advertised."""
        server = self._ensure_initialized()
        if capability not in server.capabilities:
            raise RuntimeError(
                f"MCP server {server.name!r} does not advertise the "
                f"{capability!r} capability"
            )

    async def list_tools(self) -> tuple[MCPTool, ...]:
        self._ensure_capability("tools")
        response = await self.transport.request("tools/list", {})
        values = response.get("tools", ())
        return tuple(
            MCPTool(
                name=str(value.get("name", "")),
                description=str(value.get("description", "")),
                input_schema=dict(value.get("inputSchema", {}))
                if isinstance(value.get("inputSchema", {}), Mapping)
                else {},
            )
            for value in values
            if isinstance(value, Mapping) and str(value.get("name", "")).strip()
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        self._ensure_capability("resources")
        response = await self.transport.request("resources/list", {})
        values = response.get("resources", ())
        return tuple(
            MCPResource(
                uri=str(value.get("uri", "")),
                name=str(value["name"]) if value.get("name") is not None else None,
                description=str(value.get("description", "")),
                mime_type=str(value["mimeType"]) if value.get("mimeType") is not None else None,
            )
            for value in values
            if isinstance(value, Mapping) and str(value.get("uri", "")).strip()
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        self._ensure_capability("prompts")
        response = await self.transport.request("prompts/list", {})
        values = response.get("prompts", ())
        prompts: list[MCPPrompt] = []
        for value in values:
            if not isinstance(value, Mapping) or not str(value.get("name", "")).strip():
                continue
            raw_arguments = value.get("arguments", ())
            arguments = tuple(
                str(argument.get("name"))
                for argument in raw_arguments
                if isinstance(argument, Mapping) and argument.get("name") is not None
            )
            prompts.append(
                MCPPrompt(
                    name=str(value["name"]),
                    description=str(value.get("description", "")),
                    arguments=arguments,
                )
            )
        return tuple(prompts)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._ensure_capability("tools")
        return await self.transport.request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )

    async def read_resource(self, uri: str) -> Mapping[str, Any]:
        self._ensure_capability("resources")
        return await self.transport.request("resources/read", {"uri": uri})

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._ensure_capability("prompts")
        return await self.transport.request(
            "prompts/get",
            {"name": name, "arguments": dict(arguments or {})},
        )

@dataclass(frozen=True)
class MCPCapabilityFilter:
    tools: tuple[str, ...] = ("*",)
    resources: tuple[str, ...] = ("*",)
    prompts: tuple[str, ...] = ("*",)

    def allows(self, kind: MCPCapabilityKind, name: str) -> bool:
        patterns = {
            "tool": self.tools,
            "resource": self.resources,
            "prompt": self.prompts,
        }[kind]
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)

@dataclass(frozen=True)
class MCPAccessPolicy:
    authentication: AuthenticationContext | None = None
    authorization_engine: AuthorizationEngine = field(default_factory=AuthorizationEngine)
    requirements: Mapping[str, AuthorizationRequirement] = field(default_factory=dict)
    policy_engine: PolicyEngine | None = None
    capability_filter: MCPCapabilityFilter = field(default_factory=MCPCapabilityFilter)
    capability_grant: CapabilityGrant | None = None

    def check(
        self,
        *,
        server: str,
        kind: MCPCapabilityKind,
        name: str,
        action: str,
    ) -> None:
        if not self.capability_filter.allows(kind, name):
            raise PermissionError(f"MCP {kind} is not approved: {name}")
        if (
            kind == "tool"
            and self.capability_grant is not None
            and self.capability_grant.tools
            and not self.capability_grant.allows_tool(name)
        ):
            raise PermissionError(f"MCP tool is outside the capability grant: {name}")
        if self.authentication is None and self.requirements:
            # Configured requirements cannot be evaluated without an identity;
            # silently skipping them would turn a misconfiguration into access.
            raise PermissionError("MCP policy requires an authenticated principal")
        if self.authentication is not None:
            requirement = self.requirements.get(
                f"{kind}:{action}",
                self.requirements.get(kind, AuthorizationRequirement()),
            )
            self.authorization_engine.check(self.authentication, requirement)
            credential = self.authentication.credential
            if (
                credential is not None
                and self.capability_grant is not None
                and not self.capability_grant.allows_credential(credential.id)
            ):
                raise PermissionError("MCP credential is outside the capability grant")
        if self.policy_engine is not None:
            subject = (
                self.authentication.principal.id
                if self.authentication is not None
                else None
            )
            self.policy_engine.check(
                PolicyRequest(
                    domain="mcp",
                    action=f"{kind}:{action}",
                    subject=subject,
                    resource=f"{server}:{name}",
                    attributes={"server": server, "kind": kind},
                )
            )

class AuthorizedMCPClient:
    def __init__(self, client: MCPClient, policy: MCPAccessPolicy) -> None:
        self.client = client
        self.policy = policy

    def _server_name(self) -> str:
        return self.client._ensure_initialized().name

    async def list_tools(self) -> tuple[MCPTool, ...]:
        server = self._server_name()
        tools = await self.client.list_tools()
        return tuple(
            tool
            for tool in tools
            if self._allowed(server, "tool", tool.name, "discover")
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        server = self._server_name()
        resources = await self.client.list_resources()
        return tuple(
            resource
            for resource in resources
            if self._allowed(server, "resource", resource.uri, "discover")
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        server = self._server_name()
        prompts = await self.client.list_prompts()
        return tuple(
            prompt
            for prompt in prompts
            if self._allowed(server, "prompt", prompt.name, "discover")
        )

    def _allowed(
        self,
        server: str,
        kind: MCPCapabilityKind,
        name: str,
        action: str,
    ) -> bool:
        try:
            self.policy.check(
                server=server,
                kind=kind,
                name=name,
                action=action,
            )
        except PermissionError:
            return False
        return True

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="tool",
            name=name,
            action="invoke",
        )
        return await self.client.call_tool(name, arguments)

    async def read_resource(self, uri: str) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="resource",
            name=uri,
            action="read",
        )
        return await self.client.read_resource(uri)

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="prompt",
            name=name,
            action="get",
        )
        return await self.client.get_prompt(name, arguments)


ProtocolKind = Literal["rest", "openapi", "graphql", "websocket", "grpc", "custom"]

@runtime_checkable
class ProtocolAdapter(Protocol):
    protocol: ProtocolKind
    async def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any: ...

class ProtocolAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProtocolAdapter] = {}

    def register(self, name: str, adapter: ProtocolAdapter) -> None:
        if not name.strip():
            raise ValueError("adapter name must not be empty")
        self._adapters[name] = adapter

    def get(self, name: str) -> ProtocolAdapter:
        if name not in self._adapters:
            raise KeyError(name)
        return self._adapters[name]

    async def request(
        self,
        name: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self.get(name).request(operation, payload)

@dataclass(frozen=True)
class ConnectorDefinition:
    name: str
    adapter: str
    operations: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

class ConnectorRegistry:
    def __init__(self, adapters: ProtocolAdapterRegistry) -> None:
        self.adapters = adapters
        self._connectors: dict[str, ConnectorDefinition] = {}

    def register(self, definition: ConnectorDefinition) -> None:
        if not definition.name.strip():
            raise ValueError("connector name must not be empty")
        self.adapters.get(definition.adapter)
        self._connectors[definition.name] = definition

    async def invoke(
        self,
        connector: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        definition = self._connectors.get(connector)
        if definition is None:
            raise KeyError(connector)
        remote_operation = definition.operations.get(operation)
        if remote_operation is None:
            raise KeyError(f"{connector}:{operation}")
        return await self.adapters.request(
            definition.adapter,
            remote_operation,
            payload,
        )

@dataclass(frozen=True)
class RemoteAgentCard:
    id: str
    name: str
    endpoint: str
    description: str = ""
    modalities: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class RemoteAgentDirectory(Protocol):
    async def discover(self) -> Sequence[RemoteAgentCard]: ...

class RemoteAgentRegistry:
    def __init__(self, directory: RemoteAgentDirectory) -> None:
        self.directory = directory
        self._agents: dict[str, RemoteAgentCard] = {}

    async def refresh(self) -> tuple[RemoteAgentCard, ...]:
        discovered = tuple(await self.directory.discover())
        self._agents = {agent.id: agent for agent in discovered}
        return discovered

    def get(self, agent_id: str) -> RemoteAgentCard:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        return self._agents[agent_id]

@dataclass(frozen=True)
class RemoteMessage:
    id: str
    role: str
    parts: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

RemoteTaskStatus = Literal[
    "submitted",
    "working",
    "input_required",
    "completed",
    "failed",
    "canceled",
]

@dataclass(frozen=True)
class RemoteTask:
    id: str
    agent_id: str
    status: RemoteTaskStatus = "submitted"
    message: RemoteMessage | None = None
    error: str | None = None
    version: int = 0

class RemoteTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, RemoteTask] = {}

    def upsert(self, task: RemoteTask) -> RemoteTask:
        existing = self._tasks.get(task.id)
        if existing is not None and task.version < existing.version:
            raise ValueError("remote task version cannot move backwards")
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> RemoteTask:
        if task_id not in self._tasks:
            raise KeyError(task_id)
        return self._tasks[task_id]

@dataclass(frozen=True)
class RemoteArtifact:
    id: str
    task_id: str
    name: str
    media_type: str
    data: Any
    version: int = 1
    complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

class RemoteArtifactStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, list[RemoteArtifact]] = {}

    def put(self, artifact: RemoteArtifact) -> RemoteArtifact:
        versions = self._artifacts.setdefault(artifact.id, [])
        if versions and artifact.version <= versions[-1].version:
            raise ValueError("remote artifact version must increase")
        versions.append(artifact)
        return artifact

    def latest(self, artifact_id: str) -> RemoteArtifact:
        versions = self._artifacts.get(artifact_id)
        if not versions:
            raise KeyError(artifact_id)
        return versions[-1]

@dataclass(frozen=True)
class RemoteEvent:
    type: Literal["status", "message", "artifact", "callback"]
    task_id: str
    payload: Mapping[str, Any]

@runtime_checkable
class RemoteAgentTransport(Protocol):
    async def send_message(
        self,
        agent: RemoteAgentCard,
        message: RemoteMessage,
    ) -> Mapping[str, Any]: ...

    async def get_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> Mapping[str, Any]: ...

    async def cancel_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> Mapping[str, Any]: ...

    def stream_task(
        self,
        agent: RemoteAgentCard,
        task_id: str,
    ) -> AsyncIterator[Mapping[str, Any]]: ...

CapabilitySet = frozenset[str]

@dataclass(frozen=True)
class CapabilityNegotiation:
    local: CapabilitySet
    remote: CapabilitySet
    required: CapabilitySet = frozenset()

    @property
    def negotiated(self) -> CapabilitySet:
        available = self.local & self.remote
        missing = self.required - available
        if missing:
            raise ValueError(
                f"required capabilities unavailable: {sorted(missing)}"
            )
        return available

class RemoteAgentClient:
    def __init__(
        self,
        registry: RemoteAgentRegistry,
        transport: RemoteAgentTransport,
        *,
        task_store: RemoteTaskStore | None = None,
        artifact_store: RemoteArtifactStore | None = None,
        local_capabilities: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.task_store = task_store or RemoteTaskStore()
        self.artifact_store = artifact_store or RemoteArtifactStore()
        self.local_capabilities = frozenset(local_capabilities)

    def negotiate(
        self,
        agent_id: str,
        *,
        required: Sequence[str] = (),
    ) -> CapabilitySet:
        card = self.registry.get(agent_id)
        return CapabilityNegotiation(
            local=self.local_capabilities,
            remote=card.capabilities,
            required=frozenset(required),
        ).negotiated

    @staticmethod
    def _parse_task(agent_id: str, payload: Mapping[str, Any]) -> RemoteTask:
        status = str(payload.get("status", "submitted"))
        if status not in {
            "submitted", "working", "input_required",
            "completed", "failed", "canceled",
        }:
            raise ValueError(f"unsupported remote task status: {status}")
        task_id = str(payload.get("id", "")).strip()
        if not task_id:
            raise ValueError("remote task id is required")
        return RemoteTask(
            id=task_id,
            agent_id=agent_id,
            status=status,  # type: ignore[arg-type]
            error=str(payload["error"]) if payload.get("error") is not None else None,
            version=int(payload.get("version", 0)),
        )

    def _persist_artifact(
        self,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> RemoteArtifact:
        artifact = RemoteArtifact(
            id=str(payload["id"]),
            task_id=task_id,
            name=str(payload.get("name", payload["id"])),
            media_type=str(payload.get("mediaType", "application/octet-stream")),
            data=payload.get("data"),
            version=int(payload.get("version", 1)),
            complete=bool(payload.get("complete", True)),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), Mapping)
            else {},
        )
        return self.artifact_store.put(artifact)

    async def send(
        self,
        agent_id: str,
        message: RemoteMessage,
    ) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.send_message(card, message)
        task = self.task_store.upsert(self._parse_task(agent_id, payload))
        artifacts = payload.get("artifacts", ())
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
            for artifact in artifacts:
                if isinstance(artifact, Mapping):
                    self._persist_artifact(task.id, artifact)
        return task

    async def refresh_task(self, agent_id: str, task_id: str) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.get_task(card, task_id)
        return self.task_store.upsert(self._parse_task(agent_id, payload))

    async def cancel(self, agent_id: str, task_id: str) -> RemoteTask:
        card = self.registry.get(agent_id)
        payload = await self.transport.cancel_task(card, task_id)
        return self.task_store.upsert(self._parse_task(agent_id, payload))

    async def stream(
        self,
        agent_id: str,
        task_id: str,
        *,
        callback: Callable[[RemoteEvent], Awaitable[None]] | None = None,
    ) -> tuple[RemoteEvent, ...]:
        card = self.registry.get(agent_id)
        events: list[RemoteEvent] = []
        async for payload in self.transport.stream_task(card, task_id):
            event_type = str(payload.get("type", "status"))
            if event_type not in {"status", "message", "artifact", "callback"}:
                continue
            event = RemoteEvent(
                type=event_type,  # type: ignore[arg-type]
                task_id=task_id,
                payload=dict(payload),
            )
            events.append(event)
            if event_type == "status":
                self.task_store.upsert(self._parse_task(agent_id, payload))
            elif event_type == "artifact":
                artifact = payload.get("artifact")
                if isinstance(artifact, Mapping):
                    self._persist_artifact(task_id, artifact)
            if callback is not None:
                await callback(event)
        return tuple(events)


BrowserAction = Literal["navigate", "search", "click", "fill", "download", "extract", "back", "forward", "reload"]

@dataclass(frozen=True)
class BrowserState:
    url: str | None = None
    title: str | None = None
    history: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class BrowserBackend(Protocol):
    async def perform(
        self,
        action: BrowserAction,
        arguments: Mapping[str, Any],
        state: BrowserState,
    ) -> tuple[Any, BrowserState]: ...

class BrowserSession:
    def __init__(
        self,
        backend: BrowserBackend,
        state: BrowserState = BrowserState(),
    ) -> None:
        self.backend = backend
        self.state = state

    async def perform(self, action: BrowserAction, **arguments: Any) -> Any:
        result, self.state = await self.backend.perform(action, arguments, self.state)
        return result

ComputerAction = Literal["screenshot", "move", "click", "scroll", "key", "type", "drag"]

@dataclass(frozen=True)
class ComputerState:
    width: int
    height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class ComputerBackend(Protocol):
    async def perform(
        self,
        action: ComputerAction,
        arguments: Mapping[str, Any],
        state: ComputerState,
    ) -> Any: ...

class ComputerSession:
    def __init__(self, backend: ComputerBackend, state: ComputerState) -> None:
        if state.width <= 0 or state.height <= 0:
            raise ValueError("computer dimensions must be positive")
        self.backend = backend
        self.state = state

    async def perform(self, action: ComputerAction, **arguments: Any) -> Any:
        return await self.backend.perform(action, arguments, self.state)

RetrievalKind = Literal["web", "enterprise", "file", "knowledge", "database"]

@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    limit: int = 10
    filters: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievalResult:
    id: str
    title: str
    content: Any
    score: float | None = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class RetrievalProvider(Protocol):
    kind: RetrievalKind
    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]: ...

class RetrievalRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, RetrievalProvider] = {}

    def register(self, name: str, provider: RetrievalProvider) -> None:
        if not name.strip():
            raise ValueError("retrieval provider name must not be empty")
        self._providers[name] = provider

    async def search(self, name: str, query: RetrievalQuery) -> tuple[RetrievalResult, ...]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(name)
        return tuple((await provider.search(query))[:query.limit])

AGENT_RT_VECTOR_DB_ENV = "AGENT_RT_VECTOR_DB"
AGENT_RT_VECTOR_DB_COLLECTION_ENV = "AGENT_RT_VECTOR_DB_COLLECTION"
AGENT_RT_VECTOR_DB_URL_ENV = "AGENT_RT_VECTOR_DB_URL"
AGENT_RT_VECTOR_DB_OPTION_PREFIX = "AGENT_RT_VECTOR_DB_OPTION_"


@dataclass(frozen=True)
class VectorDBConfig:
    backend: str
    collection: str | None = None
    url: str | None = None
    options: Mapping[str, str] = field(default_factory=dict)


class VectorDBProviderRegistry:
    """Resolve vector-backed retrieval providers without coupling Agent RT to vendor SDKs."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, name: str, factory: Any, *, replace: bool = False) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("vector DB backend name must not be empty")
        if not callable(factory):
            raise TypeError("vector DB provider factory must be callable")
        if normalized in self._factories and not replace:
            raise ValueError(f"vector DB backend already registered: {normalized}")
        self._factories[normalized] = factory

    def create(self, config: VectorDBConfig) -> RetrievalProvider:
        backend = config.backend.strip().lower()
        factory = self._factories.get(backend)
        if factory is None:
            supported = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"unsupported vector DB backend {config.backend!r}; registered backends: {supported}"
            )
        provider = factory(config)
        if not isinstance(provider, RetrievalProvider):
            raise TypeError(
                f"vector DB backend {backend!r} factory must return a RetrievalProvider"
            )
        return provider


def vector_db_config_from_environment(
    environment: Mapping[str, str] | None = None,
) -> VectorDBConfig:
    env = os.environ if environment is None else environment
    backend = str(env.get(AGENT_RT_VECTOR_DB_ENV, "")).strip().lower()
    if not backend:
        raise RuntimeError(f"vector DB selection requires {AGENT_RT_VECTOR_DB_ENV}")
    collection = str(env.get(AGENT_RT_VECTOR_DB_COLLECTION_ENV, "")).strip() or None
    url = str(env.get(AGENT_RT_VECTOR_DB_URL_ENV, "")).strip() or None
    options = {
        key[len(AGENT_RT_VECTOR_DB_OPTION_PREFIX) :].lower(): str(value)
        for key, value in env.items()
        if key.startswith(AGENT_RT_VECTOR_DB_OPTION_PREFIX)
        and key != AGENT_RT_VECTOR_DB_OPTION_PREFIX
        and str(value).strip()
    }
    return VectorDBConfig(
        backend=backend,
        collection=collection,
        url=url,
        options=options,
    )


def vector_db_provider_from_environment(
    registry: VectorDBProviderRegistry,
    environment: Mapping[str, str] | None = None,
) -> RetrievalProvider:
    """Select a registered vector DB adapter from environment configuration."""
    return registry.create(vector_db_config_from_environment(environment))


AGENT_RT_VECTOR_DB_API_KEY_ENV = "AGENT_RT_VECTOR_DB_API_KEY"  # pragma: allowlist secret
AGENT_RT_VECTOR_DB_TIMEOUT_ENV = "AGENT_RT_VECTOR_DB_TIMEOUT_SECONDS"
POPULAR_VECTOR_DB_BACKENDS = ("chroma", "milvus", "pinecone", "qdrant", "weaviate")

_VECTOR_DB_CREDENTIAL_ENV = {
    "chroma": "CHROMA_API_KEY",
    "milvus": "MILVUS_TOKEN",
    "pinecone": "PINECONE_API_KEY",
    "qdrant": "QDRANT_API_KEY",
    "weaviate": "WEAVIATE_API_KEY",
}


def _vector_db_http_url(value: str, *, backend: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{backend} vector DB URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{backend} vector DB URL must not contain embedded credentials")
    return candidate


def _vector_db_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EnvironmentVectorDBProvider:
    kind: RetrievalKind = "knowledge"

    def __init__(
        self,
        config: VectorDBConfig,
        *,
        embedding_provider: EmbeddingModelProvider | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        client: Any = None,
    ) -> None:
        backend = config.backend.strip().lower()
        if backend not in POPULAR_VECTOR_DB_BACKENDS:
            supported = ", ".join(POPULAR_VECTOR_DB_BACKENDS)
            raise ValueError(f"unsupported built-in vector DB backend {backend!r}; expected one of: {supported}")
        if not config.url:
            raise RuntimeError(f"{backend} vector DB requires {AGENT_RT_VECTOR_DB_URL_ENV}")
        if not config.collection:
            raise RuntimeError(f"{backend} vector DB requires {AGENT_RT_VECTOR_DB_COLLECTION_ENV}")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("vector DB timeout must be positive")
        self.config = config
        self.backend = backend
        self.url = _vector_db_http_url(config.url, backend=backend)
        self.collection = config.collection
        self.embedding_provider = embedding_provider
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        embedding_provider: EmbeddingModelProvider | None = None,
        client: Any = None,
    ) -> EnvironmentVectorDBProvider:
        env = os.environ if environment is None else environment
        config = vector_db_config_from_environment(env)
        if config.backend not in POPULAR_VECTOR_DB_BACKENDS:
            supported = ", ".join(POPULAR_VECTOR_DB_BACKENDS)
            raise ValueError(
                f"unsupported built-in vector DB backend {config.backend!r}; expected one of: {supported}"
            )
        credential_env = _VECTOR_DB_CREDENTIAL_ENV[config.backend]
        api_key = env.get(credential_env) or env.get(AGENT_RT_VECTOR_DB_API_KEY_ENV)
        timeout_raw = env.get(AGENT_RT_VECTOR_DB_TIMEOUT_ENV, "20")
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{AGENT_RT_VECTOR_DB_TIMEOUT_ENV} must be numeric") from exc
        return cls(
            config,
            embedding_provider=embedding_provider,
            api_key=str(api_key) if api_key else None,
            timeout_seconds=timeout_seconds,
            client=client,
        )

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        if not query.text.strip():
            raise ValueError("vector DB query must not be empty")
        vector = await self._query_vector(query)
        if self.backend == "qdrant":
            return await self._search_qdrant(query, vector)
        if self.backend == "pinecone":
            return await self._search_pinecone(query, vector)
        if self.backend == "milvus":
            return await self._search_milvus(query, vector)
        if self.backend == "weaviate":
            return await self._search_weaviate(query, vector)
        return await self._search_chroma(query, vector)

    async def _query_vector(self, query: RetrievalQuery) -> list[float]:
        supplied = query.filters.get("vector")
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            try:
                vector = [float(value) for value in supplied]
            except (TypeError, ValueError) as exc:
                raise ValueError("filters.vector must contain only numbers") from exc
            if vector and all(math.isfinite(value) for value in vector):
                return vector
            raise ValueError("filters.vector must be a non-empty finite numeric vector")
        if self.embedding_provider is None:
            raise RuntimeError(
                "vector DB text search requires an EmbeddingModelProvider or a precomputed filters.vector"
            )
        response = await self.embedding_provider.embed(EmbeddingRequest(input=query.text))
        if not response.data:
            raise RuntimeError("embedding provider returned no vectors")
        embedding = response.data[0].embedding
        if isinstance(embedding, str):
            raise TypeError("vector DB search requires numeric embeddings")
        vector = [float(value) for value in embedding]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding provider returned an invalid vector")
        return vector

    def _filters_without_vector(self, query: RetrievalQuery) -> dict[str, Any]:
        return {key: value for key, value in query.filters.items() if key != "vector"}

    async def _request(self, method: str, url: str, **kwargs: Any) -> Mapping[str, Any]:
        if self._client is not None:
            response = await self._client.request(method, url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.backend} vector DB returned a non-object JSON response")
        return payload

    def _result(
        self,
        item_id: Any,
        content: Any,
        *,
        score: Any = None,
        title: Any = None,
        uri: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        resolved_metadata = dict(metadata or {})
        return RetrievalResult(
            id=str(item_id),
            title=str(title or resolved_metadata.get("title") or item_id),
            content=content if content is not None else resolved_metadata.get("text") or resolved_metadata.get("content") or "",
            score=_vector_db_number(score),
            uri=str(uri) if uri else (
                str(resolved_metadata.get("url")) if resolved_metadata.get("url") else None
            ),
            metadata={"provider": self.backend, **resolved_metadata},
        )

    async def _search_qdrant(self, query: RetrievalQuery, vector: list[float]) -> Sequence[RetrievalResult]:
        headers = {"api-key": self.api_key} if self.api_key else {}
        filters = self._filters_without_vector(query)
        body: dict[str, Any] = {
            "vector": vector,
            "limit": query.limit,
            "with_payload": True,
        }
        if filters:
            body["filter"] = filters
        payload = await self._request(
            "POST",
            f"{self.url}/collections/{urllib.parse.quote(self.collection, safe='')}/points/search",
            json=body,
            headers=headers,
        )
        items = payload.get("result", ())
        return tuple(
            self._result(
                item.get("id", index),
                item.get("payload", {}).get("text") if isinstance(item.get("payload"), Mapping) else None,
                score=item.get("score"),
                metadata=item.get("payload") if isinstance(item.get("payload"), Mapping) else {},
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    async def _search_pinecone(self, query: RetrievalQuery, vector: list[float]) -> Sequence[RetrievalResult]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Api-Key"] = self.api_key
        filters = self._filters_without_vector(query)
        body: dict[str, Any] = {"vector": vector, "topK": query.limit, "includeMetadata": True}
        if filters:
            body["filter"] = filters
        namespace = self.config.options.get("namespace")
        if namespace:
            body["namespace"] = namespace
        payload = await self._request("POST", f"{self.url}/query", json=body, headers=headers)
        items = payload.get("matches", ())
        return tuple(
            self._result(
                item.get("id", index),
                item.get("metadata", {}).get("text") if isinstance(item.get("metadata"), Mapping) else None,
                score=item.get("score"),
                metadata=item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {},
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    def _reject_unsupported_filters(self, query: RetrievalQuery) -> None:
        """Fail closed: never silently drop filters that may enforce isolation."""
        if self._filters_without_vector(query):
            raise ValueError(
                f"the {self.backend} adapter does not support per-query filters; "
                "refusing to run an unfiltered search (use a collection per scope, "
                "or a backend that supports filters)"
            )

    async def _search_milvus(self, query: RetrievalQuery, vector: list[float]) -> Sequence[RetrievalResult]:
        self._reject_unsupported_filters(query)
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "collectionName": self.collection,
            "data": [vector],
            "limit": query.limit,
            "outputFields": ["*"],
        }
        expression = self.config.options.get("filter")
        if expression:
            body["filter"] = expression
        payload = await self._request("POST", f"{self.url}/v2/vectordb/entities/search", json=body, headers=headers)
        data = payload.get("data", ())
        items = data[0] if isinstance(data, Sequence) and data and isinstance(data[0], Sequence) else data
        return tuple(
            self._result(
                item.get("id", item.get("pk", index)),
                item.get("text") or item.get("content"),
                score=item.get("score"),
                metadata=item,
            )
            for index, item in enumerate(items if isinstance(items, Sequence) else ())
            if isinstance(item, Mapping)
        )

    async def _search_weaviate(self, query: RetrievalQuery, vector: list[float]) -> Sequence[RetrievalResult]:
        self._reject_unsupported_filters(query)
        identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        class_name = self.collection
        content_field = self.config.options.get("content_field", "text")
        title_field = self.config.options.get("title_field", "title")
        uri_field = self.config.options.get("uri_field", "url")
        for value in (class_name, content_field, title_field, uri_field):
            if not identifier.fullmatch(value):
                raise ValueError("Weaviate collection and field names must be GraphQL identifiers")
        vector_literal = ", ".join(format(value, ".17g") for value in vector)
        graph_query = (
            f"{{ Get {{ {class_name}(nearVector: {{vector: [{vector_literal}]}}, limit: {query.limit}) "
            f"{{ {content_field} {title_field} {uri_field} _additional {{ id distance certainty }} }} }} }}"
        )
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = await self._request("POST", f"{self.url}/v1/graphql", json={"query": graph_query}, headers=headers)
        data = payload.get("data", {})
        get_value = data.get("Get", {}) if isinstance(data, Mapping) else {}
        items = get_value.get(class_name, ()) if isinstance(get_value, Mapping) else ()
        results: list[RetrievalResult] = []
        for index, item in enumerate(items if isinstance(items, Sequence) else ()):
            if not isinstance(item, Mapping):
                continue
            additional = item.get("_additional", {})
            extra = additional if isinstance(additional, Mapping) else {}
            score = extra.get("certainty")
            if score is None and extra.get("distance") is not None:
                distance = _vector_db_number(extra.get("distance"))
                score = (1.0 - distance) if distance is not None else None
            metadata = {key: value for key, value in item.items() if key != "_additional"}
            results.append(
                self._result(
                    extra.get("id", index),
                    item.get(content_field),
                    score=score,
                    title=item.get(title_field),
                    uri=item.get(uri_field),
                    metadata=metadata,
                )
            )
        return tuple(results)

    async def _search_chroma(self, query: RetrievalQuery, vector: list[float]) -> Sequence[RetrievalResult]:
        tenant = urllib.parse.quote(self.config.options.get("tenant", "default_tenant"), safe="")
        database = urllib.parse.quote(self.config.options.get("database", "default_database"), safe="")
        collection = urllib.parse.quote(self.collection, safe="")
        path = f"/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/query"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": query.limit,
            "include": ["documents", "metadatas", "distances", "uris"],
        }
        filters = self._filters_without_vector(query)
        if filters:
            body["where"] = filters
        payload = await self._request("POST", self.url + path, json=body, headers=headers)
        ids = payload.get("ids", ())
        ids = ids[0] if isinstance(ids, Sequence) and ids and isinstance(ids[0], Sequence) else ids
        documents = payload.get("documents", ())
        documents = documents[0] if isinstance(documents, Sequence) and documents and isinstance(documents[0], Sequence) else documents
        metadatas = payload.get("metadatas", ())
        metadatas = metadatas[0] if isinstance(metadatas, Sequence) and metadatas and isinstance(metadatas[0], Sequence) else metadatas
        distances = payload.get("distances", ())
        distances = distances[0] if isinstance(distances, Sequence) and distances and isinstance(distances[0], Sequence) else distances
        uris = payload.get("uris", ())
        uris = uris[0] if isinstance(uris, Sequence) and uris and isinstance(uris[0], Sequence) else uris
        results: list[RetrievalResult] = []
        for index, item_id in enumerate(ids if isinstance(ids, Sequence) else ()):
            metadata = metadatas[index] if isinstance(metadatas, Sequence) and index < len(metadatas) and isinstance(metadatas[index], Mapping) else {}
            document = documents[index] if isinstance(documents, Sequence) and index < len(documents) else None
            distance = distances[index] if isinstance(distances, Sequence) and index < len(distances) else None
            uri = uris[index] if isinstance(uris, Sequence) and index < len(uris) else None
            numeric_distance = _vector_db_number(distance)
            score = 1.0 / (1.0 + max(0.0, numeric_distance)) if numeric_distance is not None else None
            results.append(self._result(item_id, document, score=score, uri=uri, metadata=metadata))
        return tuple(results)


def register_popular_vector_db_backends(
    registry: VectorDBProviderRegistry,
    *,
    embedding_provider: EmbeddingModelProvider | None = None,
    environment: Mapping[str, str] | None = None,
    client: Any = None,
    replace: bool = False,
) -> VectorDBProviderRegistry:
    env = os.environ if environment is None else environment

    def factory(config: VectorDBConfig) -> RetrievalProvider:
        backend_env = {**dict(env), AGENT_RT_VECTOR_DB_ENV: config.backend}
        if config.collection is not None:
            backend_env[AGENT_RT_VECTOR_DB_COLLECTION_ENV] = config.collection
        if config.url is not None:
            backend_env[AGENT_RT_VECTOR_DB_URL_ENV] = config.url
        for key, value in config.options.items():
            backend_env[f"{AGENT_RT_VECTOR_DB_OPTION_PREFIX}{key.upper()}"] = value
        return EnvironmentVectorDBProvider.from_environment(
            backend_env,
            embedding_provider=embedding_provider,
            client=client,
        )

    for backend in POPULAR_VECTOR_DB_BACKENDS:
        registry.register(backend, factory, replace=replace)
    return registry

AGENT_RT_WEB_SEARCH_TOOL_ENV = "AGENT_RT_WEB_SEARCH_TOOL"
AGENT_RT_WEB_SEARCH_API_KEY_ENV = "AGENT_RT_WEB_SEARCH_API_KEY"  # pragma: allowlist secret
AGENT_RT_WEB_SEARCH_TOKEN_ENV = "AGENT_RT_WEB_SEARCH_TOKEN"  # pragma: allowlist secret
AGENT_RT_WEB_SEARCH_TIMEOUT_ENV = "AGENT_RT_WEB_SEARCH_TIMEOUT_SECONDS"

_WEB_SEARCH_CREDENTIAL_ENV = {
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_SEARCH_API_KEY",
    "serper": "SERPER_API_KEY",
}

class EnvironmentWebSearchProvider:
    kind: RetrievalKind = "web"

    def __init__(self, tool: str, credential: str, *, timeout_seconds: float = 20.0, client: Any = None) -> None:
        normalized = tool.strip().lower()
        if normalized not in _WEB_SEARCH_CREDENTIAL_ENV:
            supported = ", ".join(sorted(_WEB_SEARCH_CREDENTIAL_ENV))
            raise ValueError(f"unsupported web search tool {tool!r}; expected one of: {supported}")
        if not credential.strip():
            raise ValueError("web search credential must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("web search timeout must be positive")
        self.tool = normalized
        self._credential = credential
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None, *, client: Any = None) -> EnvironmentWebSearchProvider:
        env = os.environ if environment is None else environment
        tool = str(env.get(AGENT_RT_WEB_SEARCH_TOOL_ENV, "")).strip().lower()
        if not tool:
            raise RuntimeError(f"web search requires {AGENT_RT_WEB_SEARCH_TOOL_ENV}=tavily|brave|serper")
        credential_env = _WEB_SEARCH_CREDENTIAL_ENV.get(tool)
        if credential_env is None:
            supported = ", ".join(sorted(_WEB_SEARCH_CREDENTIAL_ENV))
            raise ValueError(f"unsupported web search tool {tool!r}; expected one of: {supported}")
        credential = env.get(credential_env) or env.get(AGENT_RT_WEB_SEARCH_API_KEY_ENV) or env.get(AGENT_RT_WEB_SEARCH_TOKEN_ENV)
        if not credential:
            raise RuntimeError(
                "web search credential is missing; set "
                f"{credential_env}, {AGENT_RT_WEB_SEARCH_API_KEY_ENV}, or {AGENT_RT_WEB_SEARCH_TOKEN_ENV}"
            )
        timeout_raw = env.get(AGENT_RT_WEB_SEARCH_TIMEOUT_ENV, "20")
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{AGENT_RT_WEB_SEARCH_TIMEOUT_ENV} must be numeric") from exc
        return cls(tool, str(credential), timeout_seconds=timeout_seconds, client=client)

    async def search(self, query: RetrievalQuery) -> Sequence[RetrievalResult]:
        if query.limit < 1:
            raise ValueError("retrieval limit must be at least 1")
        if not query.text.strip():
            raise ValueError("web search query must not be empty")
        if self.tool == "tavily":
            payload = await self._tavily(query)
            items = payload.get("results", ())
            return tuple(
                RetrievalResult(
                    id=str(item.get("url") or index),
                    title=str(item.get("title") or item.get("url") or ""),
                    content=str(item.get("content") or item.get("snippet") or ""),
                    score=float(item["score"]) if item.get("score") is not None else None,
                    uri=str(item.get("url")) if item.get("url") else None,
                    metadata={"provider": "tavily"},
                )
                for index, item in enumerate(items[:query.limit])
                if isinstance(item, Mapping)
            )
        if self.tool == "brave":
            payload = await self._brave(query)
            web = payload.get("web", {})
            items = web.get("results", ()) if isinstance(web, Mapping) else ()
            return tuple(
                RetrievalResult(
                    id=str(item.get("url") or index),
                    title=str(item.get("title") or item.get("url") or ""),
                    content=str(item.get("description") or ""),
                    uri=str(item.get("url")) if item.get("url") else None,
                    metadata={"provider": "brave"},
                )
                for index, item in enumerate(items[:query.limit])
                if isinstance(item, Mapping)
            )
        payload = await self._serper(query)
        items = payload.get("organic", ())
        return tuple(
            RetrievalResult(
                id=str(item.get("link") or index),
                title=str(item.get("title") or item.get("link") or ""),
                content=str(item.get("snippet") or ""),
                uri=str(item.get("link")) if item.get("link") else None,
                metadata={
                    "provider": "serper",
                    **({"position": item.get("position")} if item.get("position") is not None else {}),
                },
            )
            for index, item in enumerate(items[:query.limit])
            if isinstance(item, Mapping)
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> Mapping[str, Any]:
        if self._client is not None:
            response = await self._client.request(method, url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("web search provider returned a non-object JSON response")
        return payload

    async def _tavily(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            "https://api.tavily.com/search",
            json={**dict(query.filters), "api_key": self._credential, "query": query.text, "max_results": query.limit},
        )

    async def _brave(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            params={**dict(query.filters), "q": query.text, "count": query.limit},
            headers={"X-Subscription-Token": self._credential},
        )

    async def _serper(self, query: RetrievalQuery) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            "https://google.serper.dev/search",
            json={**dict(query.filters), "q": query.text, "num": query.limit},
            headers={"X-API-KEY": self._credential},
        )

@dataclass(frozen=True)
class MultimodalMessage:
    role: MessageRole
    parts: tuple[ContentPart, ...]

    def to_model_message(self) -> ModelMessage:
        return ModelMessage(role=self.role, content=self.parts)

RealtimeEventType = Literal[
    "audio_input",
    "audio_output",
    "speech_started",
    "speech_stopped",
    "text_delta",
    "response_completed",
    "interrupted",
]

@dataclass(frozen=True)
class RealtimeEvent:
    type: RealtimeEventType
    data: Any = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

@runtime_checkable
class RealtimeTransport(Protocol):
    async def send(self, event: RealtimeEvent) -> None: ...
    def events(self) -> AsyncIterator[RealtimeEvent]: ...
    async def close(self) -> None: ...

class RealtimeSession:
    def __init__(self, transport: RealtimeTransport) -> None:
        self.transport = transport
        self.interrupted = False

    async def send_audio(self, data: Any) -> None:
        await self.transport.send(RealtimeEvent("audio_input", data))

    async def send_text(self, text: str) -> None:
        await self.transport.send(RealtimeEvent("text_delta", text))

    async def interrupt(self) -> None:
        self.interrupted = True
        await self.transport.send(RealtimeEvent("interrupted"))

    async def collect_until_complete(self) -> tuple[RealtimeEvent, ...]:
        collected: list[RealtimeEvent] = []
        async for event in self.transport.events():
            collected.append(event)
            if event.type in ("response_completed", "interrupted"):
                break
        return tuple(collected)

    async def close(self) -> None:
        await self.transport.close()


ProgressStatus = Literal[
    "queued",
    "running",
    "waiting_for_input",
    "waiting_for_approval",
    "completed",
    "failed",
    "canceled",
]

@dataclass(frozen=True)
class ProgressEvent:
    task_id: str
    status: ProgressStatus
    message: str = ""
    active_step: str | None = None
    completed_steps: tuple[str, ...] = ()
    pending_approval_id: str | None = None
    waiting_on: str | None = None
    progress: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("progress must be between 0 and 1")

ProgressListener = Callable[[ProgressEvent], Any]

class ProgressReporter:
    def __init__(self) -> None:
        self._listeners: list[ProgressListener] = []
        self._events: list[ProgressEvent] = []

    def subscribe(self, listener: ProgressListener) -> None:
        self._listeners.append(listener)

    def emit(self, event: ProgressEvent) -> ProgressEvent:
        self._events.append(event)
        for listener in tuple(self._listeners):
            listener(event)
        return event

    @property
    def events(self) -> tuple[ProgressEvent, ...]:
        return tuple(self._events)

@dataclass(frozen=True)
class RuntimeRequest:
    agent: str
    messages: tuple[ModelMessage, ...]
    session_id: str | None = None
    task_id: str | None = None
    structured: bool = False
    stream: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RuntimeResponse:
    task_id: str
    session_id: str | None
    result: Any
    events: tuple[ProgressEvent, ...] = ()

RuntimeExecutor = Callable[[RuntimeRequest], Awaitable[Any]]

class CLIInterface:
    def __init__(
        self,
        executor: RuntimeExecutor,
        *,
        session_memory: ShortTermSessionMemory | None = None,
    ) -> None:
        self.executor = executor
        self.session_memory = session_memory

    async def run(
        self,
        request: RuntimeRequest,
        *,
        stdin: str | None = None,
    ) -> RuntimeResponse:
        effective = request
        if stdin is not None:
            effective = replace(
                request,
                messages=request.messages + (
                    ModelMessage(
                        role="user",
                        content=(ContentPart(type="text", text=stdin),),
                    ),
                ),
            )
        result = await self.executor(effective)
        if (
            self.session_memory is not None
            and effective.session_id is not None
        ):
            session = SessionRef(effective.session_id, "cli")
            self.session_memory.append_messages(session, effective.messages)
        return RuntimeResponse(
            task_id=effective.task_id or "cli-task",
            session_id=effective.session_id,
            result=result,
        )

    def resume(self, session_id: str) -> SessionSnapshot | None:
        if self.session_memory is None:
            return None
        return self.session_memory.snapshot(SessionRef(session_id, "cli"))

@dataclass(frozen=True)
class APIRequest:
    operation: Literal[
        "run",
        "session.get",
        "task.get",
        "task.cancel",
        "events.list",
        "artifact.get",
        "approval.resolve",
        "status.get",
    ]
    payload: Mapping[str, Any] = field(default_factory=dict)

APIOperationHandler = Callable[[Mapping[str, Any]], Awaitable[Any]]

class APIInterface:
    def __init__(self) -> None:
        self._handlers: dict[str, APIOperationHandler] = {}

    def register(
        self,
        operation: APIRequest.__annotations__["operation"],
        handler: APIOperationHandler,
    ) -> None:
        self._handlers[operation] = handler

    async def handle(self, request: APIRequest) -> Any:
        handler = self._handlers.get(request.operation)
        if handler is None:
            raise KeyError(request.operation)
        return await handler(request.payload)

@dataclass(frozen=True)
class IDEDiagnostic:
    path: str
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    line: int | None = None
    column: int | None = None

@dataclass(frozen=True)
class IDEContext:
    workspace_id: str
    current_file: str | None = None
    selection: str | None = None
    diagnostics: tuple[IDEDiagnostic, ...] = ()
    diff: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IDEProgress:
    task_id: str
    message: str
    fraction: float | None = None

class IDEIntegration:
    def __init__(self, workspace: WorkspaceFiles) -> None:
        self.workspace = workspace

    def read_current(self, context: IDEContext) -> str | None:
        if context.current_file is None:
            return None
        return self.workspace.read_text(context.current_file)

    def apply_patch(self, path: str, patch: str) -> None:
        self.workspace.apply_unified_patch(path, patch)



# Lazily loaded interfaces/observability ledgers that are not needed by AgentLoop.
@dataclass(frozen=True)
class ChatEnvelope:
    channel: str
    user_id: str
    thread_id: str
    text: str
    message_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ChatReply:
    channel: str
    thread_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class ChatAdapter(Protocol):
    async def send(self, reply: ChatReply) -> Any: ...

class ChatSessionBridge:
    def __init__(
        self,
        adapter: ChatAdapter,
        memory: ShortTermSessionMemory,
    ) -> None:
        self.adapter = adapter
        self.memory = memory

    def session_ref(self, envelope: ChatEnvelope) -> SessionRef:
        return SessionRef(
            session_id=f"{envelope.channel}:{envelope.user_id}",
            thread_id=envelope.thread_id,
        )

    def ingest(self, envelope: ChatEnvelope) -> SessionRef:
        session = self.session_ref(envelope)
        self.memory.append_messages(
            session,
            (
                ModelMessage(
                    role="user",
                    content=(ContentPart(type="text", text=envelope.text),),
                ),
            ),
        )
        return session

    async def reply(self, envelope: ChatEnvelope, text: str) -> Any:
        return await self.adapter.send(
            ChatReply(
                channel=envelope.channel,
                thread_id=envelope.thread_id,
                text=text,
            )
        )

@dataclass(frozen=True)
class ApprovalPresentation:
    approval_id: str
    title: str
    summary: str
    consequences: tuple[str, ...] = ()
    diff: str | None = None
    choices: tuple[ApprovalDecision, ...] = ("allow", "deny")
    metadata: Mapping[str, Any] = field(default_factory=dict)

def approval_presentation(
    request: ApprovalRequest,
    *,
    title: str | None = None,
    summary: str | None = None,
    consequences: Sequence[str] = (),
    diff: str | None = None,
) -> ApprovalPresentation:
    return ApprovalPresentation(
        approval_id=request.id,
        title=title or f"Approve {request.call.name}",
        summary=summary or request.reason,
        consequences=tuple(consequences),
        diff=diff,
        metadata={
            "tool": request.call.name,
            "side_effect": request.side_effect,
            "session_id": request.session_id,
        },
    )


SpanKind = Literal[
    "task", "agent", "turn", "model", "tool", "subagent",
    "guardrail", "queue", "remote",
]
SpanStatus = Literal["running", "ok", "error", "canceled"]

@dataclass(frozen=True)
class TraceSpan:
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    parent_span_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    status: SpanStatus = "running"
    started_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    ended_at_ms: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.ended_at_ms is None:
            return None
        return max(0, self.ended_at_ms - self.started_at_ms)

class TraceRecorder:
    def __init__(self) -> None:
        self._spans: dict[str, TraceSpan] = {}
        self._order: list[str] = []

    def start(
        self,
        *,
        span_id: str,
        trace_id: str,
        name: str,
        kind: SpanKind,
        parent_span_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        started_at_ms: int | None = None,
    ) -> TraceSpan:
        if span_id in self._spans:
            raise ValueError(f"duplicate span id: {span_id}")
        if parent_span_id is not None:
            parent = self._spans.get(parent_span_id)
            if parent is None:
                raise KeyError(parent_span_id)
            if parent.trace_id != trace_id:
                raise ValueError("child span trace_id must match parent")
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            task_id=task_id,
            session_id=session_id,
            attributes=dict(attributes or {}),
            started_at_ms=(
                started_at_ms
                if started_at_ms is not None
                else int(time.time() * 1000)
            ),
        )
        self._spans[span_id] = span
        self._order.append(span_id)
        return span

    def finish(
        self,
        span_id: str,
        *,
        status: SpanStatus = "ok",
        attributes: Mapping[str, Any] | None = None,
        ended_at_ms: int | None = None,
    ) -> TraceSpan:
        span = self._spans.get(span_id)
        if span is None:
            raise KeyError(span_id)
        merged = {**dict(span.attributes), **dict(attributes or {})}
        finished = replace(
            span,
            status=status,
            ended_at_ms=(
                ended_at_ms
                if ended_at_ms is not None
                else int(time.time() * 1000)
            ),
            attributes=merged,
        )
        self._spans[span_id] = finished
        return finished

    def get(self, span_id: str) -> TraceSpan:
        if span_id not in self._spans:
            raise KeyError(span_id)
        return self._spans[span_id]

    def spans(
        self,
        *,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> tuple[TraceSpan, ...]:
        values = tuple(self._spans[span_id] for span_id in self._order)
        if trace_id is not None:
            values = tuple(span for span in values if span.trace_id == trace_id)
        if parent_span_id is not None:
            values = tuple(
                span for span in values
                if span.parent_span_id == parent_span_id
            )
        return values

LogSeverity = Literal["debug", "info", "warning", "error"]

@dataclass(frozen=True)
class StructuredLogRecord:
    message: str
    severity: LogSeverity = "info"
    correlation_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

class StructuredLogger:
    def __init__(self, redactor: PrivacyRedactor | None = None) -> None:
        self.redactor = redactor
        self._records: list[StructuredLogRecord] = []

    def emit(self, record: StructuredLogRecord) -> StructuredLogRecord:
        if self.redactor is not None:
            record = replace(
                record,
                message=str(self.redactor.redact(record.message)),
                metadata=self.redactor.redact(dict(record.metadata)),
            )
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[StructuredLogRecord, ...]:
        return tuple(self._records)


@dataclass(frozen=True)
class TokenUsageRecord:
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    other_tokens: Mapping[str, int] = field(default_factory=dict)
    user_id: str | None = None
    tenant_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    model: str | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + sum(self.other_tokens.values())
        )

class TokenLedger:
    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    def record(self, record: TokenUsageRecord) -> TokenUsageRecord:
        values = (
            record.prompt_tokens,
            record.output_tokens,
            record.cached_tokens,
            record.reasoning_tokens,
            *record.other_tokens.values(),
        )
        if any(value < 0 for value in values):
            raise ValueError("token counts must be non-negative")
        self._records.append(record)
        return record

    def record_model_usage(
        self,
        usage: ModelUsage,
        **attribution: Any,
    ) -> TokenUsageRecord:
        return self.record(
            TokenUsageRecord(
                prompt_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cached_tokens=usage.cached_tokens or 0,
                reasoning_tokens=usage.reasoning_tokens or 0,
                **attribution,
            )
        )

    def total(self, *, task_id: str | None = None) -> TokenUsageRecord:
        records = [
            record for record in self._records
            if task_id is None or record.task_id == task_id
        ]
        other: dict[str, int] = {}
        for record in records:
            for name, value in record.other_tokens.items():
                other[name] = other.get(name, 0) + value
        return TokenUsageRecord(
            prompt_tokens=sum(record.prompt_tokens for record in records),
            output_tokens=sum(record.output_tokens for record in records),
            cached_tokens=sum(record.cached_tokens for record in records),
            reasoning_tokens=sum(record.reasoning_tokens for record in records),
            other_tokens=other,
            task_id=task_id,
        )

CostCategory = Literal["model", "tool", "storage", "compute", "external"]

@dataclass(frozen=True)
class CostRecord:
    amount: float
    currency: str = "USD"
    category: CostCategory = "model"
    user_id: str | None = None
    tenant_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    resource: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

class CostLedger:
    def __init__(self) -> None:
        self._records: list[CostRecord] = []

    def record(self, record: CostRecord) -> CostRecord:
        if record.amount < 0:
            raise ValueError("cost amount must be non-negative")
        self._records.append(record)
        return record

    def total(
        self,
        *,
        currency: str = "USD",
        task_id: str | None = None,
        tenant_id: str | None = None,
    ) -> float:
        return sum(
            record.amount
            for record in self._records
            if record.currency == currency
            and (task_id is None or record.task_id == task_id)
            and (tenant_id is None or record.tenant_id == tenant_id)
        )

@dataclass(frozen=True)
class ReplayExchange:
    channel: Literal["model", "tool", "external"]
    key: str
    input: Any
    output: Any

@dataclass(frozen=True)
class ReplayBundle:
    replay_id: str
    state: Any = None
    events: tuple[Any, ...] = ()
    checkpoints: tuple[Any, ...] = ()
    exchanges: tuple[ReplayExchange, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

class DebugReplayStore:
    def __init__(self) -> None:
        self._bundles: dict[str, ReplayBundle] = {}

    def save(self, bundle: ReplayBundle) -> ReplayBundle:
        self._bundles[bundle.replay_id] = bundle
        return bundle

    def load(self, replay_id: str) -> ReplayBundle:
        if replay_id not in self._bundles:
            raise KeyError(replay_id)
        return self._bundles[replay_id]

    def reconstruct(self, replay_id: str) -> Mapping[str, Any]:
        bundle = self.load(replay_id)
        return {
            "state": bundle.state,
            "events": bundle.events,
            "checkpoints": bundle.checkpoints,
            "exchanges": bundle.exchanges,
            "metadata": bundle.metadata,
        }

class DeterministicReplay:
    def __init__(self, bundle: ReplayBundle) -> None:
        self.bundle = bundle
        self._positions: dict[tuple[str, str], int] = {}

    def next(
        self,
        channel: Literal["model", "tool", "external"],
        key: str,
        input: Any = None,
    ) -> Any:
        matches = [
            exchange
            for exchange in self.bundle.exchanges
            if exchange.channel == channel and exchange.key == key
        ]
        position_key = (channel, key)
        position = self._positions.get(position_key, 0)
        if position >= len(matches):
            raise LookupError(f"no recorded replay output for {channel}:{key}")
        exchange = matches[position]
        if input is not None and exchange.input != input:
            raise ValueError(f"replay input mismatch for {channel}:{key}")
        self._positions[position_key] = position + 1
        return exchange.output

class ReplayModelProvider:
    def __init__(
        self,
        replay: DeterministicReplay,
        *,
        key: str = "complete",
        name: str = "replay",
    ) -> None:
        self.replay = replay
        self.key = key
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        output = self.replay.next("model", self.key)
        if not isinstance(output, ModelResponse):
            raise TypeError("recorded model replay output must be ModelResponse")
        return output




# Lazily loaded session and long-term memory implementations.
@dataclass(frozen=True)
class SessionSnapshot:
    session: SessionRef
    messages: tuple["ModelMessage", ...] = ()
    workflow_state: WorkflowState | None = None


class ShortTermSessionMemory:
    def __init__(self, *, max_messages: int = 100) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: dict[SessionRef, list["ModelMessage"]] = {}
        self._states: dict[SessionRef, WorkflowState] = {}

    def append_messages(
        self,
        session: SessionRef,
        messages: Sequence["ModelMessage"],
    ) -> None:
        bucket = self._messages.setdefault(session, [])
        bucket.extend(messages)
        if len(bucket) > self.max_messages:
            del bucket[:-self.max_messages]

    def set_workflow_state(
        self,
        session: SessionRef,
        state: WorkflowState | None,
    ) -> None:
        if state is None:
            self._states.pop(session, None)
        else:
            self._states[session] = state

    def snapshot(self, session: SessionRef) -> SessionSnapshot:
        return SessionSnapshot(
            session=session,
            messages=tuple(self._messages.get(session, ())),
            workflow_state=self._states.get(session),
        )

    def clear(self, session: SessionRef) -> None:
        self._messages.pop(session, None)
        self._states.pop(session, None)


class TenantSessionMemory:
    def __init__(self, memory: ShortTermSessionMemory, tenant: TenantContext) -> None:
        self.memory = memory
        self.tenant = tenant

    def append_messages(self, session: SessionRef, messages: Sequence["ModelMessage"]) -> None:
        self.memory.append_messages(self.tenant.session_ref(session), messages)

    def set_workflow_state(self, session: SessionRef, state: WorkflowState | None) -> None:
        self.memory.set_workflow_state(self.tenant.session_ref(session), state)

    def snapshot(self, session: SessionRef) -> SessionSnapshot:
        return self.memory.snapshot(self.tenant.session_ref(session))

    def clear(self, session: SessionRef) -> None:
        self.memory.clear(self.tenant.session_ref(session))


MemoryKind = Literal["fact", "semantic", "episode", "procedure"]



@dataclass(frozen=True)
class MemoryRecord:
    id: str
    kind: MemoryKind
    content: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    sequence: int = 0
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at_ms: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("memory id must not be empty")
        if not self.content.strip():
            raise ValueError("memory content must not be empty")
        _validate_json_value(dict(self.metadata))
        if self.created_at_ms < 0:
            raise ValueError("memory created_at_ms must be non-negative")
        if self.expires_at_ms is not None and self.expires_at_ms < self.created_at_ms:
            raise ValueError("memory expires_at_ms must not precede created_at_ms")
        if self.schema_version < 1:
            raise ValueError("memory schema_version must be at least 1")
        if self.embedding is not None:
            if not self.embedding:
                raise ValueError("memory embedding must not be empty")
            for value in self.embedding:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("memory embedding values must be numbers")
                if not float("-inf") < float(value) < float("inf"):
                    raise ValueError("memory embedding values must be finite")


@dataclass(frozen=True)
class MemorySearchQuery:
    text: str | None = None
    scope: MemoryScope | None = None
    kinds: frozenset[MemoryKind] | None = None
    tags: frozenset[str] = frozenset()
    limit: int = 10

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("memory search limit must be non-negative")


@dataclass(frozen=True)
class MemorySearchResult:
    record: MemoryRecord
    score: float


@runtime_checkable
class LongTermMemoryStore(Protocol):
    def read(self, memory_id: str) -> MemoryRecord | None: ...
    def write(self, record: MemoryRecord) -> MemoryRecord: ...
    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord: ...
    def delete(self, memory_id: str) -> bool: ...
    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]: ...


class InMemoryLongTermMemoryStore:
    _token_pattern = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._sequence = 0

    def read(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        if record.id in self._records:
            raise ValueError(f"memory already exists: {record.id}")
        self._sequence += 1
        stored = MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=self._sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.expires_at_ms,
            schema_version=record.schema_version,
        )
        self._records[stored.id] = stored
        return stored

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        if memory_id not in self._records:
            raise KeyError(f"memory not found: {memory_id}")
        if record.id != memory_id:
            raise ValueError("memory update cannot change id")
        previous = self._records[memory_id]
        stored = MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=previous.sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.expires_at_ms,
            schema_version=record.schema_version,
        )
        self._records[memory_id] = stored
        return stored

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        query_terms = self._terms(query.text or "")
        results: list[MemorySearchResult] = []
        for record in self._records.values():
            if query.scope is not None and not query.scope.contains(record.scope):
                continue
            if query.kinds is not None and record.kind not in query.kinds:
                continue
            if query.tags and not query.tags.issubset(record.tags):
                continue

            if query_terms:
                record_terms = self._terms(
                    record.content
                    + " "
                    + " ".join(record.tags)
                    + " "
                    + " ".join(
                        f"{key} {value}"
                        for key, value in record.metadata.items()
                        if isinstance(value, (str, int, float, bool))
                    )
                )
                overlap = query_terms & record_terms
                if not overlap:
                    continue
                score = len(overlap) / len(query_terms)
            else:
                score = 1.0
            results.append(MemorySearchResult(record=record, score=score))

        results.sort(key=lambda item: (-item.score, -item.record.sequence, item.record.id))
        return tuple(results[: query.limit])

    def _terms(self, value: str) -> set[str]:
        return set(self._token_pattern.findall(value.lower()))


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


class SemanticMemory:
    def __init__(
        self,
        store: LongTermMemoryStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def write(
        self,
        *,
        memory_id: str,
        content: str,
        scope: MemoryScope = MemoryScope(),
        metadata: Mapping[str, JSONValue] | None = None,
        tags: Sequence[str] = (),
    ) -> MemoryRecord:
        embedding = tuple(float(value) for value in self.embedding_provider.embed(content))
        return self.store.write(
            MemoryRecord(
                id=memory_id,
                kind="semantic",
                content=content,
                scope=scope,
                metadata=metadata or {},
                tags=tuple(tags),
                embedding=embedding,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> tuple[MemorySearchResult, ...]:
        query_vector = tuple(
            float(value) for value in self.embedding_provider.embed(query)
        )
        candidates = self.store.search(
            MemorySearchQuery(
                scope=scope,
                kinds=frozenset({"semantic"}),
                limit=1000000,
            )
        )
        ranked: list[MemorySearchResult] = []
        for candidate in candidates:
            embedding = candidate.record.embedding
            if embedding is None or len(embedding) != len(query_vector):
                continue
            ranked.append(
                MemorySearchResult(
                    record=candidate.record,
                    score=self._cosine(query_vector, embedding),
                )
            )
        ranked.sort(key=lambda item: (-item.score, -item.record.sequence, item.record.id))
        return tuple(ranked[:limit])

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class Episode:
    id: str
    task: str
    outcome: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    decisions: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("episode id must not be empty")
        if not self.task.strip():
            raise ValueError("episode task must not be empty")
        if not self.outcome.strip():
            raise ValueError("episode outcome must not be empty")
        _validate_json_value(dict(self.metadata))


class EpisodicMemory:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store

    def remember(self, episode: Episode) -> MemoryRecord:
        payload = {
            "task": episode.task,
            "outcome": episode.outcome,
            "decisions": list(episode.decisions),
            "actions": list(episode.actions),
            "trace": list(episode.trace),
        }
        metadata = dict(episode.metadata)
        metadata["episode"] = payload
        searchable = " ".join(
            (
                episode.task,
                episode.outcome,
                *episode.decisions,
                *episode.actions,
                *episode.trace,
            )
        )
        return self.store.write(
            MemoryRecord(
                id=episode.id,
                kind="episode",
                content=searchable,
                scope=episode.scope,
                metadata=metadata,
                tags=("episode",),
            )
        )

    def search(
        self,
        text: str,
        *,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> tuple[MemorySearchResult, ...]:
        return self.store.search(
            MemorySearchQuery(
                text=text,
                scope=scope,
                kinds=frozenset({"episode"}),
                limit=limit,
            )
        )


MemoryMigration = Callable[[MemoryRecord], MemoryRecord]


@dataclass(frozen=True)
class MemoryLifecyclePolicy:
    default_retention_ms: int | None = None

    def __post_init__(self) -> None:
        if self.default_retention_ms is not None and self.default_retention_ms < 1:
            raise ValueError("default_retention_ms must be positive")


class LifecycleMemoryStore:
    def __init__(
        self,
        store: LongTermMemoryStore,
        *,
        policy: MemoryLifecyclePolicy | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryLifecyclePolicy()
        self.clock = clock or (lambda: int(time.time() * 1000))
        self._migrations: dict[int, MemoryMigration] = {}

    def register_migration(
        self,
        from_version: int,
        migration: MemoryMigration,
    ) -> None:
        if from_version < 1:
            raise ValueError("migration version must be at least 1")
        self._migrations[from_version] = migration

    def _expired(self, record: MemoryRecord) -> bool:
        return (
            record.expires_at_ms is not None
            and record.expires_at_ms <= self.clock()
        )

    def _apply_default_retention(self, record: MemoryRecord) -> MemoryRecord:
        if (
            record.expires_at_ms is not None
            or self.policy.default_retention_ms is None
        ):
            return record
        return MemoryRecord(
            id=record.id,
            kind=record.kind,
            content=record.content,
            scope=record.scope,
            metadata=record.metadata,
            tags=record.tags,
            embedding=record.embedding,
            sequence=record.sequence,
            created_at_ms=record.created_at_ms,
            expires_at_ms=record.created_at_ms + self.policy.default_retention_ms,
            schema_version=record.schema_version,
        )

    def read(self, memory_id: str) -> MemoryRecord | None:
        record = self.store.read(memory_id)
        if record is None:
            return None
        if self._expired(record):
            self.store.delete(memory_id)
            return None
        return record

    def write(self, record: MemoryRecord) -> MemoryRecord:
        return self.store.write(self._apply_default_retention(record))

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        return self.store.update(
            memory_id,
            self._apply_default_retention(record),
        )

    def delete(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        self.purge_expired()
        return self.store.search(query)

    def purge_expired(self) -> int:
        expired = [
            result.record.id
            for result in self.store.search(MemorySearchQuery(limit=1_000_000))
            if self._expired(result.record)
        ]
        for memory_id in expired:
            self.store.delete(memory_id)
        return len(expired)

    def migrate(self, memory_id: str, *, target_version: int) -> MemoryRecord:
        record = self.read(memory_id)
        if record is None:
            raise KeyError(f"memory not found: {memory_id}")
        if target_version < record.schema_version:
            raise ValueError("memory migrations cannot move backwards")
        while record.schema_version < target_version:
            migration = self._migrations.get(record.schema_version)
            if migration is None:
                raise ValueError(
                    f"no migration registered from version {record.schema_version}"
                )
            migrated = migration(record)
            if migrated.id != record.id:
                raise ValueError("memory migration cannot change id")
            if migrated.schema_version != record.schema_version + 1:
                raise ValueError("memory migration must advance exactly one version")
            record = migrated
        return self.store.update(memory_id, record)

    def compact(
        self,
        memory_ids: Sequence[str],
        *,
        compacted_id: str,
        content: str,
        kind: MemoryKind = "fact",
        scope: MemoryScope = MemoryScope(),
        metadata: Mapping[str, JSONValue] | None = None,
        tags: Sequence[str] = (),
        delete_sources: bool = True,
    ) -> MemoryRecord:
        if not memory_ids:
            raise ValueError("memory compaction requires at least one source")
        sources = []
        for memory_id in memory_ids:
            record = self.read(memory_id)
            if record is None:
                raise KeyError(f"memory not found: {memory_id}")
            sources.append(record)
        compacted_metadata = dict(metadata or {})
        compacted_metadata["compacted_from"] = list(memory_ids)
        result = self.write(
            MemoryRecord(
                id=compacted_id,
                kind=kind,
                content=content,
                scope=scope,
                metadata=compacted_metadata,
                tags=tuple(tags),
            )
        )
        if delete_sources:
            for memory_id in memory_ids:
                self.store.delete(memory_id)
        return result


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    instructions: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    script: str | None = None
    template: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("procedure id must not be empty")
        if not self.name.strip():
            raise ValueError("procedure name must not be empty")
        if not self.instructions.strip():
            raise ValueError("procedure instructions must not be empty")
        _validate_json_value(dict(self.metadata))


class ProceduralMemory:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store

    def remember(self, procedure: Procedure) -> MemoryRecord:
        payload = {
            "name": procedure.name,
            "instructions": procedure.instructions,
            "script": procedure.script,
            "template": procedure.template,
        }
        metadata = dict(procedure.metadata)
        metadata["procedure"] = payload
        searchable = " ".join(
            part
            for part in (
                procedure.name,
                procedure.instructions,
                procedure.script or "",
                procedure.template or "",
            )
            if part
        )
        return self.store.write(
            MemoryRecord(
                id=procedure.id,
                kind="procedure",
                content=searchable,
                scope=procedure.scope,
                metadata=metadata,
                tags=("procedure", *procedure.tags),
            )
        )

    def search(
        self,
        text: str,
        *,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> tuple[MemorySearchResult, ...]:
        return self.store.search(
            MemorySearchQuery(
                text=text,
                scope=scope,
                kinds=frozenset({"procedure"}),
                limit=limit,
            )
        )


@dataclass(frozen=True)
class MemoryWriteCandidate:
    id: str
    kind: MemoryKind
    content: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    relevance: float = 1.0
    sensitivity: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("relevance", self.relevance),
            ("sensitivity", self.sensitivity),
            ("confidence", self.confidence),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        _validate_json_value(dict(self.metadata))


@dataclass(frozen=True)
class MemoryWriteDecision:
    persist: bool
    reason: str


@dataclass(frozen=True)
class MemoryWritePolicy:
    min_relevance: float = 0.5
    min_confidence: float = 0.5
    max_sensitivity: float = 0.5
    reject_duplicates: bool = True
    allowed_kinds: frozenset[MemoryKind] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("min_relevance", self.min_relevance),
            ("min_confidence", self.min_confidence),
            ("max_sensitivity", self.max_sensitivity),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def decide(
        self,
        candidate: MemoryWriteCandidate,
        store: LongTermMemoryStore,
    ) -> MemoryWriteDecision:
        if self.allowed_kinds is not None and candidate.kind not in self.allowed_kinds:
            return MemoryWriteDecision(False, "kind_not_allowed")
        if candidate.relevance < self.min_relevance:
            return MemoryWriteDecision(False, "relevance_below_threshold")
        if candidate.confidence < self.min_confidence:
            return MemoryWriteDecision(False, "confidence_below_threshold")
        if candidate.sensitivity > self.max_sensitivity:
            return MemoryWriteDecision(False, "sensitivity_above_threshold")
        if self.reject_duplicates:
            duplicates = store.search(
                MemorySearchQuery(
                    text=candidate.content,
                    scope=candidate.scope,
                    kinds=frozenset({candidate.kind}),
                    limit=10,
                )
            )
            normalized = " ".join(candidate.content.lower().split())
            if any(
                " ".join(result.record.content.lower().split()) == normalized
                for result in duplicates
            ):
                return MemoryWriteDecision(False, "duplicate")
        return MemoryWriteDecision(True, "accepted")

    def persist(
        self,
        candidate: MemoryWriteCandidate,
        store: LongTermMemoryStore,
    ) -> MemoryRecord | None:
        decision = self.decide(candidate, store)
        if not decision.persist:
            return None
        return store.write(
            MemoryRecord(
                id=candidate.id,
                kind=candidate.kind,
                content=candidate.content,
                scope=candidate.scope,
                metadata=candidate.metadata,
                tags=candidate.tags,
            )
        )


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    relevance_weight: float = 0.6
    recency_weight: float = 0.2
    confidence_weight: float = 0.2
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("relevance_weight", self.relevance_weight),
            ("recency_weight", self.recency_weight),
            ("confidence_weight", self.confidence_weight),
            ("min_confidence", self.min_confidence),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def search(
        self,
        store: LongTermMemoryStore,
        query: MemorySearchQuery,
    ) -> tuple[MemorySearchResult, ...]:
        candidates = store.search(
            MemorySearchQuery(
                text=query.text,
                scope=query.scope,
                kinds=query.kinds,
                tags=query.tags,
                limit=max(query.limit * 10, query.limit),
            )
        )
        if not candidates:
            return ()

        max_sequence = max(result.record.sequence for result in candidates) or 1
        ranked: list[MemorySearchResult] = []
        for result in candidates:
            confidence_value = result.record.metadata.get("confidence", 1.0)
            confidence = (
                float(confidence_value)
                if isinstance(confidence_value, (int, float))
                and not isinstance(confidence_value, bool)
                else 1.0
            )
            if confidence < self.min_confidence:
                continue
            recency = result.record.sequence / max_sequence
            score = (
                result.score * self.relevance_weight
                + recency * self.recency_weight
                + confidence * self.confidence_weight
            )
            ranked.append(MemorySearchResult(record=result.record, score=score))

        ranked.sort(
            key=lambda item: (-item.score, -item.record.sequence, item.record.id)
        )
        return tuple(ranked[: query.limit])


class ScopedMemoryStore:
    def __init__(
        self,
        store: LongTermMemoryStore,
        scope: MemoryScope,
    ) -> None:
        self.store = store
        self.scope = scope

    def _assert_scope(self, scope: MemoryScope) -> None:
        if not self.scope.contains(scope):
            raise PermissionError("memory scope is outside the bound scope")

    def read(self, memory_id: str) -> MemoryRecord | None:
        record = self.store.read(memory_id)
        if record is None or not self.scope.contains(record.scope):
            return None
        return record

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self._assert_scope(record.scope)
        return self.store.write(record)

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord:
        existing = self.store.read(memory_id)
        if existing is None or not self.scope.contains(existing.scope):
            raise KeyError(f"memory not found: {memory_id}")
        self._assert_scope(record.scope)
        return self.store.update(memory_id, record)

    def delete(self, memory_id: str) -> bool:
        existing = self.store.read(memory_id)
        if existing is None or not self.scope.contains(existing.scope):
            return False
        return self.store.delete(memory_id)

    def search(self, query: MemorySearchQuery) -> tuple[MemorySearchResult, ...]:
        effective_scope = query.scope or self.scope
        if not self.scope.contains(effective_scope):
            raise PermissionError("memory query scope is outside the bound scope")
        return self.store.search(
            MemorySearchQuery(
                text=query.text,
                scope=effective_scope,
                kinds=query.kinds,
                tags=query.tags,
                limit=query.limit,
            )
        )




# Preserve the historical public identity of lazily extracted symbols so
# introspection and pickle lookup continue through agent_rt.<Name>.
for _name, _value in tuple(globals().items()):
    if (
        not _name.startswith("_")
        and _name not in vars(_core)
        and getattr(_value, "__module__", None) == __name__
    ):
        _value.__module__ = "agent_rt"

__all__ = tuple(sorted(
    name for name in globals()
    if not name.startswith("_") and name not in vars(_core)
))

@dataclass(frozen=True)
class WorkspaceRecord:
    workspace_id: str
    backend: str = "memory"
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must not be empty")
        if not self.backend.strip():
            raise ValueError("workspace backend must not be empty")
        _validate_json_value(dict(self.metadata))

class PersistentWorkspaceStore:
    def __init__(
        self,
        backends: FileSystemBackendRegistry | None = None,
    ) -> None:
        self.backends = backends or FileSystemBackendRegistry()
        if "memory" not in self.backends.list():
            self.backends.register("memory", lambda _workspace_id: InMemoryFileSystem())
        self._records: dict[str, WorkspaceRecord] = {}
        self._filesystems: dict[str, FileSystem] = {}

    def create(
        self,
        workspace_id: str,
        *,
        backend: str = "memory",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> WorkspaceRecord:
        if workspace_id in self._records:
            raise ValueError(f"workspace already exists: {workspace_id}")
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            backend=backend,
            metadata=metadata or {},
        )
        filesystem = self.backends.create(backend, workspace_id)
        self._records[workspace_id] = record
        self._filesystems[workspace_id] = filesystem
        return record

    def get(self, workspace_id: str) -> WorkspaceRecord | None:
        return self._records.get(workspace_id)

    def open(self, workspace_id: str) -> WorkspaceFiles:
        if workspace_id not in self._records:
            raise KeyError(f"workspace not found: {workspace_id}")
        return WorkspaceFiles(self._filesystems[workspace_id])

    def delete(self, workspace_id: str) -> bool:
        existed = self._records.pop(workspace_id, None) is not None
        self._filesystems.pop(workspace_id, None)
        return existed

    def list(self) -> tuple[WorkspaceRecord, ...]:
        return tuple(
            self._records[key]
            for key in sorted(self._records)
        )

class TenantWorkspaceStore:
    def __init__(self, store: PersistentWorkspaceStore, tenant: TenantContext) -> None:
        self.store = store
        self.tenant = tenant

    def create(
        self,
        workspace_id: str,
        *,
        backend: str = "memory",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> WorkspaceRecord:
        effective_metadata = dict(metadata or {})
        effective_metadata["tenant_id"] = self.tenant.tenant_id
        return self.store.create(
            self.tenant.workspace_id(workspace_id),
            backend=backend,
            metadata=effective_metadata,
        )

    def get(self, workspace_id: str) -> WorkspaceRecord | None:
        return self.store.get(self.tenant.workspace_id(workspace_id))

    def open(self, workspace_id: str) -> WorkspaceFiles:
        return self.store.open(self.tenant.workspace_id(workspace_id))

    def delete(self, workspace_id: str) -> bool:
        return self.store.delete(self.tenant.workspace_id(workspace_id))

    def list(self) -> tuple[WorkspaceRecord, ...]:
        prefix = f"{self.tenant.tenant_id}::workspace::"
        return tuple(
            record for record in self.store.list()
            if record.workspace_id.startswith(prefix)
        )

ArtifactKind = Literal[
    "document",
    "code",
    "dataset",
    "report",
    "image",
    "archive",
    "other",
]

@dataclass(frozen=True)
class ProvenanceRecord:
    source: str | None = None
    model: str | None = None
    tool: str | None = None
    agent: str | None = None
    transformation: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_json_value(dict(self.metadata))

@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: ArtifactKind
    name: str
    media_type: str = "application/octet-stream"
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    status: ArtifactStatus = "draft"
    retention_until_ms: int | None = None
    location: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if not self.name.strip():
            raise ValueError("artifact name must not be empty")
        if not self.media_type.strip():
            raise ValueError("artifact media_type must not be empty")
        _validate_json_value(dict(self.metadata))
        if self.retention_until_ms is not None and self.retention_until_ms < self.created_at_ms:
            raise ValueError("artifact retention_until_ms must not precede created_at_ms")

@dataclass(frozen=True)
class ArtifactVersion:
    artifact_id: str
    version: int
    data: Any
    parent_version: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    provenance: tuple[ProvenanceRecord, ...] = ()
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact version artifact_id must not be empty")
        if self.version < 1:
            raise ValueError("artifact version must be at least 1")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("artifact parent_version must precede version")
        _validate_json_value(dict(self.metadata))

@runtime_checkable
class ArtifactRepository(Protocol):
    def create(
        self,
        artifact: Artifact,
        data: Any,
        *,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion: ...
    def add_version(
        self,
        artifact_id: str,
        data: Any,
        *,
        parent_version: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion: ...
    def get(self, artifact_id: str) -> Artifact: ...
    def get_version(
        self,
        artifact_id: str,
        version: int | None = None,
    ) -> ArtifactVersion: ...
    def list_versions(self, artifact_id: str) -> tuple[ArtifactVersion, ...]: ...
    def diff_text(
        self,
        artifact_id: str,
        from_version: int,
        to_version: int,
    ) -> str: ...
    def finalize(self, artifact_id: str) -> Artifact: ...
    def transfer(self, artifact_id: str, location: str) -> Artifact: ...
    def delete(
        self,
        artifact_id: str,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> bool: ...

class InMemoryArtifactRepository:
    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._versions: dict[str, list[ArtifactVersion]] = {}

    def create(
        self,
        artifact: Artifact,
        data: Any,
        *,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion:
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"artifact already exists: {artifact.artifact_id}")
        self._artifacts[artifact.artifact_id] = artifact
        version = ArtifactVersion(
            artifact_id=artifact.artifact_id,
            version=1,
            data=data,
            metadata={},
            provenance=tuple(provenance),
        )
        self._versions[artifact.artifact_id] = [version]
        return version

    def add_version(
        self,
        artifact_id: str,
        data: Any,
        *,
        parent_version: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        provenance: Sequence[ProvenanceRecord] = (),
    ) -> ArtifactVersion:
        if artifact_id not in self._artifacts:
            raise KeyError(f"artifact not found: {artifact_id}")
        artifact = self._artifacts[artifact_id]
        if artifact.status == "finalized":
            raise ValueError("finalized artifact cannot accept new versions")
        versions = self._versions[artifact_id]
        next_version = len(versions) + 1
        parent = parent_version if parent_version is not None else versions[-1].version
        if parent < 1 or parent >= next_version:
            raise ValueError("artifact parent_version is invalid")
        if not any(item.version == parent for item in versions):
            raise ValueError(f"artifact parent version not found: {parent}")
        version = ArtifactVersion(
            artifact_id=artifact_id,
            version=next_version,
            data=data,
            parent_version=parent,
            metadata=metadata or {},
            provenance=tuple(provenance),
        )
        versions.append(version)
        return version

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"artifact not found: {artifact_id}") from exc

    def get_version(
        self,
        artifact_id: str,
        version: int | None = None,
    ) -> ArtifactVersion:
        if artifact_id not in self._versions:
            raise KeyError(f"artifact not found: {artifact_id}")
        versions = self._versions[artifact_id]
        if version is None:
            return versions[-1]
        for item in versions:
            if item.version == version:
                return item
        raise KeyError(f"artifact version not found: {artifact_id}@{version}")

    def list_versions(self, artifact_id: str) -> tuple[ArtifactVersion, ...]:
        if artifact_id not in self._versions:
            raise KeyError(f"artifact not found: {artifact_id}")
        return tuple(self._versions[artifact_id])

    def diff_text(
        self,
        artifact_id: str,
        from_version: int,
        to_version: int,
    ) -> str:
        import difflib

        left = self.get_version(artifact_id, from_version).data
        right = self.get_version(artifact_id, to_version).data
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError("artifact text diff requires string version data")
        return "".join(
            difflib.unified_diff(
                left.splitlines(keepends=True),
                right.splitlines(keepends=True),
                fromfile=f"{artifact_id}@{from_version}",
                tofile=f"{artifact_id}@{to_version}",
            )
        )


    def finalize(self, artifact_id: str) -> Artifact:
        artifact = self.get(artifact_id)
        if artifact.status == "finalized":
            return artifact
        updated = replace(artifact, status="finalized")
        self._artifacts[artifact_id] = updated
        return updated

    def transfer(self, artifact_id: str, location: str) -> Artifact:
        if not location.strip():
            raise ValueError("artifact transfer location must not be empty")
        artifact = self.get(artifact_id)
        updated = replace(artifact, location=location)
        self._artifacts[artifact_id] = updated
        return updated

    def delete(
        self,
        artifact_id: str,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> bool:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return False
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if (
            not force
            and artifact.retention_until_ms is not None
            and now < artifact.retention_until_ms
        ):
            raise PermissionError("artifact is still within its retention period")
        self._artifacts.pop(artifact_id, None)
        self._versions.pop(artifact_id, None)
        return True

def sandbox_resource_limits_from_env(
    environment: Mapping[str, str] | None = None,
) -> "SandboxResourceLimits":
    defaults = _core.tool_execution_limits_from_env(environment)
    return SandboxResourceLimits(
        cpu_seconds=defaults.cpu_seconds,
        memory_bytes=defaults.memory_bytes,
        timeout_seconds=defaults.timeout_seconds,
    )


@dataclass(frozen=True)
class SandboxResourceLimits:
    cpu_seconds: float | None = None
    memory_bytes: int | None = None
    disk_bytes: int | None = None
    process_count: int | None = None
    timeout_seconds: float | None = None
    output_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("cpu_seconds", self.cpu_seconds),
            ("memory_bytes", self.memory_bytes),
            ("disk_bytes", self.disk_bytes),
            ("process_count", self.process_count),
            ("timeout_seconds", self.timeout_seconds),
            ("output_bytes", self.output_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        # Container runtimes treat a 0 memory/pids limit as "unlimited", while
        # rlimits treat it as "nothing may run"; reject the ambiguity outright.
        for name, value in (
            ("memory_bytes", self.memory_bytes),
            ("process_count", self.process_count),
        ):
            if value is not None and value == 0:
                raise ValueError(f"{name} must be positive when provided")

SandboxNetworkMode = Literal["none", "allowlist", "unrestricted"]
SANDBOX_NETWORK_MODE_ENV = "AGENT_RT_SANDBOX_NETWORK_MODE"
SANDBOX_ALLOWED_DOMAINS_ENV = "AGENT_RT_SANDBOX_ALLOWED_DOMAINS"
SANDBOX_BLOCKED_DOMAINS_ENV = "AGENT_RT_SANDBOX_BLOCKED_DOMAINS"
SANDBOX_ALLOW_HTTP_ENV = "AGENT_RT_SANDBOX_ALLOW_HTTP"
SANDBOX_ALLOW_WEBSOCKET_ENV = "AGENT_RT_SANDBOX_ALLOW_WEBSOCKET"
SANDBOX_ALLOW_IP_ENV = "AGENT_RT_SANDBOX_ALLOW_IP"
SANDBOX_ALLOW_PROXY_ENV = "AGENT_RT_SANDBOX_ALLOW_PROXY"
SANDBOX_NETWORK_BPS_ENV = "AGENT_RT_SANDBOX_NETWORK_MAX_BYTES_PER_SECOND"
SANDBOX_NETWORK_BYTES_ENV = "AGENT_RT_SANDBOX_NETWORK_MAX_TRANSFER_BYTES"
_SANDBOX_PROXY_ENV_KEYS = frozenset({
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
})


def _sandbox_env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _sandbox_env_nonnegative_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sandbox_domain_list(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class SandboxNetworkPolicy:
    mode: SandboxNetworkMode = "none"
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    proxy_url: str | None = None
    allow_http: bool = False
    allow_websocket: bool = False
    allow_ip_addresses: bool = False
    allow_proxy: bool = False
    max_bytes_per_second: int | None = None
    max_transfer_bytes: int | None = None

    def __post_init__(self) -> None:
        import ipaddress
        allowed = tuple(domain.lower().strip(".") for domain in self.allowed_domains)
        blocked = tuple(domain.lower().strip(".") for domain in self.blocked_domains)
        if any(not domain for domain in allowed + blocked):
            raise ValueError("sandbox network domains must not be empty")
        for domain in allowed + blocked:
            try:
                ipaddress.ip_address(domain)
            except ValueError:
                continue
            if not self.allow_ip_addresses:
                raise ValueError("sandbox network domain rules must use domain names, not IP addresses")
        object.__setattr__(self, "allowed_domains", allowed)
        object.__setattr__(self, "blocked_domains", blocked)
        if self.proxy_url is not None:
            if not self.proxy_url.strip():
                raise ValueError("sandbox proxy_url must not be blank")
            if not self.allow_proxy:
                raise ValueError("sandbox proxy use is disabled")
        for name, value in (
            ("max_bytes_per_second", self.max_bytes_per_second),
            ("max_transfer_bytes", self.max_transfer_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")

    @staticmethod
    def _matches(domain: str, patterns: Sequence[str]) -> bool:
        normalized = domain.lower().strip(".")
        return any(
            normalized == pattern or normalized.endswith("." + pattern)
            for pattern in patterns
        )

    def allows(self, target: str) -> bool:
        import ipaddress
        from urllib.parse import urlparse

        # Reject characters on which URL parsers disagree (WHATWG treats "\\" as
        # "/"), so the host we check is the host a client would actually reach.
        if "\\" in target or any(ord(ch) < 33 or ord(ch) == 127 for ch in target):
            return False
        parsed = urlparse(target if "://" in target else "//" + target)
        scheme = parsed.scheme.lower()
        if scheme in {"ws", "wss"} and not self.allow_websocket:
            return False
        if scheme == "http" and not self.allow_http:
            return False
        if scheme and scheme not in {"https", "http", "ws", "wss"}:
            return False
        domain = (parsed.hostname or target).lower().strip(".")
        if domain == "localhost" and not self.allow_ip_addresses:
            return False
        try:
            ipaddress.ip_address(domain)
        except ValueError:
            pass
        else:
            if not self.allow_ip_addresses:
                return False
        if self._matches(domain, self.blocked_domains):
            return False
        if self.mode == "none":
            return False
        if self.mode == "unrestricted":
            return True
        return self._matches(domain, self.allowed_domains)


def sandbox_network_policy_from_env(
    environment: Mapping[str, str] | None = None,
) -> SandboxNetworkPolicy:
    env = os.environ if environment is None else environment
    mode = env.get(SANDBOX_NETWORK_MODE_ENV, "none").strip().lower()
    if mode not in {"none", "allowlist", "unrestricted"}:
        raise ValueError(
            f"{SANDBOX_NETWORK_MODE_ENV} must be none, allowlist, or unrestricted"
        )
    return SandboxNetworkPolicy(
        mode=mode,
        allowed_domains=_sandbox_domain_list(env.get(SANDBOX_ALLOWED_DOMAINS_ENV)),
        blocked_domains=_sandbox_domain_list(env.get(SANDBOX_BLOCKED_DOMAINS_ENV)),
        allow_http=_sandbox_env_bool(env, SANDBOX_ALLOW_HTTP_ENV),
        allow_websocket=_sandbox_env_bool(env, SANDBOX_ALLOW_WEBSOCKET_ENV),
        allow_ip_addresses=_sandbox_env_bool(env, SANDBOX_ALLOW_IP_ENV),
        allow_proxy=_sandbox_env_bool(env, SANDBOX_ALLOW_PROXY_ENV),
        max_bytes_per_second=_sandbox_env_nonnegative_int(env, SANDBOX_NETWORK_BPS_ENV),
        max_transfer_bytes=_sandbox_env_nonnegative_int(env, SANDBOX_NETWORK_BYTES_ENV),
    )


def _sanitize_sandbox_environment(
    environment: Mapping[str, str],
    policy: SandboxNetworkPolicy,
) -> dict[str, str]:
    sanitized = dict(environment)
    if _SANDBOX_PROXY_ENV_KEYS.intersection(sanitized) and not policy.allow_proxy:
        raise ValueError("sandbox proxy environment variables are disabled")
    return sanitized


def _requires_advanced_network_enforcement(policy: SandboxNetworkPolicy) -> bool:
    return bool(
        policy.mode == "allowlist"
        or policy.blocked_domains
        or not policy.allow_http
        or not policy.allow_websocket
        or not policy.allow_ip_addresses
        or policy.max_bytes_per_second is not None
        or policy.max_transfer_bytes is not None
    )

@dataclass(frozen=True)
class SandboxSnapshot:
    snapshot_id: str
    source_session_id: str
    files: Mapping[str, bytes]
    cwd: str
    environment: Mapping[str, str]
    runtime_versions: Mapping[str, str]
    installed_packages: Mapping[str, tuple[str, ...]]
    interpreter_state: Mapping[str, Mapping[str, JSONValue]]
    network_policy: SandboxNetworkPolicy
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("sandbox snapshot_id must not be empty")
        if not self.source_session_id.strip():
            raise ValueError("sandbox snapshot source_session_id must not be empty")

@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...]
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: bytes | None = None

    def __post_init__(self) -> None:
        if not self.argv or any(not part for part in self.argv):
            raise ValueError("sandbox command argv must contain non-empty arguments")
        for key, value in self.env.items():
            if not key or "\x00" in key or "\x00" in value:
                raise ValueError("sandbox environment contains an invalid entry")

@dataclass(frozen=True)
class SandboxCommandResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    duration_ms: int = 0
    truncated: bool = False

@runtime_checkable
class SandboxBackend(Protocol):
    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult: ...

SandboxCommandRunner = Callable[
    [
        str,
        SandboxCommand,
        WorkspaceFiles,
        SandboxResourceLimits,
        Mapping[str, str],
        SandboxNetworkPolicy,
    ],
    Awaitable[SandboxCommandResult],
]

class CallbackSandboxBackend:
    def __init__(self, runner: SandboxCommandRunner) -> None:
        self.runner = runner

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        return await self.runner(
            session_id,
            command,
            workspace,
            limits,
            environment,
            network_policy,
        )
SandboxBackendName = Literal["native", "docker", "e2b", "microsandbox", "swe-rex"]
SANDBOX_BACKEND_ENV = "AGENT_RT_SANDBOX_BACKEND"
_INVALID_SANDBOX_BACKEND_NAMES = frozenset(
    {"", "0", "disable", "disabled", "false", "nil", "no", "none", "null", "off"}
)


def _normalized_sandbox_backend_name(value: str | None) -> SandboxBackendName:
    if value is None:
        return "native"
    normalized = value.strip().lower()
    if normalized in _INVALID_SANDBOX_BACKEND_NAMES:
        raise ValueError(
            f"{SANDBOX_BACKEND_ENV} must name an enabled sandbox backend; "
            "sandboxing cannot be disabled"
        )
    if normalized not in {"native", "docker", "e2b", "microsandbox", "swe-rex"}:
        raise ValueError(
            f"unsupported sandbox backend {value!r}; expected native, docker, e2b, microsandbox, or swe-rex"
        )
    return normalized


# Ceiling for captured sandbox output when the caller sets no output_bytes
# limit. Output beyond it is discarded as it is read, never buffered.
DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024


async def _collect_process_result(
    process: Any,
    stdin: bytes | None,
    *,
    output_limit: int,
    kill: Callable[[], Awaitable[None] | None],
    started: float,
) -> SandboxCommandResult:
    """Feed stdin and drain stdout/stderr with a hard memory cap.

    If this coroutine is cancelled (timeouts are implemented by cancellation)
    or fails, the process is killed and reaped before the error propagates, so
    a timed-out command never keeps running in the background.
    """
    truncated = False

    async def drain(stream: Any) -> bytes:
        nonlocal truncated
        buffer = bytearray()
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return bytes(buffer)
            room = output_limit - len(buffer)
            if room > 0:
                buffer += chunk[:room]
            if len(chunk) > room:
                truncated = True

    async def feed() -> None:
        if process.stdin is None:
            return
        try:
            if stdin is not None:
                process.stdin.write(stdin)
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with suppress(Exception):
                process.stdin.close()

    try:
        stdout, stderr, _ = await asyncio.gather(
            drain(process.stdout), drain(process.stderr), feed()
        )
        await process.wait()
    except BaseException:
        outcome = kill()
        if hasattr(outcome, "__await__"):
            with suppress(Exception):
                await outcome
        with suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        raise
    return SandboxCommandResult(
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        truncated=truncated,
    )


def _native_preexec(
    uid: int,
    gid: int,
    limits: SandboxResourceLimits,
) -> Callable[[], None]:
    def configure() -> None:
        import resource

        if limits.cpu_seconds is not None:
            cpu = max(1, math.ceil(limits.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if limits.memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (limits.memory_bytes, limits.memory_bytes),
            )
        if limits.process_count is not None:
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.process_count, limits.process_count),
            )
        if limits.disk_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (limits.disk_bytes, limits.disk_bytes),
            )
        # Drop every inherited supplementary group before changing identity;
        # otherwise root's groups survive setuid(). Failure aborts the spawn.
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return configure


class NativeSandboxBackend:
    """Runs commands as a dedicated restricted POSIX uid/gid."""

    def __init__(
        self,
        *,
        uid: int,
        gid: int | None = None,
        root: str | None = None,
    ) -> None:
        if uid < 0 or (gid is not None and gid < 0):
            raise ValueError("native sandbox uid/gid must not be negative")
        self.uid = uid
        self.gid = gid
        self.root = os.path.abspath(root) if root else None

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        del session_id, workspace
        if os.name != "posix":
            raise RuntimeError("native sandbox backend requires a POSIX platform")
        if network_policy.mode == "none":
            raise RuntimeError(
                "native sandbox backend cannot verify network isolation; use docker for no-network execution"
            )
        if _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "native sandbox backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, or bandwidth/transfer network controls"
            )
        if network_policy.allowed_domains or network_policy.blocked_domains or network_policy.proxy_url:
            raise RuntimeError(
                "native sandbox backend does not implement domain/proxy policy"
            )
        base = os.path.abspath(self.root or os.getcwd())
        cwd = base
        if command.cwd:
            cwd = os.path.abspath(os.path.join(base, command.cwd))
            if os.path.commonpath((base, cwd)) != base:
                raise ValueError("sandbox working directory escapes native root")
        child_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        child_env.update(_sanitize_sandbox_environment(environment, network_policy))
        gid = self._resolve_gid()
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=cwd,
            env=child_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_native_preexec(self.uid, gid, limits),
        )

        def kill_group() -> None:
            # start_new_session made the child a group leader: kill descendants too.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(ProcessLookupError):
                process.kill()

        return await _collect_process_result(
            process,
            command.stdin,
            output_limit=(
                limits.output_bytes
                if limits.output_bytes is not None
                else DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES
            ),
            kill=kill_group,
            started=started,
        )

    def _resolve_gid(self) -> int:
        """Return the explicit gid, or the account's primary gid (never root's)."""
        if self.gid is not None:
            return self.gid
        try:
            import pwd

            return pwd.getpwuid(self.uid).pw_gid
        except (ImportError, KeyError) as exc:
            raise RuntimeError(
                "native sandbox could not derive a gid for the restricted uid; "
                "set AGENT_RT_SANDBOX_GID"
            ) from exc


class DockerSandboxBackend:
    """Runs each command in an ephemeral, capability-dropped Docker container."""

    def __init__(
        self,
        image: str,
        *,
        docker_binary: str = "docker",
        workspace_root: str | None = None,
        user: str | None = None,
        read_only_root: bool = False,
    ) -> None:
        if not image.strip():
            raise ValueError("docker sandbox image must not be empty")
        if not docker_binary.strip():
            raise ValueError("docker sandbox binary must not be empty")
        self.image = image
        self.docker_binary = docker_binary
        self.workspace_root = (
            os.path.abspath(workspace_root) if workspace_root else None
        )
        if self.workspace_root is not None and any(
            character in self.workspace_root for character in ",\n\r\x00"
        ):
            # ``--mount`` options are comma separated; a comma would let the
            # path inject extra mount options.
            raise ValueError(
                "docker sandbox workspace_root must not contain commas or newlines"
            )
        if user is not None and not user.strip():
            raise ValueError("docker sandbox user must not be empty when provided")
        self.user = user.strip() if user else None
        self.read_only_root = read_only_root

    @staticmethod
    def _container_workdir(cwd: str) -> str:
        normalized = posixpath.normpath(cwd.replace("\\", "/"))
        if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise ValueError("sandbox working directory must stay inside /workspace")
        return "/workspace" if normalized == "." else f"/workspace/{normalized}"

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        del session_id, workspace
        if network_policy.mode != "none" and _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "docker backend cannot enforce HTTPS/domain-only, websocket/IP/proxy, allow/block-list, or bandwidth/transfer network controls"
            )
        if limits.cpu_seconds is not None or limits.disk_bytes is not None:
            raise RuntimeError("docker backend cannot enforce cpu_seconds/disk_bytes limits")
        container = f"agent-rt-{uuid.uuid4().hex[:16]}"
        argv = [
            self.docker_binary,
            "run",
            "--rm",
            "-i",
            "--name",
            container,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if self.user is not None:
            argv += ["--user", self.user]
        if self.read_only_root:
            argv += ["--read-only", "--tmpfs", "/tmp"]  # nosec B108 - container-internal tmpfs mount, not a host temp path
        if network_policy.mode == "none":
            argv += ["--network", "none"]
        if limits.memory_bytes is not None:
            argv += [
                "--memory",
                str(limits.memory_bytes),
                "--memory-swap",
                str(limits.memory_bytes),
            ]
        if limits.process_count is not None:
            argv += ["--pids-limit", str(limits.process_count)]
        if self.workspace_root is not None:
            argv += [
                "--mount",
                f"type=bind,src={self.workspace_root},dst=/workspace",
                "--workdir",
                self._container_workdir(command.cwd or "."),
            ]
        elif command.cwd:
            argv += ["--workdir", command.cwd]
        effective_environment = _sanitize_sandbox_environment(environment, network_policy)
        if network_policy.proxy_url:
            effective_environment.setdefault("HTTPS_PROXY", network_policy.proxy_url)
        for key, value in effective_environment.items():
            argv += ["--env", f"{key}={value}"]
        argv.append(self.image)
        argv.extend(command.argv)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def kill_container() -> None:
            # Killing the docker client does not stop the container.
            with suppress(ProcessLookupError):
                process.kill()
            killer = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "kill",
                container,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()

        return await _collect_process_result(
            process,
            command.stdin,
            output_limit=(
                limits.output_bytes
                if limits.output_bytes is not None
                else DEFAULT_SANDBOX_OUTPUT_LIMIT_BYTES
            ),
            kill=kill_container,
            started=started,
        )


class E2BSandboxBackend:
    """Persistent E2B cloud sandboxes, keyed by harness session id."""

    def __init__(self, *, template: str | None = None) -> None:
        self.template = template.strip() if template and template.strip() else None
        self._sandboxes: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _load_sdk(self) -> Any:
        try:
            return importlib.import_module("e2b")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "E2B sandbox support requires the optional 'e2b' package; "
                "install agent-rt[sandbox-e2b]"
            ) from exc

    async def _sandbox(self, session_id: str) -> Any:
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            return sandbox
        # One creation per session: concurrent first commands must not each
        # start (and then leak) a billable cloud sandbox.
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            sandbox = self._sandboxes.get(session_id)
            if sandbox is not None:
                return sandbox
            sdk = self._load_sdk()
            create = sdk.Sandbox.create
            if self.template:
                sandbox = await asyncio.to_thread(create, self.template)
            else:
                sandbox = await asyncio.to_thread(create)
            self._sandboxes[session_id] = sandbox
            return sandbox

    async def close_session(self, session_id: str) -> bool:
        """Terminate the cloud sandbox owned by ``session_id`` (best effort)."""
        sandbox = self._sandboxes.pop(session_id, None)
        self._locks.pop(session_id, None)
        if sandbox is None:
            return False
        kill = getattr(sandbox, "kill", None)
        if callable(kill):
            await asyncio.to_thread(kill)
        return True

    async def close(self) -> None:
        for session_id in tuple(self._sandboxes):
            await self.close_session(session_id)

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        del workspace
        if (
            network_policy.mode != "unrestricted"
            or network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "E2B backend requires unrestricted network policy; "
                "custom domain/proxy enforcement belongs in the E2B template"
            )
        if any(
            value is not None
            for value in (
                limits.cpu_seconds,
                limits.memory_bytes,
                limits.disk_bytes,
                limits.process_count,
            )
        ):
            raise RuntimeError("E2B backend does not enforce per-command resource limits")
        if command.stdin is not None:
            raise RuntimeError("E2B backend does not support command stdin")
        sandbox = await self._sandbox(session_id)
        started = time.monotonic()
        result = await asyncio.to_thread(
            sandbox.commands.run,
            shlex.join(command.argv),
            envs=dict(environment),
            cwd=command.cwd or None,
            timeout=60 if limits.timeout_seconds is None else limits.timeout_seconds,
        )
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout).encode(),
            stderr=str(result.stderr).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )


class MicrosandboxBackend:
    """Persistent microsandbox microVMs, keyed by harness session id."""

    def __init__(self, image: str) -> None:
        if not image.strip():
            raise ValueError("microsandbox image must not be empty")
        self.image = image.strip()
        self._sandboxes: dict[str, Any] = {}
        self._names: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Network mode each microVM was created with. A microVM's network is
        # fixed at creation, so a later policy change cannot be applied.
        self._network_modes: dict[str, str] = {}

    def _load_sdk(self) -> Any:
        try:
            return importlib.import_module("microsandbox")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "microsandbox support requires the optional 'microsandbox' package; "
                "install agent-rt[sandbox-microsandbox]"
            ) from exc

    async def _sandbox(
        self,
        session_id: str,
        network_policy: SandboxNetworkPolicy,
    ) -> Any:
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            self._assert_network_unchanged(session_id, network_policy)
            return sandbox
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            sandbox = self._sandboxes.get(session_id)
            if sandbox is not None:
                self._assert_network_unchanged(session_id, network_policy)
                return sandbox
            return await self._create_sandbox(session_id, network_policy)

    def _assert_network_unchanged(
        self, session_id: str, network_policy: SandboxNetworkPolicy
    ) -> None:
        created_with = self._network_modes.get(session_id)
        if created_with is not None and created_with != network_policy.mode:
            raise RuntimeError(
                "microsandbox cannot change a running microVM's network policy "
                f"(created with {created_with!r}, now {network_policy.mode!r}); "
                "close the session and start a new one"
            )

    async def close_session(self, session_id: str) -> bool:
        """Stop the microVM owned by ``session_id`` (best effort)."""
        sandbox = self._sandboxes.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._names.pop(session_id, None)
        self._network_modes.pop(session_id, None)
        if sandbox is None:
            return False
        for method_name in ("stop", "kill", "close"):
            method = getattr(sandbox, method_name, None)
            if callable(method):
                outcome = method()
                if hasattr(outcome, "__await__"):
                    await outcome
                break
        return True

    async def close(self) -> None:
        for session_id in tuple(self._sandboxes):
            await self.close_session(session_id)

    async def _create_sandbox(
        self,
        session_id: str,
        network_policy: SandboxNetworkPolicy,
    ) -> Any:
        if network_policy.mode != "none" and _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "microsandbox backend cannot enforce the requested advanced network controls"
            )
        if (
            network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "microsandbox backend currently supports only whole-sandbox none/unrestricted network policies"
            )
        sdk = self._load_sdk()
        if network_policy.mode == "none":
            network = sdk.Network.none()
        elif network_policy.mode == "unrestricted":
            network = sdk.Network.allow_all()
        else:
            raise RuntimeError(
                "microsandbox backend does not yet map harness domain allowlists"
            )
        name = self._names.setdefault(
            session_id,
            f"agent-rt-{uuid.uuid4().hex[:20]}",
        )
        sandbox = await sdk.Sandbox.create(name, image=self.image, network=network)
        self._sandboxes[session_id] = sandbox
        self._network_modes[session_id] = network_policy.mode
        return sandbox

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        del workspace
        if any(
            value is not None
            for value in (
                limits.cpu_seconds,
                limits.memory_bytes,
                limits.disk_bytes,
                limits.process_count,
            )
        ):
            raise RuntimeError(
                "microsandbox backend does not map harness per-command "
                "cpu/memory/disk/process limits"
            )
        if command.stdin is not None:
            raise RuntimeError("microsandbox backend does not support command stdin")
        sandbox = await self._sandbox(session_id, network_policy)
        started = time.monotonic()
        result = await sandbox.exec(
            command.argv[0],
            list(command.argv[1:]),
            cwd=command.cwd or None,
            env=dict(environment) or None,
            timeout=limits.timeout_seconds,
        )
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout_text).encode(),
            stderr=str(result.stderr_text).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )


class SWEReXSandboxBackend:
    """SWE-ReX remote runtime adapter.

    The endpoint must already be backed by an isolated SWE-ReX deployment. Because
    the remote execution API cannot change sandbox networking or resource limits
    per command, unsupported policies fail closed here.
    """

    def __init__(self, url: str, *, api_key: str | None = None) -> None:
        normalized = url.strip().rstrip("/")
        if not normalized:
            raise ValueError("SWE-ReX URL must not be empty")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("SWE-ReX URL must start with http:// or https://")
        self.url = normalized
        self.api_key = api_key or None
        self._runtime: Any | None = None

    def _load_runtime(self) -> tuple[Any, Any]:
        try:
            remote = importlib.import_module("swerex.runtime.remote")
            abstract = importlib.import_module("swerex.runtime.abstract")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SWE-ReX support requires the optional 'swe-rex' package; "
                "install agent-rt[sandbox-swe-rex]"
            ) from exc
        return remote.RemoteRuntime, abstract.Command

    def _get_runtime(self) -> tuple[Any, Any]:
        RemoteRuntime, Command = self._load_runtime()
        if self._runtime is None:
            self._runtime = RemoteRuntime(
                host=self.url,
                port=None,
                auth_token=self.api_key,
                timeout=30.0,
            )
        return self._runtime, Command

    async def execute(
        self,
        session_id: str,
        command: SandboxCommand,
        *,
        workspace: WorkspaceFiles,
        limits: SandboxResourceLimits,
        environment: Mapping[str, str],
        network_policy: SandboxNetworkPolicy,
    ) -> SandboxCommandResult:
        del session_id, workspace
        if network_policy.mode != "unrestricted":
            raise RuntimeError(
                "SWE-ReX remote backend cannot enforce harness network isolation; use a preconfigured isolated deployment and request unrestricted mode"
            )
        if _requires_advanced_network_enforcement(network_policy):
            raise RuntimeError(
                "SWE-ReX remote backend cannot enforce the requested advanced network controls"
            )
        if (
            network_policy.allowed_domains
            or network_policy.blocked_domains
            or network_policy.proxy_url
        ):
            raise RuntimeError(
                "SWE-ReX remote backend does not implement domain/proxy network policy"
            )
        if any(
            value is not None
            for value in (
                limits.cpu_seconds,
                limits.memory_bytes,
                limits.disk_bytes,
                limits.process_count,
            )
        ):
            raise RuntimeError(
                "SWE-ReX remote backend does not enforce per-command "
                "cpu/memory/disk/process limits"
            )
        if command.stdin is not None:
            raise RuntimeError("SWE-ReX remote backend does not support command stdin")
        runtime, Command = self._get_runtime()
        started = time.monotonic()
        result = await runtime.execute(
            Command(
                command=list(command.argv),
                timeout=limits.timeout_seconds,
                shell=False,
                check=False,
                env=dict(environment) or None,
                cwd=command.cwd or None,
            )
        )
        if result.exit_code is None:
            raise RuntimeError("SWE-ReX returned no exit code")
        return SandboxCommandResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout).encode(),
            stderr=str(result.stderr).encode(),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )


def sandbox_backend_from_env(
    environment: Mapping[str, str] | None = None,
) -> SandboxBackend:
    env = os.environ if environment is None else environment
    if "AGENT_RT_DISABLE_SANDBOX" in env:
        raise ValueError(
            "AGENT_RT_DISABLE_SANDBOX is not supported; sandboxing cannot be disabled"
        )
    backend_name = _normalized_sandbox_backend_name(env.get(SANDBOX_BACKEND_ENV))
    if backend_name == "native":
        raw_uid = env.get("AGENT_RT_SANDBOX_UID")
        if raw_uid is None or not raw_uid.strip():
            raise ValueError(
                "native sandbox requires AGENT_RT_SANDBOX_UID for a restricted user"
            )
        try:
            uid = int(raw_uid)
            gid = (
                int(env["AGENT_RT_SANDBOX_GID"])
                if env.get("AGENT_RT_SANDBOX_GID", "").strip()
                else None
            )
        except ValueError as exc:
            raise ValueError("native sandbox uid/gid must be integers") from exc
        return NativeSandboxBackend(
            uid=uid,
            gid=gid,
            root=env.get("AGENT_RT_SANDBOX_NATIVE_ROOT"),
        )
    if backend_name == "docker":
        image = env.get("AGENT_RT_SANDBOX_DOCKER_IMAGE", "").strip()
        if not image:
            raise ValueError(
                "docker sandbox requires AGENT_RT_SANDBOX_DOCKER_IMAGE"
            )
        return DockerSandboxBackend(
            image,
            docker_binary=env.get("AGENT_RT_SANDBOX_DOCKER_BINARY", "docker"),
            workspace_root=env.get("AGENT_RT_SANDBOX_WORKSPACE_ROOT"),
            user=env.get("AGENT_RT_SANDBOX_DOCKER_USER"),
            read_only_root=_sandbox_env_bool(env, "AGENT_RT_SANDBOX_DOCKER_READ_ONLY"),
        )
    if backend_name == "e2b":
        return E2BSandboxBackend(template=env.get("AGENT_RT_SANDBOX_E2B_TEMPLATE"))
    if backend_name == "microsandbox":
        image = env.get("AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE", "").strip()
        if not image:
            raise ValueError(
                "microsandbox backend requires AGENT_RT_SANDBOX_MICROSANDBOX_IMAGE"
            )
        return MicrosandboxBackend(image)
    url = env.get("AGENT_RT_SANDBOX_SWEREX_URL", "").strip()
    if not url:
        raise ValueError("SWE-ReX backend requires AGENT_RT_SANDBOX_SWEREX_URL")
    return SWEReXSandboxBackend(
        url,
        api_key=env.get("AGENT_RT_SANDBOX_SWEREX_API_KEY"),
    )


@runtime_checkable
class SandboxPackageManager(Protocol):
    async def install(
        self,
        session_id: str,
        runtime: str,
        packages: Sequence[str],
        *,
        workspace: WorkspaceFiles,
        environment: Mapping[str, str],
        limits: SandboxResourceLimits,
        network_policy: SandboxNetworkPolicy,
    ) -> Sequence[str]: ...

SandboxPackageInstaller = Callable[
    [
        str,
        str,
        Sequence[str],
        WorkspaceFiles,
        Mapping[str, str],
        SandboxResourceLimits,
        SandboxNetworkPolicy,
    ],
    Awaitable[Sequence[str]],
]

class CallbackSandboxPackageManager:
    def __init__(self, installer: SandboxPackageInstaller) -> None:
        self.installer = installer

    async def install(
        self,
        session_id: str,
        runtime: str,
        packages: Sequence[str],
        *,
        workspace: WorkspaceFiles,
        environment: Mapping[str, str],
        limits: SandboxResourceLimits,
        network_policy: SandboxNetworkPolicy,
    ) -> Sequence[str]:
        return await self.installer(
            session_id,
            runtime,
            packages,
            workspace,
            environment,
            limits,
            network_policy,
        )

CodeInterpreterRunner = Callable[
    [
        str,
        str,
        str,
        dict[str, JSONValue],
        WorkspaceFiles,
        Mapping[str, str],
        SandboxResourceLimits,
        SandboxNetworkPolicy,
    ],
    Awaitable[SandboxCommandResult],
]

class CodeInterpreterRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, CodeInterpreterRunner] = {}

    def register(
        self,
        runtime: str,
        runner: CodeInterpreterRunner,
        *,
        replace: bool = False,
    ) -> None:
        if not runtime.strip():
            raise ValueError("interpreter runtime must not be empty")
        if runtime in self._runners and not replace:
            raise ValueError(f"interpreter runtime already registered: {runtime}")
        self._runners[runtime] = runner

    def get(self, runtime: str) -> CodeInterpreterRunner:
        try:
            return self._runners[runtime]
        except KeyError as exc:
            raise KeyError(f"interpreter runtime not registered: {runtime}") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._runners))

class SandboxSession:
    """A sandbox session: backend, workspace files, limits, and network policy.

    Snapshots, restore, and clones capture only the in-process ``workspace``
    files and session settings. They do not capture the disk of a native,
    Docker, or cloud backend, so state written by commands inside the sandbox
    is outside their scope.
    """

    def __init__(
        self,
        session_id: str,
        backend: SandboxBackend,
        *,
        workspace: WorkspaceFiles | None = None,
        limits: SandboxResourceLimits | None = None,
        interpreters: CodeInterpreterRegistry | None = None,
        package_manager: SandboxPackageManager | None = None,
        network_policy: SandboxNetworkPolicy | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("sandbox session_id must not be empty")
        self.session_id = session_id
        self.backend = backend
        self.workspace = workspace or WorkspaceFiles(InMemoryFileSystem())
        defaults = sandbox_resource_limits_from_env()
        self.limits = defaults if limits is None else SandboxResourceLimits(
            cpu_seconds=(limits.cpu_seconds if limits.cpu_seconds is not None else defaults.cpu_seconds),
            memory_bytes=(limits.memory_bytes if limits.memory_bytes is not None else defaults.memory_bytes),
            disk_bytes=limits.disk_bytes,
            process_count=limits.process_count,
            timeout_seconds=(limits.timeout_seconds if limits.timeout_seconds is not None else defaults.timeout_seconds),
            output_bytes=limits.output_bytes,
        )
        self.interpreters = interpreters or CodeInterpreterRegistry()
        self.package_manager = package_manager
        self.network_policy = network_policy or sandbox_network_policy_from_env()
        self._snapshots: dict[str, SandboxSnapshot] = {}
        self.environment: dict[str, str] = {}
        self.cwd = ""
        self.runtime_versions: dict[str, str] = {}
        self.installed_packages: dict[str, tuple[str, ...]] = {}
        self.interpreter_state: dict[str, dict[str, JSONValue]] = {}

    async def close(self) -> None:
        """Release backend resources (cloud sandboxes, microVMs) for this session."""
        close_session = getattr(self.backend, "close_session", None)
        if callable(close_session):
            await close_session(self.session_id)

    def set_environment(self, values: Mapping[str, str]) -> None:
        for key, value in values.items():
            if not key or "\x00" in key or "\x00" in value:
                raise ValueError("sandbox environment contains an invalid entry")
            self.environment[key] = value

    def unset_environment(self, *keys: str) -> None:
        for key in keys:
            self.environment.pop(key, None)

    def set_working_directory(self, path: str) -> None:
        normalized = InMemoryFileSystem._normalize(path)
        self.cwd = normalized

    def set_runtime_version(self, runtime: str, version: str) -> None:
        if not runtime.strip() or not version.strip():
            raise ValueError("runtime and version must not be empty")
        self.runtime_versions[runtime] = version

    def runtime_version(self, runtime: str) -> str | None:
        return self.runtime_versions.get(runtime)

    def set_network_policy(self, policy: SandboxNetworkPolicy) -> None:
        self.network_policy = policy

    def network_allows(self, target: str) -> bool:
        return self.network_policy.allows(target)

    def _snapshot_files(self) -> dict[str, bytes]:
        files: dict[str, bytes] = {}

        def walk(path: str = "") -> None:
            for item in self.workspace.list(path):
                if item.is_directory:
                    walk(item.path)
                else:
                    files[item.path] = self.workspace.filesystem.read(item.path)

        walk()
        return files

    def create_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        snapshot = SandboxSnapshot(
            snapshot_id=snapshot_id,
            source_session_id=self.session_id,
            files=self._snapshot_files(),
            cwd=self.cwd,
            environment=dict(self.environment),
            runtime_versions=dict(self.runtime_versions),
            installed_packages={
                runtime: tuple(packages)
                for runtime, packages in self.installed_packages.items()
            },
            interpreter_state={
                runtime: dict(state)
                for runtime, state in self.interpreter_state.items()
            },
            network_policy=self.network_policy,
        )
        if snapshot_id in self._snapshots:
            raise ValueError(f"sandbox snapshot already exists: {snapshot_id}")
        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"sandbox snapshot not found: {snapshot_id}") from exc

    def restore_snapshot(self, snapshot_id: str) -> SandboxSnapshot:
        snapshot = self.get_snapshot(snapshot_id)
        self.workspace.clear()
        for path, data in snapshot.files.items():
            self.workspace.filesystem.write(path, data)
        self.cwd = snapshot.cwd
        self.environment.clear()
        self.environment.update(snapshot.environment)
        self.runtime_versions.clear()
        self.runtime_versions.update(snapshot.runtime_versions)
        self.installed_packages.clear()
        self.installed_packages.update(
            {
                runtime: tuple(packages)
                for runtime, packages in snapshot.installed_packages.items()
            }
        )
        self.interpreter_state.clear()
        self.interpreter_state.update(
            {
                runtime: dict(state)
                for runtime, state in snapshot.interpreter_state.items()
            }
        )
        self.network_policy = snapshot.network_policy
        return snapshot

    def clone_from_snapshot(
        self,
        snapshot_id: str,
        new_session_id: str,
    ) -> "SandboxSession":
        snapshot = self.get_snapshot(snapshot_id)
        clone = SandboxSession(
            new_session_id,
            self.backend,
            limits=self.limits,
            interpreters=self.interpreters,
            package_manager=self.package_manager,
            network_policy=snapshot.network_policy,
        )
        clone._snapshots[snapshot_id] = snapshot
        clone.restore_snapshot(snapshot_id)
        return clone

    def install_packages(self, runtime: str, packages: Sequence[str]) -> tuple[str, ...]:
        if not runtime.strip():
            raise ValueError("runtime must not be empty")
        normalized = tuple(dict.fromkeys(item.strip() for item in packages if item.strip()))
        self.installed_packages[runtime] = normalized
        return normalized

    async def install_runtime_packages(
        self,
        runtime: str,
        packages: Sequence[str],
    ) -> tuple[str, ...]:
        if self.package_manager is None:
            raise RuntimeError("sandbox package manager is not configured")
        normalized = tuple(dict.fromkeys(item.strip() for item in packages if item.strip()))
        installed = await self.package_manager.install(
            self.session_id,
            runtime,
            normalized,
            workspace=self.workspace,
            environment=dict(self.environment),
            limits=self.limits,
            network_policy=self.network_policy,
        )
        recorded = tuple(dict.fromkeys(item.strip() for item in installed if item.strip()))
        self.installed_packages[runtime] = recorded
        return recorded

    def runtime_packages(self, runtime: str) -> tuple[str, ...]:
        return self.installed_packages.get(runtime, ())

    async def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        effective = SandboxCommand(
            argv=command.argv,
            cwd=command.cwd or self.cwd,
            env=command.env,
            stdin=command.stdin,
        )
        environment = dict(self.environment)
        environment.update(command.env)
        environment = _sanitize_sandbox_environment(environment, self.network_policy)
        operation = self.backend.execute(
            self.session_id,
            effective,
            workspace=self.workspace,
            limits=self.limits,
            environment=environment,
            network_policy=self.network_policy,
        )
        if self.limits.timeout_seconds is None:
            result = await operation
        else:
            try:
                result = await asyncio.wait_for(
                    operation,
                    timeout=self.limits.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError("sandbox command exceeded execution timeout") from exc
        return self._bounded_result(result)

    async def run_code(
        self,
        runtime: str,
        code: str,
    ) -> SandboxCommandResult:
        runner = self.interpreters.get(runtime)
        state = self.interpreter_state.setdefault(runtime, {})
        operation = runner(
            self.session_id,
            runtime,
            code,
            state,
            self.workspace,
            dict(self.environment),
            self.limits,
            self.network_policy,
        )
        if self.limits.timeout_seconds is None:
            result = await operation
        else:
            try:
                result = await asyncio.wait_for(
                    operation,
                    timeout=self.limits.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError("sandbox interpreter exceeded execution timeout") from exc
        return self._bounded_result(result)

    def _bounded_result(self, result: SandboxCommandResult) -> SandboxCommandResult:
        limit = self.limits.output_bytes
        if limit is None:
            return result
        remaining = limit
        stdout = result.stdout[:remaining]
        remaining -= len(stdout)
        stderr = result.stderr[:remaining]
        truncated = result.truncated or len(stdout) < len(result.stdout) or len(stderr) < len(result.stderr)
        return SandboxCommandResult(
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=result.duration_ms,
            truncated=truncated,
        )

def sandbox_shell_tool(
    session: SandboxSession,
) -> ToolHandler:
    async def handler(
        arguments: Mapping[str, Any],
        _cancellation_token: CancellationToken | None,
    ) -> Any:
        argv = arguments.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            raise ValueError("shell tool requires argv array")
        command = SandboxCommand(
            argv=tuple(str(item) for item in argv),
            cwd=str(arguments.get("cwd", "")),
            env={
                str(key): str(value)
                for key, value in dict(arguments.get("env", {})).items()
            },
        )
        result = await session.execute(command)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "duration_ms": result.duration_ms,
            "truncated": result.truncated,
        }

    return handler

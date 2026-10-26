"""Plans, plan tracking, verification, and planner execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable

PlanStepStatus = Literal[
    "pending", "ready", "in_progress", "completed", "blocked", "failed"
]


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

    def complete(
        self, step_id: str, result: Any = None, *, note: str | None = None
    ) -> PlanStep:
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
            step.id: step for step in self.plan.steps if step.status == "completed"
        }
        merged = tuple(completed.get(step.id, step) for step in steps)
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

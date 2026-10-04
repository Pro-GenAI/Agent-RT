from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    input: Any
    expected: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("evaluation case id must not be empty")


@dataclass(frozen=True)
class EvaluationToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSample:
    case: EvaluationCase
    output: Any
    trajectory: tuple[Any, ...] = ()
    tool_calls: tuple[EvaluationToolCall, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationGrade:
    grader: str
    score: float
    passed: bool
    details: str = ""

    def __post_init__(self) -> None:
        if not self.grader.strip():
            raise ValueError("grader name must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("evaluation score must be between 0 and 1")


@runtime_checkable
class EvaluationGrader(Protocol):
    name: str

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade: ...


EvaluationExecutor = Callable[[EvaluationCase], Awaitable[EvaluationSample | Any]]


@dataclass(frozen=True)
class EvaluationCaseResult:
    sample: EvaluationSample
    grades: tuple[EvaluationGrade, ...]

    @property
    def passed(self) -> bool:
        return all(grade.passed for grade in self.grades)

    @property
    def score(self) -> float:
        if not self.grades:
            return 1.0
        return sum(grade.score for grade in self.grades) / len(self.grades)


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[EvaluationCaseResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.case_count if self.case_count else 1.0

    @property
    def mean_score(self) -> float:
        return (
            sum(result.score for result in self.results) / self.case_count
            if self.case_count
            else 1.0
        )


class EvaluationRunner:
    def __init__(self, executor: EvaluationExecutor, graders: Sequence[EvaluationGrader]) -> None:
        self.executor = executor
        self.graders = tuple(graders)

    async def run(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []
        for case in cases:
            executed = await self.executor(case)
            sample = (
                executed
                if isinstance(executed, EvaluationSample)
                else EvaluationSample(case=case, output=executed)
            )
            if sample.case.id != case.id:
                raise ValueError("evaluation executor returned a sample for a different case")
            grades = tuple([await grader.grade(sample) for grader in self.graders])
            results.append(EvaluationCaseResult(sample=sample, grades=grades))
        return EvaluationReport(tuple(results))


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> tuple[str, ...]:
    issues: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str) and not _matches_type(value, expected):
        return (f"{path}: expected {expected}",)

    enum_values = schema.get("enum")
    if isinstance(enum_values, Sequence) and not isinstance(enum_values, (str, bytes)):
        if value not in enum_values:
            issues.append(f"{path}: value is not in enum")

    if isinstance(value, dict):
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    issues.append(f"{path}.{key}: required property is missing")

        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    issues.extend(validate_schema(value[key], child_schema, f"{path}.{key}"))
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        issues.append(f"{path}.{key}: additional property is not allowed")

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                issues.extend(validate_schema(item, item_schema, f"{path}[{index}]"))

    return tuple(issues)


class ExactMatchGrader:
    def __init__(self, *, name: str = "exact_match") -> None:
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        passed = sample.output == sample.case.expected
        return EvaluationGrade(
            self.name,
            1.0 if passed else 0.0,
            passed,
            "output matched expected value" if passed else "output did not match expected value",
        )


class SchemaGrader:
    def __init__(self, schema: Mapping[str, Any], *, name: str = "schema") -> None:
        self.schema = dict(schema)
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        issues = validate_schema(sample.output, self.schema)
        return EvaluationGrade(
            self.name,
            1.0 if not issues else 0.0,
            not issues,
            "; ".join(issues),
        )


class AssertionGrader:
    def __init__(
        self,
        assertion: Callable[[EvaluationSample], bool | tuple[bool, str]],
        *,
        name: str = "assertion",
    ) -> None:
        self.assertion = assertion
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        result = self.assertion(sample)
        passed, details = result if isinstance(result, tuple) else (bool(result), "")
        return EvaluationGrade(self.name, 1.0 if passed else 0.0, passed, details)


class ToolUseGrader:
    def __init__(
        self,
        expected_tools: Sequence[str],
        *,
        allow_extra: bool = False,
        require_order: bool = False,
        argument_assertions: Mapping[str, Callable[[Mapping[str, Any]], bool]] | None = None,
        name: str = "tool_use",
    ) -> None:
        self.expected_tools = tuple(expected_tools)
        self.allow_extra = allow_extra
        self.require_order = require_order
        self.argument_assertions = dict(argument_assertions or {})
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        actual = tuple(call.name for call in sample.tool_calls)
        if self.require_order:
            sequence_ok = (
                actual == self.expected_tools
                if not self.allow_extra
                else actual[: len(self.expected_tools)] == self.expected_tools
            )
        else:
            expected_set = set(self.expected_tools)
            actual_set = set(actual)
            sequence_ok = expected_set.issubset(actual_set) and (
                self.allow_extra or actual_set == expected_set
            )

        argument_failures = [
            call.name
            for call in sample.tool_calls
            if call.name in self.argument_assertions
            and not self.argument_assertions[call.name](call.arguments)
        ]
        passed = sequence_ok and not argument_failures
        details = f"actual={list(actual)}"
        if argument_failures:
            details += f"; invalid_arguments={argument_failures}"
        return EvaluationGrade(self.name, 1.0 if passed else 0.0, passed, details)


class TrajectoryGrader:
    def __init__(
        self,
        *,
        required: Sequence[Any] = (),
        forbidden: Sequence[Any] = (),
        require_order: bool = False,
        name: str = "trajectory",
    ) -> None:
        self.required = tuple(required)
        self.forbidden = tuple(forbidden)
        self.require_order = require_order
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        trajectory = sample.trajectory
        forbidden_hits = [item for item in self.forbidden if item in trajectory]
        if self.require_order:
            position = 0
            for event in trajectory:
                if position < len(self.required) and event == self.required[position]:
                    position += 1
            required_ok = position == len(self.required)
        else:
            required_ok = all(item in trajectory for item in self.required)
        passed = required_ok and not forbidden_hits
        details = (
            "trajectory satisfied requirements"
            if passed
            else f"required_ok={required_ok}; forbidden_hits={forbidden_hits}"
        )
        return EvaluationGrade(self.name, 1.0 if passed else 0.0, passed, details)


JudgeCallable = Callable[
    [EvaluationSample, str],
    Awaitable[float | EvaluationGrade | Mapping[str, Any]],
]


class LLMJudgeGrader:
    def __init__(
        self,
        judge: JudgeCallable,
        rubric: str,
        *,
        threshold: float = 0.5,
        name: str = "llm_judge",
    ) -> None:
        if not rubric.strip():
            raise ValueError("judge rubric must not be empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("judge threshold must be between 0 and 1")
        self.judge = judge
        self.rubric = rubric
        self.threshold = threshold
        self.name = name

    async def grade(self, sample: EvaluationSample) -> EvaluationGrade:
        judged = await self.judge(sample, self.rubric)
        if isinstance(judged, EvaluationGrade):
            return judged
        details = ""
        if isinstance(judged, Mapping):
            score = float(judged.get("score", 0.0))
            details = str(judged.get("details", ""))
        else:
            score = float(judged)
        if score != score:  # NaN must fail, not clamp to a passing 1.0
            score = 0.0
        score = max(0.0, min(1.0, score))
        return EvaluationGrade(self.name, score, score >= self.threshold, details)


SAFETY_EVALUATION_CATEGORIES = (
    "prompt_injection",
    "permission_boundary",
    "secret_handling",
    "unsafe_tool_use",
    "data_exfiltration",
)


@dataclass(frozen=True)
class SafetyEvaluationSuite:
    cases: tuple[EvaluationCase, ...]

    def __post_init__(self) -> None:
        for case in self.cases:
            category = case.metadata.get("safety_category")
            if category not in SAFETY_EVALUATION_CATEGORIES:
                raise ValueError(
                    f"evaluation case {case.id} has unsupported safety_category {category!r}"
                )

    async def run(self, runner: EvaluationRunner) -> EvaluationReport:
        return await runner.run(self.cases)


@dataclass(frozen=True)
class GoldenRegressionDataset:
    name: str
    version: str
    cases: tuple[EvaluationCase, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("golden dataset name must not be empty")
        if not self.version.strip():
            raise ValueError("golden dataset version must not be empty")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("golden dataset case ids must be unique")

    def get(self, case_id: str) -> EvaluationCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(case_id)


@dataclass(frozen=True)
class ProductionFeedbackRecord:
    id: str
    input: Any
    observed_output: Any
    expected_output: Any = None
    source: str = "production"
    reviewed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_evaluation_case(self) -> EvaluationCase:
        if not self.reviewed:
            raise ValueError("production feedback must be reviewed before evaluation conversion")
        metadata = {
            **dict(self.metadata),
            "source": self.source,
            "observed_output": self.observed_output,
        }
        return EvaluationCase(
            id=self.id,
            input=self.input,
            expected=self.expected_output,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    executor: EvaluationExecutor

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment variant name must not be empty")


@dataclass(frozen=True)
class ExperimentResult:
    reports: Mapping[str, EvaluationReport]
    baseline: str

    def score_delta(self, variant: str) -> float:
        if self.baseline not in self.reports:
            raise KeyError(self.baseline)
        if variant not in self.reports:
            raise KeyError(variant)
        return self.reports[variant].mean_score - self.reports[self.baseline].mean_score

    def pass_rate_delta(self, variant: str) -> float:
        if self.baseline not in self.reports:
            raise KeyError(self.baseline)
        if variant not in self.reports:
            raise KeyError(variant)
        return self.reports[variant].pass_rate - self.reports[self.baseline].pass_rate


class ABExperimentRunner:
    def __init__(self, graders: Sequence[EvaluationGrader]) -> None:
        self.graders = tuple(graders)

    async def run(
        self,
        cases: Sequence[EvaluationCase],
        variants: Sequence[ExperimentVariant],
        *,
        baseline: str,
    ) -> ExperimentResult:
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise ValueError("experiment variant names must be unique")
        if baseline not in names:
            raise ValueError("baseline must name one of the experiment variants")

        reports: dict[str, EvaluationReport] = {}
        for variant in variants:
            reports[variant.name] = await EvaluationRunner(
                variant.executor,
                self.graders,
            ).run(cases)
        return ExperimentResult(reports=reports, baseline=baseline)

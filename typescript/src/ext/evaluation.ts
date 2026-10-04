export interface EvaluationCase {
  id: string;
  input: unknown;
  expected?: unknown;
  metadata?: Record<string, unknown>;
}

export interface EvaluationToolCall {
  name: string;
  arguments?: Record<string, unknown>;
}

export interface EvaluationSample {
  case: EvaluationCase;
  output: unknown;
  trajectory?: unknown[];
  toolCalls?: EvaluationToolCall[];
  metadata?: Record<string, unknown>;
}

export interface EvaluationGrade {
  grader: string;
  score: number;
  passed: boolean;
  details?: string;
}

export interface EvaluationGrader {
  readonly name: string;
  grade(sample: EvaluationSample): Promise<EvaluationGrade>;
}

export interface EvaluationCaseResult {
  sample: EvaluationSample;
  grades: EvaluationGrade[];
  passed: boolean;
  score: number;
}

export interface EvaluationReport {
  results: EvaluationCaseResult[];
  caseCount: number;
  passedCount: number;
  passRate: number;
  meanScore: number;
}

export type EvaluationExecutor = (evaluationCase: EvaluationCase) => Promise<EvaluationSample | unknown>;

export class EvaluationRunner {
  constructor(readonly executor: EvaluationExecutor, readonly graders: EvaluationGrader[]) {}

  async run(cases: EvaluationCase[]): Promise<EvaluationReport> {
    const results: EvaluationCaseResult[] = [];
    for (const evaluationCase of cases) {
      if (!evaluationCase.id.trim()) throw new Error("evaluation case id must not be empty");
      const executed = await this.executor(evaluationCase);
      const sample: EvaluationSample =
        typeof executed === "object" && executed !== null && "case" in executed && "output" in executed
          ? executed as EvaluationSample
          : { case: evaluationCase, output: executed };
      if (sample.case.id !== evaluationCase.id) {
        throw new Error("evaluation executor returned a sample for a different case");
      }
      const grades = await Promise.all(this.graders.map((grader) => grader.grade(sample)));
      for (const grade of grades) {
        if (!grade.grader.trim()) throw new Error("grader name must not be empty");
        if (grade.score < 0 || grade.score > 1) throw new Error("evaluation score must be between 0 and 1");
      }
      const passed = grades.every((grade) => grade.passed);
      const score = grades.length === 0 ? 1 : grades.reduce((sum, grade) => sum + grade.score, 0) / grades.length;
      results.push({ sample, grades, passed, score });
    }
    const caseCount = results.length;
    const passedCount = results.filter((result) => result.passed).length;
    return {
      results,
      caseCount,
      passedCount,
      passRate: caseCount === 0 ? 1 : passedCount / caseCount,
      meanScore: caseCount === 0 ? 1 : results.reduce((sum, result) => sum + result.score, 0) / caseCount,
    };
  }
}

function matchesType(value: unknown, expected: string): boolean {
  if (expected === "object") return typeof value === "object" && value !== null && !Array.isArray(value);
  if (expected === "array") return Array.isArray(value);
  if (expected === "string") return typeof value === "string";
  if (expected === "boolean") return typeof value === "boolean";
  if (expected === "null") return value === null;
  if (expected === "integer") return typeof value === "number" && Number.isInteger(value);
  if (expected === "number") return typeof value === "number" && Number.isFinite(value);
  return true;
}

export function validateEvaluationSchema(value: unknown, schema: Record<string, unknown>, path = "$"): string[] {
  const issues: string[] = [];
  const expected = schema.type;
  if (typeof expected === "string" && !matchesType(value, expected)) return [path + ": expected " + expected];

  const enumValues = schema.enum;
  if (Array.isArray(enumValues) && !enumValues.some((candidate) => Object.is(candidate, value))) {
    issues.push(path + ": value is not in enum");
  }

  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const objectValue = value as Record<string, unknown>;
    const required = schema.required;
    if (Array.isArray(required)) {
      for (const propertyName of required) {
        if (typeof propertyName === "string" && !(propertyName in objectValue)) {
          issues.push(path + "." + propertyName + ": required property is missing");
        }
      }
    }
    const properties = schema.properties;
    if (typeof properties === "object" && properties !== null && !Array.isArray(properties)) {
      const propertyMap = properties as Record<string, unknown>;
      for (const [key, childSchema] of Object.entries(propertyMap)) {
        if (key in objectValue && typeof childSchema === "object" && childSchema !== null && !Array.isArray(childSchema)) {
          issues.push(...validateEvaluationSchema(objectValue[key], childSchema as Record<string, unknown>, path + "." + key));
        }
      }
      if (schema.additionalProperties === false) {
        for (const key of Object.keys(objectValue)) {
          if (!(key in propertyMap)) issues.push(path + "." + key + ": additional property is not allowed");
        }
      }
    }
  }

  if (Array.isArray(value)) {
    const itemSchema = schema.items;
    if (typeof itemSchema === "object" && itemSchema !== null && !Array.isArray(itemSchema)) {
      value.forEach((item, index) => issues.push(...validateEvaluationSchema(item, itemSchema as Record<string, unknown>, path + "[" + index + "]")));
    }
  }
  return issues;
}

export class ExactMatchGrader implements EvaluationGrader {
  constructor(readonly name = "exact_match") {}
  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const passed = JSON.stringify(sample.output) === JSON.stringify(sample.case.expected);
    return { grader: this.name, score: passed ? 1 : 0, passed, details: passed ? "output matched expected value" : "output did not match expected value" };
  }
}

export class SchemaGrader implements EvaluationGrader {
  constructor(readonly schema: Record<string, unknown>, readonly name = "schema") {}
  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const issues = validateEvaluationSchema(sample.output, this.schema);
    return { grader: this.name, score: issues.length === 0 ? 1 : 0, passed: issues.length === 0, details: issues.join("; ") };
  }
}

export class AssertionGrader implements EvaluationGrader {
  constructor(readonly assertion: (sample: EvaluationSample) => boolean | [boolean, string], readonly name = "assertion") {}
  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const result = this.assertion(sample);
    const pair: [boolean, string] = Array.isArray(result) ? result : [Boolean(result), ""];
    return { grader: this.name, score: pair[0] ? 1 : 0, passed: pair[0], details: pair[1] };
  }
}

export class ToolUseGrader implements EvaluationGrader {
  readonly expectedTools: string[];
  readonly argumentAssertions: Record<string, (argumentsValue: Record<string, unknown>) => boolean>;

  constructor(
    expectedTools: string[],
    readonly allowExtra = false,
    readonly requireOrder = false,
    argumentAssertions: Record<string, (argumentsValue: Record<string, unknown>) => boolean> = {},
    readonly name = "tool_use",
  ) {
    this.expectedTools = [...expectedTools];
    this.argumentAssertions = { ...argumentAssertions };
  }

  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const calls = sample.toolCalls ?? [];
    const actual = calls.map((call) => call.name);
    const expectedSet = new Set(this.expectedTools);
    const actualSet = new Set(actual);
    const sequenceOk = this.requireOrder
      ? (this.allowExtra ? this.expectedTools.every((name, index) => actual[index] === name) : JSON.stringify(actual) === JSON.stringify(this.expectedTools))
      : this.expectedTools.every((name) => actualSet.has(name)) && (this.allowExtra || [...actualSet].every((name) => expectedSet.has(name)));
    const invalidArguments = calls
      .filter((call) => call.name in this.argumentAssertions && !this.argumentAssertions[call.name](call.arguments ?? {}))
      .map((call) => call.name);
    const passed = sequenceOk && invalidArguments.length === 0;
    return { grader: this.name, score: passed ? 1 : 0, passed, details: "actual=" + JSON.stringify(actual) + "; invalidArguments=" + JSON.stringify(invalidArguments) };
  }
}

export class TrajectoryGrader implements EvaluationGrader {
  constructor(readonly required: unknown[] = [], readonly forbidden: unknown[] = [], readonly requireOrder = false, readonly name = "trajectory") {}
  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const trajectory = sample.trajectory ?? [];
    const forbiddenHits = this.forbidden.filter((item) => trajectory.includes(item));
    let requiredOk: boolean;
    if (this.requireOrder) {
      let position = 0;
      for (const event of trajectory) {
        if (position < this.required.length && Object.is(event, this.required[position])) position += 1;
      }
      requiredOk = position === this.required.length;
    } else {
      requiredOk = this.required.every((item) => trajectory.includes(item));
    }
    const passed = requiredOk && forbiddenHits.length === 0;
    return { grader: this.name, score: passed ? 1 : 0, passed, details: passed ? "trajectory satisfied requirements" : "requiredOk=" + requiredOk + "; forbiddenHits=" + JSON.stringify(forbiddenHits) };
  }
}

export type JudgeCallable = (sample: EvaluationSample, rubric: string) => Promise<number | EvaluationGrade | { score: number; details?: string }>;

export class LLMJudgeGrader implements EvaluationGrader {
  constructor(readonly judge: JudgeCallable, readonly rubric: string, readonly threshold = 0.5, readonly name = "llm_judge") {
    if (!rubric.trim()) throw new Error("judge rubric must not be empty");
    if (threshold < 0 || threshold > 1) throw new Error("judge threshold must be between 0 and 1");
  }

  async grade(sample: EvaluationSample): Promise<EvaluationGrade> {
    const judged = await this.judge(sample, this.rubric);
    if (typeof judged === "object" && judged !== null && "grader" in judged) return judged as EvaluationGrade;
    const rawScore = typeof judged === "number" ? judged : judged.score;
    const score = Math.max(0, Math.min(1, rawScore));
    const details = typeof judged === "number" ? "" : judged.details ?? "";
    return { grader: this.name, score, passed: score >= this.threshold, details };
  }
}


export const SAFETY_EVALUATION_CATEGORIES = [
  "prompt_injection",
  "permission_boundary",
  "secret_handling",
  "unsafe_tool_use",
  "data_exfiltration",
] as const;

export class SafetyEvaluationSuite {
  readonly cases: EvaluationCase[];

  constructor(cases: EvaluationCase[]) {
    for (const evaluationCase of cases) {
      const category = evaluationCase.metadata?.safetyCategory;
      if (!SAFETY_EVALUATION_CATEGORIES.includes(category as (typeof SAFETY_EVALUATION_CATEGORIES)[number])) {
        throw new Error("unsupported safety category for evaluation case " + evaluationCase.id);
      }
    }
    this.cases = [...cases];
  }

  async run(runner: EvaluationRunner): Promise<EvaluationReport> {
    return runner.run(this.cases);
  }
}

export class GoldenRegressionDataset {
  readonly cases: EvaluationCase[];

  constructor(
    readonly name: string,
    readonly version: string,
    cases: EvaluationCase[],
    readonly description = "",
  ) {
    if (!name.trim()) throw new Error("golden dataset name must not be empty");
    if (!version.trim()) throw new Error("golden dataset version must not be empty");
    const ids = cases.map((evaluationCase) => evaluationCase.id);
    if (new Set(ids).size !== ids.length) throw new Error("golden dataset case ids must be unique");
    this.cases = [...cases];
  }

  get(caseId: string): EvaluationCase {
    const found = this.cases.find((evaluationCase) => evaluationCase.id === caseId);
    if (!found) throw new Error("unknown evaluation case " + caseId);
    return found;
  }
}

export interface ProductionFeedbackRecord {
  id: string;
  input: unknown;
  observedOutput: unknown;
  expectedOutput?: unknown;
  source?: string;
  reviewed: boolean;
  metadata?: Record<string, unknown>;
}

export function productionFeedbackToEvaluationCase(record: ProductionFeedbackRecord): EvaluationCase {
  if (!record.reviewed) throw new Error("production feedback must be reviewed before evaluation conversion");
  return {
    id: record.id,
    input: record.input,
    expected: record.expectedOutput,
    metadata: {
      ...(record.metadata ?? {}),
      source: record.source ?? "production",
      observedOutput: record.observedOutput,
    },
  };
}

export interface ExperimentVariant {
  name: string;
  executor: EvaluationExecutor;
}

export class ExperimentResult {
  constructor(
    readonly reports: Record<string, EvaluationReport>,
    readonly baseline: string,
  ) {}

  scoreDelta(variant: string): number {
    if (!(this.baseline in this.reports)) throw new Error("unknown baseline " + this.baseline);
    if (!(variant in this.reports)) throw new Error("unknown variant " + variant);
    return this.reports[variant].meanScore - this.reports[this.baseline].meanScore;
  }

  passRateDelta(variant: string): number {
    if (!(this.baseline in this.reports)) throw new Error("unknown baseline " + this.baseline);
    if (!(variant in this.reports)) throw new Error("unknown variant " + variant);
    return this.reports[variant].passRate - this.reports[this.baseline].passRate;
  }
}

export class ABExperimentRunner {
  constructor(readonly graders: EvaluationGrader[]) {}

  async run(
    cases: EvaluationCase[],
    variants: ExperimentVariant[],
    baseline: string,
  ): Promise<ExperimentResult> {
    const names = variants.map((variant) => variant.name);
    if (new Set(names).size !== names.length) throw new Error("experiment variant names must be unique");
    if (!names.includes(baseline)) throw new Error("baseline must name one of the experiment variants");

    const reports: Record<string, EvaluationReport> = {};
    for (const variant of variants) {
      reports[variant.name] = await new EvaluationRunner(
        variant.executor,
        this.graders,
      ).run(cases);
    }
    return new ExperimentResult(reports, baseline);
  }
}

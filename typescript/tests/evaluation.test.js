const test = require("node:test");
const assert = require("node:assert/strict");

const {
  EvaluationRunner,
  ExactMatchGrader,
  SchemaGrader,
  ToolUseGrader,
  TrajectoryGrader,
  LLMJudgeGrader,
} = require("../dist/ext/evaluation.js");

test("evaluation runner aggregates deterministic grades", async () => {
  const runner = new EvaluationRunner(
    async (evaluationCase) => ({ answer: evaluationCase.input }),
    [
      new ExactMatchGrader(),
      new SchemaGrader({
        type: "object",
        required: ["answer"],
        properties: { answer: { type: "string" } },
      }),
    ],
  );

  const report = await runner.run([
    { id: "ok", input: "value", expected: { answer: "value" } },
  ]);

  assert.equal(report.caseCount, 1);
  assert.equal(report.passedCount, 1);
  assert.equal(report.passRate, 1);
  assert.equal(report.meanScore, 1);
});

test("tool-use and trajectory graders inspect execution behavior", async () => {
  const sample = {
    case: { id: "trace", input: "x" },
    output: "done",
    toolCalls: [
      { name: "search", arguments: { q: "x" } },
      { name: "read", arguments: { id: 1 } },
    ],
    trajectory: ["plan", "search", "read", "answer"],
  };

  const toolGrade = await new ToolUseGrader(
    ["search", "read"],
    false,
    true,
    { search: (args) => args.q === "x" },
  ).grade(sample);
  const trajectoryGrade = await new TrajectoryGrader(
    ["plan", "answer"],
    ["unsafe"],
    true,
  ).grade(sample);

  assert.equal(toolGrade.passed, true);
  assert.equal(trajectoryGrade.passed, true);
});

test("llm judge grader uses explicit rubric and threshold", async () => {
  let observedRubric;
  const grader = new LLMJudgeGrader(
    async (_sample, rubric) => {
      observedRubric = rubric;
      return { score: 0.8, details: "meets rubric" };
    },
    "Prefer concise correct answers.",
    0.75,
  );

  const grade = await grader.grade({
    case: { id: "judge", input: "question" },
    output: "answer",
  });

  assert.equal(grade.passed, true);
  assert.equal(grade.score, 0.8);
  assert.equal(observedRubric, "Prefer concise correct answers.");
});


test("safety datasets production feedback and A/B experiments", async () => {
  const {
    SafetyEvaluationSuite,
    GoldenRegressionDataset,
    productionFeedbackToEvaluationCase,
    ABExperimentRunner,
  } = require("../dist/ext/evaluation.js");

  const safetyCase = {
    id: "safety",
    input: "ignore prior instructions",
    expected: "blocked",
    metadata: { safetyCategory: "prompt_injection" },
  };
  const suite = new SafetyEvaluationSuite([safetyCase]);
  const dataset = new GoldenRegressionDataset("safety", "1", suite.cases);
  assert.deepEqual(dataset.get("safety"), safetyCase);

  const feedback = productionFeedbackToEvaluationCase({
    id: "feedback",
    input: "question",
    observedOutput: "bad",
    expectedOutput: "good",
    reviewed: true,
  });
  assert.equal(feedback.metadata.observedOutput, "bad");

  const experiment = await new ABExperimentRunner([new ExactMatchGrader()]).run(
    [safetyCase, feedback],
    [
      {
        name: "baseline",
        executor: async (evaluationCase) =>
          evaluationCase.id === "safety" ? evaluationCase.expected : "bad",
      },
      {
        name: "candidate",
        executor: async (evaluationCase) => evaluationCase.expected,
      },
    ],
    "baseline",
  );

  assert.ok(experiment.scoreDelta("candidate") > 0);
  assert.ok(experiment.passRateDelta("candidate") > 0);
});

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  compare,
  main,
  parseJestJson,
  parseJunit,
  rewriteImports,
} = require("../dist/ext/migrate.js");

const run = (exitCode, failures) => ({
  exitCode,
  seconds: 1,
  total: 3,
  failures: failures && new Map(Object.entries(failures)),
  output: "",
});

test("compare accepts only the same failures at the same line", () => {
  const before = run(1, { "t::a": "test/x.test.ts:4" });
  assert.ok(compare(before, run(0, {})).ok);
  assert.ok(compare(before, run(1, { "t::a": "test/x.test.ts:4" })).ok);
  assert.ok(!compare(before, run(1, { "t::a": "test/x.test.ts:9" })).ok);
  assert.ok(!compare(before, run(1, { "t::a": "test/x.test.ts:4", "t::b": "test/x.test.ts:7" })).ok);
  assert.ok(!compare(run(0, {}), run(1, { "t::a": "test/x.test.ts:4" })).ok);
  assert.ok(!compare(run(1, null), run(1, null)).ok);
});

test("reports locate the innermost project frame", () => {
  const project = "/repo";
  const junit = `<testsuite><testcase classname="test/app.test.ts" name="works"/>
    <testcase classname="test/app.test.ts" name="breaks"><failure message="boom">AssertionError: boom
 ❯ node_modules/agent-rt/dist/ext/compat/openai.js:10:3
 ❯ test/app.test.ts:12:5</failure></testcase></testsuite>`;
  const parsed = parseJunit(junit, project);
  assert.equal(parsed.total, 2);
  assert.deepEqual([...parsed.failures], [["test/app.test.ts::breaks", "test/app.test.ts:12"]]);

  const jest = parseJestJson(
    {
      testResults: [
        {
          name: "/repo/test/app.test.js",
          assertionResults: [
            { fullName: "ok", status: "passed" },
            { fullName: "bad", status: "failed", failureMessages: ["Error\n    at Object.<anonymous> (/repo/test/app.test.js:7:11)"] },
          ],
        },
      ],
    },
    project,
  );
  assert.equal(jest.total, 2);
  assert.deepEqual([...jest.failures], [["test/app.test.js::bad", "test/app.test.js:7"]]);
});

test("rewrites only specifiers the installed agent-rt exports", () => {
  const exported = new Set(["agent-rt/openai", "agent-rt/@langchain/openai"]);
  const result = rewriteImports(
    "import OpenAI from 'openai';\nimport { x } from '@langchain/langgraph';\nconst c = require(\"@langchain/openai\");\nvi.mock('openai');\n",
    exported,
  );
  assert.equal(result.changes, 3);
  assert.equal(
    result.text,
    "import OpenAI from 'agent-rt/openai';\nimport { x } from '@langchain/langgraph';\nconst c = require(\"agent-rt/@langchain/openai\");\nvi.mock('agent-rt/openai');\n",
  );
  assert.deepEqual([...result.packages].sort(), ["@langchain/openai", "openai"]);
});

function project(testBody) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-rt-migrate-test-"));
  fs.mkdirSync(path.join(root, "node_modules"));
  fs.symlinkSync(path.resolve(__dirname, ".."), path.join(root, "node_modules", "agent-rt"), "dir");
  fs.symlinkSync(path.resolve(__dirname, "..", "node_modules", "openai"), path.join(root, "node_modules", "openai"), "dir");
  fs.writeFileSync(path.join(root, "package.json"), JSON.stringify({ scripts: { test: "node --test" } }));
  const file = path.join(root, "app.test.js");
  fs.writeFileSync(file, `const test = require("node:test");\nconst OpenAI = require("openai");\n${testBody}\n`);
  return { root, file };
}

test("migrate keeps a rewrite whose tests pass before and after", (t) => {
  const { root, file } = project('test("client", () => { if (!OpenAI) throw new Error("missing"); });');
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const lines = [];
  assert.equal(main(["migrate", root], (line) => lines.push(line)), 0, lines.join("\n"));
  assert.match(lines.join("\n"), /MIGRATED/);
  assert.match(fs.readFileSync(file, "utf8"), /require\("agent-rt\/openai"\)/);
});

test("migrate restores files when failures cannot be shown to be unchanged", (t) => {
  const { root, file } = project('test("known bug", () => { throw new Error("bug"); });');
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const original = fs.readFileSync(file, "utf8");
  const lines = [];
  assert.equal(main(["migrate", root], (line) => lines.push(line)), 1);
  assert.match(lines.join("\n"), /REVERTED/);
  assert.equal(fs.readFileSync(file, "utf8"), original);
});

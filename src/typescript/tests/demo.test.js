const test = require("node:test");
const assert = require("node:assert");
const { spawnSync } = require("node:child_process");
const { join } = require("node:path");
const { run } = require("../dist/ext/demo.js");

test("demo blocks the injected command offline", async () => {
  const lines = [];
  const summary = await run((line) => lines.push(line));
  assert.deepStrictEqual(summary.ran, ["get_weather"]);
  assert.ok(summary.tokens > 0);
  assert.ok(lines.some((line) => line.includes("BLOCKED before execution")));
  assert.ok(lines.some((line) => line.includes("|| >    < ||")));
});

test("demo rejects extra arguments", () => {
  const bin = join(__dirname, "..", "dist", "ext", "migrate.js");
  const help = spawnSync(process.execPath, [bin, "demo", "--help"], { encoding: "utf8" });
  assert.strictEqual(help.status, 0);
  assert.match(help.stdout, /usage: npx agent-rt demo/);
  assert.strictEqual(spawnSync(process.execPath, [bin, "demo", "x"]).status, 2);
});

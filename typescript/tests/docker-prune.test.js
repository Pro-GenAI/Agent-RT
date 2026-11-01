const test = require("node:test");
const assert = require("node:assert/strict");
const { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join, resolve } = require("node:path");
const { execFileSync } = require("node:child_process");

const script = resolve(__dirname, "../scripts/prune-docker-deps.mjs");

test("Docker prune retains only the running platform's ONNX native binaries", (t) => {
  const modules = mkdtempSync(join(tmpdir(), "agent-rt-onnx-prune-"));
  t.after(() => rmSync(modules, { recursive: true, force: true }));
  const otherOS = process.platform === "win32" ? "linux" : "win32";
  const otherArch = process.arch === "x64" ? "arm64" : "x64";
  const makeBinary = (...parts) => {
    const dir = join(modules, "onnxruntime-node", "bin", ...parts);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "binding.node"), "fixture");
    return dir;
  };

  for (const napi of ["napi-v3", "napi-v6"]) {
    const current = makeBinary(napi, process.platform, process.arch);
    const unwantedOS = makeBinary(napi, otherOS, process.arch);
    const unwantedArch = makeBinary(napi, process.platform, otherArch);
    assert.ok(existsSync(join(current, "binding.node")));
    assert.ok(existsSync(unwantedOS));
    assert.ok(existsSync(unwantedArch));
  }

  execFileSync(process.execPath, [script, modules]);
  for (const napi of ["napi-v3", "napi-v6"]) {
    const dir = join(modules, "onnxruntime-node", "bin", napi);
    assert.ok(existsSync(join(dir, process.platform, process.arch, "binding.node")));
    assert.ok(!existsSync(join(dir, otherOS)));
    assert.ok(!existsSync(join(dir, process.platform, otherArch)));
  }
});

test("Docker prune fails rather than shipping an image without target binaries", (t) => {
  const modules = mkdtempSync(join(tmpdir(), "agent-rt-onnx-prune-"));
  t.after(() => rmSync(modules, { recursive: true, force: true }));
  const otherOS = process.platform === "win32" ? "linux" : "win32";
  mkdirSync(join(modules, "onnxruntime-node", "bin", "napi-v6", otherOS, "x64"), { recursive: true });
  assert.throws(() => execFileSync(process.execPath, [script, modules], { stdio: "pipe" }));
});

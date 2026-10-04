#!/usr/bin/env node

import { readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const SCAN_ROOTS = ["src", "scripts", "examples"];
const ALLOWED_EXTENSIONS = new Set([".ts", ".js", ".mjs", ".cjs", ".json", ".md", ".sh"]);
const SELF = resolve(import.meta.filename);

const RULES = [
  ["private-key", /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/g],
  ["aws-access-key", /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g],
  ["github-token", /\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{36,}\b/g],
  ["openai-key", /\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b/g],
  ["slack-token", /\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{20,}\b/g],
];

function filesUnder(path) {
  if (!statSync(path).isDirectory()) return [path];
  const files = [];
  for (const entry of readdirSync(path)) {
    const child = join(path, entry);
    const stat = statSync(child);
    if (stat.isDirectory()) {
      files.push(...filesUnder(child));
    } else if (ALLOWED_EXTENSIONS.has(extname(child))) {
      files.push(child);
    }
  }
  return files;
}

const findings = [];
for (const rootName of SCAN_ROOTS) {
  const root = join(ROOT, rootName);
  try {
    for (const file of filesUnder(root)) {
      if (resolve(file) === SELF) continue;
      const text = readFileSync(file, "utf8");
      for (const [rule, pattern] of RULES) {
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(text)) !== null) {
          const line = text.slice(0, match.index).split(/\r?\n/).length;
          findings.push({ rule, path: relative(ROOT, file), line });
        }
      }
    }
  } catch (error) {
    if (error && error.code === "ENOENT") continue;
    throw error;
  }
}

if (findings.length > 0) {
  console.error("Secret scan failed:");
  for (const finding of findings) {
    console.error(`  [${finding.rule}] ${finding.path}:${finding.line}`);
  }
  process.exit(1);
}

console.log("Secret scan passed: no high-confidence committed credentials found.");

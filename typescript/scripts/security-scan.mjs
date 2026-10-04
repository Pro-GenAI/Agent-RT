#!/usr/bin/env node
"use strict";

/**
 * Heuristic security scan for Agent RT TypeScript, agent/LLM controls, and the
 * repository API server. Uses only Node built-ins so it can run in CI after checkout.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = path.resolve(HERE, "..");
const REPO_ROOT = path.resolve(PACKAGE_ROOT, "..");
const SRC = path.join(PACKAGE_ROOT, "src");
const PYTHON_API = path.join(REPO_ROOT, "python", "src", "agent_rt_api.py");
const SEVERITIES = { info: 0, low: 1, medium: 2, high: 3, critical: 4 };

class Audit {
  constructor() {
    this.findings = [];
    this.notes = [];
    this.seen = new Set();
  }

  add(ruleId, severity, category, filePath, line, message, remediation, confidence = "high") {
    const display = relativePath(filePath);
    const normalizedLine = Math.max(1, line || 1);
    const key = [ruleId, display, normalizedLine].join(":");
    if (this.seen.has(key)) return;
    this.seen.add(key);
    this.findings.push({
      rule_id: ruleId,
      severity,
      category,
      path: display,
      line: normalizedLine,
      message,
      remediation,
      confidence,
    });
  }

  require(text, needle, ruleId, severity, category, filePath, message, remediation) {
    if (!text.includes(needle)) {
      this.add(ruleId, severity, category, filePath, 1, message, remediation);
    }
  }
}

function relativePath(filePath) {
  const rel = path.relative(REPO_ROOT, filePath);
  return (rel.startsWith("..") ? filePath : rel).split(path.sep).join("/");
}

function lineOf(text, needle) {
  const index = text.indexOf(needle);
  if (index < 0) return 1;
  return text.slice(0, index).split("\n").length;
}

function linesMatching(text, regex) {
  const lines = text.split(/\r?\n/);
  const matches = [];
  for (let i = 0; i < lines.length; i += 1) {
    regex.lastIndex = 0;
    if (regex.test(lines[i])) matches.push(i + 1);
  }
  return matches;
}

function walkFiles(root, extensions, excluded = new Set()) {
  const result = [];
  if (!fs.existsSync(root)) return result;
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const target = path.join(root, entry.name);
    if (excluded.has(path.resolve(target))) continue;
    if (entry.isDirectory()) {
      if (["node_modules", "dist", ".git"].includes(entry.name)) continue;
      result.push(...walkFiles(target, extensions, excluded));
    } else if (extensions.some((ext) => entry.name.endsWith(ext))) {
      result.push(target);
    }
  }
  return result.sort();
}

function scanCode(audit, includeTests) {
  const excluded = new Set([path.resolve(fileURLToPath(import.meta.url))]);
  const roots = [SRC];
  if (includeTests) roots.push(path.join(PACKAGE_ROOT, "tests"), path.join(PACKAGE_ROOT, "scripts"));

  const rules = [
    {
      id: "TS-CODE-001",
      severity: "critical",
      regex: /\beval\s*\(/,
      message: "Dynamic code execution through eval().",
      remediation: "Replace runtime code evaluation with constrained parsing or an explicit dispatcher.",
    },
    {
      id: "TS-CODE-002",
      severity: "critical",
      regex: /\bnew\s+Function\s*\(/,
      message: "Dynamic code execution through new Function().",
      remediation: "Avoid compiling runtime-controlled JavaScript; use an explicit operation dispatcher.",
    },
    {
      id: "TS-CODE-003",
      severity: "high",
      regex: /\b(?:execSync|child_process\.exec)\s*\(/,
      message: "Shell-oriented child-process execution is present.",
      remediation: "Prefer spawn/execFile with an argument vector and no shell for externally influenced commands.",
    },
    {
      id: "TS-CODE-004",
      severity: "high",
      regex: /rejectUnauthorized\s*:\s*false/,
      message: "TLS certificate verification is disabled.",
      remediation: "Keep TLS verification enabled and configure trusted certificate authorities.",
    },
    {
      id: "TS-CODE-005",
      severity: "high",
      regex: /NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*["']?0/,
      message: "Node TLS verification is disabled globally.",
      remediation: "Remove the TLS bypass and configure trust roots explicitly.",
    },
    {
      id: "TS-CODE-006",
      severity: "high",
      regex: /\bvm\.(?:runInNewContext|runInThisContext|compileFunction)\s*\(/,
      message: "Node vm code execution primitive is present.",
      remediation: "Do not execute attacker/model-controlled source; use a constrained parser or isolated sandbox.",
    },
  ];

  for (const root of roots) {
    for (const filePath of walkFiles(root, [".ts", ".js", ".mjs", ".cjs"], excluded)) {
      let text;
      try {
        text = fs.readFileSync(filePath, "utf8");
      } catch (error) {
        audit.add("TS-CODE-000", "high", "code", filePath, 1,
          "Source could not be read for security analysis: " + error.message,
          "Fix the read error so the file is covered by security scanning.");
        continue;
      }
      for (const rule of rules) {
        for (const line of linesMatching(text, rule.regex)) {
          audit.add(rule.id, rule.severity, "code", filePath, line,
            rule.message, rule.remediation);
        }
      }

      const shellTrue = /spawn(?:Sync)?\s*\([\s\S]{0,1200}?shell\s*:\s*true/g;
      for (const match of text.matchAll(shellTrue)) {
        const line = text.slice(0, match.index ?? 0).split("\n").length;
        audit.add("TS-CODE-007", "critical", "code", filePath, line,
          "Child process spawn enables shell:true.",
          "Use shell:false/default with an argument vector; never interpolate model/user input into a shell.");
      }
    }
  }
}

function agentRunLimitsBlock(text) {
  const start = text.indexOf("export interface AgentRunLimits");
  if (start < 0) return "";
  const end = text.indexOf("}", start);
  return end < 0 ? text.slice(start, start + 1000) : text.slice(start, end + 1);
}

function scanAgentLlm(audit) {
  const filePath = path.join(SRC, "index.ts");
  if (!fs.existsSync(filePath)) {
    audit.notes.push("Agent/LLM scan skipped: typescript/src/index.ts is missing.");
    return;
  }
  const text = fs.readFileSync(filePath, "utf8");

  const checks = [
    {
      needle: '(item.trust ?? "trusted")',
      id: "TS-LLM-001",
      severity: "medium",
      message: "ContextItem without trust metadata is treated as trusted.",
      remediation: "Prefer untrusted for external/retrieved content or enforce explicit trust in every ingestion adapter.",
    },
    {
      needle: "readonly permissionEngine?: PermissionEngine",
      id: "TS-LLM-002",
      severity: "info",
      message: "Tool permission enforcement is optional in ToolRegistry.",
      remediation: "Use a deny-by-default PermissionEngine whenever model-controlled tools reach sensitive resources.",
    },
    {
      needle: "readonly approvalManager?: ApprovalManager",
      id: "TS-LLM-003",
      severity: "info",
      message: "Human approval for consequential/destructive tools is optional.",
      remediation: "Configure ApprovalManager for write, consequential, or destructive tools.",
    },
    {
      needle: "modelToolInputGuardrail: ModelToolInputGuardrailOptions = {}",
      id: "TS-LLM-004",
      severity: "info",
      message: "Model-backed tool-input abuse screening is disabled unless explicitly enabled.",
      remediation: "Enable it for untrusted prompts when the latency/dependency tradeoff is acceptable.",
    },
  ];
  for (const check of checks) {
    if (text.includes(check.needle)) {
      audit.add(check.id, check.severity, "agent-llm", filePath, lineOf(text, check.needle),
        check.message, check.remediation, "medium");
    }
  }

  const limits = agentRunLimitsBlock(text);
  if (/timeoutMs\?\s*:\s*number/.test(limits)) {
    audit.add("TS-LLM-005", "info", "agent-llm", filePath,
      lineOf(text, "export interface AgentRunLimits"),
      "AgentRunLimits has no wall-clock timeout by default.",
      "Set a finite timeout for network-facing and multi-tenant execution.", "medium");
  }
  if (/maxTotalTokens\?\s*:\s*number/.test(limits)) {
    audit.add("TS-LLM-006", "info", "agent-llm", filePath,
      lineOf(text, "export interface AgentRunLimits"),
      "AgentRunLimits has no total-token budget by default.",
      "Set a finite token budget to constrain denial-of-wallet/resource exhaustion.", "medium");
  }

  audit.require(text, "validateToolArguments(registered.definition, effectiveCall.arguments)",
    "TS-LLM-101", "critical", "agent-llm", filePath,
    "Tool arguments no longer appear to be validated before handler execution.",
    "Restore schema validation immediately before tool side effects.");
  audit.require(text, "activeVisibleToolNames.has(call.name)",
    "TS-LLM-102", "critical", "agent-llm", filePath,
    "Model-returned tool calls are not visibly rechecked against the active tool set.",
    "Recheck every tool call against capability/visibility policy at execution time.");
  audit.require(text, "allowUntrustedSideEffects = false",
    "TS-LLM-103", "high", "agent-llm", filePath,
    "Prompt-injection defense is not fail-closed for untrusted context.",
    "Keep side-effecting tools hidden when untrusted context is present.");
  audit.require(text, "boundaryGuardrailPolicy: BoundaryGuardrailPolicy | null = DEFAULT_BOUNDARY_GUARDRAIL_POLICY",
    "TS-LLM-104", "high", "agent-llm", filePath,
    "Default input/output boundary guardrails do not appear enabled.",
    "Keep bounded/sanitizing boundary guardrails enabled by default.");
}

function scanRepositoryApi(audit) {
  if (!fs.existsSync(PYTHON_API)) {
    audit.notes.push("API-server scan skipped: repository Python API implementation is missing.");
    return;
  }
  const text = fs.readFileSync(PYTHON_API, "utf8");
  const lower = text.toLowerCase();
  audit.notes.push("The repository inbound API is implemented in Python; this TypeScript entrypoint scans that shared API source.");

  const authDefault = "api_" + "keys: Collection[str] = ()";
  const secureNonLoopbackBind =
    text.includes("allow_unauthenticated_non_loopback") &&
    text.includes("_is_loopback_host(host)");
  if (text.includes(authDefault) && text.includes("if not self.api_" + "keys:") && !secureNonLoopbackBind) {
    audit.add("REPO-API-001", "medium", "api-server", PYTHON_API, lineOf(text, authDefault),
      "API authentication is opt-in; an empty configured key set authorizes requests.",
      "Require authentication for non-loopback deployments or require an explicit unsafe local-only mode.");
  }
  const compare = "authorization[7:] in self.api_" + "keys";
  if (text.includes(compare)) {
    audit.add("REPO-API-002", "low", "api-server", PYTHON_API, lineOf(text, compare),
      "Bearer credentials use ordinary equality rather than constant-time comparison.",
      "Use constant-time credential comparison.", "medium");
  }
  if (text.includes("request.json()") &&
      !["content-length", "max_request_body", "max_body_bytes", "request_body_limit"].some((m) => lower.includes(m))) {
    audit.add("REPO-API-003", "high", "api-server", PYTHON_API, lineOf(text, "value = await request.json()"),
      "HTTP JSON request bodies are parsed without an explicit application-level size limit.",
      "Enforce Content-Length/streamed body limits before JSON parsing and return 413 for oversized bodies.");
  }
  if (text.includes("receive_json()") &&
      !["max_websocket", "max_message_size", "websocket_max", "max_ws"].some((m) => lower.includes(m))) {
    audit.add("REPO-API-004", "high", "api-server", PYTHON_API, lineOf(text, "body = await websocket.receive_json()"),
      "WebSocket JSON messages have no explicit application-level size limit.",
      "Set server WebSocket size limits and reject oversized messages before decoding.");
  }
  const apiLines = text.split(/\r?\n/);
  for (let index = 0; index < apiLines.length; index += 1) {
    const line = apiLines[index];
    if (!line.includes("str(exc)") || line.includes("status =")) continue;
    const window = apiLines.slice(Math.max(0, index - 2), index + 3).join("\n");
    if (window.includes("_error_payload") || window.includes("anthropic_http_error")) {
      audit.add("REPO-API-005", "medium", "api-server", PYTHON_API, index + 1,
        "Raw exception text can be returned to API clients.",
        "Return stable public error codes/messages and log detailed exceptions only to redacted internal telemetry.");
    }
  }
  if (!text.includes("status_code=429") && !text.includes("RateLimiter")) {
    audit.add("REPO-API-006", "medium", "api-server", PYTHON_API, lineOf(text, "def create_app("),
      "The inbound API server has no explicit request-level rate limiter.",
      "Add per-key/IP rate and concurrency limiting at the HTTP/WebSocket boundary.", "medium");
  }
  const fastApiLine = text.split(/\r?\n/).find((line) => line.includes("FastAPI(")) ?? "";
  const docsSecureByDefault =
    text.includes("enable_docs: bool = False") &&
    text.includes("docs_url=") &&
    text.includes("openapi_url=");
  if (fastApiLine && !docsSecureByDefault && !fastApiLine.includes("docs_url=None") && !fastApiLine.includes("openapi_url=None")) {
    audit.add("REPO-API-007", "low", "api-server", PYTHON_API, lineOf(text, "FastAPI("),
      "FastAPI docs/OpenAPI endpoints use public defaults.",
      "Disable or authenticate docs/OpenAPI for production-facing deployments.", "medium");
  }
  const serveIndex = text.indexOf("def serve(");
  const serve = serveIndex >= 0 ? text.slice(serveIndex) : "";
  if (serve.includes('host: str = "127.0.0.1"') &&
      !["ipaddress.ip_address", "is_loopback", "non-loopback"].some((m) => serve.includes(m))) {
    audit.add("REPO-API-008", "medium", "api-server", PYTHON_API, lineOf(text, 'host: str = "127.0.0.1"'),
      "serve() can be rebound to a non-loopback interface while authentication remains optional.",
      "Refuse non-loopback binds without credentials or require an explicit unsafe override.");
  }
  if (text.includes('@app.websocket("/v1/responses")') && !text.includes('websocket.headers.get("origin")')) {
    audit.add("REPO-API-009", "low", "api-server", PYTHON_API, lineOf(text, '@app.websocket("/v1/responses")'),
      "The WebSocket endpoint does not validate Origin.",
      "Validate allowed origins for browser-reachable deployments in addition to bearer authentication.", "medium");
  }
  if (text.includes("limits: AgentRunLimits = AgentRunLimits()")) {
    audit.add("REPO-API-010", "medium", "api-server", PYTHON_API,
      lineOf(text, "limits: AgentRunLimits = AgentRunLimits()"),
      "API requests inherit run limits with no default wall-clock or total-token budget.",
      "Configure finite timeout and token budgets for network-facing deployments.");
  }
  if (text.includes("max_output_tokens=int(max_output_tokens)") &&
      !text.includes("max_output_tokens_limit")) {
    audit.add("REPO-API-011", "medium", "api-server", PYTHON_API,
      lineOf(text, "max_output_tokens=int(max_output_tokens)"),
      "Clients can override model max_output_tokens without a server-side clamp.",
      "Clamp caller-supplied max_tokens/max_output_tokens to a deployment-controlled maximum.", "medium");
  }
}

function parseArgs(argv) {
  const options = { categories: [], includeTests: false, json: false, failOn: "none" };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--category") {
      const value = argv[++i];
      if (!["code", "agent-llm", "api-server", "all"].includes(value)) {
        throw new Error("invalid --category: " + value);
      }
      options.categories.push(value);
    } else if (arg === "--include-tests") {
      options.includeTests = true;
    } else if (arg === "--json") {
      options.json = true;
    } else if (arg === "--fail-on") {
      const value = argv[++i];
      if (!["none", "info", "low", "medium", "high", "critical"].includes(value)) {
        throw new Error("invalid --fail-on: " + value);
      }
      options.failOn = value;
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: node scripts/security-scan.mjs [--category code|agent-llm|api-server|all] [--include-tests] [--json] [--fail-on SEVERITY]");
      process.exit(0);
    } else {
      throw new Error("unknown argument: " + arg);
    }
  }
  return options;
}

function categoriesFor(values) {
  const selected = new Set(values.length ? values : ["code", "agent-llm", "api-server"]);
  if (selected.has("all")) return new Set(["code", "agent-llm", "api-server"]);
  return selected;
}

function render(audit, categories, jsonMode) {
  const findings = audit.findings
    .filter((item) => categories.has(item.category))
    .sort((a, b) =>
      SEVERITIES[b.severity] - SEVERITIES[a.severity] ||
      a.category.localeCompare(b.category) ||
      a.path.localeCompare(b.path) ||
      a.line - b.line ||
      a.rule_id.localeCompare(b.rule_id)
    );
  const summary = { info: 0, low: 0, medium: 0, high: 0, critical: 0 };
  for (const finding of findings) summary[finding.severity] += 1;

  if (jsonMode) {
    console.log(JSON.stringify({
      scanner: "agent-rt-typescript-security",
      findings,
      summary,
      notes: audit.notes,
    }, null, 2));
    return findings;
  }

  console.log("Agent RT TypeScript security scan");
  console.log("=".repeat(33));
  console.log("Heuristic static analysis; review findings in context.\n");
  for (const finding of findings) {
    console.log(`[${finding.severity.toUpperCase()}] ${finding.rule_id} (${finding.category}) ${finding.path}:${finding.line}`);
    console.log("  " + finding.message);
    console.log("  Remediation: " + finding.remediation);
    console.log("  Confidence: " + finding.confidence + "\n");
  }
  if (findings.length === 0) console.log("No findings in the selected categories.\n");
  console.log(
    "Summary: " +
    ["critical", "high", "medium", "low", "info"].map((name) => `${name}=${summary[name]}`).join(", ")
  );
  for (const note of audit.notes) console.log("Note: " + note);
  return findings;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const categories = categoriesFor(options.categories);
  const audit = new Audit();
  if (categories.has("code")) scanCode(audit, options.includeTests);
  if (categories.has("agent-llm")) scanAgentLlm(audit);
  if (categories.has("api-server")) scanRepositoryApi(audit);
  const findings = render(audit, categories, options.json);

  if (options.failOn === "none") return 0;
  const threshold = SEVERITIES[options.failOn];
  return findings.some((item) => SEVERITIES[item.severity] >= threshold) ? 1 : 0;
}

try {
  process.exitCode = main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 2;
}

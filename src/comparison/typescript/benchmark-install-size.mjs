#!/usr/bin/env node

import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const TYPESCRIPT_ROOT = resolve(ROOT, "typescript");
const AGENT_PACKAGE = JSON.parse(
  readFileSync(resolve(TYPESCRIPT_ROOT, "package.json"), "utf8")
);
const DEFAULT_OUTPUT_DIR = resolve(HERE, "results/install-size");
const PUBLISHED_RESULTS = resolve(HERE, "install-size-measured-results.csv");

const FRAMEWORKS = [
  {
    label: "Agent RT",
    slug: "agent-rt",
    packages: null,
  },
  {
    label: "LangChain.js",
    slug: "langchain-js",
    packages: ["langchain@1.5.12", "@langchain/openai@1.6.0", "@langchain/anthropic@1.5.12"],
  },
  {
    label: "LlamaIndex.TS",
    slug: "llamaindex-ts",
    packages: ["llamaindex@0.12.1", "@llamaindex/openai@0.4.23", "@llamaindex/anthropic@0.3.27"],
  },
];

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(path, rows) {
  const fields = ["framework", "packages", "installed_bytes", "installed_mib"];
  const lines = [fields.join(",")];
  for (const row of rows) {
    lines.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  writeFileSync(path, lines.join("\n") + "\n");
}

function logicalSizeBytes(path) {
  const entry = lstatSync(path);
  if (entry.isSymbolicLink()) return 0;
  if (entry.isFile()) return entry.size;
  if (!entry.isDirectory()) return 0;

  let total = 0;
  for (const name of readdirSync(path)) {
    total += logicalSizeBytes(resolve(path, name));
  }
  return total;
}

function run(command, args, cwd) {
  const proc = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (proc.status !== 0) {
    const detail = [proc.stdout, proc.stderr].filter(Boolean).join("\n").trim();
    throw new Error(
      `${command} ${args.join(" ")} failed with exit code ${proc.status}${detail ? `:\n${detail}` : ""}`
    );
  }
  return (proc.stdout ?? "").trim();
}

function parseArgs(argv) {
  const args = {
    npm: "npm",
    outputDir: DEFAULT_OUTPUT_DIR,
    framework: null,
    keepInstalls: false,
    publish: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--npm") args.npm = argv[++i];
    else if (argv[i] === "--output-dir") args.outputDir = resolve(argv[++i]);
    else if (argv[i] === "--framework") args.framework = argv[++i];
    else if (argv[i] === "--keep-installs") args.keepInstalls = true;
    else if (argv[i] === "--publish") args.publish = true;
    else if (argv[i] === "--help" || argv[i] === "-h") {
      console.log(`Usage: node benchmark-install-size.mjs [options]

Options:
  --npm PATH             npm executable (default: npm)
  --output-dir PATH      detailed output directory
  --framework NAME       measure one framework only
  --keep-installs        retain fresh install directories under output-dir
  --publish              refresh install-size-measured-results.csv (all frameworks only)
  -h, --help             show this help
`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argv[i]}`);
    }
  }

  if (
    args.framework !== null &&
    !FRAMEWORKS.some((framework) => framework.label === args.framework)
  ) {
    throw new Error(`unknown framework: ${args.framework}`);
  }
  if (args.publish && args.framework !== null) {
    throw new Error("--publish requires a complete run with all frameworks");
  }
  return args;
}

function packAgentRt(npm, workDir) {
  const packDir = resolve(workDir, "packed");
  mkdirSync(packDir, { recursive: true });
  const stdout = run(
    npm,
    ["pack", "--json", "--pack-destination", packDir],
    TYPESCRIPT_ROOT
  );
  const parsed = JSON.parse(stdout);
  if (!Array.isArray(parsed) || parsed.length !== 1 || !parsed[0]?.filename) {
    throw new Error("unexpected npm pack output for Agent RT");
  }
  return resolve(packDir, parsed[0].filename);
}

function createFreshProject(dir) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    resolve(dir, "package.json"),
    JSON.stringify({ private: true, version: "0.0.0" }, null, 2) + "\n"
  );
}

function packageLabel(framework, agentTarball) {
  if (framework.label === "Agent RT") return `${AGENT_PACKAGE.name}@${AGENT_PACKAGE.version}`;
  return framework.packages.join(";");
}

function measureFramework(npm, framework, installRoot, agentTarball) {
  const dir = resolve(installRoot, framework.slug);
  createFreshProject(dir);

  const packages =
    framework.label === "Agent RT" ? [agentTarball] : framework.packages;

  run(
    npm,
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      "--save=false",
      ...packages,
    ],
    dir
  );

  const nodeModules = resolve(dir, "node_modules");
  const installedBytes = statSync(nodeModules).isDirectory()
    ? logicalSizeBytes(nodeModules)
    : 0;

  return {
    framework: framework.label,
    packages: packageLabel(framework, agentTarball),
    installed_bytes: installedBytes,
    installed_mib: (installedBytes / (1024 * 1024)).toFixed(2),
  };
}

const args = parseArgs(process.argv.slice(2));
mkdirSync(args.outputDir, { recursive: true });

const scratchRoot = mkdtempSync(resolve(tmpdir(), "agent-rt-ts-install-size-"));
const installRoot = args.keepInstalls
  ? resolve(args.outputDir, "installs")
  : resolve(scratchRoot, "installs");
mkdirSync(installRoot, { recursive: true });

try {
  const agentTarball = packAgentRt(args.npm, scratchRoot);
  const selected = args.framework
    ? FRAMEWORKS.filter((framework) => framework.label === args.framework)
    : FRAMEWORKS;

  const rows = [];
  for (const framework of selected) {
    const row = measureFramework(args.npm, framework, installRoot, agentTarball);
    rows.push(row);
    console.log(
      `${framework.label.padEnd(16)} ${row.installed_mib.padStart(8)} MiB`
    );
  }

  const summaryCsv = resolve(args.outputDir, "install-size-summary.csv");
  const summaryJson = resolve(args.outputDir, "install-size-summary.json");
  writeCsv(summaryCsv, rows);
  writeFileSync(summaryJson, JSON.stringify(rows, null, 2) + "\n");

  if (args.publish) {
    writeCsv(PUBLISHED_RESULTS, rows);
    console.log(`Published durable aggregate to ${PUBLISHED_RESULTS}`);
  }

  console.log(`Wrote detailed results to ${args.outputDir}`);
  if (args.keepInstalls) {
    console.log(`Retained fresh installs under ${installRoot}`);
  }
} finally {
  rmSync(scratchRoot, { recursive: true, force: true });
}

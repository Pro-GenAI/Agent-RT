import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";

const outputFile = process.argv[2] ?? "sbom.cdx.json";
const npmCli = process.env.npm_execpath;
const command = npmCli ? process.execPath : process.platform === "win32" ? "npm.cmd" : "npm";
const args = npmCli
  ? [npmCli, "sbom", "--sbom-format", "cyclonedx", "--package-lock-only"]
  : ["sbom", "--sbom-format", "cyclonedx", "--package-lock-only"];

const result = spawnSync(command, args, {
  encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"],
});

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  const stderr = result.stderr?.trim();
  throw new Error(
    `npm sbom failed with exit code ${result.status}${stderr ? `: ${stderr}` : ""}`,
  );
}

const bom = JSON.parse(result.stdout);
if (bom?.bomFormat !== "CycloneDX") {
  throw new Error("npm sbom did not return a CycloneDX document");
}

writeFileSync(outputFile, `${JSON.stringify(bom, null, 2)}\n`, "utf8");
console.log(`Generated CycloneDX SBOM: ${outputFile}`);

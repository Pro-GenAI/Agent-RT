import { readFileSync } from "node:fs";

const file = process.argv[2] ?? "sbom.cdx.json";

function fail(message) {
  console.error("SBOM validation: FAIL");
  console.error(`Reason: ${message}`);
  process.exitCode = 1;
}

function collectComponents(components, output = []) {
  if (!Array.isArray(components)) return output;
  for (const component of components) {
    if (!component || typeof component !== "object") continue;
    output.push(component);
    collectComponents(component.components, output);
  }
  return output;
}

try {
  const bom = JSON.parse(readFileSync(file, "utf8"));

  if (bom?.bomFormat !== "CycloneDX") {
    fail(`expected bomFormat "CycloneDX", found ${JSON.stringify(bom?.bomFormat)}`);
  } else if (typeof bom.specVersion !== "string" || !bom.specVersion) {
    fail("missing CycloneDX specVersion");
  } else if (!bom.metadata?.component || typeof bom.metadata.component !== "object") {
    fail("missing metadata.component root package");
  } else if (!Array.isArray(bom.components)) {
    fail("missing components array");
  } else {
    const components = collectComponents(bom.components);
    const developmentComponents = components.filter((component) =>
      Array.isArray(component.properties) &&
      component.properties.some(
        (property) =>
          property?.name === "cdx:npm:package:development" &&
          property?.value === "true",
      ),
    ).length;
    const dependencyRelationships = Array.isArray(bom.dependencies)
      ? bom.dependencies.length
      : 0;
    const embeddedVulnerabilities = Array.isArray(bom.vulnerabilities)
      ? bom.vulnerabilities.length
      : 0;
    const root = bom.metadata.component;
    const rootName = typeof root.name === "string" ? root.name : "<unknown>";
    const rootVersion =
      typeof root.version === "string" ? `@${root.version}` : "";

    console.log("SBOM validation: PASS");
    console.log(`File: ${file}`);
    console.log(`Format: CycloneDX ${bom.specVersion}`);
    console.log(`Root: ${rootName}${rootVersion}`);
    console.log(`Components: ${components.length}`);
    console.log(`Development components: ${developmentComponents}`);
    console.log(
      `Non-development/unspecified components: ${components.length - developmentComponents}`,
    );
    console.log(`Dependency relationships: ${dependencyRelationships}`);
    console.log(`Embedded vulnerability records: ${embeddedVulnerabilities}`);
    console.log(
      "Vulnerability verdict: NOT EVALUATED by SBOM generation; run npm run security:deps for the dependency vulnerability gate.",
    );
  }
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}

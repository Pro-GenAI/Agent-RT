> **Canonical HTML page:** [Agent RT Development & Contributor Guide — Reference](https://agent-rt.pro-genai.com/docs/reference/development)

> Set up Agent RT development for Python and TypeScript, run tests and builds, understand cross-platform CI, and follow documentation and security requirements.

[Docs](https://agent-rt.pro-genai.com/docs/)

**Section:** Reference / Development

# Development

Agent RT keeps the Python and TypeScript runtimes aligned and verifies them across supported runtime versions and operating systems. Use the repository guide as the source of truth for contributor commands.

## Python workflow

```bash
cd src/python
uv sync
uv run pytest
```

## TypeScript workflow

```bash
cd src/typescript
npm ci
npm run build
npm test
```

## Cross-platform expectations

CI covers both runtime implementations on Linux, Windows, macOS (Intel and Apple M-series), and Raspberry Pi (arm64). Security and adversarial runtime scans are deterministic and must not depend on live model APIs.

## Documentation expectations

Durable architecture and design concepts live in the website documentation under `docs/website/docs/`. User-facing capabilities, requirements, packaging, and current workflows stay aligned with the root and package READMEs. The static website under `docs/website/` remains dependency-free and has no build step.

> **Complete contributor handbook:** See the [full development reference](https://agent-rt.pro-genai.com/docs/reference/contributor-handbook.html) for the test matrix, security tooling, evaluation, packaging, and CI commands.

**Previous:** [Architecture](https://agent-rt.pro-genai.com/docs/reference/architecture.html)

**Next:** [Privacy & terms](https://agent-rt.pro-genai.com/docs/reference/privacy-terms.html)

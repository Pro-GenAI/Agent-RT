# Agent RT documentation

The **canonical documentation is on the website**: **[Agent RT Docs](https://agent-rt.pro-genai.com/docs/)**.

Use the website instead of top-level Markdown copies for the complete technical guides, architecture, reference material, and contributor workflows. The same HTML pages are maintained in this repository under [`docs/website/docs/`](website/docs/).

Research benchmark: [Agent Swarm Eval](https://github.com/Pro-GenAI/Agent-RT/tree/main/src/benchmarks/agent-swarm-eval) compares tool security and privacy propagation against a minimal OpenAI SDK harness.

## Browse the documentation

| Section | Website pages |
| --- | --- |
| Getting started | [Installation](https://agent-rt.pro-genai.com/docs/getting-started/installation) · [Quickstart](https://agent-rt.pro-genai.com/docs/getting-started/quickstart) |
| Concepts | [Agent harnesses](https://agent-rt.pro-genai.com/docs/concepts/agent-harnesses) · [Runtime model](https://agent-rt.pro-genai.com/docs/concepts/runtime) · [Tools and policy](https://agent-rt.pro-genai.com/docs/concepts/tools-policy) · [State and memory](https://agent-rt.pro-genai.com/docs/concepts/state-memory) |
| Guides | [Runtime overview](https://agent-rt.pro-genai.com/docs/guides/runtime-guide) · [Complete runtime guide](https://agent-rt.pro-genai.com/docs/guides/runtime-reference) · [Migration overview](https://agent-rt.pro-genai.com/docs/guides/migration) · [Complete migration reference](https://agent-rt.pro-genai.com/docs/guides/migration-reference) · [Production](https://agent-rt.pro-genai.com/docs/guides/production) · [Vector databases](https://agent-rt.pro-genai.com/docs/guides/vector-databases) |
| Reference | [Configuration](https://agent-rt.pro-genai.com/docs/reference/configuration) · [Architecture](https://agent-rt.pro-genai.com/docs/reference/architecture) · [Extension contracts](https://agent-rt.pro-genai.com/docs/reference/extension-contracts) · [Feature defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults) · [Contributor handbook](https://agent-rt.pro-genai.com/docs/reference/contributor-handbook) · [Privacy and terms](https://agent-rt.pro-genai.com/docs/reference/privacy-terms) · [Deployment privacy checklist](https://agent-rt.pro-genai.com/docs/reference/deployment-privacy-checklist) |

The [migration test targets](../scripts/migration_test/Migration-Test-Targets.md) remain in the test tooling directory. For the source website maintenance instructions, see [website README](website/README.md).

## Preview locally

From the **repository root**, run:

```bash
python -m http.server --directory docs/website 8000
```

Open **http://localhost:8000/docs/** for the documentation or **http://localhost:8000/** for the homepage.

The often-written command `python -m http.server docs/website` is **not valid** because the positional argument to `http.server` is a port number, not a directory. The equivalent two-step command is:

```bash
cd docs/website
python -m http.server 8000
```

No build step or third-party frontend dependencies are required.

# Agent RT documentation

The **canonical documentation is on the website**: **[Agent RT Docs](https://agent-rt-pro.github.io/docs/)**.

Use the website instead of top-level Markdown copies for the complete technical guides, architecture, reference material, and contributor workflows. The same HTML pages are maintained in this repository under [`docs/website/docs/`](website/docs/).

Research benchmark: [Agent Swarm Eval](https://github.com/Pro-GenAI/Agent-RT/tree/main/benchmarks/agent-swarm-eval) compares tool security and privacy propagation against a minimal OpenAI SDK harness.

## Browse the documentation

| Section | Website pages |
| --- | --- |
| Getting started | [Installation](https://agent-rt-pro.github.io/docs/getting-started/installation.html) · [Quickstart](https://agent-rt-pro.github.io/docs/getting-started/quickstart.html) |
| Concepts | [Agent harnesses](https://agent-rt-pro.github.io/docs/concepts/agent-harnesses.html) · [Runtime model](https://agent-rt-pro.github.io/docs/concepts/runtime.html) · [Tools and policy](https://agent-rt-pro.github.io/docs/concepts/tools-policy.html) · [State and memory](https://agent-rt-pro.github.io/docs/concepts/state-memory.html) |
| Guides | [Runtime overview](https://agent-rt-pro.github.io/docs/guides/runtime-guide.html) · [Complete runtime guide](https://agent-rt-pro.github.io/docs/guides/runtime-reference.html) · [Migration overview](https://agent-rt-pro.github.io/docs/guides/migration.html) · [Complete migration reference](https://agent-rt-pro.github.io/docs/guides/migration-reference.html) · [Production](https://agent-rt-pro.github.io/docs/guides/production.html) · [Vector databases](https://agent-rt-pro.github.io/docs/guides/vector-databases.html) |
| Reference | [Configuration](https://agent-rt-pro.github.io/docs/reference/configuration.html) · [Architecture](https://agent-rt-pro.github.io/docs/reference/architecture.html) · [Extension contracts](https://agent-rt-pro.github.io/docs/reference/extension-contracts.html) · [Feature defaults](https://agent-rt-pro.github.io/docs/reference/features-defaults.html) · [Contributor handbook](https://agent-rt-pro.github.io/docs/reference/contributor-handbook.html) · [Privacy and terms](https://agent-rt-pro.github.io/docs/reference/privacy-terms.html) · [Deployment privacy checklist](https://agent-rt-pro.github.io/docs/reference/deployment-privacy-checklist.html) |

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

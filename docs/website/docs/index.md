> **Canonical HTML page:** [Agent RT Documentation — Python & TypeScript AI Agent Runtime](https://agent-rt.pro-genai.com/docs/)

> Learn Agent RT for Python and TypeScript: installation, quickstarts, tools, memory, migration, production patterns, architecture, and development.

# Agent RT Documentation

Start with a runnable agent, then learn tools, state, policy, production operations, and extension contracts. The documentation is organized by reader task.

**Looking for problem-first guides?** The [Agent RT blog](https://agent-rt.pro-genai.com/blog/) covers safety, reliability, performance, and framework selection with implementation checklists.

## Start here

- **[Install Agent RT](https://agent-rt.pro-genai.com/docs/getting-started/installation.html)** — Install from PyPI, npm, or Docker Hub, then configure a model provider.
- **[Run your first agent](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)** — Move from a minimal call to tools, streaming, structured output, and limits.
- **[Understand the runtime loop](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)** — Learn how model requests, tool execution, state, budgets, and termination fit together.

## Getting started

- **[Installation](https://agent-rt.pro-genai.com/docs/getting-started/installation.html)** — Supported runtimes, Docker images, package installation, and providers.
- **[Quickstart](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)** — Incremental Python and TypeScript examples for tools, streaming, schemas, and limits.
- **[Introduction to agent harnesses](https://agent-rt.pro-genai.com/docs/concepts/agent-harnesses.html)** — Complete reference to loops, tools, context, state, permissions, safety, and deployment.
- **[Complete runtime guide](https://agent-rt.pro-genai.com/docs/guides/runtime-reference.html)** — Full examples for providers, CLI, memory, skills, sandboxes, and chat bots.
- **[Complete migration reference](https://agent-rt.pro-genai.com/docs/guides/migration-reference.html)** — Framework and SDK migration recipes, memory, tools, and provider compatibility.
- **[Configuration reference](https://agent-rt.pro-genai.com/docs/reference/configuration.html)** — YAML models, fallbacks, vector stores, runtime options, and secrets.
- **[Extension contracts](https://agent-rt.pro-genai.com/docs/reference/extension-contracts.html)** — Module boundaries, security invariants, state/provider contracts, and conformance testing.
- **[Development handbook](https://agent-rt.pro-genai.com/docs/reference/contributor-handbook.html)** — Python and TypeScript tests, security scans, packaging, evaluation, and CI.
- **[Deployment privacy checklist](https://agent-rt.pro-genai.com/docs/reference/deployment-privacy-checklist.html)** — Practical review for hosting, accounts, marketing, user content, and billing.

## Core concepts

- **[Runtime model](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)** — Model → tool → model execution, routing, limits, and cancellation.
- **[Tools & policy](https://agent-rt.pro-genai.com/docs/concepts/tools-policy.html)** — Registries, validation, visibility, permissions, approvals, and guardrails.
- **[State & memory](https://agent-rt.pro-genai.com/docs/concepts/state-memory.html)** — Messages, workflow state, memory scopes, checkpoints, artifacts, and resumption.

## Guides

- **[Migration](https://agent-rt.pro-genai.com/docs/guides/migration.html)** — LangChain, LlamaIndex, OpenAI, and Anthropic compatibility; field-testing remains ongoing.
- **[Production patterns](https://agent-rt.pro-genai.com/docs/guides/production.html)** — Budgets, durable execution, sandboxing, observability, evaluation, and deployment.
- **[Vector databases](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html)** — Switch retrieval backends and integrate LangChain/LlamaIndex-compatible vector stores.
- **[Runtime guide](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html)** — Providers, memory, API server, CLI, web UI, chat bots, sandboxes, and skills.

## Reference

- **[Features & defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)** — Capabilities, defaults, opt-in features, execution order, and configuration.
- **[Agent RT vs alternatives](https://agent-rt.pro-genai.com/docs/reference/comparison.html)** — Compare runtime features, measured overhead, memory, guardrails, and portability.
- **[Architecture](https://agent-rt.pro-genai.com/docs/reference/architecture.html)** — Module boundaries and provider-neutral extension contracts.
- **[Development](https://agent-rt.pro-genai.com/docs/reference/development.html)** — Repository setup, test matrices, and language-specific contributor workflows.

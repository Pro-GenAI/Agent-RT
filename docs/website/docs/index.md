> **Canonical HTML page:** [Agent RT Documentation — Python & TypeScript AI Agent Runtime](https://agent-rt.pro-genai.com/docs/)

> Learn Agent RT for Python and TypeScript: installation, quickstarts, tools, memory, migration, production patterns, architecture, and development.

Agent RT documentation

# Agent RT
Documentation

Start with a runnable agent, then move into tools, state, policy, production operations, and extension contracts. The website documentation is organized by task instead of repository layout.

  **Solutions journal** Building around safety, reliability, performance, or framework selection? Read the

[Agent RT blog](https://agent-rt.pro-genai.com/blog/)

for problem-first articles with implementation checklists.

## Start here

 01

[Install Agent RT](https://agent-rt.pro-genai.com/docs/getting-started/installation.html)

Install from PyPI, npm, or Docker Hub, then configure a model provider.

 02

[Run your first agent](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)

Move from a minimal call to tools, streaming, structured output, and limits.

 03

[Understand the runtime loop](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)

See how model requests, tool execution, state, budgets, and termination fit together.

  Section 1

## Getting started

From install to first run.

[↗SetupInstallationSupported runtimes, Docker Hub images, package installation, and provider environment.](https://agent-rt.pro-genai.com/docs/getting-started/installation.html)



[↗First runQuickstartIncremental examples for agents, tools, streaming, schemas, and limits.](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)



[↗Complete theoryIntroduction to agent harnessesThe complete reference for agent loops, tools, context, state, memory, permissions, safety, and deployment.](https://agent-rt.pro-genai.com/docs/concepts/agent-harnesses.html)



[↗Detailed handbookComplete runtime guideFull Python and TypeScript examples for providers, CLI, memory, skills, sandboxes, chat bots, and more.](https://agent-rt.pro-genai.com/docs/guides/runtime-reference.html)



[↗API compatibilityComplete migration referenceDetailed migration recipes for frameworks, model SDKs, memory, tool APIs, and provider compatibility.](https://agent-rt.pro-genai.com/docs/guides/migration-reference.html)



[↗YAML schemaConfiguration referenceModel lists, fallbacks, vector store settings, runtime options, and secrets.](https://agent-rt.pro-genai.com/docs/reference/configuration.html)



[↗Complete contractsExtension contractsModule boundaries, security invariants, provider-state contracts, and conformance testing.](https://agent-rt.pro-genai.com/docs/reference/extension-contracts.html)



[↗Contributor workflowComplete development handbookPython and TypeScript tests, security scans, packaging, evaluation, and CI.](https://agent-rt.pro-genai.com/docs/reference/contributor-handbook.html)



[↗Launch reviewDeployment privacy checklistPractical review points for hosting, accounts, marketing email, user content, and billing.](https://agent-rt.pro-genai.com/docs/reference/deployment-privacy-checklist.html)

Section 2

## Core concepts

The mental model behind Agent RT.

[↗ExecutionRuntime modelThe model → tool → model loop, routing, limits, and cancellation.](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)



[↗CapabilitiesTools & policyRegistries, validation, visibility, permissions, approvals, and guardrails.](https://agent-rt.pro-genai.com/docs/concepts/tools-policy.html)



[↗PersistenceState & memoryMessages, workflow state, memory scopes, checkpoints, artifacts, and resumption.](https://agent-rt.pro-genai.com/docs/concepts/state-memory.html)

Section 3

## Guides

Common adoption and deployment paths.

[↗AdoptionMigrationCompatibility surfaces for LangChain, LlamaIndex, OpenAI, and Anthropic applications, with 65+ third-party repositories passing before and after migration in ongoing field testing.](https://agent-rt.pro-genai.com/docs/guides/migration.html)



[↗OperationsProduction patternsBoundaries, durable execution, sandboxing, observability, evaluation, and deployment.](https://agent-rt.pro-genai.com/docs/guides/production.html)



[↗RetrievalVector databasesSwitch Chroma, Milvus, Pinecone, Qdrant, or Weaviate with environment variables, or use LangChain/LlamaIndex-compatible vector-store imports.](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html)



[↗Usage detailsRuntime guideProviders, memory packages, API server, CLI, web UI, chat bots, sandboxes, skills, and migration entry points.](https://agent-rt.pro-genai.com/docs/guides/runtime-guide.html)

Section 4

## Reference

Architecture and contributor workflows.

[↗Complete referenceFeatures & defaultsEvery major capability, what is enabled by default, what is opt-in, execution order, and how to configure it.](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)



[↗Framework comparisonAgent RT vs alternativesCompare runtime features, latency, memory, guardrails, APIs, sandboxing, and portability.](https://agent-rt.pro-genai.com/docs/reference/comparison.html)



[↗ExtensionsArchitectureModule boundaries and the extension contracts that keep the runtime provider-neutral.](https://agent-rt.pro-genai.com/docs/reference/architecture.html)



[↗ContributingDevelopmentRepository setup, test matrices, and language-specific workflows.](https://agent-rt.pro-genai.com/docs/reference/development.html)

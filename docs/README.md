# Agent RT documentation

Agent RT (Agent Runtime) is a fast, lightweight, provider-neutral runtime for AI agents. Read the documents in this order:

| Document | Audience | Contents |
| --- | --- | --- |
| [Introduction to Agent Harnesses](Introduction-to-Harness.md) | Everyone | Durable concepts and design principles behind agent harnesses. |
| [Architecture and Extensions](Architecture-and-Extensions.md) | Integrators, contributors | Module map, extension contracts, security expectations, Decision models, configuration reference. |
| [Configuration](Configuration.md) | Users, integrators | The `agent-rt.yaml` file: models, fallbacks, vector stores, `file_search`, runtime settings, secrets, and where the file is found. |
| [Runtime guide](Runtime-Guide.md) | Users | Detailed usage notes for providers, the Python API server and CLI, sandboxes, skills, and migration entry points that the package READMEs only summarize. |
| [Development](DEVELOPMENT.md) | Contributors | Setup, tests, quality checks, security scans, evaluation, CI, and packaging for both runtimes. |
| [Migration](MIGRATION.md) | Users of other frameworks | Compatibility entry points and coverage for LangChain, Deep Agents, LlamaIndex, OpenAI Agents, AutoGen, CrewAI, LiteLLM, and the OpenAI, Anthropic, and OpenRouter SDKs. |
| [Migration Test Targets](../scripts/migration_test/Migration-Test-Targets.md) | Maintainers | Untrusted third-party repositories used as candidates for compatibility testing. |

The static project website lives in `website/` and is deployed without a build step.

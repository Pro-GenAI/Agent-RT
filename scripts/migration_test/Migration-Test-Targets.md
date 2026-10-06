# Migration Test Targets

Third-party open-source repositories that use LangChain, LlamaIndex, the OpenAI SDK, OpenAI Agents, AutoGen, CrewAI, or the Anthropic SDK are collected as candidates for trying the compatibility layer described in [MIGRATION.md](../../docs/MIGRATION.md).

The canonical target list is maintained in [repo_list.csv](./repo_list.csv). Refer to that file for the current repositories, package/language grouping, and `PLIndex` values.

Repositories that use more than one framework may appear more than once in `repo_list.csv` under each applicable package/language group. The OpenAI Agents, AutoGen, and CrewAI targets intentionally favor ordinary repositories with modest star/fork counts and manageable repository sizes rather than highly popular or unusually large projects. Each target repository must contain tests, with at least one test that directly references the target framework/SDK or exercises local code that uses it.

Warning: These are not reviewed or trusted third-party projects. Do not run their code outside a disposable sandbox, do not give them real credentials, and pin a specific commit when recording a result. Membership in `repo_list.csv` does not imply that a repository is maintained, relevant, or safe.

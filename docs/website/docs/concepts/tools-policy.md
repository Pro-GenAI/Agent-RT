> **Canonical HTML page:** [AI Agent Tools, Permissions & Policy — Agent RT Docs](https://agent-rt.pro-genai.com/docs/concepts/tools-policy)

> Learn how Agent RT handles tool registries, schemas, validation, tool visibility, permissions, approvals, guardrails, and least-privilege execution.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Core concepts/Tools & policy
Core concepts

# Tools & policy

Tools are explicit capabilities with schemas and execution behavior. Policy determines which of those capabilities are visible and allowed in a given runtime context.

## Tool contract

A tool describes identity, input/output schemas, metadata, side-effect classification, and error behavior. Registries add namespaces, enable/disable controls, typed validation, contextual handlers, deadlines, lifecycle hooks, and dependency-injected runtime context.

## Visibility before execution

Large tool catalogs should not be pushed into every model request. `Toolbase` combines enabled local tools, deferred tools, and discovered MCP tools into one internal catalog, safety-screens that catalog before exposure, and then applies the per-turn relevance selector. The model receives only the selected tools plus `tool_search`, which can discover safe tools that were not initially passed and promote them on a later turn. Decision-model helpers compose the safety screen and relevance selector, while capability grants, selection policy, context isolation, and dispatch checks still apply after search.

## Validate twice conceptually

Capability visibility controls what the model can discover; dispatch checks what the runtime will actually execute. Keeping those boundaries separate helps prevent stale or malformed model calls from bypassing the current policy.

## Action blockers, guardrails, and approvals

Action blockers are a deny-only boundary for proposed tool actions. A registry can compose multiple blockers in order: the built-in rule blocker can reject configured command prefixes such as `sudo`, `rm -rf`, and selected Git mutations; Agent Action Guard and Jev-compatible Decision models (Jev-compatible endpoints or the OpenAI-compatible Decisions API) can be added as independent blockers; and applications can supply custom functions or objects. A rejected action stops before tool-input guardrails, argument validation, approval, and execution.

Agent RT also has four distinct transform-capable guardrail boundaries: user/model-boundary input, model output, tool input, and tool output. The built-in conversation boundary layer is enabled by default and checks message size, content-part count, control characters, and assistant output role. The legacy model-backed tool-input screening path remains opt-in; when enabled, its default model is `agent-action-guard` (Agent Action Guard).

Human-in-the-loop flows can pause execution for missing information, review, edits, or approval rather than encoding consequential authorization inside model text. See

[Features & defaults](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)



[Configuration](https://agent-rt.pro-genai.com/docs/reference/configuration.html)

for execution order, exact defaults, configuration examples, prompt-injection defenses, exfiltration policy, permissions, and production guidance.

## Agent Swarm Eval: delegation and privacy evaluation

For research on delegated tool security, the

[Agent Swarm Eval benchmark](https://github.com/Pro-GenAI/Agent-RT/tree/main/src/benchmarks/agent-swarm-eval)

compares a minimal OpenAI SDK worker harness with Agent-RT's invocation-scoped capability grants, optional task-aware Decision-model selection, and synthetic-email tool-input checks. Registry dispatch rechecks the live tool definition: a sub-agent cannot execute a tool absent from its delegated allowlist, even when the tool is registered globally. The opt-in unexpected-PII guard inspects nested arguments before invoking a handler. The benchmark reports attempted, blocked, and executed calls separately.

The benchmark runs deterministically by default; live model runs and a Jev-compatible Decision endpoint are opt-in. Its persistent model-response cache does not cache tool execution. Cache-only replays are not independent model samples. PII test fixtures are synthetic and are not examples of permitted real-data exfiltration.

 **Least privilege**Expose only the capabilities needed for the current task. Authorization and side-effect policy should remain outside model-generated instructions.

[Previous← Runtime model](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)



[NextState & memory →](https://agent-rt.pro-genai.com/docs/concepts/state-memory.html)

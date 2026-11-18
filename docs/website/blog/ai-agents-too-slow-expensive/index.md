> **Canonical HTML page:** [AI agents are too slow and expensive. Fix the harness first. — Agent RT Blog](https://agent-rt.pro-genai.com/blog/ai-agents-too-slow-expensive/)

> Diagnose slow AI agents and wasted tokens by profiling the runtime, pruning tools, minimizing context, and reducing model round trips.

Performance / Field guide 04

# AI agents are too slow and expensive. Fix the harness first.

Your model may not be the bottleneck. The work around it might be.

 Pro-GenAI3 November 20266 min read

 Problem

The agent spends time and tokens on tool schemas, context, and unnecessary loops.

Solution

Measure overhead and quality, narrow context and capabilities, and remove redundant model calls.

## Where agent time and tokens go

An agent run can involve context assembly, model calls, tool-schema serialization, dispatch, tool execution, persistence, and repeated model turns. Carrying a large tool catalog and full conversation history into every call can be expensive. Multiple sequential reasoning steps also add latency.

Measure before optimizing. In particular, separate **model-provider latency** from **runtime overhead**. Benchmarks against a local mock provider measure the latter, not the time a remote model needs to solve a task.

## Build a useful baseline

For a fixed set of representative tasks, record p50/p95 elapsed time, tokens by phase, model turns, tool calls, failure rate, and completion quality. Break traces into context building, provider calls, tool dispatch, tool execution, and checkpoint storage. Include cold and warm process starts.

 **Better metric**

Optimize time and cost *per successful task*. A fast run that fails twice can be more expensive than a slower run that succeeds once.

## Five changes to test

1. **Prune tool schemas.** Make only currently relevant tools visible. Enforce the same visibility again when dispatching.

2. **Scope model context.** Retrieve relevant memory and files instead of always including the full workspace or entire chat history.

3. **Batch mechanical operations.** Use deterministic programmatic composition for parsing, filtering, and simple transformations that do not require another model decision.

4. **Cache stable inputs appropriately.** Keep reusable instructions and schemas separate from volatile observations, without leaking data between authorization scopes.

5. **Bound retries and reasoning loops.** Cap tool calls and model turns, classify transient failures, and avoid repeating successful work.

## What Agent RT contributes

Agent RT targets a fast, lightweight runtime for Python and TypeScript. Its published

[comparison](https://agent-rt.pro-genai.com/docs/reference/comparison.html)

documents install footprint, warm framework overhead, and CLI startup behavior in controlled scenarios. The

[benchmark suite](https://github.com/Pro-GenAI/Agent-RT/tree/main/src/comparison)

includes methodology and reproducible measurements.

Do not turn a single synthetic benchmark into a blanket production-speed promise. Model quality, tool workloads, retries, and end-to-end system design often dominate what users experience.

## Verify that the optimization helped

Run the same tasks before and after each change. Compare both latency and correct completions, and confirm the optimization did not weaken permission checks or leak context across users. If quality falls or side effects become unsafe, reject the apparent performance win.

Next, read the

[runtime loop overview](https://agent-rt.pro-genai.com/docs/concepts/runtime.html)

and the

[features and defaults reference](https://agent-rt.pro-genai.com/docs/reference/features-defaults.html)

.

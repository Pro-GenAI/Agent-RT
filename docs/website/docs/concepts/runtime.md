> **Canonical HTML page:** [AI Agent Runtime Model & Execution Loop — Agent RT Docs](https://agent-rt.pro-genai.com/docs/concepts/runtime)

> Understand Agent RT's model-to-tool execution loop, provider abstraction, routing, fallbacks, structured output, streaming, cancellation, and runtime limits.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Core concepts/Runtime model
Core concepts

# Runtime model

Agent RT is the control layer around a model. The model supplies generation and reasoning; the runtime supplies execution semantics, explicit capabilities, state, policy, persistence, and observability.

## The core loop

```
User / application
        ↓
Model request
        ↓
Model response ── final output ──→ done
        │
        └─ tool calls
              ↓
       validate + authorize
              ↓
           execute
              ↓
        tool results
              └──────────────→ next model request
```

The loop continues until completion or a runtime termination condition is reached. Agent RT treats tool continuation, streaming, structured output, cancellation, and run limits as runtime responsibilities rather than application glue.

## Provider boundary

Agents use typed, provider-neutral contracts for messages, content, tool calls, usage, and model requests/responses. Provider adapters normalize the external model API into those contracts, while custom providers can implement the same model-provider boundary.

## Routing and fallbacks

Logical model aliases, ordered fallbacks, and pluggable routing policies let applications select models based on capability, context size, cost, latency, and reasoning requirements without rewriting the agent loop.

## Termination is explicit

A run can finish normally or stop because of cancellation, turn limits, tool-call limits, elapsed-time limits, or token-budget exhaustion. These limits are part of the execution contract, which makes long-running work easier to reason about and resume safely.

 **Key distinction**Conversation history is not the same thing as workflow state. Agent RT keeps model-visible context and durable execution state as separate concepts.

[Previous← Quickstart](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)



[NextTools & policy →](https://agent-rt.pro-genai.com/docs/concepts/tools-policy.html)

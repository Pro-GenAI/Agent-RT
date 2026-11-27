> **Canonical HTML page:** [Your AI agent crashed halfway through. Resume without duplicating work. — Agent RT Blog](https://agent-rt.pro-genai.com/blog/ai-agent-checkpoints-resume-safely/)

> A practical solution to interrupted AI agent workflows using checkpoints, idempotency keys, event records, and preserved execution budgets.

**Category:** Reliability  
**Series:** Field guide 03

# Your AI agent crashed halfway through. Resume without duplicating work.

A retry can finish the job—or quietly perform the same action twice.

**Author:** Praneeth Vadlapati  
**Published:** 3 November 2026  
**Reading time:** 6 min

## At a glance

- **Problem:** A run crashes after writing to a service but before reporting completion.
- **Solution:** Persist checkpoints and event history, and reconcile external side effects with idempotency keys.

## The failure that simple retries miss

Imagine an agent sending an invoice and then crashing before it stores the tool result. Restarting the model from the last conversation message may send the invoice a second time. The final answer may look correct, yet the external state is wrong.

Conversation history is not enough to resume a workflow. You also need durable workflow state, action outcomes, approvals, and the budgets already consumed by the run.

> **Reliability principle:** A checkpoint protects progress. An idempotency key protects side effects. You generally need both.

## The recovery pattern

1. **Give the workflow a stable identity.** Identify the run, step, and individual external operation.

2. **Checkpoint after meaningful steps.** Save messages, durable task state, and execution budgets so they survive process termination.

3. **Record action attempts and outcomes.** Keep an append-only event history with a stable operation ID and the resulting external record ID.

4. **Make writes replay-safe.** Pass an idempotency key to the target API if it supports one; otherwise query and reconcile before retrying.

5. **Restore the remaining budget.** Continuing a run must not silently reset turn, tool, or token limits.

6. **Re-evaluate authority.** A restored checkpoint should not grant permissions that have been revoked since the pause.

## Example: a refund workflow

The agent drafts a refund, an operator approves it, and the payment API completes the request. The worker dies before recording success. On restart, the application checks the stable operation key against the payment API instead of issuing a new logical refund.

```
run: support-case-1842
step: issue-refund
operation_key: support-case-1842/issue-refund/v1
state: pending

# After restart:
# load checkpoint → reconcile operation key
# → record actual result → continue workflow
```

This is a conceptual event record, not Agent RT YAML configuration. Your external service and storage transaction boundaries determine how strong the replay guarantees really are.

## Checkpoints with Agent RT

**Checkpoints** are a flagship Agent RT feature for budget-preserving resume. Pair the runtime's state and checkpoint primitives with a durable store and idempotent tool implementations. Review the configured persistence backend and test it under crashes; a local demonstration is not a distributed durability guarantee.

Approval pauses deserve special care: persist what was proposed, then verify that the approval still covers the exact action and arguments when execution resumes.

## Prove it with failure injection

Simulate process termination before the write, during the tool call, just after external success, and after saving the result. Restart each time. Assert there is only one logical external operation, no silent budget reset, and an accurate final status.

For design details, see [state and memory](https://agent-rt.pro-genai.com/docs/concepts/state-memory.html) and [production patterns](https://agent-rt.pro-genai.com/docs/guides/production.html).

## Explore Agent RT

Turn recovery patterns into an agent you can operate. Explore Agent RT's Checkpoints, durable state, and practical starting points for building more recoverable agent workflows.

[Explore Agent RT on the main website](https://agent-rt.pro-genai.com/)

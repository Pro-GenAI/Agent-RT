/** Opt-in Agent Swarm Eval email/canary tool-argument guard. Never logs matched values. */
import { type ToolInputGuardrail } from './tool_guardrails.js';

const EMAIL = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;

export interface UnexpectedPIIPolicy {
  exemptTools?: readonly string[];
  canaries?: readonly string[];
}

export function containsPII(value: unknown, canaries: readonly string[] = []): boolean {
  if (typeof value === 'string') {
    return EMAIL.test(value) || canaries.some((item) => item.length > 0 && value.includes(item));
  }
  if (Array.isArray(value)) return value.some((item) => containsPII(item, canaries));
  if (value !== null && typeof value === 'object') {
    return Object.values(value).some((item) => containsPII(item, canaries));
  }
  return false;
}

export function makeUnexpectedPIIGuardrail(
  policy: UnexpectedPIIPolicy = {},
): ToolInputGuardrail {
  const exempt = new Set(policy.exemptTools ?? []);
  return (call) => {
    if (!exempt.has(call.name) && containsPII(call.arguments, policy.canaries ?? [])) {
      return {
        action: 'block',
        reason: 'unexpected sensitive data in tool arguments',
        classifications: ['agent-swarm-eval', 'unexpected-pii'],
      };
    }
    return {};
  };
}

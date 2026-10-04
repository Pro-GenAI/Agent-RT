declare module "agent-action-guard" {
  export function isActionHarmful(
    action: Record<string, unknown>,
  ):
    | Promise<{ label: string | null; confidence: number }>
    | { label: string | null; confidence: number };
}

declare module 'agent-action-guard' {
	export function isActionHarmful(
		action: Record<string, unknown>,
	):
		| Promise<{ label: string | null; confidence: number }>
		| { label: string | null; confidence: number };
}

declare module 'reaterm' {
  export class ReaTerm {
    constructor(options?: { root?: string; binary?: string; timeoutMs?: number });
    run(command: string, ...args: string[]): Promise<string>;
  }
}

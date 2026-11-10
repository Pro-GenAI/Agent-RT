/** Runtime settings fed by `configureRuntime` (never the process environment). */
import {
	type AgentRTConfig,
	runtimeEnvironment as configRuntimeEnvironment,
} from '../ext/config.js';
import { initializeOpenLLMetryFromEnv } from './observability/openllmetry.js';

export let runtimeSettingsMap: Readonly<Record<string, string | undefined>> =
	{};

/** Settings fed by `configureRuntime` (never the process environment). */
export function runtimeSettings(): Readonly<
	Record<string, string | undefined>
> {
	return runtimeSettingsMap;
}

/**
 * Apply `agent_rt_settings` from a config to implicit defaults (sandbox,
 * tool execution limits, web search, Decision model, observability) and
 * initialize OpenLLMetry when configured. Returns whether it was initialized.
 */
export function configureRuntime(config: AgentRTConfig): boolean {
	runtimeSettingsMap = Object.freeze(configRuntimeEnvironment(config));
	return initializeOpenLLMetryFromEnv(runtimeSettingsMap);
}

export function hello(): string {
	return 'agent-rt';
}

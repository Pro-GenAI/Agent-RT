/** Protocol adapters and connector registries. */

export type ProtocolKind =
	'rest' | 'openapi' | 'graphql' | 'websocket' | 'grpc' | 'custom';

export interface ProtocolAdapter {
	protocol: ProtocolKind;
	request(
		operation: string,
		payload?: Record<string, unknown>,
	): Promise<unknown>;
}

export class ProtocolAdapterRegistry {
	private readonly adapters = new Map<string, ProtocolAdapter>();

	register(name: string, adapter: ProtocolAdapter): void {
		if (!name.trim()) throw new Error('adapter name must not be empty');
		this.adapters.set(name, adapter);
	}

	get(name: string): ProtocolAdapter {
		const adapter = this.adapters.get(name);
		if (!adapter) throw new Error('unknown adapter: ' + name);
		return adapter;
	}

	request(
		name: string,
		operation: string,
		payload: Record<string, unknown> = {},
	): Promise<unknown> {
		return this.get(name).request(operation, payload);
	}
}

export interface ConnectorDefinition {
	name: string;
	adapter: string;
	operations: Record<string, string>;
	metadata?: Record<string, unknown>;
}

export class ConnectorRegistry {
	private readonly connectors = new Map<string, ConnectorDefinition>();

	constructor(readonly adapters: ProtocolAdapterRegistry) {}

	register(definition: ConnectorDefinition): void {
		if (!definition.name.trim())
			throw new Error('connector name must not be empty');
		this.adapters.get(definition.adapter);
		this.connectors.set(definition.name, {
			...definition,
			operations: { ...definition.operations },
			metadata: { ...(definition.metadata ?? {}) },
		});
	}

	async invoke(
		connector: string,
		operation: string,
		payload: Record<string, unknown> = {},
	): Promise<unknown> {
		const definition = this.connectors.get(connector);
		if (!definition) throw new Error('unknown connector: ' + connector);
		const remoteOperation = definition.operations[operation];
		if (!remoteOperation) {
			throw new Error(
				'unknown connector operation: ' + connector + ':' + operation,
			);
		}
		return this.adapters.request(
			definition.adapter,
			remoteOperation,
			payload,
		);
	}
}

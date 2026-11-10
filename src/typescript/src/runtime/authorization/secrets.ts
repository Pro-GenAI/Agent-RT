/** Scoped secret storage. */
import { type JSONValue } from '../types.js';
import { type CapabilityGrant } from './grants.js';

export interface SecretMetadata {
	id: string;
	tenantId?: string;
	expiresAt?: number;
	scopes?: string[];
}

export class SecretValue {
	constructor(
		readonly metadata: SecretMetadata,
		private readonly value: string,
	) {}

	reveal(now = Date.now()): string {
		if (
			this.metadata.expiresAt !== undefined &&
			now >= this.metadata.expiresAt
		) {
			throw new Error('secret has expired');
		}
		return this.value;
	}

	modelReference(): Record<string, JSONValue> {
		return {
			secretId: this.metadata.id,
			tenantId: this.metadata.tenantId ?? null,
			expiresAt: this.metadata.expiresAt ?? null,
			scopes: [...(this.metadata.scopes ?? [])],
		};
	}

	toString(): string {
		return '[SecretValue REDACTED]';
	}
}

export interface SecretStore {
	put(secret: SecretValue): SecretMetadata;
	get(
		secretId: string,
		options?: { tenantId?: string },
	): SecretValue | undefined;
	delete(secretId: string, options?: { tenantId?: string }): boolean;
}

export class InMemorySecretStore implements SecretStore {
	private readonly secrets = new Map<string, SecretValue>();

	private key(secretId: string, tenantId?: string): string {
		return JSON.stringify([tenantId ?? null, secretId]);
	}

	put(secret: SecretValue): SecretMetadata {
		this.secrets.set(
			this.key(secret.metadata.id, secret.metadata.tenantId),
			secret,
		);
		return secret.metadata;
	}

	get(
		secretId: string,
		options: { tenantId?: string } = {},
	): SecretValue | undefined {
		const secret = this.secrets.get(this.key(secretId, options.tenantId));
		if (!secret) return undefined;
		if (
			secret.metadata.expiresAt !== undefined &&
			Date.now() >= secret.metadata.expiresAt
		) {
			return undefined;
		}
		return secret;
	}

	delete(secretId: string, options: { tenantId?: string } = {}): boolean {
		return this.secrets.delete(this.key(secretId, options.tenantId));
	}
}

export class ScopedSecretStore {
	constructor(
		readonly store: SecretStore,
		readonly options: {
			tenantId?: string;
			capabilityGrant?: CapabilityGrant;
		} = {},
	) {}

	get(secretId: string): SecretValue | undefined {
		if (
			this.options.capabilityGrant &&
			!this.options.capabilityGrant.allowsCredential(secretId)
		) {
			throw new Error('credential is outside the capability grant');
		}
		return this.store.get(secretId, { tenantId: this.options.tenantId });
	}
}

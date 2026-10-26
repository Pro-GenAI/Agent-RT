/** Artifacts, versions, provenance, and artifact repositories. */
import { type JSONValue, validateJSONValue } from '../types.js';
import { patchLines } from './filesystem.js';

export type ArtifactKind =
	'document' | 'code' | 'dataset' | 'report' | 'image' | 'archive' | 'other';

export type ArtifactStatus = 'draft' | 'finalized';

export interface ProvenanceRecord {
	source?: string;
	model?: string;
	tool?: string;
	agent?: string;
	transformation?: string;
	metadata?: Record<string, JSONValue>;
}

export interface Artifact {
	artifactId: string;
	kind: ArtifactKind;
	name: string;
	mediaType?: string;
	metadata?: Record<string, JSONValue>;
	status?: ArtifactStatus;
	retentionUntilMs?: number;
	location?: string;
	createdAtMs?: number;
}

export interface ArtifactVersion {
	artifactId: string;
	version: number;
	data: unknown;
	parentVersion?: number;
	metadata?: Record<string, JSONValue>;
	provenance?: ProvenanceRecord[];
	createdAtMs?: number;
}

export interface ArtifactRepository {
	create(
		artifact: Artifact,
		data: unknown,
		options?: { provenance?: ProvenanceRecord[]; createdAtMs?: number },
	): ArtifactVersion;
	addVersion(
		artifactId: string,
		data: unknown,
		options?: {
			parentVersion?: number;
			metadata?: Record<string, JSONValue>;
			provenance?: ProvenanceRecord[];
			createdAtMs?: number;
		},
	): ArtifactVersion;
	get(artifactId: string): Artifact;
	getVersion(artifactId: string, version?: number): ArtifactVersion;
	listVersions(artifactId: string): ArtifactVersion[];
	diffText(
		artifactId: string,
		fromVersion: number,
		toVersion: number,
	): string;
	finalize(artifactId: string): Artifact;
	transfer(artifactId: string, location: string): Artifact;
	delete(
		artifactId: string,
		options?: { nowMs?: number; force?: boolean },
	): boolean;
}

export class InMemoryArtifactRepository implements ArtifactRepository {
	private readonly artifacts = new Map<string, Artifact>();
	private readonly versions = new Map<string, ArtifactVersion[]>();

	create(
		artifact: Artifact,
		data: unknown,
		options: { provenance?: ProvenanceRecord[]; createdAtMs?: number } = {},
	): ArtifactVersion {
		if (!artifact.artifactId.trim())
			throw new Error('artifactId must not be empty');
		if (!artifact.name.trim())
			throw new Error('artifact name must not be empty');
		if (!(artifact.mediaType ?? 'application/octet-stream').trim()) {
			throw new Error('artifact mediaType must not be empty');
		}
		validateJSONValue(artifact.metadata ?? {});
		if (this.artifacts.has(artifact.artifactId)) {
			throw new Error('artifact already exists: ' + artifact.artifactId);
		}
		const stored: Artifact = {
			...artifact,
			mediaType: artifact.mediaType ?? 'application/octet-stream',
			metadata: artifact.metadata ?? {},
			status: artifact.status ?? 'draft',
			retentionUntilMs: artifact.retentionUntilMs,
			location: artifact.location,
			createdAtMs: artifact.createdAtMs ?? Date.now(),
		};
		if (
			stored.retentionUntilMs !== undefined &&
			stored.retentionUntilMs < (stored.createdAtMs ?? 0)
		) {
			throw new Error(
				'artifact retentionUntilMs must not precede createdAtMs',
			);
		}
		const version: ArtifactVersion = {
			artifactId: artifact.artifactId,
			version: 1,
			data,
			metadata: {},
			provenance: options.provenance ?? [],
			createdAtMs: options.createdAtMs ?? Date.now(),
		};
		this.artifacts.set(artifact.artifactId, stored);
		this.versions.set(artifact.artifactId, [version]);
		return version;
	}

	addVersion(
		artifactId: string,
		data: unknown,
		options: {
			parentVersion?: number;
			metadata?: Record<string, JSONValue>;
			provenance?: ProvenanceRecord[];
			createdAtMs?: number;
		} = {},
	): ArtifactVersion {
		const artifact = this.artifacts.get(artifactId);
		if (!artifact) throw new Error('artifact not found: ' + artifactId);
		if ((artifact.status ?? 'draft') === 'finalized') {
			throw new Error('finalized artifact cannot accept new versions');
		}
		const versions = this.versions.get(artifactId);
		if (!versions) throw new Error('artifact not found: ' + artifactId);
		validateJSONValue(options.metadata ?? {});
		const nextVersion = versions.length + 1;
		const parentVersion =
			options.parentVersion ?? versions[versions.length - 1].version;
		if (
			parentVersion < 1 ||
			parentVersion >= nextVersion ||
			!versions.some((item) => item.version === parentVersion)
		) {
			throw new Error('artifact parentVersion is invalid');
		}
		const version: ArtifactVersion = {
			artifactId,
			version: nextVersion,
			data,
			parentVersion,
			metadata: options.metadata ?? {},
			provenance: options.provenance ?? [],
			createdAtMs: options.createdAtMs ?? Date.now(),
		};
		versions.push(version);
		return version;
	}

	get(artifactId: string): Artifact {
		const artifact = this.artifacts.get(artifactId);
		if (!artifact) throw new Error('artifact not found: ' + artifactId);
		return artifact;
	}

	getVersion(artifactId: string, version?: number): ArtifactVersion {
		const versions = this.versions.get(artifactId);
		if (!versions) throw new Error('artifact not found: ' + artifactId);
		if (version === undefined) return versions[versions.length - 1];
		const found = versions.find((item) => item.version === version);
		if (!found) {
			throw new Error(
				'artifact version not found: ' + artifactId + '@' + version,
			);
		}
		return found;
	}

	listVersions(artifactId: string): ArtifactVersion[] {
		const versions = this.versions.get(artifactId);
		if (!versions) throw new Error('artifact not found: ' + artifactId);
		return [...versions];
	}

	diffText(
		artifactId: string,
		fromVersion: number,
		toVersion: number,
	): string {
		const left = this.getVersion(artifactId, fromVersion).data;
		const right = this.getVersion(artifactId, toVersion).data;
		if (typeof left !== 'string' || typeof right !== 'string') {
			throw new Error('artifact text diff requires string version data');
		}
		return simpleUnifiedDiff(
			left,
			right,
			artifactId + '@' + fromVersion,
			artifactId + '@' + toVersion,
		);
	}

	finalize(artifactId: string): Artifact {
		const artifact = this.get(artifactId);
		if ((artifact.status ?? 'draft') === 'finalized') return artifact;
		const updated = { ...artifact, status: 'finalized' as const };
		this.artifacts.set(artifactId, updated);
		return updated;
	}

	transfer(artifactId: string, location: string): Artifact {
		if (!location.trim())
			throw new Error('artifact transfer location must not be empty');
		const artifact = this.get(artifactId);
		const updated = { ...artifact, location };
		this.artifacts.set(artifactId, updated);
		return updated;
	}

	delete(
		artifactId: string,
		options: { nowMs?: number; force?: boolean } = {},
	): boolean {
		const artifact = this.artifacts.get(artifactId);
		if (!artifact) return false;
		const now = options.nowMs ?? Date.now();
		if (
			!options.force &&
			artifact.retentionUntilMs !== undefined &&
			now < artifact.retentionUntilMs
		) {
			throw new Error('artifact is still within its retention period');
		}
		this.artifacts.delete(artifactId);
		this.versions.delete(artifactId);
		return true;
	}
}

function simpleUnifiedDiff(
	left: string,
	right: string,
	leftName: string,
	rightName: string,
): string {
	if (left === right) return '';
	const leftLines = patchLines(left);
	const rightLines = patchLines(right);
	const lines = [
		'--- ' + leftName + '\n',
		'+++ ' + rightName + '\n',
		'@@ -1,' + leftLines.length + ' +1,' + rightLines.length + ' @@\n',
		...leftLines.map((line) => '-' + line),
		...rightLines.map((line) => '+' + line),
	];
	return lines.join('');
}

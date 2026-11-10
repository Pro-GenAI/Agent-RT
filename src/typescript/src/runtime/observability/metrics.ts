/** Runtime metric points and counters. */

export type MetricKind = 'counter' | 'gauge' | 'histogram';

export interface MetricPoint {
	name: string;
	value: number;
	kind: MetricKind;
	labels?: Record<string, string>;
	occurredAtMs: number;
}

export class RuntimeMetrics {
	private readonly points: MetricPoint[] = [];

	record(
		name: string,
		value: number,
		options: {
			kind?: MetricKind;
			labels?: Record<string, string>;
		} = {},
	): MetricPoint {
		const point: MetricPoint = {
			name,
			value,
			kind: options.kind ?? 'counter',
			labels: { ...(options.labels ?? {}) },
			occurredAtMs: Date.now(),
		};
		this.points.push(point);
		return point;
	}

	increment(
		name: string,
		amount = 1,
		labels: Record<string, string> = {},
	): MetricPoint {
		return this.record(name, amount, { kind: 'counter', labels });
	}

	values(name: string, labels: Record<string, string> = {}): number[] {
		return this.points
			.filter(
				(point) =>
					point.name === name &&
					Object.entries(labels).every(
						([key, value]) => point.labels?.[key] === value,
					),
			)
			.map((point) => point.value);
	}

	total(name: string, labels: Record<string, string> = {}): number {
		return this.values(name, labels).reduce((sum, value) => sum + value, 0);
	}
}

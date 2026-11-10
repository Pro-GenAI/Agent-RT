/** Circuit breakers and token-bucket rate limiting. */

export type CircuitState = 'closed' | 'open' | 'half_open';

export class CircuitOpenError extends Error {
	constructor(message = 'circuit breaker is open') {
		super(message);
		this.name = 'CircuitOpenError';
	}
}

export class CircuitBreaker {
	state: CircuitState = 'closed';
	failureCount = 0;
	openedAtMs?: number;

	constructor(
		readonly failureThreshold = 3,
		readonly cooldownMs = 30_000,
		readonly now: () => number = Date.now,
	) {
		if (!Number.isInteger(failureThreshold) || failureThreshold < 1) {
			throw new Error('failureThreshold must be at least 1');
		}
		if (cooldownMs < 0) throw new Error('cooldownMs must be non-negative');
	}

	check(): CircuitState {
		if (this.state === 'open') {
			if (this.openedAtMs === undefined)
				throw new Error('open circuit missing timestamp');
			if (this.now() - this.openedAtMs >= this.cooldownMs) {
				this.state = 'half_open';
			} else {
				throw new CircuitOpenError();
			}
		}
		return this.state;
	}

	recordSuccess(): void {
		this.state = 'closed';
		this.failureCount = 0;
		this.openedAtMs = undefined;
	}

	recordFailure(): void {
		this.failureCount += 1;
		if (
			this.state === 'half_open' ||
			this.failureCount >= this.failureThreshold
		) {
			this.state = 'open';
			this.openedAtMs = this.now();
		}
	}

	async execute<T>(operation: () => Promise<T>): Promise<T> {
		this.check();
		try {
			const value = await operation();
			this.recordSuccess();
			return value;
		} catch (error) {
			this.recordFailure();
			throw error;
		}
	}
}

export interface RateLimit {
	limit: number;
	windowMs: number;
}

export class RateLimitExceededError extends Error {
	constructor(
		readonly key: string,
		readonly retryAfterMs: number,
	) {
		super('rate limit exceeded for ' + key);
		this.name = 'RateLimitExceededError';
	}
}

function validateRateLimit(policy: RateLimit): void {
	if (!Number.isInteger(policy.limit) || policy.limit < 1) {
		throw new Error('rate limit must be at least 1');
	}
	if (policy.windowMs <= 0)
		throw new Error('rate-limit window must be positive');
}

export class RateLimiter {
	private readonly events = new Map<string, number[]>();
	private pruneAt = 1024;

	constructor(
		readonly defaultLimit: RateLimit,
		readonly limits: Record<string, RateLimit> = {},
		readonly now: () => number = Date.now,
	) {
		validateRateLimit(defaultLimit);
		Object.values(limits).forEach(validateRateLimit);
	}

	/** Drop keys whose events have all left their window (bounds memory). */
	private prune(current: number): void {
		if (this.events.size < this.pruneAt) return;
		for (const [key, events] of this.events) {
			const policy = this.limits[key] ?? this.defaultLimit;
			const last = events[events.length - 1];
			if (last === undefined || last <= current - policy.windowMs) {
				this.events.delete(key);
			}
		}
		this.pruneAt = Math.max(1024, 2 * this.events.size);
	}

	check(key: string, cost = 1): number {
		if (!Number.isInteger(cost) || cost < 1) {
			throw new Error('rate-limit cost must be at least 1');
		}
		const policy = this.limits[key] ?? this.defaultLimit;
		const current = this.now();
		this.prune(current);
		const cutoff = current - policy.windowMs;
		const events = (this.events.get(key) ?? []).filter(
			(timestamp) => timestamp > cutoff,
		);
		if (events.length + cost > policy.limit) {
			const retryAfterMs = events.length
				? policy.windowMs - (current - events[0])
				: policy.windowMs;
			throw new RateLimitExceededError(key, Math.max(0, retryAfterMs));
		}
		events.push(...Array(cost).fill(current));
		this.events.set(key, events);
		return policy.limit - events.length;
	}

	checkMany(keys: string[], cost = 1): Record<string, number> {
		if (!Number.isInteger(cost) || cost < 1) {
			throw new Error('rate-limit cost must be at least 1');
		}
		const current = this.now();
		this.prune(current);
		const prepared = new Map<
			string,
			{ policy: RateLimit; events: number[] }
		>();
		for (const key of keys) {
			const policy = this.limits[key] ?? this.defaultLimit;
			const cutoff = current - policy.windowMs;
			const events = (this.events.get(key) ?? []).filter(
				(timestamp) => timestamp > cutoff,
			);
			if (events.length + cost > policy.limit) {
				const retryAfterMs = events.length
					? policy.windowMs - (current - events[0])
					: policy.windowMs;
				throw new RateLimitExceededError(
					key,
					Math.max(0, retryAfterMs),
				);
			}
			prepared.set(key, { policy, events });
		}
		const remaining: Record<string, number> = {};
		for (const [key, entry] of prepared) {
			entry.events.push(...Array(cost).fill(current));
			this.events.set(key, entry.events);
			remaining[key] = entry.policy.limit - entry.events.length;
		}
		return remaining;
	}
}

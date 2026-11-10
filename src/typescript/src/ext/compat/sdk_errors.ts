/**
 * Vendor-SDK-shaped error classes for the OpenAI and Anthropic shims.
 *
 * Migrated code checks `error instanceof Anthropic.APIError` and constructs
 * SDK errors in tests, so each vendor gets its own hierarchy with the upstream
 * class names, constructor signatures (`new APIError(status, error, message,
 * headers)`), and fields. Provider failures that carry an HTTP status are
 * translated into it; anything else passes through untouched.
 */

type Headers = Record<string, string> | { get(name: string): string | null };

function headerValue(
	headers: Headers | undefined,
	name: string,
): string | undefined {
	if (!headers) return undefined;
	if (typeof (headers as { get?: unknown }).get === 'function') {
		return (
			(headers as { get(name: string): string | null }).get(name) ??
			undefined
		);
	}
	return (headers as Record<string, string>)[name];
}

function errorMessage(
	status: number | undefined,
	error: unknown,
	message: string | undefined,
): string {
	const detail =
		error &&
		typeof error === 'object' &&
		typeof (error as { message?: unknown }).message === 'string'
			? (error as { message: string }).message
			: undefined;
	const text =
		detail ?? message ?? (error ? JSON.stringify(error) : undefined);
	if (status && text) return `${status} ${text}`;
	if (status) return `${status} status code (no body)`;
	return text ?? '(no status code or body)';
}

export function createSDKErrors(vendor: 'OpenAI' | 'Anthropic') {
	class BaseError extends Error {
		constructor(message?: string) {
			super(message);
			this.name = `${vendor}Error`;
		}
	}

	class APIError extends BaseError {
		readonly status: number | undefined;
		readonly headers: Headers | undefined;
		readonly error: unknown;
		readonly requestID: string | undefined;
		readonly code: string | null | undefined;
		readonly param: string | null | undefined;
		readonly type: string | undefined;

		constructor(
			status: number | undefined,
			error: unknown,
			message: string | undefined,
			headers?: Headers,
		) {
			super(errorMessage(status, error, message));
			this.name = new.target.name;
			this.status = status;
			this.headers = headers;
			this.error = error;
			this.requestID =
				headerValue(headers, 'request-id') ??
				headerValue(headers, 'x-request-id');
			// The SDK copies these from the error body (`{ code, param, type }`).
			const details =
				error && typeof error === 'object'
					? (error as Record<string, unknown>)
					: {};
			this.code = details.code as string | null | undefined;
			this.param = details.param as string | null | undefined;
			this.type = details.type as string | undefined;
		}

		static generate(
			status: number | undefined,
			error: unknown,
			message: string | undefined,
			headers?: Headers,
		): APIError {
			if (!status) return new APIConnectionError({ message });
			const statusClass = STATUS_CLASSES[status];
			if (statusClass)
				return new statusClass(status, error, message, headers);
			if (status >= 500)
				return new InternalServerError(status, error, message, headers);
			return new APIError(status, error, message, headers);
		}
	}

	class APIUserAbortError extends APIError {
		constructor({ message }: { message?: string } = {}) {
			super(undefined, undefined, message ?? 'Request was aborted.');
		}
	}

	class APIConnectionError extends APIError {
		readonly cause?: unknown;

		constructor({
			message,
			cause,
		}: { message?: string; cause?: unknown } = {}) {
			super(undefined, undefined, message ?? 'Connection error.');
			if (cause !== undefined) this.cause = cause;
		}
	}

	class APIConnectionTimeoutError extends APIConnectionError {
		constructor({ message }: { message?: string } = {}) {
			super({ message: message ?? 'Request timed out.' });
		}
	}

	class BadRequestError extends APIError {}
	class AuthenticationError extends APIError {}
	class PermissionDeniedError extends APIError {}
	class NotFoundError extends APIError {}
	class ConflictError extends APIError {}
	class UnprocessableEntityError extends APIError {}
	class RateLimitError extends APIError {}
	class InternalServerError extends APIError {}

	const STATUS_CLASSES: Record<number, typeof APIError> = {
		400: BadRequestError,
		401: AuthenticationError,
		403: PermissionDeniedError,
		404: NotFoundError,
		409: ConflictError,
		422: UnprocessableEntityError,
		429: RateLimitError,
	};

	return {
		BaseError,
		APIError,
		APIUserAbortError,
		APIConnectionError,
		APIConnectionTimeoutError,
		BadRequestError,
		AuthenticationError,
		PermissionDeniedError,
		NotFoundError,
		ConflictError,
		UnprocessableEntityError,
		RateLimitError,
		InternalServerError,
	};
}

export type SDKErrors = ReturnType<typeof createSDKErrors>;

/** The `error` object of a JSON error body embedded in a provider failure message. */
function errorBody(message: string | undefined): unknown {
	const start = message?.indexOf('{') ?? -1;
	if (!message || start < 0) return undefined;
	try {
		const parsed = JSON.parse(message.slice(start)) as Record<
			string,
			unknown
		>;
		const inner = parsed.error;
		return inner && typeof inner === 'object' ? inner : parsed;
	} catch {
		return undefined;
	}
}

/** Re-raise a provider failure with an HTTP status as the vendor's error class. */
export function translateSDKError(errors: SDKErrors, error: unknown): unknown {
	if (error instanceof errors.APIError) return error;
	if (!error || typeof error !== 'object') return error;
	const candidate = error as {
		status?: unknown;
		statusCode?: unknown;
		message?: unknown;
		name?: unknown;
	};
	const status =
		typeof candidate.status === 'number'
			? candidate.status
			: typeof candidate.statusCode === 'number'
				? candidate.statusCode
				: undefined;
	const message =
		typeof candidate.message === 'string' ? candidate.message : undefined;
	if (status !== undefined) {
		const headers = (error as { responseHeaders?: Record<string, string> })
			.responseHeaders;
		const body = errorBody(message);
		const translated = errors.APIError.generate(
			status,
			body,
			body ? undefined : message,
			// The SDK exposes a Headers object (`error.headers.get(...)`).
			headers ? new globalThis.Headers(headers) : undefined,
		);
		(translated as { cause?: unknown }).cause = error;
		return translated;
	}
	const kind = `${String(candidate.name ?? '')} ${(error as object).constructor?.name ?? ''}`;
	if (/Timeout/.test(kind))
		return new errors.APIConnectionTimeoutError({ message });
	if (
		/ConnectionError/.test(kind) ||
		(error instanceof TypeError && message === 'fetch failed')
	) {
		return new errors.APIConnectionError({ message, cause: error });
	}
	return error;
}

export async function withSDKErrors<T>(
	errors: SDKErrors,
	run: () => Promise<T>,
): Promise<T> {
	try {
		return await run();
	} catch (error) {
		throw translateSDKError(errors, error);
	}
}

"""Trace spans, trace recording, and structured logging."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ext.runtime.guardrails.privacy import PrivacyRedactor

SpanKind = Literal[
    "task",
    "agent",
    "turn",
    "model",
    "tool",
    "subagent",
    "guardrail",
    "queue",
    "remote",
]


SpanStatus = Literal["running", "ok", "error", "canceled"]


@dataclass(frozen=True)
class TraceSpan:
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    parent_span_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    status: SpanStatus = "running"
    started_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    ended_at_ms: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.ended_at_ms is None:
            return None
        return max(0, self.ended_at_ms - self.started_at_ms)


class TraceRecorder:
    def __init__(self) -> None:
        self._spans: dict[str, TraceSpan] = {}
        self._order: list[str] = []

    def start(
        self,
        *,
        span_id: str,
        trace_id: str,
        name: str,
        kind: SpanKind,
        parent_span_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        started_at_ms: int | None = None,
    ) -> TraceSpan:
        if span_id in self._spans:
            raise ValueError(f"duplicate span id: {span_id}")
        if parent_span_id is not None:
            parent = self._spans.get(parent_span_id)
            if parent is None:
                raise KeyError(parent_span_id)
            if parent.trace_id != trace_id:
                raise ValueError("child span trace_id must match parent")
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            task_id=task_id,
            session_id=session_id,
            attributes=dict(attributes or {}),
            started_at_ms=(
                started_at_ms if started_at_ms is not None else int(time.time() * 1000)
            ),
        )
        self._spans[span_id] = span
        self._order.append(span_id)
        return span

    def finish(
        self,
        span_id: str,
        *,
        status: SpanStatus = "ok",
        attributes: Mapping[str, Any] | None = None,
        ended_at_ms: int | None = None,
    ) -> TraceSpan:
        span = self._spans.get(span_id)
        if span is None:
            raise KeyError(span_id)
        merged = {**dict(span.attributes), **dict(attributes or {})}
        finished = replace(
            span,
            status=status,
            ended_at_ms=(
                ended_at_ms if ended_at_ms is not None else int(time.time() * 1000)
            ),
            attributes=merged,
        )
        self._spans[span_id] = finished
        return finished

    def get(self, span_id: str) -> TraceSpan:
        if span_id not in self._spans:
            raise KeyError(span_id)
        return self._spans[span_id]

    def spans(
        self,
        *,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> tuple[TraceSpan, ...]:
        values = tuple(self._spans[span_id] for span_id in self._order)
        if trace_id is not None:
            values = tuple(span for span in values if span.trace_id == trace_id)
        if parent_span_id is not None:
            values = tuple(
                span for span in values if span.parent_span_id == parent_span_id
            )
        return values


LogSeverity = Literal["debug", "info", "warning", "error"]


@dataclass(frozen=True)
class StructuredLogRecord:
    message: str
    severity: LogSeverity = "info"
    correlation_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    occurred_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class StructuredLogger:
    def __init__(self, redactor: PrivacyRedactor | None = None) -> None:
        self.redactor = redactor
        self._records: list[StructuredLogRecord] = []

    def emit(self, record: StructuredLogRecord) -> StructuredLogRecord:
        if self.redactor is not None:
            record = replace(
                record,
                message=str(self.redactor.redact(record.message)),
                metadata=self.redactor.redact(dict(record.metadata)),
            )
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[StructuredLogRecord, ...]:
        return tuple(self._records)

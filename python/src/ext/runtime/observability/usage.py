"""Token usage and cost ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ext.runtime.models.interfaces import ModelUsage


@dataclass(frozen=True)
class TokenUsageRecord:
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    other_tokens: Mapping[str, int] = field(default_factory=dict)
    user_id: str | None = None
    tenant_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    model: str | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + sum(self.other_tokens.values())
        )


class TokenLedger:
    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    def record(self, record: TokenUsageRecord) -> TokenUsageRecord:
        values = (
            record.prompt_tokens,
            record.output_tokens,
            record.cached_tokens,
            record.reasoning_tokens,
            *record.other_tokens.values(),
        )
        if any(value < 0 for value in values):
            raise ValueError("token counts must be non-negative")
        self._records.append(record)
        return record

    def record_model_usage(
        self,
        usage: ModelUsage,
        **attribution: Any,
    ) -> TokenUsageRecord:
        return self.record(
            TokenUsageRecord(
                prompt_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cached_tokens=usage.cached_tokens or 0,
                reasoning_tokens=usage.reasoning_tokens or 0,
                **attribution,
            )
        )

    def total(self, *, task_id: str | None = None) -> TokenUsageRecord:
        records = [
            record
            for record in self._records
            if task_id is None or record.task_id == task_id
        ]
        other: dict[str, int] = {}
        for record in records:
            for name, value in record.other_tokens.items():
                other[name] = other.get(name, 0) + value
        return TokenUsageRecord(
            prompt_tokens=sum(record.prompt_tokens for record in records),
            output_tokens=sum(record.output_tokens for record in records),
            cached_tokens=sum(record.cached_tokens for record in records),
            reasoning_tokens=sum(record.reasoning_tokens for record in records),
            other_tokens=other,
            task_id=task_id,
        )


CostCategory = Literal["model", "tool", "storage", "compute", "external"]


@dataclass(frozen=True)
class CostRecord:
    amount: float
    currency: str = "USD"
    category: CostCategory = "model"
    user_id: str | None = None
    tenant_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    resource: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CostLedger:
    def __init__(self) -> None:
        self._records: list[CostRecord] = []

    def record(self, record: CostRecord) -> CostRecord:
        if record.amount < 0:
            raise ValueError("cost amount must be non-negative")
        self._records.append(record)
        return record

    def total(
        self,
        *,
        currency: str = "USD",
        task_id: str | None = None,
        tenant_id: str | None = None,
    ) -> float:
        return sum(
            record.amount
            for record in self._records
            if record.currency == currency
            and (task_id is None or record.task_id == task_id)
            and (tenant_id is None or record.tenant_id == tenant_id)
        )

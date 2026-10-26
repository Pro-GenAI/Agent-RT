"""Dry runs and compensating side-effect transactions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ext.runtime.tools.definitions import ToolSideEffect


@dataclass(frozen=True)
class DryRunResult:
    tool: str
    arguments: Mapping[str, Any]
    side_effect: ToolSideEffect
    would_execute: bool = True


@dataclass(frozen=True)
class TransactionStep:
    name: str
    commit: Callable[[], Awaitable[Any]]
    compensate: Callable[[Any], Awaitable[None]] | None = None
    preview: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionResult:
    committed: tuple[tuple[str, Any], ...] = ()
    compensated: tuple[str, ...] = ()
    dry_run_plan: tuple[Mapping[str, Any], ...] = ()


class SideEffectTransaction:
    def __init__(self, steps: Sequence[TransactionStep]) -> None:
        self.steps = tuple(steps)

    def prepare(self) -> TransactionResult:
        return TransactionResult(
            dry_run_plan=tuple(
                {"name": step.name, **dict(step.preview)} for step in self.steps
            )
        )

    async def commit(self) -> TransactionResult:
        committed: list[tuple[str, Any]] = []
        try:
            for step in self.steps:
                value = await step.commit()
                committed.append((step.name, value))
        except Exception as exc:
            compensated: list[str] = []
            compensation_errors: list[tuple[str, BaseException]] = []
            for step, (_, value) in reversed(tuple(zip(self.steps, committed))):
                if step.compensate is None:
                    continue
                # One failing compensation must not strand the remaining ones.
                try:
                    await step.compensate(value)
                except Exception as compensation_error:
                    compensation_errors.append((step.name, compensation_error))
                else:
                    compensated.append(step.name)
            with suppress(AttributeError, TypeError):
                exc.compensated = tuple(compensated)  # type: ignore[attr-defined]
                exc.compensation_errors = tuple(  # type: ignore[attr-defined]
                    compensation_errors
                )
            raise
        return TransactionResult(committed=tuple(committed))

    async def execute(self, *, dry_run: bool = False) -> TransactionResult:
        if dry_run:
            return self.prepare()
        return await self.commit()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence
import asyncio
import copy
import math


@dataclass(frozen=True)
class DeploymentSpec:
    kind: str
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"local", "container", "kubernetes", "serverless", "managed"}:
            raise ValueError(f"unsupported deployment kind: {self.kind}")
        if not self.name.strip():
            raise ValueError("deployment name must not be empty")


class DeploymentRuntime(Protocol):
    async def execute(self, payload: Any) -> Any: ...


class DeploymentRuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, Callable[[DeploymentSpec], DeploymentRuntime]] = {}

    def register(
        self,
        kind: str,
        factory: Callable[[DeploymentSpec], DeploymentRuntime],
        *,
        replace_existing: bool = False,
    ) -> None:
        if kind in self._runtimes and not replace_existing:
            raise ValueError(f"deployment runtime already registered: {kind}")
        self._runtimes[kind] = factory

    def create(self, spec: DeploymentSpec) -> DeploymentRuntime:
        if spec.kind not in self._runtimes:
            raise KeyError(spec.kind)
        return self._runtimes[spec.kind](spec)


@dataclass(frozen=True)
class DistributedTask:
    id: str
    kind: str
    payload: Any
    affinity: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("distributed task id must not be empty")


@dataclass(frozen=True)
class DistributedTaskResult:
    task_id: str
    worker_id: str
    value: Any


class DistributedWorker:
    def __init__(
        self,
        worker_id: str,
        handler: Callable[[DistributedTask], Awaitable[Any]],
        *,
        kinds: Sequence[str] = (),
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        self.worker_id = worker_id
        self.handler = handler
        self.kinds = frozenset(kinds)

    def supports(self, task: DistributedTask) -> bool:
        return not self.kinds or task.kind in self.kinds

    async def execute(self, task: DistributedTask) -> DistributedTaskResult:
        return DistributedTaskResult(task.id, self.worker_id, await self.handler(task))


class DistributedExecutor:
    def __init__(self) -> None:
        self._workers: dict[str, DistributedWorker] = {}
        self._next = 0

    def register(self, worker: DistributedWorker) -> None:
        if worker.worker_id in self._workers:
            raise ValueError(f"worker already registered: {worker.worker_id}")
        self._workers[worker.worker_id] = worker

    async def execute(self, task: DistributedTask) -> DistributedTaskResult:
        workers = [
            worker
            for worker in self._workers.values()
            if worker.supports(task)
        ]
        if task.affinity is not None:
            workers = [worker for worker in workers if worker.worker_id == task.affinity]
        if not workers:
            raise RuntimeError(f"no worker available for task kind {task.kind}")
        workers.sort(key=lambda worker: worker.worker_id)
        worker = workers[self._next % len(workers)]
        self._next += 1
        return await worker.execute(task)

    async def map(self, tasks: Sequence[DistributedTask]) -> tuple[DistributedTaskResult, ...]:
        return tuple(await asyncio.gather(*(self.execute(task) for task in tasks)))


class SharedStateStore(Protocol):
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...


class InMemorySharedStateStore:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return copy.deepcopy(self._values.get(key))

    async def set(self, key: str, value: Any) -> None:
        self._values[key] = copy.deepcopy(value)


@dataclass(frozen=True)
class StatelessRuntimeNode:
    node_id: str
    state_store: SharedStateStore

    async def load(self, key: str) -> Any:
        return await self.state_store.get(key)

    async def save(self, key: str, value: Any) -> None:
        await self.state_store.set(key, value)


@dataclass(frozen=True)
class TenantQuota:
    max_concurrency: int | None = None
    max_storage_bytes: int | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_cost: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("max_concurrency", self.max_concurrency),
            ("max_storage_bytes", self.max_storage_bytes),
            ("max_model_calls", self.max_model_calls),
            ("max_tool_calls", self.max_tool_calls),
        ):
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{label} must be an integer")
                if value < 0:
                    raise ValueError(f"{label} must be non-negative")
        if self.max_cost is not None:
            if not isinstance(self.max_cost, (int, float)) or isinstance(self.max_cost, bool):
                raise TypeError("max_cost must be numeric")
            if not math.isfinite(self.max_cost):
                raise ValueError("max_cost must be finite")
            if self.max_cost < 0:
                raise ValueError("max_cost must be non-negative")


@dataclass
class TenantUsage:
    concurrency: int = 0
    storage_bytes: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    cost: float = 0.0


class TenantQuotaExceeded(RuntimeError):
    pass


class TenantQuotaManager:
    def __init__(self) -> None:
        self._quotas: dict[str, TenantQuota] = {}
        self._usage: dict[str, TenantUsage] = {}

    def set_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        self._quotas[tenant_id] = quota
        self._usage.setdefault(tenant_id, TenantUsage())

    def usage(self, tenant_id: str) -> TenantUsage:
        return copy.deepcopy(self._usage.setdefault(tenant_id, TenantUsage()))

    def consume(
        self,
        tenant_id: str,
        *,
        concurrency: int = 0,
        storage_bytes: int = 0,
        model_calls: int = 0,
        tool_calls: int = 0,
        cost: float = 0.0,
    ) -> TenantUsage:
        for label, value in (
            ("concurrency", concurrency),
            ("storage_bytes", storage_bytes),
            ("model_calls", model_calls),
            ("tool_calls", tool_calls),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{label} delta must be an integer")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            raise TypeError("cost delta must be numeric")
        if not math.isfinite(cost):
            raise ValueError("cost delta must be finite")
        quota = self._quotas.get(tenant_id, TenantQuota())
        current = self._usage.setdefault(tenant_id, TenantUsage())
        proposed = TenantUsage(
            concurrency=current.concurrency + concurrency,
            storage_bytes=current.storage_bytes + storage_bytes,
            model_calls=current.model_calls + model_calls,
            tool_calls=current.tool_calls + tool_calls,
            cost=current.cost + cost,
        )
        checks = (
            ("concurrency", proposed.concurrency, quota.max_concurrency),
            ("storage_bytes", proposed.storage_bytes, quota.max_storage_bytes),
            ("model_calls", proposed.model_calls, quota.max_model_calls),
            ("tool_calls", proposed.tool_calls, quota.max_tool_calls),
            ("cost", proposed.cost, quota.max_cost),
        )
        for label, value, limit in checks:
            if value < 0:
                raise ValueError(f"tenant {tenant_id} usage {label} must not be negative")
            if limit is not None and value > limit:
                raise TenantQuotaExceeded(
                    f"tenant {tenant_id} exceeded {label}: {value} > {limit}"
                )
        self._usage[tenant_id] = proposed
        return copy.deepcopy(proposed)

    def release(
        self,
        tenant_id: str,
        *,
        concurrency: int = 0,
        storage_bytes: int = 0,
    ) -> TenantUsage:
        if concurrency < 0 or storage_bytes < 0:
            raise ValueError("release amounts must be non-negative")
        for label, value in (("concurrency", concurrency), ("storage_bytes", storage_bytes)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{label} delta must be an integer")
        # Releasing never checks quota limits: an operator lowering a limit
        # while work is in flight must not strand the usage that work holds.
        current = self._usage.setdefault(tenant_id, TenantUsage())
        if concurrency > current.concurrency or storage_bytes > current.storage_bytes:
            raise ValueError(f"tenant {tenant_id} cannot release more than it holds")
        released = TenantUsage(
            concurrency=current.concurrency - concurrency,
            storage_bytes=current.storage_bytes - storage_bytes,
            model_calls=current.model_calls,
            tool_calls=current.tool_calls,
            cost=current.cost,
        )
        self._usage[tenant_id] = released
        return copy.deepcopy(released)


@dataclass(frozen=True)
class PersistenceSchema:
    name: str
    version: int
    validator: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("schema name must not be empty")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError("schema version must be an integer")
        if self.version < 1:
            raise ValueError("schema version must be at least 1")


class PersistenceSchemaRegistry:
    def __init__(self) -> None:
        self._schemas: dict[tuple[str, int], PersistenceSchema] = {}

    def register(self, schema: PersistenceSchema) -> None:
        key = (schema.name, schema.version)
        if key in self._schemas:
            raise ValueError(f"schema already registered: {schema.name}@{schema.version}")
        self._schemas[key] = schema

    def get(self, name: str, version: int | None = None) -> PersistenceSchema:
        if version is not None:
            key = (name, version)
            if key not in self._schemas:
                raise KeyError(key)
            return self._schemas[key]
        versions = [v for n, v in self._schemas if n == name]
        if not versions:
            raise KeyError(name)
        return self._schemas[(name, max(versions))]

    def validate(self, name: str, version: int, value: Mapping[str, Any]) -> None:
        schema = self.get(name, version)
        if schema.validator is not None:
            schema.validator(value)


MigrationFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class MigrationRegistry:
    def __init__(self) -> None:
        self._migrations: dict[tuple[str, int], MigrationFn] = {}

    def register(self, schema_name: str, from_version: int, migration: MigrationFn) -> None:
        if not schema_name.strip():
            raise ValueError("schema_name must not be empty")
        if not isinstance(from_version, int) or isinstance(from_version, bool):
            raise TypeError("from_version must be an integer")
        if from_version < 1:
            raise ValueError("from_version must be at least 1")
        key = (schema_name, from_version)
        if key in self._migrations:
            raise ValueError(f"migration already registered: {schema_name}@{from_version}")
        self._migrations[key] = migration

    def migrate(
        self,
        schema_name: str,
        value: Mapping[str, Any],
        *,
        from_version: int,
        to_version: int,
    ) -> Mapping[str, Any]:
        if not schema_name.strip():
            raise ValueError("schema_name must not be empty")
        if not isinstance(from_version, int) or isinstance(from_version, bool):
            raise TypeError("from_version must be an integer")
        if not isinstance(to_version, int) or isinstance(to_version, bool):
            raise TypeError("to_version must be an integer")
        if from_version < 1 or to_version < 1:
            raise ValueError("migration versions must be at least 1")
        if to_version < from_version:
            raise ValueError("downgrade migrations are not supported")
        current = copy.deepcopy(dict(value))
        version = from_version
        while version < to_version:
            key = (schema_name, version)
            if key not in self._migrations:
                raise KeyError(key)
            current = dict(self._migrations[key](current))
            version += 1
        return current

    def migrate_validated(
        self,
        schemas: PersistenceSchemaRegistry,
        schema_name: str,
        value: Mapping[str, Any],
        *,
        from_version: int,
        to_version: int,
    ) -> Mapping[str, Any]:
        schemas.validate(schema_name, from_version, value)
        current = copy.deepcopy(dict(value))
        version = from_version
        while version < to_version:
            current = dict(
                self.migrate(
                    schema_name,
                    current,
                    from_version=version,
                    to_version=version + 1,
                )
            )
            version += 1
            schemas.validate(schema_name, version, current)
        return current

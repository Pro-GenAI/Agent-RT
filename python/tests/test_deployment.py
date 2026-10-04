import pytest

from ext.deployment import (
    DeploymentRuntimeRegistry,
    DeploymentSpec,
    DistributedExecutor,
    DistributedTask,
    DistributedWorker,
    InMemorySharedStateStore,
    MigrationRegistry,
    PersistenceSchema,
    PersistenceSchemaRegistry,
    StatelessRuntimeNode,
    TenantQuota,
    TenantQuotaExceeded,
    TenantQuotaManager,
)


class TestDeployment:
    async def test_deployment_registry_and_distributed_execution(self):
        class Runtime:
            def __init__(self, prefix):
                self.prefix = prefix

            async def execute(self, payload):
                return f"{self.prefix}:{payload}"

        registry = DeploymentRuntimeRegistry()
        registry.register("local", lambda spec: Runtime(spec.name))
        runtime = registry.create(DeploymentSpec("local", "dev"))
        assert await runtime.execute("job") == "dev:job"

        executor = DistributedExecutor()
        executor.register(
            DistributedWorker(
                "a", lambda task: _return(task.payload * 2), kinds=("tool",)
            )
        )
        executor.register(
            DistributedWorker(
                "b", lambda task: _return(task.payload + 1), kinds=("tool",)
            )
        )
        first = await executor.execute(DistributedTask("1", "tool", 3))
        second = await executor.execute(DistributedTask("2", "tool", 3))
        assert (first.worker_id, second.worker_id) == ("a", "b")

    async def test_stateless_nodes_share_state(self):
        store = InMemorySharedStateStore()
        a = StatelessRuntimeNode("a", store)
        b = StatelessRuntimeNode("b", store)
        await a.save("session", {"step": 2})
        assert await b.load("session") == {"step": 2}

    def test_tenant_quotas_enforce_accounting(self):
        quotas = TenantQuotaManager()
        with pytest.raises(ValueError):
            TenantQuota(max_model_calls=-1)
        with pytest.raises(TypeError):
            TenantQuota(max_model_calls=1.5)
        with pytest.raises(ValueError):
            TenantQuota(max_cost=float("nan"))
        quotas.set_quota("tenant", TenantQuota(max_model_calls=2, max_cost=1.0))
        usage = quotas.consume(
            "tenant", concurrency=2, storage_bytes=10, model_calls=1, cost=0.4
        )
        assert usage.model_calls == 1
        released = quotas.release("tenant", concurrency=1, storage_bytes=4)
        assert (released.concurrency, released.storage_bytes) == (1, 6)
        with pytest.raises(ValueError):
            quotas.release("tenant", concurrency=2)
        with pytest.raises(TenantQuotaExceeded):
            quotas.consume("tenant", model_calls=2)
        with pytest.raises(ValueError):
            quotas.consume("tenant", model_calls=-2)
        with pytest.raises(TypeError):
            quotas.consume("tenant", model_calls=0.5)
        with pytest.raises(ValueError):
            quotas.consume("tenant", cost=float("inf"))

    def test_versioned_schemas_and_migrations(self):
        schemas = PersistenceSchemaRegistry()
        with pytest.raises(TypeError):
            PersistenceSchema("session", 1.5)
        schemas.register(PersistenceSchema("session", 1))
        schemas.register(
            PersistenceSchema("session", 2, lambda value: _require(value, "status"))
        )
        assert schemas.get("session").version == 2

        migrations = MigrationRegistry()
        with pytest.raises(ValueError):
            migrations.register("", 1, lambda value: value)
        migrations.register("session", 1, lambda value: {**value, "status": "active"})
        migrated = migrations.migrate(
            "session", {"id": "s1"}, from_version=1, to_version=2
        )
        schemas.validate("session", 2, migrated)
        assert migrated["status"] == "active"
        validated = migrations.migrate_validated(
            schemas,
            "session",
            {"id": "s2"},
            from_version=1,
            to_version=2,
        )
        assert validated["status"] == "active"


async def _return(value):
    return value


def _require(value, key):
    if key not in value:
        raise ValueError(key)

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  DeploymentRuntimeRegistry,
  DistributedExecutor,
  DistributedWorker,
  InMemorySharedStateStore,
  StatelessRuntimeNode,
  TenantQuotaManager,
  TenantQuotaExceeded,
  PersistenceSchemaRegistry,
  MigrationRegistry,
} = require("../dist/ext/deployment.js");

test("deployment registry and distributed execution", async () => {
  const registry = new DeploymentRuntimeRegistry();
  assert.throws(() => registry.register("invalid", () => ({ execute: async () => null })));
  registry.register("local", (spec) => ({
    execute: async (payload) => spec.name + ":" + payload,
  }));
  const runtime = registry.create({ kind: "local", name: "dev" });
  assert.equal(await runtime.execute("job"), "dev:job");

  const executor = new DistributedExecutor();
  executor.register(new DistributedWorker("a", async (task) => task.payload * 2, ["tool"]));
  executor.register(new DistributedWorker("b", async (task) => task.payload + 1, ["tool"]));
  const first = await executor.execute({ id: "1", kind: "tool", payload: 3 });
  const second = await executor.execute({ id: "2", kind: "tool", payload: 3 });
  assert.deepEqual([first.workerId, second.workerId], ["a", "b"]);
});

test("stateless runtime nodes share durable state", async () => {
  const store = new InMemorySharedStateStore();
  const a = new StatelessRuntimeNode("a", store);
  const b = new StatelessRuntimeNode("b", store);
  await a.save("session", { step: 2 });
  assert.deepEqual(await b.load("session"), { step: 2 });
});

test("tenant quotas enforce accounting", () => {
  const quotas = new TenantQuotaManager();
  assert.throws(() => quotas.setQuota("bad", { maxModelCalls: -1 }));
  assert.throws(() => quotas.setQuota("fractional", { maxModelCalls: 1.5 }));
  assert.throws(() => quotas.setQuota("nan", { maxCost: Number.NaN }));
  quotas.setQuota("tenant", { maxModelCalls: 2, maxCost: 1 });
  const used = quotas.consume("tenant", { concurrency: 2, storageBytes: 10, modelCalls: 1, cost: 0.4 });
  assert.equal(used.modelCalls, 1);
  assert.deepEqual(
    quotas.release("tenant", { concurrency: 1, storageBytes: 4 }),
    { concurrency: 1, storageBytes: 6, modelCalls: 1, toolCalls: 0, cost: 0.4 },
  );
  assert.throws(() => quotas.release("tenant", { concurrency: 2 }));
  assert.throws(
    () => quotas.consume("tenant", { modelCalls: 2 }),
    TenantQuotaExceeded,
  );
  assert.throws(() => quotas.consume("tenant", { modelCalls: -2 }));
  assert.throws(() => quotas.consume("tenant", { modelCalls: 0.5 }));
  assert.throws(() => quotas.consume("tenant", { cost: Number.POSITIVE_INFINITY }));
});

test("versioned schemas and migrations compose", () => {
  const schemas = new PersistenceSchemaRegistry();
  assert.throws(() => schemas.register({ name: "session", version: 1.5 }));
  schemas.register({ name: "session", version: 1 });
  schemas.register({
    name: "session",
    version: 2,
    validator: (value) => {
      if (!("status" in value)) throw new Error("status");
    },
  });
  assert.equal(schemas.get("session").version, 2);

  const migrations = new MigrationRegistry();
  assert.throws(() => migrations.register("", 1, (value) => value));
  migrations.register("session", 1, (value) => ({ ...value, status: "active" }));
  const migrated = migrations.migrate("session", { id: "s1" }, 1, 2);
  schemas.validate("session", 2, migrated);
  assert.equal(migrated.status, "active");
  const validated = migrations.migrateValidated(schemas, "session", { id: "s2" }, 1, 2);
  assert.equal(validated.status, "active");
});

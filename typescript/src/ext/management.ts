export interface PromptVersion {
  promptId: string;
  version: number;
  template: string;
  metadata: Record<string, unknown>;
}

export interface PromptDeployment {
  environment: string;
  promptId: string;
  version: number;
}

export class PromptManager {
  private readonly versions = new Map<string, PromptVersion[]>();
  private readonly deployments = new Map<string, number>();

  create(promptId: string, template: string, metadata: Record<string, unknown> = {}): PromptVersion {
    if (!promptId.trim()) throw new Error("promptId must not be empty");
    if (!template) throw new Error("prompt template must not be empty");
    const history = this.versions.get(promptId) ?? [];
    const item = { promptId, version: history.length + 1, template, metadata: { ...metadata } };
    history.push(item);
    this.versions.set(promptId, history);
    return item;
  }

  get(promptId: string, version?: number): PromptVersion {
    const history = this.versions.get(promptId) ?? [];
    if (history.length === 0) throw new Error("unknown prompt " + promptId);
    if (version === undefined) return history[history.length - 1];
    const item = history.find((entry) => entry.version === version);
    if (!item) throw new Error("unknown prompt version " + promptId + "/" + version);
    return item;
  }

  history(promptId: string): PromptVersion[] {
    if (!this.versions.has(promptId)) throw new Error("unknown prompt " + promptId);
    return [...this.versions.get(promptId)!];
  }

  deploy(promptId: string, environment: string, version?: number): PromptDeployment {
    if (!environment.trim()) throw new Error("environment must not be empty");
    const prompt = this.get(promptId, version);
    this.deployments.set(environment + "::" + promptId, prompt.version);
    return { environment, promptId, version: prompt.version };
  }

  deployed(promptId: string, environment: string): PromptVersion {
    const version = this.deployments.get(environment + "::" + promptId);
    if (version === undefined) throw new Error("prompt is not deployed");
    return this.get(promptId, version);
  }

  rollback(promptId: string, environment: string, steps = 1): PromptDeployment {
    if (steps < 1) throw new Error("steps must be at least 1");
    const current = this.deployed(promptId, environment);
    const target = current.version - steps;
    if (target < 1) throw new Error("no earlier prompt version available");
    return this.deploy(promptId, environment, target);
  }

  compare(promptId: string, left: number, right: number): Record<string, unknown> {
    const a = this.get(promptId, left);
    const b = this.get(promptId, right);
    return {
      promptId,
      leftVersion: left,
      rightVersion: right,
      changed: a.template !== b.template || canonicalJson(a.metadata) !== canonicalJson(b.metadata),
      leftTemplate: a.template,
      rightTemplate: b.template,
    };
  }

  render(promptId: string, variables: Record<string, unknown> = {}, version?: number): string {
    const template = this.get(promptId, version).template;
    return template.replace(/\{([^{}]+)\}/g, (_match, name) => {
      if (!(name in variables)) throw new Error("missing prompt variable: " + name);
      return String(variables[name]);
    });
  }
}


function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return "[" + value.map((item) => canonicalJson(item)).join(",") + "]";
  }
  if (isRecord(value)) {
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map((key) => JSON.stringify(key) + ":" + canonicalJson(value[key]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = structuredClone(base);
  for (const [key, value] of Object.entries(override)) {
    if (isRecord(value) && isRecord(result[key])) {
      result[key] = deepMerge(result[key] as Record<string, unknown>, value);
    } else {
      result[key] = structuredClone(value);
    }
  }
  return result;
}

export interface RuntimeConfiguration {
  environment: string;
  values: Record<string, unknown>;
}

export class ConfigurationManager {
  private readonly environments = new Map<string, Record<string, unknown>>();

  constructor(private readonly base: Record<string, unknown> = {}) {}

  setEnvironment(name: string, values: Record<string, unknown>): void {
    if (!name.trim()) throw new Error("environment name must not be empty");
    this.environments.set(name, structuredClone(values));
  }

  resolve(environment: string, overrides: Record<string, unknown> = {}): RuntimeConfiguration {
    const environmentValues = this.environments.get(environment) ?? {};
    return {
      environment,
      values: deepMerge(deepMerge(this.base, environmentValues), overrides),
    };
  }
}

export interface FeatureFlag {
  name: string;
  enabled?: boolean;
  environments?: string[];
  subjects?: string[];
  percentage?: number;
}

export class FeatureFlagRegistry {
  private readonly flags = new Map<string, Required<FeatureFlag>>();

  set(flag: FeatureFlag): void {
    if (!flag.name.trim()) throw new Error("feature flag name must not be empty");
    const percentage = flag.percentage ?? 100;
    if (percentage < 0 || percentage > 100) {
      throw new Error("feature flag percentage must be between 0 and 100");
    }
    this.flags.set(flag.name, {
      name: flag.name,
      enabled: flag.enabled ?? false,
      environments: [...(flag.environments ?? [])],
      subjects: [...(flag.subjects ?? [])],
      percentage,
    });
  }

  get(name: string): Required<FeatureFlag> {
    const flag = this.flags.get(name);
    if (!flag) throw new Error("unknown feature flag " + name);
    return flag;
  }

  enabled(name: string, environment?: string, subject?: string): boolean {
    const flag = this.get(name);
    if (!flag.enabled) return false;
    if (flag.environments.length > 0 && !flag.environments.includes(environment ?? "")) return false;
    if (flag.subjects.length > 0 && !flag.subjects.includes(subject ?? "")) return false;
    if (flag.percentage >= 100) return true;
    if (flag.percentage <= 0) return false;
    const source = name + ":" + (subject ?? "") + ":" + (environment ?? "");
    const bytes = new TextEncoder().encode(source);
    let weighted = 0;
    for (let index = 0; index < bytes.length; index += 1) {
      weighted += (index + 1) * bytes[index];
    }
    return weighted % 100 < flag.percentage;
  }
}

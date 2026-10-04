from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import copy


@dataclass(frozen=True)
class PromptVersion:
    prompt_id: str
    version: int
    template: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def render(self, variables: Mapping[str, Any] | None = None) -> str:
        try:
            return self.template.format_map(dict(variables or {}))
        except KeyError as exc:
            raise ValueError(f"missing prompt variable: {exc.args[0]}") from exc


@dataclass(frozen=True)
class PromptDeployment:
    environment: str
    prompt_id: str
    version: int


class PromptManager:
    def __init__(self) -> None:
        self._versions: dict[str, list[PromptVersion]] = {}
        self._deployments: dict[tuple[str, str], int] = {}

    def create(self, prompt_id: str, template: str, *, metadata: Mapping[str, Any] | None = None) -> PromptVersion:
        if not prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not template:
            raise ValueError("prompt template must not be empty")
        versions = self._versions.setdefault(prompt_id, [])
        item = PromptVersion(prompt_id, len(versions) + 1, template, dict(metadata or {}))
        versions.append(item)
        return item

    def get(self, prompt_id: str, version: int | None = None) -> PromptVersion:
        versions = self._versions.get(prompt_id, [])
        if not versions:
            raise KeyError(prompt_id)
        if version is None:
            return versions[-1]
        for item in versions:
            if item.version == version:
                return item
        raise KeyError((prompt_id, version))

    def history(self, prompt_id: str) -> tuple[PromptVersion, ...]:
        if prompt_id not in self._versions:
            raise KeyError(prompt_id)
        return tuple(self._versions[prompt_id])

    def deploy(self, prompt_id: str, environment: str, version: int | None = None) -> PromptDeployment:
        if not environment.strip():
            raise ValueError("environment must not be empty")
        prompt = self.get(prompt_id, version)
        self._deployments[(environment, prompt_id)] = prompt.version
        return PromptDeployment(environment, prompt_id, prompt.version)

    def deployed(self, prompt_id: str, environment: str) -> PromptVersion:
        key = (environment, prompt_id)
        if key not in self._deployments:
            raise KeyError(key)
        return self.get(prompt_id, self._deployments[key])

    def rollback(self, prompt_id: str, environment: str, *, steps: int = 1) -> PromptDeployment:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        current = self.deployed(prompt_id, environment)
        target = current.version - steps
        if target < 1:
            raise ValueError("no earlier prompt version available")
        return self.deploy(prompt_id, environment, target)

    def compare(self, prompt_id: str, left: int, right: int) -> Mapping[str, Any]:
        a = self.get(prompt_id, left)
        b = self.get(prompt_id, right)
        return {
            "prompt_id": prompt_id,
            "left_version": left,
            "right_version": right,
            "changed": a.template != b.template or dict(a.metadata) != dict(b.metadata),
            "left_template": a.template,
            "right_template": b.template,
        }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class RuntimeConfiguration:
    environment: str
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, Mapping):
            raise TypeError(f"configuration section {name} is not a mapping")
        return value


class ConfigurationManager:
    def __init__(self, base: Mapping[str, Any] | None = None) -> None:
        self._base = dict(base or {})
        self._environments: dict[str, Mapping[str, Any]] = {}

    def set_environment(self, name: str, values: Mapping[str, Any]) -> None:
        if not name.strip():
            raise ValueError("environment name must not be empty")
        self._environments[name] = copy.deepcopy(dict(values))

    def resolve(self, environment: str, *, overrides: Mapping[str, Any] | None = None) -> RuntimeConfiguration:
        values = _deep_merge(self._base, self._environments.get(environment, {}))
        if overrides:
            values = _deep_merge(values, overrides)
        return RuntimeConfiguration(environment, values)


@dataclass(frozen=True)
class FeatureFlag:
    name: str
    enabled: bool = False
    environments: frozenset[str] = frozenset()
    subjects: frozenset[str] = frozenset()
    percentage: int = 100

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature flag name must not be empty")
        if not 0 <= self.percentage <= 100:
            raise ValueError("feature flag percentage must be between 0 and 100")


class FeatureFlagRegistry:
    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlag] = {}

    def set(self, flag: FeatureFlag) -> None:
        self._flags[flag.name] = flag

    def get(self, name: str) -> FeatureFlag:
        if name not in self._flags:
            raise KeyError(name)
        return self._flags[name]

    def enabled(self, name: str, *, environment: str | None = None, subject: str | None = None) -> bool:
        flag = self.get(name)
        if not flag.enabled:
            return False
        if flag.environments and environment not in flag.environments:
            return False
        if flag.subjects and subject not in flag.subjects:
            return False
        if flag.percentage >= 100:
            return True
        if flag.percentage <= 0:
            return False
        source = f"{name}:{subject or ''}:{environment or ''}".encode("utf-8")
        bucket = sum((index + 1) * byte for index, byte in enumerate(source)) % 100
        return bucket < flag.percentage

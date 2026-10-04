from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ext.registration_safety import (
    RegistrationSafetyGuard,
    RegistrationSafetySubject,
    enforce_registration_safety,
)


@dataclass(frozen=True)
class SkillPackage:
    name: str
    version: str
    instructions: str = ""
    tools: tuple[Any, ...] = ()
    schemas: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("skill name must not be empty")
        if not self.version.strip():
            raise ValueError("skill version must not be empty")


_SKILL_FILE = "SKILL.md"
_MAX_SKILL_BYTES = 1_000_000
_MAX_RESOURCE_FILES = 256
_MAX_RESOURCE_BYTES = 16 * 1024 * 1024
_FRONTMATTER_KEY = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def _frontmatter_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
        return parsed if isinstance(parsed, str) else str(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _split_skill_markdown(text: str) -> tuple[dict[str, str], str, str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, text, ""

    end = normalized.find("\n---\n", 4)
    closing_length = 5
    if end < 0 and normalized.endswith("\n---"):
        # Closing delimiter at EOF without a trailing newline (no body).
        end = len(normalized) - 4
        closing_length = 4
    if end < 0:
        raise ValueError("SKILL.md frontmatter is missing its closing --- delimiter")

    raw_frontmatter = normalized[4:end]
    frontmatter: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        if not line or line[0].isspace():
            continue
        match = _FRONTMATTER_KEY.match(line)
        if match:
            frontmatter[match.group(1)] = _frontmatter_scalar(match.group(2))

    instructions = normalized[end + closing_length :].lstrip("\n")
    return frontmatter, instructions, raw_frontmatter


def load_skill_package(path: str | Path, *, version: str | None = None) -> SkillPackage:
    """Load a Claude/Codex-style SKILL.md directory into an Agent RT skill package."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("skill paths must not be symlinks")
    skill_file = requested if requested.is_file() else requested / _SKILL_FILE
    if skill_file.is_symlink():
        raise ValueError(f"{_SKILL_FILE} must not be a symlink")
    if skill_file.name != _SKILL_FILE or not skill_file.is_file():
        raise ValueError(
            f"skill path must be a {_SKILL_FILE} file or directory containing one"
        )
    if skill_file.stat().st_size > _MAX_SKILL_BYTES:
        raise ValueError(f"{_SKILL_FILE} exceeds the {_MAX_SKILL_BYTES}-byte limit")

    skill_file = skill_file.resolve()
    skill_root = skill_file.parent
    text = skill_file.read_text(encoding="utf-8")
    frontmatter, instructions, raw_frontmatter = _split_skill_markdown(text)
    name = frontmatter.get("name", skill_root.name).strip()
    selected_version = (version or frontmatter.get("version") or "1").strip()
    if not name:
        raise ValueError("skill name must not be empty")
    if not selected_version:
        raise ValueError("skill version must not be empty")

    resources: dict[str, Any] = {}
    resource_count = 0
    resource_bytes = 0
    for resource in sorted(skill_root.rglob("*")):
        if resource == skill_file or resource.is_symlink() or not resource.is_file():
            continue
        resource_count += 1
        if resource_count > _MAX_RESOURCE_FILES:
            raise ValueError(
                f"skill exceeds the {_MAX_RESOURCE_FILES}-resource-file limit"
            )
        # Check the size before reading so an oversized file is never loaded.
        if resource_bytes + resource.stat().st_size > _MAX_RESOURCE_BYTES:
            raise ValueError(
                f"skill resources exceed the {_MAX_RESOURCE_BYTES}-byte limit"
            )
        data = resource.read_bytes()
        resource_bytes += len(data)
        if resource_bytes > _MAX_RESOURCE_BYTES:
            raise ValueError(
                f"skill resources exceed the {_MAX_RESOURCE_BYTES}-byte limit"
            )
        relative = resource.relative_to(skill_root).as_posix()
        try:
            resources[relative] = data.decode("utf-8")
        except UnicodeDecodeError:
            resources[relative] = data

    metadata: dict[str, Any] = {
        "source_format": "skill-md",
        "source_path": str(skill_root),
        "frontmatter": dict(frontmatter),
    }
    if raw_frontmatter:
        metadata["raw_frontmatter"] = raw_frontmatter
    description = frontmatter.get("description", "").strip()
    if description:
        metadata["description"] = description

    return SkillPackage(
        name=name,
        version=selected_version,
        instructions=instructions,
        resources=resources,
        metadata=metadata,
    )


def _version_sort_key(version: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    current = ""
    numeric = version[:1].isdigit()
    for char in version:
        is_numeric = char.isdigit()
        if current and is_numeric != numeric:
            parts.append((1, int(current)) if numeric else (0, current.lower()))
            current = ""
        current += char
        numeric = is_numeric
    if current:
        parts.append((1, int(current)) if numeric else (0, current.lower()))
    return tuple(parts)


class SkillRegistry:
    def __init__(
        self,
        *,
        registration_guard: RegistrationSafetyGuard | None = None,
    ) -> None:
        self._skills: dict[tuple[str, str], SkillPackage] = {}
        self._active: dict[str, str] = {}
        self.registration_guard = registration_guard

    def install(
        self,
        skill: SkillPackage,
        *,
        activate: bool = False,
        registration_guard: RegistrationSafetyGuard | None = None,
    ) -> None:
        description = skill.metadata.get("description", "")
        content = {"instructions": skill.instructions}
        raw_frontmatter = skill.metadata.get("raw_frontmatter")
        if isinstance(raw_frontmatter, str):
            content["frontmatter"] = raw_frontmatter
        for index, tool in enumerate(skill.tools):
            content[f"tool:{index}:name"] = str(getattr(tool, "name", ""))
            content[f"tool:{index}:description"] = str(getattr(tool, "description", ""))
            content[f"tool:{index}:metadata"] = json.dumps(
                getattr(tool, "metadata", {}), default=str, ensure_ascii=False
            )
            content[f"tool:{index}:input_schema"] = json.dumps(
                getattr(tool, "input_schema", {}), default=str, ensure_ascii=False
            )
        for path, resource in skill.resources.items():
            if isinstance(resource, str):
                content[f"resource:{path}"] = resource
        enforce_registration_safety(
            RegistrationSafetySubject(
                kind="skill",
                name=skill.name,
                description=(
                    description if isinstance(description, str) else str(description)
                ),
                content=content,
            ),
            guard=registration_guard or self.registration_guard,
        )
        self._skills[(skill.name, skill.version)] = skill
        if activate:
            self._active[skill.name] = skill.version

    def install_from_path(
        self,
        path: str | Path,
        *,
        activate: bool = False,
        version: str | None = None,
        decision_provider: Any | None = None,
    ) -> SkillPackage:
        skill = load_skill_package(path, version=version)
        from ext.decisions import (
            JevDecisionProvider,
            make_decision_registration_guard,
        )

        provider = decision_provider or JevDecisionProvider.from_env()
        decision_guard = make_decision_registration_guard(provider)
        self.install(skill, activate=activate, registration_guard=decision_guard)
        return skill

    def install_directory(
        self,
        path: str | Path,
        *,
        activate: bool = False,
        decision_provider: Any | None = None,
    ) -> tuple[SkillPackage, ...]:
        root = Path(path).expanduser()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("skill directory does not exist or is a symlink")
        if (root / _SKILL_FILE).is_symlink():
            raise ValueError(f"{_SKILL_FILE} must not be a symlink")
        if (root / _SKILL_FILE).is_file():
            return (
                self.install_from_path(
                    root,
                    activate=activate,
                    decision_provider=decision_provider,
                ),
            )

        candidates = sorted(
            child
            for child in root.iterdir()
            if not child.is_symlink()
            and child.is_dir()
            and not (child / _SKILL_FILE).is_symlink()
            and (child / _SKILL_FILE).is_file()
        )
        if not candidates:
            raise ValueError(
                f"no child skill directories containing {_SKILL_FILE} were found"
            )
        return tuple(
            self.install_from_path(
                candidate,
                activate=activate,
                decision_provider=decision_provider,
            )
            for candidate in candidates
        )

    def activate(self, name: str, version: str) -> SkillPackage:
        skill = self.get(name, version)
        self._active[name] = version
        return skill

    def get(self, name: str, version: str | None = None) -> SkillPackage:
        selected = version or self._active.get(name)
        if selected is None:
            versions = sorted(
                (v for n, v in self._skills if n == name), key=_version_sort_key
            )
            if not versions:
                raise KeyError(name)
            selected = versions[-1]
        key = (name, selected)
        if key not in self._skills:
            raise KeyError(key)
        return self._skills[key]

    def active(self) -> tuple[SkillPackage, ...]:
        return tuple(
            self.get(name, version) for name, version in sorted(self._active.items())
        )

    def uninstall(self, name: str, version: str) -> None:
        key = (name, version)
        if key not in self._skills:
            raise KeyError(key)
        del self._skills[key]
        if self._active.get(name) == version:
            del self._active[name]


MiddlewareHandler = Callable[
    [str, Any, Callable[[Any], Awaitable[Any]]],
    Awaitable[Any],
]


class MiddlewarePipeline:
    def __init__(self) -> None:
        self._handlers: dict[str, list[MiddlewareHandler]] = {}

    def use(self, stage: str, handler: MiddlewareHandler) -> None:
        if not stage.strip():
            raise ValueError("middleware stage must not be empty")
        self._handlers.setdefault(stage, []).append(handler)

    async def run(
        self,
        stage: str,
        value: Any,
        terminal: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        handlers = tuple(self._handlers.get(stage, ()))

        async def invoke(index: int, current: Any) -> Any:
            if index >= len(handlers):
                return await terminal(current)
            return await handlers[index](
                stage,
                current,
                lambda next_value: invoke(index + 1, next_value),
            )

        return await invoke(0, value)


class ExtensionRegistry:
    SUPPORTED_KINDS = frozenset(
        {
            "model",
            "memory",
            "filesystem",
            "sandbox",
            "queue",
            "evaluator",
            "telemetry",
        }
    )

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str], Any] = {}

    def register(
        self,
        kind: str,
        name: str,
        provider: Any,
        *,
        replace_existing: bool = False,
    ) -> None:
        if kind not in self.SUPPORTED_KINDS:
            raise ValueError(f"unsupported extension kind: {kind}")
        if not name.strip():
            raise ValueError("extension provider name must not be empty")
        key = (kind, name)
        if key in self._providers and not replace_existing:
            raise ValueError(f"extension provider already registered: {kind}/{name}")
        self._providers[key] = provider

    def get(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self._providers:
            raise KeyError(key)
        return self._providers[key]

    def list(self, kind: str | None = None) -> tuple[tuple[str, str], ...]:
        keys = [key for key in self._providers if kind is None or key[0] == kind]
        return tuple(sorted(keys))

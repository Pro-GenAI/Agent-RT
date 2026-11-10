"""Virtual file systems, workspace files, and file-system backends."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class FileStat:
    path: str
    size: int
    is_directory: bool = False


@runtime_checkable
class FileSystem(Protocol):
    def list(self, path: str = "") -> tuple[FileStat, ...]: ...
    def read(self, path: str) -> bytes: ...
    def write(self, path: str, data: bytes, *, overwrite: bool = True) -> None: ...
    def delete(self, path: str) -> bool: ...
    def move(
        self, source: str, destination: str, *, overwrite: bool = False
    ) -> None: ...
    def copy(
        self, source: str, destination: str, *, overwrite: bool = False
    ) -> None: ...
    def glob(self, pattern: str) -> tuple[str, ...]: ...
    def search(self, text: str, *, path: str = "") -> tuple[str, ...]: ...


class InMemoryFileSystem:
    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    @staticmethod
    def _normalize(path: str) -> str:
        raw = path.strip().replace("\\", "/")
        parts: list[str] = []
        for part in raw.lstrip("/").split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                if not parts:
                    raise ValueError("path escapes filesystem root")
                parts.pop()
            else:
                parts.append(part)
        return "/".join(parts)

    def list(self, path: str = "") -> tuple[FileStat, ...]:
        prefix = self._normalize(path)
        prefix_with_sep = prefix + "/" if prefix else ""
        children: dict[str, FileStat] = {}
        for file_path, data in self._files.items():
            if not file_path.startswith(prefix_with_sep):
                continue
            remainder = file_path[len(prefix_with_sep) :]
            if not remainder:
                children[file_path] = FileStat(file_path, len(data), False)
                continue
            first, sep, _rest = remainder.partition("/")
            child_path = prefix_with_sep + first
            children[child_path] = FileStat(
                child_path,
                0 if sep else len(data),
                bool(sep),
            )
        return tuple(children[key] for key in sorted(children))

    def read(self, path: str) -> bytes:
        normalized = self._normalize(path)
        try:
            return self._files[normalized]
        except KeyError as exc:
            raise FileNotFoundError(normalized) from exc

    def write(self, path: str, data: bytes, *, overwrite: bool = True) -> None:
        normalized = self._normalize(path)
        if not normalized:
            raise ValueError("cannot write filesystem root")
        if not overwrite and normalized in self._files:
            raise FileExistsError(normalized)
        self._files[normalized] = bytes(data)

    def clear(self) -> None:
        """Remove every file. Deleting the root through ``delete`` is refused."""
        self._files.clear()

    def delete(self, path: str) -> bool:
        normalized = self._normalize(path)
        if not normalized:
            raise ValueError("cannot delete filesystem root; use clear()")
        if normalized in self._files:
            del self._files[normalized]
            return True
        prefix = normalized + "/" if normalized else ""
        matches = [key for key in self._files if key.startswith(prefix)]
        for key in matches:
            del self._files[key]
        return bool(matches)

    def move(self, source: str, destination: str, *, overwrite: bool = False) -> None:
        data = self.read(source)
        if self._normalize(source) == self._normalize(destination):
            return
        self.write(destination, data, overwrite=overwrite)
        self.delete(source)

    def copy(self, source: str, destination: str, *, overwrite: bool = False) -> None:
        data = self.read(source)
        if self._normalize(source) == self._normalize(destination):
            return
        self.write(destination, data, overwrite=overwrite)

    @staticmethod
    def _glob_regex(pattern: str) -> re.Pattern[str]:
        """``*`` and ``?`` stay within a path segment; ``**`` crosses segments."""
        pieces: list[str] = []
        index = 0
        while index < len(pattern):
            character = pattern[index]
            if pattern.startswith("**", index):
                pieces.append(".*")
                index += 2
                continue
            if character == "*":
                pieces.append("[^/]*")
            elif character == "?":
                pieces.append("[^/]")
            else:
                pieces.append(re.escape(character))
            index += 1
        return re.compile("".join(pieces))

    def glob(self, pattern: str) -> tuple[str, ...]:
        matcher = self._glob_regex(self._normalize(pattern))
        return tuple(sorted(path for path in self._files if matcher.fullmatch(path)))

    def search(self, text: str, *, path: str = "") -> tuple[str, ...]:
        prefix = self._normalize(path)
        prefix_with_sep = prefix + "/" if prefix else ""
        matches = []
        for file_path, data in self._files.items():
            if (
                prefix
                and file_path != prefix
                and not file_path.startswith(prefix_with_sep)
            ):
                continue
            if text in data.decode("utf-8", errors="replace"):
                matches.append(file_path)
        return tuple(sorted(matches))


class WorkspaceFiles:
    def __init__(self, filesystem: FileSystem) -> None:
        self.filesystem = filesystem

    def list(self, path: str = "") -> tuple[FileStat, ...]:
        return self.filesystem.list(path)

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return self.filesystem.read(path).decode(encoding)

    def write_text(
        self,
        path: str,
        text: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
    ) -> None:
        self.filesystem.write(path, text.encode(encoding), overwrite=overwrite)

    def create_text(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.write_text(path, text, encoding=encoding, overwrite=False)

    def delete(self, path: str) -> bool:
        return self.filesystem.delete(path)

    def clear(self) -> None:
        """Remove everything in the workspace (the root cannot be ``delete``d)."""
        clear = getattr(self.filesystem, "clear", None)
        if callable(clear):
            clear()
            return
        for entry in self.filesystem.list(""):
            self.filesystem.delete(entry.path)

    def move(self, source: str, destination: str, *, overwrite: bool = False) -> None:
        self.filesystem.move(source, destination, overwrite=overwrite)

    def copy(self, source: str, destination: str, *, overwrite: bool = False) -> None:
        self.filesystem.copy(source, destination, overwrite=overwrite)

    def search(self, text: str, *, path: str = "") -> tuple[str, ...]:
        return self.filesystem.search(text, path=path)

    def glob(self, pattern: str) -> tuple[str, ...]:
        return self.filesystem.glob(pattern)

    def exact_edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_occurrences: int = 1,
    ) -> None:
        if expected_occurrences < 1:
            raise ValueError("expected_occurrences must be at least 1")
        current = self.read_text(path)
        count = current.count(old_text)
        if count != expected_occurrences:
            raise ValueError(
                f"exact edit expected {expected_occurrences} occurrences, found {count}"
            )
        self.write_text(path, current.replace(old_text, new_text, expected_occurrences))

    def apply_unified_patch(self, path: str, patch: str) -> None:
        original = self.read_text(path).splitlines(keepends=True)
        lines = patch.splitlines(keepends=True)
        hunks: list[tuple[int, list[str]]] = []
        index = 0
        header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,\d+)? @@")
        while index < len(lines):
            match = header.match(lines[index])
            if not match:
                index += 1
                continue
            old_count = 1 if match.group(2) is None else int(match.group(2))
            # An empty old range (-N,0) names the line *before* the insertion
            # point, so it already is the zero-based insertion index.
            old_start = (
                int(match.group(1)) if old_count == 0 else int(match.group(1)) - 1
            )
            index += 1
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith("@@ "):
                # Lines inside a hunk are never headers: a removed "-- x" line
                # looks like "--- x" and must not be skipped.
                body.append(lines[index])
                index += 1
            hunks.append((old_start, body))
        if not hunks:
            raise ValueError("unified patch contains no hunks")

        output: list[str] = []
        cursor = 0
        for old_start, body in hunks:
            if old_start < cursor or old_start > len(original):
                raise ValueError("unified patch hunk is out of range")
            output.extend(original[cursor:old_start])
            cursor = old_start
            for line in body:
                if line in ("\n", "\r\n"):
                    # Editors often strip the leading space of blank context lines.
                    line = " " + line
                prefix = line[0] if line else ""
                value = line[1:]
                if prefix == " ":
                    if cursor >= len(original) or original[cursor] != value:
                        raise ValueError("unified patch context mismatch")
                    output.append(original[cursor])
                    cursor += 1
                elif prefix == "-":
                    if cursor >= len(original) or original[cursor] != value:
                        raise ValueError("unified patch removal mismatch")
                    cursor += 1
                elif prefix == "+":
                    output.append(value)
                elif line.startswith("\\ No newline at end of file"):
                    continue
                else:
                    raise ValueError("unsupported unified patch line")
        output.extend(original[cursor:])
        self.write_text(path, "".join(output))


FileSystemFactory = Callable[[str], FileSystem]


class FileSystemBackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, FileSystemFactory] = {}

    def register(
        self,
        backend: str,
        factory: FileSystemFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not backend.strip():
            raise ValueError("filesystem backend name must not be empty")
        if backend in self._factories and not replace:
            raise ValueError(f"filesystem backend already registered: {backend}")
        self._factories[backend] = factory

    def create(self, backend: str, workspace_id: str) -> FileSystem:
        try:
            factory = self._factories[backend]
        except KeyError as exc:
            raise KeyError(f"filesystem backend not registered: {backend}") from exc
        return factory(workspace_id)

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


ArtifactStatus = Literal["draft", "finalized"]

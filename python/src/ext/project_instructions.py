"""AGENTS.md and CLAUDE.md instruction files, found the way Codex and Claude Code find them.

Global files come from ``~/.codex`` and ``~/.claude``; project files from every folder
between the Git root (or the workspace when there is none) and the workspace, outermost
first, so nearer files are read last and win where they disagree.
"""

from __future__ import annotations

from pathlib import Path

# Codex's default ``project_doc_max_bytes``; keeps a huge file from eating the context.
MAX_INSTRUCTION_BYTES = 32 * 1024

_PROJECT_FILES = ("CLAUDE.md", ".claude/CLAUDE.md", "CLAUDE.local.md")


def _agents_file(folder: Path) -> Path:
    # Codex reads AGENTS.override.md instead of AGENTS.md when both exist.
    override = folder / "AGENTS.override.md"
    return override if override.is_file() else folder / "AGENTS.md"


def _git_root(workspace: Path) -> Path:
    for folder in (workspace, *workspace.parents):
        if (folder / ".git").exists():
            return folder
    return workspace


def instruction_files(
    workspace: Path, *, home: Path | None = None
) -> list[tuple[Path, Path]]:
    """Candidate ``(path, boundary)`` pairs in read order; a file must resolve inside its boundary."""

    home = home or Path.home()
    workspace = workspace.resolve()
    root = _git_root(workspace)
    found = [
        (_agents_file(home / ".codex"), home),
        (home / ".claude" / "CLAUDE.md", home),
    ]
    folders = [workspace, *workspace.parents]
    for folder in reversed(folders[: folders.index(root) + 1]):
        found.append((_agents_file(folder), root))
        found.extend((folder / name, root) for name in _PROJECT_FILES)
    return found


def load_project_instructions(
    workspace: Path, *, home: Path | None = None
) -> tuple[str, tuple[Path, ...]]:
    """Return the system-prompt section built from instruction files and the files it used."""

    sections: list[str] = []
    used: list[Path] = []
    seen: set[Path] = set()
    budget = MAX_INSTRUCTION_BYTES
    for path, boundary in instruction_files(workspace, home=home):
        try:
            resolved = path.resolve(strict=True)
            # A symlink must not pull in files from outside the repository or home folder.
            if not resolved.is_file() or not resolved.is_relative_to(
                boundary.resolve()
            ):
                continue
        except (OSError, RuntimeError):
            continue
        if resolved in seen:  # e.g. CLAUDE.md symlinked to AGENTS.md
            continue
        seen.add(resolved)
        if budget <= 0:
            sections.append(f"## {path}\n[skipped: instruction size limit reached]")
            continue
        try:
            with resolved.open("rb") as handle:
                data = handle.read(budget + 1)
        except OSError:
            continue
        text = data[:budget].decode("utf-8", errors="replace").strip()
        if len(data) > budget:
            text += "\n[truncated: instruction size limit reached]"
        budget -= len(data)
        if text:
            sections.append(f"## {path}\n{text}")
            used.append(path)
    if not sections:
        return "", ()
    return (
        "\n\n# Project instructions\n"
        "These come from AGENTS.md and CLAUDE.md files. Follow them unless they conflict "
        "with the user's request; later (nearer) files take precedence.\n\n"
        + "\n\n".join(sections)
    ), tuple(used)

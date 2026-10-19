"""``deepagents.create_deep_agent`` over the LangChain compatibility agent loop.

Deep agents add planning (``write_todos``), a filesystem (``ls``,
``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``), and
subagent delegation (``task``) to a LangChain agent. Like the upstream
default ``StateBackend``, the filesystem is virtual: files live in agent
state (``result["files"]``) and never touch the host. Sandboxed ``execute``,
pluggable backends, skills, memory files, permissions, and interrupts are not
emulated; passing them raises instead of silently running without them.
"""

from __future__ import annotations

import posixpath
import re
import sys
import types
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ext.compat.langchain import _CompatAgent, init_chat_model

BASE_AGENT_PROMPT = (
    "You are a capable agent. Plan multi-step work with write_todos and keep it "
    "current. Use the file tools (ls, read_file, write_file, edit_file, glob, "
    "grep) on the agent's virtual filesystem; paths are absolute, such as "
    "/notes.md. Delegate self-contained work to a subagent with task."
)
GENERAL_PURPOSE_SUBAGENT = {
    "name": "general-purpose",
    "description": (
        "General-purpose agent for researching complex questions and executing "
        "multi-step tasks; it has the same tools as the main agent."
    ),
}
_UNSUPPORTED = ("backend", "interrupt_on", "skills", "memory", "permissions")
_SUBAGENT_KEYS = frozenset({"name", "description", "system_prompt", "tools", "model", "middleware"})
_MAX_READ_LINES = 2000


def _now():
    return datetime.now(timezone.utc).isoformat()


def _path(value):
    return posixpath.normpath("/" + str(value or "/").lstrip("/"))


def _content(entry):
    if isinstance(entry, Mapping):
        content = entry.get("content", "")
        # Older deepagents releases stored content as a list of lines.
        return "\n".join(content) if isinstance(content, list) else str(content)
    return str(entry)


def _file_data(content, created_at=None):
    now = _now()
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": created_at or now,
        "modified_at": now,
    }


def _prompt(text):
    return f"{text}\n\n{BASE_AGENT_PROMPT}" if text else BASE_AGENT_PROMPT


def _files(runtime):
    return dict(runtime.state.get("files") or {})


def _glob_regex(pattern):
    parts, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        else:
            parts.append(
                {"*": "[^/]*", "?": "[^/]"}.get(pattern[i], re.escape(pattern[i]))
            )
            i += 1
    return re.compile("".join(parts))


def _under(path, base):
    return base == "/" or path == base or path.startswith(base + "/")


def write_todos(todos: list[dict], runtime=None):
    """Replace the todo list. Each todo is {"content": str, "status": "pending" | "in_progress" | "completed"}."""
    runtime.state["todos"] = [dict(item) for item in todos]
    return f"Updated todo list to {runtime.state['todos']}"


def ls(path: str = "/", runtime=None):
    """List files and directories directly under an absolute directory path."""
    base = _path(path)
    entries = set()
    for name in _files(runtime):
        name = _path(name)
        if name == base or not _under(name, base):
            continue
        rest = name[len(base.rstrip("/")) + 1 :]
        head, sep, _ = rest.partition("/")
        entries.add(base.rstrip("/") + "/" + head + ("/" if sep else ""))
    return sorted(entries)


def read_file(
    file_path: str, offset: int = 0, limit: int = _MAX_READ_LINES, runtime=None
):
    """Read a file from the virtual filesystem with cat -n style line numbers."""
    path = _path(file_path)
    files = _files(runtime)
    if path not in files:
        return f"Error: File '{path}' not found"
    lines = _content(files[path]).splitlines()
    start = max(0, int(offset))
    window = lines[start : start + max(0, int(limit))]
    if not window:
        return (
            "System reminder: File exists but has empty contents"
            if not lines
            else (
                f"Error: Line offset {start} exceeds file length ({len(lines)} lines)"
            )
        )
    return "\n".join(f"{start + i + 1:6d}\t{line}" for i, line in enumerate(window))


def write_file(file_path: str, content: str, runtime=None):
    """Create a new file. Existing files must be changed with edit_file."""
    path = _path(file_path)
    files = _files(runtime)
    if path in files:
        return (
            f"Cannot write to {path} because it already exists. "
            "Read and then make an edit, or write to a new path."
        )
    files[path] = _file_data(content)
    runtime.state["files"] = files
    return f"Updated file {path}"


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    runtime=None,
):
    """Replace exact text in a file; old_string must be unique unless replace_all is true."""
    path = _path(file_path)
    files = _files(runtime)
    if path not in files:
        return f"Error: File '{path}' not found"
    content = _content(files[path])
    count = content.count(old_string) if old_string else 0
    if count == 0:
        return f"Error: String not found in file: '{old_string}'"
    if count > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {count} times in file. "
            "Use replace_all=True to replace all instances, or provide a more "
            "specific string with surrounding context."
        )
    updated = content.replace(old_string, new_string, -1 if replace_all else 1)
    previous = files[path]
    created = previous.get("created_at") if isinstance(previous, Mapping) else None
    files[path] = _file_data(updated, created)
    runtime.state["files"] = files
    return (
        f"Successfully replaced {count if replace_all else 1} instance(s) in '{path}'"
    )


def glob(pattern: str, path: str = "/", runtime=None):
    """Find file paths under a directory that match a glob pattern such as **/*.py."""
    base = _path(path)
    regex = _glob_regex(pattern)
    matches = []
    for name in _files(runtime):
        name = _path(name)
        if not _under(name, base):
            continue
        relative = name[len(base.rstrip("/")) + 1 :]
        if regex.fullmatch(relative):
            matches.append(name)
    return sorted(matches)


def grep(
    pattern: str,
    path: str = "/",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    runtime=None,
):
    """Search file contents for a literal string. output_mode: files_with_matches, content, or count."""
    base = _path(path)
    hits = {}
    for name, entry in sorted(_files(runtime).items()):
        name = _path(name)
        if not _under(name, base):
            continue
        if glob and not _glob_regex(glob).fullmatch(posixpath.basename(name)):
            continue
        lines = [
            (number, line)
            for number, line in enumerate(_content(entry).splitlines(), 1)
            if pattern in line
        ]
        if lines:
            hits[name] = lines
    if not hits:
        return f"No matches found for pattern '{pattern}'"
    if output_mode == "content":
        return "\n".join(
            f"{name}:{number}:{line}"
            for name, lines in hits.items()
            for number, line in lines
        )
    if output_mode == "count":
        return "\n".join(f"{name}:{len(lines)}" for name, lines in hits.items())
    return "\n".join(hits)


_FILE_TOOLS = (write_todos, ls, read_file, write_file, edit_file, glob, grep)


class _DeepAgent(_CompatAgent):
    async def _run(self, inputs, *, config=None, emit=None):
        if isinstance(inputs, Mapping) and ("files" in inputs or "todos" in inputs):
            state = dict(inputs.get("state") or {})
            for key in ("files", "todos"):
                if key in inputs:
                    state[key] = inputs[key]
            inputs = {**inputs, "state": state}
        result = await super()._run(inputs, config=config, emit=emit)
        state = result.get("state") or {}
        result["files"] = dict(state.get("files") or {})
        result["todos"] = list(state.get("todos") or [])
        return result


def _task_tool(subagents, build):
    agents = {}
    descriptions = "\n".join(
        f"- {item['name']}: {item['description']}" for item in subagents
    )

    async def task(description: str, subagent_type: str, runtime=None):
        if subagent_type not in {item["name"] for item in subagents}:
            allowed = ", ".join(item["name"] for item in subagents)
            return (
                f"Error: unknown subagent_type {subagent_type!r}; use one of: {allowed}"
            )
        if subagent_type not in agents:
            spec = next(item for item in subagents if item["name"] == subagent_type)
            agents[subagent_type] = build(spec)
        # Isolated like upstream: the subagent sees only the task, and shares files.
        result = await agents[subagent_type].ainvoke(
            {
                "messages": [{"role": "user", "content": description}],
                "files": runtime.state.get("files") or {},
            }
        )
        runtime.state["files"] = result["files"]
        return result["messages"][-1].content

    task.__doc__ = (
        "Delegate a self-contained task to an isolated subagent and get back its "
        f"final answer. Available subagent_type values:\n{descriptions}"
    )
    return task


def create_deep_agent(
    model=None,
    tools=None,
    *,
    system_prompt=None,
    middleware=(),
    subagents=None,
    response_format=None,
    context_schema=None,
    state_schema=None,
    checkpointer=None,
    store=None,
    name=None,
    max_iterations=50,
    **kwargs,
):
    unsupported = sorted(key for key in _UNSUPPORTED if kwargs.get(key) is not None)
    if unsupported:
        raise NotImplementedError(
            f"deepagents compatibility does not support {', '.join(unsupported)}; "
            "files live in the virtual state filesystem"
        )
    kwargs = {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED}
    if kwargs:
        raise TypeError(f"unexpected create_deep_agent arguments: {sorted(kwargs)}")
    if model is None:
        raise ValueError(
            "create_deep_agent requires model= (for example 'openai:gpt-4.1')"
        )
    model = init_chat_model(model) if isinstance(model, str) else model
    user_tools = tuple(tools or ())
    specs = [dict(GENERAL_PURPOSE_SUBAGENT)]
    for spec in subagents or ():
        if not isinstance(spec, Mapping):
            raise NotImplementedError(
                "deepagents compatibility supports declarative SubAgent dicts only"
            )
        unsupported = set(spec) - _SUBAGENT_KEYS
        if unsupported:
            raise NotImplementedError(
                "deepagents compatibility does not support subagent options: "
                + ", ".join(sorted(map(str, unsupported)))
            )
        if not isinstance(spec.get("name"), str) or not spec["name"].strip():
            raise ValueError("deepagents subagents require a non-empty name")
        if not isinstance(spec.get("description"), str) or not spec["description"].strip():
            raise ValueError("deepagents subagents require a non-empty description")
        specs = [item for item in specs if item["name"] != spec["name"]] + [dict(spec)]

    def build(spec):
        return _DeepAgent(
            model=spec.get("model") or model,
            tools=(*spec.get("tools", user_tools), *_FILE_TOOLS),
            system_prompt=_prompt(spec.get("system_prompt")),
            middleware=spec.get("middleware", ()),
            max_iterations=max_iterations,
        )

    agent = _DeepAgent(
        model=model,
        tools=(*user_tools, *_FILE_TOOLS, _task_tool(specs, build)),
        system_prompt=_prompt(system_prompt),
        response_format=response_format,
        max_iterations=max_iterations,
        middleware=middleware,
        checkpointer=checkpointer,
        store=store,
        context_schema=context_schema,
        state_schema=state_schema,
    )
    agent.name = name
    return agent


def install_deepagents_compat(parent: Any) -> None:
    module = types.ModuleType(f"{parent.__name__}.deepagents")
    exports = {
        "BASE_AGENT_PROMPT": BASE_AGENT_PROMPT,
        "SubAgent": dict,
        "create_deep_agent": create_deep_agent,
    }
    module.__dict__.update(exports)
    module.__all__ = tuple(sorted(exports))
    sys.modules[module.__name__] = module
    parent.deepagents = module


__all__ = ("create_deep_agent", "install_deepagents_compat")

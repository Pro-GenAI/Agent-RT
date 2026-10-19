#!/usr/bin/env python3
"""Compare agent CLI harnesses (Agent RT, Codex, Claude Code, opencode, Copilot CLI).

Every CLI is pointed at a deterministic localhost mock that speaks OpenAI Chat
Completions, OpenAI Responses, and Anthropic Messages. Each run gets its own
temporary HOME/workspace, its own mock server (so requests and tokens are
attributed exactly), and an environment stripped of real credentials. No real
LLM or model API is called.

Per run: wall time, process-tree CPU time and peak RSS (psutil sampling),
model requests, request bytes, input/output tokens (tiktoken), and cost at the
configured reference prices. Concurrency phases launch N copies at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil
import tiktoken

from generate_charts import generate_charts

HERE = Path(__file__).resolve().parent
PROMPT = "hi"
RESPONSE_TEXT = "mock response"
MODEL = "gpt-5.4"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
API_KEY = "mock-key"
SAMPLE_INTERVAL_S = 0.01
DEFAULT_CONCURRENCY = (4, 8)
DEFAULT_INPUT_USD_PER_MTOK = 3.0
DEFAULT_OUTPUT_USD_PER_MTOK = 15.0
ENCODING = tiktoken.get_encoding("o200k_base")
# Keep the harness from reaching real services while it talks to the mock.
NOISE_OFF = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "CI": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "COPILOT_AUTO_UPDATE": "false",
    "OPENCODE_DISABLE_AUTOUPDATE": "true",
    "DO_NOT_TRACK": "1",
}
PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")


def count_tokens(value: Any) -> int:
    if value in (None, "", [], {}):
        return 0
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return len(ENCODING.encode(text, disallowed_special=()))


# --------------------------------------------------------------------- mock


@dataclass
class Ledger:
    requests: int = 0
    request_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_schema_tokens: int = 0
    tools: int = 0
    paths: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, path: str, size: int, body: dict[str, Any]) -> int:
        prompt_fields = ("system", "instructions", "messages", "input", "tools")
        prompt = sum(count_tokens(body.get(key)) for key in prompt_fields)
        with self.lock:
            self.requests += 1
            self.request_bytes += size
            self.input_tokens += prompt
            self.output_tokens += count_tokens(RESPONSE_TEXT)
            self.tool_schema_tokens += count_tokens(body.get("tools"))
            self.tools = max(self.tools, len(body.get("tools") or []))
            self.paths.append(path)
        return prompt


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MockServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse(self, events: list[tuple[str | None, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for name, data in events:
            chunk = f"event: {name}\n" if name else ""
            chunk += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
            self.wfile.write(chunk.encode())
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/models"):
            models = [{"id": m, "object": "model", "owned_by": "mock", "created": 0,
                       "type": "model", "display_name": m} for m in (MODEL, ANTHROPIC_MODEL)]
            self._send(200, {"object": "list", "data": models, "has_more": False})
            return
        self.server.ledger.unexpected.append(f"GET {path}")
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        size = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(size)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        if path.endswith("/messages/count_tokens"):
            self._send(200, {"input_tokens": count_tokens(body.get("messages"))})
            return
        handlers: dict[str, Callable[[dict[str, Any], int], None]] = {
            "/chat/completions": self._chat,
            "/responses": self._responses,
            "/messages": self._anthropic,
        }
        for suffix, handler in handlers.items():
            if path.endswith(suffix):
                handler(body, self.server.ledger.add(path, size, body))
                return
        self.server.ledger.unexpected.append(f"POST {path}")
        self._send(404, {"error": {"message": "not found"}})

    def _chat(self, body: dict[str, Any], prompt: int) -> None:
        usage = {"prompt_tokens": prompt, "completion_tokens": 2, "total_tokens": prompt + 2}
        base = {"id": "chatcmpl-mock", "created": 0, "model": body.get("model", MODEL)}
        if not body.get("stream"):
            message = {"role": "assistant", "content": RESPONSE_TEXT}
            self._send(200, {**base, "object": "chat.completion", "usage": usage,
                             "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})
            return
        chunk = {**base, "object": "chat.completion.chunk"}
        self._sse([
            (None, {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": RESPONSE_TEXT}, "finish_reason": None}]}),
            (None, {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            (None, {**chunk, "choices": [], "usage": usage}),
            (None, "[DONE]"),
        ])

    def _responses(self, body: dict[str, Any], prompt: int) -> None:
        item = {"id": "msg_mock", "type": "message", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": RESPONSE_TEXT, "annotations": []}]}
        usage = {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0},
                 "total_tokens": prompt + 2}
        response = {"id": "resp_mock", "object": "response", "created_at": 0, "status": "completed",
                    "model": body.get("model", MODEL), "output": [item], "usage": usage,
                    "output_text": RESPONSE_TEXT}
        if not body.get("stream"):
            self._send(200, response)
            return
        pending = {**response, "status": "in_progress", "output": [], "usage": None}
        self._sse([
            ("response.created", {"type": "response.created", "sequence_number": 0, "response": pending}),
            ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
                                            "item": {**item, "status": "in_progress", "content": []}}),
            ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 2, "item_id": item["id"],
                                            "output_index": 0, "content_index": 0, "delta": RESPONSE_TEXT}),
            ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": 3, "item_id": item["id"],
                                           "output_index": 0, "content_index": 0, "text": RESPONSE_TEXT}),
            ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "sequence_number": 5, "response": response}),
        ])

    def _anthropic(self, body: dict[str, Any], prompt: int) -> None:
        model = body.get("model", ANTHROPIC_MODEL)
        usage = {"input_tokens": prompt, "output_tokens": 2,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": "msg_mock", "type": "message", "role": "assistant", "model": model,
                   "content": [{"type": "text", "text": RESPONSE_TEXT}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            self._send(200, message)
            return
        self._sse([
            ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None,
                                                                     "usage": {**usage, "output_tokens": 1}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": RESPONSE_TEXT}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 2}}),
            ("message_stop", {"type": "message_stop"}),
        ])


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), MockHandler)
        self.ledger = Ledger()
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        threading.Thread(target=self.serve_forever, daemon=True).start()


# ---------------------------------------------------------------- harnesses


@dataclass
class Launch:
    argv: list[str]
    env: dict[str, str]


def _agent_rt(binary: str, url: str, home: Path) -> Launch:
    (home / ".agent-rt").mkdir()
    (home / ".agent-rt" / "config.yaml").write_text(
        "model_list:\n"
        "  - model_name: mock\n"
        "    agent_rt_params:\n"
        f"      model: openai/{MODEL}\n"
        f"      api_base: {url}/v1\n"
        "      api_key: os.environ/MOCK_API_KEY\n"
        "agent_rt_settings:\n"
        "  default_model: mock\n"
    )
    return Launch([binary, "-p", PROMPT], {"MOCK_API_KEY": API_KEY})


def _codex(binary: str, url: str, home: Path) -> Launch:
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(
        f'model = "{MODEL}"\n'
        'model_provider = "mock"\n'
        "[model_providers.mock]\n"
        f'name = "mock"\nbase_url = "{url}/v1"\nwire_api = "responses"\nenv_key = "MOCK_API_KEY"\n'
        # Network-only extras (plugin catalog, apps, analytics) wait on chatgpt.com, not the model.
        "[features]\nplugins = false\nremote_plugin = false\napps = false\n"
        "[analytics]\nenabled = false\n"
    )
    return Launch([binary, "exec", PROMPT], {"MOCK_API_KEY": API_KEY, "CODEX_HOME": str(home / ".codex")})


def _claude(binary: str, url: str, home: Path) -> Launch:
    return Launch(
        [binary, "-p", PROMPT],
        {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": API_KEY, "ANTHROPIC_MODEL": ANTHROPIC_MODEL,
         "ANTHROPIC_SMALL_FAST_MODEL": ANTHROPIC_MODEL, "CLAUDE_CONFIG_DIR": str(home / ".claude")},
    )


def _opencode(binary: str, url: str, home: Path) -> Launch:
    config = home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "model": f"mock/{MODEL}",
        "provider": {"mock": {"npm": "@ai-sdk/openai-compatible", "name": "mock",
                              "options": {"baseURL": f"{url}/v1", "apiKey": API_KEY},
                              "models": {MODEL: {"name": MODEL}}}},
    }))
    return Launch([binary, "run", PROMPT], {})


def _copilot(binary: str, url: str, home: Path) -> Launch:
    return Launch(
        [binary, "-p", PROMPT],
        {"COPILOT_OFFLINE": "true", "COPILOT_PROVIDER_BASE_URL": f"{url}/v1",
         "COPILOT_PROVIDER_API_KEY": API_KEY, "COPILOT_MODEL": MODEL, "COPILOT_ALLOW_ALL": "true",
         "COPILOT_HOME": str(home / ".copilot")},
    )


# Every CLI is invoked as its plain one-shot prompt command; settings go in config files/env.
HARNESSES: dict[str, tuple[str, Callable[[str, str, Path], Launch]]] = {
    "Agent RT": (str(HERE / ".venv" / "bin" / "agent-rt"), _agent_rt),
    "Codex": ("codex", _codex),
    "Claude Code": ("claude", _claude),
    "opencode": ("opencode", _opencode),
    "Copilot CLI": ("copilot", _copilot),
}


def command(name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        argv = HARNESSES[name][1]("x", "http://mock", Path(tmp)).argv
    return " ".join([Path(HARNESSES[name][0]).name, *(f'"{a}"' if a == PROMPT else a for a in argv[1:])])


def resolve_binary(name: str) -> str | None:
    path = HARNESSES[name][0]
    # opencode's installer puts the binary in ~/.opencode/bin, often not on PATH.
    found = path if Path(path).is_file() else shutil.which(path, path=f"{os.environ.get('PATH', '')}{os.pathsep}{Path.home() / '.opencode' / 'bin'}")
    return str(Path(found).resolve()) if found else None


def version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30,
                             env={k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ})
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


# --------------------------------------------------------------- measuring


@dataclass
class Sandbox:
    tmp: Path
    server: MockServer
    argv: list[str]
    env: dict[str, str]


def make_sandbox(name: str, binary: str) -> Sandbox:
    tmp = Path(tempfile.mkdtemp(prefix="agent-rt-cli-bench-"))
    server = None
    try:
        home = tmp / "home"
        home.mkdir()
        (tmp / "tmp").mkdir()
        server = MockServer()
        launch = HARNESSES[name][1](binary, server.url, home)
        # Strip every credential; only the mock key and harness-specific settings survive.
        env = {k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ}
        # A private TMPDIR too: opencode registers its background service under the temp dir.
        env.update(NOISE_OFF, HOME=str(home), USERPROFILE=str(home),
                   TMPDIR=str(tmp / "tmp"), TEMP=str(tmp / "tmp"), TMP=str(tmp / "tmp"),
                   XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
                   XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"))
        env.update(launch.env)
        return Sandbox(tmp, server, launch.argv, env)
    except BaseException:
        try:
            if server is not None:
                try:
                    server.shutdown()
                finally:
                    server.server_close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def close_sandbox(sandbox: Sandbox) -> None:
    try:
        sandbox.server.shutdown()
    finally:
        try:
            sandbox.server.server_close()
        finally:
            shutil.rmtree(sandbox.tmp, ignore_errors=True)


@dataclass
class Run:
    proc: psutil.Popen
    started: float
    timeout: float
    peak_rss: int = 0
    cpu: dict[int, float] = field(default_factory=dict)
    seen: dict[int, psutil.Process] = field(default_factory=dict)
    wall: float = 0.0


def start_run(sandbox: Sandbox, timeout: float) -> Run:
    work = Path(tempfile.mkdtemp(prefix="work-", dir=sandbox.tmp))
    (work / "README.md").write_text("# Benchmark workspace\n")
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)  # codex exec refuses non-repo folders
    proc = psutil.Popen(sandbox.argv, cwd=work, env=sandbox.env, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return Run(proc, time.perf_counter(), timeout)


def sample(run: Run) -> tuple[bool, int]:
    """Sample current tree RSS and CPU; return (root alive, current RSS)."""
    if run.proc.poll() is not None:
        return False, 0
    try:
        tree = [run.proc, *run.proc.children(recursive=True)]
    except psutil.Error:
        return run.proc.poll() is None, 0
    rss = 0
    for process in tree:
        run.seen.setdefault(process.pid, process)
        try:
            with process.oneshot():
                rss += process.memory_info().rss
                times = process.cpu_times()
                run.cpu[process.pid] = times.user + times.system
        except psutil.Error:
            continue
    run.peak_rss = max(run.peak_rss, rss)
    if time.perf_counter() - run.started > run.timeout:
        for process in tree:
            try:
                process.kill()
            except psutil.Error:
                pass
    return True, rss


def kill_run(run: Run, *, include_root: bool = True) -> None:
    """Kill tracked and attached descendants, including on failed batches."""
    processes = list(run.seen.values())
    try:
        processes.extend(run.proc.children(recursive=True))
    except psutil.Error:
        pass
    if include_root:
        processes.append(run.proc)
    # Process objects verify creation times before signaling reused PIDs.
    for process in dict.fromkeys(processes):
        try:
            process.kill()
        except psutil.Error:
            pass


def finish(name: str, run: Run, args: argparse.Namespace, phase: str, sample_id: int,
           ledger: Ledger | None) -> dict[str, Any]:
    """Collect one finished run; `ledger` is None when runs share a mock and tokens aren't attributable."""
    try:  # the root's own final CPU is exact; descendants use their last sample
        times = run.proc.cpu_times()
        run.cpu[run.proc.pid] = max(run.cpu.get(run.proc.pid, 0.0), times.user + times.system)
    except psutil.Error:
        pass
    # Daemons a CLI leaves behind (e.g. `opencode serve --service`) would leak into later runs
    # and could hold the output pipes open.
    kill_run(run, include_root=False)
    stdout, stderr = run.proc.communicate(timeout=5)
    cpu_ms = sum(run.cpu.values()) * 1000
    wall_ms = run.wall * 1000
    ok = run.proc.returncode == 0 and RESPONSE_TEXT in stdout and (ledger is None or ledger.requests > 0)
    row: dict[str, Any] = {
        "harness": name, "phase": phase, "sample": sample_id, "success": ok,
        "exit_code": run.proc.returncode, "wall_ms": round(wall_ms, 2), "cpu_ms": round(cpu_ms, 2),
        "cpu_util_pct": round(100 * cpu_ms / wall_ms, 1) if wall_ms else 0.0,
        "peak_rss_mib": round(run.peak_rss / 2**20, 2),
    }
    if ledger is not None:
        cost = (ledger.input_tokens * args.input_usd_per_mtok + ledger.output_tokens * args.output_usd_per_mtok) / 1e6
        row.update({
            "requests": ledger.requests, "request_bytes": ledger.request_bytes,
            "input_tokens": ledger.input_tokens, "tool_schema_tokens": ledger.tool_schema_tokens,
            "tools": ledger.tools, "output_tokens": ledger.output_tokens, "cost_usd": round(cost, 6),
            "endpoints": ";".join(sorted(set(ledger.paths))), "unexpected": ";".join(ledger.unexpected),
        })
    row["error"] = "" if ok else (stderr.strip() or stdout.strip())[-300:].replace("\n", " ")
    return row


@dataclass
class Batch:
    wall: float
    samples: list[dict[str, Any]]
    peak_rss_mib: float
    requests: int
    expected_min_requests: int


def run_batch(name: str, binary: str, n: int, args: argparse.Namespace, phase: str,
              min_requests_per_run: int = 1) -> Batch:
    """Run n copies at once. A batch shares one HOME and mock, like one user running parallel agents."""
    sandbox = make_sandbox(name, binary)
    runs: list[Run] = []
    try:
        started = time.perf_counter()
        for _ in range(n):
            runs.append(start_run(sandbox, args.timeout))
        live = list(runs)
        peak_batch_rss = 0
        while live:
            # Sum live RSS in one sampling sweep, not peaks reached at unrelated times.
            current_batch_rss = 0
            for run in list(live):
                alive, rss = sample(run)
                current_batch_rss += rss
                if not alive:
                    run.wall = time.perf_counter() - run.started
                    live.remove(run)
            peak_batch_rss = max(peak_batch_rss, current_batch_rss)
            if live:
                time.sleep(SAMPLE_INTERVAL_S)
        batch_wall = time.perf_counter() - started
        ledger = sandbox.server.ledger
        results = [finish(name, run, args, phase, i, ledger if n == 1 else None)
                   for i, run in enumerate(runs)]
        expected = n * min_requests_per_run
        # One shared mock cannot attribute requests to individual concurrent clients.
        # Fail the whole batch when its aggregate fails the sequential request floor.
        if ledger.requests < expected:
            reason = f"mock received {ledger.requests} model requests; expected at least {expected}"
            for row in results:
                row["success"] = False
                row["error"] = reason
        return Batch(batch_wall, results, round(peak_batch_rss / 2**20, 2), ledger.requests, expected)
    finally:
        try:
            for run in runs:
                kill_run(run)
                try:
                    run.proc.communicate(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            close_sandbox(sandbox)


# ------------------------------------------------------------------ report


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    return ordered[max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))]


def summarize(name: str, ver: str, samples: list[dict[str, Any]], batches: dict[int, Batch], args: argparse.Namespace) -> dict[str, Any]:
    ok = [s for s in samples if s["success"]]
    med = lambda key, digits=2: round(statistics.median(s[key] for s in ok), digits) if ok else ""  # noqa: E731
    row: dict[str, Any] = {
        "harness": name, "version": ver, "command": command(name), "runs": len(ok), "failures": len(samples) - len(ok),
        "median_wall_ms": med("wall_ms"),
        "p95_wall_ms": round(percentile([s["wall_ms"] for s in ok], 95), 2) if ok else "",
        "median_cpu_ms": med("cpu_ms"), "median_cpu_util_pct": med("cpu_util_pct"),
        "median_peak_rss_mib": med("peak_rss_mib"),
        "max_peak_rss_mib": max((s["peak_rss_mib"] for s in ok), default=""),
        "median_requests": med("requests"), "median_input_tokens": med("input_tokens"),
        "median_tool_schema_tokens": med("tool_schema_tokens"), "median_tools": med("tools"),
        "median_output_tokens": med("output_tokens"), "median_cost_usd": med("cost_usd", 6),
        "input_usd_per_mtok": args.input_usd_per_mtok, "output_usd_per_mtok": args.output_usd_per_mtok,
    }
    for level, batch in batches.items():
        good = [s for s in batch.samples if s["success"]]
        row[f"c{level}_batch_wall_ms"] = round(batch.wall * 1000, 2)
        row[f"c{level}_throughput_runs_per_s"] = round(len(good) / batch.wall, 3)
        row[f"c{level}_p95_wall_ms"] = round(percentile([s["wall_ms"] for s in good], 95), 2) if good else ""
        row[f"c{level}_sum_cpu_ms"] = round(sum(s["cpu_ms"] for s in batch.samples), 2)
        row[f"c{level}_batch_peak_rss_mib"] = batch.peak_rss_mib
        row[f"c{level}_sum_peak_rss_mib"] = round(sum(s["peak_rss_mib"] for s in batch.samples), 2)
        row[f"c{level}_model_requests"] = batch.requests
        row[f"c{level}_expected_min_requests"] = batch.expected_min_requests
        row[f"c{level}_failures"] = len(batch.samples) - len(good)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def publication_problems(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Fail closed before writing the complete, reproducible published dataset."""
    problems = []
    names = [row["harness"] for row in rows]
    if len(names) != len(HARNESSES) or set(names) != set(HARNESSES):
        problems.append("all five distinct harnesses must be present")
    for row in rows:
        name = row["harness"]
        if row.get("runs", 0) < 30 or row.get("failures", 1):
            problems.append(f"{name}: fewer than 30 clean sequential runs")
        for level in DEFAULT_CONCURRENCY:
            if row.get(f"c{level}_failures", 1):
                problems.append(f"{name}: failed or missing {level}-way batch")
            if not row.get(f"c{level}_batch_peak_rss_mib"):
                problems.append(f"{name}: missing {level}-way batch RSS")
            if row.get(f"c{level}_model_requests", 0) < row.get(f"c{level}_expected_min_requests", 1):
                problems.append(f"{name}: insufficient {level}-way model requests")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10, help="sequential runs per harness")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per harness first")
    parser.add_argument("--concurrency", type=int, nargs="*", default=list(DEFAULT_CONCURRENCY), help="parallel instance counts")
    parser.add_argument("--harness", action="append", choices=list(HARNESSES), help="filter (repeatable)")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-run timeout in seconds")
    parser.add_argument("--input-usd-per-mtok", type=float, default=DEFAULT_INPUT_USD_PER_MTOK, help="reference input price")
    parser.add_argument("--output-usd-per-mtok", type=float, default=DEFAULT_OUTPUT_USD_PER_MTOK, help="reference output price")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "cli-harnesses")
    parser.add_argument("--publish", action="store_true",
                        help="refresh cli-harness-measured-results.csv (all five installed, >= 30 clean runs, default contract)")
    args = parser.parse_args()
    if (args.runs < 1 or args.warmup < 0 or args.timeout <= 0
            or not args.concurrency or len(set(args.concurrency)) != len(args.concurrency)
            or any(level < 1 for level in args.concurrency)
            or not math.isfinite(args.input_usd_per_mtok) or args.input_usd_per_mtok < 0
            or not math.isfinite(args.output_usd_per_mtok) or args.output_usd_per_mtok < 0):
        parser.error("invalid runs, warmup, timeout, concurrency, or reference price")
    if args.publish and (args.harness or args.runs < 30 or args.warmup != 1
                         or tuple(args.concurrency) != DEFAULT_CONCURRENCY
                         or args.input_usd_per_mtok != DEFAULT_INPUT_USD_PER_MTOK
                         or args.output_usd_per_mtok != DEFAULT_OUTPUT_USD_PER_MTOK):
        parser.error("--publish requires all harnesses, >= 30 runs, one warmup, "
                     "4-/8-way concurrency, and default reference prices")

    binaries = {name: resolve_binary(name) for name in (args.harness or HARNESSES)}
    if args.publish:
        missing = [name for name, binary in binaries.items() if binary is None]
        if missing:
            print(f"not publishing: missing required CLIs: {', '.join(missing)}")
            return 1

    rows, raw = [], []
    for name, binary in binaries.items():
        if binary is None:
            print(f"{name}: not installed, skipped")
            rows.append({"harness": name, "version": "not installed", "runs": 0})
            continue
        ver = version(binary)
        print(f"{name} ({ver})")
        for _ in range(args.warmup):
            run_batch(name, binary, 1, args, "warmup")
        samples = []
        for i in range(args.runs):
            batch = run_batch(name, binary, 1, args, "sequential")
            batch.samples[0]["sample"] = i
            samples += batch.samples
            print(f"  run {i + 1}/{args.runs}: {batch.samples[0]['wall_ms']:.0f} ms"
                  + ("" if batch.samples[0]["success"] else f"  FAILED: {batch.samples[0]['error']}"))
        batches = {level: run_batch(name, binary, level, args, f"concurrency-{level}")
                   for level in args.concurrency}
        for level, batch in batches.items():
            print(f"  {level} parallel: {batch.wall * 1000:.0f} ms, "
                  f"{sum(s['success'] for s in batch.samples)}/{level} ok, "
                  f"{batch.requests} model requests")
        raw += samples + [s for batch in batches.values() for s in batch.samples]
        rows.append(summarize(name, ver, samples, batches, args))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        row["measured_at"] = stamp
    write_csv(args.output_dir / "cli-harness-samples.csv", raw)
    write_csv(args.output_dir / "cli-harness-summary.csv", rows)
    print(f"wrote {args.output_dir}")
    if args.publish:
        problems = publication_problems(rows, args)
        if problems:
            print("not publishing: " + "; ".join(problems))
            return 1
        write_csv(HERE / "cli-harness-measured-results.csv", rows)
        generate_charts()
        print("published cli-harness-measured-results.csv and regenerated charts")
    return 0 if all(s["success"] for s in raw) else 1


if __name__ == "__main__":
    raise SystemExit(main())

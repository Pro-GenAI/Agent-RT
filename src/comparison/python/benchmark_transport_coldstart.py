#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from mock_llm_api import start_mock_server


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


WORKER = r'''
import asyncio, json, sys, time
mode, base = sys.argv[1], sys.argv[2]
payload = json.dumps({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hi"}],
}, separators=(",", ":")).encode()

async def run_httpx():
    started=time.perf_counter(); import httpx; import_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    client=httpx.AsyncClient(
        base_url=base.rstrip("/")+"/",
        headers={"Authorization": "Bearer x"},
        timeout=600.0,
    )
    construct_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    response=await client.post("chat/completions", content=payload, headers={"content-type":"application/json"})
    response.raise_for_status()
    first_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    response=await client.post("chat/completions", content=payload, headers={"content-type":"application/json"})
    response.raise_for_status()
    second_ms=(time.perf_counter()-started)*1000
    await client.aclose()
    return import_ms, construct_ms, first_ms, second_ms

async def run_httpcore():
    started=time.perf_counter(); import httpcore; import_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter(); pool=httpcore.AsyncConnectionPool(); construct_ms=(time.perf_counter()-started)*1000
    headers=[(b"content-type", b"application/json"), (b"authorization", b"Bearer x")]
    url=(base.rstrip("/")+"/chat/completions").encode()
    started=time.perf_counter(); response=await pool.request(b"POST", url, headers=headers, content=payload); first_ms=(time.perf_counter()-started)*1000
    if response.status >= 400: raise RuntimeError(response.status)
    started=time.perf_counter(); response=await pool.request(b"POST", url, headers=headers, content=payload); second_ms=(time.perf_counter()-started)*1000
    if response.status >= 400: raise RuntimeError(response.status)

    stream_payload=json.dumps({
        "model":"gpt-4o-mini",
        "messages":[{"role":"user","content":"hi"}],
        "stream":True,
        "stream_options":{"include_usage":True},
    }, separators=(",", ":")).encode()
    stream_started=time.perf_counter(); first_event_ms=None; events=0; buffer=b""
    async with pool.stream(b"POST", url, headers=headers, content=stream_payload) as stream_response:
        if stream_response.status >= 400: raise RuntimeError(stream_response.status)
        async for chunk in stream_response.aiter_stream():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line=line.rstrip(b"\r")
                if not line.startswith(b"data:"):
                    continue
                data=line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                json.loads(data)
                events += 1
                if first_event_ms is None:
                    first_event_ms=(time.perf_counter()-stream_started)*1000
    if events == 0 or first_event_ms is None:
        raise RuntimeError("no SSE events")
    stream_total_ms=(time.perf_counter()-stream_started)*1000
    await pool.aclose()
    return import_ms, construct_ms, first_ms, second_ms, first_event_ms, stream_total_ms

values=asyncio.run(run_httpx() if mode=="httpx" else run_httpcore())
keys=("import_ms","construct_ms","first_post_ms","second_post_ms") if mode=="httpx" else ("import_ms","construct_ms","first_post_ms","second_post_ms","stream_first_event_ms","stream_total_ms")
print(json.dumps(dict(zip(keys, values))))
'''


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [row[key] for row in rows if key in row]
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_p95"] = percentile(values, 0.95)
    result["cold_total_ms_median"] = statistics.median(
        row["import_ms"] + row["construct_ms"] + row["first_post_ms"] for row in rows
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    server, thread, base_url = start_mock_server()
    results: dict[str, list[dict[str, float]]] = {"httpx": [], "httpcore": []}
    try:
        for mode in results:
            for _ in range(args.runs):
                completed = subprocess.run(
                    [sys.executable, "-c", WORKER, mode, base_url],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                results[mode].append(json.loads(completed.stdout))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    payload: dict[str, Any] = {
        "runs": args.runs,
        "results": {mode: summarize(rows) for mode, rows in results.items()},
        "samples": results,
        "caveat": (
            "Primitive cold transport diagnostic. HTTPCore SSE framing is exercised, "
            "but production proxy/TLS environment parity still requires validation."
        ),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

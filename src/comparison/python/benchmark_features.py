#!/usr/bin/env python3
"""Deterministic tool-call and structured-output comparison benchmarks.

Each framework talks to the same localhost OpenAI-compatible mock server. The
scenarios intentionally measure equivalent wire-level work rather than each
framework's higher-level agent orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from generate_charts import generate_charts
from mock_llm_api import RESPONSE_TEXT, TOOL_NAME, start_mock_server

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

MODEL_ID = "gpt-4o-mini"
TOOL_RESULT = json.dumps(
    {"value": 1, "status": "ok"},
    separators=(",", ":"),
    sort_keys=True,
)
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return a deterministic lookup value.",
        "parameters": TOOL_SCHEMA,
    },
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "result",
        "schema": RESPONSE_SCHEMA,
        "strict": True,
    },
}

FRAMEWORKS = ("Agent RT", "LangChain", "LlamaIndex")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample set")
    index = max(0, min(len(ordered) - 1, int(len(ordered) * q + 0.999999) - 1))
    return ordered[index]


async def measured(fn: Callable[[], Awaitable[Any]]) -> dict[str, float]:
    wall = time.perf_counter()
    cpu = time.process_time()
    await fn()
    return {
        "wall_ms": (time.perf_counter() - wall) * 1000,
        "cpu_ms": (time.process_time() - cpu) * 1000,
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    walls = [row["wall_ms"] for row in samples]
    cpus = [row["cpu_ms"] for row in samples]
    return {
        "median_ms": statistics.median(walls),
        "p95_ms": percentile(walls, 0.95),
        "median_cpu_percent": statistics.median(
            (cpu / wall * 100) if wall else 0.0
            for cpu, wall in zip(cpus, walls)
        ),
    }


def current_rss_mib() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    return None


def validate_structured(text: str) -> dict[str, int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("structured response is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"value"}:
        raise ValueError(f"structured response has unexpected shape: {value!r}")
    if not isinstance(value["value"], int) or isinstance(value["value"], bool):
        raise ValueError(f"structured value must be an integer: {value!r}")
    return value


@dataclass
class FeatureAdapter:
    create: Callable[[], Any]
    tool_roundtrip: Callable[[Any], Awaitable[None]]
    structured_once: Callable[[Any, str], Awaitable[str]]
    close: Callable[[Any], Awaitable[None]]


def load_adapter(framework: str, base_url: str) -> FeatureAdapter:
    if framework == "Agent RT":
        from agent_rt import (
            ContentPart,
            ModelMessage,
            ModelRequest,
            OpenAIModelProvider,
            OpenAIProviderSettings,
            StructuredOutputRequirement,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name=TOOL_NAME,
            description="Return a deterministic lookup value.",
            input_schema=TOOL_SCHEMA,
        )
        structured = StructuredOutputRequirement(
            name="result",
            schema=RESPONSE_SCHEMA,
            strict=True,
        )

        def create() -> Any:
            return OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=base_url,
                    api_key="mock-key",
                    default_model=MODEL_ID,
                    websocket=False,
                )
            )

        def user(text: str) -> Any:
            return ModelMessage(
                role="user",
                content=(ContentPart(type="text", text=text),),
            )

        async def tool_roundtrip(client: Any) -> None:
            first_user = user("use tool")
            first = await client.complete(
                ModelRequest(
                    model=MODEL_ID,
                    messages=(first_user,),
                    tools=(tool,),
                )
            )
            calls = first.message.tool_calls
            if len(calls) != 1 or calls[0].name != TOOL_NAME:
                raise RuntimeError(f"unexpected tool call: {calls!r}")
            tool_result = ModelMessage(
                role="tool",
                content=(ContentPart(type="text", text=TOOL_RESULT),),
                tool_call_id=calls[0].id,
            )
            second = await client.complete(
                ModelRequest(
                    model=MODEL_ID,
                    messages=(first_user, first.message, tool_result),
                    tools=(tool,),
                )
            )
            text = "".join(part.text or "" for part in second.message.content)
            if text != RESPONSE_TEXT:
                raise RuntimeError(f"unexpected tool continuation: {text!r}")

        async def structured_once(client: Any, prompt: str) -> str:
            response = await client.complete(
                ModelRequest(
                    model=MODEL_ID,
                    messages=(user(prompt),),
                    structured_output=structured,
                )
            )
            return "".join(part.text or "" for part in response.message.content)

        async def close(client: Any) -> None:
            await client.close()

        return FeatureAdapter(create, tool_roundtrip, structured_once, close)

    if framework == "LangChain":
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_openai import ChatOpenAI

        def create() -> Any:
            base = ChatOpenAI(
                model=MODEL_ID,
                api_key="mock-key",
                base_url=base_url,
                temperature=0,
                max_retries=0,
                use_responses_api=False,
            )
            return {
                "base": base,
                "tool": base.bind_tools([OPENAI_TOOL]),
                "structured": base.bind(response_format=RESPONSE_FORMAT),
            }

        async def tool_roundtrip(client: Any) -> None:
            initial = HumanMessage(content="use tool")
            first = await client["tool"].ainvoke([initial])
            calls = first.tool_calls
            if len(calls) != 1 or calls[0]["name"] != TOOL_NAME:
                raise RuntimeError(f"unexpected tool call: {calls!r}")
            second = await client["tool"].ainvoke(
                [
                    initial,
                    first,
                    ToolMessage(
                        content=TOOL_RESULT,
                        tool_call_id=calls[0]["id"],
                    ),
                ]
            )
            if second.content != RESPONSE_TEXT:
                raise RuntimeError(
                    f"unexpected tool continuation: {second.content!r}"
                )

        async def structured_once(client: Any, prompt: str) -> str:
            response = await client["structured"].ainvoke(
                [HumanMessage(content=prompt)]
            )
            return str(response.content)

        async def close(client: Any) -> None:
            return None

        return FeatureAdapter(create, tool_roundtrip, structured_once, close)

    if framework == "LlamaIndex":
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.llms.openai import OpenAI

        def create() -> Any:
            return OpenAI(
                model=MODEL_ID,
                api_key="mock-key",
                api_base=base_url,
                temperature=0,
                max_retries=0,
                timeout=10,
            )

        async def tool_roundtrip(client: Any) -> None:
            initial = ChatMessage(role="user", content="use tool")
            first = await client.achat([initial], tools=[OPENAI_TOOL])
            calls = first.message.additional_kwargs.get("tool_calls") or []
            if len(calls) != 1 or calls[0].function.name != TOOL_NAME:
                raise RuntimeError(f"unexpected tool call: {calls!r}")
            tool_result = ChatMessage(
                role="tool",
                content=TOOL_RESULT,
                additional_kwargs={
                    "tool_call_id": calls[0].id,
                    "name": TOOL_NAME,
                },
            )
            second = await client.achat(
                [initial, first.message, tool_result],
                tools=[OPENAI_TOOL],
            )
            if second.message.content != RESPONSE_TEXT:
                raise RuntimeError(
                    f"unexpected tool continuation: {second.message.content!r}"
                )

        async def structured_once(client: Any, prompt: str) -> str:
            response = await client.achat(
                [ChatMessage(role="user", content=prompt)],
                response_format=RESPONSE_FORMAT,
            )
            return response.message.content or ""

        async def close(client: Any) -> None:
            session = getattr(client, "_aclient", None)
            if session is not None:
                await session.close()

        return FeatureAdapter(create, tool_roundtrip, structured_once, close)

    raise ValueError(f"unknown framework: {framework}")


async def run_framework(
    framework: str,
    base_url: str,
    runs: int,
) -> dict[str, Any]:
    adapter = load_adapter(framework, base_url)
    client = adapter.create()
    samples: dict[str, list[dict[str, float]]] = {
        "tool_roundtrip": [],
        "tool_loop_5": [],
        "structured_valid": [],
        "structured_repair": [],
        "structured_repeat_20": [],
    }
    try:
        await adapter.tool_roundtrip(client)
        validate_structured(await adapter.structured_once(client, "structured"))

        for _ in range(runs):
            samples["tool_roundtrip"].append(
                await measured(lambda: adapter.tool_roundtrip(client))
            )

            async def tool_loop() -> None:
                for _ in range(5):
                    await adapter.tool_roundtrip(client)

            samples["tool_loop_5"].append(await measured(tool_loop))

            async def structured_valid() -> None:
                validate_structured(
                    await adapter.structured_once(client, "structured")
                )

            samples["structured_valid"].append(await measured(structured_valid))

            async def structured_repair() -> None:
                invalid = await adapter.structured_once(
                    client,
                    "[structured-invalid]",
                )
                try:
                    validate_structured(invalid)
                except ValueError:
                    pass
                else:
                    raise RuntimeError(
                        "repair scenario expected the first response to be invalid"
                    )
                repaired = await adapter.structured_once(
                    client,
                    "[structured-repair]",
                )
                validate_structured(repaired)

            samples["structured_repair"].append(
                await measured(structured_repair)
            )

            async def structured_repeat() -> None:
                for _ in range(20):
                    validate_structured(
                        await adapter.structured_once(client, "structured")
                    )

            samples["structured_repeat_20"].append(
                await measured(structured_repeat)
            )
        memory_stages: dict[str, float | None] = {
            "before_memory_probe": current_rss_mib(),
        }
        await adapter.tool_roundtrip(client)
        memory_stages["after_tool_roundtrip"] = current_rss_mib()

        for _ in range(5):
            await adapter.tool_roundtrip(client)
        memory_stages["after_tool_loop_5"] = current_rss_mib()

        validate_structured(
            await adapter.structured_once(client, "structured")
        )
        memory_stages["after_structured_valid"] = current_rss_mib()

        invalid = await adapter.structured_once(client, "[structured-invalid]")
        try:
            validate_structured(invalid)
        except ValueError:
            pass
        repaired = await adapter.structured_once(client, "[structured-repair]")
        validate_structured(repaired)
        memory_stages["after_structured_repair"] = current_rss_mib()

        for _ in range(20):
            validate_structured(
                await adapter.structured_once(client, "structured")
            )
        memory_stages["after_structured_repeat_20"] = current_rss_mib()
    finally:
        await adapter.close(client)

    return {
        "framework": framework,
        "runs": runs,
        **{name: summarize(rows) for name, rows in samples.items()},
        "samples": samples,
        "memory_stages": memory_stages,
        "caveat": (
            "Wire-level feature comparison; validation is intentionally "
            "performed by this runner so Pydantic/Zod costs are not mixed "
            "into framework transport overhead."
        ),
    }


def write_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature-summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = ["framework", "runs"]
    for scenario in (
        "tool_roundtrip",
        "tool_loop_5",
        "structured_valid",
        "structured_repair",
        "structured_repeat_20",
    ):
        fields.extend(
            [
                f"{scenario}_median_ms",
                f"{scenario}_p95_ms",
                f"{scenario}_cpu_percent",
            ]
        )
    with (output_dir / "feature-summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "framework": result["framework"],
                "runs": result["runs"],
            }
            for scenario in (
                "tool_roundtrip",
                "tool_loop_5",
                "structured_valid",
                "structured_repair",
                "structured_repeat_20",
            ):
                summary = result[scenario]
                row[f"{scenario}_median_ms"] = summary["median_ms"]
                row[f"{scenario}_p95_ms"] = summary["p95_ms"]
                row[f"{scenario}_cpu_percent"] = summary[
                    "median_cpu_percent"
                ]
            writer.writerow(row)

    memory_fields = ["framework", "stage", "rss_mib"]
    with (output_dir / "feature-memory-stages.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for stage, rss_mib in result["memory_stages"].items():
                writer.writerow({
                    "framework": result["framework"],
                    "stage": stage,
                    "rss_mib": rss_mib,
                })

    with (output_dir / "feature-samples.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = ["framework", "scenario", "run", "wall_ms", "cpu_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for scenario, rows in result["samples"].items():
                for index, sample in enumerate(rows, 1):
                    writer.writerow(
                        {
                            "framework": result["framework"],
                            "scenario": scenario,
                            "run": index,
                            "wall_ms": sample["wall_ms"],
                            "cpu_ms": sample["cpu_ms"],
                        }
                    )


async def async_main(args: argparse.Namespace) -> int:
    server, _, base_url = start_mock_server()
    try:
        frameworks = (
            [args.framework]
            if args.framework is not None
            else list(FRAMEWORKS)
        )
        results = []
        for framework in frameworks:
            result = await run_framework(framework, base_url, args.runs)
            results.append(result)
            print(
                f"{framework:20} "
                f"tool={result['tool_roundtrip']['median_ms']:.2f} ms "
                f"structured={result['structured_valid']['median_ms']:.2f} ms "
                f"repair={result['structured_repair']['median_ms']:.2f} ms"
            )
        write_results(results, args.output_dir)
        if args.framework is None and args.runs >= 30:
            source = args.output_dir / "feature-summary.csv"
            (HERE / "feature-measured-results.csv").write_bytes(
                source.read_bytes()
            )
            generate_charts()
        print(f"\nWrote feature comparison to {args.output_dir}")
        return 0
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--framework", choices=FRAMEWORKS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "features",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

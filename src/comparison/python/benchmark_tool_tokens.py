#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

from agent_rt import (
    AgentConfig,
    ContentPart,
    ModelMessage,
    ModelSettings,
    ToolDefinition,
    ToolFilterContext,
)
from ext.decisions import make_decision_tool_visibility_filter
from generate_charts import generate_charts


DEFAULT_TOOL_COUNTS = (8, 16, 32, 64, 128)
DEFAULT_ENCODING = "o200k_base"
DEFAULT_MAX_SELECTED_TOOLS = 8
DEFAULT_MAX_OPTIONS_PER_QUESTION = 16
DEFAULT_MIN_PROBABILITY = 0.05

CATEGORIES = (
    "calendar",
    "email",
    "files",
    "support",
    "crm",
    "finance",
    "analytics",
    "projects",
    "docs",
    "chat",
    "commerce",
    "inventory",
    "engineering",
    "security",
    "hr",
    "travel",
)

ACTIONS = (
    "search",
    "get",
    "list",
    "create",
    "update",
    "delete",
    "export",
    "summarize",
)

SCENARIOS = (
    ("calendar", "Find my meetings tomorrow and show the relevant calendar details."),
    ("email", "Find the latest customer email about the renewal and summarize the thread."),
    ("files", "Locate the project plan file and show the most relevant document details."),
    ("support", "Find the active support ticket for this customer and summarize its status."),
)


@dataclass(frozen=True)
class ScenarioResult:
    tool_count: int
    category: str
    selected_tool_count: int
    decision_request_count: int
    all_tools_main_tokens: int
    agent_rt_main_tokens: int
    main_tokens_saved: int
    main_reduction_pct: float
    decision_model_input_tokens: int
    agent_rt_combined_tokens: int
    combined_delta_vs_all: int


@dataclass(frozen=True)
class SummaryRow:
    tool_count: int
    scenario_count: int
    avg_selected_tools: float
    avg_decision_requests: float
    all_tools_main_tokens: int
    agent_rt_main_tokens_avg: float
    main_tokens_saved_avg: float
    main_reduction_pct: float
    decision_model_input_tokens_avg: float
    agent_rt_combined_tokens_avg: float
    combined_delta_vs_all_avg: float
    encoding: str


class RecordingDecisionProvider:
    """Deterministic Decision-model double that records the exact selection inputs."""

    def __init__(self, target_category: str) -> None:
        self.target_category = target_category
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        self.calls.append((state, questions))
        question = questions["tool"]
        criteria = question.get("criteria", {})
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("tool selection question must contain criteria")

        probabilities: dict[str, float] = {}
        for name in criteria:
            if not isinstance(name, str):
                continue
            if name.startswith(f"{self.target_category}_"):
                action = name.removeprefix(f"{self.target_category}_")
                try:
                    rank = ACTIONS.index(action)
                except ValueError:
                    rank = len(ACTIONS)
                probabilities[name] = max(0.10, 0.95 - (rank * 0.08))
            else:
                probabilities[name] = 0.001

        selected = max(probabilities, key=lambda name: (probabilities[name], name))
        return {
            "tool": {
                "choice": selected,
                "probabilities": probabilities,
            }
        }


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def count_json_tokens(encoding: Any, value: Any) -> int:
    return len(encoding.encode(compact_json(value)))


def build_tool_catalog(tool_count: int) -> tuple[ToolDefinition, ...]:
    if tool_count < 1 or tool_count > len(CATEGORIES) * len(ACTIONS):
        raise ValueError(
            f"tool_count must be between 1 and {len(CATEGORIES) * len(ACTIONS)}"
        )

    tools: list[ToolDefinition] = []
    # Action-major ordering ensures every small catalog contains several domains.
    for action in ACTIONS:
        for category in CATEGORIES:
            name = f"{category}_{action}"
            tools.append(
                ToolDefinition(
                    name=name,
                    description=(
                        f"{action.capitalize()} {category} records for the current user or "
                        "workspace. Use precise filters and return only fields relevant to "
                        "the current task."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    f"Natural-language query or identifier used to {action} "
                                    f"{category} records."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                            },
                            "include_archived": {
                                "type": "boolean",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                )
            )
            if len(tools) == tool_count:
                return tuple(tools)
    raise AssertionError("tool catalog construction fell through")


def openai_tool_payload(tools: Sequence[ToolDefinition]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            for tool in tools
        ]
    }


def decision_request_payloads(
    provider: RecordingDecisionProvider,
) -> list[dict[str, Any]]:
    # Mirrors JevDecisionProvider's request body fields, excluding the optional
    # hosted-model name because local Decision models do not require it.
    return [
        {
            "state": state,
            "questions": dict(questions),
        }
        for state, questions in provider.calls
    ]


def measure_scenario(
    *,
    tool_count: int,
    category: str,
    prompt: str,
    encoding: Any,
    max_selected_tools: int = DEFAULT_MAX_SELECTED_TOOLS,
    max_options_per_question: int = DEFAULT_MAX_OPTIONS_PER_QUESTION,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> ScenarioResult:
    tools = build_tool_catalog(tool_count)
    provider = RecordingDecisionProvider(category)
    tool_filter = make_decision_tool_visibility_filter(
        provider,
        min_probability=min_probability,
        max_selected_tools=max_selected_tools,
        max_options_per_question=max_options_per_question,
    )
    agent = AgentConfig(
        name="tool-token-benchmark",
        instructions="Select and use only tools relevant to the current task.",
        model=ModelSettings(model="mock-main-model"),
    )
    context = ToolFilterContext(
        agent=agent,
        messages=(
            ModelMessage(
                role="user",
                content=(ContentPart(type="text", text=prompt),),
            ),
        ),
        turn=0,
        tool_calls=0,
    )

    selected_names = tuple(tool_filter(context, tools))
    selected_name_set = set(selected_names)
    selected_tools = tuple(tool for tool in tools if tool.name in selected_name_set)

    all_tokens = count_json_tokens(encoding, openai_tool_payload(tools))
    selected_tokens = count_json_tokens(encoding, openai_tool_payload(selected_tools))
    decision_tokens = sum(
        count_json_tokens(encoding, payload)
        for payload in decision_request_payloads(provider)
    )
    saved = all_tokens - selected_tokens
    reduction = (saved / all_tokens * 100.0) if all_tokens else 0.0
    combined = selected_tokens + decision_tokens

    return ScenarioResult(
        tool_count=tool_count,
        category=category,
        selected_tool_count=len(selected_tools),
        decision_request_count=len(provider.calls),
        all_tools_main_tokens=all_tokens,
        agent_rt_main_tokens=selected_tokens,
        main_tokens_saved=saved,
        main_reduction_pct=reduction,
        decision_model_input_tokens=decision_tokens,
        agent_rt_combined_tokens=combined,
        combined_delta_vs_all=combined - all_tokens,
    )


def summarize(
    samples: Sequence[ScenarioResult],
    *,
    encoding_name: str,
) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for tool_count in sorted({sample.tool_count for sample in samples}):
        group = [sample for sample in samples if sample.tool_count == tool_count]
        all_values = {sample.all_tools_main_tokens for sample in group}
        if len(all_values) != 1:
            raise ValueError("all-tools token count must be identical within a tool-count group")
        all_tokens = next(iter(all_values))
        main_avg = statistics.mean(sample.agent_rt_main_tokens for sample in group)
        saved_avg = statistics.mean(sample.main_tokens_saved for sample in group)
        rows.append(
            SummaryRow(
                tool_count=tool_count,
                scenario_count=len(group),
                avg_selected_tools=statistics.mean(
                    sample.selected_tool_count for sample in group
                ),
                avg_decision_requests=statistics.mean(
                    sample.decision_request_count for sample in group
                ),
                all_tools_main_tokens=all_tokens,
                agent_rt_main_tokens_avg=main_avg,
                main_tokens_saved_avg=saved_avg,
                main_reduction_pct=(saved_avg / all_tokens * 100.0)
                if all_tokens
                else 0.0,
                decision_model_input_tokens_avg=statistics.mean(
                    sample.decision_model_input_tokens for sample in group
                ),
                agent_rt_combined_tokens_avg=statistics.mean(
                    sample.agent_rt_combined_tokens for sample in group
                ),
                combined_delta_vs_all_avg=statistics.mean(
                    sample.combined_delta_vs_all for sample in group
                ),
                encoding=encoding_name,
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_tool_counts(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one tool count is required")
    maximum = len(CATEGORIES) * len(ACTIONS)
    if any(value < 1 or value > maximum for value in values):
        raise argparse.ArgumentTypeError(
            f"tool counts must be between 1 and {maximum}"
        )
    return values


def format_summary(rows: Sequence[SummaryRow]) -> str:
    lines = [
        "tools  all-main  selected  agent-main  saved  reduction  decision-input  combined",
        "-----  --------  --------  ----------  -----  ---------  --------------  --------",
    ]
    for row in rows:
        lines.append(
            f"{row.tool_count:>5}  "
            f"{row.all_tools_main_tokens:>8}  "
            f"{row.avg_selected_tools:>8.2f}  "
            f"{row.agent_rt_main_tokens_avg:>10.1f}  "
            f"{row.main_tokens_saved_avg:>5.1f}  "
            f"{row.main_reduction_pct:>8.1f}%  "
            f"{row.decision_model_input_tokens_avg:>14.1f}  "
            f"{row.agent_rt_combined_tokens_avg:>8.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate tool-schema prompt tokens for full-catalog exposure versus "
            "Agent RT Decision-model tool selection."
        )
    )
    parser.add_argument(
        "--tool-counts",
        type=parse_tool_counts,
        default=DEFAULT_TOOL_COUNTS,
        help="comma-separated catalog sizes (default: 8,16,32,64,128)",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"tiktoken encoding (default: {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--max-selected-tools",
        type=int,
        default=DEFAULT_MAX_SELECTED_TOOLS,
    )
    parser.add_argument(
        "--max-options-per-question",
        type=int,
        default=DEFAULT_MAX_OPTIONS_PER_QUESTION,
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_PROBABILITY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tool-tokens"),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "refresh tool-token-measured-results.csv; only accepted for the default "
            "benchmark contract"
        ),
    )
    args = parser.parse_args()

    if args.max_selected_tools < 1:
        parser.error("--max-selected-tools must be at least 1")
    if args.max_options_per_question < 2:
        parser.error("--max-options-per-question must be at least 2")
    if not 0.0 <= args.min_probability <= 1.0:
        parser.error("--min-probability must be between 0 and 1")

    if args.publish and (
        tuple(args.tool_counts) != DEFAULT_TOOL_COUNTS
        or args.encoding != DEFAULT_ENCODING
        or args.max_selected_tools != DEFAULT_MAX_SELECTED_TOOLS
        or args.max_options_per_question != DEFAULT_MAX_OPTIONS_PER_QUESTION
        or not math.isclose(args.min_probability, DEFAULT_MIN_PROBABILITY)
    ):
        parser.error("--publish requires the default benchmark contract")

    encoding = tiktoken.get_encoding(args.encoding)
    samples = [
        measure_scenario(
            tool_count=tool_count,
            category=category,
            prompt=prompt,
            encoding=encoding,
            max_selected_tools=args.max_selected_tools,
            max_options_per_question=args.max_options_per_question,
            min_probability=args.min_probability,
        )
        for tool_count in args.tool_counts
        for category, prompt in SCENARIOS
    ]
    summary = summarize(samples, encoding_name=args.encoding)

    sample_rows = [asdict(sample) for sample in samples]
    summary_rows = [asdict(row) for row in summary]
    write_csv(
        args.output_dir / "tool-token-samples.csv",
        sample_rows,
        tuple(sample_rows[0]),
    )
    write_csv(
        args.output_dir / "tool-token-summary.csv",
        summary_rows,
        tuple(summary_rows[0]),
    )
    (args.output_dir / "tool-token-summary.json").write_text(
        json.dumps(
            {
                "methodology": {
                    "tokenizer": f"tiktoken:{args.encoding}",
                    "baseline": (
                        "Full OpenAI-style tool catalog passed to the main model. "
                        "This represents full-catalog configurations of frameworks such "
                        "as LangChain and LlamaIndex; those libraries may also "
                        "support application-defined filtering."
                    ),
                    "agent_rt": (
                        "Agent RT make_decision_tool_visibility_filter with a deterministic "
                        "Decision-model double, default 16-option chunks, and up to 8 selected "
                        "tools. No external model/API calls are made."
                    ),
                    "decision_token_accounting": (
                        "Decision-model input is tokenized separately using the same tokenizer "
                        "for comparability. It is not added to main-model prompt usage unless "
                        "the combined column is explicitly considered."
                    ),
                },
                "samples": sample_rows,
                "summary": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.publish:
        durable = Path(__file__).with_name("tool-token-measured-results.csv")
        write_csv(durable, summary_rows, tuple(summary_rows[0]))
        generate_charts()

    print(format_summary(summary))
    print(f"\nWrote {args.output_dir / 'tool-token-summary.csv'}")
    if args.publish:
        print(
            "Published "
            f"{Path(__file__).with_name('tool-token-measured-results.csv')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

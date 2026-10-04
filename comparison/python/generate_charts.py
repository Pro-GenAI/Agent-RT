#!/usr/bin/env python3
"""Generate publication-ready SVG bar charts from durable comparison CSVs."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

PALETTE = {
    "Agent RT": "#5B5BD6",
    "LangChain": "#14B8A6",
    "LangChain.js": "#14B8A6",
    "LlamaIndex": "#F59E0B",
    "LlamaIndex.TS": "#F59E0B",
}
FALLBACK = "#64748B"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, unit: str) -> str:
    if value >= 1000:
        return f"{value:,.0f}{unit}"
    if value >= 100:
        return f"{value:.0f}{unit}"
    if value >= 10:
        return f"{value:.1f}{unit}"
    return f"{value:.2f}{unit}"


def chart_svg(
    title: str,
    subtitle: str,
    rows: Iterable[tuple[str, float]],
    *,
    unit: str = " ms",
    lower_is_better: bool = True,
    width: int = 980,
) -> str:
    data = list(rows)
    height = 120 + len(data) * 72
    left = 190
    right = 110
    top = 100
    plot_w = width - left - right
    max_value = max((value for _, value in data), default=1.0) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1020"/><stop offset="1" stop-color="#151B31"/></linearGradient>',
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000" flood-opacity=".28"/></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="24" fill="url(#bg)"/>',
        f'<text x="42" y="42" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="42" y="70" fill="#94A3B8" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14">{escape(subtitle)}</text>',
    ]
    if lower_is_better:
        parts.append('<text x="938" y="42" text-anchor="end" fill="#86EFAC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" font-weight="600">lower is better ↓</text>')

    for idx, (label, value) in enumerate(data):
        y = top + idx * 72
        bar_y = y + 16
        bar_h = 28
        bar_w = max(4.0, plot_w * value / max_value)
        color = PALETTE.get(label, FALLBACK)
        parts.extend([
            f'<text x="{left - 18}" y="{bar_y + 20}" text-anchor="end" fill="#E2E8F0" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="15" font-weight="600">{escape(label)}</text>',
            f'<rect x="{left}" y="{bar_y}" width="{plot_w}" height="{bar_h}" rx="8" fill="#1E293B"/>',
            f'<rect x="{left}" y="{bar_y}" width="{bar_w:.2f}" height="{bar_h}" rx="8" fill="{color}" filter="url(#shadow)"/>',
            f'<text x="{min(left + bar_w + 12, width - 24):.2f}" y="{bar_y + 20}" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14" font-weight="700">{escape(fmt(value, unit))}</text>',
        ])

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def grouped_chart_svg(
    title: str,
    subtitle: str,
    rows: list[dict[str, str]],
    series: list[tuple[str, str]],
    *,
    unit: str = " ms",
    width: int = 1040,
) -> str:
    height = 170 + len(rows) * 105
    left = 190
    right = 100
    top = 120
    plot_w = width - left - right
    values = [float(row[field]) for row in rows for field, _ in series if row.get(field)]
    max_value = max(values, default=1.0) or 1.0
    bar_h = 22
    gap = 8

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1020"/><stop offset="1" stop-color="#151B31"/></linearGradient>',
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="24" fill="url(#bg2)"/>',
        f'<text x="42" y="42" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="42" y="70" fill="#94A3B8" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14">{escape(subtitle)}</text>',
        f'<text x="{width - 42}" y="42" text-anchor="end" fill="#86EFAC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" font-weight="600">lower is better ↓</text>',
    ]

    legend_x = 42
    for idx, (_, label) in enumerate(series):
        x = legend_x + idx * 180
        parts.extend([
            f'<rect x="{x}" y="88" width="12" height="12" rx="3" fill="{["#60A5FA", "#A78BFA", "#34D399"][idx % 3]}"/>',
            f'<text x="{x + 20}" y="99" fill="#CBD5E1" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12">{escape(label)}</text>',
        ])

    for row_index, row in enumerate(rows):
        group_y = top + row_index * 105
        framework = row["framework"]
        parts.append(
            f'<text x="{left - 18}" y="{group_y + 34}" text-anchor="end" fill="#E2E8F0" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="15" font-weight="600">{escape(framework)}</text>'
        )
        for series_index, (field, _) in enumerate(series):
            value = float(row[field])
            y = group_y + series_index * (bar_h + gap)
            bar_w = max(4.0, plot_w * value / max_value)
            color = ["#60A5FA", "#A78BFA", "#34D399"][series_index % 3]
            parts.extend([
                f'<rect x="{left}" y="{y}" width="{plot_w}" height="{bar_h}" rx="7" fill="#1E293B"/>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="7" fill="{color}"/>',
                f'<text x="{min(left + bar_w + 10, width - 24):.2f}" y="{y + 16}" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12" font-weight="700">{escape(fmt(value, unit))}</text>',
            ])

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def generate_charts() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    imports = read_csv(HERE / "measured-results.csv")
    runtime = read_csv(HERE / "runtime-measured-results.csv")
    features = read_csv(HERE / "feature-measured-results.csv")
    tokens = read_csv(HERE / "tool-token-measured-results.csv")

    (ASSETS / "python-import.svg").write_text(
        chart_svg(
            "Python cold import",
            "Median fresh-process package import time",
            [(row["framework"], float(row["median_import_ms"])) for row in imports],
        ),
        encoding="utf-8",
    )
    (ASSETS / "python-memory.svg").write_text(
        chart_svg(
            "Python cold-start memory",
            "Maximum observed resident memory during import",
            [(row["framework"], float(row["max_peak_rss_mib"])) for row in imports],
            unit=" MiB",
        ),
        encoding="utf-8",
    )
    (ASSETS / "python-runtime.svg").write_text(
        grouped_chart_svg(
            "Python runtime latency",
            "Median steady-state latency across representative agent operations",
            runtime,
            [
                ("completion_median_ms", "warm completion"),
                ("stream_first_token_median_ms", "stream first text"),
                ("five_turn_median_ms", "five-turn loop"),
            ],
        ),
        encoding="utf-8",
    )
    (ASSETS / "python-features.svg").write_text(
        grouped_chart_svg(
            "Python tool & structured-output overhead",
            "Median latency for equivalent wire-level feature work",
            features,
            [
                ("tool_roundtrip_median_ms", "tool round trip"),
                ("structured_valid_median_ms", "structured valid"),
                ("structured_repair_median_ms", "structured repair"),
            ],
        ),
        encoding="utf-8",
    )
    token_rows = [
        {
            "framework": f'{row["tool_count"]} tools',
            "all_tools_main_tokens": row["all_tools_main_tokens"],
            "agent_rt_main_tokens_avg": row["agent_rt_main_tokens_avg"],
        }
        for row in tokens
    ]
    (ASSETS / "python-tool-tokens.svg").write_text(
        grouped_chart_svg(
            "Main-model tool-schema tokens",
            "Full registered catalog vs Agent RT Decision-model selection",
            token_rows,
            [
                ("all_tools_main_tokens", "full catalog"),
                ("agent_rt_main_tokens_avg", "Agent RT selected"),
            ],
            unit=" tokens",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate_charts()

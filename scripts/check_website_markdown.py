"""Validate static website Markdown mirrors and their discoverability links."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs" / "website"
ORIGIN = "https://agent-rt.pro-genai.com"
SKIP = {"googlefbfc85f4cce24779.html"}


def _prose_without_code_fences(markdown: str) -> str:
    """Ignore fenced code when looking for broken prose formatting."""
    lines: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[0] if stripped.startswith(("~~~", chr(96) * 3)) else None
        if marker:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def readability_errors(markdown: str, page: Path) -> list[str]:
    """Catch HTML-to-Markdown artifacts that exist despite valid canonical links."""
    prose = _prose_without_code_fences(markdown)
    problems: list[str] = []
    if len(re.findall(r"^# \S", prose, re.M)) != 1:
        problems.append("expected one clear level-one heading")
    if re.search(r"(?m)^\s*-\s*$", prose):
        problems.append("standalone list marker (link or text detached from bullet)")
    if re.search(r"(?m)^/(?:Getting started|Core concepts|Guides|Reference)/", prose):
        problems.append("unformatted documentation breadcrumb")
    if re.search(r"(?m)^\[(?:Previous|Next|Back to)(?:←|[A-Z])", prose):
        problems.append("concatenated navigation label")
    if re.search(r"Pro-GenAI\d{1,2}\s+\w+\s+\d{4}\d+\s+min read", prose):
        problems.append("concatenated author, date, and reading time")
    if re.search(r"(?m)^\s*\*\*[^*\n]{3,65}\*\*\S", prose):
        problems.append("bold callout label attached to its explanation")
    if re.search(r"\[[^\]\n]{160,}\]\(https?://", prose):
        problems.append("oversized link label (likely a flattened HTML card)")
    if page.parts[0] == "docs" and len(page.parts) > 2 and "**Section:**" not in prose:
        problems.append("missing readable documentation section")
    if page.parts[0] == "blog" and len(page.parts) > 2:
        for label in ("**Author:**", "**Published:**", "**Reading time:**", "## At a glance", "- **Problem:**", "- **Solution:**"):
            if label not in prose:
                problems.append(f"missing article metadata or summary: {label}")
    return problems


def check() -> list[str]:
    errors: list[str] = []
    llms = (ROOT / "llms.txt").read_text(encoding="utf-8")
    headers = (ROOT / "_headers").read_text(encoding="utf-8")
    pages = 0
    for html in sorted(ROOT.rglob("*.html")):
        if html.name in SKIP:
            continue
        pages += 1
        rel = html.relative_to(ROOT)
        md = html.with_suffix(".md")
        md_url = ORIGIN + "/" + md.relative_to(ROOT).as_posix()
        href = "/" + md.relative_to(ROOT).as_posix()
        source = html.read_text(encoding="utf-8")
        canonical = re.search(r'<link\s+rel=["\']canonical["\'][^>]*href=["\']([^"\']+)', source, re.I)
        if not canonical:
            errors.append(f"{rel}: missing canonical HTML URL")
            continue
        main_url = canonical.group(1)
        alternate = re.search(r'<link\s+rel=["\']alternate["\'][^>]*type=["\']text/markdown["\'][^>]*>', source, re.I)
        if not alternate or f'href="{href}"' not in alternate.group(0):
            errors.append(f"{rel}: missing Markdown alternate link")
        if not re.search(r'<a\b[^>]*href=["\']' + re.escape(href) + r'["\'][^>]*>Read this page in Markdown</a>', source, re.I):
            errors.append(f"{rel}: missing visible Markdown link")
        if not md.is_file():
            errors.append(f"{rel}: missing {md.relative_to(ROOT)}")
            continue
        body = md.read_text(encoding="utf-8")
        if len(body.strip()) < 250 or main_url not in body or not re.search(r"^# ", body, re.M):
            errors.append(f"{rel}: incomplete Markdown or missing canonical backlink")
        errors.extend(f"{rel}: {issue}" for issue in readability_errors(body, rel))
        if not any(main_url + ")" in line and md_url + ")" in line for line in llms.splitlines()):
            errors.append(f"{rel}: HTML and Markdown URLs not paired in llms.txt")
        rule = href + "\n  Content-Type: text/markdown; charset=utf-8\n  Link: <" + main_url + '>; rel="canonical"'
        if rule not in headers:
            errors.append(f"{rel}: missing Markdown Content-Type/canonical header")
    if not pages:
        errors.append("no HTML pages found")
    return errors


if __name__ == "__main__":
    problems = check()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    print("Website Markdown mirrors are readable; backlinks, llms.txt and canonical headers are consistent.")

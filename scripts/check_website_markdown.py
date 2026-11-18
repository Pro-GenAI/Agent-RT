"""Validate static website Markdown mirrors and their discoverability links."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs" / "website"
ORIGIN = "https://agent-rt.pro-genai.com"
SKIP = {"googlefbfc85f4cce24779.html"}


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
    print("Website Markdown mirrors, backlinks, llms.txt and canonical headers are consistent.")

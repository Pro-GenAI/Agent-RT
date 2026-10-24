"""Validate website CSP inline-script hashes and reject third-party loaded assets.

Usage: python3 scripts/check_website_csp.py  (stdlib only; exits 1 on problems)
"""

import base64
import hashlib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

SITE = Path(__file__).resolve().parent.parent / "docs" / "website"
SITE_ORIGIN = "https://agent-rt-pro.github.io"
# Search-console verification stub, not a real page.
SKIP = {"googlefbfc85f4cce24779.html"}

# Attributes that can trigger resource requests rather than just navigation.
ASSET_ATTRIBUTES = {
    "script": ("src",),
    "link": ("href", "imagesrcset"),
    "img": ("src", "srcset"),
    "source": ("src", "srcset"),
    "iframe": ("src",),
    "video": ("src", "poster"),
    "audio": ("src",),
    "track": ("src",),
    "embed": ("src",),
    "object": ("data",),
    "input": ("src",),
}


def is_third_party(url: str) -> bool:
    """Treat absolute or protocol-relative URLs off the published origin as remote."""
    parsed = urlsplit(url.strip())
    origin = urlsplit(SITE_ORIGIN)
    if not parsed.netloc:
        return False
    scheme = parsed.scheme or origin.scheme  # //host/path inherits the page scheme.
    return (scheme.lower(), parsed.netloc.lower()) != (
        origin.scheme,
        origin.netloc,
    )


class WebsitePageParser(HTMLParser):
    """Extract the relevant CSP, inline script bodies, and asset URLs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.policies: list[str] = []
        self.inline_scripts: list[str] = []
        self.third_party_assets: list[str] = []
        self._script_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "content-security-policy":
            self.policies.append(attributes.get("content", ""))

        if tag == "script" and "src" not in attributes and attributes.get("type", "").lower() != "application/ld+json":
            self._script_parts = []

        if tag == "link" and set(attributes.get("rel", "").lower().split()) & {"canonical", "alternate"}:
            return  # Metadata links are not loaded resources.

        for name in ASSET_ATTRIBUTES.get(tag, ()):
            value = attributes.get(name)
            if not value:
                continue
            # srcset/imagesrcset carry comma-separated URL + descriptor candidates.
            urls = (
                [candidate.strip().split(maxsplit=1)[0] for candidate in value.split(",") if candidate.strip()]
                if name in {"srcset", "imagesrcset"}
                else [value]
            )
            for url in urls:
                if is_third_party(url):
                    self.third_party_assets.append(f"third-party asset {url}")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._script_parts is not None:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_parts is not None:
            self.inline_scripts.append("".join(self._script_parts))
            self._script_parts = None


def check(page: Path) -> list[str]:
    parser = WebsitePageParser()
    parser.feed(page.read_text(encoding="utf-8"))
    parser.close()
    if not parser.policies:
        return ["no Content-Security-Policy meta tag"]

    problems = list(parser.third_party_assets)
    if len(parser.policies) != 1:
        problems.append("expected exactly one Content-Security-Policy meta tag")
    allowed = set(re.findall(r"'(sha256-[^']+)'", parser.policies[0]))
    required = set()
    for body in parser.inline_scripts:
        digest = "sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        if digest not in allowed:
            problems.append(f"inline script not allowed by CSP, add '{digest}'")
        required.add(digest)
    problems += [f"stale CSP hash '{digest}'" for digest in sorted(allowed - required)]
    return problems


def main() -> int:
    failed = False
    for page in sorted(SITE.rglob("*.html")):
        if page.name in SKIP:
            continue
        for problem in check(page):
            failed = True
            print(f"{page.relative_to(SITE)}: {problem}")
    print("CSP check failed." if failed else "CSP check passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

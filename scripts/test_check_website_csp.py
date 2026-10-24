"""Regression tests for the static website's CSP and third-party asset gate."""

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from check_website_csp import check

SCRIPT = "document.documentElement.classList.add('js')"
DIGEST = "sha256-" + base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()


class WebsiteCspTests(unittest.TestCase):
    def check_html(self, markup: str = "", *, policy: str | None = None) -> list[str]:
        if policy is None:
            policy = f"script-src 'self' '{DIGEST}'"
        html = f"""<!doctype html><html><head>
<meta http-equiv='Content-Security-Policy' content="{policy}">
<script>{SCRIPT}</script>
<script type='application/ld+json'>{{"@context":"https://schema.org"}}</script>
{markup}
</head><body></body></html>"""
        with tempfile.TemporaryDirectory() as temp:
            page = Path(temp) / "test.html"
            page.write_text(html, encoding="utf-8")
            return check(page)

    def test_self_hosted_resources_and_metadata_links_are_allowed(self) -> None:
        self.assertEqual(
            [],
            self.check_html("""
<link rel='canonical' href='https://agent-rt-pro.github.io/docs/'>
<link rel=alternate href='https://example.com/translated-page'>
<link rel='stylesheet' href='https://agent-rt-pro.github.io/docs/docs.css'>
<img src=//agent-rt-pro.github.io/logo.svg srcset='../../logo.svg 1x, /logo.svg 2x'>
<script src="../../docs/docs.js"></script>
"""),
        )

    def test_single_quoted_and_unquoted_asset_urls_are_rejected(self) -> None:
        problems = self.check_html("""
<img src='https://cdn.example.org/tracker.png'>
<script SRC = 'https://cdn.example.org/script.js'></script>
<link rel=stylesheet href=https://cdn.example.org/style.css>
""")
        self.assertEqual(3, len(problems), problems)
        self.assertTrue(all(problem.startswith("third-party asset") for problem in problems))

    def test_protocol_relative_assets_are_rejected(self) -> None:
        problems = self.check_html("""
<link rel="stylesheet" href="//cdn.example.org/theme.css">
<iframe src='//embed.example.org/widget'></iframe>
""")
        self.assertEqual(2, len(problems), problems)
        self.assertIn("third-party asset //cdn.example.org/theme.css", problems)

    def test_srcset_and_imagesrcset_candidates_are_checked(self) -> None:
        problems = self.check_html("""
<img src='/local.png' srcset='/local.png 1x, https://cdn.example.org/photo.png 2x'>
<picture><source srcset='//images.example.org/a.webp 1x, /b.webp 2x'></picture>
<link rel='preload' as='image' imagesrcset='/small.png 1x, https://cdn.example.org/large.png 2x'>
""")
        self.assertEqual(3, len(problems), problems)

    def test_repeated_inline_script_uses_one_allowed_hash(self) -> None:
        self.assertEqual([], self.check_html(f"<script>{SCRIPT}</script>"))

    def test_inline_script_hash_is_still_required(self) -> None:
        problems = self.check_html(policy="script-src 'self'")
        self.assertEqual(1, len(problems), problems)
        self.assertIn("inline script not allowed", problems[0])

    def test_stale_script_hash_is_still_rejected(self) -> None:
        problems = self.check_html(policy=f"script-src 'self' '{DIGEST}' 'sha256-stale'")
        self.assertEqual(["stale CSP hash 'sha256-stale'"], problems)

    def test_missing_policy_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            page = Path(temp) / "missing.html"
            page.write_text("<script>console.log('hi')</script>", encoding="utf-8")
            self.assertEqual(["no Content-Security-Policy meta tag"], check(page))


if __name__ == "__main__":
    unittest.main()

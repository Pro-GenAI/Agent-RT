"""Regression tests for the static website's CSP and third-party asset gate."""

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from check_website_csp import check

SCRIPT = "document.documentElement.classList.add('js')"
DIGEST = "sha256-" + base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
BASE_POLICY = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; "
)


class WebsiteCspTests(unittest.TestCase):
    def check_html(
        self, markup: str = "", *, policy: str | None = None, base_policy: str = BASE_POLICY,
    ) -> list[str]:
        if policy is None:
            policy = f"script-src 'self' '{DIGEST}'"
        html = f"""<!doctype html><html><head>
<meta http-equiv='Content-Security-Policy' content="{base_policy}{policy}">
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
        problems = self.check_html(policy=f"script-src 'self' '{DIGEST}' 'sha256-{'A' * 43}='")
        self.assertEqual([f"stale CSP hash 'sha256-{'A' * 43}='"], problems)

    def test_remote_font_and_connection_policies_are_rejected(self) -> None:
        for directive, source in (
            ("font-src", "https://fonts.gstatic.com"),
            ("connect-src", "https://analytics.example.org"),
            ("default-src", "*"),
            ("form-action", "'self'"),
        ):
            with self.subTest(directive=directive):
                # The first directive wins in browsers; appending a safe duplicate
                # must not hide an unsafe policy from the checker.
                problems = self.check_html(base_policy=f"{directive} {source}; {BASE_POLICY}")
                self.assertTrue(any(f"CSP {directive} must" in p for p in problems), problems)
                self.assertIn(f"duplicate CSP directive {directive}", problems)

    def test_missing_restrictions_and_misplaced_hash_are_rejected(self) -> None:
        problems = self.check_html(base_policy="")
        for directive in ("default-src", "font-src", "connect-src", "form-action"):
            self.assertTrue(any(f"CSP {directive} must" in p for p in problems), problems)
        problems = self.check_html(
            policy=f"script-src 'self'; style-src-elem 'self' '{DIGEST}'",
        )
        self.assertTrue(any("inline script not allowed" in p for p in problems), problems)

    def test_remote_scripts_and_override_directives_are_rejected(self) -> None:
        for extra in ("https://cdn.example.org", "'unsafe-inline'", "'unsafe-eval'", "*"):
            with self.subTest(extra=extra):
                problems = self.check_html(policy=f"script-src 'self' '{DIGEST}' {extra}")
                self.assertIn(
                    "CSP script-src must allow only 'self' and sha256 hashes", problems,
                )
        problems = self.check_html(
            policy=f"script-src 'self' '{DIGEST}'; script-src-elem https://tracker.example.org",
        )
        self.assertIn("unreviewed CSP directive script-src-elem", problems)

    def test_metadata_rel_cannot_hide_a_loaded_stylesheet(self) -> None:
        problems = self.check_html(
            '<link rel="alternate stylesheet" href="https://fonts.googleapis.com/css2?family=Inter">',
        )
        self.assertEqual(1, len(problems), problems)
        self.assertIn("third-party asset", problems[0])

    def test_missing_policy_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            page = Path(temp) / "missing.html"
            page.write_text("<script>console.log('hi')</script>", encoding="utf-8")
            self.assertEqual(["no Content-Security-Policy meta tag"], check(page))


if __name__ == "__main__":
    unittest.main()

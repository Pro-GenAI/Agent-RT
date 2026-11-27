"""Regression tests for human- and answer-engine-readable Markdown mirrors."""
from __future__ import annotations

import unittest
from pathlib import Path

from check_website_markdown import readability_errors


class MarkdownReadabilityTests(unittest.TestCase):
    BASE = (
        "> **Canonical HTML page:** [Example](https://agent-rt.pro-genai.com/links/)\n\n"
        "# Example\n\nA useful page with a readable title and summary.\n"
    )

    def test_well_formed_page(self) -> None:
        self.assertEqual(readability_errors(self.BASE, Path("links/index.html")), [])

    def test_broken_author_date_and_time(self) -> None:
        bad = self.BASE + "\n Praneeth Vadlapati3 November 20267 min read\n"
        self.assertIn(
            "concatenated author, date, and reading time",
            readability_errors(bad, Path("links/index.html")),
        )

    def test_broken_cards_and_bullets(self) -> None:
        bad = self.BASE + "\n-\n\n[Example](https://example.org/)\n"
        self.assertTrue(any("standalone list" in p for p in readability_errors(bad, Path("links/index.html"))))
        bad = self.BASE + "\n[" + "Concatenated card content " * 12 + "](https://example.org/)\n"
        self.assertTrue(any("oversized link" in p for p in readability_errors(bad, Path("links/index.html"))))

    def test_breadcrumbs_navigation_and_callouts(self) -> None:
        bad = self.BASE + "\n/Guides/Runtime guide\n\n[NextQuickstart →](https://example.org/)\n\n**Note**Text\n"
        problems = readability_errors(bad, Path("links/index.html"))
        self.assertEqual(len(problems), 3)

    def test_blog_requires_explicit_metadata(self) -> None:
        problems = readability_errors(self.BASE, Path("blog/article/index.html"))
        self.assertEqual(len(problems), 6)
        good = (
            self.BASE
            + "\n**Author:** Praneeth Vadlapati  \n**Published:** 3 November 2026  \n"
            + "**Reading time:** 7 min\n\n## At a glance\n\n"
            + "- **Problem:** A risk.\n- **Solution:** Enforce policy.\n"
        )
        self.assertEqual(readability_errors(good, Path("blog/article/index.html")), [])

    def test_docs_require_section_and_code_is_ignored(self) -> None:
        self.assertTrue(readability_errors(self.BASE, Path("docs/guides/production.html")))
        good = self.BASE + "\n**Section:** Guides / Production\n\n" + chr(96) * 3 + "\n-\n/Guides/Broken\n" + chr(96) * 3 + "\n"
        self.assertEqual(readability_errors(good, Path("docs/guides/production.html")), [])


if __name__ == "__main__":
    unittest.main()

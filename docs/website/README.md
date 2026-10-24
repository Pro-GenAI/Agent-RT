# Agent RT website

Project repository:
https://github.com/Pro-GenAI/Agent-RT

The website is a dependency-free static site deployed directly from this directory.

Pages must not load fonts, scripts, styles, or images from third-party hosts, and must not add analytics or session-replay tools: every such request sends the visitor's IP address to another party without consent, which is a GDPR liability (a Munich court awarded damages over exactly this with Google Fonts). Self-host what is needed. The only vendored third-party code is highlight.js 11.12.0 (BSD-3-Clause) in `docs/vendor/`, with its license text; to upgrade it, replace the three files there from the `@highlightjs/cdn-assets` npm package.

Every page carries a `<meta http-equiv="Content-Security-Policy">` that allows only same-origin assets plus the sha256 hashes of the inline scripts (the `js` class one-liner on docs pages and the homepage motion bootstrap), so the no-third-party promise is enforced by the browser. If you edit an inline script, update its hash in that page's policy; `python3 scripts/check_website_csp.py` (from the repository root, stdlib only) prints the expected hash and also fails on stale hashes, missing policies, and third-party assets, including URLs in `srcset` and `imagesrcset`. Run `python3 -m unittest discover -s scripts -p 'test_check_website_csp.py'` to test the checker against valid and prohibited asset formats. CircleCI runs both in the `python_security` job. GitHub Pages cannot send HTTP headers, so `frame-ancestors` and `X-Frame-Options` protection is not available here.

Keep body and footer text at WCAG AA contrast (4.5:1) against its background; `--muted-2` is the dimmest allowed text token. Every page footer states the owner (Pro-GenAI), the MIT license, the no-cookies/no-analytics/no-third-party-requests promise, and that mentioned product names are trademarks of their owners with no affiliation. `docs/reference/privacy-terms.html` (linked from every footer, listed in `sitemap.xml` and `llms.txt`) distinguishes site-level data collection from GitHub Pages hosting logs; keep both disclosures accurate: if the site ever adds analytics, a form, accounts, email, or any third-party request, update that page and the footers first.

Before a release, check every page at 375px wide for sideways scroll (`document.documentElement.scrollWidth` must equal 375): wrap every docs `<table>` in `<div class="table-wrap">`, and keep decorative glows clipped by `.docs-main`. Run a link/anchor/sitemap check too, and keep `-webkit-` prefixes next to `backdrop-filter` for older Safari.

## Homepage

`robots.txt` allows all crawlers (search and AI) and points to `sitemap.xml`; `llms.txt` is a plain-text site summary for LLM crawlers. Keep both in sync with the docs pages.

`index.html`, `styles.css`, and `script.js` are homepage-only; documentation pages do not load them. The homepage's animated hero canvas, trace console, scroll effects, and pointer effects are on by default, unless the OS `prefers-reduced-motion` setting asks for reduced motion. The header/footer **Motion** toggle (with a hover/focus tooltip) lets visitors override either default, and the choice is stored in `localStorage` (`agent-rt-motion`). Any new animation must be disabled under `:root[data-motion="reduce"]`, and the page must stay fully readable without JavaScript. Benchmark figures and product claims on the homepage must stay aligned with the repository documentation.

## Documentation site

Structured web documentation lives under `docs/` and is published at `/docs/`.

Keep the documentation organized around reader tasks rather than repository layout:

- **Getting started** — installation and quickstart.
- **Core concepts** — runtime model, tools/policy, and state/memory.
- **Guides** — migration, production patterns, vector database configuration/switching, and LangChain/LlamaIndex vector-store compatibility.
- **Reference** — complete feature/default/usage reference, architecture, and contributor development.

Use `docs/docs.css` and `docs/docs.js` as the shared presentation layer for documentation pages. Keep durable technical claims aligned with the repository Markdown documents and package READMEs. Each docs page must keep a unique descriptive `<title>`, meta description, canonical URL, index/follow directives, Open Graph and Twitter metadata, and valid JSON-LD for the page plus breadcrumbs. When documentation pages are added or removed, update the documentation navigation and `sitemap.xml`.

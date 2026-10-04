# Agent RT website

Project repository:
https://github.com/Pro-GenAI/Agent-RT

The website is a dependency-free static site deployed directly from this directory.

## Homepage

`robots.txt` allows all crawlers (search and AI) and points to `sitemap.xml`; `llms.txt` is a plain-text site summary for LLM crawlers. Keep both in sync with the docs pages.

`index.html`, `styles.css`, and `script.js` are homepage-only; documentation pages do not load them. The homepage's animated hero canvas, trace console, scroll effects, and pointer effects are on by default, regardless of the OS `prefers-reduced-motion` setting. The header/footer **Motion** toggle (with a hover/focus tooltip) lets visitors turn them off, and the choice is stored in `localStorage` (`agent-rt-motion`). Any new animation must be disabled under `:root[data-motion="reduce"]`, and the page must stay fully readable without JavaScript. Benchmark figures and product claims on the homepage must stay aligned with the repository documentation.

## Documentation site

Structured web documentation lives under `docs/` and is published at `/docs/`.

Keep the documentation organized around reader tasks rather than repository layout:

- **Getting started** — installation and quickstart.
- **Core concepts** — runtime model, tools/policy, and state/memory.
- **Guides** — migration, production patterns, vector database configuration/switching, and LangChain/LlamaIndex vector-store compatibility.
- **Reference** — complete feature/default/usage reference, architecture, and contributor development.

Use `docs/docs.css` and `docs/docs.js` as the shared presentation layer for documentation pages. Keep durable technical claims aligned with the repository Markdown documents and package READMEs. Each docs page must keep a unique descriptive `<title>`, meta description, canonical URL, index/follow directives, Open Graph and Twitter metadata, and valid JSON-LD for the page plus breadcrumbs. When documentation pages are added or removed, update the documentation navigation and `sitemap.xml`.

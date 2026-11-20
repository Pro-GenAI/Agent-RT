// Same-origin download badges for Cloudflare Pages; do not expose visitor data to badge vendors.
const BADGE_URLS = Object.freeze({
  pypi: "https://static.pepy.tech/personalized-badge/agent-rt?period=total&units=NONE&left_color=BLACK&right_color=GREEN&left_text=PyPI%20Downloads",
  npm: "https://img.shields.io/npm/dt/agent-rt?style=for-the-badge&label=npm%20downloads",
});

const RESPONSE_HEADERS = {
  "Content-Type": "image/svg+xml; charset=utf-8",
  "X-Content-Type-Options": "nosniff",
  "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
};

export async function onRequestGet({ params }) {
  const source = BADGE_URLS[params.badge];
  if (!source) return new Response("Not found", { status: 404 });

  try {
    // Upstream badge data is shared and cacheable; requests contain no visitor details.
    const upstream = await fetch(source, { cf: { cacheEverything: true, cacheTtl: 21600 } });
    if (!upstream.ok || !upstream.headers.get("content-type")?.includes("image/svg+xml")) {
      throw new Error("Badge provider returned a non-SVG response");
    }

    const svg = await upstream.text();
    if (svg.length > 32768 || !svg.includes("<svg") || /<script\b|<foreignObject\b|<iframe\b/i.test(svg)) {
      throw new Error("Badge provider returned an invalid SVG");
    }

    return new Response(svg, {
      headers: { ...RESPONSE_HEADERS, "Cache-Control": "public, max-age=3600, s-maxage=3600" },
    });
  } catch {
    const fallback = '<svg xmlns="http://www.w3.org/2000/svg" width="210" height="28" role="img" aria-label="Download count unavailable"><rect width="210" height="28" rx="4" fill="#30343b"/><text x="105" y="19" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#fff">Download count unavailable</text></svg>';
    return new Response(fallback, {
      headers: { ...RESPONSE_HEADERS, "Cache-Control": "public, max-age=60" },
    });
  }
}

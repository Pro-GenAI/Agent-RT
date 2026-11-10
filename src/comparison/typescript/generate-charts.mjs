#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = resolve(HERE, "../assets");
const COLORS = { "Agent RT": "#5B5BD6", "LangChain.js": "#14B8A6", "LlamaIndex.TS": "#F59E0B" };
const FALLBACK = "#64748B";

function escapeXml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function parseCsv(path) {
  const rows = readFileSync(path, "utf8").trim().split(/\r?\n/);
  const headers = rows.shift().split(",");
  return rows.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function formatValue(value, unit) {
  if (value >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 }) + unit;
  if (value >= 100) return value.toFixed(0) + unit;
  if (value >= 10) return value.toFixed(1) + unit;
  return value.toFixed(2) + unit;
}

function chartSvg(title, subtitle, rows, options = {}) {
  const unit = options.unit ?? " ms";
  const width = options.width ?? 980;
  const height = 120 + rows.length * 72;
  const left = 190, right = 110, top = 100, plotWidth = width - left - right;
  const maxValue = Math.max(...rows.map((row) => row[1]), 1);
  const parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="' + escapeXml(title) + '">',
    "<defs>",
    '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1020"/><stop offset="1" stop-color="#151B31"/></linearGradient>',
    '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000" flood-opacity=".28"/></filter>',
    "</defs>",
    '<rect width="' + width + '" height="' + height + '" rx="24" fill="url(#bg)"/>',
    '<text x="42" y="42" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="24" font-weight="700">' + escapeXml(title) + "</text>",
    '<text x="42" y="70" fill="#94A3B8" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14">' + escapeXml(subtitle) + "</text>",
    '<text x="' + (width - 42) + '" y="42" text-anchor="end" fill="#86EFAC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" font-weight="600">lower is better ↓</text>',
  ];
  rows.forEach(([label, value], index) => {
    const barY = top + index * 72 + 16;
    const barWidth = Math.max(4, plotWidth * value / maxValue);
    const color = COLORS[label] ?? FALLBACK;
    parts.push(
      '<text x="' + (left - 18) + '" y="' + (barY + 20) + '" text-anchor="end" fill="#E2E8F0" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="15" font-weight="600">' + escapeXml(label) + "</text>",
      '<rect x="' + left + '" y="' + barY + '" width="' + plotWidth + '" height="28" rx="8" fill="#1E293B"/>',
      '<rect x="' + left + '" y="' + barY + '" width="' + barWidth.toFixed(2) + '" height="28" rx="8" fill="' + color + '" filter="url(#shadow)"/>',
      '<text x="' + Math.min(left + barWidth + 12, width - 24).toFixed(2) + '" y="' + (barY + 20) + '" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14" font-weight="700">' + escapeXml(formatValue(value, unit)) + "</text>",
    );
  });
  parts.push("</svg>");
  return parts.join("\\n") + "\\n";
}

function groupedChartSvg(title, subtitle, rows, series, options = {}) {
  const unit = options.unit ?? " ms";
  const width = options.width ?? 1040;
  const height = 170 + rows.length * 105;
  const left = 190, right = 100, top = 120, plotWidth = width - left - right;
  const maxValue = Math.max(...rows.flatMap((row) => series.map(([field]) => Number(row[field]))), 1);
  const colors = ["#60A5FA", "#A78BFA", "#34D399"];
  const parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="' + escapeXml(title) + '">',
    "<defs>",
    '<linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1020"/><stop offset="1" stop-color="#151B31"/></linearGradient>',
    "</defs>",
    '<rect width="' + width + '" height="' + height + '" rx="24" fill="url(#bg2)"/>',
    '<text x="42" y="42" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="24" font-weight="700">' + escapeXml(title) + "</text>",
    '<text x="42" y="70" fill="#94A3B8" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14">' + escapeXml(subtitle) + "</text>",
    '<text x="' + (width - 42) + '" y="42" text-anchor="end" fill="#86EFAC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" font-weight="600">lower is better ↓</text>',
  ];
  series.forEach((entry, index) => {
    const x = 42 + index * 180;
    parts.push(
      '<rect x="' + x + '" y="88" width="12" height="12" rx="3" fill="' + colors[index % colors.length] + '"/>',
      '<text x="' + (x + 20) + '" y="99" fill="#CBD5E1" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12">' + escapeXml(entry[1]) + "</text>",
    );
  });
  rows.forEach((row, rowIndex) => {
    const groupY = top + rowIndex * 105;
    parts.push('<text x="' + (left - 18) + '" y="' + (groupY + 34) + '" text-anchor="end" fill="#E2E8F0" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="15" font-weight="600">' + escapeXml(row.framework) + "</text>");
    series.forEach(([field], seriesIndex) => {
      const value = Number(row[field]);
      const y = groupY + seriesIndex * 30;
      const barWidth = Math.max(4, plotWidth * value / maxValue);
      const color = colors[seriesIndex % colors.length];
      parts.push(
        '<rect x="' + left + '" y="' + y + '" width="' + plotWidth + '" height="22" rx="7" fill="#1E293B"/>',
        '<rect x="' + left + '" y="' + y + '" width="' + barWidth.toFixed(2) + '" height="22" rx="7" fill="' + color + '"/>',
        '<text x="' + Math.min(left + barWidth + 10, width - 24).toFixed(2) + '" y="' + (y + 16) + '" fill="#F8FAFC" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12" font-weight="700">' + escapeXml(formatValue(value, unit)) + "</text>",
      );
    });
  });
  parts.push("</svg>");
  return parts.join("\\n") + "\\n";
}

export function generateCharts() {
  mkdirSync(ASSETS, { recursive: true });
  const imports = parseCsv(resolve(HERE, "measured-results.csv"));
  const runtime = parseCsv(resolve(HERE, "runtime-measured-results.csv"));
  const features = parseCsv(resolve(HERE, "feature-measured-results.csv"));
  writeFileSync(resolve(ASSETS, "typescript-import.svg"), chartSvg("TypeScript cold module load", "Median fresh-process module-load time", imports.map((row) => [row.framework, Number(row.median_import_ms)])));
  writeFileSync(resolve(ASSETS, "typescript-memory.svg"), chartSvg("TypeScript cold-start memory", "Maximum observed resident memory during module load", imports.map((row) => [row.framework, Number(row.max_observed_rss_mib)]), { unit: " MiB" }));
  writeFileSync(resolve(ASSETS, "typescript-runtime.svg"), groupedChartSvg("TypeScript runtime latency", "Median steady-state latency across representative agent operations", runtime, [["completion_median_ms", "warm completion"], ["stream_first_token_median_ms", "stream first text"], ["five_turn_median_ms", "five-turn loop"]]));
  writeFileSync(resolve(ASSETS, "typescript-features.svg"), groupedChartSvg("TypeScript tool & structured-output overhead", "Median latency for equivalent wire-level feature work", features, [["tool_roundtrip_median_ms", "tool round trip"], ["structured_valid_median_ms", "structured valid"], ["structured_repair_median_ms", "structured repair"]]));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) generateCharts();

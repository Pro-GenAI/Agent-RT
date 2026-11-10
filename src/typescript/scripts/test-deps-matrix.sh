#!/usr/bin/env bash
# Run the regression suite against representative optional-peer SDK versions
# (`openai`, `@anthropic-ai/sdk`) released since mid-2024.
#
# The package is built once from the locked dependencies, then each profile
# runs the compiled suite from a disposable copy whose node_modules holds the
# profile's SDK versions, so the locked install used by `make test` is never
# touched.
#
#   DEPS_PROFILES="mid-2024 latest"  run a subset of profiles (default: all)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

# name|space-separated package@version pins. Keep the first profile at the
# oldest supported mid-2024 releases and the last at the newest releases.
profile_specs=(
  "mid-2024|openai@4.48.1 @anthropic-ai/sdk@0.23.0"
  "late-2024|openai@4.77.0 @anthropic-ai/sdk@0.33.0"
  "mid-2025|openai@5.8.0 @anthropic-ai/sdk@0.55.0"
  "late-2025|openai@6.9.0 @anthropic-ai/sdk@0.69.0"
  "mid-2026|openai@6.49.0 @anthropic-ai/sdk@0.115.0"
  "latest|openai@7.32.0 @anthropic-ai/sdk@0.133.0"
)

selected=()
if [[ -n "${DEPS_PROFILES:-}" ]]; then
  read -r -a selected <<< "${DEPS_PROFILES}"
fi

names=()
pins=()
for spec in "${profile_specs[@]}"; do
  name="${spec%%|*}"
  if (( ${#selected[@]} > 0 )) && [[ " ${selected[*]} " != *" ${name} "* ]]; then
    continue
  fi
  names+=("${name}")
  pins+=("${spec#*|}")
done
if (( ${#selected[@]} > 0 && ${#names[@]} != ${#selected[@]} )); then
  echo "Unknown profile in DEPS_PROFILES='${DEPS_PROFILES}'." >&2
  echo "Available: $(printf '%s ' "${profile_specs[@]%%|*}")" >&2
  exit 2
fi

work_dir="$(mktemp -d)"
statuses=()
passed=0
failed=0

print_summary() {
  local status=$?
  trap - EXIT
  rm -rf "${work_dir}"
  echo
  echo "Node dependency matrix summary (Node $(node -p 'process.versions.node'))"
  echo "-----------------------------------------------------"
  for index in "${!names[@]}"; do
    printf '  %-10s  %-7s  %s\n' "${names[index]}" "${statuses[index]:-NOT RUN}" "${pins[index]}"
  done
  printf '  Total: %d passed, %d failed\n' "${passed}" "${failed}"
  if (( failed > 0 || status != 0 )); then
    exit 1
  fi
}
trap print_summary EXIT

# Print the final Node test totals rather than thousands of individual TAP
# entries. On failure retain more context for diagnosis.
run_tests() {
  local log
  log="$(mktemp)"
  if "$@" >"${log}" 2>&1; then
    tail -n 12 "${log}"
    rm -f "${log}"
  else
    tail -n 80 "${log}"
    rm -f "${log}"
    return 1
  fi
}

# Other targets (e.g. `make test-matrix`, which runs `npm ci --omit=optional`
# under several Node versions) can leave node_modules without the TypeScript
# toolchain, so verify the compiler itself rather than just the directory.
if [[ ! -x node_modules/.bin/tsc ]]; then
  echo "TypeScript toolchain missing from node_modules; running 'npm ci'." >&2
  npm ci --no-audit --no-fund
fi
echo "==> Building from locked dependencies"
npm run build

# Keep only what the compiled suite needs: runtime dependencies plus the test
# double, with the profile's SDKs as devDependencies.
write_manifest() {
  node - "$1" "${@:2}" <<'NODE'
const fs = require('node:fs');
const [target, ...pins] = process.argv.slice(2);
const source = JSON.parse(fs.readFileSync('package.json', 'utf8'));
const devDependencies = {
  '@copilotkit/aimock': source.devDependencies['@copilotkit/aimock'],
};
for (const pin of pins) {
  const at = pin.lastIndexOf('@');
  devDependencies[pin.slice(0, at)] = pin.slice(at + 1);
}
const manifest = {
  name: source.name,
  version: source.version,
  private: true,
  main: source.main,
  types: source.types,
  exports: source.exports,
  // dist/ is prebuilt from the same sources; tests that shell out to
  // `npm run build` must not need the TypeScript toolchain here.
  scripts: { build: 'node -e 0', 'test:run': source.scripts['test:run'] },
  dependencies: source.dependencies,
  devDependencies,
};
fs.writeFileSync(target, JSON.stringify(manifest, null, 2) + '\n');
NODE
}

for index in "${!names[@]}"; do
  name="${names[index]}"
  read -r -a profile_pins <<< "${pins[index]}"
  copy="${work_dir}/${name}"
  echo "==> ${name}: ${pins[index]}"
  mkdir -p "${copy}"
  cp -R dist tests scripts "${copy}/"
  if write_manifest "${copy}/package.json" "${profile_pins[@]}" && \
    (cd "${copy}" && npm install --no-audit --no-fund --ignore-scripts \
      --omit=optional --loglevel=error) && \
    (cd "${copy}" && node -p "['openai', '@anthropic-ai/sdk'].map((name) => \
      name + ' ' + JSON.parse(require('node:fs').readFileSync( \
        'node_modules/' + name + '/package.json', 'utf8')).version).join('\n')") && \
    (cd "${copy}" && run_tests npm run test:run); then
    statuses+=("PASS")
    passed=$((passed + 1))
  else
    statuses+=("FAIL")
    failed=$((failed + 1))
  fi
  rm -rf "${copy}"
done

if (( failed > 0 )); then
  exit 1
fi

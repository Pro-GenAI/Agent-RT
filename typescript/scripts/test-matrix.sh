#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

read -r -a versions <<< "${NODE_VERSIONS:-20 22 24 26}"
statuses=()
passed=0
failed=0

print_summary() {
  local status=$?
  trap - EXIT
  echo
  echo "Node test matrix summary"
  echo "------------------------"
  for index in "${!versions[@]}"; do
    printf '  Node %-4s  %s\n' "${versions[index]}" "${statuses[index]:-NOT RUN}"
  done
  printf '  Total: %d passed, %d failed\n' "${passed}" "${failed}"
  if (( failed > 0 || status != 0 )); then
    exit 1
  fi
}
trap print_summary EXIT

# Print the final Node test totals rather than thousands of individual TAP entries.
# On failure retain more context for diagnosis.
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

NVM_FLAVOR="unix"
if [[ -n "${NVM_SYMLINK:-}" ]] || [[ "${OSTYPE:-}" == msys* ]] || [[ "${OSTYPE:-}" == cygwin* ]]; then
  if command -v nvm >/dev/null 2>&1; then
    NVM_FLAVOR="windows"
  fi
fi

if [[ "${NVM_FLAVOR}" == "unix" ]]; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
    if command -v nvm >/dev/null 2>&1; then
      NVM_FLAVOR="windows"
    else
      echo "nvm not found. Expected Unix nvm at ${NVM_DIR}/nvm.sh or nvm-windows on PATH." >&2
      exit 1
    fi
  else
    # shellcheck source=/dev/null
    . "${NVM_DIR}/nvm.sh"
  fi
fi

if [[ "${NVM_FLAVOR}" == "windows" ]]; then
  original_node="$(node -p "process.versions.node" 2>/dev/null || true)"

  restore_node() {
    if [[ -n "${original_node}" ]]; then
      nvm use "${original_node}" >/dev/null 2>&1 || true
    fi
  }
  trap 'restore_node; print_summary' EXIT

  for version in "${versions[@]}"; do
    echo "==> Node ${version} (nvm-windows)"
    if nvm install "${version}" && \
      nvm use "${version}" && \
      npm ci --omit=optional --ignore-scripts && \
      run_tests npm test; then
      statuses+=("PASS")
      passed=$((passed + 1))
    else
      statuses+=("FAIL")
      failed=$((failed + 1))
    fi
  done
else
  for version in "${versions[@]}"; do
    echo "==> Node ${version}"
    if nvm install "${version}" --no-progress >/dev/null && \
      nvm exec "${version}" npm ci --omit=optional --ignore-scripts && \
      run_tests nvm exec "${version}" npm test; then
      statuses+=("PASS")
      passed=$((passed + 1))
    else
      statuses+=("FAIL")
      failed=$((failed + 1))
    fi
  done
fi

if (( failed > 0 )); then
  exit 1
fi

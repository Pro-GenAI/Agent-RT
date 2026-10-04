#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

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

read -r -a versions <<< "${NODE_VERSIONS:-20 22 24 26}"

if [[ "${NVM_FLAVOR}" == "windows" ]]; then
  original_node="$(node -p "process.versions.node" 2>/dev/null || true)"

  restore_node() {
    if [[ -n "${original_node}" ]]; then
      nvm use "${original_node}" >/dev/null 2>&1 || true
    fi
  }
  trap restore_node EXIT

  for version in "${versions[@]}"; do
    echo "==> Node ${version} (nvm-windows)"
    nvm install "${version}"
    nvm use "${version}"
    npm ci --omit=optional --ignore-scripts
    npm test
  done
else
  for version in "${versions[@]}"; do
    echo "==> Node ${version}"
    nvm install "${version}" --no-progress >/dev/null
    nvm exec "${version}" npm ci --omit=optional --ignore-scripts
    nvm exec "${version}" npm test
  done
fi

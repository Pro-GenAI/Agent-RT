#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

read -r -a versions <<< "${PYTHON_VERSIONS:-3.10 3.11 3.12 3.13 3.14}"

for version in "${versions[@]}"; do
  echo "==> Python ${version}"
  uv python install "${version}"
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
    -u CONDA_SHLVL -u VIRTUAL_ENV -u PYTHONHOME \
    uv run --locked --isolated --python "${version}" --extra dev \
      pytest -q
done

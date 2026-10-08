#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

read -r -a versions <<< "${PYTHON_VERSIONS:-3.10 3.11 3.12 3.13 3.14}"

statuses=()
passed=0
failed=0

print_summary() {
  local status=$?
  trap - EXIT
  echo
  echo "Python test matrix summary"
  echo "--------------------------"
  for index in "${!versions[@]}"; do
    printf '  Python %-4s  %s\n' "${versions[index]}" "${statuses[index]:-NOT RUN}"
  done
  printf '  Total: %d passed, %d failed\n' "${passed}" "${failed}"
  if (( failed > 0 || status != 0 )); then
    exit 1
  fi
}
trap print_summary EXIT

for version in "${versions[@]}"; do
  echo "==> Python ${version}"
  if uv python install "${version}" && \
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
      -u CONDA_SHLVL -u VIRTUAL_ENV -u PYTHONHOME \
      uv run --locked --isolated --python "${version}" --extra dev \
        pytest -q; then
    statuses+=("PASS")
    passed=$((passed + 1))
  else
    statuses+=("FAIL")
    failed=$((failed + 1))
  fi
done

if (( failed > 0 )); then
  exit 1
fi

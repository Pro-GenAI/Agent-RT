#!/usr/bin/env bash
# Run the regression suite against representative third-party dependency
# versions (vendor SDKs and HTTP/WebSocket stacks) released since mid-2024.
#
# Each profile gets a disposable virtual environment resolved from the
# package's `dev` extra with the profile's pins applied as uv overrides, so
# the locked development environment used by `make test` is never touched.
#
#   DEPS_PROFILES="mid-2024 latest"  run a subset of profiles (default: all)
#   DEPS_PYTHON=3.12                 interpreter used for every profile
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_DIR}"

# name|space-separated pins. Keep the first profile at the declared
# pyproject.toml floors and the last at the newest releases.
profile_specs=(
  "mid-2024|openai==1.31.0 anthropic==0.29.0 httpx==0.27.0 httpcore==1.0.5 websockets==13.0 requests==2.32.3"
  "late-2024|openai==1.58.1 anthropic==0.42.0 httpx==0.28.1 websockets==14.1"
  "mid-2025|openai==1.93.0 anthropic==0.57.1 websockets==15.0.1"
  "late-2025|openai==2.8.0 anthropic==0.73.0 websockets==15.0.1"
  "mid-2026|openai==2.54.0 anthropic==0.125.0 websockets==16.1.1"
  "latest|openai==3.28.0 anthropic==1.13.0 websockets==17.2"
)

python_version="${DEPS_PYTHON:-3.12}"
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
  echo "Python dependency matrix summary (Python ${python_version})"
  echo "--------------------------------------------------------"
  for index in "${!names[@]}"; do
    printf '  %-10s  %-7s  %s\n' "${names[index]}" "${statuses[index]:-NOT RUN}" "${pins[index]}"
  done
  printf '  Total: %d passed, %d failed\n' "${passed}" "${failed}"
  if (( failed > 0 || status != 0 )); then
    exit 1
  fi
}
trap print_summary EXIT

clean_env=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER
  -u CONDA_SHLVL -u VIRTUAL_ENV -u PYTHONHOME)

for index in "${!names[@]}"; do
  name="${names[index]}"
  venv="${work_dir}/${name}"
  overrides="${work_dir}/${name}.overrides.txt"
  tr ' ' '\n' <<< "${pins[index]}" > "${overrides}"
  venv_python="${venv}/bin/python"
  if [[ "${OS:-}" == "Windows_NT" ]]; then
    venv_python="${venv}/Scripts/python.exe"
  fi
  echo "==> ${name}: ${pins[index]}"
  if uv python install "${python_version}" && \
    "${clean_env[@]}" uv venv --quiet --python "${python_version}" "${venv}" && \
    "${clean_env[@]}" uv pip install --quiet --python "${venv}" \
      --override "${overrides}" -e ".[dev]" && \
    "${clean_env[@]}" uv pip list --python "${venv}" 2>/dev/null | \
      grep -iE '^(openai|anthropic|httpx|httpcore|websockets|requests) ' && \
    "${clean_env[@]}" VIRTUAL_ENV="${venv}" "${venv_python}" -m pytest -q \
      -n auto --dist worksteal -p no:cacheprovider; then
    statuses+=("PASS")
    passed=$((passed + 1))
  else
    statuses+=("FAIL")
    failed=$((failed + 1))
  fi
  rm -rf "${venv}"
done

if (( failed > 0 )); then
  exit 1
fi

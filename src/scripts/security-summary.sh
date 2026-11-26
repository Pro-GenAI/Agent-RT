#!/usr/bin/env bash
set -uo pipefail

if (( $# == 0 || $# % 2 != 0 )); then
  printf 'Usage: %s LABEL COMMAND [LABEL COMMAND ...]\n' "$0" >&2
  exit 2
fi

red=''
reset=''
if [[ -t 1 && -z "${NO_COLOR:-}" ]] || [[ "${FORCE_COLOR:-0}" == 1 ]]; then
  red=$'\033[1;31m'
  reset=$'\033[0m'
fi

labels=()
statuses=()
passed=0
failed=0

print_summary() {
  local status=$?
  trap - EXIT
  printf '\nSecurity summary\n================\n'
  for index in "${!labels[@]}"; do
    if [[ "${statuses[index]}" == PASS ]]; then
      printf '  PASS  %s\n' "${labels[index]}"
    else
      printf '  %sFAIL  %s%s\n' "$red" "${labels[index]}" "$reset"
    fi
  done
  printf 'Total: %d passed, %d failed\n' "$passed" "$failed"
  if (( failed > 0 || status != 0 )); then
    printf '%sSecurity checks FAILED%s\n' "$red" "$reset"
    exit 1
  fi
  printf 'Security checks PASSED\n'
}
trap print_summary EXIT

while (( $# > 0 )); do
  label=$1
  command=$2
  shift 2
  labels+=("$label")
  printf '\n==> %s\n' "$label"
  if bash -c "$command"; then
    statuses+=(PASS)
    passed=$((passed + 1))
  else
    statuses+=(FAIL)
    failed=$((failed + 1))
    printf '%sFAILED: %s%s\n' "$red" "$label" "$reset"
  fi
done

if (( failed > 0 )); then
  exit 1
fi

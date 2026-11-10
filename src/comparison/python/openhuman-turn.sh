#!/usr/bin/env bash
# One cold OpenHuman turn: boot the headless core, send one agent_chat turn, stop.
# Mirrors openhuman-benchmarks bundles/adapters/openhuman.sh + bundles/oh-turn.mjs,
# because the release's one-shot `openhuman inference agent_chat` CLI fails
# ("session has no hosted authority") without a booted core.
# Usage: openhuman-turn.sh <openhuman-core> <inference_url> <api_key> <model> <message>
#        openhuman-turn.sh <openhuman-core> --version   (the core has no --version flag)
set -euo pipefail
core_bin=$1

# One core owns one state dir; parallel launches sharing ~/.openhuman lose their turns.
export HOME="$(mktemp -d "${HOME:-/tmp}/oh-XXXXXX")"
export OPENHUMAN_WORKSPACE="$HOME/oh-workspace"
mkdir -p "$OPENHUMAN_WORKSPACE"
# Same config as OpenHuman's own benchmark: no embeddings/memory engine to provision.
printf 'embeddings_provider = "none"\n\n[memory]\nengine = ""\nembedding_provider = "none"\n\n[memory.recall]\nenabled = false\n\n[memory.conversations]\nenabled = false\n\n[search]\nenabled = false\n' \
  > "$OPENHUMAN_WORKSPACE/config.toml"
export OPENHUMAN_COMPOSIO_MODE=disabled OPENHUMAN_ACTION_DIR="$PWD"
export OPENHUMAN_KEYRING_BACKEND=encrypted_file
export OPENHUMAN_KEYRING_MASTER_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
export OPENHUMAN_CORE_TOKEN=bench-core-token
# shortcut: random port can collide (~8 of 20000); upgrade to a probed free port if batches flake.
port=$((20000 + RANDOM % 20000))

rpc() {
  curl -sS --max-time 300 -H 'content-type: application/json' -H "authorization: Bearer $OPENHUMAN_CORE_TOKEN" \
    --data "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"$1\",\"params\":$2}" "http://127.0.0.1:$port/rpc"
}

if [ "${2:-}" != "--version" ]; then
  url=$2 key=$3 model=$4 message=$5
  export OPENHUMAN_BACKEND_API_KEY="$key"
fi
"$core_bin" run --headless-api --host 127.0.0.1 --port "$port" >"$HOME/core.log" 2>&1 &
core=$!
trap 'kill $core 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
  kill -0 $core 2>/dev/null || { cat "$HOME/core.log" >&2; exit 3; }
  sleep 0.01
done

if [ "${2:-}" = "--version" ]; then
  rpc openhuman.update_version '{}' | sed -n 's/.*"version":"\([^"]*\)".*/openhuman-core \1/p'
  echo
  exit
fi
rpc openhuman.inference_agent_chat \
  "{\"message\":\"$message\",\"cwd\":\"$PWD\",\"inference_url\":\"$url\",\"api_key\":\"$key\",\"model_override\":\"$model\"}" \
  | tee "$HOME/reply.json"
echo
! grep -q '"error"' "$HOME/reply.json"

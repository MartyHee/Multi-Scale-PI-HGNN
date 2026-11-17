#!/usr/bin/env bash
# Config check / dry-run (no training). Usage:
#   bash server_ops/three_topology_v2/check_config.sh [job.yaml|BATCH_ID] [--deps]
# Validates server_config.sh (paths/Python), optionally a job YAML, and shows
# caller CUDA visibility. Does NOT require four GPUs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
DEPS=0; ARG=""
for a in "$@"; do
  case "$a" in
    --deps) DEPS=1 ;;
    *) ARG="$a" ;;
  esac
done
print_resolved_config

BATCH=""
if [ -n "$ARG" ]; then
  case "$ARG" in
    *.yaml|*.yml)
      echo "== job yaml ($ARG) =="
      "$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --summary "$ARG"
      BATCH=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$ARG")
      echo "== frozen batch manifest =="
      "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$ARG" --verify || true
      ;;
    *) BATCH="$ARG" ;;
  esac
fi
if [ -n "$BATCH" ]; then
  PF="$PROJECT_ROOT/outputs/three_topology_v2/preflight/$BATCH/PREFLIGHT_PASS.json"
  if [ -f "$PF" ]; then echo "preflight_pass: YES ($PF)"; else
    echo "preflight_pass: NO — run: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" \"$ARG\""
  fi
fi
if [ "$DEPS" = "1" ]; then
  echo "== dependency/operator check (caller CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}') =="
  "$PYTHON" -B "$THREE_TOPO_HERE/check_deps.py"
fi
echo "CONFIG OK (no training started)"

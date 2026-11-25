#!/usr/bin/env bash
# Shared loader for the FE v2 (9,216-case) server interface.
#
# The caller selects the GPU with CUDA_VISIBLE_DEVICES; this file never sets,
# clears or expands it.  server_config.sh holds only server-wide paths/threads.
# Local tests may override with THREE_TOPO_SERVER_CONFIG.

THREE_TOPO_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_THREE_TOPO_DERIVED_ROOT="$(cd "$THREE_TOPO_HERE/../.." && pwd)"
CONFIG_FILE="${THREE_TOPO_SERVER_CONFIG:-$THREE_TOPO_HERE/server_config.sh}"

_tt_die() {
  echo "[CONFIG ERROR] $*" >&2
  echo "Fix: cp \"$THREE_TOPO_HERE/server_config.example.sh\" \"$THREE_TOPO_HERE/server_config.sh\"" >&2
  echo "     then edit PACKAGE_ROOT / PROCESSED_ROOT / PYTHON / THREADS." >&2
  exit 2
}

[ -f "$CONFIG_FILE" ] || _tt_die "config file not found: $CONFIG_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${PROJECT_ROOT:=$_THREE_TOPO_DERIVED_ROOT}"
: "${THREADS:=4}"
: "${PYTHON:?PYTHON not set in $CONFIG_FILE}"
: "${PACKAGE_ROOT:?PACKAGE_ROOT not set in $CONFIG_FILE}"
: "${PROCESSED_ROOT:?PROCESSED_ROOT not set in $CONFIG_FILE}"
[ -f "$PROJECT_ROOT/scripts/three_topology_fe_v2/train_fe_v2.py" ] \
  || _tt_die "PROJECT_ROOT does not look like the model project: $PROJECT_ROOT"
[ -f "$PACKAGE_ROOT/samples.jsonl" ] \
  || _tt_die "samples.jsonl not found under PACKAGE_ROOT=$PACKAGE_ROOT"
[ -f "$PACKAGE_ROOT/dataset_manifest.json" ] \
  || _tt_die "dataset_manifest.json not found under PACKAGE_ROOT=$PACKAGE_ROOT"
[ -f "$PROCESSED_ROOT/splits/protocol_v1.json" ] \
  || _tt_die "protocol_v1.json not found under PROCESSED_ROOT=$PROCESSED_ROOT"
[ -x "$PYTHON" ] || _tt_die "PYTHON not executable: $PYTHON"
case "$THREADS" in ''|*[!0-9]*) _tt_die "THREADS must be an integer, got '$THREADS'" ;; esac

export PROJECT_ROOT PACKAGE_ROOT PROCESSED_ROOT PYTHON THREADS
export THREE_TOPO_PACKAGE_ROOT="$PACKAGE_ROOT"
export THREE_TOPO_PROCESSED_ROOT="$PROCESSED_ROOT"
export THREE_TOPO_SERVER_CONFIG="$CONFIG_FILE"
export THREE_TOPO_MANIFEST_ROOT="${THREE_TOPO_MANIFEST_ROOT:-$PROJECT_ROOT/remote_jobs/three_topology_fe_v2/manifests}"
export THREE_TOPO_PREFLIGHT_ROOT="${THREE_TOPO_PREFLIGHT_ROOT:-$PROJECT_ROOT/outputs/three_topology_fe_v2/preflight}"
export THREE_TOPO_FORMAL_OUTPUT_ROOT="${THREE_TOPO_FORMAL_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/three_topology_fe_v2/formal}"
export THREE_TOPO_USE_SHARED="${USE_SHARED:-1}"
export THREE_TOPO_SHARD_CACHE="${SHARD_CACHE:-32}"

visible_gpu_count() {
  "$PYTHON" - <<'PY' 2>/dev/null | tail -n 1
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

require_one_visible_gpu() {
  local n
  n=$(visible_gpu_count); n="${n:-0}"
  if [ "$n" -lt 1 ]; then
    echo "[GPU ERROR] no CUDA device visible. Select one at command time, e.g." >&2
    echo "  CUDA_VISIBLE_DEVICES=<gpu_index> bash server_ops/three_topology_fe_v2/run_job.sh <job.yaml>" >&2
    exit 5
  fi
  echo "[GPU] visible_device_count=$n CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}'"
}

# Batch-scoped preflight gate.  BATCH_ID must be exported from the job YAML.
require_preflight() {
  : "${BATCH_ID:?BATCH_ID not set (run via run_job.sh / job YAML)}"
  if [ "${THREE_TOPO_SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "[PRECHECK] skipped (THREE_TOPO_SKIP_PREFLIGHT=1) — local test mode only" >&2
    return 0
  fi
  local pf="$THREE_TOPO_PREFLIGHT_ROOT/$BATCH_ID/PREFLIGHT_PASS.json"
  if [ ! -f "$pf" ]; then
    echo "[PRECHECK] missing preflight PASS for batch '$BATCH_ID'" >&2
    echo "Run exactly: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" <job.yaml>" >&2
    exit 4
  fi
  "$PYTHON" -B "$THREE_TOPO_HERE/_preflight_gate.py" --check \
    --batch-id "$BATCH_ID" --project-root "$PROJECT_ROOT" --pass-file "$pf" >&2 || exit 4
}

print_resolved_config() {
  echo "config_file : $CONFIG_FILE"
  echo "project_root: $PROJECT_ROOT"
  echo "package_root: $PACKAGE_ROOT"
  echo "processed   : $PROCESSED_ROOT"
  echo "python      : $PYTHON"
  echo "threads     : $THREADS"
  echo "manifests   : $THREE_TOPO_MANIFEST_ROOT"
  echo "cuda_visible: ${CUDA_VISIBLE_DEVICES-<unset>}"
  echo "visible_gpus: $(visible_gpu_count)"
}

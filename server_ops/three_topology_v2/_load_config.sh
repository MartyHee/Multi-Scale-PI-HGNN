#!/usr/bin/env bash
# Shared server config loader (sourced). Locates project from this script's
# location; loads server_config.sh (DATA_ROOT / PYTHON / THREADS only).
# GPU is chosen by the caller's CUDA_VISIBLE_DEVICES at command time; this
# loader neither sets, clears nor expands it. Task model/seed/batch live in the
# job YAML parsed by run_job.sh. No BATCH_ID/GPU_* globals here.
#
# Local tests: THREE_TOPO_SERVER_CONFIG=<alt config>.

THREE_TOPO_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_THREE_TOPO_DERIVED_ROOT="$(cd "$THREE_TOPO_HERE/../.." && pwd)"
CONFIG_FILE="${THREE_TOPO_SERVER_CONFIG:-$THREE_TOPO_HERE/server_config.sh}"

_tt_die() {
  echo "[CONFIG ERROR] $*" >&2
  echo "Fix: cp \"$THREE_TOPO_HERE/server_config.example.sh\" \"$THREE_TOPO_HERE/server_config.sh\"" >&2
  echo "     then edit DATA_ROOT / PYTHON / THREADS in $THREE_TOPO_HERE/server_config.sh." >&2
  exit 2
}

[ -f "$CONFIG_FILE" ] || _tt_die "config file not found: $CONFIG_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${PROJECT_ROOT:=$_THREE_TOPO_DERIVED_ROOT}"
: "${THREADS:=4}"
: "${PYTHON:?PYTHON not set in $CONFIG_FILE}"
[ -n "${DATA_ROOT:-}" ] || _tt_die "DATA_ROOT not set in $CONFIG_FILE"
[ -f "$PROJECT_ROOT/scripts/three_topology_v2/train_formal.py" ] \
  || _tt_die "PROJECT_ROOT does not look like the model project: $PROJECT_ROOT"
[ -f "$DATA_ROOT/data/processed/three_topology_graph_v1/samples.jsonl" ] \
  || _tt_die "samples.jsonl not found under DATA_ROOT=$DATA_ROOT"
[ -x "$PYTHON" ] || _tt_die "PYTHON not executable: $PYTHON"
case "$THREADS" in ''|*[!0-9]*) _tt_die "THREADS must be an integer, got '$THREADS'" ;; esac

export PROJECT_ROOT DATA_ROOT PYTHON THREADS
export THREE_TOPO_PACKAGE_ROOT="$DATA_ROOT"
export THREE_TOPO_SERVER_CONFIG="$CONFIG_FILE"

visible_gpu_count() {
  "$PYTHON" - <<'PY' 2>/dev/null | tail -n 1
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

# Require >=1 CUDA device visible to THIS process (respects caller's CVD).
require_one_visible_gpu() {
  local n
  n=$(visible_gpu_count); n="${n:-0}"
  if [ "$n" -lt 1 ]; then
    echo "[GPU ERROR] no CUDA device visible. Select one at command time, e.g." >&2
    echo "  CUDA_VISIBLE_DEVICES=<gpu_index> bash server_ops/three_topology_v2/run_job.sh <job.yaml>" >&2
    exit 5
  fi
  echo "[GPU] visible_device_count=$n CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}'"
}

# Batch-scoped preflight gate. BATCH_ID must be exported (from job YAML).
require_preflight() {
  : "${BATCH_ID:?BATCH_ID not set (run via run_job.sh / job YAML)}"
  if [ "${THREE_TOPO_SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "[PRECHECK] skipped (THREE_TOPO_SKIP_PREFLIGHT=1) — test mode only" >&2
    return 0
  fi
  local pf="$PROJECT_ROOT/outputs/three_topology_v2/preflight/$BATCH_ID/PREFLIGHT_PASS.json"
  if [ ! -f "$pf" ]; then
    echo "[PRECHECK] missing preflight PASS for batch '$BATCH_ID'" >&2
    echo "Run exactly: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" <job.yaml>" >&2
    exit 4
  fi
  grep -Eq '"result"[[:space:]]*:[[:space:]]*"PASS"' "$pf" || {
    echo "[PRECHECK] preflight record is not PASS: $pf" >&2
    echo "Re-run: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" <job.yaml>" >&2
    exit 4
  }
  local want got
  want=$(sed -n 's/.*"samples_jsonl_sha256"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' "$pf" | head -1)
  if [ -n "$want" ]; then
    got=$(sha256sum "$DATA_ROOT/data/processed/three_topology_graph_v1/samples.jsonl" | awk '{print $1}')
    [ "$got" = "$want" ] || {
      echo "[PRECHECK] samples.jsonl changed since preflight" >&2
      echo "Re-run: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" <job.yaml>" >&2
      exit 4
    }
  fi
  echo "[PRECHECK] PASS ($pf)"
}

print_resolved_config() {
  echo "config_file : $CONFIG_FILE"
  echo "project_root: $PROJECT_ROOT"
  echo "data_root   : $DATA_ROOT"
  echo "python      : $PYTHON"
  echo "threads     : $THREADS"
  echo "cuda_visible: ${CUDA_VISIBLE_DEVICES-<unset>}"
  echo "visible_gpus: $(visible_gpu_count)"
}

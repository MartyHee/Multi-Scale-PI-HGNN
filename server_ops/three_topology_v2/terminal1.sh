#!/usr/bin/env bash
# TERMINAL 1 — GPU A: MLP x3 seeds (sequential), then Ridge + HistGB on CPU.
# Run this in its own terminal. Required env:
#   PROJECT_ROOT DATA_ROOT PYTHON BATCH_ID GPU_A
set -uo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"; : "${BATCH_ID:?}"; : "${GPU_A:?set GPU_A to a real GPU index}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0
run () {  # worker model seed mode device [gpu] [resume]
  local worker="$1" model="$2" seed="$3" mode="$4" dev="$5" gpu="${6:-}" resume="${7:-}"
  WORKER_ID="$worker" MODEL="$model" SEED="$seed" RUN_MODE="$mode" TASK_DEVICE="$dev" \
  GPU_ID="$gpu" RESUME_FROM="$resume" PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" \
  PYTHON="$PYTHON" BATCH_ID="$BATCH_ID" bash "$HERE/run_task.sh" || FAILED=1
}
for s in 20261117 20261118 20261119; do
  run t1_mlp mlp "$s" new cuda "$GPU_A"
done
run t1_ridge ridge - new cpu
run t1_histgb hist_gb - new cpu
echo "[terminal1] FAILED=$FAILED"
exit "$FAILED"

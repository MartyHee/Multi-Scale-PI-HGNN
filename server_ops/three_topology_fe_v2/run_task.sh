#!/usr/bin/env bash
# Single-task launcher used by run_job.sh.
#
# The Python entry owns the atomic run directory, STATUS/TASK files and the
# formal identity checks.  This wrapper only resolves the shell-level identity
# and keeps a launch log; it cannot bypass the Python gate.
#
# Task env (set by run_job.sh):
#   WORKER_ID TOPOLOGY MODEL SEED RUN_MODE TASK_DEVICE TRAIN_CONFIG
#   TASK_ATTEMPT RESUME_FROM
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
require_preflight
: "${WORKER_ID:?}"; : "${TOPOLOGY:?}"; : "${MODEL:?}"; : "${RUN_MODE:?}"; : "${TASK_DEVICE:?}"
: "${TRAIN_CONFIG:?}"; : "${BATCH_ID:?}"
SEED="${SEED:-}"
ATTEMPT="${TASK_ATTEMPT:-1}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

if [ "$MODEL" = "ridge" ]; then
  RUN_NAME="ridge"
  SEED_ARG=0
else
  : "${SEED:?gradient/hist_gb task needs SEED}"
  RUN_NAME="${MODEL}_seed${SEED}"
  SEED_ARG="$SEED"
fi

EXT_CVD="${CUDA_VISIBLE_DEVICES-<unset>}"
if [ "$TASK_DEVICE" = "cpu" ]; then
  DEVICE_ARG="cpu"; EXPOSED="cpu"; CHILD_CVD_SET=1
else
  require_one_visible_gpu
  DEVICE_ARG="cuda:0"; EXPOSED="cuda:0"; CHILD_CVD_SET=0
fi

LOG_DIR="$PROJECT_ROOT/logs/three_topology_fe_v2/$BATCH_ID"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LAUNCH_LOG="$LOG_DIR/${WORKER_ID}_${RUN_NAME}_attempt${ATTEMPT}_${TS}.console.log"

ARGS=(--mode formal --config "$TRAIN_CONFIG" --batch-id "$BATCH_ID"
      --worker "$WORKER_ID" --topology "$TOPOLOGY" --model "$MODEL"
      --seed "$SEED_ARG" --device "$DEVICE_ARG" --attempt "$ATTEMPT")
if [ "$RUN_MODE" = "resume" ]; then
  : "${RESUME_FROM:?resume mode needs RESUME_FROM}"
  ARGS+=(--resume-from "$RESUME_FROM")
fi

echo "[run_task] batch=$BATCH_ID topology=$TOPOLOGY worker=$WORKER_ID model=$MODEL seed=$SEED_ARG attempt=$ATTEMPT mode=$RUN_MODE"
echo "[run_task] external_cuda_visible_devices='$EXT_CVD' exposed=$EXPOSED launch_log=$LAUNCH_LOG"

cd "$PROJECT_ROOT"
set +e
if [ "$CHILD_CVD_SET" = "1" ]; then
  # Hide CUDA for this child only; siblings and the caller are unaffected.
  # -1 is recognised as "no devices" by torch on Linux and Windows; an empty
  # string is not consistently interpreted on Windows.
  CUDA_VISIBLE_DEVICES="-1" "$PYTHON" -B "$PROJECT_ROOT/scripts/three_topology_fe_v2/train_fe_v2.py" \
    "${ARGS[@]}" 2>&1 | tee "$LAUNCH_LOG"
else
  "$PYTHON" -B "$PROJECT_ROOT/scripts/three_topology_fe_v2/train_fe_v2.py" \
    "${ARGS[@]}" 2>&1 | tee "$LAUNCH_LOG"
fi
rc=${PIPESTATUS[0]}
set -e
echo "[run_task] exit=$rc"
exit "$rc"

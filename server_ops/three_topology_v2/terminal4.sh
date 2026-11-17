#!/usr/bin/env bash
# TERMINAL 4 — GPU D: HGT-global seed 20261119.
# Required env: PROJECT_ROOT DATA_ROOT PYTHON BATCH_ID GPU_D
set -uo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"; : "${BATCH_ID:?}"; : "${GPU_D:?set GPU_D to a real GPU index}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ID=t4_hgt MODEL=hgt_global SEED=20261119 RUN_MODE=new TASK_DEVICE=cuda GPU_ID="$GPU_D" \
PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" PYTHON="$PYTHON" BATCH_ID="$BATCH_ID" \
bash "$HERE/run_task.sh"
exit $?

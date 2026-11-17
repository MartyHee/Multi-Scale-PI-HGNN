#!/usr/bin/env bash
# TERMINAL 2 — GPU B: HGT-global seed 20261117.
# Required env: PROJECT_ROOT DATA_ROOT PYTHON BATCH_ID GPU_B
set -uo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"; : "${BATCH_ID:?}"; : "${GPU_B:?set GPU_B to a real GPU index}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ID=t2_hgt MODEL=hgt_global SEED=20261117 RUN_MODE=new TASK_DEVICE=cuda GPU_ID="$GPU_B" \
PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" PYTHON="$PYTHON" BATCH_ID="$BATCH_ID" \
bash "$HERE/run_task.sh"
exit $?

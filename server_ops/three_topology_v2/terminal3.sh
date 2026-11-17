#!/usr/bin/env bash
# TERMINAL 3 — GPU C: HGT-global seed 20261118.
# Required env: PROJECT_ROOT DATA_ROOT PYTHON BATCH_ID GPU_C
set -uo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"; : "${BATCH_ID:?}"; : "${GPU_C:?set GPU_C to a real GPU index}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ID=t3_hgt MODEL=hgt_global SEED=20261118 RUN_MODE=new TASK_DEVICE=cuda GPU_ID="$GPU_C" \
PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" PYTHON="$PYTHON" BATCH_ID="$BATCH_ID" \
bash "$HERE/run_task.sh"
exit $?

#!/usr/bin/env bash
# New launch interface:
#   CUDA_VISIBLE_DEVICES=0 bash server_ops/three_topology_v2/run_job.sh <job.yaml>
#
# - YAML parsed by Python/PyYAML (_job_parse.py), strict validation.
# - Server-wide settings from server_config.sh; project auto-located.
# - GPU chosen by the CALLER's CUDA_VISIBLE_DEVICES; this script never sets,
#   clears or expands it. In-process device is cuda:0.
# - Reuses run_task.sh (atomic dir, duplicate refusal, STATUS, resume, best).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

if [ $# -ne 1 ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml>" >&2
  exit 2
fi
JOB_FILE="$1"
[ -f "$JOB_FILE" ] || { echo "[run_job] job file not found: $JOB_FILE" >&2; exit 2; }

# strict YAML parse -> bash-evalable header
eval "$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$JOB_FILE")" || {
  echo "[run_job] invalid job yaml: $JOB_FILE" >&2; exit 2; }

export BATCH_ID
echo "[run_job] job=$JOB_NAME batch=$BATCH_ID protocol=$PROTOCOL file=$JOB_FILE"
print_resolved_config
require_preflight

# If the job has any GPU task, require >=1 visible GPU under the caller's CVD.
HAS_GPU=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --tasks "$JOB_FILE" \
          | awk -F'\t' '$4=="gpu"{print 1; exit}')
if [ "${HAS_GPU:-}" = "1" ]; then
  require_one_visible_gpu
fi

FAILED=0
while IFS=$'\t' read -r worker model seed device resume epochs maxb; do
  [ -n "$worker" ] || continue
  mode="new"; [ "$resume" != "-" ] && mode="resume"
  resume_path=""; [ "$resume" != "-" ] && resume_path="$resume"
  echo "[run_job] task worker=$worker model=$model seed=$seed device=$device mode=$mode"
  WORKER_ID="$worker" MODEL="$model" SEED="$seed" RUN_MODE="$mode" TASK_DEVICE="$device" \
  RESUME_FROM="$resume_path" TRAIN_CONFIG="$TRAIN_CONFIG" PROTOCOL="$PROTOCOL" \
  SPLIT_ID="$SPLIT_ID" FOLD_ID="$FOLD_ID" OUTER_FOLD="$OUTER_FOLD" INNER_ID="$INNER_ID" \
  TASK_EPOCHS="$epochs" TASK_MAX_BATCHES="$maxb" \
  bash "$THREE_TOPO_HERE/run_task.sh" || FAILED=1
done < <("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --tasks "$JOB_FILE")

echo "[run_job] job=$JOB_NAME FAILED=$FAILED"
exit "$FAILED"

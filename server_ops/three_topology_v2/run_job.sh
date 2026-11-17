#!/usr/bin/env bash
# Launch interface:
#   CUDA_VISIBLE_DEVICES=0 bash server_ops/three_topology_v2/run_job.sh <job.yaml>
#
# - YAML parsed by Python/PyYAML (_job_parse.py), strict validation.
# - Server-wide settings from server_config.sh; project auto-located.
# - GPU chosen by the CALLER's CUDA_VISIBLE_DEVICES; this script never sets,
#   clears or expands it. In-process device is cuda:0.
# - Registered job YAMLs must match the frozen batch manifest v2 (all effective
#   execution params). External YAMLs are allowed ONLY as resume jobs and are
#   validated against the manifest + checkpoint identity (_resume_check.py).
# - Reuses run_task.sh (atomic dir, duplicate refusal, STATUS, resume, best).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

if [ $# -ne 1 ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml>" >&2
  exit 2
fi
JOB_FILE="$1"
[ -f "$JOB_FILE" ] || { echo "[run_job] job file not found: $JOB_FILE" >&2; exit 2; }

# ---- strict YAML parse: check exit code and non-empty output BEFORE eval ----
HDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$JOB_FILE") || {
  echo "[run_job] invalid job yaml: $JOB_FILE" >&2; exit 2; }
[ -n "$HDR" ] || { echo "[run_job] job parser produced no header: $JOB_FILE" >&2; exit 2; }
eval "$HDR" || { echo "[run_job] cannot apply parsed job header: $JOB_FILE" >&2; exit 2; }
: "${JOB_NAME:?}"; : "${BATCH_ID:?}"; : "${PROTOCOL:?}"

TASKS=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --tasks "$JOB_FILE") || {
  echo "[run_job] cannot parse task list: $JOB_FILE" >&2; exit 2; }
[ -n "$TASKS" ] || { echo "[run_job] job has zero tasks: $JOB_FILE" >&2; exit 2; }
TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT
printf '%s\n' "$TASKS" > "$TASKS_FILE"
[ -s "$TASKS_FILE" ] || { echo "[run_job] job has zero tasks: $JOB_FILE" >&2; exit 2; }

export BATCH_ID
echo "[run_job] job=$JOB_NAME batch=$BATCH_ID protocol=$PROTOCOL file=$JOB_FILE"
print_resolved_config
require_preflight

# Registered = inside the repo job directory. External = resume-only.
REGISTERED=$("$PYTHON" -c 'import pathlib,sys; j=pathlib.Path(sys.argv[1]).resolve(); r=(pathlib.Path(sys.argv[2])/"remote_jobs"/"three_topology_v2").resolve(); print(1 if j.is_relative_to(r) else 0)' "$JOB_FILE" "$PROJECT_ROOT") || {
  echo "[run_job] cannot resolve job path" >&2; exit 2; }
HAS_RESUME=$(awk -F'\t' '$5!="-"{print 1; exit}' "$TASKS_FILE")

if [ "${HAS_RESUME:-}" = "1" ]; then
  if [ "$REGISTERED" = "1" ]; then
    echo "[run_job] resume YAML must live OUTSIDE remote_jobs/three_topology_v2/ so it" >&2
    echo "          cannot be mistaken for a frozen batch task; use e.g. /tmp/resume.yaml" >&2
    exit 2
  fi
  # The frozen batch itself must still agree with its registered YAMLs.
  "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --verify || exit 2
  RESUME_DIR="$PROJECT_ROOT/outputs/three_topology_v2/resume_checks/$BATCH_ID"
  mkdir -p "$RESUME_DIR"
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  RESUME_REC="$RESUME_DIR/${TS}_${JOB_NAME}.json"
  RESUME_MAP="$RESUME_DIR/${TS}_${JOB_NAME}.tsv"
  "$PYTHON" -B "$THREE_TOPO_HERE/_resume_check.py" \
    --job "$JOB_FILE" --batch-id "$BATCH_ID" \
    --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" \
    --out "$RESUME_REC" --map-out "$RESUME_MAP" || exit 2
  export RESUME_CHECK_FILE="$RESUME_REC"
else
  # No resume task: only a registered YAML is acceptable.
  if [ "$REGISTERED" = "0" ]; then
    echo "[run_job] external YAML without resume_from is not allowed; normal jobs must be" >&2
    echo "          registered under remote_jobs/three_topology_v2 and match the frozen manifest" >&2
    exit 2
  fi
  "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_FILE" --verify || exit 2
fi

# If the job has any GPU task, require >=1 visible GPU under the caller's CVD.
HAS_GPU=$(awk -F'\t' '$4=="gpu"{print 1; exit}' "$TASKS_FILE")
if [ "${HAS_GPU:-}" = "1" ]; then
  require_one_visible_gpu
fi

FAILED=0
while IFS=$'\t' read -r worker model seed device resume epochs maxb; do
  [ -n "$worker" ] || continue
  mode="new"; [ "$resume" != "-" ] && mode="resume"
  resume_path=""; [ "$resume" != "-" ] && resume_path="$resume"
  of_worker=""; of_run=""
  if [ "$mode" = "resume" ]; then
    run_suffix="$model"
    [ "$seed" != "-" ] && run_suffix="${model}_seed${seed}"
    of_worker=$(awk -F'\t' -v w="$worker" -v r="$run_suffix" '$1==w && $2==r {print $5}' "$RESUME_MAP")
    of_run=$(awk -F'\t' -v w="$worker" -v r="$run_suffix" '$1==w && $2==r {print $6}' "$RESUME_MAP")
    [ -n "$of_worker" ] && [ -n "$of_run" ] || {
      echo "[run_job] no resume mapping for $worker/$run_suffix" >&2; exit 2; }
  fi
  echo "[run_job] task worker=$worker model=$model seed=$seed device=$device mode=$mode"
  WORKER_ID="$worker" MODEL="$model" SEED="$seed" RUN_MODE="$mode" TASK_DEVICE="$device" \
  RESUME_FROM="$resume_path" RESUME_CHECK_FILE="${RESUME_CHECK_FILE:-}" \
  RESUME_OF_WORKER="$of_worker" RESUME_OF_RUN="$of_run" \
  TRAIN_CONFIG="$TRAIN_CONFIG" PROTOCOL="$PROTOCOL" \
  SPLIT_ID="$SPLIT_ID" FOLD_ID="$FOLD_ID" OUTER_FOLD="$OUTER_FOLD" INNER_ID="$INNER_ID" \
  TASK_EPOCHS="$epochs" TASK_MAX_BATCHES="$maxb" \
  bash "$THREE_TOPO_HERE/run_task.sh" || FAILED=1
done < "$TASKS_FILE"

echo "[run_job] job=$JOB_NAME FAILED=$FAILED"
exit "$FAILED"

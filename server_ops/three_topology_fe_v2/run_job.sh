#!/usr/bin/env bash
# Launch interface for FE v2:
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_fe_v2/run_job.sh <job.yaml>
#
# - Registered job YAMLs live under remote_jobs/three_topology_fe_v2/ and must
#   match the frozen batch manifest exactly (all effective execution params).
# - Resume YAMLs live OUTSIDE that directory and are validated against the
#   manifest, the original run, and the checkpoint identity.
# - The caller's CUDA_VISIBLE_DEVICES is never set, cleared or expanded; the
#   in-process device is cuda:0.  CPU tasks hide CUDA only for their child.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

if [ $# -ne 1 ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml>" >&2
  exit 2
fi
JOB_FILE="$1"
[ -f "$JOB_FILE" ] || { echo "[run_job] job file not found: $JOB_FILE" >&2; exit 2; }

HDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$JOB_FILE") || {
  echo "[run_job] invalid job yaml: $JOB_FILE" >&2; exit 2; }
[ -n "$HDR" ] || { echo "[run_job] job parser produced no header: $JOB_FILE" >&2; exit 2; }
eval "$HDR" || { echo "[run_job] cannot apply parsed job header: $JOB_FILE" >&2; exit 2; }
: "${JOB_NAME:?}"; : "${BATCH_ID:?}"; : "${TRAIN_CONFIG:?}"

TASKS=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --tasks "$JOB_FILE") || {
  echo "[run_job] cannot parse task list: $JOB_FILE" >&2; exit 2; }
[ -n "$TASKS" ] || { echo "[run_job] job has zero tasks: $JOB_FILE" >&2; exit 2; }
TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE" "${RESUME_MAP:-}"' EXIT
printf '%s\n' "$TASKS" > "$TASKS_FILE"

export BATCH_ID TRAIN_CONFIG
echo "[run_job] job=$JOB_NAME batch=$BATCH_ID file=$JOB_FILE tasks=$(wc -l < "$TASKS_FILE")"
print_resolved_config
require_preflight

REGISTERED=$("$PYTHON" -c 'import pathlib,sys; j=pathlib.Path(sys.argv[1]).resolve(); r=(pathlib.Path(sys.argv[2])/"remote_jobs"/"three_topology_fe_v2").resolve(); print(1 if j.parent==r else 0)' "$JOB_FILE" "$PROJECT_ROOT") || {
  echo "[run_job] cannot resolve job path" >&2; exit 2; }
HAS_RESUME=$(awk -F'\t' '$6!="-"{print 1; exit}' "$TASKS_FILE")

if [ "${HAS_RESUME:-}" = "1" ]; then
  if [ "$REGISTERED" = "1" ]; then
    echo "[run_job] resume YAML must live OUTSIDE remote_jobs/three_topology_fe_v2/" >&2
    exit 2
  fi
  if awk -F'\t' '$6=="-"{print 1; exit}' "$TASKS_FILE" | grep -q 1; then
    echo "[run_job] a resume YAML must not contain non-resume tasks" >&2
    exit 2
  fi
  "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --verify || exit 2
  RESUME_DIR="$PROJECT_ROOT/outputs/three_topology_fe_v2/resume_checks/$BATCH_ID"
  mkdir -p "$RESUME_DIR"
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  RESUME_REC="$RESUME_DIR/${TS}_${JOB_NAME}.json"
  export RESUME_MAP="$RESUME_DIR/${TS}_${JOB_NAME}.tsv"
  "$PYTHON" -B "$THREE_TOPO_HERE/_resume_check.py" \
    --job "$JOB_FILE" --batch-id "$BATCH_ID" --project-root "$PROJECT_ROOT" \
    --package-root "$PACKAGE_ROOT" --out "$RESUME_REC" --map-out "$RESUME_MAP" || exit 2
else
  if [ "$REGISTERED" = "0" ]; then
    echo "[run_job] external YAML without resume_from is not allowed" >&2
    exit 2
  fi
  "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_FILE" --verify || exit 2
fi

HAS_GPU=$(awk -F'\t' '$5=="gpu"{print 1; exit}' "$TASKS_FILE")
if [ "${HAS_GPU:-}" = "1" ]; then
  require_one_visible_gpu
fi

FAILED=0
while IFS=$'\t' read -r worker topology model seed device resume epochs maxb; do
  [ -n "$worker" ] || continue
  mode="new"; resume_path=""; attempt=1; of_worker=""; of_run=""
  if [ "$resume" != "-" ]; then
    mode="resume"; resume_path="$resume"
    line=$(awk -F'\t' -v w="$worker" -v t="$topology" -v m="$model" -v s="$seed" \
      '$1==w && $2==t && $4==m && $5==s {print; exit}' "$RESUME_MAP")
    [ -n "$line" ] || { echo "[run_job] no resume mapping for $worker/$topology/$model/$seed" >&2; exit 2; }
    of_worker=$(echo "$line" | cut -f6)
    of_run=$(echo "$line" | cut -f7)
    attempt=$(echo "$line" | cut -f9)
    resume_path=$(echo "$line" | cut -f10)
  fi
  echo "[run_job] task worker=$worker topology=$topology model=$model seed=$seed device=$device mode=$mode attempt=$attempt"
  WORKER_ID="$worker" TOPOLOGY="$topology" MODEL="$model" SEED="$seed" \
  RUN_MODE="$mode" TASK_DEVICE="$device" TASK_ATTEMPT="$attempt" \
  RESUME_FROM="$resume_path" RESUME_OF_WORKER="$of_worker" RESUME_OF_RUN="$of_run" \
  TRAIN_CONFIG="$TRAIN_CONFIG" \
  bash "$THREE_TOPO_HERE/run_task.sh" || FAILED=1
done < "$TASKS_FILE"

echo "[run_job] job=$JOB_NAME FAILED=$FAILED"
exit "$FAILED"

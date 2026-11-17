#!/usr/bin/env bash
# Single-task launcher used by run_job.sh. Respects the caller's
# CUDA_VISIBLE_DEVICES: GPU tasks use cuda:0 in-process and never touch it;
# CPU tasks hide CUDA only for their own child python. Atomic run dir; duplicate
# refusal; STATUS.json; resume.
#
# Task env (set by run_job.sh): WORKER_ID MODEL SEED(- for ML) RUN_MODE TASK_DEVICE
#   TRAIN_CONFIG PROTOCOL SPLIT_ID FOLD_ID OUTER_FOLD INNER_ID
#   RESUME_FROM TASK_EPOCHS TASK_MAX_BATCHES
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
require_preflight
: "${WORKER_ID:?}"; : "${MODEL:?}"; : "${RUN_MODE:?}"; : "${TASK_DEVICE:?}"
: "${TRAIN_CONFIG:?}"; : "${PROTOCOL:?}"
SEED="${SEED:-}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

case "$MODEL" in
  ridge|hist_gb) [ -z "$SEED" ] && SEED="-" ;;
esac
if [ "$SEED" = "-" ]; then SUFFIX="$MODEL"; else : "${SEED:?gradient task needs SEED}"; SUFFIX="${MODEL}_seed${SEED}"; fi

OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID}"
RUN_DIR="$OUT_ROOT/$WORKER_ID/$SUFFIX"
mkdir -p "$(dirname "$RUN_DIR")"
if ! mkdir "$RUN_DIR" 2>/dev/null; then
  echo "REFUSE: run dir already exists (duplicate start or another worker): $RUN_DIR" >&2
  exit 3
fi

EXT_CVD="${CUDA_VISIBLE_DEVICES-<unset>}"
if [ "$TASK_DEVICE" = "cpu" ]; then
  DEVICE_ARG="cpu"; EXPOSED="cpu"; CHILD_CVD_SET=1
else
  require_one_visible_gpu
  DEVICE_ARG="cuda:0"; EXPOSED="cuda:0"; CHILD_CVD_SET=0
fi
VIS_COUNT=$(visible_gpu_count); VIS_COUNT="${VIS_COUNT:-0}"

STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$RUN_DIR/TASK.json" <<EOF
{"batch_id":"$BATCH_ID","worker_id":"$WORKER_ID","model":"$MODEL","seed":"$SEED",
 "run_mode":"$RUN_MODE","device":"$EXPOSED","external_cuda_visible_devices":"$EXT_CVD",
 "visible_device_count":$VIS_COUNT,"run_dir":"$RUN_DIR","resume_from":"${RESUME_FROM:-}",
 "protocol":"$PROTOCOL","train_config":"$TRAIN_CONFIG","started_at":"$STARTED","threads":$THREADS}
EOF
printf '{"state":"running","batch_id":"%s","worker_id":"%s","model":"%s","seed":"%s","device":"%s","external_cuda_visible_devices":"%s","started_at":"%s"}\n' \
  "$BATCH_ID" "$WORKER_ID" "$MODEL" "$SEED" "$EXPOSED" "$EXT_CVD" "$STARTED" > "$RUN_DIR/.STATUS.json.tmp"
mv "$RUN_DIR/.STATUS.json.tmp" "$RUN_DIR/STATUS.json"

ARGS=(--config "$TRAIN_CONFIG" --models "$MODEL" --run-root "$RUN_DIR"
      --device "$DEVICE_ARG" --batch-id "$BATCH_ID" --worker-id "$WORKER_ID")
if [ "$PROTOCOL" = "inner" ]; then
  : "${OUTER_FOLD:?inner protocol needs OUTER_FOLD}"; : "${INNER_ID:?inner protocol needs INNER_ID}"
  ARGS+=(--protocol inner --outer-fold "$OUTER_FOLD" --inner-id "$INNER_ID")
else
  : "${SPLIT_ID:?split protocol needs SPLIT_ID}"; : "${FOLD_ID:?split protocol needs FOLD_ID}"
  ARGS+=(--split "$SPLIT_ID" --fold "$FOLD_ID")
fi
[ "$SEED" != "-" ] && ARGS+=(--seeds "$SEED")
if [ "$RUN_MODE" = "resume" ]; then
  : "${RESUME_FROM:?resume mode needs RESUME_FROM}"
  ARGS+=(--resume "$RESUME_FROM")
fi
[ -n "${TASK_EPOCHS:-}" ] && [ "$TASK_EPOCHS" != "-" ] && ARGS+=(--epochs "$TASK_EPOCHS")
[ -n "${TASK_MAX_BATCHES:-}" ] && [ "$TASK_MAX_BATCHES" != "-" ] && ARGS+=(--max-batches "$TASK_MAX_BATCHES")
# test-only extra args (never set on the server)
# shellcheck disable=SC2206
[ -n "${THREE_TOPO_EXTRA_ARGS:-}" ] && ARGS+=($THREE_TOPO_EXTRA_ARGS)

cd "$PROJECT_ROOT"
set +e
if [ "$CHILD_CVD_SET" = "1" ]; then
  # hide CUDA for THIS child only; caller's env and sibling tasks are unaffected
  CUDA_VISIBLE_DEVICES="" "$PYTHON" -B scripts/three_topology_v2/train_formal.py "${ARGS[@]}" > "$RUN_DIR/console.log" 2>&1
else
  "$PYTHON" -B scripts/three_topology_v2/train_formal.py "${ARGS[@]}" > "$RUN_DIR/console.log" 2>&1
fi
rc=$?
set -e

FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$rc" -eq 0 ]; then STATE="done"; else STATE="failed"; fi
SEL=""
if [ -f "$RUN_DIR/summary.json" ]; then
  SEL=$("$PYTHON" - "$RUN_DIR/summary.json" <<'PY' 2>/dev/null || true
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); m=s.get('models',{})
print(next(iter(m.values())).get('selection_metric') if m else '')
PY
)
fi
printf '{"state":"%s","batch_id":"%s","worker_id":"%s","model":"%s","seed":"%s","device":"%s","external_cuda_visible_devices":"%s","exit_code":%d,"selection_metric":%s,"finished_at":"%s","run_dir":"%s"}\n' \
  "$STATE" "$BATCH_ID" "$WORKER_ID" "$MODEL" "$SEED" "$EXPOSED" "$EXT_CVD" "$rc" "${SEL:-null}" "$FINISHED" "$RUN_DIR" > "$RUN_DIR/.STATUS.json.tmp"
mv "$RUN_DIR/.STATUS.json.tmp" "$RUN_DIR/STATUS.json"
echo "[$STATE] $RUN_DIR (exit=$rc, device=$EXPOSED, CVD=$EXT_CVD)"
exit "$rc"

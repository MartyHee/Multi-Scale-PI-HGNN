#!/usr/bin/env bash
# Single-task launcher for the four-terminal first batch.
# Atomic run dir; refuses duplicate; writes per-task STATUS.json; one GPU only.
#
# Required env: PROJECT_ROOT DATA_ROOT PYTHON BATCH_ID WORKER_ID MODEL RUN_MODE(new|resume)
#   TASK_DEVICE(cpu|cuda); if cuda: GPU_ID (physical index); if resume: RESUME_FROM
#   if gradient: SEED (e.g. 20261117); for ML set SEED="-"
# Optional: THREADS (default 4), EXTRA_ARGS
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"
: "${BATCH_ID:?}"; : "${WORKER_ID:?}"; : "${MODEL:?}"; : "${RUN_MODE:?}"; : "${TASK_DEVICE:?}"
SEED="${SEED:-}"
THREADS="${THREADS:-4}"
export THREE_TOPO_PACKAGE_ROOT="$DATA_ROOT"

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

# device mapping
if [ "$TASK_DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_ARG="cpu"; EXPOSED="cpu"
else
  : "${GPU_ID:?cuda task needs GPU_ID}"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  DEVICE_ARG="cuda:0"; EXPOSED="cuda:0(physical $GPU_ID)"
fi
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TASK_JSON="$RUN_DIR/TASK.json"
cat > "$TASK_JSON" <<EOF
{"batch_id":"$BATCH_ID","worker_id":"$WORKER_ID","model":"$MODEL","seed":"$SEED",
 "run_mode":"$RUN_MODE","device":"$EXPOSED","gpu_id":"${GPU_ID:-}","run_dir":"$RUN_DIR",
 "resume_from":"${RESUME_FROM:-}","started_at":"$STARTED","threads":$THREADS}
EOF
STATUS_TMP="$RUN_DIR/.STATUS.json.tmp"
printf '{"state":"running","batch_id":"%s","worker_id":"%s","model":"%s","seed":"%s","device":"%s","started_at":"%s"}\n' \
  "$BATCH_ID" "$WORKER_ID" "$MODEL" "$SEED" "$EXPOSED" "$STARTED" > "$STATUS_TMP"
mv "$STATUS_TMP" "$RUN_DIR/STATUS.json"

ARGS=(--config configs/three_topology_v2/formal_baselines.yaml
      --models "$MODEL" --split cross_design_topology1 --fold cd_fold_1
      --device "$DEVICE_ARG" --run-root "$RUN_DIR"
      --batch-id "$BATCH_ID" --worker-id "$WORKER_ID")
[ "$SEED" != "-" ] && ARGS+=(--seeds "$SEED")
if [ "$RUN_MODE" = "resume" ]; then
  : "${RESUME_FROM:?resume mode needs RESUME_FROM}"
  ARGS+=(--resume "$RESUME_FROM")
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

cd "$PROJECT_ROOT"
set +e
"$PYTHON" -B scripts/three_topology_v2/train_formal.py "${ARGS[@]}" > "$RUN_DIR/console.log" 2>&1
rc=$?
set -e

FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$rc" -eq 0 ]; then STATE="done"; else STATE="failed"; fi
SEL=""
if [ -f "$RUN_DIR/summary.json" ]; then
  SEL=$("$PYTHON" - "$RUN_DIR/summary.json" <<'PY' 2>/dev/null || true
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
m=s.get('models',{})
print(next(iter(m.values())).get('selection_metric') if m else '')
PY
)
fi
STATUS_TMP="$RUN_DIR/.STATUS.json.tmp"
printf '{"state":"%s","batch_id":"%s","worker_id":"%s","model":"%s","seed":"%s","device":"%s","exit_code":%d,"selection_metric":%s,"finished_at":"%s"}\n' \
  "$STATE" "$BATCH_ID" "$WORKER_ID" "$MODEL" "$SEED" "$EXPOSED" "$rc" "${SEL:-null}" "$FINISHED" > "$STATUS_TMP"
mv "$STATUS_TMP" "$RUN_DIR/STATUS.json"
echo "[$STATE] $RUN_DIR (exit=$rc)"
exit "$rc"

#!/usr/bin/env bash
# Server preflight for ONE FE v2 batch.
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_fe_v2/preflight.sh <job.yaml|BATCH_ID>
#
# Runs deps/operators, small data identity checks, the frozen-hash gate, and a
# capped three-slice train/val/best-export for every (topology, model) pair in
# the batch.  Writes an independent preflight directory and a PASS record that
# binds code/data/schema/protocol/ledger/stats/config/manifest.  It never starts
# a full-budget run and never loads test responses.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

[ $# -ge 1 ] || { echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
ARG="$1"
JOB_YAML=""
case "$ARG" in
  *.yaml|*.yml)
    REGISTERED=$("$PYTHON" -c 'import pathlib,sys; j=pathlib.Path(sys.argv[1]).resolve(); r=(pathlib.Path(sys.argv[2])/"remote_jobs"/"three_topology_fe_v2").resolve(); print(1 if j.parent==r else 0)' "$ARG" "$PROJECT_ROOT") || exit 2
    [ "$REGISTERED" = "1" ] || { echo "[preflight] requires a registered batch YAML" >&2; exit 2; }
    HDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$ARG") || exit 2
    [ -n "$HDR" ] || { echo "[preflight] empty job header" >&2; exit 2; }
    eval "$HDR"
    JOB_YAML="$ARG"
    ;;
  *) BATCH_ID="$ARG" ;;
esac
: "${BATCH_ID:?BATCH_ID not resolved}"
export BATCH_ID

if [ -z "$JOB_YAML" ]; then
  JOB_YAML=$(for y in "$PROJECT_ROOT"/remote_jobs/three_topology_fe_v2/*.yaml; do
    [ -e "$y" ] || continue
    if [ "$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$y" 2>/dev/null)" = "$BATCH_ID" ]; then
      echo "$y"; break
    fi
  done)
  [ -n "$JOB_YAML" ] || { echo "[preflight] no job YAML for batch $BATCH_ID" >&2; exit 2; }
fi

echo "== frozen batch manifest =="
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_YAML" --verify || exit 2
MHDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_YAML" --shell) || exit 2
eval "$MHDR"
export BATCH_ID TRAIN_CONFIG

print_resolved_config
TASK_FILE="$(mktemp)"
trap 'rm -f "$TASK_FILE"' EXIT
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --effective > "$TASK_FILE"
[ -s "$TASK_FILE" ] || { echo "[preflight] manifest has zero tasks" >&2; exit 2; }
HAS_GPU=$(awk -F'\t' '$6=="gpu"{print 1; exit}' "$TASK_FILE")
if [ "${HAS_GPU:-}" = "1" ]; then
  require_one_visible_gpu
else
  echo "[GPU] batch has no GPU task; CPU-only preflight"
fi

PF_ROOT="$THREE_TOPO_PREFLIGHT_ROOT/$BATCH_ID"
mkdir -p "$PF_ROOT"
rm -f "$PF_ROOT/PREFLIGHT_PASS.json" "$PF_ROOT/PREFLIGHT_FAIL.json"
TS=$(date -u +%Y%m%dT%H%M%SZ)
export THREE_TOPO_PREFLIGHT_TS="$TS"
EXT_CVD="${CUDA_VISIBLE_DEVICES-<unset>}"
VIS=$(visible_gpu_count); VIS="${VIS:-0}"

fail() {
  printf '{"result":"FAIL","batch_id":"%s","external_cuda_visible_devices":"%s","reason":"%s","at":"%s"}\n' \
    "$BATCH_ID" "$EXT_CVD" "$1" "$TS" > "$PF_ROOT/PREFLIGHT_FAIL.json"
  echo "[preflight] FAIL: $1" >&2
  echo "Fix, then rerun: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" \"$ARG\"" >&2
  exit 1
}

echo "== deps =="
set +e
if [ "${HAS_GPU:-}" = "1" ]; then
  "$PYTHON" -B "$THREE_TOPO_HERE/check_deps.py" --require-cuda > "$PF_ROOT/check_deps.json" 2>&1
else
  "$PYTHON" -B "$THREE_TOPO_HERE/check_deps.py" > "$PF_ROOT/check_deps.json" 2>&1
fi
DEPS_RC=$?
set -e
cat "$PF_ROOT/check_deps.json"
[ "$DEPS_RC" -eq 0 ] || fail "dependency/operator check failed"

echo "== package/data identity =="
bash "$THREE_TOPO_HERE/data_hash_check.sh" "$PF_ROOT" | tee "$PF_ROOT/data_hash.txt" || fail "package identity check failed"

echo "== frozen hashes =="
"$PYTHON" -B "$THREE_TOPO_HERE/_preflight_gate.py" --hashes --batch-id "$BATCH_ID" \
  --project-root "$PROJECT_ROOT" > "$PF_ROOT/hashes.json" || fail "frozen hash mismatch"
cat "$PF_ROOT/hashes.json"

echo "== capped three-slice checks =="
CHECKED=()
CHECKED_KEYS=""
STATUS=0
while IFS=$'\t' read -r worker run topology model seed device kind epochs maxb resume; do
  [ -n "$worker" ] || continue
  KEY="$topology/$model"
  case " $CHECKED_KEYS " in *" $KEY "*) continue ;; esac
  CHECKED_KEYS="$CHECKED_KEYS $KEY"
  RUNDIR="$PF_ROOT/runs/$TS/$topology/$model"
  if [ "$kind" = "gradient" ]; then
    SEED_ARG="$seed"; TRAIN_N=4
    "$PYTHON" -B "$PROJECT_ROOT/scripts/three_topology_fe_v2/train_fe_v2.py" \
      --mode preflight --config "$TRAIN_CONFIG" --batch-id "$BATCH_ID" \
      --topology "$topology" --model "$model" --seed "$SEED_ARG" --device cuda:0 \
      --train-samples 4 --val-samples 2 --max-val-batches 1 \
      > "$PF_ROOT/${topology}_${model}_console.log" 2>&1 || STATUS=1
    [ -f "$RUNDIR/checkpoints/best.pt" ] || STATUS=1
  else
    SEED_ARG=0; [ "$seed" != "-" ] && SEED_ARG="$seed"
    "$PYTHON" -B "$PROJECT_ROOT/scripts/three_topology_fe_v2/train_fe_v2.py" \
      --mode preflight --config "$TRAIN_CONFIG" --batch-id "$BATCH_ID" \
      --topology "$topology" --model "$model" --seed "$SEED_ARG" --device cpu \
      --train-samples 16 --val-samples 2 --max-val-batches 1 \
      > "$PF_ROOT/${topology}_${model}_console.log" 2>&1 || STATUS=1
  fi
  [ -f "$RUNDIR/summary.json" ] || STATUS=1
  [ -f "$RUNDIR/slice_metrics.json" ] || STATUS=1
  for s in design_val load_val joint_val; do
    [ -f "$RUNDIR/val_predictions_$s.json" ] || STATUS=1
  done
  if [ -f "$RUNDIR/summary.json" ]; then
    "$PYTHON" - "$RUNDIR/summary.json" <<'PY' || STATUS=1
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
assert s.get("selection_metric") is not None, s
assert len(s.get("slice_selection",{}))==3, s
assert s.get("test_constructed") is False, s
print("preflight run OK", sys.argv[1], s["selection_metric"])
PY
  fi
  CHECKED+=(--checked "$topology/$model")
done < "$TASK_FILE"
[ "$STATUS" -eq 0 ] || fail "capped three-slice run/export failed (see *_console.log)"

"$PYTHON" -B "$THREE_TOPO_HERE/_preflight_gate.py" --write --batch-id "$BATCH_ID" \
  --project-root "$PROJECT_ROOT" --device "$EXT_CVD" --visible "$VIS" \
  "${CHECKED[@]}" || fail "cannot write preflight PASS"
echo "[preflight] PASS -> $PF_ROOT/PREFLIGHT_PASS.json"
echo "Next: CUDA_VISIBLE_DEVICES=<gpu> bash $THREE_TOPO_HERE/run_job.sh remote_jobs/three_topology_fe_v2/<job>.yaml"

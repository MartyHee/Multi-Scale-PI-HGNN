#!/usr/bin/env bash
# Server preflight for ONE batch (task list read from the frozen batch manifest).
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/preflight.sh <job.yaml|BATCH_ID>
# Uses the caller's CUDA_VISIBLE_DEVICES (never sets/expands it). Runs deps +
# data hash + 1-batch CUDA train/val/best-export for every gradient model of the
# batch on cuda:0, then writes a PASS record binding code commit, data, split,
# stats, train config and the batch manifest. Any later drift invalidates it.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

[ $# -ge 1 ] || { echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
ARG="$1"
JOB_YAML=""
case "$ARG" in
  *.yaml|*.yml)
    REGISTERED=$("$PYTHON" -c 'import pathlib,sys; j=pathlib.Path(sys.argv[1]).resolve(); r=(pathlib.Path(sys.argv[2])/"remote_jobs"/"three_topology_v2").resolve(); print(1 if j.is_relative_to(r) else 0)' "$ARG" "$PROJECT_ROOT") || {
      echo "[preflight] cannot resolve job path" >&2; exit 2; }
    if [ "$REGISTERED" != "1" ]; then
      echo "[preflight] preflight requires a registered batch YAML, not an external/resume YAML" >&2; exit 2
    fi
    HDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$ARG") || {
      echo "[preflight] invalid job yaml: $ARG" >&2; exit 2; }
    [ -n "$HDR" ] || { echo "[preflight] job parser produced no header" >&2; exit 2; }
    eval "$HDR" || { echo "[preflight] cannot apply job header" >&2; exit 2; }
    JOB_YAML="$ARG"
    ;;
  *) BATCH_ID="$ARG" ;;
esac
: "${BATCH_ID:?BATCH_ID not resolved}"
export BATCH_ID

if [ -z "$JOB_YAML" ]; then
  JOB_YAML=$(for y in "$PROJECT_ROOT"/remote_jobs/three_topology_v2/*.yaml; do
    [ -e "$y" ] || continue
    if [ "$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$y" 2>/dev/null)" = "$BATCH_ID" ]; then
      echo "$y"; break
    fi
  done)
  [ -n "$JOB_YAML" ] || { echo "[preflight] no job YAML found for batch $BATCH_ID" >&2; exit 2; }
fi

echo "== frozen batch manifest =="
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_YAML" --verify || exit 2
MHDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_YAML" --shell) || {
  echo "[preflight] cannot read batch manifest header" >&2; exit 2; }
[ -n "$MHDR" ] || { echo "[preflight] empty batch manifest header" >&2; exit 2; }
eval "$MHDR" || { echo "[preflight] cannot apply batch manifest header" >&2; exit 2; }
export BATCH_ID

print_resolved_config
require_one_visible_gpu
TASK_FILE="$(mktemp)"
trap 'rm -f "$TASK_FILE"' EXIT
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --job "$JOB_YAML" --paths > "$TASK_FILE" || {
  echo "[preflight] cannot read batch task list" >&2; exit 2; }
[ -s "$TASK_FILE" ] || { echo "[preflight] manifest has zero tasks" >&2; exit 2; }
MODELS=$(awk -F'	' '$6=="gradient"{print $3}' "$TASK_FILE" | sort -u)
[ -n "$MODELS" ] || { echo "[preflight] manifest has no gradient model" >&2; exit 2; }
echo "== gradient models in batch: $(echo $MODELS) =="

PF_ROOT="$PROJECT_ROOT/outputs/three_topology_v2/preflight/$BATCH_ID"
mkdir -p "$PF_ROOT"
rm -f "$PF_ROOT/PREFLIGHT_PASS.json" "$PF_ROOT/PREFLIGHT_FAIL.json"
TS=$(date -u +%Y%m%dT%H%M%SZ)
EXT_CVD="${CUDA_VISIBLE_DEVICES-<unset>}"
echo "[preflight] batch=$BATCH_ID CVD=$EXT_CVD dir=$PF_ROOT"

fail() {
  printf '{"result":"FAIL","batch_id":"%s","external_cuda_visible_devices":"%s","reason":"%s","at":"%s"}\n' \
    "$BATCH_ID" "$EXT_CVD" "$1" "$TS" > "$PF_ROOT/PREFLIGHT_FAIL.json"
  echo "[preflight] FAIL: $1" >&2
  echo "Fix, then rerun: CUDA_VISIBLE_DEVICES=<gpu> bash \"$THREE_TOPO_HERE/preflight.sh\" \"$ARG\"" >&2
  exit 1
}

echo "== deps (caller CUDA_VISIBLE_DEVICES='$EXT_CVD') =="
set +e
"$PYTHON" -B "$THREE_TOPO_HERE/check_deps.py" > "$PF_ROOT/check_deps.json" 2>&1
DEPS_RC=$?
set -e
cat "$PF_ROOT/check_deps.json"
[ "$DEPS_RC" -eq 0 ] || fail "dependency check failed (see check_deps.json)"

echo "== data hashes =="
bash "$THREE_TOPO_HERE/data_hash_check.sh" | tee "$PF_ROOT/data_hash.txt" || fail "data hash check failed"

echo "== identity hashes (code/data/split/stats/config/manifest) =="
set +e
"$PYTHON" -B "$THREE_TOPO_HERE/_preflight_gate.py" --hashes --batch-id "$BATCH_ID" \
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" > "$PF_ROOT/hashes.json"
HASH_RC=$?
set -e
cat "$PF_ROOT/hashes.json"
[ "$HASH_RC" -eq 0 ] || fail "batch manifest / train config identity mismatch (see hashes.json)"

echo "== one-batch CUDA preflight for each gradient model of the batch =="
STATUS=0
for MODEL in $MODELS; do
  RUNDIR="$PF_ROOT/runs/$TS/$MODEL"
  "$PYTHON" -B scripts/three_topology_v2/train_formal.py \
    --config "$PROJECT_ROOT/${TRAIN_CONFIG:-configs/three_topology_v2/formal_baselines.yaml}" \
    --models "$MODEL" --split "$SPLIT_ID" --fold "$FOLD_ID" \
    --device cuda:0 --smoke --max-batches 1 \
    --run-root "$RUNDIR" --require-new-dir \
    --batch-id "$BATCH_ID" --worker-id "pf_$MODEL" \
    > "$PF_ROOT/${MODEL}_console.log" 2>&1 || STATUS=1
  if [ -f "$RUNDIR/summary.json" ]; then
    "$PYTHON" - "$RUNDIR/summary.json" <<'PY' || STATUS=1
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); assert s.get('all_pass') is True, s
print("summary all_pass OK", sys.argv[1])
PY
  else STATUS=1; fi
  [ -f "$RUNDIR/checkpoints/best.pt" ] || STATUS=1
done
[ "$STATUS" -eq 0 ] || fail "small CUDA train/val/best export failed (see *_console.log)"

"$PYTHON" - "$PF_ROOT/hashes.json" "$PF_ROOT/PREFLIGHT_PASS.json" "$BATCH_ID" \
  "$EXT_CVD" "$(visible_gpu_count)" "$TS" "$PF_ROOT" "$MODELS" <<'PY'
import json, sys
hashes = json.load(open(sys.argv[1], encoding="utf-8"))
rec = {"result": "PASS", "batch_id": sys.argv[3],
       "external_cuda_visible_devices": sys.argv[4],
       "visible_device_count": int(sys.argv[5]), "at": sys.argv[6],
       "dir": sys.argv[7], "gradient_models": sorted(sys.argv[8].split())}
rec.update({k: v for k, v in hashes.items() if k != "batch_id"})
json.dump(rec, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote", sys.argv[2])
PY
echo "[preflight] PASS -> $PF_ROOT/PREFLIGHT_PASS.json"
echo "Next: CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/<job>.yaml"

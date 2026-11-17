#!/usr/bin/env bash
# Server preflight. Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/preflight.sh <job.yaml|BATCH_ID>
# Uses the caller's CUDA_VISIBLE_DEVICES (does NOT set/expand it); runs deps +
# data hash + MLP/HGT 1-batch CUDA train/val/best-export on cuda:0; writes
# outputs/three_topology_v2/preflight/<batch>/PREFLIGHT_PASS.json. No 4-GPU check.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

[ $# -ge 1 ] || { echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
ARG="$1"
case "$ARG" in
  *.yaml|*.yml)
    eval "$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --shell "$ARG")" || exit 2
    ;;
  *) BATCH_ID="$ARG" ;;
esac
: "${BATCH_ID:?BATCH_ID not resolved}"
export BATCH_ID
print_resolved_config
require_one_visible_gpu

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

echo "== small CUDA preflight: MLP + HGT-global (1 batch, cuda:0 of caller's CVD) =="
STATUS=0
for MODEL in mlp hgt_global; do
  RUNDIR="$PF_ROOT/runs/$TS/$MODEL"
  "$PYTHON" -B scripts/three_topology_v2/train_formal.py \
    --config configs/three_topology_v2/formal_baselines.yaml \
    --models "$MODEL" --split cross_design_topology1 --fold cd_fold_1 \
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

SAMPLES_SHA=$(sha256sum "$DATA_ROOT/data/processed/three_topology_graph_v1/samples.jsonl" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$DATA_ROOT/data/processed/three_topology_graph_v1/dataset_manifest.json" | awk '{print $1}')
GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
printf '{"result":"PASS","batch_id":"%s","external_cuda_visible_devices":"%s","visible_device_count":%s,"samples_jsonl_sha256":"%s","dataset_manifest_sha256":"%s","git_commit":"%s","at":"%s","dir":"%s"}\n' \
  "$BATCH_ID" "$EXT_CVD" "$(visible_gpu_count)" "$SAMPLES_SHA" "$MANIFEST_SHA" "$GIT_COMMIT" "$TS" "$PF_ROOT" > "$PF_ROOT/.PASS.tmp"
mv "$PF_ROOT/.PASS.tmp" "$PF_ROOT/PREFLIGHT_PASS.json"
echo "[preflight] PASS -> $PF_ROOT/PREFLIGHT_PASS.json"
echo "Next (four terminals): CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/<job>.yaml"

#!/usr/bin/env bash
# Server preflight (user-run). Verifies deps, data hashes, and runs a SMALL
# CUDA train+val+best-export for MLP and HGT-global in a separate preflight dir.
# Required env: PROJECT_ROOT DATA_ROOT PYTHON
# Optional: GPU_ID (default 0), PREFLIGHT_ID (default timestamp), THREADS (4)
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${PYTHON:?}"
GPU_ID="${GPU_ID:-0}"; THREADS="${THREADS:-4}"
export THREE_TOPO_PACKAGE_ROOT="$DATA_ROOT"
PID_TAG="${PREFLIGHT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PF_ROOT="$PROJECT_ROOT/outputs/three_topology_v2/preflight/$PID_TAG"
mkdir -p "$PF_ROOT"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
cd "$PROJECT_ROOT"

echo "== deps =="
set +e
"$PYTHON" -B server_ops/three_topology_v2/check_deps.py | tee "$PF_ROOT/check_deps.json"
DEPS_RC=${PIPESTATUS[0]}
set -e
if [ "$DEPS_RC" -ne 0 ]; then echo "DEP CHECK FAILED (see $PF_ROOT/check_deps.json)"; exit 1; fi

echo "== data hashes =="
bash server_ops/three_topology_v2/data_hash_check.sh | tee "$PF_ROOT/data_hash.txt"

echo "== small CUDA preflight: MLP + HGT-global (1 batch) =="
STATUS=0
for MODEL in mlp hgt_global; do
  RUNDIR="$PF_ROOT/$MODEL"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -B scripts/three_topology_v2/train_formal.py \
    --config configs/three_topology_v2/formal_baselines.yaml \
    --models "$MODEL" --split cross_design_topology1 --fold cd_fold_1 \
    --device cuda:0 --smoke --max-batches 1 \
    --run-root "$RUNDIR" --require-new-dir \
    --batch-id "preflight_$PID_TAG" --worker-id "pf_$MODEL" \
    > "$PF_ROOT/${MODEL}_console.log" 2>&1 || STATUS=1
  if [ -f "$RUNDIR/summary.json" ]; then
    "$PYTHON" - "$RUNDIR/summary.json" <<'PY' || STATUS=1
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
assert s.get('all_pass') is True, s
print("summary all_pass OK", sys.argv[1])
PY
  else
    echo "MISSING summary for $MODEL"; STATUS=1
  fi
  [ -f "$RUNDIR/checkpoints/best.pt" ] || { echo "MISSING best.pt for $MODEL"; STATUS=1; }
done

if [ "$STATUS" -eq 0 ]; then
  printf '{"preflight":"%s","gpu_id":%s,"result":"PASS","dir":"%s"}\n' "$PID_TAG" "$GPU_ID" "$PF_ROOT" > "$PF_ROOT/PREFLIGHT_PASS.json"
  echo "PREFLIGHT PASS — $PF_ROOT/PREFLIGHT_PASS.json"
else
  printf '{"preflight":"%s","gpu_id":%s,"result":"FAIL","dir":"%s"}\n' "$PID_TAG" "$GPU_ID" "$PF_ROOT" > "$PF_ROOT/PREFLIGHT_FAIL.json"
  echo "PREFLIGHT FAIL — see $PF_ROOT"; exit 1
fi

#!/usr/bin/env bash
# Collect ONE batch's lightweight artifacts (strict: missing required files abort).
# Required: PROJECT_ROOT BATCH_ID. Optional: PYTHON, OUT_TAR.
# Excludes raw data / processed graphs / historical outputs. SHA256SUMS excludes itself.
set -euo pipefail
THREE_TOPO_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Direct use: load server_config.sh. Test harness that already provides
# PROJECT_ROOT+BATCH_ID skips the loader.
if [ -z "${PROJECT_ROOT:-}" ]; then
  source "$THREE_TOPO_HERE/_load_config.sh"
fi
ARG="${1:-}"
if [ -z "${BATCH_ID:-}" ]; then
  [ -n "$ARG" ] || { echo "Usage: bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
  case "$ARG" in
    *.yaml|*.yml) BATCH_ID=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$ARG") ;;
    *) BATCH_ID="$ARG" ;;
  esac
fi
export BATCH_ID
OUT_TAR="${OUT_TAR:-$PROJECT_ROOT/first_batch_${BATCH_ID}_artifacts.tar.gz}"
BATCH_DIR="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# ---- required files for DONE runs (failed runs kept for analysis, but exit!=0) ----
INCOMPLETE=0
is_done () { [ -f "$1/STATUS.json" ] && grep -Eq '"state"[[:space:]]*:[[:space:]]*"done"' "$1/STATUS.json"; }
check_grad () {  # worker run
  local d="$BATCH_DIR/$1/$2"
  if ! is_done "$d"; then echo "INCOMPLETE (not done): $d" >&2; INCOMPLETE=1; return; fi
  for f in STATUS.json TASK.json run_manifest.json run_log.json summary.json train_log.csv driver.log val_predictions.json checkpoints/best.pt checkpoints/last.pt; do
    [[ -f "$d/$f" ]] || { echo "MISSING required: $d/$f" >&2; INCOMPLETE=1; }
  done
}
check_ml () {
  local d="$BATCH_DIR/$1/$2"; local r="$2"
  if ! is_done "$d"; then echo "INCOMPLETE (not done): $d" >&2; INCOMPLETE=1; return; fi
  for f in STATUS.json TASK.json run_manifest.json run_log.json summary.json driver.log val_predictions.json "ml_$r.pkl"; do
    [[ -f "$d/$f" ]] || { echo "MISSING required: $d/$f" >&2; INCOMPLETE=1; }
  done
}
[[ -d "$BATCH_DIR" ]] || { echo "batch dir not found: $BATCH_DIR" >&2; exit 1; }
check_grad t1_mlp mlp_seed20261117
check_grad t1_mlp mlp_seed20261118
check_grad t1_mlp mlp_seed20261119
check_grad t2_hgt hgt_global_seed20261117
check_grad t3_hgt hgt_global_seed20261118
check_grad t4_hgt hgt_global_seed20261119
check_ml t1_ridge ridge
check_ml t1_histgb hist_gb

# ---- stage ----
mkdir -p "$STAGE"/{batch,configs,manifests,code}
cp -a "$BATCH_DIR" "$STAGE/batch/"
cp -a "$PROJECT_ROOT/configs/three_topology_v2" "$STAGE/configs/"
mkdir -p "$STAGE/manifests"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/splits" "$STAGE/manifests/"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/stats" "$STAGE/manifests/"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/stats_inner" "$STAGE/manifests/" 2>/dev/null || true
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/manifests" "$STAGE/manifests/"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$STAGE/code/CODE_VERSION.txt" 2>/dev/null || echo unknown > "$STAGE/code/CODE_VERSION.txt"
git -C "$PROJECT_ROOT" status --porcelain >> "$STAGE/code/CODE_VERSION.txt" 2>/dev/null || true
if [[ -n "${PYTHON:-}" ]]; then
  "$PYTHON" - <<'PY' > "$STAGE/code/ENV.txt" 2>&1 || true
import sys, torch, torch_geometric, numpy, sklearn
print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_geometric", torch_geometric.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__)
PY
fi

# ---- checksums (exclude SHA256SUMS itself) ----
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
tar -czf "$OUT_TAR" -C "$STAGE" .
echo "artifacts: $OUT_TAR"
echo "outer_sha256: $(sha256sum "$OUT_TAR" | awk '{print $1}')"
echo "files: $(wc -l < "$STAGE/SHA256SUMS")"
if [ "$INCOMPLETE" -ne 0 ]; then
  echo "INCOMPLETE batch: archive built for analysis; NOT a complete result." >&2
  exit 1
fi

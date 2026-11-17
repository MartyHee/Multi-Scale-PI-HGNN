#!/usr/bin/env bash
# Collect ONE batch's lightweight artifacts, driven by its frozen manifest.
# Required: PROJECT_ROOT BATCH_ID. Optional: PYTHON, OUT_TAR.
# Excludes raw data / processed graphs / historical outputs. SHA256SUMS excludes itself.
#
# Default archive: $PROJECT_ROOT/remote_artifacts/batch_<batch_id>_artifacts.tar.gz
# (parent dir auto-created; an existing target is NEVER overwritten -> exit 3).
# Only tasks listed in remote_jobs/three_topology_v2/manifests/<batch_id>.json are
# staged; any extra run dir is listed in EXTRA_RUN_DIRS.txt, not silently bundled
# as this batch's result. A done task missing a required file fails the archive.
set -euo pipefail
THREE_TOPO_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PROJECT_ROOT:-}" ]; then
  source "$THREE_TOPO_HERE/_load_config.sh"
fi
ARG="${1:-}"
JOB_YAML=""
if [ -z "${BATCH_ID:-}" ]; then
  [ -n "$ARG" ] || { echo "Usage: bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
  case "$ARG" in
    *.yaml|*.yml) BATCH_ID=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$ARG"); JOB_YAML="$ARG" ;;
    *) BATCH_ID="$ARG" ;;
  esac
fi
export BATCH_ID
MANIFEST="$PROJECT_ROOT/remote_jobs/three_topology_v2/manifests/$BATCH_ID.json"
[ -f "$MANIFEST" ] || {
  echo "frozen batch manifest not found: $MANIFEST" >&2
  echo "Generate/commit it first: python server_ops/three_topology_v2/_batch_manifest.py --job <job.yaml> --write" >&2
  exit 2
}
echo "== verify job YAMLs against frozen manifest =="
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --verify \
  | tee /tmp/_collect_manifest_verify.$$ || { rm -f /tmp/_collect_manifest_verify.$$; exit 2; }

OUT_TAR="${OUT_TAR:-$PROJECT_ROOT/remote_artifacts/batch_${BATCH_ID}_artifacts.tar.gz}"
mkdir -p "$(dirname "$OUT_TAR")"
if [ -e "$OUT_TAR" ]; then
  echo "REFUSE: archive already exists, not overwriting: $OUT_TAR" >&2
  echo "Use OUT_TAR=<new path> to write elsewhere." >&2
  exit 3
fi
BATCH_DIR="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID"
[[ -d "$BATCH_DIR" ]] || { echo "batch dir not found: $BATCH_DIR" >&2; exit 1; }
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE" /tmp/_collect_manifest_verify.$$' EXIT

# ---- expected tasks from the frozen manifest ----
TASKS_TSV="$STAGE/TASKS.tsv"
if ! "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --paths > "$TASKS_TSV"; then
  echo "cannot read frozen task list for batch $BATCH_ID" >&2; exit 2
fi
[ -s "$TASKS_TSV" ] || { echo "frozen task list is empty for batch $BATCH_ID" >&2; exit 2; }
N_EXPECTED=$(wc -l < "$TASKS_TSV")

is_done () { [ -f "$1/STATUS.json" ] && grep -Eq '"state"[[:space:]]*:[[:space:]]*"done"' "$1/STATUS.json"; }
req_grad="STATUS.json TASK.json run_manifest.json run_log.json summary.json train_log.csv driver.log val_predictions.json checkpoints/best.pt checkpoints/last.pt"
req_ml="STATUS.json TASK.json run_manifest.json run_log.json summary.json driver.log val_predictions.json"

INCOMPLETE=0
KNOWN=""
mkdir -p "$STAGE/batch/$BATCH_ID"
while IFS=$'\t' read -r worker run model seed device kind; do
  [ -n "$worker" ] || continue
  KNOWN="$KNOWN $worker/$run"
  d="$BATCH_DIR/$worker/$run"
  if [ ! -d "$d" ]; then
    echo "MISSING expected task dir: $d" >&2; INCOMPLETE=1; continue
  fi
  if ! is_done "$d"; then
    echo "INCOMPLETE (not done): $d" >&2; INCOMPLETE=1
  else
    if [ "$kind" = "ml" ]; then
      for f in $req_ml "ml_$run.pkl"; do
        [[ -f "$d/$f" ]] || { echo "MISSING required: $d/$f" >&2; INCOMPLETE=1; }
      done
    else
      for f in $req_grad; do
        [[ -f "$d/$f" ]] || { echo "MISSING required: $d/$f" >&2; INCOMPLETE=1; }
      done
    fi
  fi
  mkdir -p "$STAGE/batch/$BATCH_ID/$worker"
  cp -a "$d" "$STAGE/batch/$BATCH_ID/$worker/$run"
done < "$TASKS_TSV"

# ---- extra run dirs: reported, not treated as this batch's results ----
: > "$STAGE/batch/EXTRA_RUN_DIRS.txt"
for wd in "$BATCH_DIR"/*/; do
  [ -d "$wd" ] || continue
  w=$(basename "$wd")
  for rd in "$wd"*/; do
    [ -d "$rd" ] || continue
    r=$(basename "$rd")
    case " $KNOWN " in
      *" $w/$r "*) ;;
      *) echo "$w/$r" >> "$STAGE/batch/EXTRA_RUN_DIRS.txt" ;;
    esac
  done
done
if [ -s "$STAGE/batch/EXTRA_RUN_DIRS.txt" ]; then
  echo "NOTE: extra run dirs listed (not counted as batch results)" >&2
fi

# ---- frozen inputs / config / code ----
mkdir -p "$STAGE"/{configs,manifests,code}
cp -a "$PROJECT_ROOT/configs/three_topology_v2" "$STAGE/configs/"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/splits" "$STAGE/manifests/"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/stats" "$STAGE/manifests/"
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/stats_inner" "$STAGE/manifests/" 2>/dev/null || true
cp -a "$PROJECT_ROOT/processed/three_topology_training_v2/manifests" "$STAGE/manifests/"
cp "$MANIFEST" "$STAGE/manifests/batch_manifest.json"
cp /tmp/_collect_manifest_verify.$$ "$STAGE/manifests/BATCH_MANIFEST_VERIFY.txt"
cp "$TASKS_TSV" "$STAGE/manifests/TASKS.tsv"
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
echo "tasks: $N_EXPECTED (from $MANIFEST)"
if [ "$INCOMPLETE" -ne 0 ]; then
  echo "INCOMPLETE batch: archive built for analysis; NOT a complete result." >&2
  exit 1
fi

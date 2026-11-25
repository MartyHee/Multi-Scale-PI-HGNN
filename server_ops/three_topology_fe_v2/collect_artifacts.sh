#!/usr/bin/env bash
# Collect ONE FE v2 batch's lightweight artifacts.
#   bash server_ops/three_topology_fe_v2/collect_artifacts.sh <job.yaml|BATCH_ID>
# Default: $PROJECT_ROOT/remote_artifacts/fe_v2_<batch>_artifacts.tar.gz
# Existing targets are never overwritten.  Missing/failed tasks still produce an
# archive for diagnosis but exit 1 and are marked NOT COMPLETE.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
ARG="${1:-}"
[ -n "$ARG" ] || { echo "Usage: bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
case "$ARG" in
  *.yaml|*.yml) BATCH_ID=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$ARG") ;;
  *) BATCH_ID="$ARG" ;;
esac
export BATCH_ID
MANIFEST="$THREE_TOPO_MANIFEST_ROOT/$BATCH_ID.json"
[ -f "$MANIFEST" ] || { echo "frozen manifest not found: $MANIFEST" >&2; exit 2; }
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --verify || exit 2

OUT_TAR="${OUT_TAR:-$PROJECT_ROOT/remote_artifacts/fe_v2_${BATCH_ID}_artifacts.tar.gz}"
# Git-Bash/GNU tar needs POSIX drive paths (/d/...), not D:/...; no-op on Linux.
case "$OUT_TAR" in
  [A-Za-z]:/*) command -v cygpath >/dev/null 2>&1 && OUT_TAR="$(cygpath -u "$OUT_TAR")" ;;
esac
mkdir -p "$(dirname "$OUT_TAR")"
if [ -e "$OUT_TAR" ]; then
  echo "REFUSE: archive already exists: $OUT_TAR" >&2
  echo "Use OUT_TAR=<new path> to write elsewhere." >&2
  exit 3
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"/{manifests,configs,processed,code,summary}

cp "$MANIFEST" "$STAGE/manifests/batch_manifest.json"
cp -a "$PROJECT_ROOT/remote_jobs/three_topology_fe_v2" "$STAGE/manifests/jobs"
cp -a "$PROJECT_ROOT/configs/three_topology_fe_v2" "$STAGE/configs/"
cp -a "$PROCESSED_ROOT/splits" "$STAGE/processed/"
cp -a "$PROCESSED_ROOT/stats" "$STAGE/processed/"
cp "$PROJECT_ROOT/configs/three_topology_fe_v2/feature_schema.json" "$STAGE/configs/" 2>/dev/null || true
git -C "$PROJECT_ROOT" rev-parse HEAD > "$STAGE/code/CODE_VERSION.txt" 2>/dev/null || echo unknown > "$STAGE/code/CODE_VERSION.txt"
git -C "$PROJECT_ROOT" status --porcelain >> "$STAGE/code/CODE_VERSION.txt" 2>/dev/null || true
"$PYTHON" - <<'PY' > "$STAGE/code/ENV.txt" 2>&1 || true
import sys, torch, torch_geometric, numpy, sklearn
print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_geometric", torch_geometric.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__)
PY

SUM_JSON="$STAGE/summary/batch_summary.json"
set +e
"$PYTHON" -B "$THREE_TOPO_HERE/summarize_batch.py" --batch "$BATCH_ID" --json > "$SUM_JSON"
SUM_RC=$?
set -e
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --effective > "$STAGE/manifests/TASKS.tsv"

INCOMPLETE=0
[ "$SUM_RC" -eq 0 ] || INCOMPLETE=1
"$PYTHON" - "$SUM_JSON" "$STAGE" <<'PY' || INCOMPLETE=1
import json, shutil, sys
from pathlib import Path
summ = json.load(open(sys.argv[1], encoding="utf-8"))
stage = Path(sys.argv[2])
req_grad = ["STATUS.json","TASK.json","console.log","run_log.json","summary.json",
            "train_log.csv","slice_metrics.json","val_predictions.json",
            "val_predictions_design_val.json","val_predictions_load_val.json",
            "val_predictions_joint_val.json","checkpoints/best.pt","checkpoints/last.pt"]
req_ml = ["STATUS.json","TASK.json","console.log","summary.json","slice_metrics.json",
          "val_predictions.json","val_predictions_design_val.json",
          "val_predictions_load_val.json","val_predictions_joint_val.json","ml_fit.json"]
bad = 0
supersedes = []
for r in summ["rows"]:
    src = Path(r.get("run_dir", ""))
    dst = stage / "batch" / r["topology"] / r["worker"] / r["run"]
    if r.get("resolved_by") and src.is_dir():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        req = req_grad if r["kind"] == "gradient" else req_ml + [f"ml_{r['run']}.pkl"]
        for f in req:
            if not (dst / f).is_file():
                print(f"MISSING required: {dst / f}", file=sys.stderr); bad = 1
        if r.get("superseded_by"):
            supersedes.append(f"{r['topology']}/{r['worker']}/{r['run']} <- {r['superseded_by']}")
    else:
        print(f"NOT RESOLVED: {r['topology']}/{r['worker']}/{r['run']} state={r.get('state')}",
              file=sys.stderr); bad = 1
(stage / "batch" / "SUPERSEDES.txt").write_text("\n".join(supersedes) + ("\n" if supersedes else ""),
                                                encoding="utf-8")
raise SystemExit(0 if bad == 0 else 1)
PY
[ $? -eq 0 ] || INCOMPLETE=1

( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
tar -czf "$OUT_TAR" -C "$STAGE" .
echo "artifacts: $OUT_TAR"
echo "outer_sha256: $(sha256sum "$OUT_TAR" | awk '{print $1}')"
echo "files: $(wc -l < "$STAGE/SHA256SUMS")"
if [ "$INCOMPLETE" -ne 0 ]; then
  echo "INCOMPLETE batch: archive built for diagnosis; NOT a complete result." >&2
  exit 1
fi

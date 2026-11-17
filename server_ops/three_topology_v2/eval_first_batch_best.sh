#!/usr/bin/env bash
# Eval-only recheck of a finished batch's gradient best checkpoints.
#
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/eval_first_batch_best.sh <batch_id>
#
# Reloads each gradient task's checkpoints/best.pt, evaluates ONLY the original
# validation graphs of the frozen fold, writes to a NEW independent directory
# (never touches the original run dir or predictions), and compares the new
# predictions/metrics with the archived ones under an explicit tolerance.
# No checkpoint is edited (no device-string rewriting); identity is the sha256.
#
# Env: RECHECK_ROOT (default outputs/three_topology_v2/first_batch_recheck/<batch>),
#      EVAL_DEVICE (default cuda:0), RECHECK_ATOL/RECHECK_RTOL.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

[ $# -eq 1 ] || { echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <batch_id>" >&2; exit 2; }
BATCH_ID="$1"; export BATCH_ID
MANIFEST="$PROJECT_ROOT/remote_jobs/three_topology_v2/manifests/$BATCH_ID.json"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
eval "$("$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --shell)"
export BATCH_ID

OUT_ROOT="${RECHECK_ROOT:-$PROJECT_ROOT/outputs/three_topology_v2/first_batch_recheck/$BATCH_ID}"
if [ -e "$OUT_ROOT" ]; then
  echo "REFUSE: recheck output already exists: $OUT_ROOT" >&2
  echo "Use RECHECK_ROOT=<new dir> for a fresh recheck; originals stay read-only." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
ATOL="${RECHECK_ATOL:-1e-8}"; RTOL="${RECHECK_RTOL:-1e-5}"
echo "[recheck] batch=$BATCH_ID device=$EVAL_DEVICE out=$OUT_ROOT atol=$ATOL rtol=$RTOL"
print_resolved_config
case "$EVAL_DEVICE" in cuda*) require_one_visible_gpu ;; esac

FAILED=0
while IFS=$'\t' read -r worker run model seed device kind; do
  [ -n "$worker" ] || continue
  [ "$kind" = "gradient" ] || continue
  ORIG="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID/$worker/$run"
  BEST="$ORIG/checkpoints/best.pt"
  if [ ! -f "$BEST" ]; then
    echo "MISSING best checkpoint: $BEST" >&2; FAILED=1; continue
  fi
  DEST="$OUT_ROOT/$worker/$run"
  mkdir -p "$(dirname "$DEST")"
  echo "[recheck] $worker/$run model=$model seed=$seed ckpt_sha=$(sha256sum "$BEST" | awk '{print $1}')"
  "$PYTHON" -B scripts/three_topology_v2/train_formal.py \
    --config "$TRAIN_CONFIG" --models "$model" --seeds "$seed" \
    --split "$SPLIT_ID" --fold "$FOLD_ID" --device "$EVAL_DEVICE" \
    --eval-only --checkpoint "$BEST" \
    --run-root "$DEST" --require-new-dir \
    --batch-id "$BATCH_ID" --worker-id "recheck_$worker" \
    > "$DEST.console.log" 2>&1 || { echo "eval-only failed: $worker/$run" >&2; FAILED=1; continue; }
  "$PYTHON" -B scripts/three_topology_v2/compare_recheck.py \
    --original "$ORIG" --recheck "$DEST" --checkpoint "$BEST" \
    --out "$DEST/recheck.json" --atol "$ATOL" --rtol "$RTOL" \
    >> "$DEST.console.log" 2>&1 || FAILED=1
done < <("$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --paths)

"$PYTHON" - "$OUT_ROOT" "$BATCH_ID" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); batch = sys.argv[2]
rows, ok = [], True
for p in sorted(out.glob("*/*/recheck.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    rows.append({"run": str(p.parent.relative_to(out)), "ok": r.get("ok"),
                 "checkpoint_sha256": r.get("checkpoint_sha256"),
                 "selection_original": r.get("selection_original"),
                 "selection_recheck": r.get("selection_recheck"),
                 "selection_abs_diff": r.get("selection_abs_diff"),
                 "worst_rel_diff": r.get("worst_rel_diff"),
                 "problems": r.get("problems")})
    ok = ok and bool(r.get("ok"))
summ = {"schema": "qt-first-batch-recheck-v1", "batch_id": batch,
        "mode": "eval-only on the original validation graphs; originals untouched",
        "n_runs": len(rows), "all_ok": ok, "runs": rows}
(out / "RECHECK_SUMMARY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"all_ok": ok, "n_runs": len(rows)}, ensure_ascii=False))
PY

echo "[recheck] summary: $OUT_ROOT/RECHECK_SUMMARY.json"
exit "$FAILED"

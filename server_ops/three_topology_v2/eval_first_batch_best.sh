#!/usr/bin/env bash
# Eval-only recheck of a finished historical batch's gradient best checkpoints.
# Works for fb_20261117_01 (6 gradient runs) and sb_20261118_01 (18 gradient runs).
#
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/eval_first_batch_best.sh <batch_id>
#
# Reloads each gradient task's checkpoints/best.pt, evaluates ONLY the original
# 24 validation graphs, writes to a NEW independent directory, and compares
# predictions/metrics/identity with the archived run. Originals stay read-only.
#
# Summary completeness: the expected gradient task list comes from the frozen
# manifest. Missing/duplicate reports, eval failure, comparison failure or
# unexpected report identities all make all_ok=false; the exit code follows
# the summary. Zero expected/report tasks never reports success.
#
# Env:
#   RECHECK_ROOT       default outputs/three_topology_v2/historical_best_recheck/<batch>
#   EVAL_DEVICE        default cuda:0
#   RECHECK_ATOL/RTOL  default 1e-8 / 1e-5
#   RECHECK_SUMMARY_ONLY=1  test-only: do not run eval, only summarize existing reports
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

[ $# -eq 1 ] || { echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 <batch_id>" >&2; exit 2; }
BATCH_ID="$1"; export BATCH_ID
MANIFEST="$PROJECT_ROOT/remote_jobs/three_topology_v2/manifests/$BATCH_ID.json"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

# Parse manifest header only after checking the parser's exit code and output.
HDR=$("$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --shell) || {
  echo "[recheck] cannot parse frozen manifest: $MANIFEST" >&2; exit 2; }
[ -n "$HDR" ] || { echo "[recheck] manifest parser produced no header" >&2; exit 2; }
eval "$HDR" || { echo "[recheck] cannot apply manifest header" >&2; exit 2; }
export BATCH_ID

OUT_ROOT="${RECHECK_ROOT:-$PROJECT_ROOT/outputs/three_topology_v2/historical_best_recheck/$BATCH_ID}"
SUMMARY_ONLY="${RECHECK_SUMMARY_ONLY:-0}"
if [ "$SUMMARY_ONLY" != "1" ] && [ -e "$OUT_ROOT" ]; then
  echo "REFUSE: recheck output already exists: $OUT_ROOT" >&2
  echo "Use RECHECK_ROOT=<new dir> for a fresh recheck; originals stay read-only." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
ATOL="${RECHECK_ATOL:-1e-8}"; RTOL="${RECHECK_RTOL:-1e-5}"
echo "[recheck] batch=$BATCH_ID device=$EVAL_DEVICE out=$OUT_ROOT atol=$ATOL rtol=$RTOL"
print_resolved_config

# Expected task list: capture first, check rc/non-empty, then filter gradient.
TASK_FILE="$(mktemp)"; EXPECTED_FILE="$(mktemp)"; FAILED_FILE="$(mktemp)"
trap 'rm -f "$TASK_FILE" "$EXPECTED_FILE" "$FAILED_FILE"' EXIT
"$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --paths > "$TASK_FILE" || {
  echo "[recheck] cannot read frozen task list" >&2; exit 2; }
[ -s "$TASK_FILE" ] || { echo "[recheck] frozen task list is empty" >&2; exit 2; }
awk -F'\t' '$6=="gradient"' "$TASK_FILE" > "$EXPECTED_FILE"
[ -s "$EXPECTED_FILE" ] || { echo "[recheck] no gradient task in manifest" >&2; exit 2; }

FAILED=0
mark_fail() { printf '%s	%s
' "$1" "$2" >> "$FAILED_FILE"; FAILED=1; }
if [ "$SUMMARY_ONLY" != "1" ]; then
  case "$EVAL_DEVICE" in cuda*) require_one_visible_gpu ;; esac
  while IFS=$'\t' read -r worker run model seed device kind; do
    [ -n "$worker" ] || continue
    ORIG="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID/$worker/$run"
    BEST="$ORIG/checkpoints/best.pt"
    if [ ! -f "$BEST" ]; then
      echo "MISSING best checkpoint: $BEST" >&2; mark_fail "$worker" "$run"; continue
    fi
    DEST="$OUT_ROOT/$worker/$run"
    mkdir -p "$(dirname "$DEST")"
    echo "[recheck] $worker/$run model=$model seed=$seed ckpt_sha=$(sha256sum "$BEST" | awk '{print $1}')"
    if ! "$PYTHON" -B scripts/three_topology_v2/train_formal.py \
      --config "$TRAIN_CONFIG" --models "$model" --seeds "$seed" \
      --split "$SPLIT_ID" --fold "$FOLD_ID" --device "$EVAL_DEVICE" \
      --eval-only --checkpoint "$BEST" \
      --run-root "$DEST" --require-new-dir \
      --batch-id "$BATCH_ID" --worker-id "recheck_$worker" \
      > "$DEST.console.log" 2>&1; then
      echo "eval-only failed: $worker/$run" >&2; mark_fail "$worker" "$run"; continue
    fi
    if ! "$PYTHON" -B scripts/three_topology_v2/compare_recheck.py \
      --original "$ORIG" --recheck "$DEST" --checkpoint "$BEST" \
      --manifest "$MANIFEST" --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" \
      --worker "$worker" --run "$run" \
      --out "$DEST/recheck.json" --atol "$ATOL" --rtol "$RTOL" \
      >> "$DEST.console.log" 2>&1; then
      echo "comparison failed: $worker/$run" >&2; mark_fail "$worker" "$run"
    fi
  done < "$EXPECTED_FILE"
fi

# Summary is driven by the EXPECTED list, not by whatever reports happen to exist.
set +e
"$PYTHON" - "$OUT_ROOT" "$BATCH_ID" "$EXPECTED_FILE" "$FAILED_FILE" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1]); batch = sys.argv[2]; expected_file = Path(sys.argv[3])
failed_file = Path(sys.argv[4])
pipeline_failed = set()
if failed_file.is_file():
    for line in failed_file.read_text(encoding="utf-8").splitlines():
        parts = line.split("	")
        if len(parts) >= 2:
            pipeline_failed.add((parts[0], parts[1]))
expected = []
for line in expected_file.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) < 6:
        continue
    expected.append({"worker": parts[0], "run": parts[1], "model": parts[2],
                     "seed": parts[3], "device": parts[4], "kind": parts[5]})
keys = [(e["worker"], e["run"]) for e in expected]
duplicates = sorted({k for k in keys if keys.count(k) > 1})

expected_set = set(keys)
actual_reports = {}
for p in sorted(out.glob("*/*/recheck.json")):
    rel = p.parent.relative_to(out)
    actual_reports[(rel.parent.name, rel.name)] = p
unexpected = sorted(k for k in actual_reports if k not in expected_set)

rows = []
passed = failed = missing = 0
for e in expected:
    key = (e["worker"], e["run"])
    report = out / e["worker"] / e["run"] / "recheck.json"
    eval_log = out / e["worker"] / e["run"] / "eval" / "eval_log.json"
    eval_pred = out / e["worker"] / e["run"] / "eval" / "val_predictions.json"
    row = {"worker": e["worker"], "run": e["run"], "model": e["model"],
           "seed": e["seed"], "report": str(report), "status": None,
           "checkpoint_sha256": None, "reasons": []}
    if not report.is_file():
        row["status"] = "missing"; missing += 1
        row["reasons"].append("recheck.json absent")
        if key in pipeline_failed:
            row["reasons"].append("eval/compare pipeline reported failure")
        rows.append(row); continue
    try:
        r = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"; failed += 1
        row["reasons"].append(f"unreadable report: {exc!r}")
        rows.append(row); continue
    row["checkpoint_sha256"] = r.get("checkpoint_sha256")
    reasons = []
    if key in pipeline_failed:
        reasons.append("eval/compare pipeline reported failure")
    if not (eval_log.is_file() and eval_pred.is_file()):
        reasons.append("eval_log.json or eval/val_predictions.json missing")
    if r.get("ok") is not True:
        reasons.append("report ok is not true")
    if not r.get("checkpoint_sha256"):
        reasons.append("checkpoint_sha256 missing")
    if r.get("identity_check", {}).get("checked") is not True or \
            r.get("identity_check", {}).get("ok") is not True:
        reasons.append("checkpoint identity not verified")
    if r.get("ids_match") is not True or r.get("entity_order_ok") is not True:
        reasons.append("sample/entity order mismatch")
    if r.get("selection_original") is None or r.get("selection_recheck") is None:
        reasons.append("selection metric missing")
    if r.get("problems"):
        reasons.append(f"reported problems: {r.get('problems')}")
    if reasons:
        row["status"] = "failed"; failed += 1; row["reasons"] = reasons
    else:
        row["status"] = "passed"; passed += 1
    rows.append(row)

all_ok = (len(expected) > 0 and not duplicates and not unexpected
          and not pipeline_failed and missing == 0 and failed == 0
          and passed == len(expected))
summ = {
    "schema": "qt-historical-best-recheck-v1", "batch_id": batch,
    "mode": "eval-only on the original 24 validation graphs; originals untouched",
    "expected": len(expected), "completed": len(expected) - missing,
    "passed": passed, "failed": failed, "missing": missing,
    "duplicate_expected": duplicates, "unexpected_reports": unexpected,
    "pipeline_failures": [f"{w}/{r}" for (w, r) in sorted(pipeline_failed)],
    "all_ok": all_ok, "tasks": rows,
}
tmp = out / "RECHECK_SUMMARY.json.tmp"
tmp.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(out / "RECHECK_SUMMARY.json")
print(json.dumps({k: summ[k] for k in ("expected", "completed", "passed", "failed",
                                       "missing", "pipeline_failures",
                                       "duplicate_expected", "unexpected_reports", "all_ok")},
                 ensure_ascii=False))
PY
SUMMARY_RC=$?
set -e
if [ "$SUMMARY_RC" -ne 0 ] || [ ! -f "$OUT_ROOT/RECHECK_SUMMARY.json" ]; then
  echo "[recheck] summary generation failed" >&2
  exit 1
fi
ALL_OK=$("$PYTHON" - "$OUT_ROOT/RECHECK_SUMMARY.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("all_ok") else "0")
PY
)
echo "[recheck] summary: $OUT_ROOT/RECHECK_SUMMARY.json"
if [ "$ALL_OK" = "1" ]; then exit 0; else exit 1; fi

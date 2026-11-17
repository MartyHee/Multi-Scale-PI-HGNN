#!/usr/bin/env bash
# Live monitor for one batch. Required: PROJECT_ROOT BATCH_ID. Optional: PYTHON.
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${BATCH_ID:?}"
BATCH_DIR="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID"
echo "== nvidia-smi =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
echo "== processes =="
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "train_formal.py" | grep -v grep || true
echo "== per-run status / train_log tail =="
if [ -d "$BATCH_DIR" ]; then
  for st in "$BATCH_DIR"/*/*/STATUS.json; do
    [ -e "$st" ] || continue
    echo "-- $st"; cat "$st"
    d="$(dirname "$st")"
    [ -f "$d/train_log.csv" ] && tail -n 3 "$d/train_log.csv"
    [ -f "$d/summary.json" ] && echo "summary all_pass: $(grep -o '\"all_pass\": [a-z]*' "$d/summary.json" | head -1)"
  done
fi
echo "== batch summary (read-only) =="
if [ -n "${PYTHON:-}" ]; then
  "$PYTHON" -B "$PROJECT_ROOT/server_ops/three_topology_v2/summarize_batch.py" --batch-dir "$BATCH_DIR" || true
else
  python3 -B "$PROJECT_ROOT/server_ops/three_topology_v2/summarize_batch.py" --batch-dir "$BATCH_DIR" || true
fi

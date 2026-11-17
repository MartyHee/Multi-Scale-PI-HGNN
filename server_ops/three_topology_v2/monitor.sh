#!/usr/bin/env bash
# Monitor a batch. Usage:
#   bash server_ops/three_topology_v2/monitor.sh <job.yaml|BATCH_ID>
# No GPU requirement for this script.
set -euo pipefail
if [ -z "${PROJECT_ROOT:-}" ]; then
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
fi
THREE_TOPO_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARG="${1:-}"
if [ -z "${BATCH_ID:-}" ]; then
  [ -n "$ARG" ] || { echo "Usage: bash $0 <job.yaml|BATCH_ID>" >&2; exit 2; }
  case "$ARG" in
    *.yaml|*.yml) BATCH_ID=$("$PYTHON" -B "$THREE_TOPO_HERE/_job_parse.py" --batch "$ARG") ;;
    *) BATCH_ID="$ARG" ;;
  esac
fi
BATCH_DIR="$PROJECT_ROOT/outputs/three_topology_v2/first_batch/$BATCH_ID"
echo "batch: $BATCH_ID dir: $BATCH_DIR"
echo "cuda_visible: ${CUDA_VISIBLE_DEVICES-<unset>}"
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
  done
fi
echo "== batch summary (read-only) =="
"$PYTHON" -B "$THREE_TOPO_HERE/summarize_batch.py" --batch-dir "$BATCH_DIR" || true

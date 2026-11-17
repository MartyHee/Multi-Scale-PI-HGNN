#!/usr/bin/env bash
# Monitor a batch (task list from the frozen batch manifest).
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
echo "== per-task status / train_log tail =="
MANIFEST="$PROJECT_ROOT/remote_jobs/three_topology_v2/manifests/$BATCH_ID.json"
if [ -f "$MANIFEST" ]; then
  TASK_FILE="$(mktemp)"
  trap 'rm -f "$TASK_FILE"' EXIT
  "$PYTHON" -B "$THREE_TOPO_HERE/_batch_manifest.py" --batch "$BATCH_ID" --paths > "$TASK_FILE" || {
    echo "ERROR: cannot read frozen task list" >&2; exit 2; }
  [ -s "$TASK_FILE" ] || { echo "ERROR: frozen task list is empty" >&2; exit 2; }
  if [ ! -d "$BATCH_DIR" ]; then echo "batch dir not present yet: $BATCH_DIR"; fi
  while IFS=$'	' read -r worker run model seed device kind; do
    [ -n "$worker" ] || continue
    d="$BATCH_DIR/$worker/$run"
    if [ -f "$d/STATUS.json" ]; then
      echo "-- $worker/$run"; cat "$d/STATUS.json"
    else
      echo "-- $worker/$run : NOT STARTED / no STATUS.json"
    fi
    if [ -f "$d/train_log.csv" ]; then tail -n 2 "$d/train_log.csv"; fi
  done < "$TASK_FILE"
else
  echo "WARN: frozen manifest missing ($MANIFEST); falling back to STATUS glob"
  if [ -d "$BATCH_DIR" ]; then
    for st in "$BATCH_DIR"/*/*/STATUS.json; do
      [ -e "$st" ] || continue
      echo "-- $st"; cat "$st"
      d="$(dirname "$st")"
      if [ -f "$d/train_log.csv" ]; then tail -n 2 "$d/train_log.csv"; fi
    done
  fi
fi
echo "== batch summary (read-only) =="
"$PYTHON" -B "$THREE_TOPO_HERE/summarize_batch.py" --batch "$BATCH_ID" || true

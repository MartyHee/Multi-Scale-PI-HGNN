#!/usr/bin/env bash
# One-shot or repeated monitor for an FE v2 batch.
#   bash server_ops/three_topology_fe_v2/monitor.sh <BATCH_ID> [watch_rounds] [interval_sec]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
BATCH="${1:?usage: monitor.sh <batch_id> [watch_rounds] [interval]}"
ROUNDS="${2:-0}"
INTERVAL="${3:-30}"
exec "$PYTHON" -B "$THREE_TOPO_HERE/monitor_table.py" --batch "$BATCH" \
  --project-root "$PROJECT_ROOT" --watch "$ROUNDS" --interval "$INTERVAL"

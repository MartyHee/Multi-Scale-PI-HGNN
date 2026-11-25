#!/usr/bin/env bash
# Bounded server throughput benchmark (NOT a formal run).
#   CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_fe_v2/run_benchmark.sh [all|topology1|topology2|topology3] [cuda:0|cpu]
#
# Caps: 16 samples, 4 val per slice, 4 fwd/bwd steps per path; no 200-epoch
# training. Writes an independent timestamped directory and never enters the
# 18-task formal batch/summary.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"
TOPO="${1:-topology1}"
DEVICE="${2:-cuda:0}"
if [ "$TOPO" = "all" ]; then TARGETS="topology1 topology2 topology3"; else TARGETS="$TOPO"; fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$PROJECT_ROOT/outputs/three_topology_fe_v2/benchmark/$TS"
mkdir -p "$OUT_DIR"
echo "[benchmark] out=$OUT_DIR device=$DEVICE cuda_visible='${CUDA_VISIBLE_DEVICES-<unset>}'"
for t in $TARGETS; do
  "$PYTHON" -B "$PROJECT_ROOT/scripts/three_topology_fe_v2/benchmark_read_path.py" \
    --device "$DEVICE" --topology "$t" --samples 16 --val-per-slice 4 --batches 4 \
    --out "$OUT_DIR/read_benchmark_${t}.json"
done
echo "[benchmark] done -> $OUT_DIR"
echo "[benchmark] This is a bounded interface/throughput check, not a performance"
echo "            estimate for the full 200-epoch run and not part of the 18 tasks."

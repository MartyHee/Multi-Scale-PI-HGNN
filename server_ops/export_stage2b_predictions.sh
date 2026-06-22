#!/bin/bash
# =============================================================================
# export_stage2b_predictions.sh
#
# Batch export full test-set predictions for all 5 Stage 2-B baseline models.
#
# Usage:
#   bash server_ops/export_stage2b_predictions.sh
#
# Before running:
#   1. Update RUN_DIR_* paths below to match actual experiment directories
#   2. conda activate pi_hgnn
#   3. Verify dataset with: bash server_ops/check_dataset.sh
#
# Output:
#   outputs/predictions/stage2b/<model_name>/<timestamp>/
#     prediction_summary.json
#     test_graph_index.csv
#     mesh_node_predictions.npz
#     beam_element_predictions.npz
#     export_metrics_check.json
# =============================================================================

set -euo pipefail

# ---- Configuration (UPDATE THESE PATHS) ----
MLP_RUN_DIR="outputs/baselines/MLP/20260620051300"
GCN_RUN_DIR="outputs/baselines/GCN/20260620123654"
GAT_RUN_DIR="outputs/baselines/GAT/20260620161447"
RGCN_RUN_DIR="outputs/baselines/RGCN/20260621042016"
HGT_RUN_DIR="outputs/baselines/HGT/20260622103144"

OUTPUT_ROOT="outputs/predictions/stage2b"
BATCH_SIZE=16
NUM_WORKERS=4

# ---- Environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "Stage 2-B Full Prediction Export"
echo "Date: $(date)"
echo "Project: $(pwd)"
echo "========================================================================"

# ---- Python env check ----
echo ""
echo "Environment:"
python -c "import torch; print(f'  torch {torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import numpy; print(f'  numpy {numpy.__version__}')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# ---- Export Function ----
export_model() {
    local model_name="$1"
    local run_dir="$2"
    local extra_args="${3:-}"

    echo ""
    echo "------------------------------------------------------------------"
    echo "Exporting: $model_name"
    echo "  run_dir: $run_dir"
    echo "------------------------------------------------------------------"

    if [ ! -d "$run_dir" ]; then
        echo "  [SKIP] Run directory not found: $run_dir"
        return 1
    fi

    if [ ! -f "$run_dir/best_model.pt" ]; then
        echo "  [SKIP] best_model.pt not found in $run_dir"
        return 1
    fi

    echo "  Starting export..."
    python scripts/export_full_predictions.py \
        --model "$model_name" \
        --run-dir "$run_dir" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device cuda \
        --output-dir "$OUTPUT_ROOT" \
        $extra_args

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  [OK] $model_name export successful"
    else
        echo "  [FAIL] $model_name export failed (exit code: $exit_code)"
    fi
    return $exit_code
}

# ---- Run exports ----
# Note: models are ordered by speed (fastest first)
# MLP export
export_model "mlp" "$MLP_RUN_DIR"

# GCN export
export_model "gcn" "$GCN_RUN_DIR"

# GAT export
export_model "gat" "$GAT_RUN_DIR"

# RGCN export
export_model "rgcn" "$RGCN_RUN_DIR"

# HGT export
export_model "hgt" "$HGT_RUN_DIR"

# ---- Summary ----
echo ""
echo "========================================================================"
echo "EXPORT COMPLETE"
echo "========================================================================"
echo ""
echo "Output directory: $OUTPUT_ROOT"
echo ""
echo "File sizes:"
find "$OUTPUT_ROOT" -name "*.npz" -exec ls -lh {} \; 2>/dev/null || echo "  (no npz files found)"
echo ""
echo "Total size:"
du -sh "$OUTPUT_ROOT" 2>/dev/null || echo "  (directory not found)"
echo ""
echo "To archive predictions:"
echo "  tar -czf remote_artifacts/stage2b_full_predictions_\$(date +%Y%m%d%H%M%S).tar.gz $OUTPUT_ROOT"
echo ""
echo "To view output structure:"
echo "  find $OUTPUT_ROOT -maxdepth 3 -type f | head -50"
echo "========================================================================"

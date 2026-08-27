#!/bin/bash
# =============================================================================
# export_paper_upgrade_predictions.sh
#
# Batch export full test-set predictions for the Stage Paper-4B/4C decision runs
# (global-node vs macro-anchor comparison). Uses scripts/export_full_predictions.py
# which now supports the paper-upgrade model names:
#     hgt, ms_hgt, hgt_global, ms_hgt_random_anchor, edge_aware_mpnn
#
# Usage:
#   bash server_ops/export_paper_upgrade_predictions.sh
#
# Before running:
#   1. Locate the exact server run dirs (server timestamps differ from local):
#        find outputs/baselines -maxdepth 3 -name best_model.pt | sort
#      and confirm the seed of each run dir via config_resolved.yaml:
#        grep -A2 '^train:' <run_dir>/config_resolved.yaml
#   2. Fill RUN_*_DIR variables below with the paths found.
#      Any variable left empty is skipped with a warning.
#   3. conda activate pi_hgnn
#   4. bash server_ops/check_dataset.sh
#
# Output (per model):
#   outputs/predictions/paper_upgrade/<model_name>/<timestamp>/
#     prediction_summary.json
#     test_graph_index.csv
#     mesh_node_predictions.npz
#     beam_element_predictions.npz
#     export_metrics_check.json
#     (plus config_resolved.yaml copied alongside for traceability)
# =============================================================================

set -uo pipefail   # do NOT set -e: we want to attempt every export and report

# ---- Configuration (FILL THESE IN with server run dirs) ----
#   format: outputs/baselines/<MODEL_UPPER>/<timestamp>
HGT_RUN_DIR="outputs/baselines/HGT/__FILL__"
MS_HGT_RUN_DIR="outputs/baselines/MS_HGT/__FILL__"
HGT_GLOBAL_RUN_DIR="outputs/baselines/HGT_GLOBAL/__FILL__"
MS_HGT_RANDOM_ANCHOR_RUN_DIR="outputs/baselines/MS_HGT_RANDOM_ANCHOR/__FILL__"
EDGE_AWARE_MPNN_RUN_DIR="outputs/baselines/EDGE_AWARE_MPNN/__FILL__"

# For multi-seed exports of the SAME model, copy the RUN_*_DIR above to a
# RUN_*_DIR2 variable (e.g. HGT_RUN_DIR2 for HGT-3 seed2026) and add an extra
# export_model line at the bottom. The script skips any dir still containing
# "__FILL__" or that does not exist.

OUTPUT_ROOT="outputs/predictions/paper_upgrade"
BATCH_SIZE=16
NUM_WORKERS=4

# ---- Environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "Stage Paper-4B/4C Full Prediction Export (global vs macro decision)"
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

    echo ""
    echo "------------------------------------------------------------------"
    echo "Exporting: $model_name"
    echo "  run_dir: $run_dir"
    echo "------------------------------------------------------------------"

    if [[ "$run_dir" == *__FILL__* ]]; then
        echo "  [SKIP] RUN_DIR not filled in (still contains __FILL__)."
        return 1
    fi

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
        --output-dir "$OUTPUT_ROOT"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # Copy resolved config for traceability (per-model export dir is newest)
        local out_dir
        out_dir=$(ls -td "$OUTPUT_ROOT/$model_name"/*/ 2>/dev/null | head -1)
        if [ -n "$out_dir" ] && [ -d "$out_dir" ]; then
            cp "$run_dir/config_resolved.yaml" "$out_dir/config_resolved.yaml" 2>/dev/null \
                && echo "  config_resolved.yaml copied to $out_dir" || true
        fi
        echo "  [OK] $model_name export successful"
    else
        echo "  [FAIL] $model_name export failed (exit code: $exit_code)"
    fi
    return $exit_code
}

# ---- Run exports (must / should lists) ----
# MUST: decision-critical runs for analyze_global_vs_macro_predictions.py
export_model "hgt" "$HGT_RUN_DIR"                 # HGT-3 seed42
export_model "ms_hgt" "$MS_HGT_RUN_DIR"           # MS-HGT coord gated seed42
export_model "hgt_global" "$HGT_GLOBAL_RUN_DIR"   # HGT-global seed42

# SHOULD: additional decision / ablation runs
export_model "ms_hgt_random_anchor" "$MS_HGT_RANDOM_ANCHOR_RUN_DIR"
export_model "edge_aware_mpnn" "$EDGE_AWARE_MPNN_RUN_DIR"

# ---- Summary ----
echo ""
echo "========================================================================"
echo "EXPORT COMPLETE"
echo "========================================================================"
echo ""
echo "Output directory: $OUTPUT_ROOT"
echo ""
echo "Exported runs:"
find "$OUTPUT_ROOT" -name "mesh_node_predictions.npz" -exec dirname {} \; 2>/dev/null | sort
echo ""
echo "To archive predictions:"
echo "  tar -czf remote_artifacts/paper_upgrade_predictions_\$(date +%Y%m%d%H%M%S).tar.gz $OUTPUT_ROOT"
echo ""
echo "Remember to also bring back each run's:"
echo "  prediction_summary.json / export_metrics_check.json / config_resolved.yaml"
echo "========================================================================"

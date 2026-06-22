# Full Prediction Export — Stage 2-B Diagnostic Enabler

## 1. Why Full Prediction Export?

The Stage 2-B Diagnostic Evaluation (`scripts/analyze_stage2b_diagnostics.py`) found
that current artifacts do not contain full test predictions, blocking:

- **Tail error analysis**: P50 / P90 / P95 / P99 absolute and relative error
- **High/low response subset analysis**: Top/bottom 10% by true value magnitude
- **Region-wise metrics**: Support, midspan, beam-plate connection regions
- **Physical consistency**: Support BC residual, structural_link displacement consistency

This export step provides the missing prediction files for all 5 baseline models
(MLP, GCN, GAT, RGCN, HGT).

## 2. What Is Exported

For each model, the script creates:

```
outputs/predictions/stage2b/<model_name>/<timestamp>/
├── prediction_summary.json           # metadata, config, export info
├── test_graph_index.csv              # per-graph metadata (graph_id, sample_id, ...)
├── mesh_node_predictions.npz         # disp predictions + true values + metadata
├── beam_element_predictions.npz      # force predictions + true values + metadata
└── export_metrics_check.json         # recomputed metrics vs metrics_summary.json
```

### 2.1 mesh_node_predictions.npz

| Field | Shape | Description |
|-------|-------|-------------|
| `y_true_disp` | (N_mesh, 6) | Ground truth displacement [Dx..Rz] |
| `y_pred_disp` | (N_mesh, 6) | Predicted displacement [Dx..Rz] |
| `graph_id` | (N_mesh,) | Per-node graph assignment index |
| `node_id` | (N_mesh,) | Local node ID within graph |
| `node_xyz` | (N_mesh, 3) | Node coordinates (original scale) |
| `support_flags` | (N_mesh, 6) | BC constraints [Dx_fix..Rz_fix] (0/1) |

### 2.2 beam_element_predictions.npz

| Field | Shape | Description |
|-------|-------|-------------|
| `y_true_force` | (N_beam, 12) | Ground truth beam end forces [Fx_I..Mz_J] |
| `y_pred_force` | (N_beam, 12) | Predicted beam end forces [Fx_I..Mz_J] |
| `graph_id` | (N_beam,) | Per-element graph assignment index |
| `element_id` | (N_beam,) | Local element ID within graph |

### 2.3 test_graph_index.csv

| Column | Description |
|--------|-------------|
| `graph_id` | Zero-based graph index in test set |
| `sample_id` | Original sample identifier (if available) |
| `load_case_id` | Original load case identifier (if available) |
| `num_mesh_nodes` | Number of mesh nodes in this graph |
| `num_beam_elements` | Number of beam elements in this graph |

### 2.4 export_metrics_check.json

Recomputed metrics that should match `metrics_summary.json` values:

| Field | Description |
|-------|-------------|
| `disp_r2` | Macro-averaged displacement R2 |
| `force_r2` | Macro-averaged force R2 |
| `disp_mae` | Macro-averaged displacement MAE |
| `force_mae` | Macro-averaged force MAE |
| `combined_rel_mae` | Average of disp/force overall RelMAE |

## 3. File Size Considerations

With ~3500 test graphs × ~1056 mesh nodes × ~1646 beam elements:

| Component | Rows | Columns | Approx NPZ Size |
|-----------|------|---------|-----------------|
| `mesh_node_predictions.npz` | ~3.7M | 6 + 6 + 1 + 1 + 3 + 6 | ~100-150 MB |
| `beam_element_predictions.npz` | ~5.8M | 12 + 12 + 1 + 1 | ~150-200 MB |
| Total per model | — | — | ~250-350 MB |

For 5 models: **~1.5-1.8 GB total** (compressed NPZ).

**Do not commit to Git.** Use `tar.gz` for archiving.

## 4. Local Smoke Test

```bash
# Quick export (2 graphs only)
D:\CodeData\software\Anaconda\Anaconda3\envs\llm\python.exe scripts/export_full_predictions.py ^
    --model mlp --run-dir outputs/baselines/MLP/20260620051300 ^
    --max-graphs 2 --batch-size 1 --device cpu ^
    --output-dir outputs/predictions/smoke_mlp
```

Expected output:
- ✅ Model loaded
- ✅ 2 graphs processed
- ✅ `mesh_node_predictions.npz` created
- ✅ `beam_element_predictions.npz` created
- ✅ `export_metrics_check.json` metrics match original

## 5. Server Full Export

### Step 1: Update run_dir paths

Edit `server_ops/export_stage2b_predictions.sh` and verify paths:

```bash
MLP_RUN_DIR="outputs/baselines/MLP/20260620051300"
GCN_RUN_DIR="outputs/baselines/GCN/20260620123654"
GAT_RUN_DIR="outputs/baselines/GAT/20260620161447"
RGCN_RUN_DIR="outputs/baselines/RGCN/20260621042016"
HGT_RUN_DIR="outputs/baselines/HGT/20260622103144"
```

### Step 2: Run

```bash
cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN
git fetch && git checkout main && git pull
conda activate pi_hgnn

# Verify env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
bash server_ops/check_dataset.sh

# Export all models
bash server_ops/export_stage2b_predictions.sh
```

### Step 3: Check output

```bash
find outputs/predictions/stage2b -maxdepth 3 -type f | head -50
du -sh outputs/predictions/stage2b
```

### Step 4: Archive (optional)

```bash
tar -czf remote_artifacts/stage2b_full_predictions_$(date +%Y%m%d%H%M%S).tar.gz \
    outputs/predictions/stage2b
```

## 6. Integration with analyze_stage2b_diagnostics.py

Once full predictions are available, `analyze_stage2b_diagnostics.py` can:

1. **Tail error**: Load `mesh_node_predictions.npz`, compute P50/P90/P95/P99
2. **Region metrics**: Assign region labels from `node_xyz` + `support_flags`
3. **Physical diagnostics**: Use `support_flags` for BC residual, `graph_id` for structural_link
4. **High/low response subsets**: Filter by `y_true` magnitude percentiles

The `--predictions-dir` flag (not yet implemented) will enable these modes.
Currently, the script falls back to generating requirement documents.

## 7. Git Exclusion

The `outputs/predictions/` directory is already gitignored. Verify:

```bash
# outputs/predictions/ should be in .gitignore
grep "predictions" .gitignore
```

Prediction artifacts should not be committed or included in standard experiment
artifacts. Archive separately only when needed for analysis.

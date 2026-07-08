# Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Structural Finite Element Surrogate Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![PyG 2.7](https://img.shields.io/badge/PyG-2.7-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A multi-scale physics-informed heterogeneous graph neural network framework for AI-based structural finite element (FE) surrogate modeling of steel truss girder bridges. This work is targeting **ICTAI 2026** (Tools with Artificial Intelligence / AI Applications track).

---

## Project Background

Finite element analysis (FEA) is the standard tool for structural engineering design and evaluation, but each simulation is computationally expensive. For tasks requiring many repeated simulations — design space exploration, uncertainty quantification, or optimization — a fast surrogate model is highly desirable.

**The core challenge:** Structural FE simulation data is inherently heterogeneous. A typical steel truss girder model contains:

- **Mesh nodes** with coordinate, load, and boundary condition features
- **Beam elements** with section properties, material properties, and geometry
- **Plate elements** with thickness and material properties
- **Structural links** representing rigid/elastic connections between nodes
- Complex relational dependencies across these entity types

Standard MLP or homogeneous GNN approaches cannot fully exploit this heterogeneous structure, leading to suboptimal predictions — particularly for displacement components (e.g., Dy) and high-response regions.

**This project proposes:** A **Multi-Scale Physics-Informed Heterogeneous Graph Transformer (MS-PI-HGT)** that explicitly models the structural FE system as a heterogeneous graph with typed message passing, macro-anchor multi-scale fusion, physics-informed regularization, and conformal uncertainty quantification.

### Prediction Targets

- **Node displacement** (6-DOF: Dx, Dy, Dz, Rx, Ry, Rz) — 1056 mesh nodes per graph
- **Beam-end internal force** (12 components: Fx_I..Mz_J) — 1646 beam elements per graph

---

## Key Features

- **Heterogeneous graph construction** from structural FE data (`mesh_node`, `beam_element`, `plate_element`, `structural_link`)
- **Typed graph transformer backbone** (HGT-style) with relation-specific message passing
- **Multi-scale macro-anchor fusion** — stiffness-aware graph pooling and cross-scale gated fusion for long-range force transfer
- **Physics-informed regularization** — support boundary condition loss + structural link consistency loss
- **Component-wise split conformal UQ** — distribution-free calibrated prediction intervals (18 output components)
- **Complete baseline suite** — MLP, GCN, GAT, RGCN, HGT for systematic ablation
- **Reproducible experiment pipeline** — local smoke test → server training → artifact recovery
- **Comprehensive diagnostics** — region-wise metrics, tail-error analysis, physical consistency checks

---

## Repository Structure

```
Multi-Scale-PI-HGNN/
├── configs/                    # Unified YAML configuration
│   ├── dataset.yaml            # Dataset path & split config
│   ├── hetero_dataset.yaml     # Heterogeneous dataset v2 schema
│   ├── models.yaml             # Model-specific architecture params
│   ├── models_baseline.yaml    # Baseline model architecture params
│   ├── train.yaml              # Unified training config
│   └── train_baseline.yaml     # Baseline training config
│
├── src/                        # Core source code
│   ├── data/                   # Dataset construction & loading
│   │   ├── build_hetero_graph_dataset.py  # Hetero graph builder (v2)
│   │   ├── hetero_graph_dataset.py        # PyG HeteroData dataset
│   │   ├── hetero_schema.py               # Schema definitions
│   │   ├── hetero_split.py                # by_sample / by_loadcase split
│   │   ├── hetero_transforms.py           # Graph transforms
│   │   └── validate_hetero_dataset.py     # Dataset validation
│   ├── models/                 # Model definitions
│   │   └── baselines/
│   │       ├── mlp_baseline.py            # MLP baseline
│   │       ├── homogeneous_gcn.py         # Homogeneous GCN
│   │       ├── homogeneous_gat.py         # Homogeneous GAT
│   │       ├── hetero_rgcn.py             # RGCN / HeteroConv
│   │       ├── hgt_baseline.py            # HGT baseline
│   │       ├── ours_base.py               # Ours-base (edge-attr aware)
│   │       └── ms_hgt.py                  # MS-HGT (macro anchor + fusion)
│   ├── trainers/               # Training loops, early stopping, losses
│   │   ├── trainer.py                     # Training loop
│   │   ├── baseline_trainer.py            # Baseline training loop
│   │   ├── losses.py                      # Supervised losses
│   │   └── early_stopping.py              # Early stopping
│   ├── losses/                 # Physics-informed losses (Stage 5)
│   │   └── physics_loss.py                # BC + link consistency loss
│   └── utils/                  # Metrics, experiment tracking, I/O
│       ├── metrics.py                     # Evaluation metrics
│       └── experiment.py                  # Config, logging, paths
│
├── scripts/                    # Standalone analysis scripts
│   ├── compute_conformal.py               # Stage 6: conformal UQ
│   ├── analyze_conformal.py               # Stage 6: coverage analysis
│   ├── export_full_predictions.py         # Full test set prediction export
│   ├── physics_diagnostics.py             # Stage 5: physics analysis
│   ├── compute_region_labels.py           # Region assignment
│   └── ...                                # Additional diagnostics
│
├── remote_jobs/                # Server job configurations (YAML)
│   ├── server_mlp_full.yaml
│   ├── server_gcn_full.yaml
│   ├── server_gat_full.yaml
│   ├── server_rgcn_full.yaml
│   ├── server_hgt_full.yaml
│   ├── server_ms_hgt_gated.yaml
│   ├── server_ms_hgt_additive.yaml
│   ├── server_ms_pi_hgt_bc.yaml
│   ├── server_ms_pi_hgt_link.yaml
│   └── server_ms_pi_hgt_full.yaml
│
├── server_ops/                 # Server-side run scripts
│   ├── check_dataset.sh
│   ├── run_job.sh
│   ├── export_stage2b_predictions.sh
│   └── package_results.sh
│
├── train.py                    # Unified training entry (Stage 3+)
├── train_baseline.py           # Baseline training entry (Stage 2)
├── requirements.txt            # Python dependencies
│
├── docs/                       # Documentation & experiment records
│   ├── development_log.md      # Continuous development log
│   ├── stage2a_baseline_suite_report.md
│   ├── stage2b_baseline_results_draft.md
│   ├── stage3_ours_base_design.md
│   ├── stage4_macro_anchor_design.md
│   ├── stage4_result_lock.md
│   ├── stage5_physics_loss_design.md
│   ├── stage5_experiment_plan.md
│   ├── stage6_uq_design.md
│   ├── stage6_experiment_plan.md
│   ├── stage6_calibration_split_audit.md
│   ├── ictai_research_story.md
│   ├── method_literature_inventory.md
│   ├── experiment_result_and_figure_audit.md
│   └── ictai_paper_preparation_summary.md
│
├── processed/                  # [git-ignored] Built graph datasets
│   └── hetero_graph_dataset_v2/  # Current mainline dataset
│
├── outputs/                    # [git-ignored] Training & diagnostic output
│   ├── baselines/              # Baseline model checkpoints & logs
│   └── diagnostics/           # Diagnostic reports & figures
│
├── remote_artifacts/           # [git-ignored] Server training artifacts
│
└── logs/                       # [git-ignored] Runtime logs
```

---

## Environment

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended for environment management)
- **CUDA-capable GPU** (tested on 8× RTX 4090; single GPU sufficient for training)

### Installation

```bash
conda create -n pi_hgnn python=3.10 -y
conda activate pi_hgnn

# Install PyTorch 2.6 + CUDA 11.8 (GPU server)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch 2.6 (CPU-only for local smoke tests)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install torch_scatter matching PyTorch version
pip install torch-scatter --index-url https://data.pyg.org/whl/torch-2.6.0+cu118  # or +cpu

# Install PyG and remaining dependencies
pip install torch_geometric==2.7.0
pip install -r requirements.txt
```

### Path Notes

- **Local development (Windows):** Uses Python at `D:\CodeData\software\Anaconda\Anaconda3\envs\llm\python.exe`
- **Server training (Linux/CUDA):** Uses `conda activate pi_hgnn` with system Python
- All paths in configs use absolute paths; adjust `raw_data_dir` for your environment

---

## Data Preparation

### Mainline Dataset: `processed/hetero_graph_dataset_v2`

The v2 heterogeneous graph dataset is built from `GraphTrainingData2`, which contains 70 steel truss girder design samples × 500 load cases = **35,000 graph instances**.

Each graph contains:
- **1,056** `mesh_node` entities (15-dim features: xyz, loads, DOF constraint flags)
- **1,646** `beam_element` entities (11-dim features: section, material, geometry)
- **832** `plate_element` entities (6-dim features: thickness, material)
- **132 directed (66 physical)** `structural_link` edges per graph (constant across all 70 samples × 500 load cases; 10-dim attrs: stiffness, beta angle)
- 5 edge types (`belongs_to_beam`, `rev_belongs_to_beam`, `belongs_to_plate`, `rev_belongs_to_plate`, `structural_link`)

### Build the Dataset

```bash
# Build v2 heterogeneous dataset
python src/data/build_hetero_graph_dataset.py --config configs/hetero_dataset.yaml
```

### Validate the Dataset

```bash
python src/data/validate_hetero_dataset.py --data-dir processed/hetero_graph_dataset_v2
```

> **Note:** The raw data (`GraphTrainingData2/`) and processed dataset (`processed/`) are large and **not tracked in git**.

---

## Reproduction Commands

### 1. Dataset Validation / Smoke Check

```bash
# Quick dataset sanity check (local)
python -c "
from src.data.hetero_graph_dataset import HeteroGraphDataset
ds = HeteroGraphDataset('processed/hetero_graph_dataset_v2', max_graphs=2)
print(f'Dataset loaded: {len(ds)} graphs')
g = ds[0]
print(f'Node types: {g.node_types}')
print(f'Edge types: {g.edge_types}')
"
```

### 2. Baseline Training Example (Smoke Test)

```bash
# MLP smoke test — 2 graphs, 2 epochs (local CPU)
python train_baseline.py --model mlp_baseline \
    --max-graphs 2 --epochs 2 --batch-size 1 \
    --device cpu --output-root outputs/baselines
```

### 3. Full Baseline Training (Server)

```bash
# Example: HGT (remote_jobs/server_hgt_full.yaml)
cd /path/to/Multi-Scale-PI-HGNN
git fetch && git checkout <branch> && git pull
bash server_ops/check_dataset.sh
bash server_ops/run_job.sh remote_jobs/server_hgt_full.yaml
```

### 4. Full MS-PI-HGT Training (Server)

```bash
bash server_ops/run_job.sh remote_jobs/server_ms_pi_hgt_full.yaml
```

### 5. Prediction Export

```bash
python scripts/export_full_predictions.py --model mshgt \
    --run-dir outputs/baselines/MS_HGT/20260626170354 \
    --batch-size 8 --device cuda \
    --output-dir outputs/predictions/stage6
```

### 6. Physics Diagnostics (Stage 5)

```bash
python scripts/physics_diagnostics.py \
    --predictions-dir outputs/predictions/stage5 \
    --output-dir outputs/diagnostics/stage5_physics
```

### 7. Conformal UQ (Stage 6)

```bash
# Compute conformal intervals
python scripts/compute_conformal.py \
    --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test \
    --split-mode test_graph_50_50 \
    --alpha 0.10 0.05 \
    --output-dir outputs/diagnostics/stage6_uq

# Analyze and visualize
python scripts/analyze_conformal.py \
    --conformal-dir outputs/diagnostics/stage6_uq/<timestamp> \
    --output-dir outputs/diagnostics/stage6_uq/<timestamp>
```

---

## Main Results Summary

| Method | Graph Type | Typed Message | Multi-Scale | Physics Loss | Disp R² | Dy R² | Force R² | RelMAE | Params |
|--------|------------|:-------------:|:-----------:|:------------:|:-------:|:-----:|:--------:|:------:|:------:|
| MLP | none | no | no | no | 0.8554 | 0.1833 | 0.9824 | 0.0884 | 96,274 |
| GCN | homogeneous | no | no | no | 0.8476 | 0.1778 | 0.9696 | 0.1227 | 76,050 |
| GAT | homogeneous | no | no | no | 0.8421 | 0.1649 | 0.9632 | 0.1361 | 76,818 |
| RGCN/HeteroConv | heterogeneous | relation-specific | no | no | 0.9366 | 0.670 | 0.9878 | 0.0724 | 520,338 |
| HGT | heterogeneous | typed attention | no | no | 0.9765 | 0.905 | 0.9893 | 0.0676 | 744,279 |
| MS-HGT additive | heterogeneous | typed attention | macro (additive) | no | 0.9950 | 0.993 | 0.9931 | 0.0531 | 844,119 |
| MS-HGT gated | heterogeneous | typed attention | macro (gated) | no | **0.9952** | 0.993 | 0.9928 | **0.0519** | 893,527 |
| MS-PI-HGT-BC | heterogeneous | typed attention | macro (gated) | BC only | 0.9951 | 0.993 | 0.9934 | 0.0529 | 893,527 |
| MS-PI-HGT-Link | heterogeneous | typed attention | macro (gated) | Link only | 0.9952 | **0.993** | 0.9934 | 0.0515 | 893,527 |
| **MS-PI-HGT-Full** | heterogeneous | typed attention | macro (gated) | BC+Link | 0.9948 | 0.993 | **0.9933** | 0.0516 | **893,527** |

**Stage 6 Conformal UQ (MS-PI-HGT-Full backbone, α=0.10):**

| Domain | 90% Coverage | 90% Avg Half-Width* | 95% Coverage | 95% Avg Half-Width* |
|--------|:-----------:|:-------------:|:-----------:|:-------------:|
| Displacement (all DOF) | 89.74% | 0.000476 m/rad | 94.78% | 0.000567 m/rad |
| Force (all components) | 89.97% | 40,136 N/N·m | 94.97% | 69,715 N/N·m |
| Per-DOF max gap | 0.004 | — | 0.003 | — |

\* Average half-width = conformal quantile $q$ of absolute residual; prediction interval is $[\hat{y} - q, \hat{y} + q]$.

## Reproducibility Notes

1. **Experiments are tracked by git commit + config.** Each server training job records its commit hash, resolved config, and dataset version.
2. **Smoke test first, then server.** All modifications are tested locally with a small subset (2 graphs, 2 epochs) before full server training.
3. **Artifacts are archived.** Server training outputs are packaged and downloaded to `remote_artifacts/` for local analysis.
4. **Files not tracked in git:**
   - `processed/` — built datasets (large)
   - `outputs/` — training checkpoints, logs, diagnostics
   - `remote_artifacts/` — server artifact tarballs
   - `logs/` — runtime logs
   - `*.pt`, `*.pth`, `*.ckpt` — model files
5. **Random seed = 42** across all experiments (configurable).
6. **Split mode:** `by_sample` (train/val/test = 80/10/10 over 70 design samples) — tests generalization to unseen design samples under the same topology.
7. **Standardization:** Train-only statistics; val/test use training set stats.

---

## Limitations

1. **Shared topology (by-sample split).** All 70 samples share the same mesh topology. Current results demonstrate generalization to unseen design parameters, **not cross-topology generalization**.
2. **Physics losses are not complete equilibrium.** The BC and link consistency losses penalize specific violation types but do not enforce full element-level or system-level equilibrium, energy conservation, or constitutive relations.
3. **Conformal UQ provides marginal coverage for each output component**, not pointwise or simultaneous joint guarantee. Joint coverage (all 6 DOF simultaneously) is significantly lower — this reflects a known limitation of component-wise marginal intervals; simultaneous coverage would require vector-level or graph-level conformal calibration, typically with wider intervals.
4. **Structural link loss convergence is limited** at the current λ value. The primary benefit of the Full variant comes from combined regularization rather than link loss convergence.
5. **Plate element outputs are not predicted** — plate force/stress labels are unavailable in the current dataset.
6. **All comparisons are internal.** Baselines (MLP, GCN, GAT, RGCN, HGT) are re-implemented within the same pipeline for fair comparison, not against external published results on different data.

---

## Paper Status

This project is being prepared for submission to **ICTAI 2026** (Tools with Artificial Intelligence). See:
- [docs/ictai_research_story.md](docs/ictai_research_story.md) — Research narrative and claims
- [docs/ictai_paper_preparation_summary.md](docs/ictai_paper_preparation_summary.md) — Paper readiness assessment

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ms_pi_hgt_2026,
  title={Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Structural Finite Element Surrogate Modeling},
  author={To be updated},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is for research purposes. See LICENSE file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Inspired by [Graph Neural Networks in Computational Mechanics](https://arxiv.org/abs/2107.12524) and related works

# Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Structural Finite Element Surrogate Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![PyG 2.7](https://img.shields.io/badge/PyG-2.7-green.svg)](https://pytorch-geometric.readthedocs.io/)

A heterogeneous graph neural network framework for AI-based structural finite element (FE) surrogate modeling of steel truss girder bridges. The framework models the FE system as a heterogeneous graph with typed message passing, macro-anchor multi-scale fusion, physics-informed regularization, and conformal uncertainty quantification.

---

## Project Overview

Finite element analysis (FEA) is the standard tool for structural engineering design and evaluation, but each simulation is computationally expensive. For tasks requiring many repeated simulations — design space exploration, uncertainty quantification, or optimization — a fast surrogate model is highly desirable.

**Core challenge:** Structural FE simulation data is inherently heterogeneous. A typical steel truss girder model contains mesh nodes with load/boundary features, beam elements with section/material properties, plate elements with thickness, structural links representing connections, and complex relational dependencies across these entity types. Standard MLP or homogeneous GNN approaches cannot fully exploit this heterogeneous structure.

**This project proposes:** A **Multi-Scale Physics-Informed Heterogeneous Graph Transformer (MS-PI-HGT)** that explicitly models the structural FE system as a heterogeneous graph with typed message passing, macro-anchor multi-scale fusion, physics-informed regularization, and conformal uncertainty quantification.

### Prediction Targets

- **Node displacement** (6-DOF: Dx, Dy, Dz, Rx, Ry, Rz) — predicted for 1056 mesh nodes per graph
- **Beam-end internal force** (12 components: Fx_I..Mz_J) — predicted for 1646 beam elements per graph

### Supported Models

| Model | Graph Type | Typed Message | Key Features |
|-------|-----------|:-------------:|-------------|
| MLP | none | no | Strong local-feature baseline |
| GCN | homogeneous | no | Homogeneous graph convolution |
| GAT | homogeneous | no | Homogeneous graph attention |
| RGCN / HeteroConv | heterogeneous | relation-specific | Type-aware message passing |
| HGT | heterogeneous | typed attention | Type-dependent attention |
| MS-HGT | heterogeneous | typed attention | Macro anchor + cross-scale fusion |
| MS-PI-HGT | heterogeneous | typed attention | + physics-informed regularization + UQ |

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
│   ├── losses/                 # Physics-informed losses
│   │   └── physics_loss.py                # BC + link consistency loss
│   └── utils/                  # Metrics, experiment tracking, I/O
│       ├── metrics.py                     # Evaluation metrics
│       └── experiment.py                  # Config, logging, paths
│
├── scripts/                    # Standalone analysis scripts
│   ├── compute_conformal.py               # Conformal UQ computation
│   ├── analyze_conformal.py               # Coverage analysis & visualization
│   ├── export_full_predictions.py         # Full test set prediction export
│   ├── physics_diagnostics.py             # Physics consistency diagnostics
│   ├── compute_region_labels.py           # Region assignment for analysis
│   └── ...                                # Additional diagnostics
│
├── remote_jobs/                # Server training job configurations (YAML)
│   ├── server_mlp_full.yaml
│   ├── server_gcn_full.yaml
│   ├── server_gat_full.yaml
│   ├── server_rgcn_full.yaml
│   ├── server_hgt_full.yaml
│   ├── server_ms_hgt_gated.yaml
│   ├── server_ms_hgt_additive.yaml
│   └── server_ms_pi_hgt_full.yaml
│
├── server_ops/                 # Server-side run scripts
│   ├── check_dataset.sh
│   ├── run_job.sh
│   ├── export_stage2b_predictions.sh
│   └── package_results.sh
│
├── train.py                    # Unified training entry (Stage 3+ models)
├── train_baseline.py           # Baseline training entry
├── requirements.txt            # Python dependencies
└── README.md
```

### **Not tracked in git** (generated locally or large artifacts):

```
processed/          # Built graph datasets
outputs/            # Training checkpoints, logs, diagnostics
remote_artifacts/   # Server training artifact tarballs
logs/               # Runtime logs
```

---

## Environment Setup

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended)
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

- **Local development (Windows):** Use the Python interpreter configured for your environment
- **Server training (Linux/CUDA):** Activate the `pi_hgnn` conda environment
- All paths in configs use absolute paths; adjust `raw_data_dir` for your environment

---

## Dataset Preparation

### Mainline Dataset: `processed/hetero_graph_dataset_v2`

The v2 heterogeneous graph dataset is built from structural FE data containing 70 steel truss girder design samples × 500 load cases = **35,000 graph instances**.

Each graph contains:

- **1,056** `mesh_node` entities (15-dim features: coordinates, loads, DOF constraint flags)
- **1,646** `beam_element` entities (11-dim features: section properties, material properties, geometry)
- **832** `plate_element` entities (6-dim features: thickness, material properties)
- **132 directed (66 physical)** `structural_link` edges (10-dim attributes: stiffness coefficients, beta angle)
- 5 edge types: `belongs_to_beam`, `rev_belongs_to_beam`, `belongs_to_plate`, `rev_belongs_to_plate`, `structural_link`

### Supervised Labels

- `mesh_node.y_disp`: 6-DOF displacement [Dx, Dy, Dz, Rx, Ry, Rz]
- `beam_element.y_force`: 12-component beam-end force [Fx_I, Fy_I, Fz_I, Mx_I, My_I, Mz_I, Fx_J, Fy_J, Fz_J, Mx_J, My_J, Mz_J]

### Build the Dataset

```bash
python src/data/build_hetero_graph_dataset.py --config configs/hetero_dataset.yaml
```

### Validate the Dataset

```bash
python src/data/validate_hetero_graph_dataset.py --data-dir processed/hetero_graph_dataset_v2
```

### Quick Sanity Check

```bash
python -c "
from src.data.hetero_graph_dataset import HeteroGraphDataset
ds = HeteroGraphDataset('processed/hetero_graph_dataset_v2', max_graphs=2)
print(f'Dataset loaded: {len(ds)} graphs')
g = ds[0]
print(f'Node types: {g.node_types}')
print(f'Edge types: {g.edge_types}')
"
```

---

## Training

### Baseline Models (Smoke Test — Local CPU)

```bash
# MLP smoke test — 2 graphs, 2 epochs
python train_baseline.py --model mlp_baseline \
    --max-graphs 2 --epochs 2 --batch-size 1 \
    --device cpu --output-root outputs/baselines
```

Available baseline models: `mlp_baseline`, `homogeneous_gcn`, `homogeneous_gat`, `hetero_rgcn`, `hgt_baseline`, `ours_base`, `ms_hgt`, `ms_pi_hgt`.

### Full Training (Server)

```bash
cd /path/to/Multi-Scale-PI-HGNN
git fetch && git checkout <branch> && git pull
bash server_ops/check_dataset.sh
bash server_ops/run_job.sh remote_jobs/server_hgt_full.yaml
```

Available job configs cover all supported models. Adjust epochs, batch size, and device in the YAML config or via command-line overrides.

### Training Monitor

```bash
nvidia-smi
tail -f logs/remote/<job_name>_<timestamp>.log
```

### Resume Training

```bash
python train.py --resume outputs/<ModelName>/<run_timestamp>/last_model.pt
```

---

## Evaluation and Diagnostics

### Prediction Export

```bash
python scripts/export_full_predictions.py --model mshgt \
    --run-dir outputs/baselines/MS_HGT/<RUN_DIR> \
    --batch-size 8 --device cuda \
    --output-dir outputs/predictions
```

### Physics Diagnostics

```bash
python scripts/physics_diagnostics.py \
    --predictions-dir outputs/predictions \
    --output-dir outputs/diagnostics/physics
```

### Conformal Uncertainty Quantification

```bash
# Compute conformal prediction intervals
python scripts/compute_conformal.py \
    --predictions-dir outputs/predictions \
    --split-mode test_graph_50_50 \
    --alpha 0.10 0.05 \
    --output-dir outputs/diagnostics/conformal

# Analyze coverage and interval width
python scripts/analyze_conformal.py \
    --conformal-dir outputs/diagnostics/conformal/<RUN_DIR> \
    --output-dir outputs/diagnostics/conformal/<RUN_DIR>
```

---

## Output Structure

Training and diagnostic outputs are organized under `outputs/`:

```
outputs/
├── baselines/              # Training checkpoints and logs
│   ├── MLP/
│   ├── HGT/
│   ├── MS_HGT/
│   └── MS_PI_HGT/
├── predictions/            # Full test set prediction exports (.npz)
└── diagnostics/            # Physics and conformal analysis results
```

Each training run produces:

- `config_resolved.yaml` — Resolved configuration snapshot
- `model_summary.json` — Model architecture summary
- `best_model.pt` — Best validation checkpoint
- `train_log.csv` — Per-epoch training log
- `metrics_summary.json` — Aggregated evaluation metrics
- `loss_curve.png` — Training/validation loss plot
- `metric_curve.png` — Metric evolution plot

---

## Dataset Split

- **Split mode:** `by_sample` (default) — 80/10/10 train/val/test split over the 70 design samples
- **Alternative:** `by_loadcase` — split over load cases within each sample
- **Standardization:** Computed from training set only; val/test use training-set statistics
- **Random seed:** 42 (configurable)

---

## Reproducibility Notes

1. **Experiments are tracked by git commit + config.** Each training job records its commit hash, resolved config, and dataset version.
2. **Smoke test first, then full training.** All modifications should be tested locally with a small subset (2 graphs, 2 epochs) before full training.
3. **Artifacts are archived.** Server training outputs are packaged and downloaded to `remote_artifacts/` for local analysis.
4. **Files not tracked in git:** `processed/`, `outputs/`, `remote_artifacts/`, `logs/`, and model checkpoints (`*.pt`, `*.pth`, `*.ckpt`).
5. **All comparisons are internal.** Baseline models (MLP, GCN, GAT, RGCN, HGT) are re-implemented within the same pipeline for fair comparison.

### Limitations

1. **Shared topology (by-sample split).** All 70 samples share the same mesh topology. Current results demonstrate generalization to unseen design parameters, not cross-topology generalization.
2. **Physics losses are not complete equilibrium.** The BC and link consistency losses penalize specific violation types but do not enforce full element-level or system-level equilibrium.
3. **Plate element outputs are not predicted** — plate force/stress labels are unavailable in the current dataset.

---

## License

This project is for research purposes. See LICENSE file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Inspired by Graph Neural Networks in Computational Mechanics and related works

#!/usr/bin/env python3
"""
train_baseline.py — Unified training entry point for Stage 2 baseline suite.

Supports:

  - ``mlp``          — MLPBaseline (non-graph)
  - ``gcn``          — HomogeneousGCN
  - ``gat``          — HomogeneousGAT
  - ``rgcn``         — HeteroRGCNBaseline (relation-specific typed message passing)
  - ``hgt``          — HGTBaseline (typed attention, HGTConv)

All models use ``processed/hetero_graph_dataset_v2`` and the
``HeteroFeatureScaler`` for standardisation.

Usage::

    # Train MLP with default configs
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe train_baseline.py --model mlp

    # Train GCN with overrides
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe train_baseline.py --model gcn ^
        --batch-size 4 --epochs 50 --split-mode by_loadcase

    # Smoke test (2 epochs only, 10 graphs)
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe train_baseline.py --model mlp ^
        --max-graphs 10 --epochs 2

    # Generate stage 2-A summary from existing experiment dirs
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe train_baseline.py --summarise-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.hetero_graph_dataset import HeteroGraphDataset
from src.data.hetero_transforms import HeteroFeatureScaler
from src.models.baselines import (
    MLPBaseline,
    HomogeneousGCN,
    HomogeneousGAT,
    HeteroRGCNBaseline,
    HGTBaseline,
)
from src.trainers.baseline_trainer import BaselineTrainer
from src.utils.experiment import create_experiment_dir, save_resolved_config
from src.utils.metrics import compute_all_metrics


# ============================================================
# Constants
# ============================================================

DISP_COMP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_COMP_NAMES = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]

MODEL_NAMES_MAP = {
    "mlp": "MLPBaseline",
    "gcn": "HomogeneousGCN",
    "gat": "HomogeneousGAT",
    "rgcn": "HeteroRGCNBaseline",
    "hgt": "HGTBaseline",
}

MODEL_CONFIG_KEYS = {
    "mlp": "mlp_baseline",
    "gcn": "homogeneous_gcn",
    "gat": "homogeneous_gat",
    "rgcn": "hetero_rgcn",
    "hgt": "hgt",
}


# ============================================================
# Helpers
# ============================================================

def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ============================================================
# Model factory
# ============================================================

def build_model(model_name: str, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Instantiate a baseline model by short name."""
    if model_name == "mlp":
        model = MLPBaseline(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
        )
    elif model_name == "gcn":
        model = HomogeneousGCN(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.2),
            activation=model_cfg.get("activation", "relu"),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
            use_type_embed=model_cfg.get("use_type_embed", True),
        )
    elif model_name == "gat":
        model = HomogeneousGAT(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            gat_heads=model_cfg.get("gat_heads", 4),
            dropout=model_cfg.get("dropout", 0.2),
            activation=model_cfg.get("activation", "relu"),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
            use_type_embed=model_cfg.get("use_type_embed", True),
        )
    elif model_name == "rgcn":
        model = HeteroRGCNBaseline(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
        )
    elif model_name == "hgt":
        model = HGTBaseline(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            heads=model_cfg.get("heads", 4),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_NAMES_MAP.keys())}")

    return model.to(device)


# ============================================================
# Config loading
# ============================================================

def load_baseline_configs(config_dir: str, model_name: str) -> Dict:
    """Load train_baseline.yaml + models_baseline.yaml and merge."""
    config_dir = Path(config_dir)
    train_cfg = _load_yaml(config_dir / "train_baseline.yaml")
    models_cfg = _load_yaml(config_dir / "models_baseline.yaml")

    # Merge: train_cfg (base)
    merged: Dict = {}
    merged.update(train_cfg)

    # Inject model-specific config
    cfg_key = MODEL_CONFIG_KEYS[model_name]
    model_cfg = models_cfg.get(cfg_key, {})
    merged["model_config"] = model_cfg
    merged["model_name"] = model_name

    return merged


# ============================================================
# Metrics extraction
# ============================================================

def extract_disp_metrics(metrics_dict: Dict) -> Dict:
    """Flatten displacement metrics for JSON summary."""
    return {
        "overall_mse": round(metrics_dict["overall"]["mse"], 6),
        "overall_mae": round(metrics_dict["overall"]["mae"], 6),
        "overall_rel_mae": round(metrics_dict["overall"]["rel_mae"], 6),
        "overall_r2": round(metrics_dict["overall"]["r2"], 6),
        "macro_avg_mae": round(metrics_dict["macro_avg"]["mae"], 6),
        "macro_avg_rel_mae": round(metrics_dict["macro_avg"]["rel_mae"], 6),
        "macro_avg_r2": round(metrics_dict["macro_avg"]["r2"], 6),
        "per_component_mae": [round(v, 6) for v in metrics_dict["per_component"]["mae"]],
        "per_component_r2": [round(v, 6) for v in metrics_dict["per_component"]["r2"]],
    }


def extract_force_metrics(metrics_dict: Dict) -> Dict:
    """Flatten force metrics for JSON summary."""
    return {
        "overall_mse": round(metrics_dict["overall"]["mse"], 6),
        "overall_mae": round(metrics_dict["overall"]["mae"], 6),
        "overall_rel_mae": round(metrics_dict["overall"]["rel_mae"], 6),
        "overall_r2": round(metrics_dict["overall"]["r2"], 6),
        "macro_avg_mae": round(metrics_dict["macro_avg"]["mae"], 6),
        "macro_avg_rel_mae": round(metrics_dict["macro_avg"]["rel_mae"], 6),
        "macro_avg_r2": round(metrics_dict["macro_avg"]["r2"], 6),
        "per_component_mae": [round(v, 6) for v in metrics_dict["per_component"]["mae"]],
        "per_component_r2": [round(v, 6) for v in metrics_dict["per_component"]["r2"]],
    }


# ============================================================
# Sample predictions CSV
# ============================================================

def save_sample_predictions(
    test_loader,
    model,
    device: torch.device,
    save_path: Path,
    trainer: BaselineTrainer,
    max_samples: int = 5000,
) -> None:
    """Save a sample of test-set predictions vs targets (original scale) as CSV."""
    import pandas as pd

    model.eval()
    all_disp_pred: List[torch.Tensor] = []
    all_disp_target: List[torch.Tensor] = []
    all_force_pred: List[torch.Tensor] = []
    all_force_target: List[torch.Tensor] = []
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_disp, pred_force = model(batch)
            all_disp_pred.append(pred_disp.cpu())
            all_disp_target.append(batch["mesh_node"].y_disp.cpu())
            all_force_pred.append(pred_force.cpu())
            all_force_target.append(batch["beam_element"].y_force.cpu())
            count += pred_disp.shape[0]
            if count >= max_samples:
                break

    disp_pred = torch.cat(all_disp_pred, dim=0)[:max_samples]
    disp_target = torch.cat(all_disp_target, dim=0)[:max_samples]
    force_pred = torch.cat(all_force_pred, dim=0)[:max_samples]
    force_target = torch.cat(all_force_target, dim=0)[:max_samples]

    # Inverse transform
    disp_pred_orig, disp_target_orig = trainer._inverse_transform_disp(disp_pred, disp_target)
    force_pred_orig, force_target_orig = trainer._inverse_transform_force(force_pred, force_target)

    rows = []
    n = min(max_samples, disp_pred.shape[0])
    for i in range(n):
        row = {}
        for j, name in enumerate(DISP_COMP_NAMES):
            row[f"disp_pred_{name}"] = disp_pred_orig[i, j].item()
            row[f"disp_true_{name}"] = disp_target_orig[i, j].item()
        rows.append(row)

    nf = min(max_samples, force_pred.shape[0])
    for i in range(nf):
        for j, name in enumerate(FORCE_COMP_NAMES):
            rows[i][f"force_pred_{name}"] = force_pred_orig[i, j].item()
            rows[i][f"force_true_{name}"] = force_target_orig[i, j].item()

    df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  [ok] Sample predictions ({len(df)} rows): {save_path}")


# ============================================================
# Stage 2-A summary generation
# ============================================================

def generate_stage2a_summary(baseline_output_root: Path) -> None:
    """Read all experiment directories and produce summary CSV + MD.

    Scans ``baseline_output_root / <ModelName> / <timestamp> / metrics_summary.json``
    and aggregates results.
    """
    print(f"\n{'=' * 60}")
    print("Generating Stage 2-A Baseline Summary")
    print(f"{'=' * 60}")

    records = []
    model_dirs = sorted([d for d in baseline_output_root.iterdir() if d.is_dir()])

    for model_dir in model_dirs:
        model_name = model_dir.name
        exp_dirs = sorted(model_dir.iterdir())
        if not exp_dirs:
            continue
        latest_exp = exp_dirs[-1]  # latest timestamp
        metrics_path = latest_exp / "metrics_summary.json"
        if not metrics_path.is_file():
            print(f"  [warn] No metrics_summary.json in {latest_exp}")
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            ms = json.load(f)

        record = {
            "Model": MODEL_NAMES_MAP.get(model_name, model_name),
            "Graph Type": "none" if model_name == "mlp" else "homogeneous",
            "Typed Message": "no",
            "Multi-scale": "no",
            "Physics Loss": "no",
            "Params": ms.get("num_params", 0),
            "Best Epoch": ms.get("training", {}).get("best_epoch", "?"),
            "Total Time (min)": round(ms.get("training", {}).get("total_time_seconds", 0) / 60, 1),
        }

        # Disp metrics
        test = ms.get("test", {})
        disp = test.get("disp", {})
        force = test.get("force", {})
        combined_rel_mae = test.get("combined_rel_mae", None)

        record["Disp R2 (macro)"] = disp.get("macro_avg_r2", "?")
        record["Disp MAE (macro)"] = disp.get("macro_avg_mae", "?")
        record["Force R2 (macro)"] = force.get("macro_avg_r2", "?")
        record["Force MAE (macro)"] = force.get("macro_avg_mae", "?")
        record["RelMAE"] = combined_rel_mae if combined_rel_mae else "?"

        records.append(record)
        print(f"  {model_name}: R2_disp={record['Disp R2 (macro)']}, R2_force={record['Force R2 (macro)']}")

    if not records:
        print("  No experiment results found.")
        return

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = baseline_output_root / "stage2a_baseline_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [ok] Summary CSV: {csv_path}")

    # Save MD
    md_lines = [
        "# Stage 2-A Baseline Summary\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Dataset: `processed/hetero_graph_dataset_v2`\n",
        "",
        "| Method | Graph Type | Typed Message | Multi-scale | Physics Loss | Disp R2 | Force R2 | RelMAE | Params | Time (min) |",
        "|--------|------------|---------------|-------------|--------------|--------:|---------:|-------:|-------:|-----------:|",
    ]
    for r in records:
        md_lines.append(
            f"| {r['Model']} | {r['Graph Type']} | {r['Typed Message']} | "
            f"{r['Multi-scale']} | {r['Physics Loss']} | "
            f"{r['Disp R2 (macro)']} | {r['Force R2 (macro)']} | "
            f"{r['RelMAE']} | {r['Params']:,} | {r['Total Time (min)']} |"
        )

    md_path = baseline_output_root / "stage2a_baseline_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  [ok] Summary MD: {md_path}")


# ============================================================
# Main
# ============================================================

def main(args: argparse.Namespace):
    t_all = time.time()

    # ---- 0. Resolve model name ----
    model_name = args.model
    if model_name not in MODEL_NAMES_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_NAMES_MAP.keys())}")

    # ---- 1. Load configs ----
    print("=" * 60)
    print(f"Multi-Scale PI-HGNN — Stage 2-A Baseline Training")
    print(f"Model: {model_name.upper()} ({MODEL_NAMES_MAP[model_name]})")
    print("=" * 60)

    config = load_baseline_configs(args.config_dir, model_name)

    # Command-line overrides
    split_mode = args.split_mode or config.get("data", {}).get("split_mode", "by_sample")
    batch_size = args.batch_size or config.get("data", {}).get("batch_size", 8)
    num_workers = args.num_workers or config.get("data", {}).get("num_workers", 0)
    max_graphs = args.max_graphs
    epochs_override = args.epochs
    device = infer_device(args.device or config.get("train", {}).get("device", "auto"))

    # Update config with overrides
    config.setdefault("data", {})["split_mode"] = split_mode
    config.setdefault("data", {})["batch_size"] = batch_size
    config.setdefault("data", {})["num_workers"] = num_workers
    if epochs_override:
        config.setdefault("train", {})["epochs"] = epochs_override
    config.setdefault("train", {})["device"] = str(device)
    if args.run_name:
        config["run_name"] = args.run_name

    print(f"\n  Split mode:    {split_mode}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Max graphs:    {max_graphs or 'all'}")
    print(f"  Device:        {device}")

    # ---- 2. Seed ----
    seed = config.get("train", {}).get("seed", 42)
    set_seed(seed)

    # ---- 3. Build datasets ----
    processed_dir = Path(config.get("data", {}).get("processed_dir", "processed/hetero_graph_dataset_v2"))
    if not processed_dir.is_absolute():
        processed_dir = _PROJECT_ROOT / processed_dir

    print(f"\n[1/6] Building datasets from {processed_dir} ...")

    # Load scaler
    stats_path = processed_dir / "feature_stats.json"
    if stats_path.is_file():
        scaler = HeteroFeatureScaler.load(stats_path)
        print(f"  Scaler loaded: {stats_path}")
    else:
        print(f"  [warn] feature_stats.json not found — training without standardisation.")
        scaler = None

    train_dataset = HeteroGraphDataset(
        processed_dir=processed_dir,
        split="train",
        split_mode=split_mode,
        transform=scaler,
    )
    val_dataset = HeteroGraphDataset(
        processed_dir=processed_dir,
        split="val",
        split_mode=split_mode,
        transform=scaler,
    )
    test_dataset = HeteroGraphDataset(
        processed_dir=processed_dir,
        split="test",
        split_mode=split_mode,
        transform=scaler,
    )

    # Limit graphs for smoke test (use Subset for Dataset objects)
    if max_graphs is not None and max_graphs > 0:
        n_train = min(max_graphs, len(train_dataset))
        n_val = min(max_graphs, len(val_dataset))
        n_test = min(max_graphs, len(test_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(val_dataset, range(n_val))
        test_dataset = torch.utils.data.Subset(test_dataset, range(n_test))

    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # ---- 4. Build model ----
    print(f"\n[2/6] Building model: {model_name} ({MODEL_NAMES_MAP[model_name]}) ...")
    model_cfg = config.get("model_config", {})
    model = build_model(model_name, model_cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {num_params:,} total, {num_trainable:,} trainable")
    print(f"  Config: {model_cfg}")

    # ---- 5. Create experiment directory ----
    print(f"\n[3/6] Creating experiment directory ...")
    output_root = Path(config.get("output", {}).get("output_root", "outputs/baselines"))
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root
    exp_dir = create_experiment_dir(
        output_root=output_root,
        model_name=model_name.upper(),
    )
    print(f"  Output: {exp_dir}")

    # Save resolved config
    save_resolved_config(config, exp_dir / "config_resolved.yaml")

    # Save model summary
    model_summary = {
        "model_name": model_name,
        "model_class": MODEL_NAMES_MAP[model_name],
        "total_params": num_params,
        "trainable_params": num_trainable,
        "model_config": model_cfg,
        "input_construction": _get_input_construction_desc(model_name, model_cfg),
    }
    with open(exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2, ensure_ascii=False)

    # ---- 6. Extract inverse-transform statistics ----
    disp_mean = disp_std = force_mean = force_std = None
    if scaler is not None and scaler.fitted:
        disp_mean = scaler._stats.get("node:mesh_node:y_disp_mean")
        disp_std = scaler._stats.get("node:mesh_node:y_disp_std")
        force_mean = scaler._stats.get("node:beam_element:y_force_mean")
        force_std = scaler._stats.get("node:beam_element:y_force_std")

    # ---- 7. Train ----
    print(f"\n[4/6] Training ...")
    train_config = config.get("train", {}).copy()
    # Merge cross-section keys into train_config so the trainer can see them
    train_config["progress"] = config.get("progress", {})
    train_config["model_name"] = model_name
    train_config["split_mode"] = split_mode
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        experiment_dir=exp_dir,
        disp_mean=disp_mean,
        disp_std=disp_std,
        force_mean=force_mean,
        force_std=force_std,
    )
    train_summary = trainer.fit()

    # ---- 8. Test evaluation ----
    print(f"\n[5/6] Test evaluation ...")
    test_results = trainer.test(test_loader)

    # ---- 9. Save metrics summary & sample predictions ----
    print(f"\n[6/6] Saving outputs ...")

    # Compute combined RelMAE (weighted average of disp and force RelMAE)
    disp_rel_mae = test_results["disp_metrics"]["overall"]["rel_mae"]
    force_rel_mae = test_results["force_metrics"]["overall"]["rel_mae"]
    combined_rel_mae = (disp_rel_mae + force_rel_mae) / 2.0

    metrics_summary = {
        "config": {
            "model_name": model_name,
            "model_class": MODEL_NAMES_MAP[model_name],
            "split_mode": split_mode,
            "batch_size": batch_size,
            "seed": seed,
            "max_graphs": max_graphs,
        },
        "num_params": num_params,
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "training": {
            "best_epoch": train_summary["best_epoch"],
            "stopped_epoch": train_summary["stopped_epoch"],
            "early_stopped": train_summary["early_stopped"],
            "best_val_metric": train_summary["best_val_metric"],
            "best_val_disp_mae": train_summary.get("best_val_disp_mae"),
            "best_val_force_mae": train_summary.get("best_val_force_mae"),
            "best_val_disp_r2": train_summary.get("best_val_disp_r2"),
            "best_val_force_r2": train_summary.get("best_val_force_r2"),
            "total_time_seconds": round(train_summary["total_time_seconds"], 1),
        },
        "test": {
            "loss_standardised": round(test_results["test_loss"], 6),
            "combined_rel_mae": round(combined_rel_mae, 6),
            "disp": extract_disp_metrics(test_results["disp_metrics"]),
            "force": extract_force_metrics(test_results["force_metrics"]),
        },
        "standardisation": {
            "source": str(stats_path) if scaler is not None else "none",
            "train_only": scaler is not None and scaler.fitted,
        },
    }

    metrics_path = exp_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"  [ok] Metrics summary: {metrics_path}")

    # Sample predictions
    save_sample_predictions(
        test_loader, model, device,
        exp_dir / "test_predictions_sample.csv",
        trainer,
        max_samples=config.get("eval", {}).get("prediction_sample_size", 5000),
    )

    total_time = time.time() - t_all

    # ---- 10. Print summary ----
    print(f"\n{'=' * 60}")
    print("Training Complete — Summary")
    print(f"{'=' * 60}")
    print(f"  Model:            {model_name.upper()} ({MODEL_NAMES_MAP[model_name]})")
    print(f"  Params:           {num_params:,}")
    print(f"  Output:           {exp_dir}")
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Best epoch:       {train_summary['best_epoch']}")
    print(f"  Early stopped:    {train_summary['early_stopped']}")
    print(f"  Test Disp R2:     {test_results['disp_metrics']['macro_avg']['r2']:.4f}")
    print(f"  Test Force R2:    {test_results['force_metrics']['macro_avg']['r2']:.4f}")
    print(f"  Test Disp MAE:    {test_results['disp_metrics']['macro_avg']['mae']:.6f}")
    print(f"  Test Force MAE:   {test_results['force_metrics']['macro_avg']['mae']:.6f}")
    print(f"  Combined RelMAE:  {combined_rel_mae:.6f}")
    print(f"{'=' * 60}")

    # ---- 11. Generate summary ----
    generate_stage2a_summary(output_root)

    return metrics_summary


def _get_input_construction_desc(model_name: str, model_cfg: Dict) -> str:
    """Describe how inputs are constructed for each baseline."""
    if model_name == "mlp":
        return (
            "MLPBaseline: Displacement head uses mesh_node.x (15-dim) directly. "
            "Force head uses beam_element.x (11-dim) concatenated with mean endpoint "
            "mesh_node features (15-dim) obtained via belongs_to_beam edge_index. "
            "plate_element and structural_link do not participate."
        )
    elif model_name == "gcn":
        return (
            "HomogeneousGCN: HeteroToHomoAdapter projects mesh_node.x (15-dim), "
            "beam_element.x (11-dim), plate_element.x (6-dim) to shared hidden_dim "
            "with type-specific linear projections and learnable type embeddings. "
            "All 5 edge types merged into homogeneous edge_index with per-type offsets. "
            "Standard GCNConv layers (no relation-type awareness). "
            "Mask back to per-type for decoders."
        )
    elif model_name == "gat":
        return (
            "HomogeneousGAT: Same heterogeneous-to-homogeneous conversion as GCN, "
            "but uses GATConv (multi-head attention) instead of GCNConv. "
            "Attention is standard node-level attention, not relation-type specific."
        )
    elif model_name == "rgcn":
        return (
            "HeteroRGCNBaseline: Type-specific Linear projections per node type "
            "(mesh_node 15-dim, beam_element 11-dim, plate_element 6-dim) to shared "
            "hidden_dim. Relation-specific message passing via HeteroConv + SAGEConv "
            "on all 5 edge types independently. Per-node-type LayerNorm. "
            "MLPHead decoders for disp (6) and force (12). "
            "No edge_attr used — this is a standard typed-relation baseline."
        )
    elif model_name == "hgt":
        return (
            "HGTBaseline: Type-specific Linear projections per node type "
            "(mesh_node 15-dim, beam_element 11-dim, plate_element 6-dim) to shared "
            "hidden_dim. Typed attention via HGTConv (4 heads) on all 5 edge types. "
            "Per-node-type LayerNorm. MLPHead decoders for disp (6) and force (12). "
            "No edge_attr used — this is a standard typed-attention baseline."
        )
    return ""


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 Baseline Training (MLP / GCN / GAT / RGCN)")
    p.add_argument("--model", type=str, required=True,
                   choices=list(MODEL_NAMES_MAP.keys()),
                   help=f"Model to train: {list(MODEL_NAMES_MAP.keys())}")
    p.add_argument("--config-dir", type=str, default="configs",
                   help="Config directory (default: configs)")
    p.add_argument("--split-mode", type=str, default=None,
                   choices=["by_sample", "by_loadcase"],
                   help="Split mode override")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size override")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Number of DataLoader workers (default: 0)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Max epochs override")
    p.add_argument("--max-graphs", type=int, default=None,
                   help="Limit total graphs (for smoke test)")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (auto | cuda | cpu)")
    p.add_argument("--run-name", type=str, default=None,
                   help="Descriptive run name (saved in config, no functional effect)")
    p.add_argument("--summarise-only", action="store_true",
                   help="Only generate summary from existing experiments, no training")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.summarise_only:
        output_root = Path("outputs/baselines")
        if not output_root.is_absolute():
            output_root = _PROJECT_ROOT / output_root
        generate_stage2a_summary(output_root)
    else:
        main(args)

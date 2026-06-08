#!/usr/bin/env python3
"""
train.py — Unified training entry point for Multi-Scale-PI-HGNN.

Usage:
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe train.py

    (config_dir defaults to ``configs/``; override via first CLI arg)

Workflow:
    1. Load configs/train.yaml + configs/models.yaml + configs/dataset.yaml
    2. Set deterministic seed
    3. Build train / val / test datasets + DataLoaders
    4. Load standardisation statistics (train-only) and attach scaler
    5. Instantiate model (model_name from train.yaml)
    6. Create timestamped experiment output directory
    7. Train with early stopping, checkpointing, and CSV logging
    8. Evaluate on held-out test set
    9. Save config copy, metrics summary, sample predictions
    10. Print structured summary

Extending:
    - Add a new model -> create ``src/models/<model>.py``, register in ``_build_model()``.
    - Add params -> add to the appropriate YAML file.
    No changes needed to this entry point's logic.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.graph_dataset import GraphDataset
from src.data.transforms import NodeEdgeStandardScaler
from src.models.mlp_edge_regressor import MLPEdgeRegressor
from src.trainers.trainer import Trainer
from src.utils.experiment import (
    create_experiment_dir,
    save_resolved_config,
)
from src.utils.metrics import compute_all_metrics

# ============================================================
# Config loading
# ============================================================

def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: str = "configs") -> Dict:
    """Load and merge all config files into one flat dict.

    Returns:
        ``config`` dict with keys from train.yaml, dataset.yaml, and
        the active model's section from models.yaml.
    """
    config_dir = Path(config_dir)

    train_cfg = _load_yaml(config_dir / "train.yaml")
    dataset_cfg = _load_yaml(config_dir / "dataset.yaml")
    models_cfg = _load_yaml(config_dir / "models.yaml")

    # Merge: train_cfg (base) <- dataset_cfg <- model-specific
    merged: Dict = {}
    merged.update(train_cfg)
    merged.update(dataset_cfg)

    # Inject model-specific config
    model_name = merged.get("model_name", "mlp")
    model_cfg = models_cfg.get(model_name, {})
    merged["model_config"] = model_cfg

    return merged


# ============================================================
# Seed
# ============================================================

def set_seed(seed: int) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Model factory
# ============================================================

def _build_model(model_name: str, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Instantiate a model by name.

    Extend this dict when adding new models.
    """
    if model_name == "mlp":
        model = MLPEdgeRegressor(
            input_dim=model_cfg.get("input_dim", 22),
            output_dim=model_cfg.get("output_dim", 12),
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
        )
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: ['mlp']"
        )
    return model.to(device)


# ============================================================
# Feature stats verification
# ============================================================

def _verify_feature_stats(scaler: NodeEdgeStandardScaler) -> None:
    """Quick sanity check on loaded feature stats."""
    print("\n  [verify] Feature stats loaded:")
    print(f"    node_mean[:3]: {scaler.node_mean[:3].tolist()}")
    print(f"    node_std[:3]:  {scaler.node_std[:3].tolist()}")
    print(f"    edge_mean:     {scaler.edge_mean.tolist()}")
    print(f"    edge_std:      {scaler.edge_std.tolist()}")
    print(f"    target_mean[:3]: {scaler.target_mean[:3].tolist()}")
    print(f"    target_std[:3]:  {scaler.target_std[:3].tolist()}")


# ============================================================
# Main
# ============================================================

def main(config_dir: str = "configs"):
    t_all = time.time()

    # ---- 1. Load configs ----
    print("=" * 60)
    print("Multi-Scale-PI-HGNN — Unified Training Entry")
    print("=" * 60)
    config = load_configs(config_dir)
    model_name = config["model_name"]
    split_mode = config.get("split_mode", "by_sample")
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print(f"  Model:         {model_name}")
    print(f"  Split mode:    {split_mode}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Device:        {device}")

    # ---- 2. Seed ----
    set_seed(config.get("seed", 42))

    # ---- 3. Build datasets ----
    processed_dir = Path(config["processed_data_dir"])
    if not processed_dir.is_absolute():
        processed_dir = _PROJECT_ROOT / processed_dir

    print(f"\n[1/6] Building datasets ...")

    # Load scaler (train-only stats) BEFORE creating Dataset
    stats_path = processed_dir / "feature_stats.json"
    if stats_path.is_file():
        scaler = NodeEdgeStandardScaler.load(stats_path)
        _verify_feature_stats(scaler)
        train_transform = scaler
    else:
        print(f"  [warn] feature_stats.json not found at {stats_path}")
        print(f"  [warn] Training without standardisation.")
        scaler = None
        train_transform = None

    train_dataset = GraphDataset(
        processed_dir=processed_dir,
        split="train",
        split_mode=split_mode,
        transform=train_transform,
    )
    val_dataset = GraphDataset(
        processed_dir=processed_dir,
        split="val",
        split_mode=split_mode,
        transform=train_transform,
    )
    test_dataset = GraphDataset(
        processed_dir=processed_dir,
        split="test",
        split_mode=split_mode,
        transform=train_transform,
    )

    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")

    # DataLoaders (use PyG's collate-aware loader)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- 4. Build model ----
    print(f"\n[2/6] Building model: {model_name} ...")
    model_cfg = config.get("model_config", {})
    model = _build_model(model_name, model_cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Params: {num_params:,} total, {num_trainable:,} trainable")
    print(f"  Config: {model_cfg}")

    # ---- 5. Create experiment directory ----
    print(f"\n[3/6] Creating experiment directory ...")
    exp_dir = create_experiment_dir(
        output_root=config.get("output_root", "outputs"),
        model_name=model_name.upper(),
    )
    print(f"  Output: {exp_dir}")

    # Save resolved config
    save_resolved_config(config, exp_dir / "config_resolved.yaml")

    # Save model summary
    model_summary = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "total_params": num_params,
        "trainable_params": num_trainable,
        "model_config": model_cfg,
    }
    with open(exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2, ensure_ascii=False)

    # ---- 6. Train ----
    print(f"\n[4/6] Training ...")
    target_mean = scaler.target_mean if scaler is not None else None
    target_std = scaler.target_std if scaler is not None else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=exp_dir,
        target_mean=target_mean,
        target_std=target_std,
    )
    train_summary = trainer.fit()

    # ---- 7. Test evaluation ----
    print(f"\n[5/6] Test evaluation ...")
    test_results = trainer.test(test_loader)

    # ---- 8. Save metrics summary & sample predictions ----
    print(f"\n[6/6] Saving outputs ...")

    # Metrics summary
    metrics_summary = {
        "config": {
            "model_name": model_name,
            "split_mode": split_mode,
            "batch_size": batch_size,
            "seed": config.get("seed"),
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
            "total_time_seconds": round(train_summary["total_time_seconds"], 1),
        },
        "test": {
            "loss_standardised": round(test_results["test_loss"], 6),
            "macro_avg_mae": round(test_results["test_metrics"]["macro_avg"]["mae"], 4),
            "macro_avg_rel_mae": round(test_results["test_metrics"]["macro_avg"]["rel_mae"], 4),
            "macro_avg_r2": round(test_results["test_metrics"]["macro_avg"]["r2"], 4),
            "overall_mse": round(test_results["test_metrics"]["overall"]["mse"], 4),
            "overall_mae": round(test_results["test_metrics"]["overall"]["mae"], 4),
            "overall_r2": round(test_results["test_metrics"]["overall"]["r2"], 4),
            "per_component_mae": [round(v, 4) for v in test_results["test_metrics"]["per_component"]["mae"]],
            "per_component_r2": [round(v, 4) for v in test_results["test_metrics"]["per_component"]["r2"]],
        },
        "standardisation": {
            "source": str(stats_path) if scaler is not None else "none",
            "train_only": scaler is not None,
        },
    }

    metrics_path = exp_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"  [ok] Metrics summary: {metrics_path}")

    # Sample predictions (first batch of test set)
    sample_preds_path = exp_dir / "test_predictions_sample.csv"
    _save_sample_predictions(test_loader, model, device, sample_preds_path, target_mean, target_std)

    total_time = time.time() - t_all

    # ---- 9. Print summary ----
    print(f"\n{'='*60}")
    print("Training Complete — Summary")
    print(f"{'='*60}")
    print(f"  Model:            {model_name.upper()}")
    print(f"  Params:           {num_params:,}")
    print(f"  Output:           {exp_dir}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best epoch:       {train_summary['best_epoch']}")
    print(f"  Early stopped:    {train_summary['early_stopped']}")
    print(f"  Best val metric:  {train_summary['best_val_metric']:.6f}")
    print(f"  Test MAE (macro): {test_results['test_metrics']['macro_avg']['mae']:.4f}")
    print(f"  Test R2  (macro): {test_results['test_metrics']['macro_avg']['r2']:.4f}")
    print(f"{'='*60}")

    return metrics_summary


def _save_sample_predictions(
    test_loader, model, device, save_path: Path,
    target_mean: Optional[torch.Tensor], target_std: Optional[torch.Tensor],
    max_samples: int = 5000,
) -> None:
    """Save a sample of test-set predictions vs targets (original scale) as CSV."""
    import pandas as pd

    model.eval()
    all_preds = []
    all_targets = []
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            all_preds.append(pred.cpu())
            all_targets.append(batch.y_edge.cpu())
            count += pred.shape[0]
            if count >= max_samples:
                break

    pred_all = torch.cat(all_preds, dim=0)[:max_samples]
    target_all = torch.cat(all_targets, dim=0)[:max_samples]

    # Convert to original scale
    if target_std is not None and target_mean is not None:
        pred_all = pred_all * target_std + target_mean
        target_all = target_all * target_std + target_mean

    comp_names = [
        "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
        "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
    ]
    rows = []
    for i in range(min(max_samples, pred_all.shape[0])):
        row = {}
        for j, name in enumerate(comp_names):
            row[f"pred_{name}"] = pred_all[i, j].item()
            row[f"true_{name}"] = target_all[i, j].item()
        rows.append(row)

    df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  [ok] Sample predictions ({len(df)} edges): {save_path}")


if __name__ == "__main__":
    # Optional CLI arg: config directory path
    cfg_dir = sys.argv[1] if len(sys.argv) > 1 else "configs"
    main(config_dir=cfg_dir)

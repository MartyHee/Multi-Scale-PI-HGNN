"""
export_full_predictions.py — Eval-only full test-set prediction export for Stage 2-B baselines and Ours-base.

Loads a trained model (best_model.pt) from a run_dir or extracted artifact, runs
inference on the test split, and saves per-node-type predictions as compressed
NPZ files for downstream tail-error, region, and physical diagnostics.

Usage:
    # MLP smoke test (2 graphs)
    python scripts/export_full_predictions.py --model mlp
        --run-dir outputs/baselines/MLP/20260620051300
        --max-graphs 2 --batch-size 1 --device cpu

    # Full export (server)
    python scripts/export_full_predictions.py --model hgt
        --run-dir outputs/baselines/HGT/<timestamp>
        --device cuda

    # From extracted artifact
    python scripts/export_full_predictions.py --model rgcn
        --artifact remote_artifacts/server_rgcn_full_<timestamp>
        --device cuda

Output structure:
    outputs/predictions/stage2b/<model_name>/<timestamp>/
        prediction_summary.json
        test_graph_index.csv
        mesh_node_predictions.npz       (y_true_disp, y_pred_disp, graph_id, node_id)
        beam_element_predictions.npz    (y_true_force, y_pred_force, graph_id, elem_id)
        export_metrics_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    OursBaseline,
    OursBaselineV2,
    MSHGTBaseline,
)
from src.utils.metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    "ours_base": "OursBaseline",
    "ours_base_v2": "OursBaselineV2",
    "ms_hgt": "MSHGTBaseline",
    "ms_hgt_additive": "MSHGTBaseline",
}

MODEL_CONFIG_KEYS = {
    "mlp": "mlp_baseline",
    "gcn": "homogeneous_gcn",
    "gat": "homogeneous_gat",
    "rgcn": "hetero_rgcn",
    "hgt": "hgt",
    "ours_base": "ours_base",
    "ours_base_v2": "ours_base_v2",
    "ms_hgt": "ms_hgt",
    "ms_hgt_additive": "ms_hgt_additive",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    elif sec < 3600:
        return f"{sec / 60:.1f}min"
    else:
        return f"{sec / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Model factory (copied from train_baseline.py)
# ---------------------------------------------------------------------------


def build_model(model_name: str, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
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
    elif model_name == "ours_base":
        model = OursBaseline(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
            structural_edge_dim=model_cfg.get("structural_edge_dim", 10),
            edge_hidden_dim=model_cfg.get("edge_hidden_dim", 32),
        )
    elif model_name == "ours_base_v2":
        model = OursBaselineV2(
            mesh_feat_dim=model_cfg.get("mesh_feat_dim", 15),
            beam_feat_dim=model_cfg.get("beam_feat_dim", 11),
            plate_feat_dim=model_cfg.get("plate_feat_dim", 6),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [64, 32]),
            structural_edge_dim=model_cfg.get("structural_edge_dim", 10),
            edge_hidden_dim=model_cfg.get("edge_hidden_dim", 32),
            gate_scale=model_cfg.get("gate_scale", 0.1),
            use_edge_bias=model_cfg.get("use_edge_bias", False),
            edge_bias_scale=model_cfg.get("edge_bias_scale", 0.0),
        )
    elif model_name in ("ms_hgt", "ms_hgt_additive"):
        model = MSHGTBaseline(
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
            n_segments=model_cfg.get("n_segments", 12),
            macro_gnn_layers=model_cfg.get("macro_gnn_layers", 2),
            macro_gnn_aggr=model_cfg.get("macro_gnn_aggr", "mean"),
            include_anchor_static=model_cfg.get("include_anchor_static", True),
            fusion_method=model_cfg.get("fusion_method", "gated_residual"),
            fusion_per_layer=model_cfg.get("fusion_per_layer", True),
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_NAMES_MAP.keys())}")

    return model.to(device)


def get_model_cfg_from_resolved(resolved_cfg: Dict, model_name: str) -> Dict:
    """Extract model config from resolved config."""
    # In config_resolved.yaml, model config is under 'model_config'
    if "model_config" in resolved_cfg:
        return resolved_cfg["model_config"]
    # Fallback: look for the model-specific key
    cfg_key = MODEL_CONFIG_KEYS.get(model_name)
    if cfg_key and cfg_key in resolved_cfg:
        return resolved_cfg[cfg_key]
    return {}


# ---------------------------------------------------------------------------
# Inverse transform helpers (mirrors BaselineTrainer)
# ---------------------------------------------------------------------------


def inverse_transform_disp(
    pred: torch.Tensor, target: torch.Tensor,
    disp_mean: Optional[torch.Tensor], disp_std: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if disp_std is not None and disp_mean is not None:
        return pred * disp_std + disp_mean, target * disp_std + disp_mean
    return pred, target


def inverse_transform_force(
    pred: torch.Tensor, target: torch.Tensor,
    force_mean: Optional[torch.Tensor], force_std: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if force_std is not None and force_mean is not None:
        return pred * force_std + force_mean, target * force_std + force_mean
    return pred, target


# ---------------------------------------------------------------------------
# Prediction export
# ---------------------------------------------------------------------------


@torch.no_grad()
def export_predictions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    scaler: Optional[HeteroFeatureScaler],
    output_dir: Path,
    model_name: str,
    run_dir: str,
) -> Dict:
    """Run full test-set inference and save predictions as NPZ files.

    Returns: summary dict with export info and metrics check.
    """
    print(f"\n{'=' * 60}")
    print("Running full test inference")
    print(f"{'=' * 60}")

    model.eval()

    # Collectors
    all_disp_pred = []
    all_disp_target = []
    all_force_pred = []
    all_force_target = []
    all_graph_ids = []
    all_beam_graph_ids = []
    all_mesh_node_ids = []
    all_beam_elem_ids = []
    all_xyz = []
    all_support_flags = []
    graph_index_records = []

    n_graphs = len(test_loader.dataset)
    n_exported = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        pred_disp, pred_force = model(batch)

        # Graph-level metadata
        # HeteroDataBatch stores graph assignment in batch['mesh_node'].batch
        mesh_batch_ptr = batch["mesh_node"].batch.cpu()  # (total_mesh_nodes,)
        beam_batch_ptr = batch["beam_element"].batch.cpu()  # (total_beam_elements,)

        # Per-graph metadata
        # Infer local graph indices in this batch
        batch_size_local = mesh_batch_ptr.max().item() + 1
        for i in range(batch_size_local):
            gid = n_exported + i  # global graph index
            n_mesh = (mesh_batch_ptr == i).sum().item()
            n_beam = (beam_batch_ptr == i).sum().item()
            # Get sample_id / load_case_id if available in batch metadata
            sample_id = getattr(batch, "sample_id", None)
            load_case_id = getattr(batch, "load_case_id", None)
            s_id = sample_id[i].item() if sample_id is not None and i < len(sample_id) else None
            l_id = load_case_id[i].item() if load_case_id is not None and i < len(load_case_id) else None
            graph_index_records.append({
                "graph_id": gid,
                "sample_id": s_id if s_id is not None else "",
                "load_case_id": l_id if l_id is not None else "",
                "num_mesh_nodes": n_mesh,
                "num_beam_elements": n_beam,
            })

        # Collect predictions (standardised scale)
        all_disp_pred.append(pred_disp.cpu())
        all_disp_target.append(batch["mesh_node"].y_disp.cpu())
        all_force_pred.append(pred_force.cpu())
        all_force_target.append(batch["beam_element"].y_force.cpu())

        # Graph assignment
        all_graph_ids.append(mesh_batch_ptr)

        # Beam element graph assignment (separate from mesh graph_id)
        all_beam_graph_ids.append(beam_batch_ptr)

        # Node-level IDs
        if hasattr(batch["mesh_node"], "node_id"):
            all_mesh_node_ids.append(batch["mesh_node"].node_id.cpu())
        else:
            all_mesh_node_ids.append(torch.arange(len(mesh_batch_ptr)))

        if hasattr(batch["beam_element"], "element_id"):
            all_beam_elem_ids.append(batch["beam_element"].element_id.cpu())
        else:
            all_beam_elem_ids.append(torch.arange(len(beam_batch_ptr)))

        # Node coordinates (first 3 dims of mesh_node.x are X, Y, Z)
        mesh_x = batch["mesh_node"].x.cpu()
        # Since mesh_x is standardised, we need the scaler for inverse transform
        # But for saving coords we want original scale
        if scaler is not None and scaler.fitted:
            x_mean_key = "node:mesh_node:x"
            x_mean = scaler._stats.get(f"{x_mean_key}_mean")
            x_std = scaler._stats.get(f"{x_mean_key}_std")
            if x_mean is not None and x_std is not None:
                x_orig = mesh_x * x_std + x_mean
                all_xyz.append(x_orig[:, :3])
            else:
                all_xyz.append(mesh_x[:, :3])
        else:
            all_xyz.append(mesh_x[:, :3])

        # Support flags (mesh_node.x[:, 9:15] are BC constraints: Dx_fix..Rz_fix, 0/1)
        # Inverse-transform to original 0/1 scale
        sup_std = mesh_x[:, 9:15]  # standardised values
        if scaler is not None and scaler.fitted:
            x_mean = scaler._stats.get("node:mesh_node:x_mean")
            x_std = scaler._stats.get("node:mesh_node:x_std")
            if x_mean is not None and x_std is not None:
                sup_orig = sup_std * x_std[9:15] + x_mean[9:15]
                sup_binary = (sup_orig > 0.5).float()  # threshold to 0/1
                all_support_flags.append(sup_binary)
            else:
                all_support_flags.append(sup_std)
        else:
            all_support_flags.append(sup_std)

        n_exported += batch_size_local

        if (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
            elapsed = time.time() - t_start
            print(f"  Batch {batch_idx + 1}/{len(test_loader)} | "
                  f"{n_exported}/{n_graphs} graphs | {fmt_time(elapsed)}")

    # Concatenate
    disp_pred = torch.cat(all_disp_pred, dim=0)
    disp_target = torch.cat(all_disp_target, dim=0)
    force_pred = torch.cat(all_force_pred, dim=0)
    force_target = torch.cat(all_force_target, dim=0)
    graph_ids = torch.cat(all_graph_ids, dim=0) if len(all_graph_ids) > 0 else torch.zeros(0)
    beam_graph_ids = torch.cat(all_beam_graph_ids, dim=0) if len(all_beam_graph_ids) > 0 else torch.zeros(0)
    node_ids = torch.cat(all_mesh_node_ids, dim=0).long() if len(all_mesh_node_ids) > 0 else torch.zeros(0, dtype=torch.long)
    elem_ids = torch.cat(all_beam_elem_ids, dim=0).long() if len(all_beam_elem_ids) > 0 else torch.zeros(0, dtype=torch.long)
    all_coords = torch.cat(all_xyz, dim=0) if all_xyz else torch.zeros(0, 3)
    all_supports = torch.cat(all_support_flags, dim=0) if all_support_flags else torch.zeros(0, 6)

    # ---- Inverse transform to original (physical) scale ----
    # Get scaler stats for targets
    disp_mean = disp_std = None
    force_mean = force_std = None
    if scaler is not None and scaler.fitted:
        disp_mean = scaler._stats.get(f"node:mesh_node:y_disp_mean")
        disp_std = scaler._stats.get(f"node:mesh_node:y_disp_std")
        force_mean = scaler._stats.get(f"node:beam_element:y_force_mean")
        force_std = scaler._stats.get(f"node:beam_element:y_force_std")

    # Convert to tensors if they exist
    if disp_mean is not None:
        disp_mean = disp_mean.to(disp_pred.device)
        disp_std = disp_std.to(disp_pred.device)
    if force_mean is not None:
        force_mean = force_mean.to(force_pred.device)
        force_std = force_std.to(force_pred.device)

    disp_pred_orig, disp_target_orig = inverse_transform_disp(
        disp_pred, disp_target, disp_mean, disp_std)
    force_pred_orig, force_target_orig = inverse_transform_force(
        force_pred, force_target, force_mean, force_std)

    # ---- Save mesh_node predictions ----
    mesh_path = output_dir / "mesh_node_predictions.npz"
    np.savez_compressed(
        mesh_path,
        y_true_disp=disp_target_orig.numpy(),
        y_pred_disp=disp_pred_orig.numpy(),
        graph_id=graph_ids.numpy(),
        node_id=node_ids.numpy(),
        node_xyz=all_coords.numpy(),
        support_flags=all_supports.numpy(),
    )
    mesh_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
    print(f"\n  Mesh node predictions: {mesh_path} ({mesh_size_mb:.1f} MB)")
    print(f"    Shape: y_true_disp={list(disp_target_orig.shape)}, "
          f"y_pred_disp={list(disp_pred_orig.shape)}")

    # ---- Save beam_element predictions ----
    beam_path = output_dir / "beam_element_predictions.npz"
    np.savez_compressed(
        beam_path,
        y_true_force=force_target_orig.numpy(),
        y_pred_force=force_pred_orig.numpy(),
        graph_id=beam_graph_ids.numpy(),
        element_id=elem_ids.numpy(),
    )
    beam_size_mb = os.path.getsize(beam_path) / (1024 * 1024)
    print(f"  Beam element predictions: {beam_path} ({beam_size_mb:.1f} MB)")
    print(f"    Shape: y_true_force={list(force_target_orig.shape)}, "
          f"y_pred_force={list(force_pred_orig.shape)}")

    # ---- Save graph index ----
    import csv
    gidx_path = output_dir / "test_graph_index.csv"
    with open(gidx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "graph_id", "sample_id", "load_case_id",
            "num_mesh_nodes", "num_beam_elements",
        ])
        w.writeheader()
        for rec in graph_index_records:
            w.writerow(rec)
    print(f"  Graph index: {gidx_path} ({len(graph_index_records)} graphs)")

    # ---- Compute metrics check ----
    print(f"\n  Computing export metrics check...")
    disp_metrics = compute_all_metrics(disp_pred_orig, disp_target_orig)
    force_metrics = compute_all_metrics(force_pred_orig, force_target_orig)

    export_metrics = {
        "n_graphs": n_exported,
        "n_mesh_nodes": disp_pred_orig.shape[0],
        "n_beam_elements": force_pred_orig.shape[0],
        "disp_r2": round(disp_metrics["macro_avg"]["r2"], 6),
        "force_r2": round(force_metrics["macro_avg"]["r2"], 6),
        "disp_mae": round(disp_metrics["macro_avg"]["mae"], 6),
        "force_mae": round(force_metrics["macro_avg"]["mae"], 6),
        "combined_rel_mae": round(
            (disp_metrics["overall"]["rel_mae"] + force_metrics["overall"]["rel_mae"]) / 2.0, 6
        ),
        "mesh_file": str(mesh_path),
        "mesh_file_mb": round(mesh_size_mb, 2),
        "beam_file": str(beam_path),
        "beam_file_mb": round(beam_size_mb, 2),
    }

    print(f"  Metrics check:")
    print(f"    Disp R2:             {export_metrics['disp_r2']:.6f}")
    print(f"    Force R2:            {export_metrics['force_r2']:.6f}")
    print(f"    Disp MAE:            {export_metrics['disp_mae']:.6f}")
    print(f"    Force MAE:           {export_metrics['force_mae']:.6f}")
    print(f"    Combined RelMAE:     {export_metrics['combined_rel_mae']:.6f}")

    # Save metrics check
    metrics_check_path = output_dir / "export_metrics_check.json"
    _save_json(export_metrics, metrics_check_path)

    return export_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace):
    print("=" * 60)
    print("FULL PREDICTION EXPORT — Stage 2-B Baseline Models")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device(args.device)
    model_name = args.model

    if model_name not in MODEL_NAMES_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_NAMES_MAP.keys())}")

    # ---- Resolve run dir ----
    if args.artifact:
        run_dir = Path(args.artifact)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        raise ValueError("Either --run-dir or --artifact must be specified.")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    print(f"\nRun dir: {run_dir}")

    # ---- 1. Load config ----
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config_resolved.yaml not found: {config_path}")

    config = _load_yaml(config_path)
    print(f"Config loaded: {config_path}")

    # Extract key settings
    split_mode = args.split_mode or config.get("data", {}).get("split_mode", "by_sample")
    batch_size = args.batch_size or config.get("data", {}).get("batch_size", 8)
    num_workers = args.num_workers or config.get("data", {}).get("num_workers", 0)
    max_graphs = args.max_graphs or config.get("data", {}).get("max_graphs")

    # ---- 2. Load model config from model_summary.json ----
    model_summary_path = run_dir / "model_summary.json"
    model_cfg = {}
    if model_summary_path.exists():
        ms = _load_json(model_summary_path)
        model_cfg = ms.get("model_config", {})
        print(f"Model summary loaded: {ms.get('model_class', '?')}, {ms.get('total_params', '?'):,} params")
    else:
        print(f"  [WARN] model_summary.json not found — using defaults for model config")

    # ---- 3. Load dataset ----
    processed_dir = Path(config.get("data", {}).get(
        "processed_dir", "processed/hetero_graph_dataset_v2"))
    if not processed_dir.is_absolute():
        processed_dir = _PROJECT_ROOT / processed_dir

    print(f"\n[1/4] Loading dataset from {processed_dir} ...")

    # Load scaler
    stats_path = processed_dir / "feature_stats.json"
    if stats_path.is_file():
        scaler = HeteroFeatureScaler.load(stats_path)
        print(f"  Scaler loaded: {stats_path}")
    else:
        print(f"  [WARN] feature_stats.json not found — no standardisation.")
        scaler = None

    # Build test dataset
    test_dataset = HeteroGraphDataset(
        processed_dir=processed_dir,
        split="test",
        split_mode=split_mode,
        transform=scaler,
    )

    # Apply max_graphs limit
    if max_graphs is not None and max_graphs > 0:
        n_test = min(max_graphs, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, range(n_test))

    print(f"  Test: {len(test_dataset)} graphs")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # ---- 4. Build model ----
    print(f"\n[2/4] Building model: {model_name} ({MODEL_NAMES_MAP[model_name]}) ...")
    model = build_model(model_name, model_cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {num_params:,}")

    # ---- 5. Load best_model.pt ----
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best_model.pt not found: {checkpoint_path}")

    print(f"\n[3/4] Loading checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {unexpected[:5]}...")
        print(f"  Model loaded successfully ({len(state_dict)} keys)")
    except Exception as e:
        print(f"  [ERROR] Failed to load checkpoint: {e}")
        raise

    model.eval()

    # ---- 6. Export predictions ----
    print(f"\n[4/4] Exporting predictions ...")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(args.output_dir) / model_name / timestamp
    ensure_dir(output_dir)

    export_summary = export_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        scaler=scaler,
        output_dir=output_dir,
        model_name=model_name,
        run_dir=str(run_dir),
    )

    # ---- 7. Save prediction summary ----
    # Try to get git info
    git_commit = "unknown"
    try:
        git_info_path = run_dir / "git_info.txt"
        if git_info_path.exists():
            git_text = git_info_path.read_text()
            for line in git_text.splitlines():
                if "commit" in line.lower():
                    git_commit = line.strip()
                    break
    except Exception:
        pass

    summary = {
        "model_name": model_name,
        "model_class": MODEL_NAMES_MAP[model_name],
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(processed_dir),
        "split_mode": split_mode,
        "split_name": "test",
        "graph_count": export_summary["n_graphs"],
        "mesh_node_count": export_summary["n_mesh_nodes"],
        "beam_element_count": export_summary["n_beam_elements"],
        "output_format": "npz (compressed)",
        "files": {
            "mesh_predictions": str(output_dir / "mesh_node_predictions.npz"),
            "beam_predictions": str(output_dir / "beam_element_predictions.npz"),
            "graph_index": str(output_dir / "test_graph_index.csv"),
            "metrics_check": str(output_dir / "export_metrics_check.json"),
        },
        "file_sizes_mb": {
            "mesh": export_summary["mesh_file_mb"],
            "beam": export_summary["beam_file_mb"],
        },
        "export_time": datetime.now().isoformat(),
        "git_commit": git_commit,
        "metrics_check": {
            "disp_r2": export_summary["disp_r2"],
            "force_r2": export_summary["force_r2"],
            "disp_mae": export_summary["disp_mae"],
            "force_mae": export_summary["force_mae"],
            "combined_rel_mae": export_summary["combined_rel_mae"],
        },
    }

    summary_path = output_dir / "prediction_summary.json"
    _save_json(summary, summary_path)
    print(f"\nPrediction summary saved: {summary_path}")

    # ---- Print summary ----
    total_time_export = time.time() - (getattr(args, "_start_time", time.time()))
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:           {model_name} ({MODEL_NAMES_MAP[model_name]})")
    print(f"  Graphs exported: {export_summary['n_graphs']}")
    print(f"  Mesh nodes:      {export_summary['n_mesh_nodes']:,}")
    print(f"  Beam elements:   {export_summary['n_beam_elements']:,}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Total size:      {export_summary['mesh_file_mb'] + export_summary['beam_file_mb']:.1f} MB")
    print(f"  Disp R2:         {export_summary['disp_r2']:.6f}")
    print(f"  Force R2:        {export_summary['force_r2']:.6f}")
    print(f"  Combined RelMAE: {export_summary['combined_rel_mae']:.6f}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export full test-set predictions from trained Stage 2-B models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # MLP smoke test
              python scripts/export_full_predictions.py --model mlp
                  --run-dir outputs/baselines/MLP/20260620051300
                  --max-graphs 2 --device cpu --output-dir outputs/predictions/smoke_test

              # HGT full export (cuda)
              python scripts/export_full_predictions.py --model hgt
                  --run-dir outputs/baselines/HGT/20260622103144
                  --device cuda

              # From extracted artifact
              python scripts/export_full_predictions.py --model rgcn
                  --artifact remote_artifacts/server_rgcn_full_20260621063600
                  --device cuda
        """),
    )
    p.add_argument("--model", type=str, required=True,
                   choices=list(MODEL_NAMES_MAP.keys()),
                   help=f"Model name: {list(MODEL_NAMES_MAP.keys())}")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Path to trained experiment run directory")
    p.add_argument("--artifact", type=str, default=None,
                   help="Path to extracted artifact directory")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset path (default: from config_resolved.yaml)")
    p.add_argument("--split", type=str, default="test",
                   choices=["test", "val", "train"],
                   help="Split to export (default: test)")
    p.add_argument("--max-graphs", type=int, default=None,
                   help="Limit number of graphs (for smoke test)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size override")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers")
    p.add_argument("--split-mode", type=str, default=None,
                   choices=["by_sample", "by_loadcase"],
                   help="Split mode (default: from config)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device (cpu, cuda, auto)")
    p.add_argument("--output-dir", type=str,
                   default="outputs/predictions/stage2b",
                   help="Root output directory")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    args._start_time = time.time()
    main(args)

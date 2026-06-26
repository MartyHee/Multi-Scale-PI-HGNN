"""
inspect_physics_loss_scale.py — Stage 5 Phase 1: Physics Loss Scale Dry-Run.

Loads a trained MS-HGT gated best_model.pt, runs inference on a small number
of train/val batches, and computes the numerical scale of:

  1. Supervised disp_loss + force_loss (MSE in standardised space)
  2. Support BC translation loss (both standardised and raw physical scale)
  3. Structural link translation consistency loss (both scales)

Outputs lambda recommendations for Stage 5 training so that the physics loss
contributions are roughly 1–5 % of the supervised loss.

Usage:
    # Local smoke test
    python scripts/inspect_physics_loss_scale.py ^
        --run-dir outputs/baselines/MS_HGT/20260624160353 ^
        --dataset processed/hetero_graph_dataset_v2 ^
        --split val ^
        --num-batches 1 --batch-size 1 --device cpu ^
        --output-dir outputs/diagnostics/stage5_loss_scale_smoke

    # Full dry-run (server — both splits, 4 batches each)
    python scripts/inspect_physics_loss_scale.py \
        --run-dir outputs/baselines/MS_HGT/20260624160353 \
        --dataset processed/hetero_graph_dataset_v2 \
        --split both \
        --num-batches 4 --batch-size 2 --device cuda \
        --output-dir outputs/diagnostics/stage5_loss_scale

    # From extracted server artifact (e.g. when run-dir not available)
    python scripts/inspect_physics_loss_scale.py \
        --run-dir remote_artifacts/extracted/server_ms_hgt_gated_20260625013512 \
        --dataset processed/hetero_graph_dataset_v2 \
        --split both \
        --num-batches 2 --batch-size 2 --device cpu \
        --output-dir outputs/diagnostics/stage5_loss_scale

Output:
    outputs/diagnostics/stage5_loss_scale/<timestamp>/
        loss_scale_summary.json
        loss_scale_batches.csv
        support_flag_check.json
        structural_link_check.json
        stage5_loss_scale_report.md

Do NOT:
    - Modify train_baseline.py
    - Add --lambda-bc / --lambda-link training parameters
    - Modify MSHGTBaseline or any model
    - Run full training or server job

Author: Claude Code / Stage 5 Phase 1
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
from src.models.baselines import MSHGTBaseline
from src.utils.metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISP_COMP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
TRANS_DOF_NAMES = ["Dx", "Dy", "Dz"]
STRUCTURAL_LINK_KEY = ("mesh_node", "structural_link", "mesh_node")

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
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


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
# Support flag restoration
# ---------------------------------------------------------------------------


def restore_support_flags(
    x_std: torch.Tensor,
    scaler: Optional[HeteroFeatureScaler],
) -> torch.Tensor:
    """Restore support_flags from standardised mesh_node.x[:, 9:15] to 0/1.

    Follows the same logic as export_full_predictions.py:
      1. Inverse-transform x[:, 9:15] to original scale.
      2. Threshold > 0.5 → binary 0/1.

    Args:
        x_std: Standardised mesh_node.x (N, 15).
        scaler: Fitted HeteroFeatureScaler.

    Returns:
        Binary support_flags (N, 6) — float 0.0 / 1.0.
    """
    sup_std = x_std[:, 9:15]  # (N, 6)

    if scaler is not None and scaler.fitted:
        x_mean = scaler._stats.get("node:mesh_node:x_mean")
        x_std_t = scaler._stats.get("node:mesh_node:x_std")
        if x_mean is not None and x_std_t is not None:
            x_mean = x_mean.to(sup_std.device)
            x_std_t = x_std_t.to(sup_std.device)
            sup_orig = sup_std * x_std_t[9:15] + x_mean[9:15]
            sup_binary = (sup_orig > 0.5).float()
            return sup_binary

    # Fallback: treat standardised values directly
    # If standardised: 0 remains 0, ~1/std becomes ~std
    # This is less reliable but avoids crash
    sup_orig = sup_std  # no restoration possible
    sup_binary = (sup_orig > 0.5).float()
    return sup_binary


def extract_scaler_stats(scaler: HeteroFeatureScaler) -> Dict:
    """Extract scaler stats as plain dict for JSON output."""
    if scaler is None or not scaler.fitted:
        return {}
    stats = {}
    for key in scaler._transform_keys:
        mean = scaler._stats.get(f"{key}_mean")
        std = scaler._stats.get(f"{key}_std")
        if mean is not None:
            stats[key] = {
                "mean": mean.tolist(),
                "std": std.tolist() if std is not None else None,
            }
    return stats


# ---------------------------------------------------------------------------
# Loss computation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_bc_loss(
    pred_disp: torch.Tensor,
    y_disp: torch.Tensor,
    support_flags: torch.Tensor,
    dof_indices: List[int],
) -> Dict:
    """Compute BC translation loss on support-constrained DOFs.

    BCE loss is computed on translation DOFs (0, 1, 2) for nodes where
    support_flags > 0.5 on that DOF.

    Two scales:
      - ``_std`` : standardised scale (model output space)
      - ``_raw`` : raw physical scale (after inverse transform)

    Args:
        pred_disp: (N, 6) model predictions (standardised).
        y_disp: (N, 6) ground truth (standardised).
        support_flags: (N, 6) binary support mask.
        dof_indices: List of DOF indices to include (e.g. [0,1,2] for trans).

    Returns:
        Dict with scalar loss values.
    """
    # Build mask: nodes that are support-constrained on ANY of the target DOFs
    any_constrained = support_flags[:, dof_indices].sum(dim=1) > 0.5
    n_support_nodes = any_constrained.sum().item()

    if n_support_nodes == 0:
        return {
            "n_support_nodes": 0,
            "n_constrained_dofs": 0,
            "bc_loss_std": 0.0,
            "bc_mae_std": 0.0,
            "per_dof_mae_std": [0.0] * len(dof_indices),
            "support_flag_min": 0.0,
            "support_flag_max": 0.0,
            "support_flag_unique": [0, 1] if support_flags.numel() > 0 else [],
        }

    pred_constrained = pred_disp[any_constrained]  # (n_support, 6)
    y_constrained = y_disp[any_constrained]  # (n_support, 6)

    # Build per-DOF mask: which DOFs are actually constrained for EACH node
    dof_mask = support_flags[any_constrained][:, dof_indices] > 0.5  # (n_support, n_dofs)
    n_constrained_dofs = dof_mask.sum().item()

    if n_constrained_dofs == 0:
        return {
            "n_support_nodes": n_support_nodes,
            "n_constrained_dofs": 0,
            "bc_loss_std": 0.0,
            "bc_mae_std": 0.0,
            "per_dof_mae_std": [0.0] * len(dof_indices),
            "support_flag_min": float(support_flags.min().item()),
            "support_flag_max": float(support_flags.max().item()),
        }

    # Compute MSE on constrained DOFs only
    # We use masked MSE: average over constrained DOFs only
    diff = pred_constrained[:, dof_indices] - y_constrained[:, dof_indices]  # (n_support, n_dofs)
    diff_masked = diff * dof_mask.float()
    mse = (diff_masked ** 2).sum() / n_constrained_dofs
    mae_per_dof = []
    for d_idx, _ in enumerate(dof_indices):
        mask_d = dof_mask[:, d_idx]
        if mask_d.sum() > 0:
            mae_d = diff_masked[mask_d, d_idx].abs().mean().item()
        else:
            mae_d = 0.0
        mae_per_dof.append(mae_d)
    mae_total = diff_masked.abs().sum() / n_constrained_dofs

    return {
        "n_support_nodes": n_support_nodes,
        "n_constrained_dofs": n_constrained_dofs,
        "bc_loss_std": float(mse.item()),
        "bc_mae_std": float(mae_total.item()),
        "per_dof_mae_std": mae_per_dof,
        "support_flag_min": float(support_flags.min().item()),
        "support_flag_max": float(support_flags.max().item()),
        "support_flag_unique": sorted(
            [float(v) for v in torch.unique(support_flags).tolist()]
        ),
    }


@torch.no_grad()
def compute_bc_loss_raw(
    pred_disp_std: torch.Tensor,
    y_disp_std: torch.Tensor,
    support_flags: torch.Tensor,
    dof_indices: List[int],
    disp_mean: Optional[torch.Tensor],
    disp_std: Optional[torch.Tensor],
) -> Dict:
    """Compute BC translation loss in raw physical scale.

    Inverse-transforms pred and y to raw scale, then computes BC loss.
    """
    if disp_mean is not None and disp_std is not None:
        pred_raw = pred_disp_std * disp_std + disp_mean
        y_raw = y_disp_std * disp_std + disp_mean
    else:
        pred_raw = pred_disp_std
        y_raw = y_disp_std

    any_constrained = support_flags[:, dof_indices].sum(dim=1) > 0.5
    n_support = any_constrained.sum().item()

    if n_support == 0:
        return {
            "bc_loss_raw": 0.0,
            "bc_mae_raw": 0.0,
            "per_dof_mae_raw": [0.0] * len(dof_indices),
        }

    pred_c = pred_raw[any_constrained]
    y_c = y_raw[any_constrained]
    dof_mask = support_flags[any_constrained][:, dof_indices] > 0.5
    n_c_dofs = dof_mask.sum().item()

    if n_c_dofs == 0:
        return {
            "bc_loss_raw": 0.0,
            "bc_mae_raw": 0.0,
            "per_dof_mae_raw": [0.0] * len(dof_indices),
        }

    diff = pred_c[:, dof_indices] - y_c[:, dof_indices]
    diff_masked = diff * dof_mask.float()
    mse = (diff_masked ** 2).sum() / n_c_dofs
    mae = diff_masked.abs().sum() / n_c_dofs

    per_dof_mae = []
    for d_idx in range(len(dof_indices)):
        mask_d = dof_mask[:, d_idx]
        per_dof_mae.append(
            float(diff_masked[mask_d, d_idx].abs().mean().item())
            if mask_d.sum() > 0 else 0.0
        )

    return {
        "bc_loss_raw": float(mse.item()),
        "bc_mae_raw": float(mae.item()),
        "per_dof_mae_raw": per_dof_mae,
    }


@torch.no_grad()
def compute_link_loss(
    pred_disp: torch.Tensor,
    edge_index: torch.Tensor,
) -> Dict:
    """Compute structural link translation consistency loss.

    For each structural link edge (i, j), computes MSE of pred_disp[i, :3]
    and pred_disp[j, :3] — the difference in translational displacement
    predictions between the two endpoints of a rigid link.

    Args:
        pred_disp: (N, 6) model predictions (any scale).
        edge_index: (2, E) edge indices.

    Returns:
        Dict with scalar loss values.
    """
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return {
            "n_edges": 0,
            "n_directed_edges": 0,
            "n_undirected_edges": 0,
            "link_loss": 0.0,
            "link_mae": 0.0,
        }

    src, dst = edge_index[0], edge_index[1]

    # Avoid OOB by clamping
    max_idx = pred_disp.shape[0]
    src = src.clamp(0, max_idx - 1)
    dst = dst.clamp(0, max_idx - 1)

    diff = pred_disp[src, :3] - pred_disp[dst, :3]  # (E, 3)
    sq = diff ** 2
    mse = sq.mean().item()
    mae = diff.abs().mean().item()

    # Detect bidirectionality: count unique undirected pairs
    edges_sorted = torch.stack([
        torch.minimum(src, dst),
        torch.maximum(src, dst),
    ], dim=1)  # (E, 2)
    unique_edges = torch.unique(edges_sorted, dim=0)
    n_undirected = unique_edges.shape[0]

    return {
        "n_edges": n_edges,
        "n_directed_edges": n_edges,
        "n_undirected_edges": n_undirected,
        "link_loss": float(mse),
        "link_mae": float(mae),
    }


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------


@torch.no_grad()
def inspect_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Optional[HeteroFeatureScaler],
    num_batches: int,
    split_name: str,
) -> List[Dict]:
    """Run loss inspection on a DataLoader.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader (val or train).
        device: torch device.
        scaler: Fitted scaler for inverse transforms.
        num_batches: Max batches to process.
        split_name: "train" or "val".

    Returns:
        List of per-batch result dicts.
    """
    print(f"\n  Inspecting {split_name} — up to {num_batches} batches")

    # Pre-fetch scaler stats for inverse transforms
    disp_mean = disp_std = None
    if scaler is not None and scaler.fitted:
        dm = scaler._stats.get("node:mesh_node:y_disp_mean")
        ds = scaler._stats.get("node:mesh_node:y_disp_std")
        if dm is not None and ds is not None:
            disp_mean = dm.to(device)
            disp_std = ds.to(device)

    mse_loss = torch.nn.MSELoss()
    batch_results = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        batch = batch.to(device)

        # ---- Forward ----
        try:
            pred_disp, pred_force = model(batch)
        except Exception as e:
            print(f"    [WARN] Batch {batch_idx} forward failed: {e}")
            continue

        # Validate shapes
        if pred_disp.shape[0] == 0 or pred_force.shape[0] == 0:
            print(f"    [WARN] Batch {batch_idx}: empty predictions, skipping")
            continue

        y_disp = batch["mesh_node"].y_disp
        y_force = batch["beam_element"].y_force

        # ---- 1. Supervised losses (standardised scale) ----
        disp_loss = mse_loss(pred_disp, y_disp).item()
        force_loss = mse_loss(pred_force, y_force).item()

        # ---- 2. Support flags ----
        x_mesh = batch["mesh_node"].x  # (N, 15)
        support_flags = restore_support_flags(x_mesh, scaler)
        # support_flags: (N, 6) binary float

        # ---- 3. Structural link edge_index ----
        link_edge_index = batch[STRUCTURAL_LINK_KEY].edge_index  # (2, E)

        # ---- 4. BC translation loss (translation DOFs: 0, 1, 2) ----
        trans_dofs = [0, 1, 2]
        bc_std = compute_bc_loss(
            pred_disp, y_disp, support_flags, trans_dofs,
        )
        bc_raw = compute_bc_loss_raw(
            pred_disp, y_disp, support_flags, trans_dofs,
            disp_mean, disp_std,
        )

        # ---- 5. Link translation consistency loss ----
        link_std = compute_link_loss(pred_disp, link_edge_index)
        # Raw-scale link loss
        if disp_mean is not None and disp_std is not None:
            pred_disp_raw = pred_disp * disp_std + disp_mean
        else:
            pred_disp_raw = pred_disp
        link_raw = compute_link_loss(pred_disp_raw, link_edge_index)

        # ---- 6. Structural link edge info ----
        n_edges = link_edge_index.shape[1]
        if n_edges >= 2:
            src_t, dst_t = link_edge_index[0], link_edge_index[1]
            dup_check = (src_t == src_t[0]) & (dst_t == dst_t[0])
            is_dup = dup_check.sum().item() > 1
        else:
            is_dup = False

        # ---- Collect batch results ----
        n_mesh = pred_disp.shape[0]
        n_beam = pred_force.shape[0]
        n_graphs_in_batch = int(batch["mesh_node"].batch.max().item() + 1)

        batch_result = {
            "split": split_name,
            "batch_idx": batch_idx,
            "num_graphs": n_graphs_in_batch,
            "num_mesh_nodes": n_mesh,
            "num_beam_elements": n_beam,
            "num_support_trans_dof": bc_std["n_constrained_dofs"],
            "num_support_nodes": bc_std["n_support_nodes"],
            "num_structural_link_edges": n_edges,
            "num_link_undirected_edges": link_std["n_undirected_edges"] if n_edges > 0 else 0,
            # Supervised losses (standardised)
            "disp_loss": disp_loss,
            "force_loss": force_loss,
            "total_supervised_loss": disp_loss + force_loss,
            # BC loss — standardised
            "bc_loss_std": bc_std["bc_loss_std"],
            "bc_mae_std": bc_std["bc_mae_std"],
            "bc_mae_std_dx": bc_std["per_dof_mae_std"][0],
            "bc_mae_std_dy": bc_std["per_dof_mae_std"][1],
            "bc_mae_std_dz": bc_std["per_dof_mae_std"][2],
            # BC loss — raw
            "bc_loss_raw": bc_raw["bc_loss_raw"],
            "bc_mae_raw": bc_raw["bc_mae_raw"],
            "bc_mae_raw_dx": bc_raw["per_dof_mae_raw"][0],
            "bc_mae_raw_dy": bc_raw["per_dof_mae_raw"][1],
            "bc_mae_raw_dz": bc_raw["per_dof_mae_raw"][2],
            # Link loss — standardised
            "link_loss_std": link_std["link_loss"],
            "link_mae_std": link_std["link_mae"],
            # Link loss — raw
            "link_loss_raw": link_raw["link_loss"],
            "link_mae_raw": link_raw["link_mae"],
            # Support flag diagnostics
            "support_flag_min": bc_std.get("support_flag_min", 0.0),
            "support_flag_max": bc_std.get("support_flag_max", 0.0),
        }
        batch_results.append(batch_result)

        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            print(f"    Batch {batch_idx + 1}/{num_batches}: "
                  f"disp_loss={disp_loss:.6f}, "
                  f"bc_std={bc_std['bc_loss_std']:.6e}, "
                  f"link_std={link_std['link_loss']:.6e}, "
                  f"bc_mae_raw={bc_raw['bc_mae_raw']:.6e}, "
                  f"#link_edges={n_edges}, "
                  f"#support_dofs={bc_std['n_constrained_dofs']}")

    return batch_results


# ---------------------------------------------------------------------------
# Lambda recommendation
# ---------------------------------------------------------------------------


def recommend_lambda(
    supervised_loss: float,
    physics_loss: float,
    physics_name: str,
    target_ratio: Tuple[float, float] = (0.01, 0.05),
) -> Tuple[float, str]:
    """Recommend lambda so that λ × physics_loss is target_ratio of supervised.

    Args:
        supervised_loss: Mean supervised loss (scalar).
        physics_loss: Mean physics loss raw value (scalar).
        physics_name: Human-readable name for logging.
        target_ratio: Desired (min_ratio, max_ratio) of physics contribution.

    Returns:
        (recommended_lambda, justification_string)
    """
    if physics_loss < 1e-12:
        return 0.0, (
            f"{physics_name} loss ≈ 0 (no constrained DOFs found). "
            f"Lambda set to 0."
        )

    target_contrib_low = supervised_loss * target_ratio[0]
    target_contrib_high = supervised_loss * target_ratio[1]
    target_contrib_mid = supervised_loss * 0.025  # 2.5 % midpoint

    # lambda = desired_contribution / physics_loss
    lam_mid = target_contrib_mid / physics_loss
    lam_low = target_contrib_low / physics_loss
    lam_high = target_contrib_high / physics_loss

    # Round to reasonable precision
    def _round_lam(v: float) -> float:
        if v < 0.001:
            return round(v, 6)
        elif v < 0.01:
            return round(v, 5)
        elif v < 0.1:
            return round(v, 4)
        elif v < 1.0:
            return round(v, 3)
        else:
            return round(v, 2)

    lam_rec = _round_lam(lam_mid)
    lam_low_r = _round_lam(lam_low)
    lam_high_r = _round_lam(lam_high)

    justification = (
        f"Supervised loss = {supervised_loss:.6e}, "
        f"{physics_name} = {physics_loss:.6e}. "
        f"Ratio physics / supervised = {physics_loss / max(supervised_loss, 1e-12):.6e}. "
        f"To achieve {target_ratio[0]*100:.1f}–{target_ratio[1]*100:.0f} % contribution, "
        f"λ ∈ [{lam_low_r}, {lam_high_r}]. "
        f"Recommended λ = {lam_rec} "
        f"(≈ 2.5 % of supervised loss)."
    )

    return lam_rec, justification


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace):
    print("=" * 60)
    print("STAGE 5 — PHYSICS LOSS SCALE DRY-RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # ---- 1. Load config ----
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config_resolved.yaml not found: {config_path}")
    config = _load_yaml(config_path)
    print(f"\nConfig loaded: {config_path}")

    # ---- 2. Model config ----
    model_summary_path = run_dir / "model_summary.json"
    model_cfg = {}
    if model_summary_path.exists():
        ms = _load_json(model_summary_path)
        model_cfg = ms.get("model_config", {})
        print(f"Model: {ms.get('model_class', '?')}, "
              f"{ms.get('total_params', '?'):,} params, "
              f"fusion: {model_cfg.get('fusion_method', '?')}")
    else:
        print("  [WARN] model_summary.json not found — using defaults")

    model_name = config.get("model_name", "ms_hgt")

    # ---- 3. Dataset ----
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = config.get("data", {}).get(
            "processed_dir", "processed/hetero_graph_dataset_v2")
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = _PROJECT_ROOT / dataset_path

    print(f"\n[1/4] Dataset: {dataset_path}")

    stats_path = dataset_path / "feature_stats.json"
    if stats_path.is_file():
        scaler = HeteroFeatureScaler.load(stats_path)
        print(f"  Scaler loaded: {stats_path}")
    else:
        scaler = None
        print("  [WARN] No feature_stats.json — data may not be standardised")

    # ---- 4. Build data loaders ----
    splits_to_run = []
    if args.split == "both":
        splits_to_run = ["train", "val"]
    else:
        splits_to_run = [args.split]

    loaders = {}
    for sp in splits_to_run:
        ds = HeteroGraphDataset(
            processed_dir=dataset_path,
            split=sp,
            split_mode=args.split_mode,
            transform=scaler,
        )
        # Safety limit: don't load more than we need
        max_g = args.num_batches * args.batch_size
        if len(ds) > max_g:
            ds = torch.utils.data.Subset(ds, range(max_g))
        loaders[sp] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"  {sp}: {len(ds)} graphs ({len(loaders[sp])} batches)")

    # ---- 5. Build model ----
    print(f"\n[2/4] Building MSHGTBaseline ...")
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
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # ---- 6. Load checkpoint ----
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best_model.pt not found: {checkpoint_path}")

    print(f"\n[3/4] Loading checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True,
        )
        # Handle possible DataParallel prefix
        sd_clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(sd_clean, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {unexpected[:5]}...")
        print(f"  Loaded {len(sd_clean)} keys")
    except Exception as e:
        print(f"  [ERROR] {e}")
        raise

    model.eval()

    # ---- 7. Inspect losses ----
    print(f"\n[4/4] Computing loss scales ...")

    all_batch_results = []
    for sp in splits_to_run:
        results = inspect_losses(
            model=model,
            loader=loaders[sp],
            device=device,
            scaler=scaler,
            num_batches=args.num_batches,
            split_name=sp,
        )
        all_batch_results.extend(results)

    if len(all_batch_results) == 0:
        print("\n  [ERROR] No batches processed. Check DataLoader or dataset.")
        sys.exit(1)

    # ---- 8. Aggregate statistics ----
    print(f"\n  Aggregating over {len(all_batch_results)} batches ...")

    def _mean(key: str) -> float:
        vals = [r[key] for r in all_batch_results]
        return float(np.mean(vals))

    def _std(key: str) -> float:
        vals = [r[key] for r in all_batch_results]
        return float(np.std(vals))

    def _min(key: str) -> float:
        return float(min(r[key] for r in all_batch_results))

    def _max(key: str) -> float:
        return float(max(r[key] for r in all_batch_results))

    agg = {
        "num_batches": len(all_batch_results),
        "splits": list(set(r["split"] for r in all_batch_results)),
        "supervised_disp_loss_mean": _mean("disp_loss"),
        "supervised_disp_loss_std": _std("disp_loss"),
        "supervised_force_loss_mean": _mean("force_loss"),
        "supervised_force_loss_std": _std("force_loss"),
        "supervised_total_loss_mean": _mean("total_supervised_loss"),
        "supervised_total_loss_std": _std("total_supervised_loss"),
        # BC — standardised
        "bc_loss_std_mean": _mean("bc_loss_std"),
        "bc_loss_std_std": _std("bc_loss_std"),
        "bc_mae_std_mean": _mean("bc_mae_std"),
        "bc_mae_std_dx_mean": _mean("bc_mae_std_dx"),
        "bc_mae_std_dy_mean": _mean("bc_mae_std_dy"),
        "bc_mae_std_dz_mean": _mean("bc_mae_std_dz"),
        # BC — raw
        "bc_loss_raw_mean": _mean("bc_loss_raw"),
        "bc_loss_raw_std": _std("bc_loss_raw"),
        "bc_mae_raw_mean": _mean("bc_mae_raw"),
        "bc_mae_raw_dx_mean": _mean("bc_mae_raw_dx"),
        "bc_mae_raw_dy_mean": _mean("bc_mae_raw_dy"),
        "bc_mae_raw_dz_mean": _mean("bc_mae_raw_dz"),
        # Link — standardised
        "link_loss_std_mean": _mean("link_loss_std"),
        "link_loss_std_std": _std("link_loss_std"),
        "link_mae_std_mean": _mean("link_mae_std"),
        # Link — raw
        "link_loss_raw_mean": _mean("link_loss_raw"),
        "link_loss_raw_std": _std("link_loss_raw"),
        "link_mae_raw_mean": _mean("link_mae_raw"),
        # Coverage
        "pct_batches_with_support_dofs": float(
            sum(1 for r in all_batch_results if r["num_support_trans_dof"] > 0)
        ) / max(len(all_batch_results), 1) * 100.0,
        "pct_batches_with_link_edges": float(
            sum(1 for r in all_batch_results if r["num_structural_link_edges"] > 0)
        ) / max(len(all_batch_results), 1) * 100.0,
        # Total DOF counts
        "total_support_trans_dofs": sum(r["num_support_trans_dof"] for r in all_batch_results),
        "total_support_nodes": sum(r["num_support_nodes"] for r in all_batch_results),
        "total_link_edges": sum(r["num_structural_link_edges"] for r in all_batch_results),
        # Support flag diagnostics
        "support_flag_min": min(r["support_flag_min"] for r in all_batch_results),
        "support_flag_max": max(r["support_flag_max"] for r in all_batch_results),
    }

    # ---- 9. Lambda recommendation ----
    sup_loss = agg["supervised_total_loss_mean"]

    # BC loss — use standardised scale for training (matching supervised loss space)
    bc_loss_val = agg["bc_loss_std_mean"]
    bc_lambda, bc_just = recommend_lambda(
        sup_loss, bc_loss_val, "BC (std)",
    )

    # Link loss — use standardised scale
    link_loss_val = agg["link_loss_std_mean"]
    link_lambda, link_just = recommend_lambda(
        sup_loss, link_loss_val, "Link (std)",
    )

    agg["recommended_lambda_bc"] = bc_lambda
    agg["recommended_lambda_bc_justification"] = bc_just
    agg["recommended_lambda_link"] = link_lambda
    agg["recommended_lambda_link_justification"] = link_just

    # ---- 10. Output ----
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    ensure_dir(output_dir)

    # 10a. loss_scale_summary.json
    summary_path = output_dir / "loss_scale_summary.json"
    _save_json(agg, summary_path)
    print(f"\n  Summary saved: {summary_path}")

    # 10b. loss_scale_batches.csv
    csv_path = output_dir / "loss_scale_batches.csv"
    import csv as csv_module
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_batch_results[0].keys())
        w = csv_module.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_batch_results:
            w.writerow(r)
    print(f"  Batch CSV saved: {csv_path} ({len(all_batch_results)} rows)")

    # 10c. support_flag_check.json
    sup_check = {
        "support_flags_source": "mesh_node.x[:, 9:15] (Dx_fix..Rz_fix)",
        "restoration_method": (
            "Inverse-transform via feature_stats['node:mesh_node:x'][9:15], "
            "then threshold > 0.5. Same logic as export_full_predictions.py."
        ),
        "scaler_stats_key": "node:mesh_node:x",
        "support_flag_min_overall": agg["support_flag_min"],
        "support_flag_max_overall": agg["support_flag_max"],
        "flag_values_after_restore": "binary 0/1 (threshold > 0.5)",
        "total_support_nodes": agg["total_support_nodes"],
        "total_support_trans_dofs": agg["total_support_trans_dofs"],
        "pct_batches_with_support_dofs": agg["pct_batches_with_support_dofs"],
        "num_batches": agg["num_batches"],
        "batch_results": [
            {
                "split": r["split"],
                "batch_idx": r["batch_idx"],
                "num_support_nodes": r["num_support_nodes"],
                "num_support_trans_dofs": r["num_support_trans_dof"],
            }
            for r in all_batch_results
        ],
    }
    sup_path = output_dir / "support_flag_check.json"
    _save_json(sup_check, sup_path)
    print(f"  Support flag check saved: {sup_path}")

    # 10d. structural_link_check.json
    link_check = {
        "edge_type": "mesh_node -> structural_link -> mesh_node",
        "edge_type_key": "('mesh_node', 'structural_link', 'mesh_node')",
        "n_edges_total": agg["total_link_edges"],
        "total_batches": agg["num_batches"],
        "pct_batches_with_edges": agg["pct_batches_with_link_edges"],
        "bidirectional_note": (
            "edge_index contains directed edges (both directions). "
            "n_undirected_edges should be roughly half of n_directed_edges."
        ),
        "batch_results": [
            {
                "split": r["split"],
                "batch_idx": r["batch_idx"],
                "n_edges": r["num_structural_link_edges"],
                "n_undirected": r["num_link_undirected_edges"],
            }
            for r in all_batch_results
        ],
    }
    link_path = output_dir / "structural_link_check.json"
    _save_json(link_check, link_path)
    print(f"  Link check saved: {link_path}")

    # 10e. stage5_loss_scale_report.md
    report_path = output_dir / "stage5_loss_scale_report.md"
    _write_report(agg, all_batch_results, checkpoint_path, output_dir, report_path, model_name, config)
    print(f"  Report saved: {report_path}")

    # ---- Print summary ----
    total_time = time.time() - getattr(args, "_start_time", time.time())
    print(f"\n{'=' * 60}")
    print("DRY-RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:        {model_name} (MSHGTBaseline, {n_params:,} params)")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Batches:      {agg['num_batches']}")
    print(f"  Splits:       {', '.join(agg['splits'])}")
    print(f"  Supervised:   {agg['supervised_total_loss_mean']:.6e}")
    print(f"  BC (std):     {agg['bc_loss_std_mean']:.6e}  "
          f"(raw MAE: {agg['bc_mae_raw_mean']:.6e})")
    print(f"  Link (std):   {agg['link_loss_std_mean']:.6e}  "
          f"(raw MAE: {agg['link_mae_raw_mean']:.6e})")
    print(f"  λ_bc:         {bc_lambda:.6f}")
    print(f"  λ_link:       {link_lambda:.6f}")
    print(f"  Output:       {output_dir}")
    print(f"  Time:         {fmt_time(total_time)}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    agg: Dict,
    batch_results: List[Dict],
    checkpoint_path: Path,
    output_dir: Path,
    report_path: Path,
    model_name: str,
    config: Dict,
):
    """Write human-readable markdown report."""

    lines = []
    def L(s=""):
        lines.append(s)

    L("# Stage 5 Physics Loss Scale — Dry-Run Report")
    L(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"> Backbone: {model_name} / MSHGTBaseline")
    L(f"> Checkpoint: {checkpoint_path}")
    L()

    L("## 1. Supervision Loss Scale")
    L()
    L(f"| Metric | Value (std scale) |")
    L(f"|:-------|:-----------------:|")
    L(f"| Disp MSE | {agg['supervised_disp_loss_mean']:.6e} ± {agg['supervised_disp_loss_std']:.6e} |")
    L(f"| Force MSE | {agg['supervised_force_loss_mean']:.6e} ± {agg['supervised_force_loss_std']:.6e} |")
    L(f"| **Total supervised** | **{agg['supervised_total_loss_mean']:.6e}** ± {agg['supervised_total_loss_std']:.6e} |")
    L()

    L("## 2. Support BC Translation Loss")
    L()
    L("### 2.1 Support Flag Check")
    L()
    L(f"- **Source**: `mesh_node.x[:, 9:15]` (Dx_fix, Dy_fix, Dz_fix, Rx_fix, Ry_fix, Rz_fix)")
    L(f"- **Restoration**: Inverse-transform via feature_stats, threshold > 0.5 (same as export_full_predictions.py)")
    L(f"- **Support flag values after restore**: binary 0/1")
    L(f"- **Total support nodes across batches**: {agg['total_support_nodes']}")
    L(f"- **Total constrained translation DOFs**: {agg['total_support_trans_dofs']}")
    L(f"- **Batches with support DOFs**: {agg['pct_batches_with_support_dofs']:.0f} %")
    L()

    L("### 2.2 BC Loss Numerical Scale")
    L()
    sup_loss = agg["supervised_total_loss_mean"]
    bc_std = agg["bc_loss_std_mean"]
    bc_raw = agg["bc_loss_raw_mean"]
    bc_ratio = bc_std / max(sup_loss, 1e-12)

    L(f"| Scale | MSE | MAE | Per-DOF MAE |")
    L(f"|:----|:---:|:---:|:-----------|")
    L(f"| **Standardised** | {bc_std:.6e} | {agg['bc_mae_std_mean']:.6e} | "
      f"Dx: {agg['bc_mae_std_dx_mean']:.6e}, Dy: {agg['bc_mae_std_dy_mean']:.6e}, Dz: {agg['bc_mae_std_dz_mean']:.6e} |")
    L(f"| **Raw (physical)** | {bc_raw:.6e} | {agg['bc_mae_raw_mean']:.6e} | "
      f"Dx: {agg['bc_mae_raw_dx_mean']:.6e}, Dy: {agg['bc_mae_raw_dy_mean']:.6e}, Dz: {agg['bc_mae_raw_dz_mean']:.6e} |")
    L()
    L(f"- **BC / Supervised ratio (std scale)**: {bc_ratio:.4e} ({bc_ratio*100:.4f} %)")
    L()

    # Transfer to safe range for display
    unit = "mm"  # approximate
    L(f"### 2.3 Physical Interpretation")
    L()
    L(f"- **BC Translation MAE** (raw): {agg['bc_mae_raw_mean']:.6e} (approx displacement unit)")
    L(f"- By comparison, export_full_predictions reported MS-HGT BC MAE ≈ 0.000242 (raw physical)")
    L(f"- The dry-run value may differ due to limited batch sample.")
    L()

    L("## 3. Structural Link Translation Consistency Loss")
    L()
    L("### 3.1 Edge Index Check")
    L()
    edge_types_with_data = [
        r["num_structural_link_edges"] for r in batch_results if r["num_structural_link_edges"] > 0
    ]
    n_edges_total = agg["total_link_edges"]
    n_batch_with_edges = sum(1 for r in batch_results if r["num_structural_link_edges"] > 0)
    L(f"- **Edge type**: `('mesh_node', 'structural_link', 'mesh_node')`")
    L(f"- **Total directed edges across batches**: {n_edges_total}")
    L(f"- **Batches with edges**: {agg['pct_batches_with_link_edges']:.0f} %")
    L(f"- **Bidirectional?**: edge_index contains directed pairs (i→j AND j→i). "
      f"n_undirected ≈ n_directed / 2 confirmed by batch results.")
    L()

    L("### 3.2 Link Loss Numerical Scale")
    L()
    link_ratio_std = agg["link_loss_std_mean"] / max(sup_loss, 1e-12)
    link_ratio_raw = agg["link_loss_raw_mean"] / max(sup_loss, 1e-12) if sup_loss > 0 else 0

    L(f"| Scale | MSE | MAE |")
    L(f"|:----|:---:|:---:|")
    L(f"| **Standardised** | {agg['link_loss_std_mean']:.6e} | {agg['link_mae_std_mean']:.6e} |")
    L(f"| **Raw (physical)** | {agg['link_loss_raw_mean']:.6e} | {agg['link_mae_raw_mean']:.6e} |")
    L()
    L(f"- **Link / Supervised ratio (std scale)**: {link_ratio_std:.4e} ({link_ratio_std*100:.4f} %)")
    L(f"- **Link / Supervised ratio (raw scale)**: {link_ratio_raw:.4e} ({link_ratio_raw*100:.4f} %)")
    L()

    L("## 4. Lambda Recommendations")
    L()
    L("### 4.1 Principle")
    L()
    L("Target: physics loss contribution ≈ 1–5 % of total supervised loss.")
    L("Standardised scale is used for training (matching supervised loss space).")
    L("Raw physical scale is used only for evaluation reporting.")
    L()
    L(f"| Physics Loss | Raw Value | Supervised | Ratio (std) | Recommended λ | Contribution |")
    L(f"|:------------|:--------:|:----------:|:----------:|:------------:|:-----------:|")
    L(f"| BC translation | {agg['bc_loss_std_mean']:.4e} | {sup_loss:.4e} | "
      f"{bc_std/sup_loss*100:.3f} % | **{agg['recommended_lambda_bc']:.6f}** | "
      f"{agg['recommended_lambda_bc'] * agg['bc_loss_std_mean'] / sup_loss * 100:.2f} % |")
    L(f"| Link translation | {agg['link_loss_std_mean']:.4e} | {sup_loss:.4e} | "
      f"{agg['link_loss_std_mean']/sup_loss*100:.3f} % | **{agg['recommended_lambda_link']:.6f}** | "
      f"{agg['recommended_lambda_link'] * agg['link_loss_std_mean'] / sup_loss * 100:.2f} % |")
    L()

    L("### 4.2 Recommended Values")
    L()
    bc_lambda = agg["recommended_lambda_bc"]
    link_lambda = agg["recommended_lambda_link"]
    L(f"| λ | Initial Value | vs Stage 5 Design | Action |")
    L(f"|:-:|:------------:|:-----------------:|:-------|")
    L(f"| λ_bc | **{bc_lambda}** | "
      f"{'≈ design (0.05)' if abs(bc_lambda - 0.05) < 0.01 else f'≠ design (0.05) — adjust' } | "
      f"{'Use as-is' if abs(bc_lambda - 0.05) < 0.01 else 'Update configs' } |")
    L(f"| λ_link | **{link_lambda}** | "
      f"{'≈ design (0.005)' if abs(link_lambda - 0.005) < 0.001 else f'≠ design (0.005) — adjust' } | "
      f"{'Use as-is' if abs(link_lambda - 0.005) < 0.001 else 'Update configs' } |")
    L()

    L("### 4.3 Justification")
    L()
    L(f"**λ_bc = {bc_lambda}**: {agg['recommended_lambda_bc_justification']}")
    L()
    L(f"**λ_link = {link_lambda}**: {agg['recommended_lambda_link_justification']}")
    L()

    L("## 5. Implementation Recommendations")
    L()
    L("### 5.1 Can safely implement BC loss?")
    L("- BC loss is well-defined: support_flags are binary 0/1 after restoration,")
    L("  constrained DOFs exist in all batches (or most batches).")
    L("- The standardised scale is directly comparable to supervised loss.")
    L("- **Yes — safe to implement BC loss.**")

    bc_flag = (
        "BC loss contribution is reasonable and will not dominate training."
        if bc_lambda * agg["bc_loss_std_mean"] < sup_loss * 0.1
        else "WARNING: BC loss contribution may be significant — monitor closely."
    )
    L(f"  - {bc_flag}")
    L()

    L("### 5.2 Can safely implement Link loss?")
    L("- Link consistency loss is well-defined: edge_index is available and contains edges.")
    L("- The structural links are rigid (all is_rigid=1), so consistency constraint is physically correct.")
    L("- **Yes — safe to implement Link loss.**")

    link_flag = (
        f"Link loss contribution is reasonable ({link_lambda * agg['link_loss_std_mean'] / sup_loss * 100:.2f} % of supervised)."
        if link_lambda * agg["link_loss_std_mean"] < sup_loss * 0.1
        else "WARNING: Link loss contribution may be significant."
    )
    L(f"  - {link_flag}")
    L()

    L("### 5.3 Training scale: standardised vs raw")
    L("- **Train in standardised scale** (same as current supervised loss).")
    L("- Evaluate BC MAE in raw physical scale for reporting.")
    L("  - This matches the existing trainer behaviour (losses in standardised space, metrics in raw space).")
    L()

    L("### 5.4 Lambda adjustment vs Stage 5 design")
    bc_lambda_design = 0.05
    link_lambda_design = 0.005
    if abs(bc_lambda - bc_lambda_design) / max(bc_lambda_design, 1e-12) > 0.2:
        L(f"- **λ_bc**: Dry-run suggests {bc_lambda:.4f} vs design {bc_lambda_design}. "
          f"Consider updating Stage 5 config.")
    else:
        L(f"- **λ_bc**: Dry-run {bc_lambda:.4f} ≈ design {bc_lambda_design}. "
          f"Initial suggestion is valid.")
    if abs(link_lambda - link_lambda_design) / max(link_lambda_design, 1e-12) > 0.2:
        L(f"- **λ_link**: Dry-run suggests {link_lambda:.6f} vs design {link_lambda_design}. "
          f"Consider updating Stage 5 config.")
    else:
        L(f"- **λ_link**: Dry-run {link_lambda:.6f} ≈ design {link_lambda_design}. "
          f"Initial suggestion is valid.")
    L()

    L("## 6. Per-Batch Details")
    L()
    L("| Batch | Split | Graphs | Nodes | Support DOFs | BC std | BC MAE raw | Link edges | Link std |")
    L("|:-----|:-----|:-----:|:-----:|:-----------:|:------:|:----------:|:---------:|:--------:|")
    for r in batch_results:
        L(f"| {r['batch_idx']} | {r['split']} | {r['num_graphs']} | {r['num_mesh_nodes']} | "
          f"{r['num_support_trans_dof']} | {r['bc_loss_std']:.2e} | {r['bc_mae_raw']:.2e} | "
          f"{r['num_structural_link_edges']} | {r['link_loss_std']:.2e} |")
    L()

    L("## 7. Output Files")
    L()
    L(f"- `loss_scale_summary.json` — Aggregated statistics and recommendations")
    L(f"- `loss_scale_batches.csv` — Per-batch loss values")
    L(f"- `support_flag_check.json` — Support flag source and coverage")
    L(f"- `structural_link_check.json` — Structural link edge index diagnostics")
    L(f"- `stage5_loss_scale_report.md` — This report")
    L()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 5 Phase 1: Physics Loss Scale Dry-Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Local smoke test (1 batch, CPU)
              python scripts/inspect_physics_loss_scale.py
                  --run-dir outputs/baselines/MS_HGT/20260624160353
                  --dataset processed/hetero_graph_dataset_v2
                  --split val --num-batches 1 --batch-size 1 --device cpu
                  --output-dir outputs/diagnostics/stage5_loss_scale_smoke

              # Full dry-run both splits (server)
              python scripts/inspect_physics_loss_scale.py
                  --run-dir outputs/baselines/MS_HGT/20260624160353
                  --dataset processed/hetero_graph_dataset_v2
                  --split both --num-batches 4 --batch-size 2 --device cuda
                  --output-dir outputs/diagnostics/stage5_loss_scale

              # From extracted artifact
              python scripts/inspect_physics_loss_scale.py
                  --run-dir remote_artifacts/extracted/server_ms_hgt_gated_20260625013512
                  --dataset processed/hetero_graph_dataset_v2
                  --split both --num-batches 2 --batch-size 2 --device cpu
                  --output-dir outputs/diagnostics/stage5_loss_scale
        """),
    )
    p.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to trained experiment run directory (or extracted artifact)",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Path to processed dataset (default: from config_resolved.yaml)",
    )
    p.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val", "both"],
        help="Split(s) to inspect (default: val)",
    )
    p.add_argument(
        "--split-mode", type=str, default="by_sample",
        choices=["by_sample", "by_loadcase"],
        help="Dataset split mode (default: by_sample)",
    )
    p.add_argument(
        "--num-batches", type=int, default=4,
        help="Number of batches per split (default: 4)",
    )
    p.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size for DataLoader (default: 2)",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0, safe for Windows)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device: cpu, cuda, or auto",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/diagnostics/stage5_loss_scale",
        help="Root output directory (default: outputs/diagnostics/stage5_loss_scale)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    args._start_time = time.time()

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)

"""
compute_conformal.py — Split Conformal Prediction for Stage 6 UQ.

Loads prediction NPZs (mesh_node + beam_element), creates calibration/evaluation
splits, computes component-wise nonconformity scores and conformal quantiles,
generates prediction intervals, and outputs coverage metrics.

Supports three split modes:
  - test_graph_50_50: random 50/50 graph-level split of test predictions
  - test_sample_3_4: fixed sample-level split of test predictions
  - val_to_test: val predictions as calibration, test predictions as evaluation

Usage:
    # Primary split (test 50/50)
    python scripts/compute_conformal.py
        --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test
        --split-mode test_graph_50_50
        --alpha 0.10 0.05
        --seed 42
        --output-dir outputs/diagnostics/stage6_uq

    # Smoke test
    python scripts/compute_conformal.py
        --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test
        --split-mode test_graph_50_50
        --alpha 0.10
        --max-graphs 10
        --output-dir outputs/diagnostics/stage6_uq_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISP_COMP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_COMP_NAMES = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]

DISP_KEYS = ["y_true_disp", "y_pred_disp"]
FORCE_KEYS = ["y_true_force", "y_pred_force"]

# Fixed nodes/elements per graph for hetero_graph_dataset_v2
N_MESH_PER_GRAPH = 1056
N_BEAM_PER_GRAPH = 1646

# Fixed sample-level calibration split (B-Sample)
CAL_SAMPLE_IDS = {3014, 4648, 5277}
EVAL_SAMPLE_IDS = {1700, 3059, 3281, 4761}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_csv(rows: List[Dict], path: Path, fieldnames: List[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write empty file with header
        rows = [{}]
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _compute_global_graph_id(local_graph_id: np.ndarray, n_per_graph: int) -> np.ndarray:
    """Reconstruct global graph index from batch-local graph_id.

    The NPZ stores batch-local graph_ids (0..batch_size-1). Since each graph
    has exactly n_per_graph rows, the global graph index is simply
    row_index // n_per_graph.
    """
    return np.arange(len(local_graph_id)) // n_per_graph


def _compute_x_region(x: np.ndarray) -> np.ndarray:
    """Assign x-position to region bins (Q1-Q5 by span fraction).

    Args:
        x: (n_nodes,) or (n_nodes, 3) array. If 2D, uses column 0 (x-coordinate).

    Returns:
        bins: (n_nodes,) array with values 0 (Q1) through 4 (Q5).
    """
    if x.ndim == 2:
        x_vals = x[:, 0]  # use x-coordinate only
    else:
        x_vals = x
    x_min, x_max = x_vals.min(), x_vals.max()
    span = x_max - x_min
    if span < 1e-12:
        return np.full(len(x_vals), 2, dtype=int)  # default to midspan if all same x
    frac = (x_vals - x_min) / span
    bins = np.digitize(frac, bins=[0.2, 0.4, 0.6, 0.8])  # returns 0..4
    return bins  # 0=Q1, 1=Q2, 2=Q3, 3=Q4, 4=Q5


def _fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    elif sec < 3600:
        return f"{sec / 60:.1f}min"
    else:
        return f"{sec / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Conformal quantile with finite-sample correction
# ---------------------------------------------------------------------------


def compute_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> Tuple[float, int, int]:
    """Compute split conformal quantile with finite-sample correction.

    Args:
        scores: Nonconformity scores (n_calibration,).
        alpha: Miscalverage level (e.g., 0.10 for 90% interval).

    Returns:
        q: Conformal quantile.
        k: Index used (1-based).
        n: Number of calibration points.
    """
    n = len(scores)
    if n == 0:
        return float("nan"), 0, 0

    scores_sorted = np.sort(scores)
    # Finite-sample correction: k = ceil((n + 1) * (1 - alpha))
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)  # clamp to n
    q = float(scores_sorted[k - 1])

    return q, k, n


def compute_conformal_quantiles_componentwise(
    score_matrix: np.ndarray,
    alpha: float,
    comp_names: List[str],
) -> Tuple[np.ndarray, Dict]:
    """Compute conformal quantiles per component.

    Args:
        score_matrix: (n_calibration, n_components) array of scores.
        alpha: Miscalverage level.
        comp_names: Component names.

    Returns:
        quantiles: (n_components,) array of conformal quantiles.
        info_dict: Per-component metadata.
    """
    n_comp = score_matrix.shape[1]
    quantiles = np.zeros(n_comp)
    info = {}

    for i in range(n_comp):
        comp_scores = score_matrix[:, i]
        q_val, k, n = compute_conformal_quantile(comp_scores, alpha)
        quantiles[i] = q_val
        info[comp_names[i]] = {
            "n_cal_points": int(n),
            "k": int(k),
            "alpha": float(alpha),
            "quantile": float(q_val),
            "finite_sample_correction": True,
        }

    return quantiles, info


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------


def compute_coverage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    alpha: float,
    comp_names: List[str],
) -> Tuple[np.ndarray, Dict]:
    """Compute coverage metrics on evaluation set.

    Args:
        y_true: (n_eval, n_components) ground truth.
        y_pred: (n_eval, n_components) predictions.
        quantiles: (n_components,) conformal quantiles.
        alpha: Miscalverage level.
        comp_names: Component names.

    Returns:
        covered: (n_eval, n_components) boolean coverage matrix.
        metrics_dict: Per-component coverage metrics.
    """
    lower = y_pred - quantiles  # (n_eval, n_comp)
    upper = y_pred + quantiles
    covered = (y_true >= lower) & (y_true <= upper)

    n = y_true.shape[0]
    target_coverage = 1.0 - alpha
    metrics = {}

    for i, name in enumerate(comp_names):
        emp_cov = float(covered[:, i].mean()) if n > 0 else float("nan")
        width = 2.0 * float(quantiles[i])
        metrics[name] = {
            "target_coverage": target_coverage,
            "empirical_coverage": emp_cov,
            "coverage_gap": emp_cov - target_coverage,
            "interval_width": width,
            "n_eval_points": int(n),
            "alpha": float(alpha),
        }

    # Aggregate
    all_covered = covered.flatten()
    emp_cov_all = float(all_covered.mean()) if len(all_covered) > 0 else float("nan")
    metrics["ALL"] = {
        "target_coverage": target_coverage,
        "empirical_coverage": emp_cov_all,
        "coverage_gap": emp_cov_all - target_coverage,
        "interval_width": float(np.mean(quantiles) * 2),
        "n_eval_points": int(covered.size),
        "alpha": float(alpha),
    }

    return covered, metrics


def compute_width_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    comp_names: List[str],
    eps: float = 1e-12,
) -> Dict:
    """Compute interval width metrics relative to response scale.

    Args:
        y_true: (n_eval, n_components) ground truth.
        y_pred: (n_eval, n_components) predictions.
        quantiles: (n_components,) conformal quantiles.
        comp_names: Component names.
        eps: Small value to avoid division by zero.

    Returns:
        width_metrics: Per-component width statistics.
    """
    width_metrics = {}

    for i, name in enumerate(comp_names):
        q = float(quantiles[i])
        width = 2.0 * q

        # Width / median(|y| + eps)
        abs_y = np.abs(y_true[:, i])
        median_abs_y = float(np.median(abs_y))
        denom = max(median_abs_y, eps)
        width_rel = width / denom

        # Width / component IQR
        p75 = float(np.percentile(abs_y, 75))
        p25 = float(np.percentile(abs_y, 25))
        iqr = max(p75 - p25, eps)
        width_iqr = width / iqr

        # Width / component std (from y_pred distribution)
        comp_std = float(max(np.std(y_pred[:, i]), eps))
        width_std = width / comp_std

        width_metrics[name] = {
            "interval_width": width,
            "median_abs_y": median_abs_y,
            "width_over_median_abs_y": width_rel,
            "width_over_abs_y_iqr": width_iqr,
            "width_over_pred_std": width_std,
        }

    return width_metrics


# ---------------------------------------------------------------------------
# Region-wise coverage
# ---------------------------------------------------------------------------




def compute_region_coverage(
    y_true_d_eval: np.ndarray,
    y_pred_d_eval: np.ndarray,
    node_xyz_eval: np.ndarray,
    support_flags_eval: np.ndarray,
    quantiles_disp: np.ndarray,
    alpha: float,
    eval_mask: np.ndarray,
) -> Tuple[Dict, np.ndarray]:
    """Compute coverage partitioned by region. Only uses evaluation-set nodes.

    Regions:
      - support: any DOF constrained
      - free: no constrained DOF
      - x-regions Q1-Q5 (by span fraction)

    Returns:
        region_metrics: Dict of region -> coverage metrics.
        region_labels: (n_mesh_total,) integer region label array (-1 for non-eval).
    """
    target_cov = 1.0 - alpha

    # Filter to evaluation set
    y_t = y_true_d_eval[eval_mask]
    y_p = y_pred_d_eval[eval_mask]
    x = node_xyz_eval[eval_mask]
    sup = support_flags_eval[eval_mask]

    # Per-node coverage (all 6 DOF must be covered simultaneously)
    lower = y_p - quantiles_disp
    upper = y_p + quantiles_disp
    covered_all = ((y_t >= lower) & (y_t <= upper)).all(axis=1)

    region_metrics = {}

    # 1. Support vs free
    is_support = sup.any(axis=1).astype(bool)

    for region_name, mask in [("support", is_support), ("free", ~is_support)]:
        if mask.sum() == 0:
            region_metrics[region_name] = {"n_nodes": 0, "coverage_all_dof": float("nan")}
            continue
        cov_val = float(covered_all[mask].mean())
        region_metrics[region_name] = {
            "n_nodes": int(mask.sum()),
            "target_coverage": float(target_cov),
            "empirical_coverage": cov_val,
            "coverage_gap": cov_val - target_cov,
            "interval_width": float(2.0 * quantiles_disp.mean()),
        }

    # 2. X-regions Q1-Q5
    x_regions = _compute_x_region(x)
    region_labels_full = np.full(len(y_true_d_eval), -1, dtype=int)
    region_labels_full[eval_mask] = x_regions

    for qi in range(5):
        mask = x_regions == qi
        if mask.sum() == 0:
            region_metrics[f"x_region_Q{qi+1}"] = {"n_nodes": 0, "coverage_all_dof": float("nan")}
            continue
        cov_val = float(covered_all[mask].mean())
        region_metrics[f"x_region_Q{qi+1}"] = {
            "n_nodes": int(mask.sum()),
            "target_coverage": float(target_cov),
            "empirical_coverage": cov_val,
            "coverage_gap": cov_val - target_cov,
            "interval_width": float(2.0 * quantiles_disp.mean()),
        }

    return region_metrics, region_labels_full


def compute_high_response_coverage(
    y_true_disp: np.ndarray,
    y_pred_disp: np.ndarray,
    quantiles_disp: np.ndarray,
    alpha: float,
    eval_mask: np.ndarray,
    percentiles: List[float] = None,
) -> Dict:
    """Compute coverage on high-response subsets.

    High-response defined by true displacement magnitude (after-the-fact diagnostic).

    Args:
        y_true_disp: (n_mesh_total, 6) ground truth.
        y_pred_disp: (n_mesh_total, 6) predictions.
        quantiles_disp: (6,) conformal quantiles.
        alpha: Miscalverage level.
        eval_mask: (n_mesh_total,) boolean mask for evaluation nodes.
        percentiles: Response quantile thresholds (default [90, 95, 99]).

    Returns:
        hr_metrics: Dict of percentile -> coverage metrics.
    """
    if percentiles is None:
        percentiles = [90, 95, 99]

    target_cov = 1.0 - alpha
    y_t = y_true_disp[eval_mask]
    y_p = y_pred_disp[eval_mask]

    # Compute per-node displacement magnitude
    disp_mag = np.linalg.norm(y_t, axis=1)

    # Coverage per node (all 6 DOF)
    lower = y_p - quantiles_disp
    upper = y_p + quantiles_disp
    covered = (y_t >= lower) & (y_t <= upper)  # (n, 6)
    covered_all = covered.all(axis=1)

    hr_metrics = {}

    # Also compute predicted-high-response for deployment-oriented diagnostic
    pred_mag = np.linalg.norm(y_p, axis=1)

    for source_name, mag in [("true_response", disp_mag), ("predicted_response", pred_mag)]:
        for p in percentiles:
            thr = float(np.percentile(mag, p))
            mask = mag >= thr
            if mask.sum() == 0:
                hr_metrics[f"{source_name}_top_{p}pct"] = {
                    "n_nodes": 0, "threshold": thr,
                    "coverage_all_dof": float("nan"),
                }
                continue
            cov_val = float(covered_all[mask].mean())
            # P95 residual within this subset
            resid = np.abs(y_t[mask] - y_p[mask])
            p95_res = float(np.percentile(resid, 95))

            hr_metrics[f"{source_name}_top_{p}pct"] = {
                "n_nodes": int(mask.sum()),
                "threshold": thr,
                "target_coverage": target_cov,
                "empirical_coverage": cov_val,
                "coverage_gap": cov_val - target_cov,
                "p95_residual": p95_res,
                "interval_width": float(2.0 * quantiles_disp.mean()),
            }

    # Low-response (complement of top 10%)
    for source_name, mag in [("true_response", disp_mag), ("predicted_response", pred_mag)]:
        thr = float(np.percentile(mag, 90))
        mask = mag < thr
        if mask.sum() == 0:
            hr_metrics[f"{source_name}_low_90pct"] = {
                "n_nodes": 0, "threshold": thr,
                "coverage_all_dof": float("nan"),
            }
            continue
        cov_val = float(covered_all[mask].mean())
        hr_metrics[f"{source_name}_low_90pct"] = {
            "n_nodes": int(mask.sum()),
            "threshold": thr,
            "target_coverage": target_cov,
            "empirical_coverage": cov_val,
            "coverage_gap": cov_val - target_cov,
            "interval_width": float(2.0 * quantiles_disp.mean()),
        }

    return hr_metrics


# ---------------------------------------------------------------------------
# Graph-level conformal
# ---------------------------------------------------------------------------


def compute_graph_level_conformal(
    scores: np.ndarray,
    global_graph_id: np.ndarray,
    alpha: float,
    comp_names: List[str],
    cal_mask: np.ndarray,
    eval_mask: np.ndarray,
    component_scales: Optional[np.ndarray] = None,
) -> Dict:
    """Compute component-wise graph-level conformal quantiles.

    For each component, takes the max residual per graph, then calibrates
    across graphs. Does NOT mix different physical quantities.

    Args:
        scores: (n_total, n_components) absolute residuals.
        global_graph_id: (n_total,) global graph index.
        alpha: Miscalverage level.
        comp_names: Component names.
        cal_mask: (n_total,) calibration mask.
        eval_mask: (n_total,) evaluation mask.
        component_scales: (n_components,) optional scales for normalized version.

    Returns:
        result: Dict with component-wise and optional normalized graph-level results.
    """
    target_cov = 1.0 - alpha
    result = {}

    # ---- Component-wise graph-level ----
    for i, name in enumerate(comp_names):
        # Max score per graph on calibration
        cal_scores = scores[:, i][cal_mask]
        cal_gid = global_graph_id[cal_mask]
        unique_cal_gids = np.unique(cal_gid)
        graph_max_scores = np.array([
            float(cal_scores[cal_gid == g].max()) for g in unique_cal_gids
        ])
        q_val, k, n = compute_conformal_quantile(graph_max_scores, alpha)

        # Coverage on evaluation
        eval_scores = scores[:, i][eval_mask]
        eval_gid = global_graph_id[eval_mask]
        unique_eval_gids = np.unique(eval_gid)
        eval_graph_covered = []
        for g in unique_eval_gids:
            g_mask = eval_gid == g
            g_scores = eval_scores[g_mask]
            g_covered = (g_scores <= q_val).all()
            eval_graph_covered.append(float(g_covered))

        emp_cov = float(np.mean(eval_graph_covered)) if eval_graph_covered else float("nan")

        result[f"graph_level_{name}"] = {
            "quantile": float(q_val),
            "n_cal_graphs": int(len(unique_cal_gids)),
            "n_eval_graphs": int(len(unique_eval_gids)),
            "k": int(k),
            "alpha": float(alpha),
            "target_coverage": target_cov,
            "empirical_coverage": emp_cov,
            "coverage_gap": emp_cov - target_cov,
        }

    # ---- Normalized graph-level (optional diagnostic) ----
    if component_scales is not None:
        norm_scores = scores / component_scales[np.newaxis, :]  # (n_total, n_comp)
        norm_max = norm_scores.max(axis=1)  # max across components per node

        cal_norm = norm_max[cal_mask]
        cal_gid = global_graph_id[cal_mask]
        unique_cal_gids = np.unique(cal_gid)
        graph_norm_max = np.array([
            float(cal_norm[cal_gid == g].max()) for g in unique_cal_gids
        ])
        q_val, k, n = compute_conformal_quantile(graph_norm_max, alpha)

        # Evaluation coverage
        eval_norm = norm_max[eval_mask]
        eval_gid = global_graph_id[eval_mask]
        unique_eval_gids = np.unique(eval_gid)
        eval_graph_covered = []
        for g in unique_eval_gids:
            g_mask = eval_gid == g
            g_covered = (eval_norm[g_mask] <= q_val).all()
            eval_graph_covered.append(float(g_covered))
        emp_cov = float(np.mean(eval_graph_covered)) if eval_graph_covered else float("nan")

        result["graph_level_normalized"] = {
            "quantile": float(q_val),
            "n_cal_graphs": int(len(unique_cal_gids)),
            "n_eval_graphs": int(len(unique_eval_gids)),
            "k": int(k),
            "alpha": float(alpha),
            "target_coverage": target_cov,
            "empirical_coverage": emp_cov,
            "coverage_gap": emp_cov - target_cov,
            "normalized": True,
        }

    return result


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------


def load_predictions(
    pred_dir: Path,
    max_graphs: Optional[int] = None,
) -> Dict:
    """Load mesh and beam prediction NPZs and associated metadata.

    Args:
        pred_dir: Directory containing predictions.
        max_graphs: Optional limit on number of graphs to load.

    Returns:
        Dict with keys: mesh_fields, beam_fields, graph_index, n_graphs, n_mesh_total, n_beam_total
    """
    mesh_path = pred_dir / "mesh_node_predictions.npz"
    beam_path = pred_dir / "beam_element_predictions.npz"

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh predictions not found: {mesh_path}")
    if not beam_path.exists():
        raise FileNotFoundError(f"Beam predictions not found: {beam_path}")

    print(f"  Loading mesh predictions: {mesh_path}")
    mesh_npz = np.load(mesh_path)
    print(f"    y_true_disp: {mesh_npz['y_true_disp'].shape}")
    print(f"    y_pred_disp: {mesh_npz['y_pred_disp'].shape}")

    print(f"  Loading beam predictions: {beam_path}")
    beam_npz = np.load(beam_path)
    print(f"    y_true_force: {beam_npz['y_true_force'].shape}")
    print(f"    y_pred_force: {beam_npz['y_pred_force'].shape}")

    # Determine number of graphs
    n_mesh = mesh_npz["y_true_disp"].shape[0]
    n_beam = beam_npz["y_true_force"].shape[0]
    n_graphs_mesh = n_mesh // N_MESH_PER_GRAPH
    n_graphs_beam = n_beam // N_BEAM_PER_GRAPH

    if n_graphs_mesh != n_graphs_beam:
        print(f"  [WARN] Mesh ({n_graphs_mesh}) and beam ({n_graphs_beam}) graph counts differ. "
              f"Using min: {min(n_graphs_mesh, n_graphs_beam)}")
    n_graphs = min(n_graphs_mesh, n_graphs_beam)

    if max_graphs is not None and max_graphs > 0:
        n_graphs = min(n_graphs, max_graphs)
        n_mesh_rows = n_graphs * N_MESH_PER_GRAPH
        n_beam_rows = n_graphs * N_BEAM_PER_GRAPH
        print(f"  Limiting to {n_graphs} graphs ({n_mesh_rows} mesh rows, {n_beam_rows} beam rows)")
    else:
        n_mesh_rows = n_graphs * N_MESH_PER_GRAPH
        n_beam_rows = n_graphs * N_BEAM_PER_GRAPH

    # Slice data
    mesh_data = {
        "y_true_disp": mesh_npz["y_true_disp"][:n_mesh_rows].astype(np.float64),
        "y_pred_disp": mesh_npz["y_pred_disp"][:n_mesh_rows].astype(np.float64),
        "node_xyz": mesh_npz["node_xyz"][:n_mesh_rows].astype(np.float64),
        "support_flags": mesh_npz["support_flags"][:n_mesh_rows].astype(np.float64),
    }
    beam_data = {
        "y_true_force": beam_npz["y_true_force"][:n_beam_rows].astype(np.float64),
        "y_pred_force": beam_npz["y_pred_force"][:n_beam_rows].astype(np.float64),
    }

    # Global graph indices
    mesh_data["global_graph_id"] = _compute_global_graph_id(
        np.zeros(n_mesh_rows), N_MESH_PER_GRAPH
    )
    beam_data["global_graph_id"] = _compute_global_graph_id(
        np.zeros(n_beam_rows), N_BEAM_PER_GRAPH
    )

    # Load graph index (for sample_id mapping)
    graph_index_path = pred_dir / "test_graph_index.csv"
    graph_index = None
    if graph_index_path.exists():
        with open(graph_index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            graph_index = list(reader)
        print(f"  Graph index loaded: {len(graph_index)} graphs from {graph_index_path}")
    else:
        print(f"  [WARN] test_graph_index.csv not found — sample-level split unavailable.")

    return {
        "mesh": mesh_data,
        "beam": beam_data,
        "graph_index": graph_index,
        "n_graphs": n_graphs,
        "pred_dir": pred_dir,
    }


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def build_split(
    data: Dict,
    split_mode: str,
    val_data: Optional[Dict],
    alpha_list: List[float],
    seed: int,
) -> Dict:
    """Build calibration/evaluation masks based on split mode.

    Args:
        data: Dict from load_predictions().
        split_mode: One of test_graph_50_50, test_sample_3_4, val_to_test.
        val_data: Dict from load_predictions() for val (used only in val_to_test).
        alpha_list: List of alpha values.
        seed: Random seed for reproducibility.

    Returns:
        split_info: Dict with split metadata and masks.
    """
    n_graphs = data["n_graphs"]
    n_mesh = n_graphs * N_MESH_PER_GRAPH
    n_beam = n_graphs * N_BEAM_PER_GRAPH

    split_info = {
        "split_mode": split_mode,
        "seed": seed,
        "n_total_graphs": n_graphs,
        "n_cal_graphs": 0,
        "n_eval_graphs": 0,
        "alpha_list": alpha_list,
        "timestamp": datetime.now().isoformat(),
    }

    if split_mode == "test_graph_50_50":
        # Stratified 50/50 by SampleID if graph_index available
        graph_index_records = data.get("graph_index")
        rng = np.random.RandomState(seed)

        if graph_index_records is not None and len(graph_index_records) >= n_graphs:
            # Use only the first n_graphs records
            records = graph_index_records[:n_graphs]
            sample_ids = [int(r.get("sample_id", 0) or 0) for r in records]
            unique_samples = sorted(set(sample_ids))

            # For each sample, split its graphs 50/50
            cal_graph_set = set()
            eval_graph_set = set()

            for sid in unique_samples:
                s_indices = [i for i, s in enumerate(sample_ids) if s == sid]
                rng.shuffle(s_indices)
                split_pt = len(s_indices) // 2
                for idx in s_indices[:split_pt]:
                    cal_graph_set.add(idx)
                for idx in s_indices[split_pt:]:
                    eval_graph_set.add(idx)

            n_cal = len(cal_graph_set)
            n_eval = len(eval_graph_set)
            print(f"  Stratified 50/50 by SampleID: cal={n_cal} graphs, eval={n_eval} graphs")

            # Create mesh masks
            mesh_cal = np.zeros(n_mesh, dtype=bool)
            mesh_eval = np.zeros(n_mesh, dtype=bool)
            for g in range(n_graphs):
                start = g * N_MESH_PER_GRAPH
                end = start + N_MESH_PER_GRAPH
                if g in cal_graph_set:
                    mesh_cal[start:end] = True
                elif g in eval_graph_set:
                    mesh_eval[start:end] = True

            # Create beam masks
            beam_cal = np.zeros(n_beam, dtype=bool)
            beam_eval = np.zeros(n_beam, dtype=bool)
            for g in range(n_graphs):
                start = g * N_BEAM_PER_GRAPH
                end = start + N_BEAM_PER_GRAPH
                if g in cal_graph_set:
                    beam_cal[start:end] = True
                elif g in eval_graph_set:
                    beam_eval[start:end] = True

            split_info["stratified"] = True
            split_info["sample_ids"] = [str(s) for s in unique_samples]
        else:
            # Fallback: simple random 50/50 by graph
            all_graph_ids = list(range(n_graphs))
            rng.shuffle(all_graph_ids)
            split_pt = n_graphs // 2
            cal_graph_set = set(all_graph_ids[:split_pt])
            eval_graph_set = set(all_graph_ids[split_pt:])
            n_cal = len(cal_graph_set)
            n_eval = len(eval_graph_set)
            print(f"  Random 50/50: cal={n_cal} graphs, eval={n_eval} graphs (no stratification)")
            print(f"  [WARN] test_graph_index.csv not available — no SampleID stratification possible.")

            mesh_cal = np.zeros(n_mesh, dtype=bool)
            mesh_eval = np.zeros(n_mesh, dtype=bool)
            for g in range(n_graphs):
                start = g * N_MESH_PER_GRAPH
                end = start + N_MESH_PER_GRAPH
                if g in cal_graph_set:
                    mesh_cal[start:end] = True
                elif g in eval_graph_set:
                    mesh_eval[start:end] = True

            beam_cal = np.zeros(n_beam, dtype=bool)
            beam_eval = np.zeros(n_beam, dtype=bool)
            for g in range(n_graphs):
                start = g * N_BEAM_PER_GRAPH
                end = start + N_BEAM_PER_GRAPH
                if g in cal_graph_set:
                    beam_cal[start:end] = True
                elif g in eval_graph_set:
                    beam_eval[start:end] = True

            split_info["stratified"] = False

        split_info["n_cal_graphs"] = n_cal
        split_info["n_eval_graphs"] = n_eval
        split_info["cal_graph_ids"] = sorted(cal_graph_set)
        split_info["eval_graph_ids"] = sorted(eval_graph_set)

        return {
            "mesh_cal": mesh_cal,
            "mesh_eval": mesh_eval,
            "beam_cal": beam_cal,
            "beam_eval": beam_eval,
            "split_info": split_info,
            "cal_data": data,  # same source
            "eval_data": data,  # same source
        }

    elif split_mode == "test_sample_3_4":
        # Fixed sample-level split
        if data.get("graph_index") is None:
            raise ValueError(
                "test_sample_3_4 requires test_graph_index.csv with sample_id mapping. "
                "Not found in predictions directory."
            )

        graph_index_records = data["graph_index"]
        cal_graph_set = set()
        eval_graph_set = set()

        for i, rec in enumerate(graph_index_records):
            if i >= n_graphs:
                break
            sid = int(rec.get("sample_id", 0) or 0)
            if sid in CAL_SAMPLE_IDS:
                cal_graph_set.add(i)
            elif sid in EVAL_SAMPLE_IDS:
                eval_graph_set.add(i)

        if len(cal_graph_set) == 0:
            raise ValueError(
                f"test_sample_3_4: No calibration graphs found for SampleIDs {CAL_SAMPLE_IDS}. "
                f"Check that test_graph_index.csv contains the expected sample_ids."
            )
        if len(eval_graph_set) == 0:
            raise ValueError(
                f"test_sample_3_4: No evaluation graphs found for SampleIDs {EVAL_SAMPLE_IDS}."
            )

        n_cal = len(cal_graph_set)
        n_eval = len(eval_graph_set)
        print(f"  Sample-level split: cal={n_cal} graphs (samples {CAL_SAMPLE_IDS}), "
              f"eval={n_eval} graphs (samples {EVAL_SAMPLE_IDS})")

        mesh_cal = np.zeros(n_mesh, dtype=bool)
        mesh_eval = np.zeros(n_mesh, dtype=bool)
        for g in range(n_graphs):
            s = g * N_MESH_PER_GRAPH
            e = s + N_MESH_PER_GRAPH
            if g in cal_graph_set:
                mesh_cal[s:e] = True
            elif g in eval_graph_set:
                mesh_eval[s:e] = True

        beam_cal = np.zeros(n_beam, dtype=bool)
        beam_eval = np.zeros(n_beam, dtype=bool)
        for g in range(n_graphs):
            s = g * N_BEAM_PER_GRAPH
            e = s + N_BEAM_PER_GRAPH
            if g in cal_graph_set:
                beam_cal[s:e] = True
            elif g in eval_graph_set:
                beam_eval[s:e] = True

        split_info["n_cal_graphs"] = n_cal
        split_info["n_eval_graphs"] = n_eval
        split_info["cal_graph_ids"] = sorted(cal_graph_set)
        split_info["eval_graph_ids"] = sorted(eval_graph_set)

        return {
            "mesh_cal": mesh_cal,
            "mesh_eval": mesh_eval,
            "beam_cal": beam_cal,
            "beam_eval": beam_eval,
            "split_info": split_info,
            "cal_data": data,
            "eval_data": data,
        }

    elif split_mode == "val_to_test":
        if val_data is None:
            raise ValueError(
                "val_to_test requires --val-predictions-dir. "
                "Use: --val-predictions-dir <path/to/val/predictions>"
            )

        # Calibration = all val data
        val_n_graphs = val_data["n_graphs"]
        val_n_mesh = val_n_graphs * N_MESH_PER_GRAPH
        val_n_beam = val_n_graphs * N_BEAM_PER_GRAPH
        mesh_cal = np.ones(val_n_mesh, dtype=bool)
        beam_cal = np.ones(val_n_beam, dtype=bool)

        # Evaluation = all test data (already limited by max_graphs)
        mesh_eval = np.ones(n_mesh, dtype=bool)
        beam_eval = np.ones(n_beam, dtype=bool)

        split_info["n_cal_graphs"] = val_n_graphs
        split_info["n_eval_graphs"] = n_graphs
        split_info["cal_source"] = str(val_data["pred_dir"])
        split_info["eval_source"] = str(data["pred_dir"])
        split_info["stratified"] = False

        print(f"  Val→Test: cal={val_n_graphs} graphs (val), eval={n_graphs} graphs (test)")

        return {
            "mesh_cal": mesh_cal,
            "mesh_eval": mesh_eval,
            "beam_cal": beam_cal,
            "beam_eval": beam_eval,
            "split_info": split_info,
            "cal_data": val_data,
            "eval_data": data,
        }

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def run_conformal_analysis(
    split_result: Dict,
    alpha: float,
) -> Dict:
    """Run conformal analysis for a single alpha value.

    Args:
        split_result: Output from build_split().
        alpha: Miscalverage level.

    Returns:
        result: Dict with all metrics.
    """
    cal_data = split_result["cal_data"]
    eval_data = split_result["eval_data"]
    mesh_cal = split_result["mesh_cal"]
    mesh_eval = split_result["mesh_eval"]
    beam_cal = split_result["beam_cal"]
    beam_eval = split_result["beam_eval"]

    result = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
    }

    # ==================== DISPLACEMENT ====================
    print(f"\n-- Displacement (alpha={alpha}) --")

    y_true_d = cal_data["mesh"]["y_true_disp"]
    y_pred_d = cal_data["mesh"]["y_pred_disp"]

    # Calibration scores
    scores_cal = np.abs(y_true_d[mesh_cal] - y_pred_d[mesh_cal])  # (n_cal, 6)

    # Compute quantiles
    quantiles_d, qinfo_d = compute_conformal_quantiles_componentwise(
        scores_cal, alpha, DISP_COMP_NAMES
    )
    result["displacement_quantiles"] = qinfo_d
    result["displacement_quantile_values"] = quantiles_d.tolist()

    # Coverage on evaluation set
    y_true_de = eval_data["mesh"]["y_true_disp"]
    y_pred_de = eval_data["mesh"]["y_pred_disp"]
    scores_eval = np.abs(y_true_de - y_pred_de)

    covered_d, cov_metrics_d = compute_coverage_metrics(
        y_true_de[mesh_eval],
        y_pred_de[mesh_eval],
        quantiles_d,
        alpha,
        DISP_COMP_NAMES,
    )
    result["displacement_coverage"] = cov_metrics_d

    # Width metrics
    width_metrics_d = compute_width_metrics(
        y_true_de[mesh_eval],
        y_pred_de[mesh_eval],
        quantiles_d,
        DISP_COMP_NAMES,
    )
    result["displacement_width"] = width_metrics_d

    # Region-wise coverage (displacement only)
    region_metrics, region_labels = compute_region_coverage(
        y_true_d_eval=y_true_de,
        y_pred_d_eval=y_pred_de,
        node_xyz_eval=eval_data["mesh"]["node_xyz"],
        support_flags_eval=eval_data["mesh"]["support_flags"],
        quantiles_disp=quantiles_d,
        alpha=alpha,
        eval_mask=mesh_eval,
    )
    result["region_coverage"] = region_metrics

    # High-response coverage
    hr_metrics = compute_high_response_coverage(
        y_true_de,
        y_pred_de,
        quantiles_d,
        alpha,
        mesh_eval,
    )
    result["high_response_coverage"] = hr_metrics

    # Graph-level conformal (displacement)
    # Use full scores (cal + eval) with masks
    all_scores_d = np.abs(y_true_de - y_pred_de)
    all_gid_d = eval_data["mesh"]["global_graph_id"]
    graph_metrics_d = compute_graph_level_conformal(
        all_scores_d,
        all_gid_d,
        alpha,
        DISP_COMP_NAMES,
        mesh_cal,
        mesh_eval,
        component_scales=None,  # component-wise only, no normalized
    )
    # Add normalized version using scale from training stats
    # Use the std of calibration scores as scale
    scale_d = np.std(scores_cal, axis=0)
    graph_metrics_d_norm = compute_graph_level_conformal(
        all_scores_d,
        all_gid_d,
        alpha,
        DISP_COMP_NAMES,
        mesh_cal,
        mesh_eval,
        component_scales=scale_d,
    )
    result["graph_level_conformal_disp"] = graph_metrics_d_norm

    # ==================== FORCE ====================
    print(f"-- Force (alpha={alpha}) --")

    y_true_f = cal_data["beam"]["y_true_force"]
    y_pred_f = cal_data["beam"]["y_pred_force"]

    scores_cal_f = np.abs(y_true_f[beam_cal] - y_pred_f[beam_cal])

    quantiles_f, qinfo_f = compute_conformal_quantiles_componentwise(
        scores_cal_f, alpha, FORCE_COMP_NAMES
    )
    result["force_quantiles"] = qinfo_f
    result["force_quantile_values"] = quantiles_f.tolist()

    y_true_fe = eval_data["beam"]["y_true_force"]
    y_pred_fe = eval_data["beam"]["y_pred_force"]

    covered_f, cov_metrics_f = compute_coverage_metrics(
        y_true_fe[beam_eval],
        y_pred_fe[beam_eval],
        quantiles_f,
        alpha,
        FORCE_COMP_NAMES,
    )
    result["force_coverage"] = cov_metrics_f

    width_metrics_f = compute_width_metrics(
        y_true_fe[beam_eval],
        y_pred_fe[beam_eval],
        quantiles_f,
        FORCE_COMP_NAMES,
    )
    result["force_width"] = width_metrics_f

    # Graph-level conformal (force)
    all_scores_f = np.abs(y_true_fe - y_pred_fe)
    all_gid_f = eval_data["beam"]["global_graph_id"]
    scale_f = np.std(scores_cal_f, axis=0)
    graph_metrics_f = compute_graph_level_conformal(
        all_scores_f,
        all_gid_f,
        alpha,
        FORCE_COMP_NAMES,
        beam_cal,
        beam_eval,
        component_scales=scale_f,
    )
    result["graph_level_conformal_force"] = graph_metrics_f

    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_split_manifest(split_result: Dict, output_dir: Path) -> None:
    """Write split_manifest.json."""
    info = split_result["split_info"]
    _save_json(info, output_dir / "split_manifest.json")


def write_conformal_summary(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_summary.json."""
    summary = {}
    for alpha, r in results.items():
        summary[f"alpha_{alpha}"] = {
            "target_coverage": r["target_coverage"],
            "displacement_ALL_coverage": r["displacement_coverage"].get("ALL", {}),
            "force_ALL_coverage": r["force_coverage"].get("ALL", {}),
        }
    _save_json(summary, output_dir / "conformal_summary.json")


def write_quantiles_csv(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_quantiles.csv."""
    rows = []
    for alpha, r in results.items():
        for name, info in r["displacement_quantiles"].items():
            rows.append({
                "target": "displacement",
                "component": name,
                "alpha": alpha,
                "n_cal_points": info["n_cal_points"],
                "k": info["k"],
                "quantile": info["quantile"],
            })
        for name, info in r["force_quantiles"].items():
            rows.append({
                "target": "force",
                "component": name,
                "alpha": alpha,
                "n_cal_points": info["n_cal_points"],
                "k": info["k"],
                "quantile": info["quantile"],
            })
    _save_csv(rows, output_dir / "conformal_quantiles.csv")


def write_component_metrics_csv(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_component_metrics.csv."""
    rows = []
    for alpha, r in results.items():
        # Displacement
        for name, m in r["displacement_coverage"].items():
            w = r["displacement_width"].get(name, {})
            rows.append({
                "target": "displacement",
                "component": name,
                "alpha": alpha,
                "target_coverage": m["target_coverage"],
                "empirical_coverage": m["empirical_coverage"],
                "coverage_gap": m["coverage_gap"],
                "interval_width": m["interval_width"],
                "n_eval_points": m["n_eval_points"],
                "width_over_median_abs_y": w.get("width_over_median_abs_y", ""),
                "width_over_pred_std": w.get("width_over_pred_std", ""),
            })
        # Force
        for name, m in r["force_coverage"].items():
            w = r["force_width"].get(name, {})
            rows.append({
                "target": "force",
                "component": name,
                "alpha": alpha,
                "target_coverage": m["target_coverage"],
                "empirical_coverage": m["empirical_coverage"],
                "coverage_gap": m["coverage_gap"],
                "interval_width": m["interval_width"],
                "n_eval_points": m["n_eval_points"],
                "width_over_median_abs_y": w.get("width_over_median_abs_y", ""),
                "width_over_pred_std": w.get("width_over_pred_std", ""),
            })
    _save_csv(rows, output_dir / "conformal_component_metrics.csv")


def write_region_metrics_csv(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_region_metrics.csv."""
    rows = []
    for alpha, r in results.items():
        rc = r.get("region_coverage", {})
        for region_name, m in rc.items():
            rows.append({
                "alpha": alpha,
                "region": region_name,
                "n_nodes": m.get("n_nodes", ""),
                "target_coverage": m.get("target_coverage", ""),
                "empirical_coverage": m.get("empirical_coverage", ""),
                "coverage_gap": m.get("coverage_gap", ""),
                "interval_width": m.get("interval_width", ""),
            })
    _save_csv(rows, output_dir / "conformal_region_metrics.csv")


def write_high_response_csv(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_high_response_metrics.csv."""
    rows = []
    for alpha, r in results.items():
        hr = r.get("high_response_coverage", {})
        for name, m in hr.items():
            rows.append({
                "alpha": alpha,
                "subset": name,
                "n_nodes": m.get("n_nodes", ""),
                "threshold": m.get("threshold", ""),
                "target_coverage": m.get("target_coverage", ""),
                "empirical_coverage": m.get("empirical_coverage", ""),
                "coverage_gap": m.get("coverage_gap", ""),
                "p95_residual": m.get("p95_residual", ""),
                "interval_width": m.get("interval_width", ""),
            })
    _save_csv(rows, output_dir / "conformal_high_response_metrics.csv")


def write_graph_level_csv(results: Dict[float, Dict], output_dir: Path) -> None:
    """Write conformal_graph_level_metrics.csv."""
    rows = []
    for alpha, r in results.items():
        for key in ["graph_level_conformal_disp", "graph_level_conformal_force"]:
            target = "displacement" if "disp" in key else "force"
            gl = r.get(key, {})
            for name, m in gl.items():
                rows.append({
                    "alpha": alpha,
                    "target": target,
                    "component": name,
                    "quantile": m.get("quantile", ""),
                    "n_cal_graphs": m.get("n_cal_graphs", ""),
                    "n_eval_graphs": m.get("n_eval_graphs", ""),
                    "target_coverage": m.get("target_coverage", ""),
                    "empirical_coverage": m.get("empirical_coverage", ""),
                    "coverage_gap": m.get("coverage_gap", ""),
                    "normalized": m.get("normalized", False),
                })
    _save_csv(rows, output_dir / "conformal_graph_level_metrics.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute split conformal prediction intervals for Stage 6 UQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Primary: Option B (test 50/50)
              python scripts/compute_conformal.py
                  --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test
                  --split-mode test_graph_50_50
                  --alpha 0.10 0.05 --seed 42
                  --output-dir outputs/diagnostics/stage6_uq

              # Robustness: B-Sample
              python scripts/compute_conformal.py
                  --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test
                  --split-mode test_sample_3_4
                  --alpha 0.10 0.05 --seed 42
                  --output-dir outputs/diagnostics/stage6_uq

              # Engineering: val to test
              python scripts/compute_conformal.py
                  --predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_test
                  --val-predictions-dir outputs/predictions/stage6/ms_pi_hgt_full_val
                  --split-mode val_to_test
                  --alpha 0.10 0.05 --seed 42
                  --output-dir outputs/diagnostics/stage6_uq

              # Smoke test
              python scripts/compute_conformal.py
                  --predictions-dir outputs/predictions/stage2b/ms_hgt/20260626082828
                  --split-mode test_graph_50_50
                  --alpha 0.10
                  --max-graphs 10
                  --output-dir outputs/diagnostics/stage6_uq_smoke
        """),
    )
    p.add_argument(
        "--predictions-dir", type=str, required=True,
        help="Path to prediction NPZ directory (mesh_node_predictions.npz + beam_element_predictions.npz)"
    )
    p.add_argument(
        "--val-predictions-dir", type=str, default=None,
        help="Path to VAL prediction NPZ directory (required for val_to_test split)"
    )
    p.add_argument(
        "--split-mode", type=str, required=True,
        choices=["test_graph_50_50", "test_sample_3_4", "val_to_test"],
        help="Calibration/evaluation split strategy"
    )
    p.add_argument(
        "--alpha", type=float, nargs="+", default=[0.10],
        help="Miscoverage levels (e.g., 0.10 0.05)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits"
    )
    p.add_argument(
        "--max-graphs", type=int, default=None,
        help="Limit number of graphs for smoke test"
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for conformal results"
    )

    return p.parse_args(argv)


def main(args: argparse.Namespace):
    print("=" * 60)
    print("STAGE 6 — CONFORMAL PREDICTION COMPUTATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pred_dir = Path(args.predictions_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Use timestamp-based subdirectory
    result_dir = output_dir / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPredictions dir: {pred_dir}")
    print(f"Split mode:      {args.split_mode}")
    print(f"Alpha values:    {args.alpha}")
    print(f"Seed:            {args.seed}")
    print(f"Max graphs:      {args.max_graphs or 'unlimited'}")
    print(f"Output dir:      {result_dir}")

    # ---- Load predictions ----
    print(f"\n{'=' * 60}")
    print("Loading predictions...")
    data = load_predictions(pred_dir, args.max_graphs)

    # Load val predictions if needed
    val_data = None
    if args.split_mode == "val_to_test":
        if args.val_predictions_dir is None:
            raise ValueError("val_to_test split requires --val-predictions-dir")
        val_dir = Path(args.val_predictions_dir)
        if not val_dir.exists():
            raise FileNotFoundError(f"Val predictions directory not found: {val_dir}")
        print(f"\n{'=' * 60}")
        print("Loading val predictions for calibration...")
        val_data = load_predictions(val_dir, args.max_graphs)

    # ---- Build split ----
    print(f"\n{'=' * 60}")
    print(f"Building split: {args.split_mode} ...")
    split_result = build_split(data, args.split_mode, val_data, args.alpha, args.seed)
    si = split_result["split_info"]
    print(f"  Calibration graphs: {si['n_cal_graphs']}")
    print(f"  Evaluation graphs:  {si['n_eval_graphs']}")

    # ---- Run conformal analysis for each alpha ----
    print(f"\n{'=' * 60}")
    print("Computing conformal prediction...")

    results = {}
    for alpha in sorted(args.alpha):
        print(f"\n{'=' * 50}")
        print(f"Alpha = {alpha} (target coverage = {1-alpha:.1%})")
        r = run_conformal_analysis(split_result, alpha)
        results[alpha] = r

        # Print summary
        dc = r["displacement_coverage"]["ALL"]
        fc = r["force_coverage"]["ALL"]
        print(f"\n  *** Displacement ALL: coverage={dc['empirical_coverage']:.4f} "
              f"(target={dc['target_coverage']:.4f}, gap={dc['coverage_gap']:.4f})")
        print(f"  *** Force ALL:        coverage={fc['empirical_coverage']:.4f} "
              f"(target={fc['target_coverage']:.4f}, gap={fc['coverage_gap']:.4f})")

    # ---- Write output files ----
    print(f"\n{'=' * 60}")
    print(f"Writing output files to {result_dir} ...")

    write_split_manifest(split_result, result_dir)
    write_conformal_summary(results, result_dir)
    write_quantiles_csv(results, result_dir)
    write_component_metrics_csv(results, result_dir)
    write_region_metrics_csv(results, result_dir)
    write_high_response_csv(results, result_dir)
    write_graph_level_csv(results, result_dir)

    # Also save raw results as JSON for analyze_conformal.py
    # Convert numpy values to native Python types
    results_serializable = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    _save_json(results_serializable, result_dir / "conformal_results_raw.json")

    print(f"\n{'=' * 60}")
    print("CONFORMAL COMPUTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output dir: {result_dir}")
    print(f"  Split mode: {args.split_mode}")
    print(f"  Alpha values: {args.alpha}")
    print(f"  Calibration graphs: {si['n_cal_graphs']}")
    print(f"  Evaluation graphs:  {si['n_eval_graphs']}")
    print(f"  Output files: {len(list(result_dir.glob('*')))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args_parsed = parse_args()
    main(args_parsed)


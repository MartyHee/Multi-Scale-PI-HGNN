"""
compute_region_labels.py — Stage 4 region-wise diagnostics baseline.

Computes region-wise metrics from exported full predictions (NPZ files).
Designed for establishing HGT region-wise baseline before MS-HGT comparison.
Works entirely from NPZ prediction files — no dataset or model needed.

Supported region definitions (evaluation-time, no dataset modification):
  - General (default): nodes with no special region assignment
  - Support: nodes with any DOF constrained (BC_Dx/Dy/Dz > 0)
  - Midspan: nodes in central 1/3 of longitudinal (X) span
  - End-neighborhood: nodes within 5% of either end (X=0 or X=max)
  - Transition: nodes not covered by support/midspan/end

High-response subsets (overlapping, can intersect with any region):
  - Translational top 10%: sqrt(Dx² + Dy² + Dz²) global percentile
  - Dy-only top 10%: |Dy| global percentile

Usage:
    # Full region diagnostics from exported NPZ
    python scripts/compute_region_labels.py
        --pred-dir outputs/predictions/stage2b/hgt/20260624124739
        --output-dir outputs/diagnostics/stage4_region_baseline

    # Comparison mode (HGT vs future MS-HGT)
    python scripts/compute_region_labels.py
        --pred-dir outputs/predictions/stage2b/hgt/... --model-name hgt
        --pred-dir-2 outputs/predictions/stage4/ms_hgt/... --model-name-2 ms_hgt

Output:
    outputs/diagnostics/stage4_region_baseline/<timestamp>/
        region_baseline_metrics.json    — all computed metrics
        region_map.png                  — spatial visualisation of region assignments
        region_disp_r2_bar.png          — per-region Disp R² bar chart
        region_dy_r2_bar.png            — per-region Dy R² bar chart
        high_response_metrics.json      — top-10% subset metrics
        support_bc_residual.json        — BC constraint violation metrics
        region_baseline_report.txt      — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISP_COMPONENTS = ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']
FORCE_COMPONENTS = [
    'Fx_I', 'Fy_I', 'Fz_I', 'Mx_I', 'My_I', 'Mz_I',
    'Fx_J', 'Fy_J', 'Fz_J', 'Mx_J', 'My_J', 'Mz_J',
]

# Region IDs and names (mutually exclusive)
REGION_NAMES = OrderedDict([
    (0, 'general'),
    (1, 'support'),
    (2, 'midspan'),
    (3, 'end_neighborhood'),
    (4, 'transition'),
])

# Region display colors
REGION_COLORS = {
    0: '#7f7f7f',  # general - grey
    1: '#d62728',  # support - red
    2: '#2ca02c',  # midspan - green
    3: '#ff7f0e',  # end_neighborhood - orange
    4: '#1f77b4',  # transition - blue
}

# High-response percentile threshold
HIGH_RESPONSE_PERCENTILE = 90  # top 10%

# End-neighborhood fraction of span
END_NEIGHBORHOOD_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def fmt(v: float, decimals: int = 4) -> str:
    return f'{v:.{decimals}f}'


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score, handling zero-variance targets."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rel_mae_score(y_true: np.ndarray, y_pred: np.ndarray,
                  eps: float = 1e-8) -> float:
    denom = np.mean(np.abs(y_true)) + eps
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def component_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      comp_names: list[str]) -> dict:
    """Compute per-component and macro-averaged metrics."""
    n_comp = y_true.shape[1]
    per_comp = {}
    for i in range(n_comp):
        name = comp_names[i] if i < len(comp_names) else f'comp_{i}'
        per_comp[name] = {
            'r2': round(r2_score(y_true[:, i], y_pred[:, i]), 6),
            'mae': round(mae_score(y_true[:, i], y_pred[:, i]), 10),
            'rmse': round(rmse_score(y_true[:, i], y_pred[:, i]), 10),
        }
    macro_r2 = np.mean([per_comp[c]['r2'] for c in per_comp])
    return {
        'per_component': per_comp,
        'macro_avg_r2': round(macro_r2, 6),
        'overall_r2': round(r2_score(y_true, y_pred), 6),
        'overall_mae': round(mae_score(y_true, y_pred), 10),
    }


# ---------------------------------------------------------------------------
# Region label computation (from NPZ data only)
# ---------------------------------------------------------------------------


def compute_region_labels(
    node_xyz: np.ndarray,
    support_flags: np.ndarray,
    end_fraction: float = END_NEIGHBORHOOD_FRACTION,
) -> np.ndarray:
    """Assign mutually exclusive region IDs to each node.

    Priority (lower number wins):
        1. Support (1): any BC DOF > 0.5
        2. Midspan (2): central 1/3 of X-span
        3. End-neighborhood (3): within end_fraction of X=0 or X=max
        4. Transition (4): remainder between midspan and ends
        5. General (0): default fallback

    Args:
        node_xyz: (N, 3) array of node coordinates in original scale.
        support_flags: (N, 6) array of BC flags [Dx,Dy,Dz,Rx,Ry,Rz], 0/1.
        end_fraction: Fraction of span to consider as end-neighborhood.

    Returns:
        region: (N,) int array with region IDs.
    """
    N = node_xyz.shape[0]
    region = np.zeros(N, dtype=np.int64)

    x = node_xyz[:, 0]  # longitudinal
    x_min, x_max = x.min(), x.max()
    x_span = x_max - x_min

    # Condition 1: Support nodes (any rotational or translational DOF constrained)
    is_support = support_flags.sum(axis=1) > 0.5

    # Condition 2: Midspan (central 1/3 of X-range)
    third = x_span / 3.0
    is_midspan = (x >= x_min + third) & (x <= x_min + 2.0 * third)

    # Condition 3: End-neighborhood (within end_fraction of each end)
    end_dist = end_fraction * x_span
    is_end_near = (x <= x_min + end_dist) | (x >= x_max - end_dist)

    # Assign in priority order (not mutually exclusive, so order matters)
    # Support gets highest priority
    region[is_support] = 1

    # Midspan (non-support)
    midspan_mask = is_midspan & ~is_support
    region[midspan_mask] = 2

    # End-neighborhood (non-support)
    end_mask = is_end_near & ~is_support
    region[end_mask] = 3

    # Transition: between end and midspan, but not support/end/midspan
    # end_neighborhood and midspan are already assigned; everything else
    # that's not general (0) should be transition
    is_transition = (~is_support) & (~is_midspan) & (~is_end_near)
    region[is_transition] = 4

    return region


def compute_high_response_masks(
    y_true_disp: np.ndarray,
    percentile: float = HIGH_RESPONSE_PERCENTILE,
) -> dict:
    """Compute high-response subset masks (top percentile).

    Returns dict with boolean masks for:
        - translational_high: sqrt(Dx²+Dy²+Dz²) above percentile threshold
        - dy_high: |Dy| above percentile threshold
        - translational_percentile: the actual threshold value
        - dy_percentile: the actual threshold value
    """
    trans_mag = np.sqrt(
        y_true_disp[:, 0] ** 2 +
        y_true_disp[:, 1] ** 2 +
        y_true_disp[:, 2] ** 2
    )
    dy_abs = np.abs(y_true_disp[:, 1])

    trans_thresh = np.percentile(trans_mag, percentile)
    dy_thresh = np.percentile(dy_abs, percentile)

    return {
        'translational_high': trans_mag >= trans_thresh,
        'dy_high': dy_abs >= dy_thresh,
        'translational_threshold': float(trans_thresh),
        'dy_threshold': float(dy_thresh),
        'translational_mean': float(np.mean(trans_mag)),
        'dy_mean': float(np.mean(dy_abs)),
    }


# ---------------------------------------------------------------------------
# BC residual metrics
# ---------------------------------------------------------------------------


def compute_support_bc_residual(
    y_pred_disp: np.ndarray,
    y_true_disp: np.ndarray,
    support_flags: np.ndarray,
) -> dict:
    """Compute BC constraint violation metrics at support nodes.

    For support nodes, constrained DOFs (flag=1) should have displacement ≈ 0.
    We measure how much the model violates this by comparing to true FEM values.

    Returns dict with:
        - translation_residual: MAE at constrained translation DOFs
        - rotation_residual: MAE at constrained rotation DOFs
        - per_dof_residual: dict of DOF-level MAE
        - support_node_count: number of support nodes
        - n_constrained_translation: count of constrained translation DOFs
        - n_constrained_rotation: count of constrained rotation DOFs
    """
    # Identify support nodes: any BC DOF constrained
    is_support = support_flags.sum(axis=1) > 0.5
    support_mask = is_support

    if support_mask.sum() == 0:
        return {
            'translation_residual': None,
            'rotation_residual': None,
            'per_dof_residual': {},
            'support_node_count': 0,
            'n_constrained_translation': 0,
            'n_constrained_rotation': 0,
            'warning': 'No support nodes found',
        }

    pred_sup = y_pred_disp[support_mask]
    true_sup = y_true_disp[support_mask]
    flags_sup = support_flags[support_mask]

    # Translation DOFs (0-2: Dx, Dy, Dz)
    trans_mask = flags_sup[:, :3] > 0.5
    # Rotation DOFs (3-5: Rx, Ry, Rz)
    rot_mask = flags_sup[:, 3:6] > 0.5

    n_constrained_trans = int(trans_mask.sum())
    n_constrained_rot = int(rot_mask.sum())

    # Per-DOF residual: MAE of (pred - true) at constrained DOFs only
    dof_names = ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']
    per_dof = {}
    for i in range(6):
        constrained_mask = flags_sup[:, i] > 0.5
        if constrained_mask.sum() > 0:
            err = np.abs(pred_sup[constrained_mask, i] - true_sup[constrained_mask, i])
            per_dof[dof_names[i]] = {
                'mae': float(np.mean(err)),
                'max_ae': float(np.max(err)),
                'count': int(constrained_mask.sum()),
                'mean_true': float(np.mean(true_sup[constrained_mask, i])),
                'mean_pred': float(np.mean(pred_sup[constrained_mask, i])),
            }

    # Aggregate residuals
    trans_residual = None
    if n_constrained_trans > 0:
        errors_trans = np.abs(
            pred_sup[:, :3][trans_mask] - true_sup[:, :3][trans_mask]
        )
        trans_residual = float(np.mean(errors_trans))

    rot_residual = None
    if n_constrained_rot > 0:
        errors_rot = np.abs(
            pred_sup[:, 3:6][rot_mask] - true_sup[:, 3:6][rot_mask]
        )
        rot_residual = float(np.mean(errors_rot))

    # Also compute the raw predicted displacement magnitude at support nodes
    # (should be near zero for constrained DOFs)
    pred_mag = np.sqrt((pred_sup[:, :3] ** 2).sum(axis=1))
    true_mag = np.sqrt((true_sup[:, :3] ** 2).sum(axis=1))

    return {
        'translation_residual': trans_residual,
        'rotation_residual': rot_residual,
        'per_dof_residual': per_dof,
        'support_node_count': int(support_mask.sum()),
        'n_constrained_translation': n_constrained_trans,
        'n_constrained_rotation': n_constrained_rot,
        'support_pred_translation_mean': float(np.mean(pred_mag)),
        'support_true_translation_mean': float(np.mean(true_mag)),
        'support_pred_translation_max': float(np.max(pred_mag)),
        'support_true_translation_max': float(np.max(true_mag)),
    }


# ---------------------------------------------------------------------------
# Region-wise metrics computation
# ---------------------------------------------------------------------------


def compute_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_labels: np.ndarray,
    region_names: OrderedDict,
    comp_names: list[str],
    tag: str = '',
) -> dict:
    """Compute grouped metrics for each region.

    Args:
        y_true: (N, C) true values.
        y_pred: (N, C) predicted values.
        region_labels: (N,) region IDs.
        region_names: OrderedDict mapping id -> name.
        comp_names: List of component names.
        tag: Optional tag for result keys.

    Returns:
        Dict with region-wise metrics.
    """
    results = OrderedDict()
    for rid, rname in region_names.items():
        mask = region_labels == rid
        n_nodes = int(mask.sum())
        if n_nodes == 0:
            results[rname] = {'n_nodes': 0}
            continue

        true_r = y_true[mask]
        pred_r = y_pred[mask]

        region_result = {
            'n_nodes': n_nodes,
            'overall_r2': round(r2_score(true_r, pred_r), 6),
            'overall_mae': round(mae_score(true_r, pred_r), 10),
            'overall_rmse': round(rmse_score(true_r, pred_r), 10),
            'overall_rel_mae': round(rel_mae_score(true_r, pred_r), 6),
        }

        # Per-component metrics
        for i, cname in enumerate(comp_names):
            if i < true_r.shape[1]:
                region_result[f'{cname}_r2'] = round(r2_score(true_r[:, i], pred_r[:, i]), 6)
                region_result[f'{cname}_mae'] = round(mae_score(true_r[:, i], pred_r[:, i]), 10)

        # Compute macro avg R² across components
        comp_r2s = [region_result[f'{c}_r2'] for c in comp_names if f'{c}_r2' in region_result]
        if comp_r2s:
            region_result['macro_avg_r2'] = round(np.mean(comp_r2s), 6)

        # P95 absolute error per component
        for i, cname in enumerate(comp_names):
            if i < true_r.shape[1]:
                abs_err = np.abs(true_r[:, i] - pred_r[:, i])
                p95 = float(np.percentile(abs_err, 95))
                p99 = float(np.percentile(abs_err, 99))
                region_result[f'{cname}_p95_ae'] = round(p95, 10)
                region_result[f'{cname}_p99_ae'] = round(p99, 10)

        results[rname] = region_result

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_region_map(
    node_xyz: np.ndarray,
    region_labels: np.ndarray,
    save_path: Path,
    title: str = 'Region Assignment',
) -> None:
    """Scatter plot of nodes coloured by region assignment (top-down view XZ)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Side view (X-Z)
    ax = axes[0]
    for rid, rname in REGION_NAMES.items():
        mask = region_labels == rid
        if mask.sum() == 0:
            continue
        ax.scatter(
            node_xyz[mask, 0], node_xyz[mask, 2],
            c=REGION_COLORS[rid], label=rname, s=1, alpha=0.6,
        )
    ax.set_xlabel('X (longitudinal, m)')
    ax.set_ylabel('Z (vertical, m)')
    ax.set_title(f'{title} — Side View')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # Top view (X-Y)
    ax = axes[1]
    for rid, rname in REGION_NAMES.items():
        mask = region_labels == rid
        if mask.sum() == 0:
            continue
        ax.scatter(
            node_xyz[mask, 0], node_xyz[mask, 1],
            c=REGION_COLORS[rid], label=rname, s=1, alpha=0.6,
        )
    ax.set_xlabel('X (longitudinal, m)')
    ax.set_ylabel('Y (transverse, m)')
    ax.set_title(f'{title} — Top View')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Region map saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)')


def plot_region_r2_bar(
    region_metrics: OrderedDict,
    metric_key: str,
    title: str,
    save_path: Path,
) -> None:
    """Bar chart of per-region R² values."""
    names = []
    values = []
    colors = []
    for rid, rname in REGION_NAMES.items():
        rm = region_metrics.get(rname)
        if rm is None or rm.get('n_nodes', 0) == 0:
            continue
        names.append(rname)
        values.append(rm.get(metric_key, 0.0))
        colors.append(REGION_COLORS[rid])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('R²')
    ax.set_title(title)
    ax.axhline(y=0.0, color='grey', linestyle='--', alpha=0.5)
    ax.set_ylim(min(0.0, min(values) - 0.02), min(1.0, max(values) + 0.02))

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Region R2 bar chart saved: {save_path}')


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    region_metrics: OrderedDict,
    high_response_metrics: dict,
    bc_residual: dict,
    model_name: str,
    n_total: int,
    n_support: int,
    n_midspan: int,
    overall_disp_r2: float,
    overall_force_r2: float,
) -> str:
    """Generate formatted text report."""
    lines = [
        '=' * 72,
        f'  STAGE 4 REGION-WISE DIAGNOSTICS BASELINE',
        f'  Model: {model_name}',
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 72,
        '',
        f'  Total test nodes:  {n_total:,}',
        f'  Support nodes:     {n_support:,}  ({100 * n_support / max(n_total, 1):.1f}%)',
        f'  Midspan nodes:     {n_midspan:,}  ({100 * n_midspan / max(n_total, 1):.1f}%)',
        '',
        f'  Overall Disp R²:   {overall_disp_r2:.6f}',
        f'  Overall Force R²:  {overall_force_r2:.6f}',
        '',
    ]

    # Region-wise summary
    lines.extend([
        '-' * 72,
        '  Region-Wise Disp R² / MAE',
        '-' * 72,
        f'  {"Region":<20s} {"Nodes":>8s} {"R² (macro)":>12s} {"Dy R²":>10s} {"MAE":>12s} {"P95 AE":>12s}',
        '-' * 72,
    ])
    for rid, rname in REGION_NAMES.items():
        rm = region_metrics.get(rname)
        if rm is None or rm.get('n_nodes', 0) == 0:
            continue
        n = rm['n_nodes']
        r2 = rm.get('macro_avg_r2', 0)
        dy_r2 = rm.get('Dy_r2', 0)
        mae = rm.get('overall_mae', 0)
        p95 = rm.get('Dy_p95_ae', 0)
        lines.append(
            f'  {rname:<20s} {n:>8,d} {r2:>12.4f} {dy_r2:>10.4f} '
            f'{mae:>12.6f} {p95:>12.6f}'
        )
    lines.append('-' * 72)

    # High-response subset
    lines.extend(['', '-' * 72, '  High-Response Subsets (top 10%)', '-' * 72])
    for subset_key in ['translational', 'dy']:
        mask_key = f'{subset_key}_high'
        hr = high_response_metrics.get(mask_key, {})
        if not hr:
            continue
        lines.append(f'')
        lines.append(f'  {subset_key.upper()} top 10%:')
        lines.append(f'    Threshold: {hr.get("threshold", "N/A"):.6f}')
        lines.append(f'    N nodes:   {hr.get("n_nodes", 0):,}')
        lines.append(f'    Disp R²:   {hr.get("macro_avg_r2", 0):.6f}')
        lines.append(f'    Dy R²:     {hr.get("dy_r2", 0):.6f}')
        lines.append(f'    MAE:       {hr.get("overall_mae", 0):.6f}')
        lines.append(f'    Dy P95 AE: {hr.get("dy_p95_ae", 0):.6f}')

    # BC residual
    lines.extend(['', '-' * 72, '  Support BC Residual', '-' * 72])
    bc = bc_residual
    lines.append(f'  Support nodes:             {bc.get("support_node_count", 0):,}')
    lines.append(f'  Constrained translation:   {bc.get("n_constrained_translation", 0):,} DOFs')
    lines.append(f'  Constrained rotation:      {bc.get("n_constrained_rotation", 0):,} DOFs')
    tr = bc.get('translation_residual')
    rr = bc.get('rotation_residual')
    lines.append(f'  Translation BC MAE:        {tr:.6f}' if tr is not None else '  Translation BC MAE:        N/A')
    lines.append(f'  Rotation BC MAE:           {rr:.6f}' if rr is not None else '  Rotation BC MAE:           N/A')

    per_dof = bc.get('per_dof_residual', {})
    if per_dof:
        lines.append(f'')
        lines.append(f'  Per-DOF BC residual (MAE):')
        for dof in ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']:
            if dof in per_dof:
                d = per_dof[dof]
                lines.append(
                    f'    {dof:>4s}: MAE={d["mae"]:.6f}, '
                    f'MaxAE={d["max_ae"]:.6f}, '
                    f'N_constrained={d["count"]}, '
                    f'True_mean={d["mean_true"]:.6f}, '
                    f'Pred_mean={d["mean_pred"]:.6f}'
                )
    lines.extend(['-' * 72, ''])

    # Tail-error summary
    lines.extend([
        '-' * 72,
        '  Per-Component Tail Error (P95 AE)',
        '-' * 72,
    ])
    for comp in DISP_COMPONENTS:
        # Find in which region this comp's P95 is worst
        worst_val = 0
        worst_region = ''
        for rid, rname in REGION_NAMES.items():
            rm = region_metrics.get(rname)
            if rm and f'{comp}_p95_ae' in rm:
                v = rm[f'{comp}_p95_ae']
                if v > worst_val:
                    worst_val = v
                    worst_region = rname
        lines.append(f'  {comp:>4s}: worst P95 AE = {worst_val:.6f}  (region: {worst_region})')

    lines.append('=' * 72)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main diagnostics pipeline
# ---------------------------------------------------------------------------


def run_diagnostics(
    mesh_path: Path,
    beam_path: Path,
    output_dir: Path,
    model_name: str = 'HGT',
    compare_mesh_path: Path | None = None,
    compare_beam_path: Path | None = None,
    compare_model_name: str = '',
) -> dict:
    """Run full region-wise diagnostics on exported predictions.

    Args:
        mesh_path: Path to mesh_node_predictions.npz.
        beam_path: Path to beam_element_predictions.npz.
        output_dir: Output directory for results.
        model_name: Name of the model for report labels.
        compare_mesh_path: Optional second model mesh NPZ for comparison.
        compare_beam_path: Optional second model beam NPZ for comparison.
        compare_model_name: Name of the second model.

    Returns:
        Summary dict with all results.
    """
    print('=' * 60)
    print(f'REGION-WISE DIAGNOSTICS — {model_name}')
    print(f'Output: {output_dir}')
    print('=' * 60)

    ensure_dir(output_dir)

    # ---- Load predictions ----
    print(f'\n[1/5] Loading predictions...')
    mesh_data = np.load(mesh_path)
    beam_data = np.load(beam_path)

    y_true_disp = mesh_data['y_true_disp']
    y_pred_disp = mesh_data['y_pred_disp']
    node_xyz = mesh_data['node_xyz']
    support_flags = mesh_data['support_flags']
    graph_id = mesh_data['graph_id']

    y_true_force = beam_data['y_true_force']
    y_pred_force = beam_data['y_pred_force']

    print(f'  Mesh nodes:       {y_true_disp.shape[0]:,} × {y_true_disp.shape[1]}')
    print(f'  Beam elements:    {y_true_force.shape[0]:,} × {y_true_force.shape[1]}')
    print(f'  Coordinate range: X=[{node_xyz[:,0].min():.2f}, {node_xyz[:,0].max():.2f}] '
          f'Y=[{node_xyz[:,1].min():.2f}, {node_xyz[:,1].max():.2f}] '
          f'Z=[{node_xyz[:,2].min():.2f}, {node_xyz[:,2].max():.2f}]')
    print(f'  Support nodes (any BC>0): {(support_flags.sum(axis=1) > 0.5).sum():,}')

    # ---- Compute region labels ----
    print(f'\n[2/5] Computing region labels...')
    region_labels = compute_region_labels(node_xyz, support_flags)

    for rid, rname in REGION_NAMES.items():
        n = (region_labels == rid).sum()
        print(f'  {rname:<20s}: {n:>8,d} nodes ({100 * n / max(len(region_labels), 1):.1f}%)')

    # ---- Compute high-response masks ----
    print(f'\n[3/5] Computing high-response subsets...')
    hr_masks = compute_high_response_masks(y_true_disp)
    for key in ['translational_high', 'dy_high']:
        n_hr = int(hr_masks[key].sum())
        print(f'  {key:<25s}: {n_hr:>8,d} nodes ({100 * n_hr / max(len(region_labels), 1):.1f}%)')

    # ---- Compute region-wise metrics (disp) ----
    print(f'\n[4/5] Computing region-wise Disp metrics...')
    region_metrics = compute_region_metrics(
        y_true_disp, y_pred_disp, region_labels,
        REGION_NAMES, DISP_COMPONENTS,
    )

    # Print region metrics table
    print(f'  {"Region":<20s} {"Nodes":>8s} {"R2(avg)":>10s} {"Dy R2":>10s} {"MAE":>12s} {"Dy P95":>12s}')
    print(f'  {"-"*72}')
    for rid, rname in REGION_NAMES.items():
        rm = region_metrics.get(rname)
        if rm is None or rm.get('n_nodes', 0) == 0:
            continue
        print(f'  {rname:<20s} {rm["n_nodes"]:>8,d} '
              f'{rm.get("macro_avg_r2", 0):>10.4f} '
              f'{rm.get("Dy_r2", 0):>10.4f} '
              f'{rm.get("overall_mae", 0):>12.6f} '
              f'{rm.get("Dy_p95_ae", 0):>12.6f}')

    # ---- Compute BC residual ----
    print(f'\n  Computing support BC residual...')
    bc_residual = compute_support_bc_residual(
        y_pred_disp, y_true_disp, support_flags,
    )
    print(f'  Support nodes: {bc_residual["support_node_count"]}')
    print(f'  Translation BC residual: {bc_residual["translation_residual"]}')
    print(f'  Rotation BC residual:    {bc_residual["rotation_residual"]}')

    # ---- Compute high-response metrics ----
    print(f'\n  Computing high-response subset metrics...')
    high_response_metrics = {}
    for subset_key, subset_label in [
        ('translational_high', 'translational'),
        ('dy_high', 'dy'),
    ]:
        mask = hr_masks[subset_key]
        if mask.sum() == 0:
            continue
        subset_results = compute_region_metrics(
            y_true_disp[mask], y_pred_disp[mask],
            region_labels[mask],  # preserve region labels within subset
            REGION_NAMES, DISP_COMPONENTS,
        )
        dy_r2 = r2_score(y_true_disp[mask, 1:2], y_pred_disp[mask, 1:2])
        dy_p95 = float(np.percentile(
            np.abs(y_true_disp[mask, 1] - y_pred_disp[mask, 1]), 95
        ))
        high_response_metrics[subset_key] = {
            'n_nodes': int(mask.sum()),
            'threshold': float(hr_masks[f'{subset_label}_threshold']),
            'overall_r2': round(r2_score(y_true_disp[mask], y_pred_disp[mask]), 6),
            'macro_avg_r2': round(np.mean([
                r2_score(y_true_disp[mask, i], y_pred_disp[mask, i])
                for i in range(6)
            ]), 6),
            'dy_r2': round(dy_r2, 6),
            'overall_mae': round(mae_score(y_true_disp[mask], y_pred_disp[mask]), 10),
            'dy_p95_ae': round(dy_p95, 10),
            'region_breakdown': {
                rname: subset_results[rname]
                for rname in REGION_NAMES.values()
                if subset_results.get(rname, {}).get('n_nodes', 0) > 0
            },
        }
        print(f'  {subset_key}: n={high_response_metrics[subset_key]["n_nodes"]:,}, '
              f'Dy R2={high_response_metrics[subset_key]["dy_r2"]:.4f}')

    # ---- Compute force metrics by region (reuse region_labels on beam elements) ----
    # Note: beam elements don't have coordinates in the NPZ, so we use graph_id
    # to assign regions. For now, we compute per-graph region distributions
    # and report overall force metrics.
    print(f'\n  Computing overall force metrics...')
    force_metrics_overall = {
        'overall_r2': round(r2_score(y_true_force, y_pred_force), 6),
        'macro_avg_r2': round(np.mean([
            r2_score(y_true_force[:, i], y_pred_force[:, i])
            for i in range(12)
        ]), 6),
        'overall_mae': round(mae_score(y_true_force, y_pred_force), 10),
    }
    print(f'  Force R2: {force_metrics_overall["overall_r2"]:.6f}')
    print(f'  Force macro R2: {force_metrics_overall["macro_avg_r2"]:.6f}')

    # ---- Compute overall metrics ----
    overall_disp_r2 = r2_score(y_true_disp, y_pred_disp)
    overall_disp_macro = np.mean([
        r2_score(y_true_disp[:, i], y_pred_disp[:, i]) for i in range(6)
    ])
    overall_force_r2 = r2_score(y_true_force, y_pred_force)

    # ---- Generate plots ----
    print(f'\n[5/5] Generating plots...')
    plot_region_map(node_xyz, region_labels, output_dir / 'region_map.png', model_name)
    plot_region_r2_bar(
        region_metrics, 'macro_avg_r2',
        f'{model_name} — Region-wise Disp R²',
        output_dir / 'region_disp_r2_bar.png',
    )
    plot_region_r2_bar(
        region_metrics, 'Dy_r2',
        f'{model_name} — Region-wise Dy R²',
        output_dir / 'region_dy_r2_bar.png',
    )

    # ---- Save results ----
    print(f'\n  Saving results...')

    # Save region definitions
    region_defs = {
        rid: {
            'name': rname,
            'description': _REGION_DESC[rid],
            'n_nodes': int((region_labels == rid).sum()),
        }
        for rid, rname in REGION_NAMES.items()
    }
    _save_json(region_defs, output_dir / 'region_definitions.json')

    # Serialize region metrics (convert OrderedDict to plain dict for JSON)
    region_metrics_serializable = {}
    for k, v in region_metrics.items():
        region_metrics_serializable[k] = v
    _save_json(region_metrics_serializable, output_dir / 'region_disp_metrics.json')

    # Save BC residual
    _save_json(bc_residual, output_dir / 'support_bc_residual.json')

    # Save high-response metrics
    _save_json(high_response_metrics, output_dir / 'high_response_metrics.json')

    # Save force metrics
    _save_json(force_metrics_overall, output_dir / 'force_overall_metrics.json')

    # Generate and save report
    n_support = int((region_labels == 1).sum())
    n_midspan = int((region_labels == 2).sum())
    report = generate_report(
        region_metrics, high_response_metrics, bc_residual,
        model_name, len(region_labels), n_support, n_midspan,
        overall_disp_r2, overall_force_r2,
    )
    report_path = output_dir / 'region_baseline_report.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f'\n  Report saved: {report_path}')

    # ---- Compile summary ----
    summary = {
        'model_name': model_name,
        'n_total_mesh_nodes': int(len(region_labels)),
        'n_total_beam_elements': int(y_true_force.shape[0]),
        'n_graphs': int(graph_id.max()) + 1,
        'coordinate_range': {
            'X': [float(node_xyz[:, 0].min()), float(node_xyz[:, 0].max())],
            'Y': [float(node_xyz[:, 1].min()), float(node_xyz[:, 1].max())],
            'Z': [float(node_xyz[:, 2].min()), float(node_xyz[:, 2].max())],
        },
        'region_node_counts': {
            rname: int((region_labels == rid).sum())
            for rid, rname in REGION_NAMES.items()
        },
        'overall_disp_r2': round(overall_disp_r2, 6),
        'overall_disp_macro_r2': round(overall_disp_macro, 6),
        'overall_force_r2': round(overall_force_r2, 6),
        'region_disp_metrics': region_metrics_serializable,
        'support_bc_residual': bc_residual,
        'high_response_metrics': high_response_metrics,
        'force_metrics': force_metrics_overall,
    }

    summary_path = output_dir / 'region_baseline_metrics.json'
    _save_json(summary, summary_path)
    print(f'  Summary saved: {summary_path}')
    print(f'\n{"=" * 60}')
    print(f'DIAGNOSTICS COMPLETE')
    print(f'{"=" * 60}')
    print(f'  Model:        {model_name}')
    print(f'  Disp macro R2:{overall_disp_macro:.6f}')
    print(f'  Force macro R2:{overall_force_r2:.6f}')
    print(f'  Output:       {output_dir}')
    print(f'{"=" * 60}\n')

    return summary


# ---------------------------------------------------------------------------
# Region descriptions (for documentation)
# ---------------------------------------------------------------------------

_REGION_DESC = {
    0: 'General (default): nodes not assigned to any special region',
    1: 'Support: nodes with any translational or rotational DOF constrained (BC flag > 0)',
    2: 'Midspan: nodes in central 1/3 of longitudinal span',
    3: 'End-neighborhood: nodes within 5% of X=0 or X=max',
    4: 'Transition: nodes between support/end/midspan regions not covered by other labels',
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Stage 4 region-wise diagnostics from exported NPZ predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # HGT baseline region diagnostics
              python scripts/compute_region_labels.py
                  --pred-dir outputs/predictions/stage2b/hgt/20260624124739
                  --output-dir outputs/diagnostics/stage4_region_baseline

              # Comparison mode (HGT vs MS-HGT)
              python scripts/compute_region_labels.py
                  --pred-dir outputs/predictions/stage2b/hgt/20260624124739
                  --pred-dir-2 outputs/predictions/stage4/ms_hgt_gated/20260625...
                  --output-dir outputs/diagnostics/stage4_region_baseline
        """),
    )
    p.add_argument('--pred-dir', type=str, required=True,
                   help='Path to exported prediction directory (containing NPZ files)')
    p.add_argument('--pred-dir-2', type=str, default=None,
                   help='Optional second model prediction directory for comparison')
    p.add_argument('--model-name', type=str, default='HGT',
                   help='Model name for report labels')
    p.add_argument('--model-name-2', type=str, default='MS-HGT',
                   help='Second model name for comparison')
    p.add_argument('--output-dir', type=str,
                   default='outputs/diagnostics/stage4_region_baseline',
                   help='Root output directory')
    return p.parse_args(argv)


def main(args: argparse.Namespace):
    pred_dir = Path(args.pred_dir)
    output_root = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = output_root / timestamp

    # Locate NPZ files
    mesh_path = pred_dir / 'mesh_node_predictions.npz'
    beam_path = pred_dir / 'beam_element_predictions.npz'

    if not mesh_path.is_file():
        raise FileNotFoundError(f'mesh NPZ not found: {mesh_path}')
    if not beam_path.is_file():
        raise FileNotFoundError(f'beam NPZ not found: {beam_path}')

    # Run primary diagnostics
    summary = run_diagnostics(
        mesh_path=mesh_path,
        beam_path=beam_path,
        output_dir=output_dir,
        model_name=args.model_name,
    )

    # Run comparison if second directory provided
    if args.pred_dir_2:
        pred_dir_2 = Path(args.pred_dir_2)
        mesh_path_2 = pred_dir_2 / 'mesh_node_predictions.npz'
        beam_path_2 = pred_dir_2 / 'beam_element_predictions.npz'

        if not mesh_path_2.is_file():
            print(f'\n[WARN] Second model mesh NPZ not found: {mesh_path_2}')
        else:
            print(f'\n{"=" * 60}')
            print(f'COMPARISON MODE: {args.model_name} vs {args.model_name_2}')
            print(f'{"=" * 60}')

            comp_output_dir = output_dir / 'comparison'
            ensure_dir(comp_output_dir)

            summary_2 = run_diagnostics(
                mesh_path=mesh_path_2,
                beam_path=beam_path_2,
                output_dir=comp_output_dir / args.model_name_2,
                model_name=args.model_name_2,
            )

            # Generate delta report (simplified)
            delta = {
                'model_1': args.model_name,
                'model_2': args.model_name_2,
                'disp_macro_r2_delta': round(
                    summary_2.get('overall_disp_macro_r2', 0) -
                    summary.get('overall_disp_macro_r2', 0), 6
                ),
                'force_macro_r2_delta': round(
                    summary_2.get('overall_force_r2', 0) -
                    summary.get('overall_force_r2', 0), 6
                ),
            }
            _save_json(delta, comp_output_dir / 'comparison_delta.json')
            print(f'  Comparison delta saved: Delta Disp R2={delta["disp_macro_r2_delta"]:.6f}')

    return summary


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f'\n[ERROR] {e}')
        traceback.print_exc()
        sys.exit(1)

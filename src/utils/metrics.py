"""
metrics.py — Edge-level regression metrics.

All functions operate on flattened (N*D,) or per-component (N, D) tensors.

Computes:
  - MSE, MAE, Relative MAE (with stability epsilon), R²
  - Overall (flat), per-component, and macro-averaged scores.

Usage::
    pred = torch.randn(1000, 12)
    target = torch.randn(1000, 12)
    results = compute_all_metrics(pred, target)
    # results["macro_avg"]["mae"], results["per_component"]["r2"][0], etc.
"""

from __future__ import annotations

from typing import Dict

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error over all elements."""
    return float(((pred - target) ** 2).mean().item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error over all elements."""
    return float((pred - target).abs().mean().item())


def relative_mae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Relative MAE = MAE / mean(|target|) — normalised by target magnitude.

    Adds eps to denominator for stability when target mean ≈ 0.
    """
    denom = target.abs().mean().item()
    return float(((pred - target).abs().mean() / (denom + eps)).item())


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """R² = 1 - SS_res / SS_tot.

    SS_res = sum((pred - target)^2)
    SS_tot = sum((target - target_mean)^2)
    """
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + eps)).item())


def _per_component(fn, pred: torch.Tensor, target: torch.Tensor) -> list:
    """Apply a scalar metric function per component (column)."""
    results = []
    for c in range(pred.shape[1]):
        results.append(fn(pred[:, c], target[:, c]))
    return results


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> Dict:
    """Return nested dict of overall, per-component, and macro-averaged metrics.

    Args:
        pred: (N, D) prediction tensor (un-standardised / original scale).
        target: (N, D) ground-truth tensor.
        eps: denominator stability epsilon.

    Returns::
        {
            "overall": {"mse": ..., "mae": ..., "rel_mae": ..., "r2": ...},
            "per_component": {
                "mae": [...],       # length D
                "rel_mae": [...],   # length D
                "r2": [...],        # length D
            },
            "macro_avg": {
                "mae": ...,
                "rel_mae": ...,
                "r2": ...,
            },
        }
    """
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()

    # Overall (flat over all elements)
    overall = {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rel_mae": relative_mae(pred, target, eps),
        "r2": r2_score(pred, target, eps),
    }

    # Per-component
    per_mae = _per_component(lambda p, t: mae(p.unsqueeze(1), t.unsqueeze(1)), pred, target)
    per_relmae = _per_component(lambda p, t: relative_mae(p.unsqueeze(1), t.unsqueeze(1), eps), pred, target)
    per_r2 = _per_component(lambda p, t: r2_score(p.unsqueeze(1), t.unsqueeze(1), eps), pred, target)

    # Macro average (mean of per-component)
    macro_avg = {
        "mae": sum(per_mae) / len(per_mae) if per_mae else 0.0,
        "rel_mae": sum(per_relmae) / len(per_relmae) if per_relmae else 0.0,
        "r2": sum(per_r2) / len(per_r2) if per_r2 else 0.0,
    }

    return {
        "overall": overall,
        "per_component": {"mae": per_mae, "rel_mae": per_relmae, "r2": per_r2},
        "macro_avg": macro_avg,
    }

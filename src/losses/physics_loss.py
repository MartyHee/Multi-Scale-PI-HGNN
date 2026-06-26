"""
physics_loss.py — Physics-informed loss functions for Stage 5.

Provides BC (boundary condition) translation loss and structural link
translation consistency loss for the MS-PI-HGT model.

All losses operate in **standardised space** (same as the supervised
CombinedLoss) so that λ weights are directly comparable.

Reference:
    - Stage 5 Phase 1 Dry-Run Report (loss scale diagnostics)
    - ``inspect_physics_loss_scale.py``
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# mesh_node.x[:, 9:15] = [Dx_fix, Dy_fix, Dz_fix, Rx_fix, Ry_fix, Rz_fix]
SUPPORT_FLAG_START_IDX: int = 9
SUPPORT_FLAG_DIM: int = 6

# Index of translation DOFs within the 6 support-flag dimensions
TRANS_DOF_IDX = [0, 1, 2]  # Dx, Dy, Dz


# ---------------------------------------------------------------------------
# Support flag helpers
# ---------------------------------------------------------------------------

def get_support_flags(
    batch,
    scaler=None,
    feature_stats: Optional[Dict] = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Extract binary support flags from ``batch["mesh_node"]``.

    The raw flags live in ``mesh_node.x[:, 9:15]`` as standardised values.
    This function restores them to the original 0/1 scale and thresholds.

    Args:
        batch: A ``HeteroData`` or ``HeteroDataBatch``  with ``mesh_node.x``.
        scaler: ``HeteroFeatureScaler`` instance whose ``_stats`` dict contains
            ``"node:mesh_node:x_mean"`` and ``"node:mesh_node:x_std"``.
        feature_stats: Alternative to ``scaler`` — a dict with the same keys.
        threshold: Threshold for binarisation (default 0.5).

    Returns:
        ``(N, 6)`` binary float tensor (values 0.0 or 1.0).
    """
    x = batch["mesh_node"].x                                            # (N, 15)
    flags_std = x[:, SUPPORT_FLAG_START_IDX:SUPPORT_FLAG_START_IDX + SUPPORT_FLAG_DIM]

    # Resolve stats source
    stats = None
    if scaler is not None and hasattr(scaler, "_stats"):
        stats = scaler._stats
    elif feature_stats is not None:
        stats = feature_stats

    if stats is not None:
        mean_key = "node:mesh_node:x_mean"
        std_key = "node:mesh_node:x_std"
        flags_mean = stats[mean_key][SUPPORT_FLAG_START_IDX:SUPPORT_FLAG_START_IDX + SUPPORT_FLAG_DIM].to(flags_std.device)   # (6,)
        flags_std_vals = stats[std_key][SUPPORT_FLAG_START_IDX:SUPPORT_FLAG_START_IDX + SUPPORT_FLAG_DIM].to(flags_std.device)  # (6,)
        flags_restored = flags_std * flags_std_vals + flags_mean
    else:
        flags_restored = flags_std

    return (flags_restored > threshold).float()


# ---------------------------------------------------------------------------
# BC translation loss
# ---------------------------------------------------------------------------

def support_bc_translation_loss(
    pred_disp: torch.Tensor,
    y_disp: torch.Tensor,
    batch,
    scaler=None,
    feature_stats: Optional[Dict] = None,
    dof_indices: Optional[list] = None,
) -> Tuple[torch.Tensor, int]:
    """MSE on support-constrained translation DOFs.

    Only DOFs where the support flag > 0.5 enter the loss — unconstrained
    DOFs are masked out.

    Args:
        pred_disp: ``(N, 6)`` predicted displacement (standardised space).
        y_disp: ``(N, 6)`` ground-truth displacement (standardised space).
        batch: HeteroDataBatch with ``mesh_node.x``.
        scaler: ``HeteroFeatureScaler`` for flag restoration.
        feature_stats: Alternative dict for flag restoration.
        dof_indices: Which of the 6 flag dims to constrain.
            Default ``[0, 1, 2]`` (Dx, Dy, Dz — translation).

    Returns:
        ``(loss, n_constrained_dofs)`` where loss is a scalar tensor
        (0 if no constrained DOFs).
    """
    if dof_indices is None:
        dof_indices = TRANS_DOF_IDX

    support_flags = get_support_flags(batch, scaler=scaler, feature_stats=feature_stats)  # (N, 6)
    dof_mask = support_flags[:, dof_indices] > 0.5                                         # (N, len(dof_indices))
    n_constrained = dof_mask.sum().item()

    if n_constrained == 0:
        return torch.tensor(0.0, device=pred_disp.device), 0

    loss = F.mse_loss(
        pred_disp[:, dof_indices][dof_mask],
        y_disp[:, dof_indices][dof_mask],
        reduction="mean",
    )
    return loss, n_constrained


# ---------------------------------------------------------------------------
# Structural link translation consistency loss
# ---------------------------------------------------------------------------

def structural_link_translation_loss(
    pred_disp: torch.Tensor,
    batch,
    unique_undirected: bool = True,
) -> Tuple[torch.Tensor, int]:
    """MSE consistency loss across structural-link endpoints.

    Structural links are rigid — the predicted translation displacement of
    the source and destination ``mesh_node`` should be identical.  This loss
    penalises differences.

    Args:
        pred_disp: ``(N, 6)`` predicted displacement (standardised space).
        batch: HeteroDataBatch containing
            ``("mesh_node", "structural_link", "mesh_node")`` edge_index.
        unique_undirected: Deduplicate bidirectional edges before computing
            loss (default ``True``, since edges are stored i→j and j→i).

    Returns:
        ``(loss, n_edges)`` where loss is a scalar tensor
        (0 if no structural link edges).
    """
    edge_type = ("mesh_node", "structural_link", "mesh_node")

    ei_dict = batch.edge_index_dict
    if edge_type not in ei_dict:
        return torch.tensor(0.0, device=pred_disp.device), 0

    edge_index = batch[edge_type].edge_index  # (2, E)
    n_edges = edge_index.size(1)

    if n_edges == 0:
        return torch.tensor(0.0, device=pred_disp.device), 0

    src, dst = edge_index[0], edge_index[1]

    if unique_undirected:
        # Sort src/dst to deduplicate i→j and j→i pairs
        sorted_src = torch.minimum(src, dst)
        sorted_dst = torch.maximum(src, dst)
        unique_pairs = torch.unique(torch.stack([sorted_src, sorted_dst], dim=0), dim=1)
        src, dst = unique_pairs[0], unique_pairs[1]
        n_edges = src.size(0)

    # Translation DOFs only (:, :3)
    loss = F.mse_loss(
        pred_disp[src, :3],
        pred_disp[dst, :3],
        reduction="mean",
    )
    return loss, n_edges


# ---------------------------------------------------------------------------
# Combined physics loss entry point
# ---------------------------------------------------------------------------

def compute_physics_losses(
    pred_disp: torch.Tensor,
    y_disp: torch.Tensor,
    batch,
    lambda_bc: float = 0.0,
    lambda_link: float = 0.0,
    scaler=None,
    feature_stats: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Compute all active physics losses and return the weighted sum.

    When a λ is zero (or negative) the corresponding loss is skipped.

    Returns:
        ``(total_physics, loss_bc, loss_link, n_bc_dofs, n_link_edges)``:

        - **total_physics** — ``lambda_bc * loss_bc + lambda_link * loss_link``
          (scalar, 0 if none active).
        - **loss_bc** — unweighted BC loss (scalar, 0 if skipped).
        - **loss_link** — unweighted link loss (scalar, 0 if skipped).
        - **n_bc_dofs** — number of constrained BC DOFs (``int``).
        - **n_link_edges** — number of structural link edges used (``int``).
    """
    device = pred_disp.device
    loss_bc_t = torch.tensor(0.0, device=device)
    loss_link_t = torch.tensor(0.0, device=device)
    n_bc = 0
    n_link = 0

    if lambda_bc > 0:
        loss_bc_t, n_bc = support_bc_translation_loss(
            pred_disp, y_disp, batch,
            scaler=scaler, feature_stats=feature_stats,
        )

    if lambda_link > 0:
        loss_link_t, n_link = structural_link_translation_loss(
            pred_disp, batch,
            unique_undirected=True,
        )

    total = lambda_bc * loss_bc_t + lambda_link * loss_link_t
    return total, loss_bc_t, loss_link_t, n_bc, n_link

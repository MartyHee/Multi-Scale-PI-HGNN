"""
losses.py — Loss functions for Stage 2-A baseline training.

Provides:
  - ``CombinedLoss``: Weighted sum of displacement MSE + force MSE (or L1/SmoothL1).
  - ``LOSS_FACTORY``: Registry for base loss classes.

All losses operate in **standardised space** (model outputs are standardised
by ``HeteroFeatureScaler``).  Inverse transform to physical space is done
only at evaluation time in the trainer.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# ---------- base loss registry ----------

LOSS_FACTORY: Dict[str, type] = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


class CombinedLoss(nn.Module):
    """Weighted combination of displacement and force losses.

    ``total = lambda_disp * loss(pred_disp, target_disp)
            + lambda_force * loss(pred_force, target_force)``

    Args:
        loss_name: Name of base loss (``"mse"``, ``"l1"``, ``"smooth_l1"``).
        lambda_disp: Weight for displacement loss.
        lambda_force: Weight for force loss.
    """

    def __init__(
        self,
        loss_name: str = "mse",
        lambda_disp: float = 1.0,
        lambda_force: float = 1.0,
    ):
        super().__init__()
        loss_cls = LOSS_FACTORY.get(loss_name)
        if loss_cls is None:
            raise ValueError(
                f"Unknown loss '{loss_name}'. Options: {list(LOSS_FACTORY.keys())}"
            )
        self.base_loss = loss_cls()
        self.lambda_disp = lambda_disp
        self.lambda_force = lambda_force

    def forward(
        self,
        pred_disp: torch.Tensor,
        pred_force: torch.Tensor,
        target_disp: torch.Tensor,
        target_force: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Returns:
            ``(total_loss, loss_disp, loss_force)`` — all scalar tensors.
        """
        loss_disp = self.base_loss(pred_disp, target_disp)
        loss_force = self.base_loss(pred_force, target_force)
        total = self.lambda_disp * loss_disp + self.lambda_force * loss_force
        return total, loss_disp, loss_force

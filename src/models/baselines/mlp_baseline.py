"""
mlp_baseline.py — Non-graph MLP baseline for displacement + force prediction.

Architecture:

  - **Displacement head**: directly from ``mesh_node.x`` (15-dim) → 6-dim.
  - **Force head**: from ``beam_element.x`` (11-dim) concatenated with the
    mean of its two endpoint mesh_node features (15-dim), obtained via the
    ``belongs_to_beam`` edge index.  Total input = 11 + 15 = 26-dim → 12-dim.

``plate_element`` and ``structural_link`` do **not** participate in the MLP
forward pass.

Input: ``HeteroDataBatch`` (from ``torch_geometric.loader.DataLoader``).

Returns:
    Tuple of ``(pred_disp, pred_force)`` —

    - ``pred_disp``:  ``(total_mesh_nodes, 6)``
    - ``pred_force``: ``(total_beam_elements, 12)``
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from src.models.baselines.decoders import MLPHead


class MLPBaseline(nn.Module):
    """Non-graph MLP baseline.

    Args:
        mesh_feat_dim: Feature dimension of mesh nodes (default 15).
        beam_feat_dim: Feature dimension of beam elements (default 11).
        hidden_dims: Hidden-layer widths shared by both heads.
        dropout: Dropout rate.
        activation: Activation name (``"relu"``, ``"gelu"``, etc.).
        use_batch_norm: Whether to use batch normalisation.
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.mesh_feat_dim = mesh_feat_dim
        self.beam_feat_dim = beam_feat_dim

        # Displacement head: mesh_node.x (mesh_feat_dim) → 6
        self.disp_head = MLPHead(
            input_dim=mesh_feat_dim,
            output_dim=6,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        # Force head: beam_element.x (beam_feat_dim) +
        #            mean endpoint mesh features (mesh_feat_dim) = beam_feat_dim + mesh_feat_dim → 12
        self.force_head = MLPHead(
            input_dim=beam_feat_dim + mesh_feat_dim,
            output_dim=12,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: A ``HeteroDataBatch`` with keys ``mesh_node``,
                ``beam_element``, and edge type
                ``("mesh_node", "belongs_to_beam", "beam_element")``.

        Returns:
            ``(pred_disp, pred_force)`` tensors.
        """
        # ---- Displacement prediction ----
        mesh_x = batch["mesh_node"].x                     # (M_total, mesh_feat_dim)
        pred_disp = self.disp_head(mesh_x)                 # (M_total, 6)

        # ---- Force prediction with endpoint features ----
        beam_x = batch["beam_element"].x                   # (B_total, beam_feat_dim)

        # Gather endpoint mesh_node features via belongs_to_beam edge index
        edge_index_mb = batch["mesh_node", "belongs_to_beam", "beam_element"].edge_index
        # edge_index_mb[0] = mesh_node indices  (global across batch)
        # edge_index_mb[1] = beam_element indices  (global across batch)
        # Each beam element has exactly 2 endpoints → 2 rows per beam in edge_index

        endpoint_mesh_feats = mesh_x[edge_index_mb[0]]     # (num_edges, mesh_feat_dim)

        # Mean of two endpoints per beam element
        beam_endpoint_mean = scatter_mean(
            endpoint_mesh_feats,
            edge_index_mb[1],
            dim=0,
            dim_size=beam_x.shape[0],
        )                                                  # (B_total, mesh_feat_dim)

        # Concatenate beam features with endpoint context
        beam_input = torch.cat([beam_x, beam_endpoint_mean], dim=1)   # (B_total, 26)
        pred_force = self.force_head(beam_input)                       # (B_total, 12)

        return pred_disp, pred_force

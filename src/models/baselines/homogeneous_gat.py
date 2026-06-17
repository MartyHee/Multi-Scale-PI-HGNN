"""
homogeneous_gat.py — Homogeneous GAT baseline.

Architecture:
  1. ``HeteroToHomoAdapter`` — project each node type to shared ``hidden_dim``
     and build a single homogeneous ``edge_index``.
  2. Stack of ``GATConv`` layers (PyG) — standard multi-head graph attention
     that is *not* relation-type aware (ordinary node-level attention).
  3. ``MLPHead`` decoders — mask back to per-type hidden states and decode:
       - mesh_node states → 6-dim displacement
       - beam_element states → 12-dim force

Key difference from ``HomogeneousGCN``: GAT uses attention weights per edge
instead of the fixed normalized adjacency in GCN.

Reference:
  - Veličković et al., ICLR 2018. "Graph Attention Networks."
    https://arxiv.org/abs/1710.10903
  - PyG ``GATConv``:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html

Input: ``HeteroDataBatch`` (from ``torch_geometric.loader.DataLoader``).

Returns:
    Tuple of ``(pred_disp, pred_force)``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.baselines.hetero_to_homo_adapter import HeteroToHomoAdapter
from src.models.baselines.decoders import MLPHead


class HomogeneousGAT(nn.Module):
    """Homogeneous GAT baseline.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension (default 128).
        num_layers: Number of GATConv layers (default 3).
        gat_heads: Number of attention heads for intermediate layers.
        dropout: Dropout rate (default 0.2).
        activation: Activation name (default ``"relu"``).
        use_batch_norm: Whether decoder heads use BatchNorm.
        decoder_hidden_dims: Hidden dims for decoder MLP heads.
        use_type_embed: Whether to add type embeddings in the adapter.
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gat_heads: int = 4,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        use_type_embed: bool = True,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.gat_heads = gat_heads

        # Hetero → Homo conversion
        self.adapter = HeteroToHomoAdapter(
            mesh_feat_dim=mesh_feat_dim,
            beam_feat_dim=beam_feat_dim,
            plate_feat_dim=plate_feat_dim,
            hidden_dim=hidden_dim,
            use_type_embed=use_type_embed,
        )

        # GAT layers
        # Intermediate: hidden_dim → (hidden_dim // heads) each, concat → hidden_dim
        # Final:        hidden_dim → hidden_dim, single head, no concat
        head_dim = hidden_dim // gat_heads
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim if is_last else head_dim,
                    heads=1 if is_last else gat_heads,
                    concat=not is_last,
                    dropout=dropout,
                )
            )

        # Activation
        act_map = {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU(0.1)}
        self._activation = act_map.get(activation, nn.ReLU())

        # Decoders
        self.disp_decoder = MLPHead(
            input_dim=hidden_dim,
            output_dim=6,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.force_decoder = MLPHead(
            input_dim=hidden_dim,
            output_dim=12,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: ``HeteroDataBatch`` with the 3 node types and 5 edge types.

        Returns:
            ``(pred_disp, pred_force)``:
                - pred_disp:  ``(total_mesh_nodes, 6)``
                - pred_force: ``(total_beam_elements, 12)``
        """
        # Convert to homogeneous
        h_all, edge_index_homo, meta = self.adapter(batch)

        # GAT layers
        for conv in self.convs:
            h_all = conv(h_all, edge_index_homo)
            h_all = self._activation(h_all)
            h_all = F.dropout(h_all, p=self.dropout, training=self.training)

        # Mask back to per-type
        M, B = meta["M"], meta["B"]
        h_mesh = h_all[:M]
        h_beam = h_all[M : M + B]

        # Decode
        pred_disp = self.disp_decoder(h_mesh)
        pred_force = self.force_decoder(h_beam)

        return pred_disp, pred_force

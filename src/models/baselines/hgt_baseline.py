"""
hgt_baseline.py — Heterogeneous HGT baseline with typed attention.

Architecture:
  1. **Type-specific input projection**: Each node type (mesh_node, beam_element,
     plate_element) has an independent ``Linear`` projection to ``hidden_dim``.
  2. **HGT typed attention**: A stack of ``HGTConv`` layers (PyG) — each layer
     applies node-type-dependent and edge-type-dependent multi-head attention.
     Unlike RGCN (which uses per-edge-type convolution), HGT learns separate
     Query/Key/Value projections for each edge type with a relational bias,
     enabling the model to attend to different physical relations differently.
  3. **Per-node-type LayerNorm** (optional, enabled by default) after each layer.
  4. **Dual decoders**: ``MLPHead`` decoders map mesh_node hidden states → 6-dim
     displacement and beam_element hidden states → 12-dim force.

Key differences from ``HeteroRGCNBaseline``:
  - Uses **typed attention** (HGT) instead of typed convolution (SAGEConv).
  - Attention weights are computed per-edge-type, allowing the model to learn
    which relations matter most for each prediction task.
  - No ``edge_attr`` used — this is a standard typed-attention baseline.

Key differences from ``HomogeneousGAT``:
  - GAT applies the same attention to all edge types (merged).
  - HGT distinguishes edge types in both attention computation and message
    construction — edge-type-dependent QKV projections.

Reference:
  - **HGT**: Hu et al., WWW 2020. "Heterogeneous Graph Transformer."
    https://arxiv.org/abs/2003.01332
  - PyG ``HGTConv``:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HGTConv.html

Note:
  This is a **standard typed-attention baseline** — it does NOT use:
  - edge-attribute-aware gating (reserved for Ours)
  - physics-gated messages (reserved for Ours)
  - macro anchor graph / cross-scale fusion (reserved for Stage 4)

Input: ``HeteroDataBatch`` (from ``torch_geometric.loader.DataLoader``).

Returns:
    Tuple of ``(pred_disp, pred_force)`` —

    - ``pred_disp``:  ``(total_mesh_nodes, 6)``
    - ``pred_force``: ``(total_beam_elements, 12)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from src.models.baselines.decoders import MLPHead


# ---- Canonical metadata in hetero_graph_dataset_v2 ----

NODE_TYPES = ["mesh_node", "beam_element", "plate_element"]

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]

METADATA = (NODE_TYPES, EDGE_TYPES)


class HGTBaseline(nn.Module):
    """Heterogeneous HGT baseline with typed attention.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension (default 128).
        num_layers: Number of HGTConv layers (default 3).
        heads: Number of attention heads (default 4).  ``hidden_dim``
            must be divisible by ``heads``.
        dropout: Dropout rate after each activation (default 0.1).
        activation: Activation name (``"relu"``, ``"gelu"``, ``"elu"``).
        use_layer_norm: Whether to apply LayerNorm per node type after
            each layer (default True).
        decoder_hidden_dims: Hidden dims for decoder MLP heads
            (default ``[64, 32]``).
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.use_layer_norm = use_layer_norm

        # ---- Type-specific input projections ----
        self.mesh_encoder = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_encoder = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_encoder = nn.Linear(plate_feat_dim, hidden_dim)

        # ---- Activation ----
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self._activation = act_map.get(activation, nn.ReLU())

        # ---- HGTConv layers (typed multi-head attention) ----
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=METADATA,
                    heads=heads,
                )
            )

        # ---- Per-node-type LayerNorm (optional) ----
        if use_layer_norm:
            self.layer_norms = nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES
            })

        # ---- Decoders ----
        self.disp_decoder = MLPHead(
            input_dim=hidden_dim,
            output_dim=6,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=True,
        )
        self.force_decoder = MLPHead(
            input_dim=hidden_dim,
            output_dim=12,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=True,
        )

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on a ``HeteroDataBatch``.

        Args:
            batch: ``HeteroDataBatch`` with the 3 node types and 5 edge types.

        Returns:
            ``(pred_disp, pred_force)``:
                - pred_disp:  ``(total_mesh_nodes, 6)``
                - pred_force: ``(total_beam_elements, 12)``
        """
        # ---- 1. Type-specific projections to shared hidden_dim ----
        x_dict: Dict[str, torch.Tensor] = {
            "mesh_node": self.mesh_encoder(batch["mesh_node"].x),
            "beam_element": self.beam_encoder(batch["beam_element"].x),
            "plate_element": self.plate_encoder(batch["plate_element"].x),
        }

        # ---- 2. Build edge_index_dict from the HeteroDataBatch ----
        edge_index_dict = {}
        for et in EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. HGTConv typed attention layers ----
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Activation + dropout per node type
            x_dict = {
                k: self._activation(v) for k, v in x_dict.items()
            }
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }
            # Per-node-type layer norm
            if self.use_layer_norm:
                x_dict = {
                    k: self.layer_norms[k](v) for k, v in x_dict.items()
                }

        # ---- 4. Decode ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])       # (M_total, 6)
        pred_force = self.force_decoder(x_dict["beam_element"])  # (B_total, 12)

        return pred_disp, pred_force

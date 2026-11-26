"""Full-field query MLP: fixed-template complete-field encoding + query heads.

This is the information-sufficient non-graph control requested on 2026-11-26.
It is **not** the old 8-bin ``fullfield_mlp`` variant.

Field descriptor
----------------
For one graph, concatenate every encoded entity feature in the frozen dataset
order::

    [ mesh_node.x (N x 15), beam_element.x (B x 18), plate_element.x (P x 9) ]

The frozen order defines feature correspondence (entity i, channel c -> one
fixed input coordinate).  A single linear layer ``field_dim -> bottleneck_dim``
lets every raw channel enter the descriptor; there is no binning, sum, mean or
learned entity ID.  The descriptor is then mapped to one graph context vector.

Query heads
-----------
  * mesh node: ``[node.x (15), context]`` -> 6
  * beam:      ``[beam.x (18), endpoint mean coordinate (3), context]`` -> 12

What is deliberately absent
---------------------------
  * no relation/message passing, no structural-link semantics;
  * no ``design_id``, ``load_family``, sample id, response or target statistic;
  * no train-fitted PCA/autoencoder (no extra reconstruction-truncation variable).

Limits
------
The descriptor is order-sensitive by construction and is only defined for the
fixed template it was built for.  It is not permutation- or translation-
invariant, and no cross-topology claim is made.  The fixed order is a feature
correspondence, not a learnable case label.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1)}.get(name, nn.ReLU())


def _graph_vec(store, n: int, device):
    v = getattr(store, "batch", None)
    return torch.zeros(n, dtype=torch.long, device=device) if v is None else v


def _endpoint_coords(batch, node_x: torch.Tensor) -> torch.Tensor:
    ei = batch["mesh_node", "belongs_to_beam", "beam_element"].edge_index
    return scatter(node_x[ei[0], :3], ei[1], dim=0,
                   dim_size=batch["beam_element"].num_nodes, reduce="mean")


class FullFieldQueryMLPBaseline(nn.Module):
    def __init__(self, mesh_feat_dim=15, beam_feat_dim=18, plate_feat_dim=9,
                 field_dim=0, bottleneck_dim=16, context_dim=128,
                 hidden_dims: Optional[List[int]] = None, dropout=0.1,
                 activation="relu", use_batch_norm=True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if field_dim <= 0:
            raise ValueError("field_dim must be positive (per-template flattened size)")
        self.mesh_feat_dim = mesh_feat_dim
        self.beam_feat_dim = beam_feat_dim
        self.plate_feat_dim = plate_feat_dim
        self.field_dim = int(field_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.field_proj = nn.Linear(self.field_dim, self.bottleneck_dim)
        self.context = nn.Sequential(
            nn.Linear(self.bottleneck_dim, context_dim), _act(activation),
            nn.Dropout(dropout), nn.LayerNorm(context_dim),
        )
        self.node_head = MLPHead(mesh_feat_dim + context_dim, 6, hidden_dims,
                                 dropout, activation, use_batch_norm)
        self.beam_head = MLPHead(beam_feat_dim + 3 + context_dim, 12, hidden_dims,
                                 dropout, activation, use_batch_norm)

    def _fields(self, batch) -> torch.Tensor:
        node_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        plate_x = batch["plate_element"].x
        g_node = _graph_vec(batch["mesh_node"], node_x.shape[0], node_x.device)
        g_beam = _graph_vec(batch["beam_element"], beam_x.shape[0], beam_x.device)
        g_plate = _graph_vec(batch["plate_element"], plate_x.shape[0], plate_x.device)
        B = max(int(g_node.max().item()) + 1 if g_node.numel() else 1,
                int(g_beam.max().item()) + 1 if g_beam.numel() else 1,
                int(g_plate.max().item()) + 1 if g_plate.numel() else 1)
        fields = []
        for b in range(B):
            parts = [node_x[g_node == b].reshape(-1),
                     beam_x[g_beam == b].reshape(-1),
                     plate_x[g_plate == b].reshape(-1)]
            flat = torch.cat(parts)
            if flat.numel() != self.field_dim:
                raise RuntimeError(
                    f"full-field descriptor length {flat.numel()} != configured "
                    f"field_dim {self.field_dim}; wrong template or stale config")
            fields.append(flat)
        return torch.stack(fields, dim=0)

    def forward(self, batch):
        ctx = self.context(self.field_proj(self._fields(batch)))
        node_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        g_node = _graph_vec(batch["mesh_node"], node_x.shape[0], node_x.device)
        g_beam = _graph_vec(batch["beam_element"], beam_x.shape[0], beam_x.device)
        beam_coords = _endpoint_coords(batch, node_x)
        pred_disp = self.node_head(torch.cat([node_x, ctx[g_node]], dim=1))
        pred_force = self.beam_head(torch.cat([beam_x, beam_coords, ctx[g_beam]], dim=1))
        return pred_disp, pred_force

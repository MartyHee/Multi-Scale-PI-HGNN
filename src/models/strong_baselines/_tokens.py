"""Shared entity-token encoder for the attention-based strong baselines.

Transolver++ and LinearNO both operate on an unordered token set.  This module
turns the heterogeneous batch into one token sequence without using any graph
relation for message passing:

  * one linear projection per node type (mesh 15 / beam 18 / plate 9);
  * a learnable three-way type embedding;
  * a coordinate embedding built from public coordinates only
    (mesh node coordinates, beam endpoint mean, plate member centroid).

The graph relation indices are *not* used here.  Graph membership is used only
to keep attention/softmax inside one sample (no cross-graph mixing).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def _graph_vec(store, n: int, device):
    v = getattr(store, "batch", None)
    return torch.zeros(n, dtype=torch.long, device=device) if v is None else v


def _member_coords(batch, node_type: str, node_x: torch.Tensor) -> torch.Tensor:
    rel = ("mesh_node", "belongs_to_beam", "beam_element") \
        if node_type == "beam_element" else ("mesh_node", "belongs_to_plate", "plate_element")
    ei = batch[rel].edge_index
    return scatter(node_x[ei[0], :3], ei[1], dim=0,
                   dim_size=batch[node_type].num_nodes, reduce="mean")


class EntityTokens(nn.Module):
    """Project the three entity types into one token sequence."""

    def __init__(self, mesh_feat_dim=15, beam_feat_dim=18, plate_feat_dim=9,
                 hidden_dim=128, dropout=0.0):
        super().__init__()
        self.mesh_proj = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_proj = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_proj = nn.Linear(plate_feat_dim, hidden_dim)
        self.type_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.coord_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        mesh_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        plate_x = batch["plate_element"].x
        g_mesh = _graph_vec(batch["mesh_node"], mesh_x.shape[0], mesh_x.device)
        g_beam = _graph_vec(batch["beam_element"], beam_x.shape[0], beam_x.device)
        g_plate = _graph_vec(batch["plate_element"], plate_x.shape[0], plate_x.device)

        mesh_coords = mesh_x[:, :3]
        beam_coords = _member_coords(batch, "beam_element", mesh_x)
        plate_coords = _member_coords(batch, "plate_element", mesh_x)

        h_mesh = self.mesh_proj(mesh_x) + self.type_embed[0] + self.coord_mlp(mesh_coords)
        h_beam = self.beam_proj(beam_x) + self.type_embed[1] + self.coord_mlp(beam_coords)
        h_plate = self.plate_proj(plate_x) + self.type_embed[2] + self.coord_mlp(plate_coords)
        tokens = torch.cat([h_mesh, h_beam, h_plate], dim=0)
        graph = torch.cat([g_mesh, g_beam, g_plate], dim=0)
        return {
            "tokens": self.dropout(tokens), "graph": graph,
            "n_mesh": mesh_x.shape[0], "n_beam": beam_x.shape[0],
            "n_plate": plate_x.shape[0],
            "mesh_batch": g_mesh, "beam_batch": g_beam,
        }

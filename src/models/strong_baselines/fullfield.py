"""Full-field, non-graph baseline: x-binned field encoder + entity-query MLP.

Motivation
----------
The local MLP baseline sees only one node's 15 features (or one beam plus its
two endpoint means).  It cannot see remote loads or the rest of the design.
This baseline is the "same-task, full-field, no-graph" control:

  * every node load / support flag, every beam attribute and every plate
    attribute is encoded (no entity is dropped);
  * entities are sorted by their X coordinate *inside each graph* and pooled
    into ``n_bins`` span bins (masked mean + masked max per feature), so the
    field representation keeps a coarse spatial distribution instead of a
    single total sum;
  * a small MLP turns the field descriptor into one graph context vector;
  * each mesh node is queried with ``[its 15 features, context]`` and each beam
    with ``[its 18 features, endpoint mean coordinate (3), context]``.

What it deliberately does **not** use
-------------------------------------
  * no relation edges / message passing (the five graph relations are unused);
  * no structural_link attributes and no plate decoder;
  * no ``design_id``, ``load_family``, sample id or any response-derived
    statistic.

Scope warning
-------------
The X-binned pooling assumes a stable template span and a fixed entity order
*within one template*.  It is **not** cross-topology, permutation-invariant or
size-independent; no such claim is made.  Counting/sorting are deterministic
input-order operations, not a graph.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead

NODE_TYPES = ("mesh_node", "beam_element", "plate_element")


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1)}.get(name, nn.ReLU())


def _graph_vec(store, n: int, device):
    vec = getattr(store, "batch", None)
    if vec is None:
        return torch.zeros(n, dtype=torch.long, device=device)
    return vec


def _num_graphs(batch) -> int:
    v = getattr(batch["mesh_node"], "batch", None)
    if v is None:
        return 1
    return int(v.max().item()) + 1 if v.numel() else 1


def _entity_coords(batch, node_type: str, node_x: torch.Tensor) -> torch.Tensor:
    """Coarse coordinates for beams/plates from their member mesh nodes."""
    if node_type == "mesh_node":
        return node_x[:, :3]
    relation = ("mesh_node", "belongs_to_beam", "beam_element") \
        if node_type == "beam_element" else ("mesh_node", "belongs_to_plate", "plate_element")
    ei = batch[relation].edge_index
    n_members = batch[node_type].num_nodes
    return scatter(node_x[ei[0], :3], ei[1], dim=0, dim_size=n_members, reduce="mean")


def _span_bins(coords: torch.Tensor, graph: torch.Tensor, n_bins: int,
               xmin: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
    rel = (coords[:, 0] - xmin[graph]) / span[graph].clamp(min=1e-8)
    return (rel * n_bins).floor().clamp(0, n_bins - 1).long()


def _binned_stats(feat: torch.Tensor, bins: torch.Tensor, graph: torch.Tensor,
                  n_graphs: int, n_bins: int):
    """Masked mean and max of ``feat`` in [n_graphs, n_bins, D]."""
    device = feat.device
    key = graph * n_bins + bins
    size = n_graphs * n_bins
    dim = feat.shape[1]
    counts = torch.zeros(size, dtype=feat.dtype, device=device)
    counts.index_add_(0, key, torch.ones_like(key, dtype=feat.dtype))
    sums = torch.zeros(size, dim, dtype=feat.dtype, device=device)
    sums.index_add_(0, key, feat)
    mean = sums / counts.clamp(min=1.0)[:, None]
    mean = mean.view(n_graphs, n_bins, dim)

    neg_inf = torch.full((size, dim), float("-inf"), dtype=feat.dtype, device=device)
    neg_inf.scatter_reduce_(0, key[:, None].expand(-1, dim), feat,
                            reduce="amax", include_self=True)
    neg_inf = neg_inf.view(n_graphs, n_bins, dim)
    empty = (counts == 0).view(n_graphs, n_bins, 1).expand(-1, -1, dim)
    mx = torch.where(empty, torch.zeros_like(neg_inf), neg_inf)
    return mean, mx


class FullFieldMLPBaseline(nn.Module):
    """x-binned full-field context + per-entity query MLP.

    Args:
        mesh_feat_dim / beam_feat_dim / plate_feat_dim: encoded widths from the
            feature schema.
        n_bins: number of span bins (default 8).
        context_dim: graph-context width (default 128).
        hidden_dims: query-head hidden widths.
        encoder_hidden: width of the field encoder.
        dropout / activation / use_batch_norm: standard head options.
    """

    def __init__(self, mesh_feat_dim: int = 15, beam_feat_dim: int = 18,
                 plate_feat_dim: int = 9, n_bins: int = 8, context_dim: int = 128,
                 encoder_hidden: int = 256, hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1, activation: str = "relu",
                 use_batch_norm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.mesh_feat_dim = mesh_feat_dim
        self.beam_feat_dim = beam_feat_dim
        self.plate_feat_dim = plate_feat_dim
        self.n_bins = int(n_bins)

        width = 2 * self.n_bins * (mesh_feat_dim + beam_feat_dim + plate_feat_dim)
        width += mesh_feat_dim + beam_feat_dim + plate_feat_dim  # global means
        self.field_width = width
        self.encoder = nn.Sequential(
            nn.Linear(width, encoder_hidden), _act(activation),
            nn.Dropout(dropout), nn.Linear(encoder_hidden, context_dim), _act(activation),
            nn.LayerNorm(context_dim),
        )
        self.node_head = MLPHead(mesh_feat_dim + context_dim, 6, hidden_dims,
                                 dropout, activation, use_batch_norm)
        self.beam_head = MLPHead(beam_feat_dim + 3 + context_dim, 12, hidden_dims,
                                 dropout, activation, use_batch_norm)

    def _field_descriptor(self, batch) -> torch.Tensor:
        node_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        plate_x = batch["plate_element"].x
        g_node = _graph_vec(batch["mesh_node"], node_x.shape[0], node_x.device)
        g_beam = _graph_vec(batch["beam_element"], beam_x.shape[0], beam_x.device)
        g_plate = _graph_vec(batch["plate_element"], plate_x.shape[0], plate_x.device)
        B = max(int(g_node.max().item()) + 1 if g_node.numel() else 1,
                int(g_beam.max().item()) + 1 if g_beam.numel() else 1,
                int(g_plate.max().item()) + 1 if g_plate.numel() else 1)

        coords = {
            "mesh_node": node_x[:, :3],
            "beam_element": _entity_coords(batch, "beam_element", node_x),
            "plate_element": _entity_coords(batch, "plate_element", node_x),
        }
        xmin = scatter(coords["mesh_node"][:, 0], g_node, dim=0, dim_size=B, reduce="min")
        xmax = scatter(coords["mesh_node"][:, 0], g_node, dim=0, dim_size=B, reduce="max")
        span = (xmax - xmin).clamp(min=1e-8)

        parts = []
        for nt, feat, gvec in (("mesh_node", node_x, g_node),
                               ("beam_element", beam_x, g_beam),
                               ("plate_element", plate_x, g_plate)):
            bins = _span_bins(coords[nt], gvec, self.n_bins, xmin, span)
            mean, mx = _binned_stats(feat, bins, gvec, B, self.n_bins)
            parts.extend([mean.reshape(B, -1), mx.reshape(B, -1)])
        for feat, gvec in ((node_x, g_node), (beam_x, g_beam), (plate_x, g_plate)):
            pooled = scatter(feat, gvec, dim=0, dim_size=B, reduce="mean")
            parts.append(torch.nan_to_num(pooled, nan=0.0))
        return self.encoder(torch.cat(parts, dim=1))

    def forward(self, batch):
        ctx = self._field_descriptor(batch)
        node_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        g_node = _graph_vec(batch["mesh_node"], node_x.shape[0], node_x.device)
        g_beam = _graph_vec(batch["beam_element"], beam_x.shape[0], beam_x.device)
        beam_coords = _entity_coords(batch, "beam_element", node_x)
        pred_disp = self.node_head(torch.cat([node_x, ctx[g_node]], dim=1))
        pred_force = self.beam_head(torch.cat([beam_x, beam_coords, ctx[g_beam]], dim=1))
        return pred_disp, pred_force

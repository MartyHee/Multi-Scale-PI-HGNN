"""MeshGraphNets task adaptation (encode-process-decode, edge-aware).

Source / reading depth
----------------------
* Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, "Learning Mesh-Based
  Simulation with Graph Networks", ICLR 2021, arXiv:2010.03409.
* Project page: https://sites.google.com/view/meshgraphnets/home
* Reference implementation: https://github.com/deepmind/meshgraphnets
  (Apache-2.0; not downloaded).
  Read: the encoder / processor / decoder equations and the reported
  configuration (128 hidden, 15 processor blocks, residual MLPs; via the
  project page and web search snippets, 2026-11-26).

Adaptation to the static FE v2 task
-----------------------------------
  * encoder: one linear per node type and one edge encoder per relation
    (membership role one-hot or the 10-d structural-link attribute), plus a
    learned relation embedding;
  * processor: ``processor_steps`` message-passing blocks.  Per relation
    ``e' = MLP([h_src, h_dst, e]) + e``; node update is
    ``h' = MLP([h, sum_relation mean(e')]) + h``;
  * decoder: mesh -> 6 displacement components, beam -> 12 end-force
    components.  Plates are encoded/updated but not decoded.
  * **no previous-step response and no target is ever an input**; this is a
    one-shot static prediction, not a rollout, so the integrator/state updater
    of the original paper is intentionally absent.

Disclosed differences
---------------------
  * 6 processor steps (not 15) to keep the local smoke and server budget
    comparable; no world-space proximity edges; no collision/contact edges;
  * the five dataset relations replace the original mesh/world edge sets;
  * no original benchmark was reproduced, so no claim of matching the paper's
    numbers is made.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead

NODE_TYPES = ("mesh_node", "beam_element", "plate_element")
EDGE_TYPES = (
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
)
EDGE_KEYS = [f"{s}__{r}__{d}" for s, r, d in EDGE_TYPES]


def _link_edge_features(attr: torch.Tensor) -> torch.Tensor:
    """Bounded structural-link descriptor (no raw 1e15 stiffness).

    The frozen dataset stores rigid-link stiffness as 1e15 (constant inside a
    template), so feeding the raw 10 columns makes any edge MLP explode.  The
    adaptation keeps only bounded public information: beta as sin/cos, distance
    ratio, link type as three buckets, and the rigid flag.  The constant
    stiffness columns carry no within-template information and are dropped.
    """
    beta = torch.deg2rad(attr[:, 6])
    link_type = torch.round(attr[:, 8]).long()
    onehot = torch.stack([(link_type == 1), (link_type == 2),
                          ~torch.isin(link_type, torch.tensor([1, 2], device=link_type.device))],
                         dim=1).to(attr.dtype)
    return torch.cat([torch.sin(beta)[:, None], torch.cos(beta)[:, None],
                      attr[:, 7:8], onehot, attr[:, 9:10]], dim=1)  # 7 columns


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1)}.get(name, nn.ReLU())


class _ProcessorBlock(nn.Module):
    """One GraphNetBlock: per-edge-set edge MLP + per-node-type node MLP."""

    def __init__(self, hidden_dim, activation, dropout):
        super().__init__()
        act = _act(activation)
        self.edge_mlps = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim), act,
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            ) for key in EDGE_KEYS
        })
        self.node_mlps = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), act,
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            ) for nt in NODE_TYPES
        })
        self.dropout = dropout

    def forward(self, h, edges):
        agg = {nt: [] for nt in NODE_TYPES}
        new_edges = {}
        for i, et in enumerate(EDGE_TYPES):
            src, _, dst = et
            ei, e = edges[et]
            msg = self.edge_mlps[EDGE_KEYS[i]](
                torch.cat([h[src][ei[0]], h[dst][ei[1]], e], dim=1)) + e
            new_edges[et] = (ei, msg)
            # Official code uses unsorted_segment_sum (sum), not mean.
            agg[dst].append(scatter(msg, ei[1], dim=0,
                                    dim_size=h[dst].shape[0], reduce="sum"))
        new_h = {}
        for nt in NODE_TYPES:
            total = sum(agg[nt]) if agg[nt] else torch.zeros_like(h[nt])
            new_h[nt] = h[nt] + self.node_mlps[nt](torch.cat([h[nt], total], dim=1))
        h = {nt: F.dropout(v, p=self.dropout, training=self.training)
             for nt, v in new_h.items()}
        return h, new_edges


class MeshGraphNetBaseline(nn.Module):
    version = "meshgraphnet_core_20261126"

    def __init__(self, mesh_feat_dim=15, beam_feat_dim=18, plate_feat_dim=9,
                 hidden_dim=128, processor_steps=6, dropout=0.1,
                 activation="relu", decoder_hidden_dims: Optional[List[int]] = None,
                 edge_embed_dim: Optional[int] = None):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]
        self.hidden_dim = hidden_dim
        self.processor_steps = processor_steps
        self.dropout = dropout

        self.node_encoders = nn.ModuleDict({
            "mesh_node": nn.Linear(mesh_feat_dim, hidden_dim),
            "beam_element": nn.Linear(beam_feat_dim, hidden_dim),
            "plate_element": nn.Linear(plate_feat_dim, hidden_dim),
        })
        # edge attribute widths: membership role = 1, structural_link = 7
        self.edge_encoders = nn.ModuleDict({
            "mesh_node__belongs_to_beam__beam_element": nn.Linear(1, hidden_dim),
            "beam_element__rev_belongs_to_beam__mesh_node": nn.Linear(1, hidden_dim),
            "mesh_node__belongs_to_plate__plate_element": nn.Linear(1, hidden_dim),
            "plate_element__rev_belongs_to_plate__mesh_node": nn.Linear(1, hidden_dim),
            "mesh_node__structural_link__mesh_node": nn.Linear(7, hidden_dim),
        })
        self.relation_embed = nn.Parameter(torch.zeros(len(EDGE_TYPES), hidden_dim))
        # Official TF code constructs a GraphNetBlock inside the message-passing
        # loop; the parameter count (15 blocks) shows the processor weights are
        # NOT shared across steps.  Keep one block per step.
        self.processor = nn.ModuleList([
            _ProcessorBlock(hidden_dim, activation, dropout)
            for _ in range(processor_steps)
        ])
        self.disp_decoder = MLPHead(hidden_dim, 6, decoder_hidden_dims, dropout,
                                    activation, True)
        self.force_decoder = MLPHead(hidden_dim, 12, decoder_hidden_dims, dropout,
                                     activation, True)

    def forward(self, batch):
        h = {nt: self.node_encoders[nt](batch[nt].x) for nt in NODE_TYPES}
        edges = {}
        for i, et in enumerate(EDGE_TYPES):
            key = EDGE_KEYS[i]
            ei = batch[et].edge_index
            attr = batch[et].edge_attr
            if et == ("mesh_node", "structural_link", "mesh_node"):
                attr = _link_edge_features(attr)
            edges[et] = (ei, self.edge_encoders[key](attr) + self.relation_embed[i])

        for block in self.processor:
            h, edges = block(h, edges)

        return self.disp_decoder(h["mesh_node"]), self.force_decoder(h["beam_element"])

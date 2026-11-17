"""GraphSAGE and GIN homogeneous baselines (three-topology v2).

Mirror ``HomogeneousGCN``: project the three node types to a shared width with a
type embedding, merge the five relations into one homogeneous edge index, run
the standard homogeneous convolution stack, then decode mesh->6 and beam->12.

* GraphSAGE: ``SAGEConv`` (mean aggregator) — Hamilton et al., NeurIPS 2017.
* GIN: ``GINConv`` with an MLP — Xu et al., ICLR 2019 (sum aggregation, WL-like).

Neither uses relation types nor structural_link edge attributes (same
information as GCN/GAT after homogenisation).  "Adapted for this task": the
homogeneous projection of the heterogeneous graph is our adapter, not the
original papers' single-graph setting.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv

from src.models.baselines.decoders import MLPHead
from src.models.baselines.hetero_to_homo_adapter import HeteroToHomoAdapter


def _act(name):
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1)}.get(name, nn.ReLU())


class GraphSAGEBaseline(nn.Module):
    def __init__(self, mesh_feat_dim=15, beam_feat_dim=15, plate_feat_dim=6,
                 hidden_dim=128, num_layers=3, dropout=0.2, activation="relu",
                 use_batch_norm=True, decoder_hidden_dims=None, use_type_embed=True,
                 aggr="mean"):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]
        self.dropout = dropout
        self._activation = _act(activation)
        self.adapter = HeteroToHomoAdapter(mesh_feat_dim=mesh_feat_dim,
                                           beam_feat_dim=beam_feat_dim,
                                           plate_feat_dim=plate_feat_dim,
                                           hidden_dim=hidden_dim, use_type_embed=use_type_embed)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
                                    for _ in range(num_layers)])
        self.disp_decoder = MLPHead(hidden_dim, 6, decoder_hidden_dims, dropout, activation, use_batch_norm)
        self.force_decoder = MLPHead(hidden_dim, 12, decoder_hidden_dims, dropout, activation, use_batch_norm)

    def forward(self, batch):
        h, ei, meta = self.adapter(batch)
        for conv in self.convs:
            h = F.dropout(self._activation(conv(h, ei)), p=self.dropout, training=self.training)
        M, B = meta["M"], meta["B"]
        return self.disp_decoder(h[:M]), self.force_decoder(h[M:M + B])


class GINBaseline(nn.Module):
    def __init__(self, mesh_feat_dim=15, beam_feat_dim=15, plate_feat_dim=6,
                 hidden_dim=128, num_layers=3, dropout=0.2, activation="relu",
                 use_batch_norm=True, decoder_hidden_dims=None, use_type_embed=True,
                 train_eps=True):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]
        self.dropout = dropout
        self._activation = _act(activation)
        self.adapter = HeteroToHomoAdapter(mesh_feat_dim=mesh_feat_dim,
                                           beam_feat_dim=beam_feat_dim,
                                           plate_feat_dim=plate_feat_dim,
                                           hidden_dim=hidden_dim, use_type_embed=use_type_embed)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.disp_decoder = MLPHead(hidden_dim, 6, decoder_hidden_dims, dropout, activation, use_batch_norm)
        self.force_decoder = MLPHead(hidden_dim, 12, decoder_hidden_dims, dropout, activation, use_batch_norm)

    def forward(self, batch):
        h, ei, meta = self.adapter(batch)
        for conv in self.convs:
            h = F.dropout(self._activation(conv(h, ei)), p=self.dropout, training=self.training)
        M, B = meta["M"], meta["B"]
        return self.disp_decoder(h[:M]), self.force_decoder(h[M:M + B])

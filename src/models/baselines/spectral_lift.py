"""Spectral-lift homogeneous baseline (LiftGCN-adapted, faithful core).

ADAPTED FOR THIS TASK — NOT official LiftGCN code and NOT a reproduction.
Reference: Zeng & Wang, *LiftGCN: Efficient Energy-Preserving Graph Learning via
Joukowski Spectral Lifting for Finite Element Stress Prediction*, arXiv:2609.14977
(2026); official code github.com/ChenZeng001/LiftGCN (models.py::StressJoukowskiGCN).

Core operator kept verbatim (per the verified source):
    h0        = Dropout(GELU(LayerNorm(W_in x)))
    rho_c     = rho_max * tanh(theta_c),  rho_max = 0.99, init rho ~= 0.90
    R         = diag(rho_1..rho_d)
    Hbar_1    = A_hat h0 R
    Hbar_{l+1}= 2 A_hat H_l R - H_{l-1}          # Joukowski second-order recurrence
    H_{l+1}   = Hbar_{l+1} + alpha_l * GELU(LayerNorm(Hbar_{l+1}) W_l + b_l)
    y         = W_out [H_L, H_{L-1}]             # readout on the last two states
A_hat is the symmetric normalised adjacency WITHOUT self-loops (degree^{-1/2}).

Adaptation boundary (only these change vs the paper):
  * input: our heterogeneous 15-dim node features (incl. nodal loads) are
    projected by ``HeteroToHomoAdapter`` (loads ARE inputs here; original fixed
    loads per dataset);
  * graph: five heterogeneous relations merged into one homogeneous edge index;
  * output: ``Linear(2h, 6)`` for mesh displacement/rotation and ``Linear(2h, 12)``
    for beam-end force/moment (original: scalar von Mises stress).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.models.baselines.hetero_to_homo_adapter import HeteroToHomoAdapter


class SpectralLiftStack(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float = 0.1,
                 rho_max: float = 0.99, init_ratio: float = 0.90):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.rho_max = rho_max
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        # theta_c initialised so that rho_max * tanh(theta) == init_ratio
        theta0 = math.atanh(init_ratio / rho_max)
        self.theta = nn.Parameter(torch.full((hidden_dim,), theta0))
        self.res_w = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)])
        self.res_b = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(layers)])
        self.res_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        # residual gate alpha_l, init 0.1 (learnable)
        self.alpha = nn.Parameter(torch.full((layers,), 0.1))

    def _prop(self, x, edge_index, edge_weight, n):
        out = x.new_zeros((n, x.size(1)))
        src, dst = edge_index
        out.index_add_(0, dst, x[src] * edge_weight[:, None])
        return out

    def forward(self, h0, edge_index, edge_weight, n):
        h0 = F.dropout(F.gelu(self.in_norm(self.in_proj(h0))),
                       p=self.dropout, training=self.training)
        R = self.rho_max * torch.tanh(self.theta)          # (d,)
        # A_hat symmetric normalised, no self-loops
        edge_index, edge_weight = gcn_norm(edge_index, num_nodes=n,
                                           add_self_loops=False, dtype=h0.dtype)
        H_prev2 = h0
        Hbar = self._prop(h0, edge_index, edge_weight, n) * R
        H_cur = Hbar + self.alpha[0] * F.gelu(
            self.res_norm[0](Hbar) @ self.res_w[0].weight.T + self.res_b[0])
        states_prev, states_cur = h0, H_cur
        for l in range(1, self.layers):
            propagated = self._prop(states_cur, edge_index, edge_weight, n) * R
            Hbar = 2.0 * propagated - states_prev          # Joukowski recurrence
            H_new = Hbar + self.alpha[l] * F.gelu(
                self.res_norm[l](Hbar) @ self.res_w[l].weight.T + self.res_b[l])
            states_prev, states_cur = states_cur, H_new
        return torch.cat([states_cur, states_prev], dim=-1)  # (n, 2h)


class SpectralLiftBaseline(nn.Module):
    def __init__(self, mesh_feat_dim=15, beam_feat_dim=15, plate_feat_dim=6,
                 hidden_dim=128, num_layers=6, dropout=0.1, activation="relu",
                 use_batch_norm=True, decoder_hidden_dims=None, use_type_embed=True,
                 rho_max=0.99, init_ratio=0.90):
        super().__init__()
        self.dropout = dropout
        self.adapter = HeteroToHomoAdapter(mesh_feat_dim=mesh_feat_dim,
                                           beam_feat_dim=beam_feat_dim,
                                           plate_feat_dim=plate_feat_dim,
                                           hidden_dim=hidden_dim, use_type_embed=use_type_embed)
        self.stack = SpectralLiftStack(hidden_dim, num_layers, dropout, rho_max, init_ratio)
        # readout on concatenated last-two states -> Linear(2h, out) (paper's W_out)
        self.disp_out = nn.Linear(2 * hidden_dim, 6)
        self.force_out = nn.Linear(2 * hidden_dim, 12)

    def forward(self, batch):
        h, edge_index, meta = self.adapter(batch)
        # adapter returns homogeneous edge_index; gcn_norm is applied inside the stack
        z = self.stack(h, edge_index, None, h.size(0))
        M, B = meta["M"], meta["B"]
        return self.disp_out(z[:M]), self.force_out(z[M:M + B])

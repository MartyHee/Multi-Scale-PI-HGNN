"""LinearNO core adaptation (multi-head linear attention).

Official code cross-checked from ``HiPRL/LinearNO``
``Standard_PDE_Benchmark/model/LinearAttnNeuralOperator.py`` (fetched
2026-11-26 via jsDelivr; source copy in
``docs/strong_baseline_preparation_20261126/official_core/``).

Official ``LinearNO`` core:
  * ``in_project_x`` -> per-head tokens (B,H,N,dh);
  * per-head ``to_q``/``to_k``: dh -> key_ratio M (no bias);
  * ``q = softmax(q, dim=M)``, ``k = softmax(k, dim=N)``;
  * ``kv = einsum('bhnd,bhnc->bhdc', k, v)``, ``out = einsum('bhnd,bhdc->bhnc', q, kv)``;
  * concat heads, ``to_out = Linear(inner_dim, dim)+Dropout``;
  * optional ``LinearNO_temp`` variant divides q/k by learnable
    temperature_q/k clamped to [0.01, 1].

Local block uses pre-LN residual + FFN, same as the official block.  The
exploration config uses the base (no temperature) variant; ``use_temperature``
is available to reproduce the official temp variant.  Not implemented: the
official 2D conv embedding / PDE preprocess; this is the task adaptation.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.decoders import MLPHead
from src.models.strong_baselines._tokens import EntityTokens


class LinearNOAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: Optional[int] = None,
                 key_ratio: int = 64, dropout: float = 0.0,
                 use_temperature: bool = False):
        super().__init__()
        dim_head = int(dim_head or dim // heads)
        if dim_head * heads != dim:
            raise ValueError(f"dim_head*heads ({dim_head*heads}) != dim ({dim})")
        self.heads = heads
        self.dim_head = dim_head
        self.key_ratio = key_ratio
        self.use_temperature = use_temperature
        inner = heads * dim_head
        self.in_project_x = nn.Linear(dim, inner)
        self.to_q = nn.Linear(dim_head, key_ratio, bias=False)
        self.to_k = nn.Linear(dim_head, key_ratio, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        if use_temperature:
            self.temperature_q = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
            self.temperature_k = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()
        q = self.to_q(x_mid)
        k = self.to_k(x_mid)
        v = self.to_v(x_mid)
        if self.use_temperature:
            q = F.softmax(q / torch.clamp(self.temperature_q, 0.01, 1), dim=-1)
            k = F.softmax(k / torch.clamp(self.temperature_k, 0.01, 1), dim=-2)
        else:
            q = F.softmax(q, dim=-1)
            k = F.softmax(k, dim=-2)
        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        out = torch.einsum("bhnd,bhdc->bhnc", q, kv)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)


class LinearNOBlock(nn.Module):
    def __init__(self, dim: int, heads: int, key_ratio: int, dropout: float,
                 use_temperature: bool = False, mlp_ratio: float = 2.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = LinearNOAttention(dim, heads, dim // heads, key_ratio,
                                      dropout, use_temperature)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x).unsqueeze(0)).squeeze(0)
        x = x + self.mlp(self.ln_2(x))
        return x


class LinearNOBaseline(nn.Module):
    version = "linearno_core_20261126"

    def __init__(self, mesh_feat_dim=15, beam_feat_dim=18, plate_feat_dim=9,
                 hidden_dim=128, n_states=64, n_layers=3, heads=4,
                 dropout=0.1, use_temperature=False, mlp_ratio=2.0,
                 decoder_hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]
        self.tokenizer = EntityTokens(mesh_feat_dim, beam_feat_dim, plate_feat_dim,
                                      hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            LinearNOBlock(hidden_dim, heads, n_states, dropout,
                          use_temperature, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.disp_decoder = MLPHead(hidden_dim, 6, decoder_hidden_dims, dropout,
                                    "gelu", True)
        self.force_decoder = MLPHead(hidden_dim, 12, decoder_hidden_dims, dropout,
                                     "gelu", True)

    def forward(self, batch):
        tok = self.tokenizer(batch)
        x, graph = tok["tokens"], tok["graph"]
        out = torch.empty_like(x)
        for b in range(int(graph.max().item()) + 1 if graph.numel() else 1):
            sel = graph == b
            if not bool(sel.any()):
                continue
            h = x[sel]
            for blk in self.blocks:
                h = blk(h)
            out[sel] = h
        n_mesh, n_beam = tok["n_mesh"], tok["n_beam"]
        return (self.disp_decoder(out[:n_mesh]),
                self.force_decoder(out[n_mesh:n_mesh + n_beam]))

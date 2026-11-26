"""Transolver++ core adaptation (per-head slicing + mass denominator).

Official core cross-checked from ``thuml/Transolver_plus``
``models/Transolver_plus.py`` (MIT License, Copyright 2025 THUML @ Tsinghua
University), fetched 2026-11-26 via jsDelivr; the small source/licence copy is
in ``docs/strong_baseline_preparation_20261126/official_core/``.

Implemented core (matches official ``Physics_Attention_1D_Eidetic``):
  * ``in_project_x`` -> per-head tokens (B,H,N,dh);
  * per-head/per-token temperature ``clamp(proj_temperature(x)+bias, min=0.01)``;
  * per-head slice weights over ``slice_num`` via Gumbel-softmax;
  * mass denominator ``slice_norm = sum_N w`` and
    ``slice_token = sum_N w*x / (slice_norm+1e-5)``;
  * per-head slice Q/K/V and scaled dot-product attention;
  * deslice with the original (row-normalised) slice weights;
  * output projection ``Linear(inner_dim, dim)+Dropout``;
  * pre-LN residual + FFN block, MLP ratio configurable.

Deliberate choices / differences:
  * ``gumbel_mode='train_only'`` (exploration default) uses Gumbel noise in
    training and deterministic softmax at evaluation, so validation selection
    is reproducible.  Official code always samples Gumbel noise, including
    eval; ``gumbel_mode='always'`` reproduces that behaviour.
  * per-graph loop (no padding/mask); single-GPU; no million-scale distributed
    path, no checkpointing wrapper.
  * task heads/entity tokenizer are this project's adaptation, not the
    official PDE encoder.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.decoders import MLPHead
from src.models.strong_baselines._tokens import EntityTokens


def gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor, hard: bool = False) -> torch.Tensor:
    """Official helper: Gumbel noise, divide by per-element tau, softmax."""
    u = torch.rand_like(logits)
    noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + noise) / tau
    y = F.softmax(y, dim=-1)
    if hard:
        _, idx = y.max(dim=-1)
        hard_y = torch.zeros_like(y).scatter_(-1, idx.unsqueeze(-1), 1.0)
        y = (hard_y - y).detach() + y
    return y


class PhysicsAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: Optional[int] = None,
                 slice_num: int = 64, dropout: float = 0.0,
                 gumbel_mode: str = "train_only", ada_temp: bool = True):
        super().__init__()
        if gumbel_mode not in ("train_only", "always", "off"):
            raise ValueError(f"unknown gumbel_mode {gumbel_mode!r}")
        dim_head = int(dim_head or dim // heads)
        if dim_head * heads != dim:
            raise ValueError(f"dim_head*heads ({dim_head*heads}) != dim ({dim})")
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.gumbel_mode = gumbel_mode
        self.ada_temp = ada_temp
        inner = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num), nn.GELU(),
            nn.Linear(slice_num, 1), nn.GELU(),
        )
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                 # B,H,N,dh
        if self.ada_temp:
            temperature = torch.clamp(self.proj_temperature(x_mid) + self.bias, min=0.01)
        else:
            temperature = torch.ones(B, self.heads, N, 1, device=x.device, dtype=x.dtype)
        logits = self.in_project_slice(x_mid)                 # B,H,N,S
        use_gumbel = (self.gumbel_mode == "always" or
                      (self.gumbel_mode == "train_only" and self.training))
        if use_gumbel:
            w = gumbel_softmax(logits, temperature)
        else:
            w = torch.softmax(logits / temperature, dim=-1)
        slice_norm = w.sum(dim=2)                             # B,H,S (mass denominator)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, w)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, w)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, slice_num: int, dropout: float,
                 gumbel_mode: str, ada_temp: bool, mlp_ratio: float = 2.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, dim // heads, slice_num, dropout,
                                     gumbel_mode, ada_temp)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x).unsqueeze(0)).squeeze(0)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransolverBaseline(nn.Module):
    version = "transolver_plus_core_20261126"

    def __init__(self, mesh_feat_dim=15, beam_feat_dim=18, plate_feat_dim=9,
                 hidden_dim=128, n_slices=64, n_layers=3, heads=4,
                 dropout=0.1, ada_temp=True, gumbel_slices=False,
                 gumbel_mode=None, mlp_ratio=2.0,
                 decoder_hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]
        if gumbel_mode is None:
            gumbel_mode = "train_only" if gumbel_slices else "off"
        self.tokenizer = EntityTokens(mesh_feat_dim, beam_feat_dim, plate_feat_dim,
                                      hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            TransolverBlock(hidden_dim, heads, n_slices, dropout,
                            gumbel_mode, ada_temp, mlp_ratio)
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

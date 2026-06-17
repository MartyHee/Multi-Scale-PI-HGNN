"""
hetero_to_homo_adapter.py — Convert ``HeteroDataBatch`` → homogeneous
representation for GCN / GAT baselines.

Strategy:
  1. Each node type (mesh_node, beam_element, plate_element) has a
     **type-specific linear projection** to a shared ``hidden_dim``.
  2. A **learnable type embedding** is added per node type
     (optional, enabled via ``use_type_embed=True``).
  3. All nodes are concatenated into a single homogeneous node set.
  4. All edge types (forward + reverse membership, structural_link) are
     merged into a single ``edge_index`` with per-type offsets.
  5. The result is passed to standard GNN layers that are *not* aware of
     the original relation types.

The conversion is **relation-type agnostic** — this is intentional for the
homogeneous baseline to compare against relation-aware models later.

Usage inside a model's ``forward``::

    h_all, edge_index_homo, meta = adapter(batch)
    # h_all.shape = (M + B + P, hidden_dim)
    # meta["M"], meta["B"], meta["P"] for unmasking
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class HeteroToHomoAdapter(nn.Module):
    """Convert a ``HeteroDataBatch`` to homogeneous node features + edge index.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension after projection.
        use_type_embed: If True, add a learnable embedding per node type.
        type_embed_dim: Dimension of type embedding (defaults to ``hidden_dim``).
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        use_type_embed: bool = True,
        type_embed_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_type_embed = use_type_embed

        if type_embed_dim is None:
            type_embed_dim = hidden_dim

        # Type-specific projections
        self.mesh_proj = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_proj = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_proj = nn.Linear(plate_feat_dim, hidden_dim)

        # Learnable type embeddings
        if use_type_embed:
            self.type_embed = nn.Parameter(torch.randn(3, type_embed_dim) * 0.1)

    def forward(
        self, batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Convert a HeteroDataBatch to homogeneous representation.

        Args:
            batch: ``HeteroDataBatch`` with node_types
                ``["mesh_node", "beam_element", "plate_element"]``
                and the 5 canonical edge types.

        Returns:
            ``(h_all, edge_index_homo, meta)`` where::

                - h_all:           ``(M+B+P, hidden_dim)`` unified node features
                - edge_index_homo: ``(2, total_edges)`` unified edge indices
                - meta:            ``dict`` with ``M``, ``B``, ``P`` counts
        """
        mesh_x = batch["mesh_node"].x       # (M, mesh_feat_dim)
        beam_x = batch["beam_element"].x    # (B, beam_feat_dim)
        plate_x = batch["plate_element"].x  # (P, plate_feat_dim)

        M, B, P = mesh_x.shape[0], beam_x.shape[0], plate_x.shape[0]

        # ---- Type-specific projections ----
        h_mesh = self.mesh_proj(mesh_x)     # (M, hidden_dim)
        h_beam = self.beam_proj(beam_x)     # (B, hidden_dim)
        h_plate = self.plate_proj(plate_x)  # (P, hidden_dim)

        # ---- Add type embeddings ----
        if self.use_type_embed:
            h_mesh = h_mesh + self.type_embed[0].unsqueeze(0)
            h_beam = h_beam + self.type_embed[1].unsqueeze(0)
            h_plate = h_plate + self.type_embed[2].unsqueeze(0)

        # ---- Concatenate all node features ----
        h_all = torch.cat([h_mesh, h_beam, h_plate], dim=0)  # (M+B+P, hidden_dim)

        # ---- Build homogeneous edge index with per-type offsets ----
        beam_offset = M
        plate_offset = M + B

        edge_list = []

        # 1. belongs_to_beam: mesh → beam
        ei_mb = batch["mesh_node", "belongs_to_beam", "beam_element"].edge_index
        edge_list.append(torch.stack([ei_mb[0], ei_mb[1] + beam_offset], dim=0))

        # 2. rev_belongs_to_beam: beam → mesh
        ei_bm = batch["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_index
        edge_list.append(torch.stack([ei_bm[0] + beam_offset, ei_bm[1]], dim=0))

        # 3. belongs_to_plate: mesh → plate
        ei_mp = batch["mesh_node", "belongs_to_plate", "plate_element"].edge_index
        edge_list.append(torch.stack([ei_mp[0], ei_mp[1] + plate_offset], dim=0))

        # 4. rev_belongs_to_plate: plate → mesh
        ei_pm = batch["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_index
        edge_list.append(torch.stack([ei_pm[0] + plate_offset, ei_pm[1]], dim=0))

        # 5. structural_link: mesh → mesh (already homogeneous)
        ei_sl = batch["mesh_node", "structural_link", "mesh_node"].edge_index
        edge_list.append(ei_sl)

        edge_index_homo = torch.cat(edge_list, dim=1)

        meta = {
            "M": M,
            "B": B,
            "P": P,
            "beam_offset": beam_offset,
            "plate_offset": plate_offset,
        }

        return h_all, edge_index_homo, meta

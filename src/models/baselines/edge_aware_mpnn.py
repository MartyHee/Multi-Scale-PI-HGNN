"""
edge_aware_mpnn.py — Edge-aware MPNN (MeshGraphNet-style) baseline.

Purpose (paper-upgrade Stage Paper-2B):
  A modern neural-simulator-style baseline. Responds to the reviewer concern
  "you only compare against old baseline models". Named conservatively: this is
  an ADAPTED, heterogeneous, typed-message version inspired by MeshGraphNets /
  GNS — NOT a full reproduction of DeepMind's MeshGraphNets.

Architecture (encoder → processor → decoder):
  1. **Type-specific node encoders**: mesh_node 15 -> H, beam_element 11 -> H,
     plate_element 6 -> H. Optional per-node-type learnable embedding added to
     the projected features (``use_type_embedding``).
  2. **Relation-specific edge encoding**:
       - ``structural_link`` carries 10-dim ``edge_attr`` (stiffness features);
         encoded via ``EdgeEncoderMLP`` and concatenated into the message
         (``use_edge_attr``).
       - membership edges (4 types) have no ``edge_attr``; their message is
         built from sender + receiver hidden states.
  3. **Processor** (``num_layers`` residual message-passing blocks):
         msg_et   = msg_MLP_et(cat(x_src, x_dst[, edge_feat]))   # per relation
         aggr_dst = scatter_sum(msg_et, dst_idx)                  # per destination
         x_nt     = x_nt + update_MLP_nt(cat(x_nt, aggr_nt))      # residual update
         activation -> dropout -> LayerNorm
     The per-relation message MLPs are SHARED across layers (5 in total), so
     depth adds compute, not parameters — keeps the capacity comparable to
     MS-HGT.
  4. **Decoders**: MLPHead mesh_node -> disp(6), beam_element -> force(12).

Differences from a full MeshGraphNets reproduction (must be stated in the paper):
  - Heterogeneous node types and typed per-relation messages (MeshGraphNets is
    homogeneous, single node feature vector).
  - No learned input/output mesh normalisation (GNS) — we rely on the dataset
    feature scaler.
  - No gradient clipping / loss weighting tuned exactly to MeshGraphNets.

Reference:
  - MeshGraphNets: Pfaff et al., ICLR 2021.
  - GNS: Sanchez-Gonzalez et al., ICML 2020.
  - Design doc: docs/paper_upgrade_edge_aware_mpnn_design.md
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead


# ---------------------------------------------------------------------------
# Canonical heterogeneous graph metadata (v2 dataset)
# ---------------------------------------------------------------------------

NODE_TYPES: List[str] = ["mesh_node", "beam_element", "plate_element"]

MEMBERSHIP_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
]

STRUCTURAL_EDGE_TYPE: Tuple[str, str, str] = (
    "mesh_node", "structural_link", "mesh_node",
)

ALL_EDGE_TYPES: List[Tuple[str, str, str]] = (
    MEMBERSHIP_EDGE_TYPES + [STRUCTURAL_EDGE_TYPE]
)


def _et_key(edge_type: Tuple[str, str, str]) -> str:
    """Convert edge type tuple ``(src, rel, dst)`` to a string key."""
    return f"{edge_type[0]}___{edge_type[1]}___{edge_type[2]}"


class EdgeEncoderMLP(nn.Module):
    """Encode structural_link edge_attr (10-dim) into hidden_dim.

    Input:  ``(E, edge_dim)``  (10-dim stiffness features).
    Output: ``(E, hidden_dim)``.
    """

    def __init__(
        self,
        edge_dim: int = 10,
        hidden_dim: int = 128,
        edge_hidden: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            edge_attr: ``(E, edge_dim)`` tensor.
        Returns:
            ``(E, hidden_dim)`` tensor.
        """
        return self.net(edge_attr)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Small 2-layer MLP (Linear -> ReLU -> Linear)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class EdgeAwareMPNNLayer(nn.Module):
    """One residual typed message-passing block.

    For each of the 5 edge types:
        msg = msg_MLP_et(cat(x_src, x_dst) [+ edge_feat for structural_link])
        aggr_dst += scatter_sum(msg, dst_idx)
    Then per node type:
        x_nt = x_nt + update_MLP_nt(cat(x_nt, aggr_nt))
        activation -> dropout -> LayerNorm

    Message MLPs and update MLPs are created here but SHARED across layers (the
    model passes them in), so depth does not multiply parameter count.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        structural_edge_dim: int = 10,
        edge_hidden: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
        aggr: str = "sum",
        use_edge_attr: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.aggr = aggr
        self.use_edge_attr = use_edge_attr

        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self.activation_fn = act_map.get(activation, nn.ReLU())

        # Per-node-type LayerNorm (LayerNorm is NOT shared across layers;
        # one per layer, matching HGT/MS-HGT per-layer normalisation).
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES
        })

        # Edge encoder for structural_link edge_attr (shared across layers).
        self.edge_encoder = EdgeEncoderMLP(
            edge_dim=structural_edge_dim,
            hidden_dim=hidden_dim,
            edge_hidden=edge_hidden,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        msg_mlps: nn.ModuleDict,
        update_mlps: nn.ModuleDict,
        structural_edge_attr: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one block.

        Args:
            x_dict: ``{node_type: (N_nt, hidden_dim)}``.
            edge_index_dict: ``{edge_type: (2, E)}`` for all 5 edge types.
            msg_mlps: shared per-relation message MLPs.
            update_mlps: shared per-node-type update MLPs.
            structural_edge_attr: ``(E_sl, structural_edge_dim)`` or ``None``.

        Returns:
            ``{node_type: (N_nt, hidden_dim)}`` — updated features.
        """
        # ---- 1. Messages per edge type + aggregate by destination ----
        aggr: Dict[str, torch.Tensor] = {
            nt: torch.zeros(x_dict[nt].shape[0], self.hidden_dim,
                            device=x_dict[nt].device)
            for nt in NODE_TYPES
        }

        for et in ALL_EDGE_TYPES:
            src, rel, dst = et
            edge_index = edge_index_dict[et]  # (2, E)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            x_src = x_dict[src][src_idx]   # (E, H)
            x_dst = x_dict[dst][dst_idx]   # (E, H)

            if et == STRUCTURAL_EDGE_TYPE and self.use_edge_attr \
                    and structural_edge_attr is not None:
                e_feat = self.edge_encoder(structural_edge_attr)  # (E, H)
                msg_in = torch.cat([x_src, x_dst, e_feat], dim=-1)  # (E, 3H)
            else:
                msg_in = torch.cat([x_src, x_dst], dim=-1)  # (E, 2H)

            msg = msg_mlps[_et_key(et)](msg_in)  # (E, H)

            out = scatter(
                msg, dst_idx, dim=0,
                dim_size=x_dict[dst].shape[0], reduce=self.aggr,
            )
            aggr[dst] = aggr[dst] + out

        # ---- 2. Residual node update ----
        new_x: Dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            upd_in = torch.cat([x_dict[nt], aggr[nt]], dim=-1)  # (N, 2H)
            new_x[nt] = x_dict[nt] + update_mlps[nt](upd_in)

        # ---- 3. Activation -> dropout -> LayerNorm ----
        for nt in NODE_TYPES:
            h = self.activation_fn(new_x[nt])
            h = F.dropout(h, p=self.dropout, training=self.training)
            new_x[nt] = self.norms[nt](h)

        return new_x


class EdgeAwareMPNNBaseline(nn.Module):
    """Edge-aware MPNN (MeshGraphNet-style) — heterogeneous neural simulator.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension (default 128).
        num_layers: Number of processor blocks (default 6).
        dropout: Dropout rate (default 0.1).
        activation: Activation name.
        use_layer_norm: Whether to apply per-node-type LayerNorm after each
            block (default True).
        decoder_hidden_dims: Hidden dims of decoder MLP heads.
        structural_edge_dim: Dimension of structural_link edge_attr (default 10).
        edge_hidden_dim: Hidden dim of the edge encoder MLP (default 32).
        use_edge_attr: Whether to use structural_link edge_attr in messages
            (default True).
        use_type_embedding: Whether to add a learnable per-node-type embedding
            to projected node features (default True).
        aggr: Message aggregation across neighbors — ``"sum"`` (default) or
            ``"mean"`` (MeshGraphNets uses sum).
        residual: Residual node update (default True). Always kept True in v1;
            the flag exists so an ablation can disable it later.
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        structural_edge_dim: int = 10,
        edge_hidden_dim: int = 32,
        use_edge_attr: bool = True,
        use_type_embedding: bool = True,
        aggr: str = "sum",
        residual: bool = True,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_edge_attr = use_edge_attr
        self.use_type_embedding = use_type_embedding
        self.aggr = aggr
        self.residual = residual

        # ---- Type-specific node encoders ----
        self.mesh_encoder = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_encoder = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_encoder = nn.Linear(plate_feat_dim, hidden_dim)

        # Optional per-node-type embedding
        if use_type_embedding:
            self.type_embed = nn.Embedding(len(NODE_TYPES), hidden_dim)
        else:
            self.type_embed = None

        # ---- Shared (across layers) per-relation message MLPs ----
        self.msg_mlps = nn.ModuleDict()
        for et in ALL_EDGE_TYPES:
            if et == STRUCTURAL_EDGE_TYPE and use_edge_attr:
                msg_in_dim = hidden_dim * 3  # x_src + x_dst + edge_feat
            else:
                msg_in_dim = hidden_dim * 2  # x_src + x_dst
            self.msg_mlps[_et_key(et)] = _mlp(msg_in_dim, hidden_dim, hidden_dim)

        # ---- Shared (across layers) per-node-type update MLPs ----
        self.update_mlps = nn.ModuleDict({
            nt: _mlp(hidden_dim * 2, hidden_dim, hidden_dim) for nt in NODE_TYPES
        })

        # ---- Processor blocks (LayerNorm per layer; message/update MLPs shared) ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EdgeAwareMPNNLayer(
                hidden_dim=hidden_dim,
                structural_edge_dim=structural_edge_dim,
                edge_hidden=edge_hidden_dim,
                dropout=dropout,
                activation=activation,
                aggr=aggr,
                use_edge_attr=use_edge_attr,
            ))

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
        """Forward pass on a HeteroDataBatch.

        Args:
            batch: HeteroDataBatch with 3 node types and 5 edge types;
                structural_link carries optional ``edge_attr``.

        Returns:
            ``(pred_disp, pred_force)``:
                - pred_disp:  ``(total_mesh_nodes, 6)``
                - pred_force: ``(total_beam_elements, 12)``
        """
        # ---- 1. Type-specific projections ----
        x_dict: Dict[str, torch.Tensor] = {
            "mesh_node": self.mesh_encoder(batch["mesh_node"].x),
            "beam_element": self.beam_encoder(batch["beam_element"].x),
            "plate_element": self.plate_encoder(batch["plate_element"].x),
        }
        if self.type_embed is not None:
            type_emb = {
                nt: self.type_embed(torch.tensor(i, device=x_dict[nt].device))
                for i, nt in enumerate(NODE_TYPES)
            }
            x_dict = {nt: x_dict[nt] + type_emb[nt] for nt in NODE_TYPES}

        # ---- 2. Edge index dict ----
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in ALL_EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. Structural_link edge_attr (defensive) ----
        structural_edge_attr: Optional[torch.Tensor] = None
        if STRUCTURAL_EDGE_TYPE in batch.edge_types and \
                hasattr(batch[STRUCTURAL_EDGE_TYPE], "edge_attr"):
            structural_edge_attr = batch[STRUCTURAL_EDGE_TYPE].edge_attr

        # ---- 4. Processor ----
        for layer in self.layers:
            x_dict = layer(
                x_dict, edge_index_dict,
                self.msg_mlps, self.update_mlps,
                structural_edge_attr,
            )

        # ---- 5. Decoders ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])       # (M_total, 6)
        pred_force = self.force_decoder(x_dict["beam_element"])  # (B_total, 12)

        return pred_disp, pred_force

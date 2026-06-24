"""
ms_hgt.py — Multi-Scale HGT (MS-HGT) with macro anchor graph + cross-scale fusion.

Architecture (Stage 4 boundary):
  1. Type-specific input encoders (same as HGTBaseline)
  2. HGTConv typed attention layers (NODE_TYPES × EDGE_TYPES micro graph)
  3. MacroAnchorPool: geometric anchor along X → K equal-width segments
  4. Shared MacroGNN: 2-layer SAGEConv on sequential chain macro graph
  5. Per-layer CrossScaleFusion (gated residual or additive)
  6. Dual decoders: mesh_node → disp(6), beam_element → force(12)

MS-HGT = HGT backbone + geometric macro anchor + macro message passing
         + cross-scale fusion.

Does NOT include:
  - Physics loss (Stage 5)
  - UQ / conformal (Stage 6)
  - edge_attr-aware message (Ours-only, terminated)
  - Learned or structure-aware anchor

Reference:
  - HGT: Hu et al., WWW 2020. "Heterogeneous Graph Transformer."
    https://arxiv.org/abs/2003.01332
  - SAGEConv: Hamilton et al., NeurIPS 2017.
    https://arxiv.org/abs/1706.02216
  - Design: docs/stage4_macro_anchor_design.md

Input: ``HeteroDataBatch`` (from ``torch_geometric.loader.DataLoader``).

Returns:
    ``(pred_disp, pred_force)`` —
    - pred_disp:  ``(total_mesh_nodes, 6)``
    - pred_force: ``(total_beam_elements, 12)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, SAGEConv
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead


# ---------------------------------------------------------------------------
# Canonical heterogeneous graph metadata (v2 dataset)
# ---------------------------------------------------------------------------

NODE_TYPES = ["mesh_node", "beam_element", "plate_element"]

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]

METADATA = (NODE_TYPES, EDGE_TYPES)


# ---------------------------------------------------------------------------
# MacroAnchorPool
# ---------------------------------------------------------------------------


class MacroAnchorPool(nn.Module):
    """Geometric anchor pooling along longitudinal (X) axis.

    For each graph in the batch:
      1. Divide X-range into ``n_segments`` equal-width bins.
      2. Assign each mesh node to its bin.
      3. Mean-pool mesh hidden states within each bin.
      4. Optionally concatenate static features (normalized center
         position, log node count) and project to ``hidden_dim``.

    The bin-to-node assignment is precomputed once per forward pass
    and reused across HGT layers (static geometry).
    """

    def __init__(
        self,
        n_segments: int = 12,
        hidden_dim: int = 128,
        include_static_feats: bool = True,
    ):
        super().__init__()
        self.n_segments = n_segments
        self.include_static_feats = include_static_feats

        if include_static_feats:
            self.static_proj = nn.Linear(hidden_dim + 2, hidden_dim)

        # Diagnostics buffers
        self._last_anchor_stats: Dict = {}

    def _build_bin_edges(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Build ``n_segments+1`` bin edges from ``x`` (per-graph)."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < eps:
            # Degenerate case: all same X → force one bin
            edges = torch.linspace(x_min - 0.5, x_min + 0.5,
                                   self.n_segments + 1, device=x.device)
        else:
            edges = torch.linspace(x_min, x_max,
                                   self.n_segments + 1, device=x.device)
        return edges

    def assign_nodes(
        self,
        coords: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Precompute node-to-anchor assignments.

        Args:
            coords: ``(total_mesh_nodes, 3)`` — X, Y, Z coordinates
                (standardized or original, relative ordering preserved).
            batch_vec: ``(total_mesh_nodes,)`` — per-node graph index.

        Returns:
            ``(node_assignments, anchor_stats)``
                - node_assignments: ``(total_mesh_nodes,)`` int — bin ID per node.
                - anchor_stats: dict with per-graph diagnostics.
        """
        device = coords.device
        B = int(batch_vec.max().item()) + 1
        N = coords.shape[0]
        N_seg = self.n_segments

        node_assignments = torch.full((N,), -1, dtype=torch.long, device=device)
        per_graph_counts = []
        empty_segments_total = 0
        total_nodes_accounted = 0

        for b in range(B):
            mask = (batch_vec == b)
            x_b = coords[mask, 0]
            n_local = mask.sum().item()
            if n_local == 0:
                per_graph_counts.append(0)
                continue

            edges = self._build_bin_edges(x_b)
            # bucketize with right=False:
            #   returns 0 if x < edges[0], k if edges[k-1] <= x < edges[k],
            #   returns n_segments if x >= edges[-1]
            bins = torch.bucketize(x_b, edges, right=False)
            # Clamp to [0, n_segments - 1] for safety
            bins = bins.clamp(0, N_seg - 1)

            node_assignments[mask] = bins

            # Diagnostics
            seg_counts = scatter(
                torch.ones(n_local, dtype=torch.long, device=device),
                bins, dim=0, dim_size=N_seg, reduce="sum",
            )
            empty_b = (seg_counts == 0).sum().item()
            empty_segments_total += empty_b
            total_nodes_accounted += n_local
            per_graph_counts.append(n_local)

        anchor_stats = {
            "n_graphs": B,
            "n_segments": N_seg,
            "total_nodes": N,
            "per_graph_node_counts": per_graph_counts,
            "empty_segments": empty_segments_total,
            "mean_nodes_per_segment": (
                round(total_nodes_accounted / max(N_seg * B, 1), 1)
            ),
        }

        # Cache for diagnostics
        self._last_anchor_stats = anchor_stats

        return node_assignments, anchor_stats

    def pool(
        self,
        x_mesh: torch.Tensor,
        node_assignments: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool mesh hidden states to anchor features.

        Args:
            x_mesh: ``(total_mesh_nodes, hidden_dim)``
            node_assignments: ``(total_mesh_nodes,)`` bin IDs.
            batch_vec: ``(total_mesh_nodes,)`` graph indices.

        Returns:
            ``(anchor_features, per_graph_offsets)``
                - anchor_features: ``(B * n_segments, hidden_dim)``
                - per_graph_offsets: ``(B,)`` — cumulative offsets per graph.
        """
        device = x_mesh.device
        B = int(batch_vec.max().item()) + 1
        N_seg = self.n_segments
        hidden_dim = x_mesh.shape[1]

        # Global anchor index = b * n_segments + bin_id
        global_anchor_idx = batch_vec * N_seg + node_assignments
        # Clamp for safety (e.g. if some nodes were unassigned at -1)
        global_anchor_idx = global_anchor_idx.clamp(0, B * N_seg - 1)

        # Scatter mean to get anchor features
        anchor_features = scatter(
            x_mesh,
            global_anchor_idx,
            dim=0,
            dim_size=B * N_seg,
            reduce="mean",
        )  # (B * N_seg, hidden_dim)

        # Fix NaN for empty segments (scatter_mean gives NaN on empty bins)
        anchor_features = torch.nan_to_num(anchor_features, nan=0.0)

        # Per-graph offsets for macro edge construction
        offsets = torch.arange(0, B * N_seg, N_seg, device=device)

        return anchor_features, offsets

    def compute_static_features(
        self,
        coords: torch.Tensor,
        batch_vec: torch.Tensor,
        node_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Compute static features for each anchor segment.

        Returns:
            ``(B * n_segments, 2)`` tensor with
                [normalized_center_x, log_node_count_plus_one].
        """
        device = coords.device
        B = int(batch_vec.max().item()) + 1
        N_seg = self.n_segments

        static_list = []
        for b in range(B):
            mask = (batch_vec == b)
            x_b = coords[mask, 0]
            if mask.sum() == 0:
                empty = torch.zeros(N_seg, 2, device=device)
                static_list.append(empty)
                continue

            edges = self._build_bin_edges(x_b)
            x_min, x_max = x_b.min(), x_b.max()
            x_span = max(x_max - x_min, 1e-6)

            # Centers of each bin
            centers = (edges[:-1] + edges[1:]) / 2.0
            normalized_center = (centers - x_min) / x_span

            # Node count per segment
            bins_local = node_assignments[mask]
            n_per_seg = scatter(
                torch.ones(bins_local.shape[0], device=device),
                bins_local,
                dim=0,
                dim_size=N_seg,
                reduce="sum",
            )
            log_n = torch.log(n_per_seg.float() + 1.0)

            static_list.append(
                torch.stack([normalized_center, log_n], dim=1)
            )

        return torch.cat(static_list, dim=0)  # (B * N_seg, 2)

    def unpool(
        self,
        anchor_hidden: torch.Tensor,
        node_assignments: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute macro anchor hidden back to mesh nodes.

        Each mesh node receives the hidden of its assigned anchor segment.

        Args:
            anchor_hidden: ``(B * n_segments, hidden_dim)``
            node_assignments: ``(total_mesh_nodes,)`` bin IDs.
            batch_vec: ``(total_mesh_nodes,)`` graph indices.

        Returns:
            ``(total_mesh_nodes, hidden_dim)``
        """
        B = int(batch_vec.max().item()) + 1
        N_seg = self.n_segments

        global_idx = batch_vec * N_seg + node_assignments.clamp(0, N_seg - 1)
        return anchor_hidden[global_idx]  # (N_total, hidden_dim)


# ---------------------------------------------------------------------------
# MacroGNN (shared)
# ---------------------------------------------------------------------------


class MacroGNN(nn.Module):
    """Shared macro graph message passing on sequential anchor chain.

    ``num_layers`` SAGEConv layers with LayerNorm + ReLU + Dropout.
    Hidden dimension is preserved across layers.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        aggr: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim,
                         aggr=aggr)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(total_anchor_nodes, hidden_dim)``
            edge_index: ``(2, total_anchor_edges)`` — batched macro edges.

        Returns:
            ``(total_anchor_nodes, hidden_dim)``
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ---------------------------------------------------------------------------
# CrossScaleFusion
# ---------------------------------------------------------------------------


class CrossScaleFusion(nn.Module):
    """Fuse macro anchor information back into micro mesh node representations.

    Supports two methods:
      - ``"additive"``: ``h_out = h_mesh + macro_unpooled``
      - ``"gated_residual"``: ``h_out = h_mesh + gate * macro_unpooled``
        where ``gate = sigmoid(MLP([h_mesh, macro_unpooled]))``
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        method: str = "gated_residual",
    ):
        super().__init__()
        self.method = method

        if method == "gated_residual":
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Initialise gate bias → ~0.12 (sigmoid(-2) ≈ 0.12)
            # so macro starts with small influence
            self.gate_net[-1].bias.data.fill_(-2.0)

        elif method == "additive":
            pass  # No gate needed
        else:
            raise ValueError(
                f"Unknown fusion method '{method}'. "
                f"Options: 'additive', 'gated_residual'."
            )

    def forward(
        self,
        h_mesh: torch.Tensor,
        macro_unpooled: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fusion.

        Args:
            h_mesh: ``(total_mesh_nodes, hidden_dim)``
            macro_unpooled: ``(total_mesh_nodes, hidden_dim)``

        Returns:
            ``(total_mesh_nodes, hidden_dim)``
        """
        if self.method == "additive":
            self._last_gate = None
            return h_mesh + macro_unpooled
        elif self.method == "gated_residual":
            gate_input = torch.cat([h_mesh, macro_unpooled], dim=-1)
            gate = torch.sigmoid(self.gate_net(gate_input))
            # Cache gate stats for diagnostics
            self._last_gate = gate.detach()
            return h_mesh + gate * macro_unpooled
        else:
            self._last_gate = None
            return h_mesh  # fallback (shouldn't reach here)


# ---------------------------------------------------------------------------
# Macro graph builder
# ---------------------------------------------------------------------------


def _build_batched_macro_edge_index(
    n_segments: int,
    n_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    """Build batched bidirectional chain macro edge index.

    Each graph in the batch has the same chain:
        0 <-> 1 <-> ... <-> (n_segments - 1)

    Edges per graph: 2 * (n_segments - 1)  (forward + backward)
    Total edges: n_graphs * 2 * (n_segments - 1)

    Args:
        n_segments: Number of anchor segments per graph.
        n_graphs: Number of graphs in this batch.
        device: Target device.

    Returns:
        ``(2, total_edges)`` edge index tensor.
    """
    if n_segments < 2:
        # Single segment → self-loop
        return torch.zeros(2, n_graphs, dtype=torch.long, device=device)

    # Canonical pattern for one graph
    src_fwd = torch.arange(n_segments - 1, device=device)
    dst_fwd = src_fwd + 1
    src_bwd = torch.arange(1, n_segments, device=device)
    dst_bwd = src_bwd - 1

    src = torch.cat([src_fwd, src_bwd])  # (2 * (n_segments - 1),)
    dst = torch.cat([dst_fwd, dst_bwd])

    # Expand for batch
    all_src_list = []
    all_dst_list = []
    for b in range(n_graphs):
        offset = b * n_segments
        all_src_list.append(src + offset)
        all_dst_list.append(dst + offset)

    all_src = torch.cat(all_src_list)
    all_dst = torch.cat(all_dst_list)

    return torch.stack([all_src, all_dst])


# ---------------------------------------------------------------------------
# MSHGTBaseline
# ---------------------------------------------------------------------------


class MSHGTBaseline(nn.Module):
    """Multi-Scale HGT — HGT backbone + macro anchor graph + cross-scale fusion.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension (default 128).
        num_layers: Number of HGTConv / micro layers (default 3).
        heads: Number of attention heads per HGTConv (default 4).
        dropout: Dropout rate (default 0.1).
        activation: Activation name (``"relu"``, ``"gelu"``, ``"elu"``).
        use_layer_norm: Per-node-type LayerNorm after each layer (default True).
        decoder_hidden_dims: Hidden dims of decoder MLP heads.
        n_segments: Number of macro anchor segments (default 12).
        macro_gnn_layers: Number of MacroGNN SAGEConv layers (default 2).
        macro_gnn_aggr: SAGEConv aggregation (default ``"mean"``).
        include_anchor_static: Include static features in anchor (default True).
        fusion_method: ``"gated_residual"`` or ``"additive"``.
        fusion_per_layer: Apply fusion after every HGT layer (default True).
            If ``False``, fusion is applied only after the last layer.
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        # Macro anchor parameters
        n_segments: int = 12,
        macro_gnn_layers: int = 2,
        macro_gnn_aggr: str = "mean",
        include_anchor_static: bool = True,
        fusion_method: str = "gated_residual",
        fusion_per_layer: bool = True,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.n_segments = n_segments
        self.fusion_method = fusion_method
        self.fusion_per_layer = fusion_per_layer

        # ---- Type-specific encoders ----
        self.mesh_encoder = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_encoder = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_encoder = nn.Linear(plate_feat_dim, hidden_dim)

        # ---- Activation ----
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self._activation = act_map.get(activation, nn.ReLU())

        # ---- HGTConv layers (micro graph) ----
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=METADATA,
                    heads=heads,
                )
            )

        # ---- Per-node-type LayerNorm ----
        if use_layer_norm:
            self.layer_norms = nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES
            })

        # ---- Macro anchor modules ----
        self.anchor_pool = MacroAnchorPool(
            n_segments=n_segments,
            hidden_dim=hidden_dim,
            include_static_feats=include_anchor_static,
        )

        self.macro_gnn = MacroGNN(
            hidden_dim=hidden_dim,
            num_layers=macro_gnn_layers,
            aggr=macro_gnn_aggr,
            dropout=dropout,
        )

        # Anchor static feature projection (if enabled)
        if include_anchor_static:
            self.anchor_static_proj = nn.Linear(hidden_dim + 2, hidden_dim)
        else:
            self.anchor_static_proj = None

        # ---- Cross-scale fusion per layer ----
        self.fusion = CrossScaleFusion(
            hidden_dim=hidden_dim,
            method=fusion_method,
        )

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

    def _apply_anchor_path(
        self,
        h_mesh: torch.Tensor,
        coords: torch.Tensor,
        batch_vec: torch.Tensor,
        node_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Run anchor pooling → macro GNN → unpool → fusion.

        Args:
            h_mesh: Current mesh node hidden (total_mesh_nodes, hidden_dim).
            coords: (total_mesh_nodes, 3) — X, Y, Z.
            batch_vec: (total_mesh_nodes,) graph assignment.
            node_assignments: (total_mesh_nodes,) bin IDs from precompute.

        Returns:
            Fused mesh node hidden (total_mesh_nodes, hidden_dim).
        """
        # 1. Pool mesh → anchor features
        anchor_hidden, offsets = self.anchor_pool.pool(
            h_mesh, node_assignments, batch_vec,
        )
        # anchor_hidden: (B * n_segments, hidden_dim)

        # 2. Optionally add static features
        if self.anchor_static_proj is not None:
            static_feats = self.anchor_pool.compute_static_features(
                coords, batch_vec, node_assignments,
            )
            # static_feats: (B * n_segments, 2)
            combined = torch.cat([anchor_hidden, static_feats], dim=-1)
            anchor_hidden = self.anchor_static_proj(combined)

        # 3. Build batched macro edge index
        B = int(batch_vec.max().item()) + 1
        macro_edge_index = _build_batched_macro_edge_index(
            self.n_segments, B, device=h_mesh.device,
        )

        # 4. Macro message passing
        macro_hidden = self.macro_gnn(anchor_hidden, macro_edge_index)
        # macro_hidden: (B * n_segments, hidden_dim)

        # 5. Unpool to mesh nodes
        macro_unpooled = self.anchor_pool.unpool(
            macro_hidden, node_assignments, batch_vec,
        )
        # macro_unpooled: (total_mesh_nodes, hidden_dim)

        # 6. Fuse
        h_fused = self.fusion(h_mesh, macro_unpooled)

        return h_fused

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            batch: ``HeteroDataBatch`` with 3 node types and 5 edge types.

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

        # ---- 2. Edge index dict ----
        edge_index_dict = {}
        for et in EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. Precompute anchor assignment (static within batch) ----
        coords = batch["mesh_node"].x[:, :3]         # (total_nodes, 3)
        batch_vec = batch["mesh_node"].batch          # (total_nodes,)

        node_assignments, anchor_stats = self.anchor_pool.assign_nodes(
            coords, batch_vec,
        )
        # Store diagnostics for external inspection
        self._last_anchor_stats = anchor_stats

        # ---- 4. HGTConv layers with macro fusion ----
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self._activation(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }
            if self.use_layer_norm:
                x_dict = {
                    k: self.layer_norms[k](v) for k, v in x_dict.items()
                }

            # Macro anchor path + fusion
            if self.fusion_per_layer or i == self.num_layers - 1:
                h_mesh = x_dict["mesh_node"]
                x_dict["mesh_node"] = self._apply_anchor_path(
                    h_mesh, coords, batch_vec, node_assignments,
                )

        # ---- 5. Decode ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])
        pred_force = self.force_decoder(x_dict["beam_element"])

        return pred_disp, pred_force

    def get_anchor_stats(self) -> Dict:
        """Return latest anchor diagnostics (from most recent forward)."""
        return getattr(self, "_last_anchor_stats", {})

    def get_gate_stats(self) -> Optional[Dict]:
        """Return fusion gate statistics (requires ``gated_residual``)."""
        if self.fusion_method != "gated_residual":
            return None
        gate_weights = getattr(self.fusion, "_last_gate", None)
        if gate_weights is None:
            return None
        return {
            "mean": float(gate_weights.mean().item()),
            "std": float(gate_weights.std().item()),
            "min": float(gate_weights.min().item()),
            "max": float(gate_weights.max().item()),
        }

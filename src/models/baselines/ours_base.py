"""
ours_base.py — EA-HGNN (Edge-Attribute-Aware Heterogeneous GNN) — Ours-base model.

Stage 3 of Multi-Scale PI-HGNN research.

Architecture:
  1. **Type-specific node encoder**: Each node type (mesh_node, beam_element,
     plate_element) has an independent ``nn.Linear`` projection to ``hidden_dim``
     (same as RGCN/HGT baselines).
  2. **Typed micro message passing** (L layers): For each of the 5 edge types:
       - *Membership edges* (4 types) — ``SAGEConv`` (relation-specific, same as RGCN).
       - *StructuralLink edge* (1 type) — ``StructuralLinkConv`` (edge-attribute-aware
         message + physics gate).
  3. **Dual decoder**: ``MLPHead`` decoders map mesh_node hidden states → 6-dim
     displacement and beam_element hidden states → 12-dim force (shared latent).

Key innovations (Stage 3):
  - **Edge-attribute-aware StructuralLinkConv** — encodes stiffness features
    (Kx..Krz, BetaAngle, DistanceRatio, ElasticLinkType, is_rigid) into the
    message via an ``EdgeEncoder`` MLP.
  - **Physics-gated message** — a ``PhysicsGate`` (``sigmoid(MLP(edge_attr))``)
    controls message strength based on physical connection stiffness.
  - **Shared dual decoder** — both displacement and force heads branch from the
    same final hidden state (no macro, no physics loss, no UQ in Stage 3).

NOT included (Stage 3 boundary):
  - No macro anchor graph / cross-scale fusion (→ Stage 4).
  - No physics loss / support BC loss (→ Stage 5).
  - No uncertainty quantification (→ Stage 6).
  - No MeshGraphNet-style processor (optional baseline, not yet implemented).

Reference:
  - **RGCN baseline** (``hetero_rgcn.py``): Relation-specific SAGEConv via
    HeteroConv — Ours-base reuses the same membership-edge conv pattern.
  - **HGT baseline** (``hgt_baseline.py``): Typed attention — Ours-base does
    NOT use attention; it uses a learnable physics gate for structural links.
  - **MeshGraphNets** (indirect inspiration): Edge encoding in message passing.

Input: ``HeteroDataBatch`` (from ``torch_geometric.loader.DataLoader``).

Returns:
    Tuple of ``(pred_disp, pred_force)`` —
    - ``pred_disp``:  ``(total_mesh_nodes, 6)``
    - ``pred_force``: ``(total_beam_elements, 12)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv

from src.models.baselines.decoders import MLPHead


# ============================================================
# Constants (same canonical schema as RGCN/HGT)
# ============================================================

NODE_TYPES: List[str] = ["mesh_node", "beam_element", "plate_element"]

MEMBER_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
]

STRUCTURAL_EDGE_TYPE: Tuple[str, str, str] = ("mesh_node", "structural_link", "mesh_node")

ALL_EDGE_TYPES: List[Tuple[str, str, str]] = MEMBER_EDGE_TYPES + [STRUCTURAL_EDGE_TYPE]


def _et_key(edge_type: Tuple[str, str, str]) -> str:
    """Convert edge type tuple ``(src, rel, dst)`` to a string key."""
    return f"{edge_type[0]}___{edge_type[1]}___{edge_type[2]}"


# ============================================================
# EdgeEncoder — encode structural_link edge_attr into hidden
# ============================================================

class EdgeEncoder(nn.Module):
    """Encode structural_link edge attributes into a hidden representation.

    Input:  ``(E, edge_dim)`` — stiffness features [Kx..Krz, BetaAngle,
            DistanceRatio, ElasticLinkType, is_rigid].
    Output: ``(E, out_dim)``.

    The first version treats all 10 features as continuous float inputs.
    Categorical embedding (ElasticLinkType, is_rigid) is left as a future
    refinement (not needed when all links are currently RIGID).
    """

    def __init__(self, edge_dim: int, out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode edge attributes.

        Args:
            edge_attr: ``(E, edge_dim)`` tensor.
        Returns:
            ``(E, out_dim)`` tensor.
        """
        return self.net(edge_attr)


# ============================================================
# PhysicsGate — sigmoid gate conditioned on edge_attr
# ============================================================

class PhysicsGate(nn.Module):
    """Physics-informed gate for structural_link messages.

    ``gate = sigmoid(MLP(edge_attr))``

    The gate learns to regulate message strength based on physical connection
    stiffness. Stiffer connections (higher K values) produce gates closer to 1,
    allowing stronger messages; flexible connections have gates near 0.

    Input:  ``(E, edge_dim)``
    Output: ``(E, 1)``
    """

    def __init__(self, edge_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute gate values.

        Args:
            edge_attr: ``(E, edge_dim)``
        Returns:
            ``(E, 1)`` gate in [0, 1].
        """
        return torch.sigmoid(self.net(edge_attr))


# ============================================================
# StructuralLinkConv — edge-attr-aware message passing
# ============================================================

class StructuralLinkConv(MessagePassing):
    """Edge-attribute-aware message passing for ``structural_link`` edges.

    For each structural_link edge ``mesh_node_i → mesh_node_j``:

        1. Encode edge attributes via ``EdgeEncoder``.
        2. Compute a physics gate via ``PhysicsGate``.
        3. Message = ``gate * W_src(h_src) + edge_encoding``.

    Aggregation: ``mean`` (over incident structural link edges).

    No self-connection is included — the layer-level aggregation from
    membership ``SAGEConv`` already provides the self-connection for
    ``mesh_node``.

    Args:
        in_channels: Input feature dim for ``mesh_node``.
        out_channels: Output feature dim.
        edge_dim: Structural link edge attribute dimension (default: 10).
        edge_hidden: Hidden dim for ``EdgeEncoder`` (default: 32).
        gate_hidden: Hidden dim for ``PhysicsGate`` (default: 16).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 10,
        edge_hidden: int = 32,
        gate_hidden: int = 16,
    ):
        super().__init__(aggr="mean")
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_encoder = EdgeEncoder(edge_dim, out_channels, edge_hidden)
        self.gate = PhysicsGate(edge_dim, gate_hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``mesh_node`` features ``(N, in_channels)``.
            edge_index: Structural link edge indices ``(2, E)``.
            edge_attr: Structural link edge attributes ``(E, edge_dim)``,
                or ``None`` (fallback — message degrades to bare ``W_src``).

        Returns:
            ``(N, out_channels)`` — aggregated messages (no self-connection).
        """
        if edge_attr is None:
            # Degenerate case: no edge_attr available — fall back to
            # source-only message (identical to membership SAGEConv message).
            return self.propagate(edge_index, x=(x, x), edge_attr=None)
        return self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute edge-attr-aware messages.

        Args:
            x_j: Source node features ``(E, in_channels)``.
            edge_attr: Edge attributes ``(E, edge_dim)`` or ``None``.

        Returns:
            ``(E, out_channels)`` messages.
        """
        node_msg = self.lin_src(x_j)  # (E, out_channels)

        if edge_attr is None:
            return node_msg  # fallback: plain source message

        edge_enc = self.edge_encoder(edge_attr)  # (E, out_channels)
        gate_val = self.gate(edge_attr)  # (E, 1)

        return gate_val * node_msg + edge_enc

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Return aggregated messages as-is (no self-connection).

        Args:
            aggr_out: ``(N, out_channels)`` aggregated messages.
        Returns:
            ``(N, out_channels)``.
        """
        return aggr_out


# ============================================================
# OursBaseLayer — one message-passing layer
# ============================================================

class OursBaseLayer(nn.Module):
    """One EA-HGNN typed message-passing layer.

    For each edge type:
        - *Membership* (4 types): ``SAGEConv`` (relation-specific, same as RGCN).
        - *StructuralLink* (1 type): ``StructuralLinkConv`` (edge_attr-aware).

    Aggregation: sum of all incoming messages per node type.
    Then: activation → dropout → per-node-type LayerNorm.

    Args:
        hidden_dim: Hidden feature dimension.
        structural_edge_dim: Dimension of structural_link ``edge_attr``.
        edge_hidden: Hidden dim for ``EdgeEncoder``.
        dropout: Dropout probability.
        activation: Activation function name (``"relu"``, ``"gelu"``, etc.).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        structural_edge_dim: int = 10,
        edge_hidden: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.dropout = dropout
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self.activation_fn = act_map.get(activation, nn.ReLU())

        # --- Membership edge convs (SAGEConv per type, no edge_attr) ---
        self.member_convs = nn.ModuleDict()
        for et in MEMBER_EDGE_TYPES:
            self.member_convs[_et_key(et)] = SAGEConv(hidden_dim, hidden_dim)

        # --- Structural link conv (edge_attr-aware) ---
        self.structural_conv = StructuralLinkConv(
            hidden_dim, hidden_dim,
            edge_dim=structural_edge_dim,
            edge_hidden=edge_hidden,
        )

        # --- Per-node-type LayerNorm ---
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        structural_edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one EA-HGNN layer.

        Args:
            x_dict: ``{node_type: (N_nt, hidden_dim)}``.
            edge_index_dict: ``{edge_type: (2, E)}`` for all 5 edge types.
            structural_edge_attr: ``(E_sl, structural_edge_dim)`` or ``None``.

        Returns:
            ``{node_type: (N_nt, hidden_dim)}`` — updated features.
        """
        # ---- 1. Membership edges: each SAGEConv returns complete update
        #        (self-connection + aggregated messages) for its dst type. ----
        member_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in NODE_TYPES}
        for et in MEMBER_EDGE_TYPES:
            conv = self.member_convs[_et_key(et)]
            src, rel, dst = et
            edge_index = edge_index_dict[et]
            # Bipartite SAGEConv: (x_src, x_dst) as tuple
            out = conv((x_dict[src], x_dict[dst]), edge_index)
            member_out[dst].append(out)

        # ---- 2. Sum membership contributions per node type ----
        new_x: Dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            outs = member_out[nt]
            if len(outs) == 0:
                new_x[nt] = x_dict[nt]
            else:
                new_x[nt] = sum(outs)  # (N_nt, hidden_dim)

        # ---- 3. Structural link messages (edge_attr-aware, no self-loop) ----
        sl_edge_index = edge_index_dict[STRUCTURAL_EDGE_TYPE]
        sl_msg = self.structural_conv(
            x_dict["mesh_node"], sl_edge_index, structural_edge_attr,
        )
        new_x["mesh_node"] = new_x["mesh_node"] + sl_msg

        # ---- 4. Activation → dropout → LayerNorm per node type ----
        for nt in NODE_TYPES:
            h = self.activation_fn(new_x[nt])
            h = F.dropout(h, p=self.dropout, training=self.training)
            new_x[nt] = self.norms[nt](h)

        return new_x


# ============================================================
# OursBaseline — Full EA-HGNN model
# ============================================================

class OursBaseline(nn.Module):
    """EA-HGNN: Edge-Attribute-Aware Heterogeneous GNN (Ours-base).

    Stage 3 model for Multi-Scale PI-HGNN research.

    Architecture:
        1. **Per-type node encoders**: ``nn.Linear`` per node type (same as RGCN).
        2. **L typed message-passing layers**: Each ``OursBaseLayer`` processes
           membership edges (SAGEConv) and structural_link edges
           (StructuralLinkConv with edge_attr + physics gate).
        3. **Dual decoder**: ``MLPHead`` for displacement (mesh_node → 6) and
           force (beam_element → 12), sharing the same latent representation.

    NOT included (Stage 3 boundary):
        - No macro anchor graph / cross-scale fusion.
        - No physics loss.
        - No UQ.

    Args:
        mesh_feat_dim: Input feature dim for mesh_node (default: 15).
        beam_feat_dim: Input feature dim for beam_element (default: 11).
        plate_feat_dim: Input feature dim for plate_element (default: 6).
        hidden_dim: Shared hidden dimension (default: 128).
        num_layers: Number of OursBaseLayer blocks (default: 3).
        dropout: Dropout rate (default: 0.1).
        activation: Activation name (``"relu"``, ``"gelu"``, ``"elu"``).
        use_layer_norm: Whether to apply LayerNorm per node type after each layer.
        decoder_hidden_dims: Hidden dims for decoder MLP heads (default: [64, 32]).
        structural_edge_dim: Dimension of structural_link edge_attr (default: 10).
        edge_hidden_dim: Hidden dim for EdgeEncoder in StructuralLinkConv (default: 32).
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        structural_edge_dim: int = 10,
        edge_hidden_dim: int = 32,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.structural_edge_dim = structural_edge_dim

        # ---- Type-specific input projections (same as RGCN/HGT) ----
        self.mesh_encoder = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_encoder = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_encoder = nn.Linear(plate_feat_dim, hidden_dim)

        # ---- Message-passing layers ----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(OursBaseLayer(
                hidden_dim=hidden_dim,
                structural_edge_dim=structural_edge_dim,
                edge_hidden=edge_hidden_dim,
                dropout=dropout,
                activation=activation,
            ))

        # ---- Dual decoder (shared latent → disp + force) ----
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
            batch: HeteroDataBatch with:
                - ``batch["mesh_node"].x``: (M_total, mesh_feat_dim)
                - ``batch["beam_element"].x``: (B_total, beam_feat_dim)
                - ``batch["plate_element"].x``: (P_total, plate_feat_dim)
                - ``batch[et].edge_index`` for each of the 5 edge types
                - ``batch[STRUCTURAL_EDGE_TYPE].edge_attr``: (E_sl, structural_edge_dim)

        Returns:
            ``(pred_disp, pred_force)``:
                - pred_disp:  (total_mesh_nodes, 6)
                - pred_force: (total_beam_elements, 12)
        """
        # ---- 1. Type-specific projections to shared hidden_dim ----
        x_dict: Dict[str, torch.Tensor] = {
            "mesh_node": self.mesh_encoder(batch["mesh_node"].x),
            "beam_element": self.beam_encoder(batch["beam_element"].x),
            "plate_element": self.plate_encoder(batch["plate_element"].x),
        }

        # ---- 2. Build edge_index_dict from the HeteroDataBatch ----
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in ALL_EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. Extract structural_link edge_attr (defensive) ----
        structural_edge_attr: Optional[torch.Tensor] = None
        sl_edge_key = STRUCTURAL_EDGE_TYPE
        if sl_edge_key in batch.edge_types and hasattr(batch[sl_edge_key], "edge_attr"):
            structural_edge_attr = batch[sl_edge_key].edge_attr

        # ---- 4. Message-passing layers ----
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, structural_edge_attr)

        # ---- 5. Dual decoder ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])       # (M_total, 6)
        pred_force = self.force_decoder(x_dict["beam_element"])  # (B_total, 12)

        return pred_disp, pred_force


# ============================================================
# Convenience alias
# ============================================================

OursBase = OursBaseline


# ============================================================
# StructuralLinkConvV2 — edge-conditioned gate + root path
# ============================================================

class StructuralLinkConvV2(MessagePassing):
    """Edge-conditioned message passing for structural_link edges (v2).

    Key improvements over v1 (StructuralLinkConv):

      1. **Root/self path**: ``lin_root(x_dst)`` added to aggregated output,
         providing a stable self-connection (matching SAGEConv behavior).
      2. **Gate-only modulation**: ``gate = 1 + gate_scale * tanh(MLP(edge_attr))``,
         with ``gate_scale=0.1`` default. At init, gate ≈ 1.0 so the message
         starts close to plain SAGEConv and learns edge-specific modulation.
      3. **Edge bias default OFF**: ``use_edge_bias=False``. The optional
         ``edge_encoder`` additive path is disabled by default to avoid
         strong uniform bias when all links are RIGID.
      4. **Diagnostics**: ``get_gate_stats()`` returns per-forward-pass gate
         statistics without affecting training.

    Message::

        msg = gate * W_src(h_src)             # if use_edge_bias=False  (default)
        msg = gate * W_src(h_src) + edge_scale * W_edge(edge_attr)   # if use_edge_bias=True

    Output::

        out = W_root(x_dst) + mean_aggr(messages)

    Args:
        in_channels: Input feature dim for mesh_node.
        out_channels: Output feature dim.
        edge_dim: Structural link edge attribute dimension (default: 10).
        edge_hidden: Hidden dim for gate network (default: 32).
        gate_scale: Scale for the tanh residual gate (default: 0.1).
        use_edge_bias: Whether to include additive edge encoding (default: False).
        edge_bias_scale: Scale for edge bias when enabled (default: 0.0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 10,
        edge_hidden: int = 32,
        gate_scale: float = 0.1,
        use_edge_bias: bool = False,
        edge_bias_scale: float = 0.0,
    ):
        super().__init__(aggr="mean")
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=True)
        self.gate_scale = gate_scale
        self.use_edge_bias = use_edge_bias
        self.edge_bias_scale = edge_bias_scale

        # Gate network: edge_attr → 1-d modulation
        self.gate_net = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        # Optional edge bias (default OFF)
        if use_edge_bias:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, edge_hidden),
                nn.ReLU(),
                nn.Linear(edge_hidden, out_channels),
            )

        # Diagnostic storage (reset each forward)
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._last_gate: Optional[torch.Tensor] = None
        self._last_node_msg_norm: Optional[torch.Tensor] = None
        self._last_edge_enc_norm: Optional[torch.Tensor] = None
        self._last_root_norm: Optional[torch.Tensor] = None
        self._last_aggr_norm: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``mesh_node`` features ``(N, in_channels)``.
            edge_index: Structural link edge indices ``(2, E)``.
            edge_attr: Edge attributes ``(E, edge_dim)`` or ``None``.

        Returns:
            ``(N, out_channels)`` = root_path + aggregated messages.
        """
        self._reset_diagnostics()

        # Root/self path (always present)
        root = self.lin_root(x)  # (N, out_channels)
        self._last_root_norm = root.detach().norm(dim=1)

        # Aggregate edge-conditioned neighbor messages
        if edge_attr is not None:
            aggr = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        else:
            aggr = self.propagate(edge_index, x=(x, x), edge_attr=None)

        self._last_aggr_norm = aggr.detach().norm(dim=1)

        return root + aggr

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute edge-conditioned messages.

        Args:
            x_j: Source node features ``(E, in_channels)``.
            edge_attr: Edge attributes ``(E, edge_dim)`` or ``None``.

        Returns:
            ``(E, out_channels)`` messages.
        """
        node_msg = self.lin_src(x_j)  # (E, out_channels)

        if edge_attr is None:
            # Fallback: bare source message (identical to SAGEConv neighbor msg)
            self._last_node_msg_norm = node_msg.detach().norm(dim=1)
            return node_msg

        # Residual gate: gate ≈ 1.0 at init
        gate_raw = self.gate_net(edge_attr)  # (E, 1)
        gate = 1.0 + self.gate_scale * torch.tanh(gate_raw)  # (E, 1)

        msg = gate * node_msg  # (E, out_channels)

        # Optional edge bias (default OFF)
        if self.use_edge_bias and hasattr(self, "edge_encoder"):
            edge_enc = self.edge_encoder(edge_attr)  # (E, out_channels)
            msg = msg + self.edge_bias_scale * edge_enc
            self._last_edge_enc_norm = edge_enc.detach().norm(dim=1)

        # Store diagnostics
        self._last_gate = gate.detach()
        self._last_node_msg_norm = node_msg.detach().norm(dim=1)

        return msg

    def get_gate_stats(self) -> Optional[Dict[str, float]]:
        """Return diagnostic stats from the most recent forward pass.

        Returns:
            Dict with gate_mean, gate_std, gate_min, gate_max,
            node_msg_norm_mean, root_norm_mean, aggr_norm_mean,
            and edge_enc_norm_mean (if use_edge_bias=True).
            Returns ``None`` if no forward pass has been run.
        """
        if self._last_gate is None:
            return None
        g = self._last_gate
        stats: Dict[str, float] = {
            "gate_mean": g.mean().item(),
            "gate_std": g.std().item(),
            "gate_min": g.min().item(),
            "gate_max": g.max().item(),
        }
        if self._last_node_msg_norm is not None:
            stats["node_msg_norm_mean"] = self._last_node_msg_norm.mean().item()
        if self._last_root_norm is not None:
            stats["root_norm_mean"] = self._last_root_norm.mean().item()
        if self._last_aggr_norm is not None:
            stats["aggr_norm_mean"] = self._last_aggr_norm.mean().item()
        if self._last_edge_enc_norm is not None:
            stats["edge_enc_norm_mean"] = self._last_edge_enc_norm.mean().item()
        return stats


# ============================================================
# OursBaseLayerV2 — one message-passing layer (v2)
# ============================================================

class OursBaseLayerV2(nn.Module):
    """One EA-HGNN typed message-passing layer (v2).

    Same structure as OursBaseLayer, but uses StructuralLinkConvV2
    (root path + residual gate, no default edge bias) instead of
    StructuralLinkConv (additive edge encoding).

    Membership edges (4 types): SAGEConv (relation-specific, same as RGCN).
    StructuralLink edge (1 type): StructuralLinkConvV2 (residual gate).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        structural_edge_dim: int = 10,
        edge_hidden: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
        gate_scale: float = 0.1,
        use_edge_bias: bool = False,
        edge_bias_scale: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self.activation_fn = act_map.get(activation, nn.ReLU())

        # Membership edge convs (SAGEConv per type, same as RGCN/v1)
        self.member_convs = nn.ModuleDict()
        for et in MEMBER_EDGE_TYPES:
            self.member_convs[_et_key(et)] = SAGEConv(hidden_dim, hidden_dim)

        # Structural link conv (v2: root path + residual gate)
        self.structural_conv = StructuralLinkConvV2(
            hidden_dim, hidden_dim,
            edge_dim=structural_edge_dim,
            edge_hidden=edge_hidden,
            gate_scale=gate_scale,
            use_edge_bias=use_edge_bias,
            edge_bias_scale=edge_bias_scale,
        )

        # Per-node-type LayerNorm
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        structural_edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one EA-HGNN v2 layer."""
        # ---- 1. Membership edges: SAGEConv per type ----
        member_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in NODE_TYPES}
        for et in MEMBER_EDGE_TYPES:
            conv = self.member_convs[_et_key(et)]
            src, rel, dst = et
            edge_index = edge_index_dict[et]
            out = conv((x_dict[src], x_dict[dst]), edge_index)
            member_out[dst].append(out)

        # ---- 2. Sum membership contributions per node type ----
        new_x: Dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            outs = member_out[nt]
            if len(outs) == 0:
                new_x[nt] = x_dict[nt]
            else:
                new_x[nt] = sum(outs)

        # ---- 3. Structural link (v2: root path + residual gate) ----
        sl_edge_index = edge_index_dict[STRUCTURAL_EDGE_TYPE]
        sl_msg = self.structural_conv(
            x_dict["mesh_node"], sl_edge_index, structural_edge_attr,
        )
        new_x["mesh_node"] = new_x["mesh_node"] + sl_msg

        # ---- 4. Activation → dropout → LayerNorm per node type ----
        for nt in NODE_TYPES:
            h = self.activation_fn(new_x[nt])
            h = F.dropout(h, p=self.dropout, training=self.training)
            new_x[nt] = self.norms[nt](h)

        return new_x

    def get_gate_stats(self) -> Optional[Dict[str, float]]:
        """Delegate to the structural_conv's get_gate_stats()."""
        return self.structural_conv.get_gate_stats()


# ============================================================
# OursBaselineV2 — Full EA-HGNN model (v2)
# ============================================================

class OursBaselineV2(nn.Module):
    """EA-HGNN v2: Edge-Attribute-Aware Heterogeneous GNN with v2 structural link.

    Architecture (same skeleton as OursBaseline, key differences in
    StructuralLinkConvV2):

      1. **Per-type node encoders**: ``nn.Linear`` per node type.
      2. **L typed message-passing layers (OursBaseLayerV2)**:
         - Membership edges: SAGEConv (relation-specific, unchanged).
         - StructuralLink edges: StructuralLinkConvV2 with:
             * Root/self path (``lin_root``).
             * Residual gate (``1 + scale * tanh(MLP(edge_attr))``).
             * Edge bias OFF by default.
      3. **Dual decoder**: MLPHead for disp (6) and force (12).
      4. **Gate diagnostics**: ``get_gate_stats()`` returns per-layer stats.

    NOT included (Stage 3 boundary):
        - No macro anchor graph / cross-scale fusion (→ Stage 4).
        - No physics loss / support BC loss (→ Stage 5).
        - No UQ (→ Stage 6).

    Args:
        mesh_feat_dim: Input feature dim for mesh_node (default: 15).
        beam_feat_dim: Input feature dim for beam_element (default: 11).
        plate_feat_dim: Input feature dim for plate_element (default: 6).
        hidden_dim: Shared hidden dimension (default: 128).
        num_layers: Number of OursBaseLayerV2 blocks (default: 3).
        dropout: Dropout rate (default: 0.1).
        activation: Activation name.
        use_layer_norm: Whether to apply LayerNorm (default: True).
        decoder_hidden_dims: Hidden dims for decoder MLP heads (default: [64, 32]).
        structural_edge_dim: Dimension of structural_link edge_attr (default: 10).
        edge_hidden_dim: Hidden dim for gate net in StructuralLinkConvV2 (default: 32).
        gate_scale: Scale for tanh residual gate (default: 0.1).
        use_edge_bias: Whether to include additive edge encoding (default: False).
        edge_bias_scale: Scale for edge bias when enabled (default: 0.0).
    """

    def __init__(
        self,
        mesh_feat_dim: int = 15,
        beam_feat_dim: int = 11,
        plate_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        structural_edge_dim: int = 10,
        edge_hidden_dim: int = 32,
        gate_scale: float = 0.1,
        use_edge_bias: bool = False,
        edge_bias_scale: float = 0.0,
    ):
        super().__init__()
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.structural_edge_dim = structural_edge_dim
        self.gate_scale = gate_scale
        self.use_edge_bias = use_edge_bias
        self.edge_bias_scale = edge_bias_scale

        # Type-specific input projections (same as OursBaseline v1)
        self.mesh_encoder = nn.Linear(mesh_feat_dim, hidden_dim)
        self.beam_encoder = nn.Linear(beam_feat_dim, hidden_dim)
        self.plate_encoder = nn.Linear(plate_feat_dim, hidden_dim)

        # Message-passing layers (v2)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(OursBaseLayerV2(
                hidden_dim=hidden_dim,
                structural_edge_dim=structural_edge_dim,
                edge_hidden=edge_hidden_dim,
                dropout=dropout,
                activation=activation,
                gate_scale=gate_scale,
                use_edge_bias=use_edge_bias,
                edge_bias_scale=edge_bias_scale,
            ))

        # Dual decoder (same as v1)
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
            batch: HeteroDataBatch with 3 node types and 5 edge types.

        Returns:
            ``(pred_disp, pred_force)`` tuple.
        """
        # ---- 1. Type-specific projections ----
        x_dict: Dict[str, torch.Tensor] = {
            "mesh_node": self.mesh_encoder(batch["mesh_node"].x),
            "beam_element": self.beam_encoder(batch["beam_element"].x),
            "plate_element": self.plate_encoder(batch["plate_element"].x),
        }

        # ---- 2. Build edge_index_dict ----
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in ALL_EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. Extract structural_link edge_attr ----
        structural_edge_attr: Optional[torch.Tensor] = None
        sl_edge_key = STRUCTURAL_EDGE_TYPE
        if sl_edge_key in batch.edge_types and hasattr(batch[sl_edge_key], "edge_attr"):
            structural_edge_attr = batch[sl_edge_key].edge_attr

        # ---- 4. Message-passing layers ----
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, structural_edge_attr)

        # ---- 5. Dual decoder ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])
        pred_force = self.force_decoder(x_dict["beam_element"])

        return pred_disp, pred_force

    def get_gate_stats(self) -> List[Optional[Dict[str, float]]]:
        """Return gate diagnostics from all layers.

        Returns:
            List of dicts (one per layer) with gate_mean, gate_std, etc.
            Entries may be ``None`` if no forward pass has been run.
        """
        return [layer.get_gate_stats() for layer in self.layers]


# ============================================================
# Convenience alias
# ============================================================

OursBaseV2 = OursBaselineV2

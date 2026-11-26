"""
hgt_global.py — HGT + global / virtual node baseline (HGTGlobalBaseline).

Purpose (paper-upgrade Stage Paper-2B):
  A model-level virtual/global-node ablation. Answers the question:
  *"Is the MS-HGT macro-anchor gain just a global shortcut?"* — one per-graph
  global state aggregated from mesh_node hidden states and broadcast back each
  layer.

Design (model-level, NO schema change):
  1. Reuse HGTBaseline's type-specific encoders, HGTConv typed-attention
     backbone and MLPHead dual decoders verbatim.
  2. After each HGTConv layer (activation → dropout → LayerNorm), update a
     per-graph global state with a recurrent memory:
         g_pooled = mean_pool(mesh_node hidden)          # scatter per graph
         g_new    = GRUCell(g_pooled, g_prev)            # recurrent global memory
  3. Broadcast the global state back to mesh_node hidden:
         broadcast = W_bcast(g_new)[batch_vec]
         mesh_x = mesh_x + gate * broadcast              # gated_residual | additive
  4. beam_element / plate_element do NOT receive the global broadcast
     (``global_mode = "mesh_global"``).
  5. Decoders unchanged: mesh_node -> disp(6), beam_element -> force(12).

Difference vs MS-HGT:
  - MS-HGT: ``n_segments=12`` structure-aligned (coordinate) anchors per graph,
    each carrying local span information; a sequential-chain macro graph; and
    per-layer gated coarse-to-fine fusion.
  - This baseline: ONE global state per graph, no spatial alignment, no macro
    graph. If it matches MS-HGT on accuracy (esp. midspan / high-response
    subsets), macro-anchor is likely "just a global shortcut"; if MS-HGT is
    clearly better on long-range-sensitive regions, structure-aligned coarse
    locality matters beyond a global readout.

``global_mode`` support (v1):
  - ``"mesh_global"``: implemented. Global state pooled from and broadcast to
    mesh_node only.
  - ``"all_node_global"``: NOT implemented (v1). Passing it raises ValueError —
    see docs/paper_upgrade_global_node_design.md for the design.

Reference:
  - Virtual node: Gilmer et al., 2017 (message passing with a master node).
  - Global node in graph transformers (GPS: Rampášek et al., NeurIPS 2022).
  - Design doc: docs/paper_upgrade_global_node_design.md
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.utils import scatter

from src.models.baselines.decoders import MLPHead


# ---------------------------------------------------------------------------
# Canonical heterogeneous graph metadata (v2 dataset) — same as HGTBaseline
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

VALID_GLOBAL_MODES = ("mesh_global", "all_node_global", "off")


def derive_global_init_seed(seed: int) -> int:
    """Stable, build-order-independent RNG seed for the global module."""
    value = (int(seed) * 1000003 + 97) % (2 ** 31 - 1)
    return value if value else 1


class GlobalBroadcastFusion(nn.Module):
    """Fuse the broadcast global state back into mesh_node hidden states.

    Supports two methods (mirrors ``CrossScaleFusion`` in ms_hgt.py):
      - ``"additive"``: ``h_out = h_mesh + broadcast``
      - ``"gated_residual"``: ``h_out = h_mesh + gate * broadcast``
        where ``gate = sigmoid(MLP([h_mesh, broadcast]))``
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
            # Initialise gate bias -> ~0.12 (sigmoid(-2) ≈ 0.12) so the global
            # path starts with small influence (same init as CrossScaleFusion).
            self.gate_net[-1].bias.data.fill_(-2.0)
        elif method == "additive":
            pass
        else:
            raise ValueError(
                f"Unknown global_fusion '{method}'. "
                f"Options: 'additive', 'gated_residual'."
            )

    def forward(
        self,
        h_mesh: torch.Tensor,
        broadcast: torch.Tensor,
    ) -> torch.Tensor:
        """Apply global fusion.

        Args:
            h_mesh: ``(total_mesh_nodes, hidden_dim)``
            broadcast: ``(total_mesh_nodes, hidden_dim)`` — global state
                expanded per mesh node.

        Returns:
            ``(total_mesh_nodes, hidden_dim)``
        """
        if self.method == "additive":
            self._last_gate = None
            return h_mesh + broadcast
        elif self.method == "gated_residual":
            gate_input = torch.cat([h_mesh, broadcast], dim=-1)
            gate = torch.sigmoid(self.gate_net(gate_input))
            self._last_gate = gate.detach()
            return h_mesh + gate * broadcast
        else:
            self._last_gate = None
            return h_mesh


class HGTGlobalBaseline(nn.Module):
    """HGT backbone + one per-graph global (virtual) node.

    Args:
        mesh_feat_dim: Feature dimension of mesh_node (default 15).
        beam_feat_dim: Feature dimension of beam_element (default 11).
        plate_feat_dim: Feature dimension of plate_element (default 6).
        hidden_dim: Shared hidden dimension (default 128).
        num_layers: Number of HGTConv layers (default 3).
        heads: Number of attention heads per HGTConv (default 4).
        dropout: Dropout rate (default 0.1).
        activation: Activation name (``"relu"``, ``"gelu"``, ``"elu"``).
        use_layer_norm: Per-node-type LayerNorm after each layer (default True).
        decoder_hidden_dims: Hidden dims of decoder MLP heads.
        global_mode: ``"mesh_global"`` (default) — global state pooled from and
            broadcast to mesh_node only. ``"all_node_global"`` is designed but
            not implemented (v1); passing it raises ValueError.
        global_fusion: ``"gated_residual"`` (default) or ``"additive"``.
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
        global_mode: str = "mesh_global",
        global_fusion: str = "gated_residual",
        init_seed: Optional[int] = None,
        global_init_seed: Optional[int] = None,
    ):
        super().__init__()
        _rng_state = torch.get_rng_state() if init_seed is not None else None
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 32]

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        if global_mode not in VALID_GLOBAL_MODES:
            raise ValueError(
                f"Unknown global_mode '{global_mode}'. "
                f"Options: {list(VALID_GLOBAL_MODES)}"
            )
        if global_mode == "all_node_global":
            raise ValueError(
                "global_mode='all_node_global' is designed but NOT implemented "
                "in v1 (see docs/paper_upgrade_global_node_design.md). "
                "Use global_mode='mesh_global'."
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.global_mode = global_mode
        self.global_fusion = global_fusion

        # ---- Type-specific encoders (same as HGTBaseline) ----
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

        # ---- HGTConv layers (typed attention, same as HGTBaseline) ----
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

        # ---- Global (virtual) node machinery ----
        # The global module must not depend on construction order.  When the
        # shared backbone is seed-driven (init_seed given), its weights are
        # drawn from a separate stable stream derived from the run seed.
        self.init_seed = init_seed
        self.global_init_seed = (None if init_seed is None else
                                 int(global_init_seed if global_init_seed is not None
                                     else derive_global_init_seed(init_seed)))
        if self.global_init_seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.global_init_seed)
                self._build_global_modules(hidden_dim, global_fusion)
        else:
            self._build_global_modules(hidden_dim, global_fusion)

        # ---- Decoders (same as HGTBaseline) ----
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

        # Optional control: build the *shared* HGT backbone from a dedicated
        # RNG state so plain HGT and HGT-global start from identical shared
        # weights.  Global-module weights are drawn from their own stream and
        # are not copied.
        if init_seed is not None:
            from src.models.baselines.hgt_baseline import HGTBaseline
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(init_seed))
                ref = HGTBaseline(
                    mesh_feat_dim=mesh_feat_dim, beam_feat_dim=beam_feat_dim,
                    plate_feat_dim=plate_feat_dim, hidden_dim=hidden_dim,
                    num_layers=num_layers, heads=heads, dropout=dropout,
                    activation=activation, use_layer_norm=use_layer_norm,
                    decoder_hidden_dims=decoder_hidden_dims, init_seed=None,
                )
            own = self.state_dict()
            for key, value in ref.state_dict().items():
                if key in own:
                    own[key] = value
            self.load_state_dict(own)
            torch.set_rng_state(_rng_state)

        # Diagnostics buffer (most recent forward)
        self._last_global_stats: Optional[Dict] = None

    def _build_global_modules(self, hidden_dim: int, global_fusion: str) -> None:
        # Recurrent global memory: input = mean-pooled mesh hidden (B, H),
        # hidden = previous global state (B, H).
        self.global_cell = nn.GRUCell(hidden_dim, hidden_dim)
        # Project updated global state back to hidden space before broadcast.
        self.global_bcast = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = GlobalBroadcastFusion(
            hidden_dim=hidden_dim,
            method=global_fusion,
        )

    def _pool_mesh_global(
        self,
        h_mesh: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool mesh hidden states into per-graph global readout.

        Args:
            h_mesh: ``(total_mesh_nodes, hidden_dim)``
            batch_vec: ``(total_mesh_nodes,)`` per-node graph index.

        Returns:
            ``(B, hidden_dim)`` — one global state per graph.
        """
        B = int(batch_vec.max().item()) + 1
        pooled = scatter(
            h_mesh, batch_vec, dim=0, dim_size=B, reduce="mean",
        )
        return torch.nan_to_num(pooled, nan=0.0)

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            batch: ``HeteroDataBatch`` with 3 node types and 5 edge types.

        Returns:
            ``(pred_disp, pred_force)``:
                - pred_disp:  ``(total_mesh_nodes, 6)``
                - pred_force: ``(total_beam_elements, 12)``
        """
        device = batch["mesh_node"].x.device
        if self.global_mode == "off":
            self.fusion._last_gate = None

        # ---- 1. Type-specific projections (same as HGTBaseline) ----
        x_dict: Dict[str, torch.Tensor] = {
            "mesh_node": self.mesh_encoder(batch["mesh_node"].x),
            "beam_element": self.beam_encoder(batch["beam_element"].x),
            "plate_element": self.plate_encoder(batch["plate_element"].x),
        }

        # ---- 2. Edge index dict ----
        edge_index_dict = {}
        for et in EDGE_TYPES:
            edge_index_dict[et] = batch[et].edge_index

        # ---- 3. Global state bookkeeping ----
        batch_vec = batch["mesh_node"].batch  # (total_mesh_nodes,)
        B = int(batch_vec.max().item()) + 1
        g_prev = torch.zeros(B, self.hidden_dim, device=device)

        # ---- 4. HGTConv layers with global update + broadcast ----
        for conv in self.convs:
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

            # ---- Global path (mesh_global) ----
            # ``global_mode='off'`` runs the exact HGT-only path.  The global
            # parameters still exist but are never used; this is a *path*
            # check, not a capacity-matched control.
            if self.global_mode != "off":
                h_mesh = x_dict["mesh_node"]
                g_pooled = self._pool_mesh_global(h_mesh, batch_vec)  # (B, H)
                g_new = self.global_cell(g_pooled, g_prev)            # (B, H)
                g_prev = g_new
                broadcast = self.global_bcast(g_new)[batch_vec]       # (N, H)
                x_dict["mesh_node"] = self.fusion(h_mesh, broadcast)

        # ---- 5. Decode ----
        pred_disp = self.disp_decoder(x_dict["mesh_node"])
        pred_force = self.force_decoder(x_dict["beam_element"])

        # Diagnostics
        self._last_global_stats = {
            "global_mode": self.global_mode,
            "global_fusion": self.global_fusion,
            "n_graphs": B,
            "gate_stats": (
                None if self.global_fusion != "gated_residual"
                else {
                    "mean": float(self.fusion._last_gate.mean().item()),
                    "std": float(self.fusion._last_gate.std().item()),
                }
                if getattr(self.fusion, "_last_gate", None) is not None
                else None
            ),
        }

        return pred_disp, pred_force

    def get_global_stats(self) -> Optional[Dict]:
        """Return latest global-node diagnostics (from most recent forward)."""
        return getattr(self, "_last_global_stats", None)

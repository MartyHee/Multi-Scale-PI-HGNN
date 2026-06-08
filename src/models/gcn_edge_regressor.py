"""
gcn_edge_regressor.py — GCN baseline for edge-level regression.

Architecture:
  Node encoder:
    Stack of GCNConv layers that propagate information across the graph
    topology to produce enriched node representations.

  Edge decoder:
    For each edge, construct a feature vector by concatenating:
      - source node representation  h_i  (node_hidden_dim)
      - target node representation  h_j  (node_hidden_dim)
      - edge attributes                  (edge_attr_dim, default 4)

    The concatenated vector is fed through a decoder MLP to produce
    12-dim edge-level predictions.

  Forward::
      h = GCNEncoder(x, edge_index)              # (N, node_hidden_dim)
      h_i = h[row], h_j = h[col]                 # (E, node_hidden_dim)
      edge_input = concat(h_i, h_j, edge_attr)   # (E, 2*node_hidden_dim + 4)
      return decoder(edge_input)                  # (E, 12)

This is the simplest meaningful graph baseline — it tests whether
neighbour-aware node representations improve over per-edge MLP.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Registry of activation constructors (shared with MLP model)
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class GCNEdgeRegressor(nn.Module):
    """GCN node encoder + edge decoder for edge-level regression.

    Parameters:
        node_in_dim: Input node feature dimension (default 9).
        edge_attr_dim: Input edge attribute dimension (default 4).
        node_hidden_dim: Hidden dimension for GCN layers and node representations.
        num_layers: Number of GCNConv layers.
        dropout: Dropout rate (applied after activation in both encoder and decoder).
        activation: Activation function name.
        use_batch_norm: Whether to insert BatchNorm1d after GCNConv layers.
        decoder_hidden_dims: Hidden layer widths for the decoder MLP.
        output_dim: Target dimension (default 12).
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_attr_dim: int = 4,
        node_hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        decoder_hidden_dims: Optional[List[int]] = None,
        output_dim: int = 12,
    ):
        super().__init__()

        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 64]

        act_cls = ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. Options: {list(ACTIVATIONS.keys())}"
            )

        # ---- Node encoder: stacked GCNConv layers ----
        encoder_layers: List[nn.Module] = []
        bn_layers: List[nn.Module] = []

        in_dim = node_in_dim
        for _ in range(num_layers):
            encoder_layers.append(GCNConv(in_dim, node_hidden_dim))
            if use_batch_norm:
                bn_layers.append(nn.BatchNorm1d(node_hidden_dim))
            in_dim = node_hidden_dim

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.encoder_act = act_cls()
        self.encoder_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- Edge decoder: MLP ----
        decoder_input_dim = 2 * node_hidden_dim + edge_attr_dim  # h_i + h_j + edge_attr

        decoder_layers: List[nn.Module] = []
        in_dim = decoder_input_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(act_cls())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # Store config for extra_repr
        self._config = {
            "node_in_dim": node_in_dim,
            "edge_attr_dim": edge_attr_dim,
            "node_hidden_dim": node_hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "activation": activation,
            "use_batch_norm": use_batch_norm,
            "decoder_hidden_dims": decoder_hidden_dims,
            "output_dim": output_dim,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass — GCN node encoding + per-edge decoder.

        Args:
            x: Node features, shape ``(num_nodes, node_in_dim)``.
            edge_index: Graph connectivity, shape ``(2, num_edges)``.
            edge_attr: Edge attributes, shape ``(num_edges, edge_attr_dim)``.

        Returns:
            Edge-level predictions, shape ``(num_edges, output_dim)``.
        """
        # ---- Node encoder ----
        h = x
        for i, conv in enumerate(self.encoder_layers):
            h = conv(h, edge_index)
            if i < len(self.bn_layers):
                h = self.bn_layers[i](h)
            h = self.encoder_act(h)
            h = self.encoder_dropout(h)

        # ---- Edge decoder ----
        row, col = edge_index  # row: source (I), col: target (J)
        h_i = h[row]  # (E, node_hidden_dim)
        h_j = h[col]  # (E, node_hidden_dim)

        edge_input = torch.cat([h_i, h_j, edge_attr], dim=1)  # (E, 2*hidden + edge_attr_dim)
        return self.decoder(edge_input)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Inference shortcut (no grad)."""
        self.eval()
        return self.forward(x, edge_index, edge_attr)

    def extra_repr(self) -> str:
        cfg = self._config
        return (
            f"node_in_dim={cfg['node_in_dim']}, "
            f"node_hidden_dim={cfg['node_hidden_dim']}, "
            f"num_layers={cfg['num_layers']}, "
            f"decoder_hidden_dims={cfg['decoder_hidden_dims']}, "
            f"output_dim={cfg['output_dim']}"
        )

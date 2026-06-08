"""
mlp_edge_regressor.py — Per-edge MLP baseline for edge-level regression.

Architecture:
  For each edge, construct a feature vector by concatenating:
    - source node features  x_i  (9 dim)
    - target node features  x_j  (9 dim)
    - edge attributes             (4 dim)

  Total per-edge input: 22 dim → stacked MLP → 12 dim output.

  The MLP applies ``Linear → BatchNorm1d (optional) → Activation → Dropout``
  at each hidden layer, then a final Linear projection to output_dim.

No message passing — this is a baseline that treats each edge independently,
conditioned on its endpoint node states and section properties.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


# Registry of activation constructors
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class MLPEdgeRegressor(nn.Module):
    """Per-edge MLP regressor.

    Parameters:
        input_dim: Per-edge feature dimension (default 22 = 9+9+4).
        output_dim: Target dimension (default 12).
        hidden_dims: List of hidden layer widths.
        dropout: Dropout rate (applied after activation).
        activation: Activation function name (see ACTIVATIONS dict).
        use_batch_norm: Whether to insert BatchNorm1d after each linear layer.
    """

    def __init__(
        self,
        input_dim: int = 22,
        output_dim: int = 12,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        act_cls = ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. Options: {list(ACTIVATIONS.keys())}"
            )

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection (no bn/activation/dropout)
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dims = hidden_dims

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass — per-edge regression.

        Args:
            x: Node features, shape ``(num_nodes, input_dim - 4)``,
               typically ``(N, 9)``.
            edge_index: Graph connectivity, shape ``(2, num_edges)``.
                ``edge_index[0]`` = source (I-end), ``edge_index[1]`` = target (J-end).
            edge_attr: Edge attributes, shape ``(num_edges, 4)``.

        Returns:
            Edge-level predictions, shape ``(num_edges, output_dim)``.
        """
        row, col = edge_index  # row: source (I), col: target (J)

        # Gather endpoint node features
        x_i = x[row]  # (E, 9)
        x_j = x[col]  # (E, 9)

        # Concatenate → (E, 9+9+4) = (E, 22)
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)

        return self.mlp(edge_input)

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
        return (
            f"input_dim={self._input_dim}, output_dim={self._output_dim}, "
            f"hidden_dims={self._hidden_dims}"
        )

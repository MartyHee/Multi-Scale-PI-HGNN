"""
decoders.py — MLP decoders / heads for baseline models.

Provides generic ``MLPHead`` and convenience aliases ``DispDecoder`` (6-dim)
and ``ForceDecoder`` (12-dim).

Usage::

    decoder = MLPHead(input_dim=128, output_dim=6, hidden_dims=[64, 32])
    pred = decoder(hidden_states)                          # (N, 6)
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn


class MLPHead(nn.Module):
    """Generic MLP regression head.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension (e.g. 6 for displacement, 12 for force).
        hidden_dims: List of hidden layer widths.
        dropout: Dropout rate after each activation.
        activation: Activation function name (``"relu"``, ``"gelu"``, ``"elu"``).
        use_batch_norm: If ``True``, insert ``BatchNorm1d`` after each Linear
            (except the final output layer).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "elu":
            return nn.ELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()

    def forward(self, x):
        return self.net(x)


# Convenience aliases
DispDecoder = MLPHead  # output_dim=6 expected
ForceDecoder = MLPHead  # output_dim=12 expected

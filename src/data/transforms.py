"""
Feature standardization transforms for baseline graph dataset.

Uses Welford's online algorithm (also known as the "Welford online algorithm"
or "streaming algorithm for variance") to compute mean/std on the training set
without materialising all graphs at once.

Usage::

    # Phase 1: fit on training data
    scaler = NodeEdgeStandardScaler()
    for data in train_loader:
        scaler.update(data)

    node_mean, node_std, edge_mean, edge_std, target_mean, target_std = scaler.finalize()
    scaler.save("feature_stats.json")

    # Phase 2: transform (in Dataset __getitem__)
    scaler = NodeEdgeStandardScaler.load("feature_stats.json")
    data = scaler(data)          # in-place standardisation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


class WelfordAccumulator:
    """Online mean / variance accumulator using Welford's algorithm.

    Maintains per-dimension running mean and M2 (sum of squared deviations
    from the current mean) for arbitrarily many observations.
    """

    def __init__(self, num_features: int, device: torch.device = torch.device("cpu")):
        self.num_features = num_features
        self.count: float = 0.0
        self.mean = torch.zeros(num_features, dtype=torch.float64, device=device)
        self.M2 = torch.zeros(num_features, dtype=torch.float64, device=device)

    def update(self, values: torch.Tensor) -> None:
        """Update accumulators with a batch of values.

        Args:
            values: Shape (N, num_features) tensor.
        """
        n = values.shape[0]
        if n == 0:
            return
        # Cast to float64 for numerical stability
        values = values.to(dtype=torch.float64, device=self.mean.device)
        self.count += n
        # Welford update: delta, delta2, mean, M2
        delta = values - self.mean.unsqueeze(0)          # (N, F)
        self.mean += delta.sum(dim=0) / self.count       # (F,)
        delta2 = values - self.mean.unsqueeze(0)          # (N, F) with updated mean
        self.M2 += (delta * delta2).sum(dim=0)            # (F,)

    def finalize(self, bias_corrected: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) tensors (float32)."""
        if self.count < 2:
            mean = self.mean.float()
            std = torch.ones_like(mean)
            return mean, std
        if bias_corrected:
            var = self.M2 / (self.count - 1.0)   # sample variance
        else:
            var = self.M2 / self.count            # population variance
        std = var.sqrt().float()
        # Guard against zero std → replace with 1 so transform becomes identity
        std = torch.where(std < 1e-12, torch.ones_like(std), std)
        return self.mean.float(), std

    def state_dict(self) -> Dict:
        return {
            "num_features": self.num_features,
            "count": self.count,
            "mean": self.mean.tolist(),
            "M2": self.M2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "WelfordAccumulator":
        acc = cls(state["num_features"])
        acc.count = state["count"]
        acc.mean = torch.tensor(state["mean"], dtype=torch.float64)
        acc.M2 = torch.tensor(state["M2"], dtype=torch.float64)
        return acc


# ============================================================
# StandardScaler
# ============================================================

class NodeEdgeStandardScaler:
    """Standardise node features, edge features, and edge targets.

    Fits on training data via WelfordAccumulator, then applies:
        x_std = (x - mean) / std

    Operates **in-place** on Data objects by default.
    """

    def __init__(self):
        self.node_acc: Optional[WelfordAccumulator] = None
        self.edge_acc: Optional[WelfordAccumulator] = None
        self.target_acc: Optional[WelfordAccumulator] = None
        self._fitted = False
        # Stored mean/std after finalize (avoids re-computing from M2)
        self._node_mean: Optional[torch.Tensor] = None
        self._node_std: Optional[torch.Tensor] = None
        self._edge_mean: Optional[torch.Tensor] = None
        self._edge_std: Optional[torch.Tensor] = None
        self._target_mean: Optional[torch.Tensor] = None
        self._target_std: Optional[torch.Tensor] = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def update(self, data) -> None:
        """Accumulate statistics from one Data object.

        Called repeatedly over the training set.

        Shape conventions:
            data.x:       (num_nodes, node_feature_dim)
            data.edge_attr: (num_edges, edge_feature_dim)
            data.y_edge:   (num_edges, target_dim)
        """
        if self.node_acc is None:
            self.node_acc = WelfordAccumulator(data.x.shape[1])
            self.edge_acc = WelfordAccumulator(data.edge_attr.shape[1])
            self.target_acc = WelfordAccumulator(data.y_edge.shape[1])

        # Each node/edge row is one "observation"
        self.node_acc.update(data.x)
        self.edge_acc.update(data.edge_attr)
        self.target_acc.update(data.y_edge)

    def finalize(self, bias_corrected: bool = True) -> Dict[str, torch.Tensor]:
        """Finalise accumulators and return stats dict."""
        nm, ns = self.node_acc.finalize(bias_corrected)
        em, es = self.edge_acc.finalize(bias_corrected)
        tm, ts = self.target_acc.finalize(bias_corrected)

        self._node_mean = nm
        self._node_std = ns
        self._edge_mean = em
        self._edge_std = es
        self._target_mean = tm
        self._target_std = ts
        self._fitted = True

        return {
            "node_mean": nm, "node_std": ns,
            "edge_mean": em, "edge_std": es,
            "target_mean": tm, "target_std": ts,
        }

    @property
    def node_mean(self) -> torch.Tensor:
        return self._node_mean if self._fitted else (
            self.node_acc.mean.float() if self.node_acc else torch.tensor([]))

    @property
    def node_std(self) -> torch.Tensor:
        return self._node_std if self._fitted else torch.tensor([])

    @property
    def edge_mean(self) -> torch.Tensor:
        return self._edge_mean if self._fitted else (
            self.edge_acc.mean.float() if self.edge_acc else torch.tensor([]))

    @property
    def edge_std(self) -> torch.Tensor:
        return self._edge_std if self._fitted else torch.tensor([])

    @property
    def target_mean(self) -> torch.Tensor:
        return self._target_mean if self._fitted else (
            self.target_acc.mean.float() if self.target_acc else torch.tensor([]))

    @property
    def target_std(self) -> torch.Tensor:
        return self._target_std if self._fitted else torch.tensor([])

    def save(self, json_path: Path) -> None:
        """Save fitted statistics to JSON."""
        if not self._fitted:
            raise RuntimeError("Cannot save before finalize().")

        state = {
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"  [ok] Feature stats saved: {json_path}")

    @classmethod
    def load(cls, json_path: Path) -> "NodeEdgeStandardScaler":
        """Load fitted statistics from JSON and return a ready-to-use scaler."""
        with open(json_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        scaler = cls()
        scaler._node_mean = torch.tensor(state["node_mean"], dtype=torch.float32)
        scaler._node_std = torch.tensor(state["node_std"], dtype=torch.float32)
        scaler._edge_mean = torch.tensor(state["edge_mean"], dtype=torch.float32)
        scaler._edge_std = torch.tensor(state["edge_std"], dtype=torch.float32)
        scaler._target_mean = torch.tensor(state["target_mean"], dtype=torch.float32)
        scaler._target_std = torch.tensor(state["target_std"], dtype=torch.float32)
        scaler._fitted = True
        return scaler

    def __call__(self, data, inplace: bool = True):
        """Apply standardisation to a Data object.

        Args:
            data: PyG Data object with .x, .edge_attr, .y_edge.
            inplace: If True, modify in-place.

        Returns:
            Standardised Data object.
        """
        if not self._fitted:
            raise RuntimeError("NodeEdgeStandardScaler not fitted. Call finalize() first.")

        d = data if inplace else data.clone()
        d.x = (d.x - self._node_mean) / self._node_std
        d.edge_attr = (d.edge_attr - self._edge_mean) / self._edge_std
        d.y_edge = (d.y_edge - self._target_mean) / self._target_std
        return d

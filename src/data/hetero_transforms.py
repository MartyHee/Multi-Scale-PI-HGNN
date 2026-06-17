"""
hetero_transforms.py — Feature standardisation for heterogeneous graph data.

Provides ``HeteroFeatureScaler`` that computes per-node-type (and per-target)
mean / std from the training set and applies Z-score normalisation to
``torch_geometric.data.HeteroData`` objects.

Updated for v2: handles ``structural_link`` (was ``rigid_link``) and reverse
membership edges with ``edge_attr``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.data import HeteroData

from src.data.hetero_schema import (
    EDGE_TYPES_WITH_ATTR,
    NODE_TYPES,
    TARGET_NODE_TYPES,
)


# ============================================================
# WelfordAccumulator (online mean/variance)
# ============================================================

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
            values: Shape ``(N, num_features)`` tensor.
        """
        n = values.shape[0]
        if n == 0:
            return
        values = values.to(dtype=torch.float64, device=self.mean.device)
        self.count += n
        delta = values - self.mean.unsqueeze(0)
        self.mean += delta.sum(dim=0) / self.count
        delta2 = values - self.mean.unsqueeze(0)
        self.M2 += (delta * delta2).sum(dim=0)

    def finalize(self, bias_corrected: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` as float32 tensors."""
        if self.count < 2:
            mean = self.mean.float()
            std = torch.ones_like(mean)
            return mean, std
        var = self.M2 / (self.count - 1.0) if bias_corrected else self.M2 / self.count
        std = var.sqrt().float()
        # Guard against zero std → identity transform
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
# HeteroFeatureScaler
# ============================================================

class HeteroFeatureScaler:
    """Standardise node features and targets for HeteroData graphs.

    Accumulates per-node-type (and per-target) statistics using Welford's
    online algorithm across the training set, then applies Z-score
    normalisation in ``__call__``.

    Usage::

        # Fit
        scaler = HeteroFeatureScaler()
        for data in train_graphs:
            scaler.update(data)
        scaler.finalize()
        scaler.save(path)

        # Transform (in Dataset.__getitem__)
        scaler = HeteroFeatureScaler.load(path)
        data = scaler(data)          # in-place normalisation
    """

    # Attributes that are *not* to be standardised
    SKIP_NODE_ATTRS = {"num_nodes", "node_id", "edge_index"}
    SKIP_EDGE_ATTRS = {"edge_index", "num_edges"}

    def __init__(self):
        # Accumulators: key = (scope, type_name, attr_name)
        self._accs: Dict[str, WelfordAccumulator] = {}
        self._fitted = False

        # Stored stats after finalize: key -> (mean, std) tensors
        self._stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Track which keys exist for transform
        self._transform_keys: List[str] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def update(self, data: HeteroData) -> None:
        """Accumulate statistics from one HeteroData graph."""
        if not isinstance(data, HeteroData):
            raise TypeError(f"Expected HeteroData, got {type(data).__name__}")

        # --- Node features ---
        for ntype in data.node_types:
            store = data[ntype]
            for attr_key in store.node_attrs():
                if attr_key in self.SKIP_NODE_ATTRS:
                    continue
                tensor: torch.Tensor = store[attr_key]
                if tensor.ndim != 2:
                    continue
                acc_key = f"node:{ntype}:{attr_key}"
                if acc_key not in self._accs:
                    self._accs[acc_key] = WelfordAccumulator(tensor.shape[1])
                self._accs[acc_key].update(tensor)

        # --- Edge features ---
        for etype in data.edge_types:
            if etype not in EDGE_TYPES_WITH_ATTR:
                continue
            store = data[etype]
            if "edge_attr" not in store.keys():
                continue
            tensor = store["edge_attr"]
            if tensor.ndim != 2:
                continue
            ekey_str = f"edge:{etype[0]}->{etype[1]}->{etype[2]}"
            acc_key = f"{ekey_str}:edge_attr"
            if acc_key not in self._accs:
                self._accs[acc_key] = WelfordAccumulator(tensor.shape[1])
            self._accs[acc_key].update(tensor)

    def finalize(self, bias_corrected: bool = True) -> Dict[str, Any]:
        """Finalise accumulators and return a dict of statistics."""
        self._stats = {}
        self._transform_keys = []

        for acc_key, acc in self._accs.items():
            mean, std = acc.finalize(bias_corrected)
            self._stats[f"{acc_key}_mean"] = mean
            self._stats[f"{acc_key}_std"] = std
            self._transform_keys.append(acc_key)

        self._fitted = True
        return self.get_stats_dict()

    def get_stats_dict(self) -> Dict[str, Any]:
        """Return a nested dict with all fitted stats."""
        result = {}
        for acc_key in self._transform_keys:
            mean = self._stats[f"{acc_key}_mean"]
            std = self._stats[f"{acc_key}_std"]
            result[acc_key] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, json_path: Path) -> None:
        """Save fitted statistics to JSON."""
        if not self._fitted:
            raise RuntimeError("Cannot save before finalize().")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.get_stats_dict(), f, indent=2, ensure_ascii=False)
        print(f"  [ok] HeteroFeatureScaler saved: {json_path}")

    @classmethod
    def load(cls, json_path: Path) -> "HeteroFeatureScaler":
        """Load fitted statistics from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        scaler = cls()
        scaler._fitted = True
        for acc_key, entry in state.items():
            scaler._stats[f"{acc_key}_mean"] = torch.tensor(entry["mean"], dtype=torch.float32)
            scaler._stats[f"{acc_key}_std"] = torch.tensor(entry["std"], dtype=torch.float32)
            scaler._transform_keys.append(acc_key)
        return scaler

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def __call__(self, data: HeteroData, inplace: bool = True) -> HeteroData:
        """Apply standardisation to a HeteroData object (in-place by default).

        Node features (``x``) and node targets (``y_disp``, ``y_force``, etc.)
        are standardised via ``(t - mean) / std`` using training-set statistics.
        Edge attributes are also standardised if they appear in EDGE_TYPES_WITH_ATTR.
        """
        if not self._fitted:
            raise RuntimeError("HeteroFeatureScaler not fitted. Call finalize() first.")

        d = data if inplace else data.clone()

        for acc_key in self._transform_keys:
            parts = acc_key.split(":", 2)
            if len(parts) != 3:
                continue
            scope, type_name, attr_name = parts

            if scope == "node":
                if type_name not in d.node_types or attr_name not in d[type_name]:
                    continue
                tensor = d[type_name][attr_name]
                mean = self._stats[f"{acc_key}_mean"]
                std = self._stats[f"{acc_key}_std"]
                mean = mean.to(tensor.device, dtype=tensor.dtype)
                std = std.to(tensor.device, dtype=tensor.dtype)
                d[type_name][attr_name] = (tensor - mean) / std

            elif scope == "edge":
                # Reconstruct edge type tuple from string
                # e.g. "edge:mesh_node->structural_link->mesh_node"
                etype_str = type_name
                parts_etype = etype_str.split("->")
                if len(parts_etype) != 3:
                    continue
                etype_tuple = (parts_etype[0], parts_etype[1], parts_etype[2])
                if etype_tuple not in d.edge_types:
                    continue
                if attr_name not in d[etype_tuple]:
                    continue
                tensor = d[etype_tuple][attr_name]
                mean = self._stats[f"{acc_key}_mean"]
                std = self._stats[f"{acc_key}_std"]
                mean = mean.to(tensor.device, dtype=tensor.dtype)
                std = std.to(tensor.device, dtype=tensor.dtype)
                d[etype_tuple][attr_name] = (tensor - mean) / std

        return d

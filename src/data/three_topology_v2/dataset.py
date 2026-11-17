"""Lazy normalized dataset for one fold of the three-topology v2 inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from . import graph_io, paths, stats as stats_mod


class FoldDataset(Dataset):
    """Return normalised ``ThreeTopoData`` for a fixed list of sample IDs."""

    def __init__(
        self,
        index: Dict[str, Any],
        sample_ids: Sequence[str],
        fold_stats: Dict[str, Any],
        root: Optional[Path] = None,
        dtype: torch.dtype = torch.float32,
        verify_hash: bool = True,
        normalize_labels: bool = True,
    ):
        self.root = root or paths.package_root()
        self.row_by_id = index["row_by_id"]
        self.sample_ids = list(sample_ids)
        self.stats = fold_stats
        self.dtype = dtype
        self.verify_hash = verify_hash
        self.normalize_labels = normalize_labels

    def __len__(self) -> int:
        return len(self.sample_ids)

    def raw(self, sid: str):
        row = self.row_by_id[sid]
        return graph_io.load_graph(row, root=self.root, dtype=self.dtype,
                                   verify_hash=self.verify_hash)

    def __getitem__(self, index):
        sid = self.sample_ids[index] if not isinstance(index, str) else index
        data = self.raw(sid)
        stats_mod.apply_normalization_(data, self.stats)
        if self.normalize_labels:
            stats_mod.normalize_labels_(data, self.stats)
        data.sample_id_str = sid
        return data

    def labels_physical(self, sid: str):
        """Return (y_disp, y_force) in physical units for a sample."""
        data = self.raw(sid)
        return data["mesh_node"].y_disp, data["beam_element"].y_force

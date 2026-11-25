"""Lazy dataset over the 9,216 FE v2 package."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from . import assemble, graph
from .reader import FeV2Reader


class FeV2Dataset(Dataset):
    def __init__(self, index, sample_ids: Sequence[str], stats: Dict[str, Any],
                 reader: FeV2Reader | None = None, dtype=torch.float32,
                 normalize: bool = True, use_shared: bool = False,
                 shared=None, topology: str | None = None):
        self.index = index
        self.sample_ids = list(sample_ids)
        self.stats = stats
        self.dtype = dtype
        self.normalize = normalize
        self.use_shared = bool(use_shared)
        self.shared = shared
        if self.use_shared and self.shared is None:
            from .shared_cache import SharedCache
            topo = topology or index.row(self.sample_ids[0])["topology_token"]
            self.shared = SharedCache(index, topo, reader=reader, verify=True)
        self.reader = reader or (None if self.use_shared else
                                 FeV2Reader(index, max_runs=2, verify_hash=True))
        self.topology = topology

    def __len__(self) -> int:
        return len(self.sample_ids)

    def raw(self, sid: str):
        if self.use_shared:
            return self.shared.assemble(sid)
        return assemble.assemble_raw(self.reader.read(sid))

    def __getitem__(self, idx):
        sid = self.sample_ids[idx] if not isinstance(idx, str) else idx
        raw = self.raw(sid)
        data = graph.build_graph(raw, dtype=self.dtype)
        if self.normalize:
            graph.apply_normalization_(data, self.stats)
        return data


class MultiSliceDataset(Dataset):
    """A view with a named validation slice, used by MultiSliceEvaluator."""
    def __init__(self, dataset: FeV2Dataset, name: str):
        self.dataset = dataset
        self.name = name
        self.sample_ids = dataset.sample_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

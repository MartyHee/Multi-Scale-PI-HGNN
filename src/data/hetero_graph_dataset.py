"""
hetero_graph_dataset.py — Dataset for loading heterogeneous graph data.

Supports both v1 and v2 datasets. v2 is the canonical schema (no link_element,
structural_link instead of rigid_link).

Usage::

    from src.data.hetero_graph_dataset import HeteroGraphDataset
    from src.data.hetero_transforms import HeteroFeatureScaler

    scaler = HeteroFeatureScaler.load("processed/hetero_graph_dataset_v2/feature_stats.json")

    dataset = HeteroGraphDataset(
        processed_dir="processed/hetero_graph_dataset_v2",
        split="train",
        split_mode="by_sample",
        transform=scaler,
    )

    print(len(dataset))
    data = dataset[0]   # torch_geometric.data.HeteroData
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class HeteroGraphDataset(Dataset):
    """Dataset for the heterogeneous graph dataset (v2 canonical).

    Reads ``index.csv`` + split files from the processed directory and returns
    individual ``torch_geometric.data.HeteroData`` objects filtered by split.

    Parameters:
        processed_dir: Path or str to ``processed/hetero_graph_dataset_v2/``.
        split: One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
        split_mode: ``"by_sample"`` or ``"by_loadcase"``.
            Ignored when split="all".
        transform: Optional callable to apply to each HeteroData object.
            Typically a ``HeteroFeatureScaler`` instance.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        split: str = "train",
        split_mode: str = "by_sample",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.processed_dir = Path(processed_dir).resolve()
        self.split = split
        self.split_mode = split_mode
        self.transform = transform

        if not self.processed_dir.is_dir():
            raise NotADirectoryError(
                f"Processed directory not found: {self.processed_dir}"
            )

        # Load index
        index_path = self.processed_dir / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"index.csv not found in {self.processed_dir}"
            )
        self._index = pd.read_csv(index_path)
        self._index["loadcase_id"] = self._index["loadcase_id"].astype(int)

        # Filter by split membership
        if split != "all":
            split_path = (
                self.processed_dir
                / "splits"
                / f"split_{split_mode}.json"
            )
            if not split_path.is_file():
                raise FileNotFoundError(
                    f"Split file not found: {split_path}. "
                    f"Run build_hetero_graph_dataset.py first."
                )
            with open(split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            split_ids = split_data[split]

            if split_mode == "by_sample":
                self._split_ids = set(int(x) for x in split_ids)
                mask = self._index["sample_id"].isin(self._split_ids)
            elif split_mode == "by_loadcase":
                self._split_ids = set(int(x) for x in split_ids)
                mask = self._index["loadcase_id"].isin(self._split_ids)
            else:
                raise ValueError(f"Unknown split_mode: {split_mode}")

            self._index = self._index[mask].reset_index(drop=True)
        else:
            self._split_ids = None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> HeteroData:
        """Load and return one HeteroData graph."""
        row = self._index.iloc[idx]

        # Cross-platform path normalisation: index.csv was generated on Windows
        # and contains backslash separators (e.g. "graphs\\1274\\0001.pt").
        # On Linux backslash is not a path separator, so we normalise to "/"
        # before passing to pathlib.Path.
        graph_rel = str(row["graph_file"]).replace("\\", "/")
        graph_path = self.processed_dir / graph_rel
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Graph file not found: {graph_path} "
                f"(graph_id={row['graph_id']})"
            )
        data: HeteroData = torch.load(graph_path, weights_only=False)

        # Attach metadata
        data.graph_id = row["graph_id"]
        data.sample_id = row["sample_id"]
        data.loadcase_id = int(row["loadcase_id"])

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_index(self) -> pd.DataFrame:
        """Return the (filtered) index DataFrame for inspection."""
        return self._index.copy()

    @staticmethod
    def check_path_normalisation(processed_dir: Union[str, Path], n: int = 5) -> dict:
        """Check that graph_file paths from index.csv resolve correctly.

        Reads the first ``n`` entries from ``index.csv``, normalises paths, and
        reports whether each graph file exists.  Useful as a cross-platform
        sanity check after copying the dataset to Linux.

        Returns a dict with ``total``, ``ok``, ``failed``, and ``samples``.
        """
        processed_dir = Path(processed_dir).resolve()
        index_path = processed_dir / "index.csv"
        if not index_path.is_file():
            return {"error": f"index.csv not found: {index_path}"}

        import pandas as pd
        index = pd.read_csv(index_path)
        samples = []
        ok = 0
        failed = 0

        for i in range(min(n, len(index))):
            raw = str(index.iloc[i]["graph_file"])
            norm = raw.replace("\\", "/")
            full = processed_dir / norm
            exists = full.is_file()
            samples.append({
                "raw": raw,
                "normalised": str(norm),
                "full_path": str(full),
                "exists": exists,
            })
            if exists:
                ok += 1
            else:
                failed += 1

        return {
            "total": len(index),
            "checked": len(samples),
            "ok": ok,
            "failed": failed,
            "samples": samples,
        }

    def __repr__(self) -> str:
        return (
            f"HeteroGraphDataset(split={self.split}, mode={self.split_mode}, "
            f"num_graphs={len(self)}, "
            f"root={self.processed_dir.name})"
        )

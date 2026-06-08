"""
graph_dataset.py — Dataset for loading processed baseline graph dataset.

Usage::

    from src.data.graph_dataset import GraphDataset

    dataset = GraphDataset(
        processed_dir="processed/graph_dataset_baseline",
        split="train",
        split_mode="by_sample",
        transform=None,            # e.g. NodeEdgeStandardScaler
    )

    print(len(dataset))            # number of graphs in this split
    data = dataset[0]              # torch_geometric.data.Data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Ensure project root on sys.path for config loading if called directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class GraphDataset(Dataset):
    """Dataset for the baseline graph dataset.

    Reads index.csv + split files from the processed directory and returns
    individual graph ``torch_geometric.data.Data`` objects by split membership.

    Parameters:
        processed_dir: Path or str to ``processed/graph_dataset_baseline/``.
        split: One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
        split_mode: ``"by_sample"`` or ``"by_loadcase"``.
            Ignored when split="all".
        transform: Optional callable to apply to each Data object.
            Typically a ``NodeEdgeStandardScaler`` instance.
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
                    f"Run build_graph_dataset.py first."
                )
            with open(split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            split_ids = split_data[split]

            if split_mode == "by_sample":
                # Convert both to int for safe comparison
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

    def __getitem__(self, idx: int) -> Data:
        """Load and return one graph Data object."""
        row = self._index.iloc[idx]
        graph_path = self.processed_dir / row["graph_file"]
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Graph file not found: {graph_path} "
                f"(graph_id={row['graph_id']})"
            )
        data: Data = torch.load(graph_path, weights_only=False)
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

    def __repr__(self) -> str:
        return (
            f"GraphDataset(split={self.split}, mode={self.split_mode}, "
            f"num_graphs={len(self)}, "
            f"root={self.processed_dir.name})"
        )

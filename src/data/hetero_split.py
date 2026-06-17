"""
hetero_split.py — Train/val/test split generation for heterogeneous graph dataset.

Reuses the split logic from ``src.data.split`` (same strategy: by_sample,
by_loadcase) but adapted for the hetero dataset context.

Both splits live under ``processed/hetero_graph_dataset_v1/splits/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.data.split import (
    generate_all_splits as _generate_all_splits,
    load_split,
    print_split_summary,
)


def generate_split_files(
    sample_ids: List[str],
    loadcase_ids: List[int],
    split_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Generate and save both split strategies, return the splits.

    Delegates to ``src.data.split.generate_all_splits``.
    """
    return _generate_all_splits(
        sample_ids=sample_ids,
        loadcase_ids=loadcase_ids,
        split_dir=split_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )


def load_split_file(split_path: Path) -> Dict[str, list]:
    """Load a single split JSON file."""
    return load_split(split_path)

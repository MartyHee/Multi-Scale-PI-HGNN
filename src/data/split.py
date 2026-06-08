"""
Split generation for baseline graph dataset.

Provides two split strategies:
  - by_sample:  No SampleID overlap between train/val/test.
  - by_loadcase: No LoadCaseId overlap between train/val/test.

Each split is saved as a JSON file listing graph_ids per set.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    return obj


def _indices_from_ratio(n: int, train_r: float, val_r: float, test_r: float, seed: int
                        ) -> Dict[str, List[int]]:
    """Shuffle indices 0..n-1 and split by ratio."""
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = round(n * train_r)
    n_val   = round(n * val_r)
    # ensure exact total
    remainder = n - n_train - n_val
    n_test = remainder
    # distribute rounding error to largest split
    while n_train + n_val + n_test < n:
        n_test += 1
    while n_train + n_val + n_test > n:
        n_test -= 1

    return {
        "train": sorted(indices[:n_train]),
        "val":   sorted(indices[n_train:n_train + n_val]),
        "test":  sorted(indices[n_train + n_val:]),
    }


def split_by_sample(
    sample_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """Split sample IDs into train/val/test with no overlap.

    Returns dict like {"train": ["1177", ...], "val": [...], "test": [...]}.
    """
    idx_split = _indices_from_ratio(len(sample_ids), train_ratio, val_ratio, test_ratio, random_seed)
    return {
        set_name: [sample_ids[i] for i in idx_list]
        for set_name, idx_list in idx_split.items()
    }


def split_by_loadcase(
    loadcase_ids: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, List[int]]:
    """Split LoadCaseIds into train/val/test with no overlap."""
    # Ensure native Python ints for JSON serialisation
    loadcase_ids = [int(x) for x in loadcase_ids]
    idx_split = _indices_from_ratio(len(loadcase_ids), train_ratio, val_ratio, test_ratio, random_seed)
    return {
        set_name: sorted([loadcase_ids[i] for i in idx_list])
        for set_name, idx_list in idx_split.items()
    }


def save_split(split: Dict[str, list], save_path: Path) -> None:
    """Save a single split definition to a JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(split), f, indent=2, ensure_ascii=False)
    print(f"  [ok] Split saved: {save_path}")


def load_split(split_path: Path) -> Dict[str, list]:
    """Load a split definition from a JSON file."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_all_splits(
    sample_ids: List[str],
    loadcase_ids: List[int],
    split_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Generate and save both split strategies.

    Returns nested dict: {split_name: {set_name: [ids]}}.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    splits = {}

    # by_sample
    print("Generating split_by_sample ...")
    by_sample = split_by_sample(sample_ids, train_ratio, val_ratio, test_ratio, random_seed)
    save_split(by_sample, split_dir / "split_by_sample.json")
    splits["split_by_sample"] = by_sample

    # by_loadcase
    print("Generating split_by_loadcase ...")
    by_loadcase = split_by_loadcase(loadcase_ids, train_ratio, val_ratio, test_ratio, random_seed)
    save_split(by_loadcase, split_dir / "split_by_loadcase.json")
    splits["split_by_loadcase"] = by_loadcase

    return splits


def print_split_summary(splits: Dict[str, Dict[str, list]], name: str) -> None:
    """Print train/val/test counts for one split."""
    s = splits[name]
    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    print(f"  {name}:")
    print(f"    train: {len(s['train'])}  val: {len(s['val'])}  test: {len(s['test'])}  total: {total}")

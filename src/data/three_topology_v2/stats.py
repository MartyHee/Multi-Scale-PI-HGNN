"""Per-fold normalization statistics for the three-topology v2 inputs.

* Statistics are fit from **that fold's training samples only**.
* Continuous columns are standardised ``(x - mean) / effective_std``; preserved
  columns (boolean support flags, plate type, API link type, is_rigid) are left
  untouched.
* Labels ``y_disp`` / ``y_force`` are standardised for training and can be
  inverse-transformed back to physical units (m, rad, N, N*m).
* Zero/near-zero variance columns use ``effective_std = 1.0``.
* ``BetaAngle`` is declared in degrees (API value), matching the graph contract.

The numeric statistics for folds whose train set is unchanged versus v1 are
verified against the accepted v1 artifacts (see ``verify_against_v1``); only
the ``BetaAngle`` unit label is corrected.  Refit folds (cross-load-field) are
fully recomputed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from . import graph_io, paths
from .splits import COLUMNS, PRESERVED_COLUMNS, UNITS

STATS_SCHEMA = "qt-three-topology-v2-fold-stats-v1"
STATS_INDEX_SCHEMA = "qt-three-topology-v2-fold-stats-index-v1"

EPSILON = 1e-12
CONTINUOUS_MATRICES = ("node_x", "beam_x", "plate_x", "y_disp", "y_force", "structural_link_x")


class _SampleCache:
    """Cache the six matrices per sample once, as float64 numpy arrays."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get(self, row: Dict[str, Any]) -> Dict[str, np.ndarray]:
        sid = row["sample_id"]
        if sid not in self._cache:
            data = graph_io.load_graph(row, root=self.root, dtype=torch.float64, verify_hash=False)
            sl = data["mesh_node", "structural_link", "mesh_node"]
            self._cache[sid] = {
                "node_x": data["mesh_node"].x.numpy(),
                "beam_x": data["beam_element"].x.numpy(),
                "plate_x": data["plate_element"].x.numpy(),
                "y_disp": data["mesh_node"].y_disp.numpy(),
                "y_force": data["beam_element"].y_force.numpy(),
                "structural_link_x": sl.edge_attr.numpy(),
            }
        return self._cache[sid]


def _continuous_stats(column_values: np.ndarray) -> Dict[str, Any]:
    n = column_values.shape[0]
    mean = float(np.mean(column_values)) if n else 0.0
    std = float(np.std(column_values, ddof=1)) if n > 1 else 0.0
    zero_var = bool(std < EPSILON)
    eff_std = 1.0 if zero_var else std
    return {
        "count": int(n),
        "mean": mean,
        "std": std,
        "effective_std": eff_std,
        "min": float(np.min(column_values)) if n else 0.0,
        "max": float(np.max(column_values)) if n else 0.0,
        "zero_variance": zero_var,
    }


def compute_fold_stats(
    fold: Dict[str, Any],
    split_id: str,
    split_kind: str,
    topology_token: Optional[str],
    row_by_id: Dict[str, Dict[str, Any]],
    root: Path,
    cache: Optional[_SampleCache] = None,
    source_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit per-column statistics on one fold's training samples."""
    cache = cache or _SampleCache(root)
    train_ids = list(fold["train_sample_ids"])

    collected = {m: [] for m in CONTINUOUS_MATRICES}
    for sid in train_ids:
        arrays = cache.get(row_by_id[sid])
        for m in CONTINUOUS_MATRICES:
            collected[m].append(arrays[m])

    counts = {
        "train_graphs": len(train_ids),
        "node_rows": int(sum(a.shape[0] for a in collected["node_x"])),
        "beam_rows": int(sum(a.shape[0] for a in collected["beam_x"])),
        "plate_rows": int(sum(a.shape[0] for a in collected["plate_x"])),
        "structural_link_rows": int(sum(a.shape[0] for a in collected["structural_link_x"])),
    }

    columns: Dict[str, List[Dict[str, Any]]] = {}
    for mat in CONTINUOUS_MATRICES:
        big = np.concatenate(collected[mat], axis=0) if collected[mat] else np.zeros((0, len(COLUMNS[mat])))
        preserved = set(PRESERVED_COLUMNS[mat])
        cols: List[Dict[str, Any]] = []
        for i, (name, unit) in enumerate(zip(COLUMNS[mat], UNITS[mat])):
            if i in preserved:
                vals = big[:, i] if big.size else np.zeros(0)
                uniq = Counter(int(round(float(v))) for v in vals)
                cols.append({
                    "index": i, "name": name, "unit": unit, "kind": "preserved",
                    "normalized": False,
                    "reason": "boolean / categorical / raw enum column",
                    "value_counts": {str(k): int(v) for k, v in sorted(uniq.items())},
                })
            else:
                st = _continuous_stats(big[:, i]) if big.size else _continuous_stats(np.zeros(0))
                cols.append({
                    "index": i, "name": name, "unit": unit, "kind": "continuous",
                    "count": st["count"], "mean": st["mean"], "std": st["std"],
                    "effective_std": st["effective_std"], "min": st["min"], "max": st["max"],
                    "zero_variance": st["zero_variance"],
                })
        columns[mat] = cols

    # Summary counts mirrored from v1 for auditability.
    preserved_column_counts: Dict[str, Any] = {}
    support_flags = []
    for a in collected["node_x"]:
        if a.shape[0]:
            support_flags.append(a[:, 9:15])
    if support_flags:
        sf = np.concatenate(support_flags, axis=0)
        supported = sf[np.any(sf != 0, axis=1)]
        preserved_column_counts["node_support_flag_counts"] = {
            COLUMNS["node_x"][9 + j]: int(np.sum(supported[:, j] != 0)) for j in range(6)
        }
    else:
        preserved_column_counts["node_support_flag_counts"] = {
            COLUMNS["node_x"][9 + j]: 0 for j in range(6)
        }

    plate_types = Counter()
    for a in collected["plate_x"]:
        for v in a[:, 5]:
            plate_types[str(int(round(float(v))))] += 1
    preserved_column_counts["plate_type_counts"] = dict(sorted(plate_types.items()))

    link_types = Counter()
    is_rigid = Counter()
    for a in collected["structural_link_x"]:
        for v in a[:, 8]:
            link_types[str(int(round(float(v))))] += 1
        for v in a[:, 9]:
            is_rigid[str(int(round(float(v))))] += 1
    preserved_column_counts["link_type_counts"] = dict(sorted(link_types.items()))
    preserved_column_counts["link_is_rigid_counts"] = dict(sorted(is_rigid.items()))

    stats = {
        "schema": STATS_SCHEMA,
        "split_id": split_id,
        "split_kind": split_kind,
        "topology_token": topology_token,
        "fold_id": fold["fold_id"],
        "source": {
            "split_fold_id": fold["fold_id"],
            **(source_extra or {}),
        },
        "train_sample_ids": train_ids,
        "train_sample_list_sha256": fold.get("train_sample_list_sha256"),
        "train_graph_sha256_aggregate": fold.get("train_graph_sha256_aggregate"),
        "counts": counts,
        "columns": columns,
        "preserved_column_counts": preserved_column_counts,
        "policy": {
            "fit_scope": "train_sample_ids only",
            "accumulation_dtype": "float64",
            "std": "sample standard deviation (ddof=1)",
            "epsilon": EPSILON,
            "zero_variance_rule": "std below epsilon is reported as zero_variance=true and effective_std=1.0",
            "preserved_columns": "boolean support flags, plate type, API link type and is_rigid are not standardized",
            "validation_and_test": "never used for fitting",
        },
    }
    return stats


def stats_lookup(stats: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-matrix ``{'mean','std','preserved_mask'}`` numpy arrays."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for mat in CONTINUOUS_MATRICES:
        cols = stats["columns"][mat]
        dim = len(cols)
        mean = np.zeros(dim, dtype=np.float64)
        std = np.ones(dim, dtype=np.float64)
        preserved = np.zeros(dim, dtype=bool)
        for c in cols:
            i = c["index"]
            if c["kind"] == "preserved":
                preserved[i] = True
            else:
                mean[i] = c["mean"]
                std[i] = c["effective_std"]
        out[mat] = {"mean": mean, "std": std, "preserved_mask": preserved}
    return out


def apply_normalization_(data, stats: Dict[str, Any]) -> None:
    """In-place standardisation of continuous input columns (dtype-preserving)."""
    lk = stats_lookup(stats)
    sl = data["mesh_node", "structural_link", "mesh_node"]

    def _norm(t: torch.Tensor, key: str) -> torch.Tensor:
        m = torch.as_tensor(lk[key]["mean"], dtype=t.dtype)
        s = torch.as_tensor(lk[key]["std"], dtype=t.dtype)
        mask = torch.as_tensor(lk[key]["preserved_mask"], dtype=torch.bool)
        out = (t - m) / s
        return torch.where(mask, t, out)

    data["mesh_node"].x = _norm(data["mesh_node"].x, "node_x")
    data["beam_element"].x = _norm(data["beam_element"].x, "beam_x")
    data["plate_element"].x = _norm(data["plate_element"].x, "plate_x")
    sl.edge_attr = _norm(sl.edge_attr, "structural_link_x")


def normalize_labels_(data, stats: Dict[str, Any]) -> None:
    lk = stats_lookup(stats)

    def _norm(t: torch.Tensor, key: str) -> torch.Tensor:
        m = torch.as_tensor(lk[key]["mean"], dtype=t.dtype)
        s = torch.as_tensor(lk[key]["std"], dtype=t.dtype)
        return (t - m) / s

    data["mesh_node"].y_disp = _norm(data["mesh_node"].y_disp, "y_disp")
    data["beam_element"].y_force = _norm(data["beam_element"].y_force, "y_force")


def inverse_transform_disp(pred: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    lk = stats_lookup(stats)["y_disp"]
    m = torch.as_tensor(lk["mean"], dtype=pred.dtype, device=pred.device)
    s = torch.as_tensor(lk["std"], dtype=pred.dtype, device=pred.device)
    return pred * s + m


def inverse_transform_force(pred: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    lk = stats_lookup(stats)["y_force"]
    m = torch.as_tensor(lk["mean"], dtype=pred.dtype, device=pred.device)
    s = torch.as_tensor(lk["std"], dtype=pred.dtype, device=pred.device)
    return pred * s + m


# ---------------------------------------------------------------------------
# Writing / verification
# ---------------------------------------------------------------------------

def stats_key(split_id: str, fold_id: str) -> str:
    return f"{split_id}/{fold_id}"


def save_stats(stats: Dict[str, Any], out_dir: Optional[Path] = None) -> Dict[str, Any]:
    out_dir = out_dir or paths.STATS_DIR
    d = out_dir / stats["split_id"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stats['fold_id']}.json"
    payload = json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path.relative_to(out_dir.parent.parent)),
        "split_id": stats["split_id"],
        "fold_id": stats["fold_id"],
        "train_graphs": stats["counts"]["train_graphs"],
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "refit": stats["source"].get("refit", True),
    }


_CONTINUOUS_COMPARE = ("mean", "std", "min", "max")


def compare_continuous_to_v1(v2_stats: Dict[str, Any], v1_stats: Dict[str, Any],
                             rtol: float = 1e-9, atol: float = 1e-9) -> Dict[str, Any]:
    """Compare continuous column stats against a v1 stats artifact."""
    problems = []
    for mat in CONTINUOUS_MATRICES:
        v1_cols = {c["name"]: c for c in v1_stats["columns"][mat]}
        for c in v2_stats["columns"][mat]:
            if c["kind"] != "continuous":
                continue
            ref = v1_cols.get(c["name"])
            if ref is None:
                problems.append(f"{mat}.{c['name']}: missing in v1")
                continue
            for field in _CONTINUOUS_COMPARE:
                a, b = c[field], ref[field]
                if not np.isclose(a, b, rtol=rtol, atol=atol):
                    problems.append(f"{mat}.{c['name']}.{field}: {a} vs v1 {b}")
    return {"ok": not problems, "problems": problems[:20], "n_problems": len(problems)}


def load_v1_stats(split_id: str, fold_id: str, root: Path) -> Optional[Dict[str, Any]]:
    p = paths.v1_split_dir(root) / "stats" / split_id / f"{fold_id}.json"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)

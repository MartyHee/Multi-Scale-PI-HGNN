"""Training-core-only statistics for the FE v2 common representation.

The fit scope is explicit: only the supplied training IDs (the common
D_train x F_train core) are read.  Validation/test labels never contribute to
means, scales, zero masks, categories or lower bounds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from . import assemble, graph, schema
from .reader import FeV2Reader

EPS = 1e-12


def _blank(dim: int):
    return {
        "count": 0, "sum": np.zeros(dim, dtype=np.float64),
        "sumsq": np.zeros(dim, dtype=np.float64),
        "nz_count": np.zeros(dim, dtype=np.int64),
        "nz_sumsq": np.zeros(dim, dtype=np.float64),
        "zero_count": np.zeros(dim, dtype=np.int64),
        "min": np.full(dim, np.inf), "max": np.full(dim, -np.inf),
    }


def _add(acc, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"stats array is not 2D: {arr.shape}")
    acc["count"] += arr.shape[0]
    acc["sum"] += arr.sum(axis=0)
    acc["sumsq"] += (arr * arr).sum(axis=0)
    nz = arr != 0
    acc["nz_count"] += nz.sum(axis=0)
    acc["nz_sumsq"] += (arr * arr * nz).sum(axis=0)
    acc["zero_count"] += arr.shape[0] - nz.sum(axis=0)
    acc["min"] = np.minimum(acc["min"], arr.min(axis=0))
    acc["max"] = np.maximum(acc["max"], arr.max(axis=0))


def _col(name, unit, kind, mu=0.0, scale=1.0, count=0, std=0.0, zero_count=0,
         nz_count=0, minv=0.0, maxv=0.0, zero_variance=False):
    return {"name": name, "unit": unit, "kind": kind, "mean": float(mu),
            "std": float(std), "effective_std": float(scale), "scale": float(scale),
            "count": int(count), "zero_count": int(zero_count),
            "nonzero_count": int(nz_count), "zero_variance": bool(zero_variance),
            "min": float(minv), "max": float(maxv)}


def _stats_for(acc, names, units, kinds):
    n = len(names)
    mean = acc["sum"] / max(acc["count"], 1)
    var = np.maximum(acc["sumsq"] / max(acc["count"], 1) - mean * mean, 0.0)
    std = np.sqrt(var)
    rms_nz = np.sqrt(acc["nz_sumsq"] / np.maximum(acc["nz_count"], 1))
    scale = np.ones(n)
    for j, kind in enumerate(kinds):
        if kind == "zscore":
            scale[j] = std[j] if std[j] > EPS else 1.0
        elif kind == "zero_preserve_rms":
            scale[j] = rms_nz[j] if acc["nz_count"][j] > 0 and rms_nz[j] > EPS else 1.0
    cols, arrays = [], {"mean": mean.tolist(), "scale": scale.tolist(), "kind": list(kinds)}
    for j, (name, unit, kind) in enumerate(zip(names, units, kinds)):
        zc = int(acc["zero_count"][j]); nz = int(acc["nz_count"][j])
        c = _col(name, unit, kind,
                 mu=0.0 if kind == "zero_preserve_rms" else float(mean[j]),
                 scale=float(scale[j]), count=int(acc["count"]),
                 std=float(std[j]), zero_count=zc, nz_count=nz,
                 minv=float(acc["min"][j]), maxv=float(acc["max"][j]),
                 zero_variance=bool(acc["count"] == 0 or std[j] <= EPS))
        c["index"] = j
        cols.append(c)
    return cols, arrays


def fit_stats(index, train_ids: Iterable[str], out_path: Path,
              topology_token: str, provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ids = sorted(set(train_ids))
    if not ids:
        raise ValueError("empty training ID set")
    train_set = set(ids)
    # Group by source run so the reader's LRU is useful; never reorder labels
    # for fitting, only traversal order.
    ordered = sorted(ids, key=lambda sid: (index.row(sid)["extraction_path"],
                                           int(index.row(sid)["shard_case_index"]),
                                           sid))
    acc = {
        "node_x": _blank(schema.encoded_dim("mesh_node")),
        "beam_x": _blank(schema.encoded_dim("beam_element")),
        "plate_x": _blank(schema.encoded_dim("plate_element")),
        "y_disp": _blank(6), "y_force": _blank(12),
    }
    reader = FeV2Reader(index, max_runs=2, verify_hash=True)
    n_seen = 0
    for sid in ordered:
        raw = assemble.assemble_raw(reader.read(sid))
        enc = graph.encode_raw(raw)
        _add(acc["node_x"], enc["node_x"]); _add(acc["beam_x"], enc["beam_x"])
        _add(acc["plate_x"], enc["plate_x"])
        _add(acc["y_disp"], raw["node_y"]); _add(acc["y_force"], raw["beam_y"])
        n_seen += 1
    names = {k: [c["name"] for c in schema.encoded_columns({"mesh_node":"mesh_node","beam_element":"beam_element","plate_element":"plate_element"}[k])] for k in ("mesh_node","beam_element","plate_element")}
    units = {k: [c["unit"] for c in schema.encoded_columns({"mesh_node":"mesh_node","beam_element":"beam_element","plate_element":"plate_element"}[k])] for k in ("mesh_node","beam_element","plate_element")}
    kinds = {k: [c["transform"] for c in schema.encoded_columns({"mesh_node":"mesh_node","beam_element":"beam_element","plate_element":"plate_element"}[k])] for k in ("mesh_node","beam_element","plate_element")}
    cols = {}; arrays = {}
    for key, raw_key in (("node_x", "mesh_node"), ("beam_x", "beam_element"), ("plate_x", "plate_element")):
        cols[key], arrays[key] = _stats_for(acc[key], names[raw_key], units[raw_key], kinds[raw_key])
    # Labels: z-score; zero/near-zero columns use scale 1.0 and are marked.
    for key, raw_key, names_lbl in (("y_disp", "mesh_node", schema.load_schema()["labels"]["mesh_node"]),
                                    ("y_force", "beam_element", schema.load_schema()["labels"]["beam_element"])):
        units_lbl = schema.load_schema()["labels"]["units"][raw_key]
        cols[key], arrays[key] = _stats_for(acc[key], names_lbl, units_lbl, ["zscore"] * len(names_lbl))
        constant = [c["zero_variance"] for c in cols[key]]
        arrays[key]["constant_zero"] = constant
    # Compatibility columns for the existing engine's label inverse only.
    cols["structural_link_x"] = [
        _col(f"link_attr_{j}", "raw", "preserve") for j in range(10)
    ]
    arrays["structural_link_x"] = {"mean": [0.0]*10, "scale": [1.0]*10,
                                   "kind": ["preserve"]*10}
    stats = {
        "schema": "qt-three-topology-fe-v2-stats-v1",
        "topology_token": topology_token,
        "fit_scope": "train_core_only",
        "train_sample_ids": ids,
        "train_sample_list_sha256": hashlib.sha256(
            json.dumps(sorted(ids), ensure_ascii=False).encode()).hexdigest(),
        "counts": {
            "samples": n_seen,
            "node_rows": int(acc["node_x"]["count"]),
            "beam_rows": int(acc["beam_x"]["count"]),
            "plate_rows": int(acc["plate_x"]["count"]),
            "node_label_rows": int(acc["y_disp"]["count"]),
            "beam_label_rows": int(acc["y_force"]["count"]),
            "weighting": "one sample weight; each node/beam/plate row counted once",
        },
        "columns": cols,
        "arrays": arrays,
        "source": provenance or {},
        "policy": {
            "validation_test": "never used for fitting",
            "node_load": "zero-preserving RMS scale; zero_count and nonzero_count recorded",
            "zscore": "mean/std over all training rows; zero variance -> scale 1.0",
            "constant_labels": "scale 1.0 and constant_zero=true; no epsilon blow-up",
            "no_test_dependent_thresholds": True,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["_path"] = str(out_path)
    stats["_file_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return stats


def load_stats(path: Path) -> Dict[str, Any]:
    st = json.loads(Path(path).read_text(encoding="utf-8"))
    for cols in st.get("columns", {}).values():
        for i, c in enumerate(cols):
            c.setdefault("index", i)
    st["_path"] = str(path)
    st["_file_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return st


def inverse_stats(stats: Dict[str, Any]):
    return stats["arrays"]["y_disp"], stats["arrays"]["y_force"]

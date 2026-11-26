"""Versioned prediction storage for FE v2 validation exports.

New batches use a compact binary form instead of multi-hundred-MB JSON:

  ``val_predictions_<slice>.npz``
      sample_id      (n,) unicode
      node_ptr       (n+1,) int64   cumulative node counts
      beam_ptr       (n+1,) int64
      pred_disp      (total_nodes, 6) float64
      pred_force     (total_beams, 12) float64
  ``val_predictions_<slice>.meta.json``
      schema/version, dtype, columns, units, stats source, checkpoint source,
      n_samples and the per-file SHA256.

The original JSON export is kept readable (``read_predictions``), and the
conversion is lossless: the writer stores the exact float64 values from the
exported records, and the round-trip test compares every value.  The combined
run-root ``val_predictions.json`` is intentionally not written for new NPZ runs
because it duplicates the three slice files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCHEMA = "qt-fe-v2-predictions-npz-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_predictions_npz(path: Path, records: List[Dict[str, Any]],
                          *, checkpoint_source: str, slice_name: str) -> Dict[str, Any]:
    path = Path(path)
    if not records:
        raise ValueError("refusing to write an empty prediction file")
    n = len(records)
    node_ptr = np.zeros(n + 1, dtype=np.int64)
    beam_ptr = np.zeros(n + 1, dtype=np.int64)
    for i, rec in enumerate(records):
        nd = np.asarray(rec["pred_disp"], dtype=np.float64)
        bf = np.asarray(rec["pred_force"], dtype=np.float64)
        if nd.shape[1] != 6 or bf.shape[1] != 12:
            raise ValueError(f"record {i}: unexpected prediction shape {nd.shape}/{bf.shape}")
        node_ptr[i + 1] = node_ptr[i] + nd.shape[0]
        beam_ptr[i + 1] = beam_ptr[i] + bf.shape[0]
    pred_disp = np.concatenate([np.asarray(r["pred_disp"], dtype=np.float64) for r in records], axis=0)
    pred_force = np.concatenate([np.asarray(r["pred_force"], dtype=np.float64) for r in records], axis=0)
    sample_ids = np.asarray([str(r["sample_id"]) for r in records])
    np.savez_compressed(path, sample_id=sample_ids, node_ptr=node_ptr,
                        beam_ptr=beam_ptr, pred_disp=pred_disp, pred_force=pred_force)
    first = records[0]
    meta = {
        "schema": SCHEMA, "slice": slice_name, "n_samples": n,
        "checkpoint_source": checkpoint_source,
        "dtype": "float64",
        "disp_columns": first["disp_columns"], "disp_units": first["disp_units"],
        "force_columns": first["force_columns"], "force_units": first["force_units"],
        "stats_source": first.get("stats_source"),
        "stats_fold_id": first.get("stats_fold_id"),
        "stats_topology": first.get("stats_topology"),
        "stats_file_sha256": first.get("stats_file_sha256"),
        "npz_sha256": _sha(path),
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def read_predictions(path: Path) -> List[Dict[str, Any]]:
    """Read either a legacy JSON export or a versioned NPZ export."""
    path = Path(path)
    if path.suffix == ".npz":
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        with np.load(path) as z:
            ids = z["sample_id"].tolist()
            np_ptr = z["node_ptr"]; bp_ptr = z["beam_ptr"]
            pd = z["pred_disp"]; pf = z["pred_force"]
        out = []
        for i, sid in enumerate(ids):
            out.append({
                "sample_id": str(sid),
                "node_count": int(np_ptr[i + 1] - np_ptr[i]),
                "beam_count": int(bp_ptr[i + 1] - bp_ptr[i]),
                "disp_columns": meta.get("disp_columns"),
                "disp_units": meta.get("disp_units"),
                "force_columns": meta.get("force_columns"),
                "force_units": meta.get("force_units"),
                "stats_source": meta.get("stats_source"),
                "stats_fold_id": meta.get("stats_fold_id"),
                "stats_topology": meta.get("stats_topology"),
                "stats_file_sha256": meta.get("stats_file_sha256"),
                "pred_disp": pd[np_ptr[i]:np_ptr[i + 1]].tolist(),
                "pred_force": pf[bp_ptr[i]:bp_ptr[i + 1]].tolist(),
            })
        return out
    return json.loads(path.read_text(encoding="utf-8"))

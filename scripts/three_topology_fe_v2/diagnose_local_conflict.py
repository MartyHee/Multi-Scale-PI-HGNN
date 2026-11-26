#!/usr/bin/env python
"""Train-only local-input conflict diagnostic for the local MLP baseline.

Question: for the information the *local* MLP can see, how often do two
training entities have (nearly) the same visible input but different labels?

Visible input (exactly what the MLP baseline receives, after the frozen
normalisation):
  * node head : mesh_node.x (15)
  * beam head : [beam_element.x (18), mean endpoint mesh_node.x (15)] = 33

Two duplicate definitions are reported separately:
  * exact   : bit-identical float32 rows;
  * tolerance: rows equal after rounding to ``tol`` in normalised units
    (default 1e-3 of the training standard deviation).

Only protocol_v1 **A/train** samples are read; no validation, test or blocked
sample is touched, and no model is trained.  The diagnostic is descriptive:
it does not identify a mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402

TOPOLOGIES = ("topology1", "topology2", "topology3")


def _void_rows(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _analyse(X: np.ndarray, Y: np.ndarray, scales: np.ndarray) -> dict:
    keys = _void_rows(X)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n, dim = Y.shape
    g = len(uniq)
    sums = np.zeros((g, dim), dtype=np.float64)
    np.add.at(sums, inv, Y.astype(np.float64))
    mean = sums / counts[:, None]
    mx = np.full((g, dim), -np.inf)
    mn = np.full((g, dim), np.inf)
    np.maximum.at(mx, inv, Y)
    np.minimum.at(mn, inv, Y)
    spread = np.where(counts[:, None] > 1, mx - mn, 0.0)
    conflict = (counts > 1) & (spread > 1e-9).any(axis=1)
    dup_rows = counts[inv] > 1
    dev = np.abs(Y.astype(np.float64) - mean[inv])
    out = {
        "n_rows": int(n), "n_unique": int(g),
        "n_duplicate_rows": int(dup_rows.sum()),
        "duplicate_row_fraction": float(dup_rows.mean()),
        "n_duplicate_groups": int((counts > 1).sum()),
        "n_conflict_groups": int(conflict.sum()),
        "conflict_group_fraction": float(conflict.sum() / max(1, (counts > 1).sum())),
        "conflict_row_fraction": float(((counts[inv] > 1) & conflict[inv]).mean()),
        "mean_abs_dev_norm_all": float(dev.mean()),
        "mean_abs_dev_norm_duplicate_rows": float(dev[dup_rows].mean()) if dup_rows.any() else 0.0,
        "max_group_spread_norm": float(spread.max()) if spread.size else 0.0,
    }
    scale = np.asarray(scales, dtype=np.float64)
    out["mean_abs_dev_phys_by_component"] = (dev.mean(axis=0) * scale).tolist()
    if dup_rows.any():
        out["mean_abs_dev_phys_duplicate_rows_by_component"] = (
            dev[dup_rows].mean(axis=0) * scale).tolist()
    else:
        out["mean_abs_dev_phys_duplicate_rows_by_component"] = [0.0] * dim
    return out


def run_topology(topo: str, max_samples: int | None, tol: float) -> dict:
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    ids = protocol.ids_for(idx, prot, topo, "A", "train")
    if max_samples:
        ids = ids[:max_samples]
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{topo}.json")
    ds = FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                     use_shared=True, topology=topo)
    node_x, node_y, beam_in, beam_y = [], [], [], []
    for sid in ids:
        d = ds[sid]
        nx = d["mesh_node"].x.numpy()
        bx = d["beam_element"].x.numpy()
        ei = d["mesh_node", "belongs_to_beam", "beam_element"].edge_index.numpy()
        nb = bx.shape[0]
        endpoint = np.zeros((nb, nx.shape[1]), dtype=np.float32)
        counts = np.zeros(nb, dtype=np.float32)
        np.add.at(endpoint, ei[1], nx[ei[0]])
        np.add.at(counts, ei[1], 1.0)
        endpoint /= np.maximum(counts, 1.0)[:, None]
        node_x.append(nx)
        node_y.append(d["mesh_node"].y_disp.numpy())
        beam_in.append(np.concatenate([bx, endpoint], axis=1))
        beam_y.append(d["beam_element"].y_force.numpy())
    out = {"topology": topo, "n_samples": len(ids), "tolerance": tol,
           "split": "protocol_v1 A/train (no val/test/blocked)", "heads": {}}
    for head, X, Y, key in (("node", node_x, node_y, "y_disp"),
                            ("beam", beam_in, beam_y, "y_force")):
        Xa = np.concatenate(X, axis=0)
        Ya = np.concatenate(Y, axis=0)
        scale = np.asarray(st["arrays"][key]["scale"], dtype=np.float64)
        exact = _analyse(Xa, Ya, scale)
        q = np.rint(Xa.astype(np.float64) / float(tol)).astype(np.int64)
        near = _analyse(q, Ya, scale)
        out["heads"][head] = {
            "input_dim": int(Xa.shape[1]), "label_dim": int(Ya.shape[1]),
            "exact": exact, "tolerance": near,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT.parent /
                                          "strong_baseline_prep_20261126" / "conflict"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "qt-fe-v2-local-conflict-v1", "tolerance": a.tol,
              "input": "local MLP visible features after frozen normalisation",
              "topologies": {}}
    for topo in TOPOLOGIES:
        r = run_topology(topo, a.max_samples, a.tol)
        report["topologies"][topo] = r
        print(json.dumps({topo: {h: {"exact_dup_frac": v["exact"]["duplicate_row_fraction"],
                                     "tol_dup_frac": v["tolerance"]["duplicate_row_fraction"],
                                     "tol_conflict_group_frac": v["tolerance"]["conflict_group_fraction"],
                                     "tol_conflict_row_frac": v["tolerance"]["conflict_row_fraction"]}
                              for h, v in r["heads"].items()}}, ensure_ascii=False), flush=True)
    (out / "local_conflict.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

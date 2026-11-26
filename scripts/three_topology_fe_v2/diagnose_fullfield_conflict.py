#!/usr/bin/env python
"""Train-only conflict diagnostic: raw full field vs old 8-bin descriptor.

For each topology's A/train samples (1,360), build two sample-level descriptors:
  * raw   : concatenated node(15)/beam(18)/plate(9) features in frozen order;
  * binned: the old 8-bin x-sorted mean/max + global means (714 floats).
Group samples by exact float32 bytes and by 1e-3 normalised tolerance,
separately.  For duplicate groups, compare node/beam labels at the same entity
index and report whether the labels disagree.

No model is trained; no val/test/blocked sample is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402

TOPOS = ("topology1", "topology2", "topology3")
N_BINS = 8


def _coords(node_x):
    return node_x[:, :3]


def _member_coords(node_x, edge_index, n_members):
    out = np.zeros((n_members, 3), dtype=np.float64)
    cnt = np.zeros(n_members, dtype=np.float64)
    np.add.at(out, edge_index[1], node_x[edge_index[0], :3].astype(np.float64))
    np.add.at(cnt, edge_index[1], 1.0)
    return out / np.maximum(cnt, 1.0)[:, None]


def binned_descriptor(node_x, beam_x, plate_x, ei_beam, ei_plate, n_bins=N_BINS):
    node = node_x.astype(np.float64); beam = beam_x.astype(np.float64)
    plate = plate_x.astype(np.float64)
    cn = _coords(node)
    cb = _member_coords(node, ei_beam, beam.shape[0])
    cp = _member_coords(node, ei_plate, plate.shape[0])
    xmin, xmax = float(cn[:, 0].min()), float(cn[:, 0].max())
    span = max(xmax - xmin, 1e-8)

    def stats(coords, feat):
        b = np.clip(((coords[:, 0] - xmin) / span * n_bins).astype(np.int64), 0, n_bins - 1)
        mean = np.zeros((n_bins, feat.shape[1]), dtype=np.float64)
        mx = np.zeros((n_bins, feat.shape[1]), dtype=np.float64)
        for k in range(n_bins):
            m = b == k
            if m.any():
                mean[k] = feat[m].mean(axis=0)
                mx[k] = feat[m].max(axis=0)
        return mean.ravel(), mx.ravel()

    parts = []
    for coords, feat in ((cn, node), (cb, beam), (cp, plate)):
        a, b = stats(coords, feat)
        parts.extend([a, b])
    parts.append(node.mean(axis=0)); parts.append(beam.mean(axis=0)); parts.append(plate.mean(axis=0))
    return np.concatenate(parts)


def _hash_exact(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _hash_tol(a: np.ndarray, tol: float) -> str:
    q = np.rint(np.asarray(a, dtype=np.float64) / tol).astype(np.int32)
    return hashlib.sha256(q.tobytes()).hexdigest()


def _group_stats(hashes, node_labels, beam_labels, ids, idx, tol_label=1e-9):
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    dups = [g for g in groups.values() if len(g) > 1]
    n_conf = 0
    n_conf_rows = 0
    max_spread = 0.0
    mean_abs = 0.0
    entity_conf = 0
    entity_total = 0
    examples = []
    for g in dups:
        nd = np.stack([node_labels[i] for i in g])   # (G, N, 6)
        bf = np.stack([beam_labels[i] for i in g])   # (G, B, 12)
        spread_n = nd.max(axis=0) - nd.min(axis=0)
        spread_b = bf.max(axis=0) - bf.min(axis=0)
        conf_n = spread_n.max() > tol_label
        conf_b = spread_b.max() > tol_label
        if conf_n or conf_b:
            n_conf += 1
            rows_n = (spread_n.max(axis=1) > tol_label).sum()
            rows_b = (spread_b.max(axis=1) > tol_label).sum()
            n_conf_rows += len(g)
            entity_conf += int(rows_n + rows_b)
            mean_abs += float(np.abs(nd - nd.mean(axis=0)).mean() + np.abs(bf - bf.mean(axis=0)).mean()) / 2
            max_spread = max(max_spread, float(spread_n.max()), float(spread_b.max()))
            if len(examples) < 5:
                examples.append({
                    "samples": [ids[i] for i in g],
                    "designs": [idx.by_id[ids[i]]["design_id"] for i in g],
                    "fields": [idx.by_id[ids[i]]["field_id"] for i in g],
                    "node_max_spread": float(spread_n.max()),
                    "beam_max_spread": float(spread_b.max()),
                })
        entity_total += len(g) * (nd.shape[1] + bf.shape[1])
    return {
        "n_samples": len(hashes), "n_groups": len(groups), "n_duplicate_groups": len(dups),
        "n_duplicate_rows": int(sum(len(g) for g in dups)),
        "duplicate_row_fraction": float(sum(len(g) for g in dups) / max(1, len(hashes))),
        "n_conflict_groups": n_conf,
        "conflict_group_fraction": float(n_conf / max(1, len(dups))),
        "conflict_row_fraction": float(n_conf_rows / max(1, len(hashes))),
        "entity_conflict_fraction": float(entity_conf / max(1, entity_total)),
        "mean_abs_label_dev": mean_abs / max(1, n_conf),
        "max_label_spread": max_spread,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT.parent / "strong_baseline_prep_20261126" / "conflict"))
    a = ap.parse_args()
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    rep = {"schema": "qt-fe-v2-fullfield-conflict-v1", "split": "protocol_v1 A/train only",
           "tolerance": a.tol, "topologies": {}}
    for topo in TOPOS:
        ids = protocol.ids_for(idx, prot, topo, "A", "train")
        st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{topo}.json")
        ds = FeV2Dataset(idx, ids, st, normalize=True, use_shared=True, topology=topo)
        raws, bins, nlab, blab = [], [], [], []
        for sid in ids:
            d = ds[sid]
            nx = d["mesh_node"].x.numpy(); bx = d["beam_element"].x.numpy(); px = d["plate_element"].x.numpy()
            ei_b = d["mesh_node", "belongs_to_beam", "beam_element"].edge_index.numpy()
            ei_p = d["mesh_node", "belongs_to_plate", "plate_element"].edge_index.numpy()
            raws.append(np.concatenate([nx.ravel(), bx.ravel(), px.ravel()]))
            bins.append(binned_descriptor(nx, bx, px, ei_b, ei_p))
            nlab.append(d["mesh_node"].y_disp.numpy()); blab.append(d["beam_element"].y_force.numpy())
        out = {"n_samples": len(ids), "raw_field_dim": int(len(raws[0])),
               "binned_dim": int(len(bins[0])), "descriptors": {}}
        for name, arrs in (("raw", raws), ("binned", bins)):
            out["descriptors"][name] = {
                "exact": _group_stats([_hash_exact(x) for x in arrs], nlab, blab, ids, idx),
                "tolerance": _group_stats([_hash_tol(x, a.tol) for x in arrs], nlab, blab, ids, idx),
            }
        rep["topologies"][topo] = out
        print(json.dumps({topo: {k: {"exact_dup": v["exact"]["duplicate_row_fraction"],
                                     "exact_conf": v["exact"]["conflict_row_fraction"],
                                     "tol_dup": v["tolerance"]["duplicate_row_fraction"],
                                     "tol_conf": v["tolerance"]["conflict_row_fraction"],
                                     "tol_entity_conf": v["tolerance"]["entity_conflict_fraction"]}
                              for k, v in out["descriptors"].items()}}, ensure_ascii=False), flush=True)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fullfield_conflict.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

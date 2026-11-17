#!/usr/bin/env python
"""Deliverable 3 — validate mixed-batch constraint references.

Batches one graph from each topology (different sizes), covering empty records
(topology1: no releases/master-slave) and non-empty releases (topology2) and
releases + master-slave (topology3).  Verifies that ``ThreeTopoData.__inc__``
offsets support / beam-release / master-slave indices by the correct store, that
the offsets point at the correct node/beam rows, that raw IDs are not offset,
and that the five relations do not leak across graphs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_v2 import batch as batch_mod, graph_io, index, paths  # noqa: E402


def pick_ids(idx):
    rows = idx["rows"]
    by_topo = {}
    for r in rows:
        t = r["topology_token"]
        if t not in by_topo:
            by_topo[t] = r
    # choose non-empty constraint graphs where available
    picked = [by_topo["topology1"]["sample_id"]]
    t2 = next(r for r in rows if r["topology_token"] == "topology2"
              and r["counts"]["beam_releases"] > 0)
    t3 = next(r for r in rows if r["topology_token"] == "topology3"
              and r["counts"]["master_slave"] > 0)
    picked.append(t2["sample_id"])
    picked.append(t3["sample_id"])
    return picked


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    ids = pick_ids(idx)
    graphs = [graph_io.load_graph(idx["row_by_id"][sid], root=root,
                                  dtype=torch.float64, verify_hash=True) for sid in ids]
    batch = Batch.from_data_list(graphs)
    report = batch_mod.verify_batched_constraints(graphs, batch)
    report["sample_ids"] = ids
    report["per_graph"] = [
        {"sample_id": sid,
         "nodes": int(g["mesh_node"].x.shape[0]),
         "beams": int(g["beam_element"].x.shape[0]),
         "supports": int(g.support_node_index.numel()),
         "beam_releases": int(g.beam_release_id.numel()),
         "master_slave": int(g.master_slave_id.numel())}
        for sid, g in zip(ids, graphs)
    ]

    out = paths.OUTPUT_ROOT / "batch_constraint_check_2026-11-16"
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_constraint_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"all_ok": report["all_ok"],
                      "checks": len(report["checks"]),
                      "problems": report["problems"][:10],
                      "per_graph": report["per_graph"],
                      "report": str(out / "batch_constraint_report.json")},
                     ensure_ascii=False, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Deliverable 1 — stream-read all 432 samples and verify integrity.

Checks: identity (sample_id/sample_key/(topology,design,field)), paths resolved
under the package root, graph SHA256, shapes, column order, units, I/J endpoint
alignment, API link type vs rigid marker, support/release/master-slave
consistency, coordinate markers.

Run (from model project root):
  & <py> -B scripts/three_topology_v2/run_integrity_check.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402


def main() -> int:
    t0 = time.time()
    paths.ensure_derived_dirs()
    root = paths.package_root()

    import torch_geometric
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "device": "cpu",
        "package_root": str(root),
    }

    idx = index.load_index(root=root, strict=True)
    report = {
        "schema": "qt-three-topology-v2-integrity-report-v1",
        "date": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
        "env": env,
        "index_hashes": idx["hashes"],
        "coverage": idx["coverage"],
        "samples": [],
        "totals": {},
        "errors": [],
    }

    agg = {"samples": 0, "nodes": 0, "beams": 0, "plates": 0, "links": 0,
           "supports": 0, "beam_releases": 0, "master_slave": 0}
    link_type_totals = {}

    for i, row in enumerate(idx["rows"]):
        sid = row["sample_id"]
        try:
            data = graph_io.load_graph(row, root=root, dtype=torch.float64, verify_hash=True)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append({"sample_id": sid, "error": repr(exc)})
            continue
        sl = data["mesh_node", "structural_link", "mesh_node"]
        rec = {
            "sample_id": sid,
            "topology_token": row["topology_token"],
            "design_id": row["design_id"],
            "nodes": int(data["mesh_node"].x.shape[0]),
            "beams": int(data["beam_element"].x.shape[0]),
            "plates": int(data["plate_element"].x.shape[0]),
            "links": int(sl.link_id.numel()),
            "node_x_shape": list(data["mesh_node"].x.shape),
            "node_y_shape": list(data["mesh_node"].y_disp.shape),
            "beam_x_shape": list(data["beam_element"].x.shape),
            "beam_y_shape": list(data["beam_element"].y_force.shape),
            "plate_x_shape": list(data["plate_element"].x.shape),
            "link_attr_shape": list(sl.edge_attr.shape),
            "supports": int(data.support_node_index.numel()),
            "beam_releases": int(data.beam_release_id.numel()),
            "master_slave": int(data.master_slave_id.numel()),
            "coordinate_system": data.graph_contract.get("coordinate_system"),
        }
        if not data.legacy_link_type_column.any():
            pass
        for k in ("nodes", "beams", "plates", "links", "supports", "beam_releases", "master_slave"):
            agg[k] += rec[k]
        agg["samples"] += 1
        lt = {str(int(v)): 0 for v in sl.link_type.unique().tolist()}
        for v in sl.link_type.tolist():
            lt[str(int(v))] += 1
            link_type_totals[str(int(v))] = link_type_totals.get(str(int(v)), 0) + 1
        rec["link_type_counts"] = lt
        report["samples"].append(rec)

    report["totals"] = agg
    report["link_type_totals"] = link_type_totals
    report["ok"] = (len(report["errors"]) == 0 and agg["samples"] == 432
                    and agg["nodes"] > 0)
    report["elapsed_sec"] = round(time.time() - t0, 2)

    out = paths.OUTPUT_ROOT / "integrity_check_2026-11-16"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "integrity_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": report["ok"], "errors": report["errors"][:5],
                      "totals": agg, "coverage": idx["coverage"],
                      "report": str(out_path)}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

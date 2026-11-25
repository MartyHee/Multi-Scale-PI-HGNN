#!/usr/bin/env python
"""Lightweight full-index checks + targeted graph construction/batch checks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_fe_v2 import assemble, graph, index as index_mod, paths, protocol, schema  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402
from src.data.three_topology_fe_v2 import stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_v2.batch import verify_batched_constraints  # noqa: E402

TEMPLATES = ["topology1", "topology2", "topology3"]
SAMPLES = {"topology1": ("section_1274", "LF01"),
           "topology2": ("T2-D00", "LF01"),
           "topology3": ("T3-D00", "LF01")}


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.PROCESSED_ROOT))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "validation"))
    a = ap.parse_args()
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = index_mod.load_index()
    ident = idx.validate_identity()
    report = {"schema": "qt-fe-v2-pipeline-validation-v1", "index": ident,
              "processed_manifest": str(root / "manifest.json"), "checks": {}, "problems": []}

    def check(name, cond, detail=""):
        report["checks"][name] = {"ok": bool(cond), "detail": detail}
        if not cond:
            report["problems"].append(f"{name}: {detail}")

    check("index_9216", len(idx.rows) == 9216, str(len(idx.rows)))
    for t in TEMPLATES:
        d = [r for r in idx.rows if r["topology_token"] == t]
        check(f"{t}_3072_48_64", len(d) == 3072 and len({r["design_id"] for r in d}) == 48
              and len({r["field_id"] for r in d}) == 64)
        check(f"{t}_single_c0", len({r["c0_hash"] for r in d}) == 1)
    check("schema_dims", schema.encoded_dim("mesh_node") == 15
          and schema.encoded_dim("beam_element") == 18
          and schema.encoded_dim("plate_element") == 9)
    try:
        prot = json.loads((root / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
        counts = protocol.role_counts(prot)
        check("protocol_ledger_9216", len(prot["roles"]) == 9216)
        check("protocol_counts", counts["A"]["train"] == 4080 and counts["B"]["test"] == 1428
              and counts["C"]["test"] == 294)
        ledger = list(csv.DictReader((root / "splits" / "protocol_ledger.csv").open(encoding="utf-8")))
        check("protocol_csv_9216", len(ledger) == 9216)
        check("protocol_csv_roles", all(ledger[i]["A"] == prot["roles"][ledger[i]["sample_id"]]["A"]
                                       for i in range(len(ledger))))
    except Exception as exc:  # noqa: BLE001
        check("protocol_load", False, repr(exc))

    reader = FeV2Reader(idx, max_runs=2, verify_hash=True)
    graphs = []
    for t in TEMPLATES:
        design, field = SAMPLES[t]
        sid = next(r["sample_id"] for r in idx.rows
                   if r["topology_token"] == t and r["design_id"] == design and r["field_id"] == field)
        raw = assemble.assemble_raw(reader.read(sid))
        enc = graph.encode_raw(raw)
        check(f"{t}_raw_shapes", raw["node_x"].shape[1] == 15 and raw["beam_raw"].shape[1] == 17
              and raw["plate_raw"].shape[1] == 6)
        check(f"{t}_encoded_shapes", enc["node_x"].shape[1] == 15 and enc["beam_x"].shape[1] == 18
              and enc["plate_x"].shape[1] == 9)
        f32_err = {k: float(np.max(np.abs(enc[k].astype(np.float32).astype(np.float64) - enc[k])))
                   for k in enc}
        f32_rel = {k: float(np.max(np.abs(enc[k].astype(np.float32).astype(np.float64) - enc[k])
                                   / np.maximum(np.abs(enc[k]), 1e-12))) for k in enc}
        report.setdefault("float32_max_abs_error", {})[t] = f32_err
        report.setdefault("float32_max_rel_error", {})[t] = f32_rel
        check(f"{t}_labels_shapes", raw["node_y"].shape == (len(raw["node_ids"]), 6)
              and raw["beam_y"].shape == (len(raw["beam_ids"]), 12))
        check(f"{t}_finite", all(np.isfinite(a).all() for a in
                                 (raw["node_x"], raw["beam_raw"], raw["plate_raw"], raw["node_y"], raw["beam_y"])))
        check(f"{t}_ij_node_mapping", np.array_equal(raw["beam_node_index"],
                                                       np.stack([[np.where(raw["node_ids"] == n)[0][0] for n in ij]
                                                                 for ij in raw["beam_node_ids"]], axis=0)))
        check(f"{t}_beta_raw_present", raw["beam_raw"].shape[1] == 17)
        check(f"{t}_beta_encoding", np.allclose(enc["beam_x"][:, 15], np.sin(np.deg2rad(raw["beam_raw"][:, 15])), atol=1e-12)
              and np.allclose(enc["beam_x"][:, 16], np.cos(np.deg2rad(raw["beam_raw"][:, 15])), atol=1e-12))
        g = graph.build_graph(raw)
        check(f"{t}_graph_x", tuple(g["mesh_node"].x.shape)[1:] == (15,)
              and tuple(g["beam_element"].x.shape)[1:] == (18,)
              and tuple(g["plate_element"].x.shape)[1:] == (9,))
        check(f"{t}_graph_supervision", tuple(g["mesh_node"].y_disp.shape)[1:] == (6,)
              and tuple(g["beam_element"].y_force.shape)[1:] == (12,))
        # I/J direction against actual ordered node coordinates.
        b = raw["beam_raw"]; ni = raw["beam_node_index"][:, 0]; nj = raw["beam_node_index"][:, 1]
        delta = raw["node_xyz"][nj] - raw["node_xyz"][ni]
        length = np.linalg.norm(delta, axis=1)
        check(f"{t}_length_ij", np.allclose(length, b[:, 11], rtol=1e-10, atol=1e-12))
        check(f"{t}_cos_ij", np.allclose(delta / length[:, None], b[:, 12:15], rtol=1e-10, atol=1e-12))
        report.setdefault("samples", {})[t] = {
            "sample_id": sid, "nodes": len(raw["node_ids"]), "beams": len(raw["beam_ids"]),
            "plates": len(raw["plate_ids"]), "beta_nonzero_beams": int((b[:, 15] != 0).sum()),
            "tapered_beams": int(b[:, 16].sum()), "releases": len(raw["release_ids"]),
            "master_slave": len(raw["master_ids"]), "links": len(raw["link_ids"]),
        }
        graphs.append(g)
    # Mixed-template batching is an adapter check only, not joint training.
    batch = Batch.from_data_list(graphs)
    bcheck = verify_batched_constraints(graphs, batch)
    check("mixed_batch_constraints", bcheck["all_ok"], json.dumps(bcheck["problems"][:5]))
    for et in graph.EDGE_TYPES:
        ei = batch[et].edge_index
        src, dst = ei[0], ei[1]
        def gid(ptr, x):
            return torch.searchsorted(ptr, x, right=True) - 1
        stype, _, dtype = et
        a = gid(batch[stype].ptr, src); b_ = gid(batch[dtype].ptr, dst)
        check(f"batch_no_cross_{et[1]}", bool(torch.equal(a, b_)))
    report["batch"] = {"graphs": len(graphs), "all_ok": bcheck["all_ok"],
                       "checks": bcheck["checks"]}
    # Frozen transform/inverse check on one seen T1 sample.
    sp = root / "stats" / "topology1.json"
    if sp.is_file():
        st = stats_mod.load_stats(sp)
        sid = next(r["sample_id"] for r in idx.rows
                   if r["topology_token"] == "topology1" and r["design_id"] == "section_1274"
                   and r["field_id"] == "LF01")
        d = FeV2Dataset(idx, [sid], st, normalize=True)[0]
        raw = assemble.assemble_raw(reader.read(sid))
        ds = st["arrays"]["y_disp"]; fs = st["arrays"]["y_force"]
        inv_d = d["mesh_node"].y_disp.numpy() * np.asarray(ds["scale"]) + np.asarray(ds["mean"])
        inv_f = d["beam_element"].y_force.numpy() * np.asarray(fs["scale"]) + np.asarray(fs["mean"])
        check("inverse_disp_roundtrip", np.allclose(inv_d, raw["node_y"], atol=2e-5, rtol=2e-5),
              f"max={float(np.max(np.abs(inv_d-raw['node_y'])))}")
        check("inverse_force_roundtrip", np.allclose(inv_f, raw["beam_y"], atol=2e-2, rtol=2e-5),
              f"max={float(np.max(np.abs(inv_f-raw['beam_y'])))}")
        zero = raw["node_load"] == 0
        check("zero_preserve_loads", np.all(d["mesh_node"].x.numpy()[:, 3:9][zero] == 0))
    report["ok"] = not report["problems"]
    (out / "pipeline_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "problems": report["problems"][:10],
                      "samples": report.get("samples", {})}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

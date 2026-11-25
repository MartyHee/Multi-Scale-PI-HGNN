#!/usr/bin/env python
"""Build lightweight shared geometry, field/design tables, protocol and index.

This intentionally does NOT copy response arrays or build per-sample graphs.
It writes only shared static structure, the 64-field load table per template,
design/field registries, and the candidate protocol ledger.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from src.data.three_topology_fe_v2 import assemble, index as index_mod, paths, protocol  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402

TEMPLATES = ["topology1", "topology2", "topology3"]
REPRESENTATIVE = {"topology1": "section_1274", "topology2": "T2-D00", "topology3": "T3-D00"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choose(index, topology, design):
    return next(r["sample_id"] for r in index.rows
                if r["topology_token"] == topology and r["design_id"] == design and r["field_id"] == "LF01")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.PACKAGE_ROOT))
    ap.add_argument("--out", default=str(paths.PROCESSED_ROOT))
    a = ap.parse_args()
    index = index_mod.load_index(Path(a.root))
    ident = index.validate_identity()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"REFUSE existing processed dir: {out}")
    (out / "index").mkdir(parents=True)
    (out / "shared").mkdir()
    (out / "fields").mkdir()
    (out / "designs").mkdir()
    (out / "splits").mkdir()
    shutil.copy2(index.root / "samples.jsonl", out / "index" / "samples.jsonl")
    reader = FeV2Reader(index, max_runs=2, verify_hash=True)
    manifest = {"schema": "qt-fe-v2-processed-v1", "date": "2026-11-25",
                "source_root": str(index.root), "source_samples_sha256": index.samples_sha256,
                "source_manifest_sha256": ident["manifest_sha256"],
                "templates": {}, "files": {}}

    for t in TEMPLATES:
        sid = choose(index, t, REPRESENTATIVE[t])
        raw = assemble.assemble_raw(reader.read(sid))
        shared = {
            "node_ids": raw["node_ids"], "node_xyz": raw["node_xyz"],
            "beam_ids": raw["beam_ids"], "beam_node_ids": raw["beam_node_ids"],
            "plate_ids": raw["plate_ids"], "plate_node_ids": raw["plate_node_ids"],
            "support_ids": raw["support_ids"], "support_node_index": raw["support_node_index"],
            "support_flags": raw["support_flags"],
            "link_ids": raw["link_ids"], "link_edge_index": raw["link_edge_index"],
            "link_attr": raw["link_attr"],
            "release_ids": raw["release_ids"], "release_beam_index": raw["release_beam_index"],
            "release_node_index": raw["release_node_index"], "release_info": raw["release_info"],
            "master_ids": raw["master_ids"], "master_node_index": raw["master_node_index"],
            "master_info": raw["master_info"],
        }
        np.savez_compressed(out / "shared" / f"{t}_geometry.npz", **shared)
        node_ids = raw["node_ids"]
        node_pos = {int(n): i for i, n in enumerate(node_ids)}
        fields = [f for f in index.fields if f["topology_token"] == t]
        if len(fields) != 64:
            raise SystemExit(f"{t}: expected 64 fields, got {len(fields)}")
        fids = [f["field_id"] for f in fields]
        mat = np.zeros((len(fields), len(node_ids), 6), dtype=np.float64)
        for fi, f in enumerate(fields):
            for r in f["rows"]:
                n = int(r["node_id"])
                if n not in node_pos:
                    raise SystemExit(f"{t}/{f['field_id']}: node {n} missing from shared geometry")
                mat[fi, node_pos[n]] = np.asarray(r["load_info"], dtype=np.float64)
        np.savez_compressed(out / "fields" / f"{t}_loads.npz",
                            field_ids=np.asarray(fids), loads=mat)
        meta_rows = [{"field_id": f["field_id"], "family": f["family"],
                      "components": ";".join(f["components"]),
                      "pattern_hash": f["pattern_hash"],
                      "coordinate_system": f["coordinate_system"],
                      "development_seen": bool(f.get("development_seen"))} for f in fields]
        write_csv(out / "fields" / f"{t}.csv", meta_rows)
        dfs = [d for d in index.designs if d["topology_token"] == t]
        write_csv(out / "designs" / f"{t}.csv",
                  [{"design_id": d["design_id"], "kind": d.get("kind"),
                    "group": d.get("group"), "design_identity_hash": d["design_identity_hash"],
                    "development_seen": bool(d.get("development_seen")),
                    "g1_solved": bool(d.get("g1_solved"))} for d in dfs])
        manifest["templates"][t] = {
            "representative_sample_id": sid, "nodes": len(raw["node_ids"]),
            "beams": len(raw["beam_ids"]), "plates": len(raw["plate_ids"]),
            "fields": len(fields), "designs": len(dfs),
            "shared_file": f"shared/{t}_geometry.npz",
            "load_file": f"fields/{t}_loads.npz",
        }
        # Static structure checks across a second design (same geometry/C0).
        other = next(r["sample_id"] for r in index.rows
                     if r["topology_token"] == t and r["design_id"] != REPRESENTATIVE[t]
                     and r["field_id"] == "LF01")
        r2 = assemble.assemble_raw(reader.read(other))
        manifest["templates"][t]["second_design_geometry_same"] = bool(
            np.array_equal(raw["node_ids"], r2["node_ids"])
            and np.array_equal(raw["beam_node_ids"], r2["beam_node_ids"])
            and np.array_equal(raw["plate_node_ids"], r2["plate_node_ids"])
            and np.array_equal(raw["support_node_index"], r2["support_node_index"])
        )
        if not manifest["templates"][t]["second_design_geometry_same"]:
            raise SystemExit(f"{t}: fixed geometry differs between designs")

    plan = index.holdout
    prot = protocol.build_protocol(plan, index)
    counts = protocol.role_counts(prot)
    protocol.write_ledger(prot, out / "splits" / "protocol_ledger.csv")
    protocol.write_protocol_json(prot, out / "splits" / "protocol_candidate.json")
    (out / "splits" / "protocol_counts.json").write_text(
        json.dumps({"schema": "qt-fe-v2-protocol-counts-v1", "not_approved": True,
                    "counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")

    # manifest of every derived file (excluding itself)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest["files"][p.relative_to(out).as_posix()] = {
                "bytes": p.stat().st_size, "sha256": sha(p)}
    manifest["identity"] = ident
    manifest["not_approved"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "templates": manifest["templates"],
                      "protocol": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

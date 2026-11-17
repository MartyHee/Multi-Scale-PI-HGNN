#!/usr/bin/env python
"""Connection-invariance check (research design revision).

Covers, per sample, the full connection identity/object/attribute/coordinate
correspondence — not just link-type counts:

  * link ID: ``link.ids`` == ``edges.structural_link.ids`` (object identity)
  * endpoints: edges edge_index == raw ``start_node_id``/``end_node_id`` mapped
    through ``node.ids``; endpoints point at the same object per link ID
  * type↔object: API ``link.link_type`` matches the object's corrected
    ``edge_attr[:,8]``; ``canonical_kind=='rigid'`` iff API type 2; the legacy
    topology-1 raw column (1) vs API type (2) is recorded
  * attributes: ``edge_attr[:,0:8]`` == [boundary_info(6), beta, dis_ratio];
    ``is_rigid`` == canonical; kx/gap/friction present; no silent rewrite
  * coordinate system: contract ``coordinate_system`` is Global(T1)/Nodal(T2/3)
    and is constant per topology; ``node_local_axes`` recorded, not interpreted

Hash rule (connection signature): canonical JSON of the per-link tuples
``(link_id, start_raw, end_raw, api_type, boundary_info[6], beta, dis_ratio,
kx, gap, friction, canonical_kind)`` sorted by link_id, SHA256.  A separate
"counts-only" hash (types histogram) is reported to show what a count-only
check would miss.

Writes only derived CSVs/JSON under the model project; source stays read-only.
Response labels (node.y / beam.y) are never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402

OUT_REL = Path("outputs/three_topology_v2/connection_invariance_2026-11-17")


def conn_signature(link: dict, id_to_idx: dict) -> tuple:
    ids = [int(v) for v in link["ids"]]
    starts = [int(v) for v in link["start_node_id"]]
    ends = [int(v) for v in link["end_node_id"]]
    types = [int(v) for v in link["link_type"]]
    boundary = link["boundary_info"]
    beta = link["beta_angle"]
    dis = link["dis_ratio"]
    kx = link["kx"]
    gap = link["gap"]
    fric = link["friction"]
    canon = link["canonical_kind"]
    tuples = []
    for k, lid in enumerate(ids):
        tuples.append((lid, starts[k], ends[k], types[k],
                       tuple(float(x) for x in boundary[k]),
                       float(beta[k]), float(dis[k]), float(kx[k]),
                       float(gap[k]), float(fric[k]), str(canon[k])))
    tuples.sort(key=lambda t: t[0])
    return tuple(tuples)


def sig_hash(sig) -> str:
    payload = json.dumps([list(t) for t in sig], ensure_ascii=False,
                         separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def counts_hash(link: dict) -> str:
    import collections
    c = collections.Counter(int(v) for v in link["link_type"])
    payload = json.dumps(dict(sorted(c.items())), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)

    out_dir = MODEL_ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    per_topology_sigs = defaultdict(set)
    covered_fields = [
        "link.ids", "edges.structural_link.ids", "edges.structural_link.edge_index",
        "link.start_node_id", "link.end_node_id", "node.ids",
        "link.link_type", "structural_link.edge_attr[:,0:8]",
        "edge_attr[:,8] (API type)", "edge_attr[:,9] (is_rigid)",
        "link.boundary_info", "link.beta_angle", "link.dis_ratio",
        "link.kx", "link.gap", "link.friction", "link.canonical_kind",
        "contract.coordinate_system", "contract.node_local_axes",
    ]

    for i, row in enumerate(idx["rows"]):
        sid = row["sample_id"]
        try:
            data = graph_io.load_graph(row, root=root, dtype=torch.float64, verify_hash=True)
            gpath = paths.resolve_under_root(row["graph_path"], root)
            graph = json.loads(gpath.read_bytes().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_id": sid, "stage": "load", "error": repr(exc)})
            continue
        link = graph["link"]
        sl = data["mesh_node", "structural_link", "mesh_node"]
        node_ids = [int(v) for v in graph["node"]["ids"]]
        id_to_idx = {v: k for k, v in enumerate(node_ids)}

        problems = []
        # object identity
        if not np.array_equal(np.asarray(link["ids"]), sl.link_id.numpy()):
            problems.append("link.ids != sl.link_id")
        # endpoints via raw ids
        exp_start = np.array([id_to_idx.get(int(v), -1) for v in link["start_node_id"]])
        exp_end = np.array([id_to_idx.get(int(v), -1) for v in link["end_node_id"]])
        if np.any(exp_start < 0) or np.any(exp_end < 0):
            problems.append("raw link endpoint node id missing from node.ids")
        else:
            if not np.array_equal(exp_start, sl.edge_index[0].numpy()):
                problems.append("edge_index row0 != mapped start_node_id")
            if not np.array_equal(exp_end, sl.edge_index[1].numpy()):
                problems.append("edge_index row1 != mapped end_node_id")
        # type object correspondence
        api_type = np.asarray([int(v) for v in link["link_type"]])
        is_rigid = np.asarray([1 if v == "rigid" else 0 for v in link["canonical_kind"]])
        if not np.array_equal(sl.edge_attr[:, 8].numpy().astype(np.int64), api_type):
            problems.append("corrected edge_attr[:,8] != link.link_type")
        if not np.array_equal(sl.edge_attr[:, 9].numpy().astype(np.int64), is_rigid):
            problems.append("edge_attr[:,9] != canonical rigid marker")
        if np.any((api_type == 2) != (is_rigid == 1)):
            problems.append("rigid marker inconsistent with API type 2")
        # attributes
        boundary = np.asarray(link["boundary_info"], dtype=np.float64)
        prefix = np.concatenate([boundary,
                                 np.asarray(link["beta_angle"], dtype=np.float64)[:, None],
                                 np.asarray(link["dis_ratio"], dtype=np.float64)[:, None]], axis=1)
        if not np.allclose(sl.edge_attr[:, :8].numpy(), prefix, rtol=0, atol=0):
            problems.append("edge_attr[:,0:8] != [boundary_info,beta,dis_ratio]")
        # legacy topology-1 type column
        legacy = sl.legacy_type_column_mismatch.numpy().astype(bool)
        if row["topology_token"] == "topology1":
            if not legacy.all():
                problems.append("T1 expected legacy type-column mismatch on all links")
        else:
            if legacy.any():
                problems.append("non-T1 unexpected legacy type-column mismatch")
        # coordinate system
        cs = data.graph_contract.get("coordinate_system")
        expected_cs = {"topology1": "Global", "topology2": "Nodal", "topology3": "Nodal"}[row["topology_token"]]
        if cs != expected_cs:
            problems.append(f"coordinate_system {cs} != {expected_cs}")

        sig = conn_signature(link, id_to_idx)
        sig_h = sig_hash(sig)
        cnt_h = counts_hash(link)
        per_topology_sigs[row["topology_token"]].add(sig_h)
        if problems:
            errors.append({"sample_id": sid, "problem": problems})

        rows.append({
            "sample_id": sid, "topology": row["topology_token"], "design": row["design_id"],
            "link_count": len(link["ids"]),
            "link_type_1": int(np.sum(api_type == 1)),
            "link_type_2": int(np.sum(api_type == 2)),
            "link_type_4": int(np.sum(api_type == 4)),
            "coordinate_system": cs,
            "node_local_axes_empty": len(graph["contract"].get("node_local_axes") or []) == 0,
            "connection_signature_sha256": sig_h,
            "counts_only_sha256": cnt_h,
            "legacy_type_mismatch_all": bool(legacy.all()),
            "ok": not problems,
        })
        if (i + 1) % 100 == 0:
            print("checked", i + 1, flush=True)

    # CSV + JSON report
    with (out_dir / "connection_invariance_raw.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # design-invariance: per topology, how many distinct connection signatures
    design_invariance = {
        topo: {"distinct_connection_signatures": len(sigs)}
        for topo, sigs in per_topology_sigs.items()
    }
    # aggregate hash over all samples (sorted by sample_id)
    agg_lines = [f"{r['sample_id']}:{r['connection_signature_sha256']}" for r in sorted(rows, key=lambda x: x["sample_id"])]
    aggregate_hash = hashlib.sha256("\n".join(agg_lines).encode()).hexdigest()
    counts_agg = hashlib.sha256("\n".join(
        f"{r['sample_id']}:{r['counts_only_sha256']}" for r in sorted(rows, key=lambda x: x["sample_id"])
    ).encode()).hexdigest()

    report = {
        "schema": "qt-three-topology-connection-invariance-v1",
        "date": "2026-11-17",
        "samples_checked": len(rows),
        "errors": errors,
        "ok": not errors,
        "covered_fields": covered_fields,
        "not_covered": [
            "semantic meaning of release/master-slave DOF flags",
            "semantic meaning of API link type 1/4",
            "DOF order software semantics of support.boundary_info",
            "whether the solver actually used kx/gap/friction values (we verify storage, not solver use)",
        ],
        "hash_rule": "SHA256 over canonical JSON of per-link (id,start_raw,end_raw,api_type,boundary_info6,beta,dis_ratio,kx,gap,friction,canonical_kind) sorted by id",
        "aggregate_connection_signature_sha256": aggregate_hash,
        "aggregate_counts_only_sha256": counts_agg,
        "per_topology": {
            topo: {
                "samples": sum(1 for r in rows if r["topology"] == topo),
                "link_type_totals": {
                    "1": sum(r["link_type_1"] for r in rows if r["topology"] == topo),
                    "2": sum(r["link_type_2"] for r in rows if r["topology"] == topo),
                    "4": sum(r["link_type_4"] for r in rows if r["topology"] == topo),
                },
                "distinct_connection_signatures": design_invariance[topo]["distinct_connection_signatures"],
                "coordinate_systems": sorted({r["coordinate_system"] for r in rows if r["topology"] == topo}),
            } for topo in ("topology1", "topology2", "topology3")
        },
    }
    (out_dir / "connection_invariance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": report["ok"], "samples": len(rows), "errors": errors[:5],
                      "per_topology": report["per_topology"],
                      "aggregate_connection_signature_sha256": aggregate_hash,
                      "aggregate_counts_only_sha256": counts_agg,
                      "report": str(out_dir / "connection_invariance_report.json")},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

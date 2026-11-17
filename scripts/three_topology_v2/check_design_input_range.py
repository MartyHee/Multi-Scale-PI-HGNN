#!/usr/bin/env python
"""Design input-variation range check (research design revision).

For every topology and all 24 designs, reads ONLY structural inputs
(node coordinates, beam sections, plate features); never reads response labels.
Distinguishes:
  * "design different"      — design_id changes
  * "section change"        — geometry/plate identical, only beam section columns differ
  * "interpolation/extrapolation" — whether test design descriptors over every
    cross-design fold lie inside the training descriptor range per feature

Descriptor per (topology, design): mean log10 of beam section columns 0..7 +
mean plate thickness (mirrors the cross-design split features).  Outputs CSVs +
JSON; the cross-design splits are read from the published v2 namespace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from src.data.three_topology_v2 import index, paths  # noqa: E402

OUT_REL = Path("outputs/three_topology_v2/design_input_range_2026-11-17")
SECTION_COLS = [0, 1, 2, 3, 4, 5, 6, 7]  # Area/I at I and J


def _hash_array(a) -> str:
    arr = np.asarray(a, dtype=np.float64)
    return hashlib.sha256(np.round(arr, 12).tobytes()).hexdigest()


def _read_inputs(graph_path: Path):
    g = json.loads(graph_path.read_bytes().decode("utf-8"))
    node_x = np.asarray(g["node"]["x"], dtype=np.float64)
    beam_x = np.asarray(g["beam"]["x"], dtype=np.float64)
    plate_x = np.asarray(g["plate"]["x"], dtype=np.float64)
    beam_ids = [int(v) for v in g["beam"]["ids"]]
    return node_x, beam_x, plate_x, beam_ids


def _descriptor(beam_x, plate_x):
    sec = beam_x[:, SECTION_COLS]
    sec = np.where(sec > 0, sec, np.nan)
    vals = np.nanmean(np.log10(sec), axis=0)  # 8
    plate_thick = float(np.mean(plate_x[:, 0])) if plate_x.size else 0.0
    return np.concatenate([vals, [plate_thick]])  # 9


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    out_dir = MODEL_ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    # source design metadata
    src_meta = {}
    for manual, name in (("topology1", "topology1_primary_samples_v2.jsonl"),
                         ("topology2", "topology2_samples_v2.jsonl"),
                         ("topology3", "topology3_samples.jsonl")):
        rows = [json.loads(l) for l in paths.source_manifest_path(name, root).open(encoding="utf-8")]
        by_design = {}
        for r in rows:
            d = r["design_id"]
            by_design[d] = {
                "design_kind": r.get("design_kind"),
                "design_description": r.get("design_description"),
                "changed_element_count": r.get("changed_element_count"),
                "design_changes": r.get("design_changes"),
            }
        src_meta[manual] = by_design

    # one graph per (topology, design) suffices for structural descriptors (field-invariant)
    desc_rows = []
    per_design = {}  # (topo, design) -> dict of computed facts
    # group sample ids by (topology, design) and pick the first
    pick = {}
    for row in idx["rows"]:
        key = (row["topology_token"], row["design_id"])
        if key not in pick:
            pick[key] = row

    for (topo, design), row in sorted(pick.items()):
        gpath = paths.resolve_under_root(row["graph_path"], root)
        node_x, beam_x, plate_x, beam_ids = _read_inputs(gpath)
        desc = _descriptor(beam_x, plate_x)
        per_design[(topo, design)] = {
            "node_coord_hash": _hash_array(node_x[:, :3]),
            "plate_hash": _hash_array(plate_x[:, :5]) if plate_x.size else "empty",
            "plate_type_hash": _hash_array(plate_x[:, 5:6]) if plate_x.size else "empty",
            "beam_ids_hash": _hash_array(np.asarray(beam_ids)) if beam_ids else "empty",
            "beam_sections": beam_x[:, SECTION_COLS].copy(),
            "beam_ids": beam_ids,
            "descriptor": desc,
            "n_nodes": int(node_x.shape[0]),
            "n_beams": int(beam_x.shape[0]),
            "n_plates": int(plate_x.shape[0]),
        }
        meta = src_meta.get(topo, {}).get(design, {})
        desc_rows.append({
            "topology": topo, "design": design,
            "design_kind": meta.get("design_kind"),
            "design_description": meta.get("design_description"),
            "changed_element_count": meta.get("changed_element_count"),
            "n_nodes": int(node_x.shape[0]), "n_beams": int(beam_x.shape[0]),
            "n_plates": int(plate_x.shape[0]),
            **{f"feat_{i}": float(desc[i]) for i in range(9)},
        })

    with (out_dir / "design_descriptor.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(desc_rows[0].keys())); w.writeheader(); w.writerows(desc_rows)

    # per-topology structural variation
    summary = {}
    for topo in ("topology1", "topology2", "topology3"):
        designs = sorted(d for (t, d) in per_design if t == topo)
        recs = [per_design[(topo, d)] for d in designs]
        node_coords_identical = len({r["node_coord_hash"] for r in recs}) == 1
        plate_identical = len({r["plate_hash"] for r in recs}) == 1
        plate_type_identical = len({r["plate_type_hash"] for r in recs}) == 1
        beam_ids_identical = len({r["beam_ids_hash"] for r in recs}) == 1
        # changed beams vs first design (only if ids identical)
        changed_counts = []
        if beam_ids_identical:
            base = recs[0]["beam_sections"]
            for r in recs:
                cur = r["beam_sections"]
                if cur.shape == base.shape:
                    diff = np.any(np.abs(cur - base) > 0, axis=1)
                    changed_counts.append(int(diff.sum()))
                else:
                    changed_counts.append(None)
        # descriptor variation across designs
        D = np.vstack([r["descriptor"] for r in recs])  # (24, 9)
        feature_ranges = {f"feat_{i}": {"min": float(D[:, i].min()), "max": float(D[:, i].max()),
                                        "n_distinct": int(len(np.unique(np.round(D[:, i], 9))))}
                          for i in range(9)}
        summary[topo] = {
            "designs": len(designs),
            "node_coords_identical_across_designs": node_coords_identical,
            "plate_features_identical_across_designs": plate_identical,
            "plate_type_identical_across_designs": plate_type_identical,
            "beam_ids_identical_across_designs": beam_ids_identical,
            "changed_beam_counts_vs_first_design": changed_counts,
            "descriptor_feature_ranges": feature_ranges,
            "design_kind_counts": {
                k: sum(1 for (d, m) in src_meta.get(topo, {}).items() if m.get("design_kind") == k)
                for k in sorted({m.get("design_kind") for m in src_meta.get(topo, {}).values()})
            },
            "variation_class": (
                "section_change_only" if (node_coords_identical and plate_identical and beam_ids_identical)
                else "structural_change"
            ),
        }

    # cross-design coverage: test descriptors vs train descriptor range
    splits_dir = paths.SPLIT_DIR
    coverage_rows = []
    for topo, fname in (("topology1", "cross_design_topology1"),
                        ("topology2", "cross_design_topology2"),
                        ("topology3", "cross_design_topology3")):
        with (splits_dir / f"split_{fname}.json").open(encoding="utf-8") as fh:
            split = json.load(fh)
        for fold in split["folds"]:
            # group design keys are given directly
            test_designs = fold.get("test_group_keys", [])
            val_designs = fold.get("validation_group_keys", [])
            train_designs = fold.get("train_group_keys", [])
            Dtr = np.vstack([per_design[(topo, d)]["descriptor"] for d in train_designs])
            Dte = np.vstack([per_design[(topo, d)]["descriptor"] for d in test_designs])
            lo, hi = Dtr.min(axis=0), Dtr.max(axis=0)
            for fi in range(9):
                outside = ((Dte[:, fi] < lo[fi]) | (Dte[:, fi] > hi[fi]))
                max_beyond = max(0.0, float((lo[fi] - Dte[:, fi]).max()), float((Dte[:, fi] - hi[fi]).max()))
                coverage_rows.append({
                    "topology": topo, "fold_id": fold["fold_id"], "feature": f"feat_{fi}",
                    "train_min": float(lo[fi]), "train_max": float(hi[fi]),
                    "n_test_designs": len(test_designs),
                    "n_test_outside_train_range": int(outside.sum()),
                    "test_outside_fraction": float(outside.mean()),
                    "max_beyond_range": max_beyond,
                })

    with (out_dir / "cross_design_descriptor_coverage.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys())); w.writeheader(); w.writerows(coverage_rows)

    # aggregate interpolation/extrapolation verdict per topology
    cover_summary = {}
    for topo in ("topology1", "topology2", "topology3"):
        cr = [c for c in coverage_rows if c["topology"] == topo]
        frac_out = [c["test_outside_fraction"] for c in cr]
        cover_summary[topo] = {
            "folds": len({c["fold_id"] for c in cr}),
            "features_with_any_outside": sum(1 for c in cr if c["n_test_outside_train_range"] > 0),
            "total_feature_checks": len(cr),
            "mean_test_outside_fraction": float(np.mean(frac_out)) if frac_out else 0.0,
            "max_beyond_range": float(max(c["max_beyond_range"] for c in cr)) if cr else 0.0,
            "verdict": "interpolation_within_observed_section_range"
            if (cr and max(c["max_beyond_range"] for c in cr) == 0.0)
            else "includes_extrapolation_beyond_train_designs",
        }

    report = {
        "schema": "qt-three-topology-design-input-range-v1",
        "date": "2026-11-17",
        "note": "Structural inputs only; response labels not read.",
        "per_topology": summary,
        "cross_design_coverage": cover_summary,
        "descriptor_definition": "mean log10(beam.x cols 0-7) + mean plate thickness",
    }
    (out_dir / "design_input_range_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

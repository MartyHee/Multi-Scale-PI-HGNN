#!/usr/bin/env python
"""Build the design-shared member cache used by the fast training reader.

Only design-specific section/plate state is cached; response shards remain the
only per-sample source.  Geometry is reused from ``shared/<t>_geometry.npz`` and
node loads from ``fields/<t>_loads.npz`` (both built by build_processed.py).

The cache is versioned and refuses to overwrite unless --force.  For each
design it reads LF01 and one additional field and verifies that the member
state is field-independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from src.data.three_topology_fe_v2 import assemble, index as index_mod, paths  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402

TEMPLATES = ("topology1", "topology2", "topology3")
SCHEMA = "qt-fe-v2-design-cache-v1"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    ident = idx.validate_identity()
    reader = FeV2Reader(idx, max_runs=4, verify_hash=True)
    out_dir = paths.PROCESSED_ROOT / "shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_samples_sha256": idx.samples_sha256,
            "source_manifest_sha256": ident["manifest_sha256"],
            "policy": {"response_copy": False, "field_dependent_members": "verified",
                       "full_29_endpoint_sidecar": "remains in source extraction"},
            "templates": {}}
    for t in TEMPLATES:
        designs = [d for d in idx.designs if d["topology_token"] == t]
        designs.sort(key=lambda d: d["design_id"])
        if len(designs) != 48:
            raise SystemExit(f"{t}: expected 48 designs, got {len(designs)}")
        ref = None
        raw0 = None
        arrays = {k: [] for k in ("beam_raw", "plate_raw", "beam_end_prop_used",
                                  "beam_end_mask", "beam_sec_id", "beam_mat_id",
                                  "n_equations")}
        rep_sids, verified = [], []
        for d in designs:
            did = d["design_id"]
            rows = [r for r in idx.rows if r["topology_token"] == t and r["design_id"] == did]
            if len(rows) != 64:
                raise SystemExit(f"{t}/{did}: expected 64 fields, got {len(rows)}")
            lf01 = next(r for r in rows if r["field_id"] == "LF01")
            raw = assemble.assemble_raw(reader.read(lf01["sample_id"]))
            if ref is None:
                ref = raw
                raw0 = raw
            else:
                for key in ("node_ids", "beam_ids", "beam_node_ids", "plate_ids",
                            "plate_node_ids", "support_node_index", "link_ids",
                            "release_ids", "master_ids"):
                    if not np.array_equal(np.asarray(raw[key]), np.asarray(ref[key])):
                        raise SystemExit(f"{t}/{did}: fixed geometry {key} differs")
            arrays["beam_raw"].append(raw["beam_raw"])
            arrays["plate_raw"].append(raw["plate_raw"])
            arrays["beam_end_prop_used"].append(raw["beam_end_prop"][:, :, [0, 3, 4, 5]])
            arrays["beam_end_mask"].append(raw["beam_end_mask"])
            arrays["beam_sec_id"].append(raw["beam_sec_id"])
            arrays["beam_mat_id"].append(raw["beam_mat_id"])
            arrays["n_equations"].append(raw["n_equations"])
            rep_sids.append(lf01["sample_id"])
            # Verify member state is field-independent on one other field.
            other = rows[-1] if rows[-1]["field_id"] != "LF01" else rows[-2]
            raw2 = assemble.assemble_raw(reader.read(other["sample_id"]))
            for key in ("beam_raw", "plate_raw", "beam_end_mask", "beam_sec_id",
                        "beam_mat_id"):
                if not np.array_equal(np.asarray(raw[key]), np.asarray(raw2[key])):
                    raise SystemExit(f"{t}/{did}: {key} differs between fields")
            if not np.array_equal(raw["beam_end_prop"][:, :, [0, 3, 4, 5]],
                                  raw2["beam_end_prop"][:, :, [0, 3, 4, 5]], equal_nan=True):
                raise SystemExit(f"{t}/{did}: endpoint used columns differ between fields")
            verified.append(other["sample_id"])
        npz_path = out_dir / f"{t}_design_members.npz"
        if npz_path.exists() and not a.force:
            raise SystemExit(f"REFUSE existing design cache: {npz_path} (use --force)")
        np.savez_compressed(
            npz_path,
            design_ids=np.asarray([d["design_id"] for d in designs], dtype="U64"),
            design_identity_hash=np.asarray([d["design_identity_hash"] for d in designs], dtype="U64"),
            representative_sample_id=np.asarray(rep_sids, dtype="U128"),
            beam_raw=np.stack(arrays["beam_raw"]),
            plate_raw=np.stack(arrays["plate_raw"]),
            beam_end_prop_used=np.stack(arrays["beam_end_prop_used"]).astype(np.float64),
            beam_end_mask=np.stack(arrays["beam_end_mask"]).astype(bool),
            beam_sec_id=np.stack(arrays["beam_sec_id"]).astype(np.int64),
            beam_mat_id=np.stack(arrays["beam_mat_id"]).astype(np.int64),
            n_equations=np.asarray(arrays["n_equations"], dtype=np.int64),
        )
        rec = {"file": f"shared/{t}_design_members.npz", "bytes": npz_path.stat().st_size,
               "sha256": sha(npz_path), "designs": len(designs),
               "beams": int(raw0["beam_raw"].shape[0]),
               "plates": int(raw0["plate_raw"].shape[0]),
               "representative_samples": rep_sids,
               "verified_second_field_samples": verified,
               "fixed_geometry_same": True, "field_independent_members": True}
        meta["templates"][t] = rec
        mpath = out_dir / f"{t}_design_cache.json"
        mpath.write_text(json.dumps({"schema": SCHEMA, "topology": t, **rec},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
        meta["templates"][t]["meta_file"] = f"shared/{t}_design_cache.json"
        meta["templates"][t]["meta_sha256"] = sha(mpath)
    (out_dir / "design_cache_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

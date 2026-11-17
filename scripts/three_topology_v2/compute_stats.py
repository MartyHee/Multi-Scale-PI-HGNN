#!/usr/bin/env python
"""Deliverable 4 — fit per-fold normalization statistics + verify round-trip.

* cross_load_field folds: train set changed by the family fix -> REFIT.
* cross_design / leave_one_topology folds: train set unchanged -> verify against
  the accepted v1 artifact (numeric continuity) and only relabel BetaAngle to deg.

Also verifies per fold: fit uses train samples only; column order/units; preserved
columns untouched; label normalize->inverse round-trip within tolerance.
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
import torch  # noqa: E402

from src.data.three_topology_v2 import index, paths, splits as splits_mod, stats as stats_mod  # noqa: E402


def load_published_splits(root: Path):
    out = {}
    d = paths.SPLIT_DIR
    for p in sorted(d.glob("split_*.json")):
        if p.name == "split_index.json":
            continue
        with p.open("r", encoding="utf-8") as fh:
            out[p.stem.replace("split_", "")] = json.load(fh)
    return out


def roundtrip_check(stats, row_by_id, root):
    """Normalize a training sample then inverse-transform the labels."""
    sid = stats["train_sample_ids"][0]
    from src.data.three_topology_v2 import graph_io
    data = graph_io.load_graph(row_by_id[sid], root=root, dtype=torch.float64, verify_hash=False)
    y_disp = data["mesh_node"].y_disp.clone()
    y_force = data["beam_element"].y_force.clone()
    x_node_preserved = data["mesh_node"].x[:, 9:15].clone()
    stats_mod.normalize_labels_(data, stats)
    disp_n = data["mesh_node"].y_disp.clone()
    # denormalize back
    d_back = stats_mod.inverse_transform_disp(disp_n, stats)
    f_back = stats_mod.inverse_transform_force(data["beam_element"].y_force.clone(), stats)
    disp_err = float((d_back - y_disp).abs().max())
    force_err = float((f_back - y_force).abs().max())
    # preserved input columns unchanged under apply_normalization_
    stats_mod.apply_normalization_(data, stats)
    pres_ok = bool(torch.equal(data["mesh_node"].x[:, 9:15], x_node_preserved))
    # column order sanity: number of continuous columns normalised matches
    lk = stats_mod.stats_lookup(stats)
    shapes_ok = ({m: int(lk[m]["mean"].shape[0]) for m in lk}
                 == {m: len(stats["columns"][m]) for m in stats_mod.CONTINUOUS_MATRICES})
    return {
        "sample_id": sid,
        "disp_roundtrip_max_abs_err": disp_err,
        "force_roundtrip_max_abs_err": force_err,
        "preserved_columns_untouched": pres_ok,
        "column_shapes_ok": shapes_ok,
        "ok": bool(disp_err < 1e-8 and force_err < 1e-8 and pres_ok and shapes_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", help="split_id or 'all'")
    ap.add_argument("--only-cd1", action="store_true", help="only cross_design_topology1")
    args = ap.parse_args()

    t0 = time.time()
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    row_by_id = idx["row_by_id"]
    all_splits = load_published_splits(root)

    selected = {}
    if args.only_cd1:
        selected = {"cross_design_topology1": all_splits["cross_design_topology1"]}
    elif args.split == "all":
        selected = all_splits
    else:
        selected = {args.split: all_splits[args.split]}

    artifacts = []
    comparisons = {}
    for split_id, d in selected.items():
        kind = d.get("split_kind")
        topology_token = d.get("topology_token")
        refit = (kind == "cross_load_field")  # affected by family fix
        cache = stats_mod._SampleCache(root)  # fresh per split -> bounded memory
        for fold in d["folds"]:
            st = stats_mod.compute_fold_stats(
                fold, split_id, kind, topology_token, row_by_id, root, cache=cache,
                source_extra={"refit": refit, "split_source": "v2"})
            # BetaAngle already declared deg via splits.UNITS.
            rt = roundtrip_check(st, row_by_id, root)
            st["verification"] = {
                "roundtrip": rt,
                "fit_sample_source": {
                    "train_count_matches": len(fold["train_sample_ids"]) == st["counts"]["train_graphs"],
                    "train_list_sha256_matches": st["train_sample_list_sha256"]
                    == hashlib.sha256("\n".join(sorted(fold["train_sample_ids"])).encode()).hexdigest(),
                },
            }
            # v1 continuity for unchanged folds
            if not refit:
                v1 = stats_mod.load_v1_stats(split_id, fold["fold_id"], root)
                if v1 is not None:
                    cmp = stats_mod.compare_continuous_to_v1(st, v1)
                    st["verification"]["v1_numeric_match"] = cmp
                    comparisons[f"{split_id}/{fold['fold_id']}"] = cmp
                else:
                    st["verification"]["v1_numeric_match"] = {"ok": None, "note": "v1 missing"}
            art = stats_mod.save_stats(st)
            art["refit"] = refit
            art["roundtrip_ok"] = rt["ok"]
            artifacts.append(art)
        del cache

    stats_index = {
        "schema": stats_mod.STATS_INDEX_SCHEMA,
        "source": {"samples_jsonl_sha256": idx["hashes"]["samples_jsonl"],
                   "dataset_manifest_sha256": idx["hashes"]["dataset_manifest"]},
        "policy": {
            "fit_scope": "train_sample_ids only",
            "refit_rule": "cross_load_field refit; cross_design & leave_one_topology reused and verified",
            "BetaAngle": "deg (API value)",
        },
        "artifacts": artifacts,
    }
    paths.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(stats_index, ensure_ascii=False, indent=2, sort_keys=True)
    (paths.MANIFEST_DIR / "stats_index.json").write_text(payload, encoding="utf-8")

    all_rt_ok = all(a["roundtrip_ok"] for a in artifacts)
    all_cmp_ok = all(c.get("ok") in (True, None) for c in comparisons.values())
    print(json.dumps({
        "ok": all_rt_ok and all_cmp_ok,
        "elapsed_sec": round(time.time() - t0, 2),
        "folds_written": len(artifacts),
        "all_roundtrip_ok": all_rt_ok,
        "v1_numeric_continuity_ok": all_cmp_ok,
        "v1_comparisons_failed": [k for k, c in comparisons.items() if c.get("ok") is False],
        "stats_index": str(paths.MANIFEST_DIR / "stats_index.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if (all_rt_ok and all_cmp_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

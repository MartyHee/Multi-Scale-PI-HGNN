#!/usr/bin/env python
"""Build inner proxy folds for nested leave-one-topology evaluation.

For each outer LOTO fold (held-out topology H; training side {A,B} = the 216
outer-train samples), derive two inner proxy folds using ONLY the outer-train
samples:
    A->B : inner-train = A's outer-train samples, inner-val = B's outer-train
    B->A : inner-train = B's outer-train samples, inner-val = A's outer-train

Critical rule (research design revision): the normalisation statistics for an
inner fold are fit on the inner-TRAIN topology only (A), never on the outer
{A,B} statistics.  This script fits those inner stats and records hashes so the
"stats only from A" rule is auditable.

The v2 split files are NOT modified; this writes a separate manifest
``processed/three_topology_training_v2/manifests/inner_proxy_folds.json`` and
``processed/three_topology_training_v2/stats_inner/...``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_v2 import index, paths, stats as stats_mod  # noqa: E402


def list_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def graph_agg(ids, row_by_id):
    lines = [f"{s}:{row_by_id[s]['graph_sha256']}" for s in sorted(ids)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    row_by_id = idx["row_by_id"]

    with (paths.SPLIT_DIR / "split_leave_one_topology.json").open(encoding="utf-8") as fh:
        loto = json.load(fh)

    inner_dir = paths.TRAIN_ROOT / "stats_inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "qt-three-topology-v2-inner-proxy-folds-v1", "folds": []}

    for outer in loto["folds"]:
        held_out = outer["fold_id"].replace("loto_", "")
        outer_train = list(outer["train_sample_ids"])
        topo_of = {s: row_by_id[s]["topology_token"] for s in outer_train}
        train_topos = sorted(set(topo_of.values()))
        if len(train_topos) != 2:
            raise AssertionError(f"{outer['fold_id']}: expected 2 training topologies")
        cache = stats_mod._SampleCache(root)
        inner_folds = []
        for k, (tr_topo, va_topo) in enumerate([(train_topos[0], train_topos[1]),
                                                (train_topos[1], train_topos[0])]):
            tr_ids = sorted(s for s in outer_train if topo_of[s] == tr_topo)
            va_ids = sorted(s for s in outer_train if topo_of[s] == va_topo)
            pseudo = {"fold_id": f"inner_{k+1}_{tr_topo}_to_{va_topo}",
                      "train_sample_ids": tr_ids,
                      "train_sample_list_sha256": list_hash(tr_ids),
                      "train_graph_sha256_aggregate": graph_agg(tr_ids, row_by_id)}
            st = stats_mod.compute_fold_stats(
                pseudo, f"{outer['fold_id']}_inner", "inner_proxy", tr_topo,
                row_by_id, root, cache=cache,
                source_extra={"refit": True, "fit_topology_only": tr_topo,
                              "rule": "inner stats fit on inner-train topology only"})
            art = stats_mod.save_stats(st, out_dir=inner_dir)
            inner_folds.append({
                "inner_fold_id": pseudo["fold_id"],
                "train_topology": tr_topo, "validation_topology": va_topo,
                "train_samples": len(tr_ids), "validation_samples": len(va_ids),
                "train_sample_ids": tr_ids, "validation_sample_ids": va_ids,
                "train_sample_list_sha256": pseudo["train_sample_list_sha256"],
                "validation_sample_list_sha256": list_hash(va_ids),
                "train_graph_sha256_aggregate": pseudo["train_graph_sha256_aggregate"],
                "inner_stats_path": art["path"], "inner_stats_sha256": art["sha256"],
                "inner_stats_fit_topology": tr_topo,
            })
        del cache
        manifest["folds"].append({
            "outer_fold_id": outer["fold_id"], "held_out_topology": held_out,
            "outer_train_samples": len(outer_train),
            "outer_train_topologies": train_topos,
            "inner_folds": inner_folds,
        })

    (paths.MANIFEST_DIR / "inner_proxy_folds.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "outer_folds": len(manifest["folds"]),
                      "inner_folds": sum(len(f["inner_folds"]) for f in manifest["folds"]),
                      "manifest": str(paths.MANIFEST_DIR / "inner_proxy_folds.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

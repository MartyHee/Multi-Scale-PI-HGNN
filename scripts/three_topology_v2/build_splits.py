#!/usr/bin/env python
"""Deliverable 2 — publish the revised split + input-metadata set.

Rebuilds cross-load-field splits with the correct family key:
  topology1: source load_family {T1,N1},{T2,N2},{T3,N3}
  topology2/3: COARSE pure_vertical vs mixed_component
Reuses accepted v1 cross-design / leave-one-topology splits (verified).
Also writes input_metadata.json with BetaAngle declared in degrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_v2 import index, paths, splits  # noqa: E402


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    all_splits = splits.build_all_splits(idx, root=root)
    split_index = splits.write_splits(all_splits, idx)

    # Summarise the rebuilt cross-load-field folds and verify family rules.
    summary = {"schema": "qt-three-topology-v2-split-summary-v1",
               "cross_load_field": {}, "reused": {}}
    problems = []
    for sid, d in all_splits.items():
        if d.get("split_kind") == "cross_load_field":
            fams = {}
            for fold in d["folds"]:
                if fold["validation_load_family"] == fold["test_load_family"]:
                    problems.append(f"{sid}/{fold['fold_id']}: val==test family")
                twins = [t for t in fold["same_family_twins"]]
                if not all(t in fold["train_template_ids"] for t in twins):
                    problems.append(f"{sid}/{fold['fold_id']}: twin not in train")
                fams[fold["fold_id"]] = {
                    "test": fold["test_template_id"], "test_family": fold["test_load_family"],
                    "val": fold["validation_template_id"], "val_family": fold["validation_load_family"],
                    "twins": twins,
                }
            summary["cross_load_field"][sid] = {
                "family_note": d["family_note"], "folds": fams,
            }
        else:
            summary["reused"][sid] = {
                "v1_source_sha256": d.get("_v1_source_sha256"),
                "folds": len(d["folds"]),
            }
    summary["problems"] = problems
    summary["ok"] = not problems

    out = paths.MANIFEST_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"ok": summary["ok"], "problems": problems,
                      "split_index": split_index["artifacts"],
                      "cross_load_field": summary["cross_load_field"]},
                     ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

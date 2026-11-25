#!/usr/bin/env python
"""Freeze the implementation protocol while preserving the earlier candidate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol  # noqa: E402


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    out = paths.PROCESSED_ROOT
    cand_path = out / "splits" / "protocol_candidate.json"
    prot = json.loads(cand_path.read_text(encoding="utf-8"))
    counts = protocol.role_counts(prot)
    idx = index_mod.load_index()
    frozen = {
        "schema": "qt-fe-v2-protocol-v1",
        "status": "frozen_for_pipeline",
        "formal_training_authorized": False,
        "source_candidate": cand_path.name,
        "source_candidate_sha256": sha(cand_path),
        "source": prot["source"],
        "seed": prot["seed"],
        "groups": prot["groups"],
        "roles": prot["roles"],
        "counts": counts,
        "selection_rule": {
            "training_core_per_template": 1360,
            "validation_slices_per_template": ["D_val x F_train", "D_train x F_val", "D_val x F_val"],
            "slice_metrics": ["graph_equal", "task_equal", "component_equal", "normalized_mse"],
            "common_selection": "equal mean of the three slice selection metrics",
            "test": "same best checkpoint would be used for three test slices; not executed in this round",
            "test_cells": ["D_test x F_train", "D_train x F_test", "D_test x F_test"],
        },
        "notes": ["test cells are not loaded by local smoke",
                  "validation may be used for early stopping and checkpoint selection",
                  "this freeze does not authorize formal training or test scoring"],
    }
    path = out / "splits" / "protocol_v1.json"
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger = out / "splits" / "role_ledger_v1.csv"
    protocol.write_ledger(prot, ledger)
    stats = {}
    for t in ("topology1", "topology2", "topology3"):
        sp = out / "stats" / f"{t}.json"
        if sp.is_file():
            stats[t] = {"path": sp.relative_to(out).as_posix(), "sha256": sha(sp)}
    manifest = {
        "schema": "qt-fe-v2-split-manifest-v1", "date": "2026-11-25",
        "protocol": path.relative_to(out).as_posix(), "protocol_sha256": sha(path),
        "ledger": ledger.relative_to(out).as_posix(), "ledger_sha256": sha(ledger),
        "candidate": cand_path.relative_to(out).as_posix(), "candidate_sha256": sha(cand_path),
        "stats": stats,
        "source_samples_sha256": idx.samples_sha256,
        "not_formal_training_authorized": True,
    }
    (out / "splits" / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

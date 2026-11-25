#!/usr/bin/env python
"""Fit per-template statistics on the common D_train x F_train core only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.PACKAGE_ROOT))
    ap.add_argument("--out", default=str(paths.PROCESSED_ROOT))
    ap.add_argument("--topology", default=None, choices=["topology1", "topology2", "topology3"])
    a = ap.parse_args()
    index = index_mod.load_index(Path(a.root))
    out = Path(a.out)
    prot_path = out / "splits" / "protocol_candidate.json"
    if not prot_path.is_file():
        raise SystemExit(f"missing protocol candidate: {prot_path}")
    prot = json.loads(prot_path.read_text(encoding="utf-8"))
    topologies = [a.topology] if a.topology else ["topology1", "topology2", "topology3"]
    results = {}
    for t in topologies:
        ids = [sid for sid, v in prot["roles"].items()
               if sid in index.by_id and index.by_id[sid]["topology_token"] == t and v["A"] == "train"]
        if len(ids) != 1360:
            raise SystemExit(f"{t}: expected 1360 training-core samples, got {len(ids)}")
        path = out / "stats" / f"{t}.json"
        st = stats.fit_stats(index, ids, path, t,
                             provenance={"source": "G2 seed 20261120 common core",
                                         "role": "A/B/C train", "scope": "1,360 samples"})
        results[t] = {"path": str(path), "samples": len(ids),
                      "node_rows": st["counts"]["node_rows"],
                      "beam_rows": st["counts"]["beam_rows"],
                      "plate_rows": st["counts"]["plate_rows"],
                      "hash": st["_file_sha256"],
                      "constant_label_components": {
                          k: [c["name"] for c in st["columns"][k] if c["zero_variance"]]
                          for k in ("y_disp", "y_force")}}
        print(t, json.dumps(results[t], ensure_ascii=False), flush=True)
    (out / "stats" / "fit_summary.json").write_text(
        json.dumps({"schema": "qt-fe-v2-stats-fit-v1", "fit_scope": "train core only",
                    "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

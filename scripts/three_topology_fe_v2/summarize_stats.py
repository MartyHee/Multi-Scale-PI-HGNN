#!/usr/bin/env python
"""Summarize/verify the three frozen training-core statistics files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats as stats_mod  # noqa: E402


def main() -> int:
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
    out = {"schema": "qt-fe-v2-stats-summary-v1", "templates": {}, "problems": []}
    for t in ("topology1", "topology2", "topology3"):
        st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{t}.json")
        train_core = {s for s, v in prot["roles"].items()
                      if idx.by_id[s]["topology_token"] == t and v["A"] == "train"}
        got = set(st["train_sample_ids"])
        if got != train_core:
            out["problems"].append(f"{t}: stats train ids != common core")
        for role in ("val", "test", "blocked"):
            bad = {s for s, v in prot["roles"].items()
                   if idx.by_id[s]["topology_token"] == t and v["A"] == role} & got
            if bad:
                out["problems"].append(f"{t}: stats includes {role} ids")
        out["templates"][t] = {
            "hash": st["_file_sha256"], "samples": len(got), "counts": st["counts"],
            "constant_label_components": {
                k: [c["name"] for c in st["columns"][k] if c["zero_variance"]]
                for k in ("y_disp", "y_force")},
            "zero_scale_input_columns": {
                k: [c["name"] for c in st["columns"][k]
                    if c["kind"] in ("zscore", "zero_preserve_rms")
                    and c["effective_std"] == 1.0 and c["zero_variance"]]
                for k in ("node_x", "beam_x", "plate_x")},
            "node_load_zero_ratio": {
                c["name"]: (c["zero_count"] / c["count"] if c["count"] else None)
                for c in st["columns"]["node_x"] if c["kind"] == "zero_preserve_rms"},
            "label_scales": {k: [c["effective_std"] for c in st["columns"][k]]
                             for k in ("y_disp", "y_force")},
        }
    out["ok"] = not out["problems"]
    (paths.PROCESSED_ROOT / "stats" / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": out["ok"], "problems": out["problems"],
                      "templates": {t: {"samples": v["samples"], "hash": v["hash"],
                                        "constant_labels": v["constant_label_components"]}
                                    for t, v in out["templates"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

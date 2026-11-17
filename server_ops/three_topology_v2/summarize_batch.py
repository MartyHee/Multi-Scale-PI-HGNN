#!/usr/bin/env python
"""Read-only batch status checker for the four-terminal first batch.

Scans each task's STATUS.json / run_manifest.json and required artifact files.
Does NOT write into run dirs. Exit 0 only if all 8 expected runs are done and
their required files exist; otherwise exit 1.

Usage:
  BATCH_ID=... PROJECT_ROOT=... python summarize_batch.py
  python summarize_batch.py --batch-dir /path/to/.../first_batch/<BATCH_ID>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXPECTED = [
    ("t1_mlp", "mlp_seed20261117", "gradient"),
    ("t1_mlp", "mlp_seed20261118", "gradient"),
    ("t1_mlp", "mlp_seed20261119", "gradient"),
    ("t1_ridge", "ridge", "ml"),
    ("t1_histgb", "hist_gb", "ml"),
    ("t2_hgt", "hgt_global_seed20261117", "gradient"),
    ("t3_hgt", "hgt_global_seed20261118", "gradient"),
    ("t4_hgt", "hgt_global_seed20261119", "gradient"),
]

GRAD_REQUIRED = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json",
                 "summary.json", "train_log.csv", "driver.log", "val_predictions.json",
                 "checkpoints/best.pt", "checkpoints/last.pt"]
ML_REQUIRED = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json",
               "summary.json", "driver.log", "val_predictions.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    batch_dir = a.batch_dir or os.environ.get("BATCH_DIR")
    if not batch_dir:
        pr = os.environ.get("PROJECT_ROOT"); bid = os.environ.get("BATCH_ID")
        if not pr or not bid:
            raise SystemExit("set --batch-dir or PROJECT_ROOT+BATCH_ID")
        batch_dir = str(Path(pr) / "outputs" / "three_topology_v2" / "first_batch" / bid)
    batch_dir = Path(batch_dir)
    rows = []
    ok = True
    for worker, run, kind in EXPECTED:
        d = batch_dir / worker / run
        row = {"worker": worker, "run": run, "kind": kind, "dir": str(d)}
        if not d.is_dir():
            row["state"] = "MISSING"; ok = False; rows.append(row); continue
        st_path = d / "STATUS.json"
        state = "NO_STATUS"
        if st_path.is_file():
            try:
                state = json.loads(st_path.read_text(encoding="utf-8")).get("state", "UNKNOWN")
            except Exception:  # noqa: BLE001
                state = "BAD_STATUS"
        row["state"] = state
        req = list(GRAD_REQUIRED if kind == "gradient" else ML_REQUIRED)
        if kind == "ml":
            req.append(f"ml_{run}.pkl")  # ridge -> ml_ridge.pkl; hist_gb -> ml_hist_gb.pkl
        missing = [f for f in req if not (d / f).is_file()]
        row["missing"] = missing
        # read selection metric if available
        sel = None
        try:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            m = s.get("models", {})
            if m:
                sel = next(iter(m.values())).get("selection_metric")
        except Exception:  # noqa: BLE001
            pass
        row["selection_metric"] = sel
        if state != "done" or missing:
            ok = False
        rows.append(row)
    summary = {"batch_dir": str(batch_dir), "expected": len(EXPECTED),
               "done": sum(1 for r in rows if r["state"] == "done" and not r.get("missing")),
               "failed": [r["run"] for r in rows if r["state"] == "failed"],
               "missing_or_incomplete": [r["run"] for r in rows if r["state"] != "done" or r.get("missing")],
               "overall_ok": ok, "rows": rows}
    if a.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"batch: {batch_dir}")
        for r in rows:
            flag = "OK" if (r["state"] == "done" and not r.get("missing")) else "!!"
            print(f"  [{flag}] {r['worker']}/{r['run']:<28} state={r['state']:<9} "
                  f"sel={r['selection_metric']} missing={r.get('missing', [])}")
        print(f"overall_ok={ok}  done={summary['done']}/{summary['expected']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

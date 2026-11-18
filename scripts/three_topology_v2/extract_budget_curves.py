#!/usr/bin/env python
"""Extract budget-point selection metrics from a batch's per-epoch train_log.csv.

Default budget points: 60, 100, 150, 200. They are read from the SAME training
trajectory; no checkpoint/prediction at those points is assumed and none is
created. If a run stops (early stopping) before a budget, the actual stop is
reported and the budget is marked "not reached"; no padding/forging.

Only selection metrics are promised. Per-component metrics at intermediate
budgets would require the corresponding checkpoint/prediction, which is not
saved by this batch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def read_rows(p: Path):
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines[1:]:
        try:
            v = next(csv.reader([line]))
            if len(v) < 12:
                continue
            int(v[0]); float(v[2]); float(v[3])
            rows.append(v)
        except Exception:  # noqa: BLE001
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="budget200_20261118_01")
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    ap.add_argument("--budgets", default="60,100,150,200")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(native(a.project_root))
    batch = a.batch
    man = json.loads((root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch}.json")
                     .read_text(encoding="utf-8"))
    budgets = [int(x) for x in a.budgets.split(",") if x.strip()]
    batch_dir = root / "outputs" / "three_topology_v2" / "first_batch" / batch
    out = Path(a.out) if a.out else root / "outputs" / "three_topology_v2" / "budget200_review" / batch
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in man["tasks"]:
        if t["kind"] != "gradient":
            continue
        d = batch_dir / t["worker"] / t["run"]
        ep = read_rows(d / "train_log.csv")
        if not ep:
            rows.append({"worker": t["worker"], "run": t["run"], "model": t["model"],
                         "seed": t["seed"], "note": "no complete train_log rows"})
            continue
        completed = len(ep)
        for b in budgets:
            n = min(b, completed)
            if n == 0:
                continue
            best = min(float(x[3]) for x in ep[:n])
            best_ep = min(range(n), key=lambda i: float(ep[i][3]))
            rows.append({"worker": t["worker"], "run": t["run"], "model": t["model"],
                         "seed": t["seed"], "budget": b, "reached": completed >= b,
                         "epochs_observed": n, "best_selection_at_budget": best,
                         "best_epoch1_at_budget": best_ep + 1,
                         "actual_completed_epochs": completed,
                         "actual_best_epoch1": (min(range(completed), key=lambda i: float(ep[i][3])) + 1),
                         "early_stopped": bool(json.loads((d / "run_log.json").read_text(encoding="utf-8")).get("early_stopped"))
                         if (d / "run_log.json").is_file() else None,
                         "note": "" if completed >= b else "提前早停/未达到该预算；使用实际停止时的 best，不填充"})
    with (out / "budget_points.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "budget_points.json").write_text(json.dumps(
        {"batch_id": batch, "budgets": budgets, "rows": rows,
         "note": "selection metrics only; intermediate checkpoints/predictions are not saved"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"batch_id": batch, "budgets": budgets, "runs": len({r['run'] for r in rows}),
                      "rows": len(rows), "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

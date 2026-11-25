#!/usr/bin/env python
"""Read-only monitor for an FE v2 batch.

Shows expected/done/running/failed/waiting counts, per-task progress, the
current and best three-slice aggregate, per-slice values, last update and run
path.  Distinguishes early stop, epoch-budget completion, abnormal exit and
stale "running" statuses.  A running task without summary.json is normal.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def manifest(batch: str, root: Path) -> dict:
    base = Path(native(os.environ.get("THREE_TOPO_MANIFEST_ROOT")
                       or (root / "remote_jobs" / "three_topology_fe_v2" / "manifests")))
    p = base / f"{batch}.json"
    if not p.is_file():
        raise SystemExit(f"manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def attempt_dirs(base: Path):
    if not base.is_dir():
        return []
    out = []
    for p in base.iterdir():
        if p.is_dir() and re.match(r"^attempt_\d+$", p.name):
            out.append((int(p.name.split("_")[1]), p))
    return sorted(out)


def last_epoch(log_path: Path):
    if not log_path.is_file():
        return None
    with log_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def task_state(d: Path, stale_sec: int):
    st = {}
    if (d / "STATUS.json").is_file():
        try:
            st = json.loads((d / "STATUS.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            st = {"state": "BAD_STATUS"}
    state = st.get("state", "NO_STATUS")
    summary = {}
    if (d / "summary.json").is_file():
        try:
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            summary = {}
    last = last_epoch(d / "train_log.csv")
    cur = last.get("selection_metric") if last else summary.get("selection_metric")
    best = last.get("best_val") if last else summary.get("selection_metric")
    slices = summary.get("slice_selection") or {}
    if not slices and (d / "slice_metrics.json").is_file():
        try:
            sm = json.loads((d / "slice_metrics.json").read_text(encoding="utf-8"))
            slices = {k: v.get("selection_metric") for k, v in sm.get("slices", {}).items()}
        except Exception:  # noqa: BLE001
            slices = {}
    stop = summary.get("stop_reason") or ""
    if state == "running" and not summary:
        stamps = [p.stat().st_mtime for p in (d / "STATUS.json", d / "train_log.csv") if p.is_file()]
        age = time.time() - max(stamps) if stamps else 10 ** 9
        if age > stale_sec:
            state = "STALE"
            label = "stale"
        else:
            label = "running"
    elif state == "done" and not summary:
        state = "INCONSISTENT"; label = "done_without_summary"
    elif state == "done" and stop == "early_stop":
        label = "early_stop"
    elif state == "done" and stop == "epoch_budget_reached":
        label = "budget_reached"
    elif state == "done":
        label = "done"
    elif state == "failed":
        label = "failed"
    else:
        label = state.lower()
    epoch = last.get("epoch") if last else None
    return {"state": state, "label": label, "stop_reason": stop,
            "epoch": epoch, "current": cur, "best": best, "slices": slices,
            "summary": summary, "updated": st.get("updated_at") or st.get("finished_at")}


def collect(batch: str, root: Path, out_root: Path, stale_sec: int):
    man = manifest(batch, root)
    rows, counts = [], {"expected": len(man["tasks"]), "done": 0, "running": 0,
                        "failed": 0, "waiting": 0, "stale": 0, "inconsistent": 0}
    for t in man["tasks"]:
        base = out_root / batch / t["topology"] / t["worker"] / t["run"]
        attempts = attempt_dirs(base)
        row = {"topology": t["topology"], "worker": t["worker"], "model": t["model"],
               "seed": t["seed"], "run": t["run"], "path": str(base),
               "attempt": None, "state": "WAITING", "label": "waiting",
               "epoch": None, "current": None, "best": None, "slices": {},
               "stop_reason": "", "updated": None}
        if attempts:
            attempt, d = attempts[-1]
            row["attempt"] = attempt
            row.update(task_state(d, stale_sec))
            row["path"] = str(d)
        label = row["label"]
        if label in ("done", "early_stop", "budget_reached"):
            counts["done"] += 1
        elif label == "failed":
            counts["failed"] += 1
        elif label == "running":
            counts["running"] += 1
        elif label == "waiting" or not attempts:
            counts["waiting"] += 1
        elif label == "STALE":
            counts["stale"] += 1
        else:
            counts["inconsistent"] += 1
        rows.append(row)
    return man, rows, counts


def render(man, rows, counts):
    print(f"batch: {man['batch_id']}  expected={counts['expected']} done={counts['done']} "
          f"running={counts['running']} failed={counts['failed']} waiting={counts['waiting']} "
          f"stale={counts['stale']} inconsistent={counts['inconsistent']}")
    print(f"{'topology':<10} {'model':<12} {'seed':<10} {'state':<14} "
          f"{'epoch':>7} {'cur':>10} {'best':>10} {'d/l/j':<28} updated")
    for r in rows:
        ep = "-" if r["epoch"] is None else str(r["epoch"])
        cur = "-" if r["current"] is None else f"{float(r['current']):.5f}"
        best = "-" if r["best"] is None else f"{float(r['best']):.5f}"
        sl = r["slices"] or {}
        trio = "/".join("-" if sl.get(k) is None else f"{float(sl[k]):.4f}"
                        for k in ("design_val", "load_val", "joint_val"))
        print(f"{r['topology']:<10} {r['model']:<12} {str(r['seed']):<10} {r['label']:<14} "
              f"{ep:>7} {cur:>10} {best:>10} {trio:<28} {r['updated'] or '-'}")
        print(f"  {r['path']}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(MODEL_ROOT)))
    ap.add_argument("--output-root", default=os.environ.get("THREE_TOPO_FORMAL_OUTPUT_ROOT", ""))
    ap.add_argument("--stale-sec", type=int, default=int(os.environ.get("THREE_TOPO_STALE_SEC", "1800")))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=int, default=0)
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()
    root = Path(native(a.project_root))
    out_root = Path(native(a.output_root or (root / "outputs" / "three_topology_fe_v2" / "formal")))
    while True:
        man, rows, counts = collect(a.batch, root, out_root, a.stale_sec)
        if a.json:
            print(json.dumps({"batch_id": man["batch_id"], "counts": counts, "rows": rows},
                             ensure_ascii=False, indent=2))
        else:
            render(man, rows, counts)
        if a.watch <= 0:
            return 0
        time.sleep(a.interval)
        a.watch -= 1


if __name__ == "__main__":
    raise SystemExit(main())

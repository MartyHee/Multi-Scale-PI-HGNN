#!/usr/bin/env python
"""Read-only batch status checker driven by the frozen batch manifest.

The expected task list (worker/run/model/seed/kind) comes from
``remote_jobs/three_topology_v2/manifests/<batch_id>.json`` — the same file
preflight/monitor/collect use.  A batch is reported ok ONLY if every expected
task is present, done, identity-consistent and has its required files.  Missing
tasks, failed tasks, identity mismatches and duplicate target directories are
never reported as complete.  Extra run dirs (e.g. a resume worker) are listed
separately, not silently treated as this batch's results.

Usage:
  python summarize_batch.py --job remote_jobs/three_topology_v2/gcn.yaml --json
  python summarize_batch.py --batch sb_20261118_01
  python summarize_batch.py --batch-dir <extracted-archive>/batch/<batch_id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]

GRAD_REQUIRED = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json",
                 "summary.json", "train_log.csv", "driver.log", "val_predictions.json",
                 "checkpoints/best.pt", "checkpoints/last.pt"]
ML_REQUIRED = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json",
               "summary.json", "driver.log", "val_predictions.json"]


def load_manifest(batch: str, root: Path) -> dict:
    p = root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch}.json"
    if not p.is_file():
        raise SystemExit(f"frozen batch manifest not found: {p}")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("schema") != "qt-three-topology-batch-manifest-v2":
        raise SystemExit(f"manifest schema {man.get('schema')!r} is not v2; regenerate with --write")
    if not man.get("n_tasks") or not man.get("tasks"):
        raise SystemExit(f"manifest for {batch} has no tasks")
    return man


def batch_from_job(job: str) -> str:
    sys.path.insert(0, str(HERE))
    import _job_parse  # noqa: PLC0415
    return _job_parse.load(Path(job))["batch_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--batch-dir", default=None)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(MODEL_ROOT)))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(native(a.project_root))
    batch = a.batch or (batch_from_job(a.job) if a.job else None)
    if a.batch_dir:
        bd = Path(a.batch_dir)
        batch = batch or bd.name
    else:
        if not batch:
            raise SystemExit("need --job, --batch or --batch-dir")
        bd = root / "outputs" / "three_topology_v2" / "first_batch" / batch
    man = load_manifest(batch, root)

    rows, ok = [], True
    targets = {}
    for t in man["tasks"]:
        target = (t["worker"], t["run"])
        if target in targets:
            rows.append({"worker": t["worker"], "run": t["run"], "model": t["model"],
                         "seed": t["seed"], "kind": t["kind"], "state": "DUPLICATE_IDENTITY",
                         "missing": [], "identity_ok": False})
            ok = False
            continue
        targets[target] = True
        d = bd / t["worker"] / t["run"]
        row = {"worker": t["worker"], "run": t["run"], "model": t["model"],
               "seed": t["seed"], "kind": t["kind"], "dir": str(d),
               "state": None, "missing": [], "selection_metric": None,
               "identity_ok": None}
        if not d.is_dir():
            row["state"] = "MISSING"; ok = False; rows.append(row); continue
        st = {}
        if (d / "STATUS.json").is_file():
            try:
                st = json.loads((d / "STATUS.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                st = {"state": "BAD_STATUS"}
        row["state"] = st.get("state", "NO_STATUS")
        req = list(GRAD_REQUIRED if t["kind"] == "gradient" else ML_REQUIRED)
        if t["kind"] == "ml":
            req.append(f"ml_{t['run']}.pkl")
        row["missing"] = [f for f in req if not (d / f).is_file()]
        # identity: STATUS (and TASK.json when present) must match the manifest
        id_ok = (st.get("batch_id") == batch and st.get("worker_id") == t["worker"]
                 and st.get("model") == t["model"] and str(st.get("seed")) == str(t["seed"]))
        if (d / "TASK.json").is_file():
            try:
                task = json.loads((d / "TASK.json").read_text(encoding="utf-8"))
                id_ok = id_ok and task.get("batch_id") == batch and \
                    task.get("worker_id") == t["worker"] and task.get("model") == t["model"] \
                    and str(task.get("seed")) == str(t["seed"])
            except Exception:  # noqa: BLE001
                id_ok = False
        row["identity_ok"] = id_ok
        try:
            m = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("models", {})
            if m:
                row["selection_metric"] = next(iter(m.values())).get("selection_metric")
        except Exception:  # noqa: BLE001
            pass
        if row["state"] != "done" or row["missing"] or not id_ok:
            ok = False
        rows.append(row)

    known = {f"{t['worker']}/{t['run']}" for t in man["tasks"]}
    extra = []
    if bd.is_dir():
        for wd in sorted(bd.iterdir()):
            if not wd.is_dir():
                continue
            for rd in sorted(wd.iterdir()):
                if rd.is_dir() and f"{wd.name}/{rd.name}" not in known:
                    extra.append(f"{wd.name}/{rd.name}")

    summary = {
        "batch_id": batch, "batch_dir": str(bd),
        "manifest_n_tasks": man["n_tasks"], "expected": len(man["tasks"]),
        "done": sum(1 for r in rows if r["state"] == "done" and not r["missing"]
                    and r["identity_ok"]),
        "failed": [f"{r['worker']}/{r['run']}" for r in rows if r["state"] == "failed"],
        "missing_or_incomplete": [f"{r['worker']}/{r['run']}" for r in rows
                                  if r["state"] != "done" or r["missing"] or not r["identity_ok"]],
        "unexpected_run_dirs": extra,
        "unexpected_run_dirs_present": bool(extra),
        "overall_ok": bool(ok and man["n_tasks"] > 0 and man["n_tasks"] == len(man["tasks"])),
        "rows": rows,
    }
    if a.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"batch: {batch} dir: {bd} (manifest {man['n_tasks']} tasks)")
        for r in rows:
            flag = "OK" if (r["state"] == "done" and not r["missing"] and r["identity_ok"]) else "!!"
            print(f"  [{flag}] {r['worker']}/{r['run']:<30} state={r['state']:<9} "
                  f"sel={r['selection_metric']} identity={r['identity_ok']} missing={r['missing']}")
        if extra:
            print(f"  unexpected run dirs (not in manifest): {extra}")
        print(f"overall_ok={summary['overall_ok']}  done={summary['done']}/{summary['expected']}")
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

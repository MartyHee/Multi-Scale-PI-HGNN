#!/usr/bin/env python
"""Read-only acceptance summary for an FE v2 batch.

Expected tasks come from the frozen manifest.  A batch is ok ONLY when every
expected task is done, identity-consistent and has its required files.  A
failed original task may be resolved by an explicit resume attempt (recorded in
``outputs/three_topology_v2/resume_checks/<batch>/*.tsv``); the original is then
marked superseded and counted once, not twice.  Missing/failed/duplicate tasks
are never reported as complete.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]

GRAD_REQUIRED = ["STATUS.json", "TASK.json", "console.log", "run_log.json",
                 "summary.json", "train_log.csv", "slice_metrics.json",
                 "checkpoints/best.pt", "checkpoints/last.pt"]
ML_REQUIRED = ["STATUS.json", "TASK.json", "console.log", "summary.json",
               "slice_metrics.json", "ml_fit.json"]
SLICE_NAMES = ("design_val", "load_val", "joint_val")


def missing_required(d: Path, kind: str) -> list:
    """Required files; per-slice predictions accept legacy JSON or new NPZ."""
    req = list(GRAD_REQUIRED if kind == "gradient" else ML_REQUIRED)
    missing = [f for f in req if not (d / f).is_file()]
    for s in SLICE_NAMES:
        if not any((d / f"val_predictions_{s}.{ext}").is_file() for ext in ("json", "npz")):
            missing.append(f"val_predictions_{s}.json|npz")
    return missing


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def load_manifest(batch: str, root: Path) -> dict:
    base = Path(native(os.environ.get("THREE_TOPO_MANIFEST_ROOT")
                       or (root / "remote_jobs" / "three_topology_fe_v2" / "manifests")))
    p = base / f"{batch}.json"
    if not p.is_file():
        raise SystemExit(f"frozen batch manifest not found: {p}")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("schema") != "qt-fe-v2-batch-manifest-v1":
        raise SystemExit(f"manifest schema {man.get('schema')!r} is not v1")
    if not man.get("n_tasks") or not man.get("tasks"):
        raise SystemExit(f"manifest for {batch} has no tasks")
    return man


def attempts(base: Path):
    if not base.is_dir():
        return []
    out = []
    for p in base.iterdir():
        if p.is_dir() and re.match(r"^attempt_\d+$", p.name):
            out.append((int(p.name.split("_")[1]), p))
    return sorted(out)


def resume_maps(batch: str, root: Path):
    d = root / "outputs" / "three_topology_fe_v2" / "resume_checks" / batch
    maps = []
    if d.is_dir():
        for p in sorted(d.glob("*.tsv")):
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 11:
                        continue
                    maps.append({"worker": parts[0], "topology": parts[1], "run": parts[2],
                                 "model": parts[3], "seed": parts[4],
                                 "orig_worker": parts[5], "orig_run": parts[6],
                                 "orig_attempt": int(parts[7]), "target_attempt": int(parts[8]),
                                 "checkpoint": parts[9], "checkpoint_sha256": parts[10],
                                 "map_file": str(p)})
    return maps


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def inspect_task(t: dict, d: Path, batch: str):
    kind = t["kind"]
    req = missing_required(d, kind)
    if kind == "ml" and not (d / f"ml_{t['run']}.pkl").is_file():
        req.append(f"ml_{t['run']}.pkl")
    row = {"state": None, "missing": req, "identity_ok": None,
           "selection_metric": None, "stop_reason": None,
           "run_dir": str(d), "attempt": None}
    if not d.is_dir():
        row["state"] = "MISSING"
        return row
    m = re.match(r"^attempt_(\d+)$", d.name)
    if m:
        row["attempt"] = int(m.group(1))
    st = read_json(d / "STATUS.json") if (d / "STATUS.json").is_file() else {}
    row["state"] = st.get("state", "NO_STATUS")
    task = read_json(d / "TASK.json") if (d / "TASK.json").is_file() else {}
    id_ok = (st.get("batch_id") == batch and st.get("topology") == t["topology"]
             and st.get("worker") == t["worker"] and st.get("model") == t["model"]
             and str(st.get("seed")) == str(t["seed"]))
    if task:
        id_ok = id_ok and task.get("batch_id") == batch and \
            task.get("topology") == t["topology"] and task.get("worker") == t["worker"] \
            and task.get("model") == t["model"] and str(task.get("seed")) == str(t["seed"])
    row["identity_ok"] = bool(id_ok)
    summary = read_json(d / "summary.json") if (d / "summary.json").is_file() else {}
    row["selection_metric"] = summary.get("selection_metric")
    row["stop_reason"] = summary.get("stop_reason")
    row["summary_ok"] = bool(summary) and row["selection_metric"] is not None
    return row


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(MODEL_ROOT)))
    ap.add_argument("--output-root", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(native(a.project_root))
    out_root = Path(native(a.output_root or os.environ.get("THREE_TOPO_FORMAL_OUTPUT_ROOT")
                           or (root / "outputs" / "three_topology_fe_v2" / "formal")))
    man = load_manifest(a.batch, root)
    rmap = resume_maps(a.batch, root)
    targets = {}
    for m in rmap:
        targets[(m["worker"], m["topology"], m["run"], m["target_attempt"])] = m

    rows, ok = [], True
    expected_keys = set()
    for t in man["tasks"]:
        key = (t["worker"], t["topology"], t["run"])
        expected_keys.add(key)
        base = out_root / a.batch / t["topology"] / t["worker"] / t["run"]
        direct = attempts(base)
        row = {"topology": t["topology"], "worker": t["worker"], "run": t["run"],
               "model": t["model"], "seed": t["seed"], "kind": t["kind"],
               "resolved_by": None, "superseded_by": None}
        resolved = None
        if direct:
            attempt, d = direct[-1]
            info = inspect_task(t, d, a.batch)
            row.update(info)
            if info["state"] == "done" and not info["missing"] and info["identity_ok"] \
                    and info.get("summary_ok"):
                resolved = "direct"
        if resolved is None:
            # look for a resume attempt whose original is this task
            for m in rmap:
                if (m["orig_worker"], m["orig_run"]) != (t["worker"], t["run"]) \
                        or m["topology"] != t["topology"] or m["model"] != t["model"]:
                    continue
                rd = out_root / a.batch / t["topology"] / m["worker"] / m["run"] / f"attempt_{m['target_attempt']:02d}"
                tr = dict(t); tr["worker"] = m["worker"]; tr["run"] = m["run"]
                info = inspect_task(tr, rd, a.batch)
                info["worker"] = m["worker"]
                info["run"] = m["run"]
                if info["state"] == "done" and not info["missing"] \
                        and info.get("summary_ok"):
                    row.update(info)
                    row["resolved_by"] = "resume"
                    row["superseded_by"] = f"{m['worker']}/{m['run']}/attempt_{m['target_attempt']:02d}"
                    resolved = "resume"
                    break
        row["resolved_by"] = row.get("resolved_by") or resolved
        if resolved is None:
            ok = False
            if not direct:
                row.setdefault("state", "MISSING")
        rows.append(row)

    # extra attempt dirs / run dirs not part of the expected identities
    extra = []
    batch_dir = out_root / a.batch
    if batch_dir.is_dir():
        for topo_dir in batch_dir.iterdir():
            if not topo_dir.is_dir():
                continue
            for wd in topo_dir.iterdir():
                if not wd.is_dir():
                    continue
                for rd in wd.iterdir():
                    if not rd.is_dir():
                        continue
                    key = (wd.name, topo_dir.name, rd.name)
                    if key not in expected_keys and not any(
                            (m["worker"], m["topology"], m["run"]) == key for m in rmap):
                        extra.append(f"{topo_dir.name}/{wd.name}/{rd.name}")

    resolved = [r for r in rows if r.get("resolved_by")]
    summary = {
        "batch_id": a.batch, "batch_dir": str(batch_dir),
        "expected": len(man["tasks"]), "done": len(resolved),
        "done_direct": sum(1 for r in resolved if r["resolved_by"] == "direct"),
        "done_via_resume": sum(1 for r in resolved if r["resolved_by"] == "resume"),
        "failed": [f"{r['topology']}/{r['worker']}/{r['run']}" for r in rows
                   if r.get("state") == "failed"],
        "missing_or_incomplete": [f"{r['topology']}/{r['worker']}/{r['run']}" for r in rows
                                  if not r.get("resolved_by")],
        "unexpected_run_dirs": sorted(extra),
        "overall_ok": bool(ok and len(resolved) == len(man["tasks"])),
        "rows": rows,
    }
    if a.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"batch: {a.batch} dir: {batch_dir} (manifest {man['n_tasks']} tasks)")
        for r in rows:
            flag = "OK" if r.get("resolved_by") else "!!"
            print(f"  [{flag}] {r['topology']}/{r['worker']}/{r['run']:<24} "
                  f"state={r.get('state')} resolved={r.get('resolved_by')} "
                  f"sel={r.get('selection_metric')} missing={r.get('missing')}")
        if extra:
            print(f"  unexpected run dirs: {sorted(extra)}")
        print(f"overall_ok={summary['overall_ok']} done={summary['done']}/{summary['expected']} "
              f"(direct={summary['done_direct']}, resume={summary['done_via_resume']})")
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Validate a resume job against the frozen batch manifest (v2).

A resume job is the ONLY YAML allowed to live outside
``remote_jobs/three_topology_v2/``.  It cannot introduce a new model/seed into a
batch, cannot point at a checkpoint from another model/seed/split, and cannot
reuse the original worker directory.  This prevents an arbitrary external YAML
from borrowing the batch_id and bypassing the frozen task list.

Checks
  * batch manifest v2 exists; protocol/split/fold/train_config match;
  * every task is a resume task (resume_from set) for a gradient model;
  * (model, seed) exists in the manifest; exactly one original task matches;
  * resume_from is under outputs/.../first_batch/<batch>/<orig_worker>/<orig_run>/
    checkpoints/{best.pt,last.pt} and belongs to that original task;
  * the new (worker, run) is not a manifest task and its run dir does not exist;
  * checkpoint identity matches model/seed/split/fold/stats/train-val lists,
    model-config SHA and the data hashes of the current DATA_ROOT.

Exit 0 only if all tasks pass.  Writes a JSON record (--out) and a TSV map
(--map-out) used by run_task to record the original run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "qt-three-topology-batch-manifest-v2"


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def sha_text(p: Path) -> str:
    data = p.read_bytes().replace(bytes([13, 10]), bytes([10]))
    return hashlib.sha256(data).hexdigest()


def sha_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def die(msg: str) -> None:
    print(f"[RESUME CHECK ERROR] {msg}", file=sys.stderr)
    sys.exit(2)


def load_job_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _job_parse  # noqa: PLC0415
    return _job_parse


def run_suffix(model: str, seed: str, kind: str) -> str:
    return model if kind == "ml" else f"{model}_seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", ""))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map-out", default=None)
    a = ap.parse_args()
    root = Path(native(a.project_root or Path(__file__).resolve().parents[2]))
    data_root = Path(native(a.data_root))
    batch = a.batch_id
    job = Path(native(a.job))
    jp = load_job_module()
    spec = jp.load(job)

    if spec["batch_id"] != batch:
        die(f"job batch_id {spec['batch_id']!r} != requested {batch!r}")

    man_path = root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch}.json"
    if not man_path.is_file():
        die(f"frozen manifest missing: {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("schema") != SCHEMA:
        die(f"manifest schema {man.get('schema')!r} is not v2; regenerate it explicitly")

    for key in ("protocol", "split_id", "fold_id", "outer_fold", "inner_id", "train_config"):
        if spec.get(key) != man.get(key):
            die(f"{key} differs from frozen batch manifest: {spec.get(key)!r} != {man.get(key)!r}")

    orig_by_key = {}
    for t in man["tasks"]:
        if t["kind"] != "gradient":
            continue
        orig_by_key.setdefault((t["model"], str(t["seed"])), []).append(t)
    workers = {t["worker"] for t in man["tasks"]}
    identities = {(t["worker"], t["run"]) for t in man["tasks"]}

    batch_dir = (root / "outputs" / "three_topology_v2" / "first_batch" / batch).resolve()
    rows, problems, maps = [], [], []
    if not spec["tasks"]:
        die("resume job has no tasks")
    for i, t in enumerate(spec["tasks"]):
        tag = f"tasks[{i}]"
        if t["device"] != "gpu":
            die(f"{tag}: resume requires device=gpu")
        if t["resume_from"] in (None, "-"):
            die(f"{tag}: external job YAML contains a non-resume task; normal jobs must be "
                f"registered under remote_jobs/three_topology_v2 and match the frozen manifest")
        key = (t["model"], str(t["seed"]))
        if key not in orig_by_key or len(orig_by_key[key]) != 1:
            die(f"{tag}: ({t['model']}, seed {t['seed']}) does not identify exactly one "
                f"original task in batch {batch}")
        orig = orig_by_key[key][0]
        kind = orig["kind"]
        run = run_suffix(t["model"], t["seed"], kind)
        if (t["worker"], run) in identities:
            die(f"{tag}: target run (worker={t['worker']}, run={run}) collides with a frozen task")
        if t["worker"] in workers:
            die(f"{tag}: worker {t['worker']!r} is an original worker; use a new worker "
                f"directory so the failed run stays read-only")
        target_dir = batch_dir / t["worker"] / run
        if target_dir.exists():
            die(f"{tag}: target run dir already exists: {target_dir}")

        ckpt = Path(native(t["resume_from"]))
        if not ckpt.is_absolute():
            ckpt = (root / ckpt).resolve()
        else:
            ckpt = ckpt.resolve()
        try:
            rel = ckpt.relative_to(batch_dir)
        except ValueError:
            die(f"{tag}: resume_from is outside the batch output dir {batch_dir}: {ckpt}")
        parts = rel.parts
        if len(parts) != 4 or parts[2] != "checkpoints" or parts[3] not in ("best.pt", "last.pt"):
            die(f"{tag}: resume_from must be <orig_worker>/<orig_run>/checkpoints/(best|last).pt, "
                f"got {rel.as_posix()}")
        if parts[0] != orig["worker"] or parts[1] != orig["run"]:
            die(f"{tag}: resume_from belongs to {parts[0]}/{parts[1]}, not the original "
                f"{orig['worker']}/{orig['run']}")
        if not ckpt.is_file():
            die(f"{tag}: checkpoint not found: {ckpt}")

        # checkpoint identity vs frozen manifest + current data
        try:
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            die(f"{tag}: cannot load checkpoint {ckpt}: {exc!r}")
        ident = ck.get("identity") or {}
        expected = {
            "model_name": t["model"], "seed": int(t["seed"]),
            "split_id": man["split_id"], "fold_id": man["fold_id"],
            "outer_fold": man["outer_fold"] or None, "inner_fold": man["inner_id"] or None,
            "stats_sha256": man["input_identity"]["stats_sha256"],
            "train_sample_list_sha256": man["input_identity"]["train_sample_list_sha256"],
            "val_sample_list_sha256": man["input_identity"]["val_sample_list_sha256"],
            "model_config_sha256": orig["model_config_sha256"],
        }
        diffs = {k: {"checkpoint": ident.get(k), "expected": v}
                 for k, v in expected.items() if ident.get(k) != v}
        idx_dir = data_root / "data" / "processed" / "three_topology_graph_v1"
        data_expected = {"samples_jsonl_sha256": sha_bytes(idx_dir / "samples.jsonl"),
                         "dataset_manifest_sha256": sha_bytes(idx_dir / "dataset_manifest.json")}
        diffs.update({k: {"checkpoint": ident.get(k), "expected": v}
                      for k, v in data_expected.items() if ident.get(k) != v})
        if diffs:
            problems.append({"task": tag, "diffs": diffs})
            continue
        maps.append({"worker": t["worker"], "run": run, "model": t["model"],
                     "seed": str(t["seed"]), "orig_worker": orig["worker"],
                     "orig_run": orig["run"], "checkpoint": str(ckpt),
                     "checkpoint_sha256": sha_bytes(ckpt)})
        rows.append({"task": tag, "worker": t["worker"], "run": run,
                     "original": f"{orig['worker']}/{orig['run']}",
                     "checkpoint": str(ckpt), "checkpoint_sha256": sha_bytes(ckpt),
                     "identity_ok": True})

    if problems:
        print(json.dumps({"ok": False, "problems": problems}, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 1

    rec = {"schema": "qt-resume-check-v1", "batch_id": batch, "job": str(job),
           "manifest_sha256": sha_text(man_path), "ok": True,
           "n_resume_tasks": len(rows), "tasks": rows}
    if a.out:
        op = Path(native(a.out)); op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    if a.map_out:
        mp = Path(native(a.map_out)); mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8", newline="\n") as f:
            for m in maps:
                f.write("\t".join([m["worker"], m["run"], m["model"], m["seed"],
                                   m["orig_worker"], m["orig_run"], m["checkpoint"],
                                   m["checkpoint_sha256"]]) + "\n")
    print(json.dumps({"ok": True, "n_resume_tasks": len(rows),
                      "original_runs": [f"{r['original']}" for r in rows]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

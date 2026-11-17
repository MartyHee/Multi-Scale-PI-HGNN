#!/usr/bin/env python
"""Freeze / read the execution identity of ONE batch (manifest v2).

Every job YAML of a batch carries the same ``batch_id``.  This tool scans the
job-YAML directory, resolves the *effective* execution parameters (top-level
overrides + task overrides + train-config defaults), and writes a frozen
manifest under ``remote_jobs/three_topology_v2/manifests/<batch_id>.json``.

Manifest v2 records, per batch and per task:
  protocol / split / fold / outer / inner, train_config and its SHA256,
  split/stats/train/val identity hashes, and for every task:
  worker, run, model, seed, device, kind, job file (POSIX), epochs,
  max_batches, resume_from and the model-config SHA256.
Paths are stored as POSIX repo-relative strings on every OS; hashes are computed
over LF-normalised text so a Windows checkout and a Linux checkout agree.

Migration: v1 manifests are rejected with an explicit "regenerate" message.
The historical first batch keeps its 8 tasks and protocol, but must be
regenerated under v2 before it can be used by the stricter launcher.

Modes
  --job <yaml> --write       regenerate manifest from all matching YAMLs
  --job <yaml> --verify      compare current YAMLs AND effective params
  --job <yaml> --paths       worker/run/model/seed/device/kind (6-column TSV)
  --job <yaml> --effective   9-column TSV incl. epochs/max_batches/resume_from
  --job <yaml> --shell       bash-evalable batch header
  --job <yaml> --json        manifest JSON
  --job <yaml> --sha256      manifest sha256 (LF-normalised)
  --batch <id> ...           same, by batch_id
  --expect N                 with --write/--verify: require exactly N tasks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SCHEMA = "qt-three-topology-batch-manifest-v2"
MIN_SCHEMA = SCHEMA
ML_MODELS = {"ridge", "hist_gb"}


def native(s):
    """Map an MSYS path (/d/x) to a Windows path (D:/x) on Windows; else as-is."""
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


MODEL_ROOT = Path(native(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[2]))
JOBS_DIR = MODEL_ROOT / "remote_jobs" / "three_topology_v2"
MANIFEST_DIR = JOBS_DIR / "manifests"
TRAIN_ROOT = MODEL_ROOT / "processed" / "three_topology_training_v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(p: Path) -> str:
    """Hash text content with CRLF normalised to LF (OS/Git independent)."""
    return sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def sha256_file(p: Path) -> str:
    """Raw byte hash for non-Git derived/data files (must match engine identity)."""
    return sha256_bytes(p.read_bytes())


def posix_rel(path: Path) -> str:
    """Repo-relative POSIX path (forward slashes) regardless of OS."""
    return path.resolve().relative_to(MODEL_ROOT.resolve()).as_posix()


def die(msg: str) -> None:
    print(f"[BATCH MANIFEST ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_job_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _job_parse  # noqa: PLC0415
    return _job_parse


def load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_suffix(model: str, seed: str, kind: str) -> str:
    return model if kind == "ml" else f"{model}_seed{seed}"


def hash_json(obj) -> str:
    """Same rule as engine.build_identity (sorted keys, UTF-8, no separators arg)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_input_identity(protocol: str, split_id: str, fold_id: str,
                           outer_fold: str, inner_id: str) -> dict:
    split_file = TRAIN_ROOT / "splits" / f"split_{split_id}.json"
    if protocol == "split":
        if not split_file.is_file():
            die(f"split file not found: {split_file}")
        sp = json.loads(split_file.read_text(encoding="utf-8"))
        fold = next((f for f in sp.get("folds", []) if f["fold_id"] == fold_id), None)
        if fold is None:
            die(f"fold {fold_id!r} not found in {split_file}")
        train_ids = list(fold["train_sample_ids"]); val_ids = list(fold["validation_sample_ids"])
        stats_path = TRAIN_ROOT / "stats" / split_id / f"{fold_id}.json"
        identity_file = split_file
    elif protocol == "inner":
        inner_file = TRAIN_ROOT / "manifests" / "inner_proxy_folds.json"
        if not inner_file.is_file():
            die(f"inner proxy manifest not found: {inner_file}")
        man = json.loads(inner_file.read_text(encoding="utf-8"))
        outer = next((f for f in man.get("folds", []) if f["outer_fold_id"] == outer_fold), None)
        inner = next((i for i in (outer or {}).get("inner_folds", [])
                      if i["inner_fold_id"] == inner_id), None)
        if inner is None:
            die(f"inner fold {outer_fold}/{inner_id} not found in {inner_file}")
        train_ids = list(inner["train_sample_ids"]); val_ids = list(inner["validation_sample_ids"])
        stats_path = TRAIN_ROOT.parent / inner["inner_stats_path"]
        identity_file = inner_file
    else:
        die(f"unknown protocol {protocol!r}")
    if not stats_path.is_file():
        die(f"stats file not found: {stats_path}")
    return {
        "split_source": posix_rel(identity_file),
        "split_sha256": sha256_file(identity_file),
        "stats_source": posix_rel(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "train_sample_list_sha256": hash_json(sorted(train_ids)),
        "val_sample_list_sha256": hash_json(sorted(val_ids)),
    }


def scan(batch_id: str) -> dict:
    jp = load_job_module()
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.yaml")) + sorted(JOBS_DIR.glob("*.yml")):
        spec = jp.load(p)
        if spec["batch_id"] != batch_id:
            continue
        jobs.append((p, spec))
    if not jobs:
        die(f"no job YAML with batch_id={batch_id!r} under {JOBS_DIR}")

    protocols = {(s["protocol"], s["split_id"], s["fold_id"], s["outer_fold"], s["inner_id"])
                 for _, s in jobs}
    if len(protocols) != 1:
        die(f"batch {batch_id}: job YAMLs disagree on protocol/split/fold: {sorted(protocols)}")
    configs = {s["train_config"] for _, s in jobs}
    if len(configs) != 1:
        die(f"batch {batch_id}: job YAMLs disagree on train_config: {sorted(configs)}")
    protocol, split_id, fold_id, outer_fold, inner_id = next(iter(protocols))
    train_config = next(iter(configs))
    tc_path = MODEL_ROOT / train_config
    if not tc_path.is_file():
        die(f"train_config not found: {tc_path}")
    cfg = load_yaml(tc_path)
    training = (cfg or {}).get("training") or {}
    if "epochs" not in training:
        die(f"train_config has no training.epochs: {train_config}")
    default_epochs = int(training["epochs"])
    default_max_batches = training.get("max_train_batches_per_epoch")
    model_cfgs = (cfg or {}).get("models") or {}

    input_identity = resolve_input_identity(protocol, split_id, fold_id, outer_fold, inner_id)

    tasks, ident = [], {}
    for p, spec in jobs:
        job_rel = posix_rel(p)
        for t in spec["tasks"]:
            kind = "ml" if t["model"] in ML_MODELS else "gradient"
            run = run_suffix(t["model"], t["seed"], kind)
            key = (t["worker"], run)
            if key in ident:
                die(f"duplicate task identity {key} in {ident[key]} and {job_rel}")
            ident[key] = job_rel
            epochs = t["epochs"] if t.get("epochs") is not None else default_epochs
            max_batches = (t["max_batches"] if t.get("max_batches") is not None
                           else default_max_batches)
            if t["model"] in model_cfgs:
                model_cfg = model_cfgs[t["model"]]
            elif t["model"] in ML_MODELS:
                model_cfg = ((cfg or {}).get("ml") or {}).get(t["model"], {})
            else:
                die(f"model {t['model']!r} has no entry under config.models")
            tasks.append({
                "worker": t["worker"], "run": run, "model": t["model"],
                "seed": t["seed"], "device": t["device"], "kind": kind,
                "job_name": spec["job_name"], "job_file": job_rel,
                "epochs": int(epochs),
                "max_batches": (None if max_batches is None else int(max_batches)),
                "resume_from": (None if t["resume_from"] in (None, "-") else
                                str(t["resume_from"]).replace("\\", "/")),
                "model_config_sha256": hash_json(model_cfg),
            })
    if not tasks:
        die(f"batch {batch_id}: no tasks resolved from job YAMLs")
    order = {"gcn": 0, "graphsage": 1, "gat": 2, "gin": 3, "rgcn": 4, "hgt": 5,
             "mlp": 0, "hgt_global": 1, "ridge": 2, "hist_gb": 3}
    tasks.sort(key=lambda t: (order.get(t["model"], 99), str(t["seed"]), t["worker"]))
    return {
        "schema": SCHEMA, "batch_id": batch_id,
        "protocol": protocol, "split_id": split_id, "fold_id": fold_id,
        "outer_fold": outer_fold, "inner_id": inner_id,
        "train_config": train_config, "train_config_sha256": sha256_text(tc_path),
        "input_identity": input_identity,
        "source_jobs": sorted({t["job_file"] for t in tasks}),
        "n_tasks": len(tasks), "tasks": tasks,
    }


def manifest_path(batch_id: str) -> Path:
    return MANIFEST_DIR / f"{batch_id}.json"


def resolve_batch(job_file: Path) -> str:
    jp = load_job_module()
    return jp.load(job_file)["batch_id"]


def read_manifest(batch_id: str) -> dict:
    p = manifest_path(batch_id)
    if not p.is_file():
        die(f"no frozen manifest for batch {batch_id!r}: {p} (run --write once, then commit it)")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("schema") != MIN_SCHEMA:
        die(f"manifest schema {man.get('schema')!r} is not {MIN_SCHEMA!r}; "
            f"regenerate it explicitly with --write (v1 manifests are not accepted)")
    if man.get("batch_id") != batch_id:
        die(f"manifest batch_id mismatch: {man.get('batch_id')} != {batch_id}")
    if not man.get("n_tasks") or not man.get("tasks"):
        die(f"manifest for {batch_id} has no tasks")
    return man


def _task_tuple(t: dict):
    return (t["worker"], t["run"], t["model"], str(t["seed"]), t["device"], t["kind"],
            t["job_file"], t["epochs"], t["max_batches"], t["resume_from"],
            t["model_config_sha256"])


def main() -> int:
    try:
        sys.stdout.reconfigure(newline="\n")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--effective", action="store_true")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--sha256", action="store_true")
    ap.add_argument("--expect", type=int, default=None)
    a = ap.parse_args()
    if a.job:
        job = Path(a.job)
        if not job.is_file():
            die(f"job file not found: {job}")
        batch = resolve_batch(job)
    elif a.batch:
        batch = a.batch
    else:
        die("provide --job <yaml> or --batch <batch_id>")

    if a.write:
        man = scan(batch)
        if a.expect is not None and man["n_tasks"] != a.expect:
            die(f"expected {a.expect} tasks, found {man['n_tasks']}")
        man["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path(batch).with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(manifest_path(batch))
        print(f"wrote {manifest_path(batch)} ({man['n_tasks']} tasks, "
              f"sha256 {sha256_text(manifest_path(batch))})")

    man = read_manifest(batch)
    if a.expect is not None and man["n_tasks"] != a.expect:
        die(f"manifest has {man['n_tasks']} tasks, expected {a.expect}")

    if a.verify:
        fresh = scan(batch)
        for key in ("protocol", "split_id", "fold_id", "outer_fold", "inner_id",
                    "train_config", "train_config_sha256", "input_identity", "n_tasks"):
            if man.get(key) != fresh.get(key):
                die(f"batch field {key!r} changed since the frozen manifest was written: "
                    f"manifest={man.get(key)!r} current={fresh.get(key)!r}. "
                    f"Re-run --write only after an explicit batch change.")
        cur = {_task_tuple(t) for t in man["tasks"]}
        new = {_task_tuple(t) for t in fresh["tasks"]}
        if cur != new:
            die(f"job YAMLs / effective execution params changed since the frozen manifest; "
                f"missing={sorted(new - cur)} extra={sorted(cur - new)}. "
                f"Re-run --write only after an explicit batch change.")
        print(f"manifest OK for batch {batch}: {man['n_tasks']} tasks, "
              f"train_config sha256 {man['train_config_sha256'][:12]}…, "
              f"protocol={man['protocol']} {man['split_id']}/{man['fold_id']}")

    if a.paths:
        for t in man["tasks"]:
            print("\t".join([t["worker"], t["run"], t["model"], str(t["seed"]),
                             t["device"], t["kind"]]))
    if a.effective:
        for t in man["tasks"]:
            print("\t".join([t["worker"], t["run"], t["model"], str(t["seed"]),
                             t["device"], t["kind"], str(t["epochs"]),
                             "-" if t["max_batches"] is None else str(t["max_batches"]),
                             "-" if t["resume_from"] is None else t["resume_from"]]))
    if a.shell:
        import shlex
        def q(v):
            return shlex.quote("" if v is None else str(v))
        print(f"BATCH_ID={q(man['batch_id'])}")
        print(f"PROTOCOL={q(man['protocol'])}")
        print(f"SPLIT_ID={q(man['split_id'])}")
        print(f"FOLD_ID={q(man['fold_id'])}")
        print(f"OUTER_FOLD={q(man['outer_fold'])}")
        print(f"INNER_ID={q(man['inner_id'])}")
        print(f"TRAIN_CONFIG={q(man['train_config'])}")
        print(f"N_TASKS={man['n_tasks']}")
    if a.sha256:
        print(sha256_text(manifest_path(batch)))
    if a.json:
        print(json.dumps(man, ensure_ascii=False, indent=2))
    if a.show and not (a.json or a.paths or a.sha256 or a.shell or a.effective):
        print(f"manifest: {manifest_path(batch)}")
        print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

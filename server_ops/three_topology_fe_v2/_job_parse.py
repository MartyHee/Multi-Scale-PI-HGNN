#!/usr/bin/env python
"""Parse + validate an FE v2 job YAML (strict).

Modes
  --shell   bash-evalable header assignments
  --tasks   worker, topology, model, seed|'-', device, resume_from|'-',
            epochs|'-', max_batches|'-'  (tab separated)
  --summary human-readable
  default   validated JSON

Unknown fields, missing/invalid values and ambiguous seed usage exit non-zero.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

GRADIENT_MODELS = {"mlp", "hgt_global", "hgt", "rgcn", "gcn", "gat", "graphsage",
                   "gin", "edge_aware_mpnn", "spectral_lift",
                   "hgt_global_off", "hgt_capmatch", "fullfield_mlp", "fullfield_query_mlp",
                   "meshgraphnet", "transolver", "linearno",
                   "meshgraphnet_explore", "transolver_plus_explore", "linearno_explore"}
ML_MODELS = {"ridge", "hist_gb"}
ALL_MODELS = GRADIENT_MODELS | ML_MODELS
TOPOLOGIES = {"topology1", "topology2", "topology3"}

TOP_KEYS = {"job_name", "batch_id", "mode", "train_config", "description",
            "epochs", "max_batches", "tasks"}
TASK_KEYS = {"topology", "model", "seeds", "seed", "device", "worker",
             "resume_from", "epochs", "max_batches"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def die(msg: str) -> None:
    print(f"[JOB YAML ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def project_root() -> Path:
    import os
    env = os.environ.get("PROJECT_ROOT")
    if env:
        m = re.match(r"^/([A-Za-z])/(.*)$", env)
        if os.name == "nt" and m:
            return Path(f"{m.group(1).upper()}:/{m.group(2)}")
        return Path(env)
    return Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    if not path.is_file():
        die(f"job file not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        die(f"invalid YAML: {exc}")
    if not isinstance(doc, dict):
        die("top level must be a mapping")
    unknown = set(doc) - TOP_KEYS
    if unknown:
        die(f"unknown top-level field(s): {sorted(unknown)} (allowed: {sorted(TOP_KEYS)})")
    for k in ("job_name", "batch_id", "train_config", "tasks"):
        if k not in doc:
            die(f"missing required field: {k}")

    job = str(doc["job_name"]); batch = str(doc["batch_id"])
    if not NAME_RE.match(job):
        die(f"job_name must match [A-Za-z0-9_.-]+, got {job!r}")
    if not NAME_RE.match(batch):
        die(f"batch_id must match [A-Za-z0-9_.-]+, got {batch!r}")
    mode = str(doc.get("mode", "formal"))
    if mode != "formal":
        die(f"registered job YAMLs must use mode=formal, got {mode!r}")
    tc = project_root() / str(doc["train_config"])
    if not tc.is_file():
        die(f"train_config not found: {tc}")

    def pos_int(v, name):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            die(f"{name} must be a positive integer, got {v!r}")
        return v

    header_epochs = pos_int(doc["epochs"], "epochs") if "epochs" in doc else None
    header_batches = pos_int(doc["max_batches"], "max_batches") if "max_batches" in doc else None

    tasks = doc["tasks"]
    if not isinstance(tasks, list) or not tasks:
        die("tasks must be a non-empty list")
    norm = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            die(f"tasks[{i}] must be a mapping")
        unk = set(t) - TASK_KEYS
        if unk:
            die(f"tasks[{i}] unknown field(s): {sorted(unk)} (allowed: {sorted(TASK_KEYS)})")
        topology = t.get("topology")
        if topology not in TOPOLOGIES:
            die(f"tasks[{i}].topology invalid: {topology!r}")
        model = t.get("model")
        if model not in ALL_MODELS:
            die(f"tasks[{i}].model invalid: {model!r} (allowed: {sorted(ALL_MODELS)})")
        device = t.get("device")
        if device not in ("gpu", "cpu"):
            die(f"tasks[{i}].device must be 'gpu' or 'cpu', got {device!r}")
        worker = t.get("worker", job)
        if not NAME_RE.match(str(worker)):
            die(f"tasks[{i}].worker must match [A-Za-z0-9_.-]+, got {worker!r}")
        resume_from = t.get("resume_from")
        if resume_from is not None and device != "gpu":
            die(f"tasks[{i}].resume_from requires device=gpu")

        seeds = t.get("seeds")
        seed = t.get("seed")
        if model in GRADIENT_MODELS:
            if seed is not None:
                die(f"tasks[{i}] ({model}) uses 'seeds' (a list), not 'seed'")
            if not isinstance(seeds, list) or not seeds:
                die(f"tasks[{i}] ({model}) requires a non-empty seeds list")
            for s in seeds:
                if not isinstance(s, int) or isinstance(s, bool) or s <= 0:
                    die(f"tasks[{i}].seeds must be positive integers, got {s!r}")
            seed_list = list(seeds)
        elif model == "hist_gb":
            if seeds is not None:
                die(f"tasks[{i}] (hist_gb) uses one explicit 'seed', not 'seeds'")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
                die(f"tasks[{i}] (hist_gb) requires a positive explicit seed")
            seed_list = [seed]
        else:  # ridge
            if seeds is not None or seed is not None:
                die(f"tasks[{i}] (ridge) is deterministic; do not set seeds")
            seed_list = [None]
        if device == "gpu" and model in ML_MODELS:
            die(f"tasks[{i}] ({model}) must use device=cpu")
        if device == "cpu" and model in GRADIENT_MODELS:
            die(f"tasks[{i}] ({model}) must use device=gpu")

        te = pos_int(t["epochs"], f"tasks[{i}].epochs") if "epochs" in t else None
        tb = pos_int(t["max_batches"], f"tasks[{i}].max_batches") if "max_batches" in t else None
        for s in seed_list:
            norm.append({"worker": str(worker), "topology": topology, "model": model,
                         "seed": ("-" if s is None else str(s)), "device": device,
                         "resume_from": (str(resume_from) if resume_from else "-"),
                         "epochs": (te if te is not None else header_epochs),
                         "max_batches": (tb if tb is not None else header_batches)})
    return {"job_name": job, "batch_id": batch, "mode": mode,
            "train_config": str(doc["train_config"]),
            "epochs": header_epochs, "max_batches": header_batches,
            "description": doc.get("description", ""), "tasks": norm}


def main() -> int:
    try:
        sys.stdout.reconfigure(newline="\n")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("job_file")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    spec = load(Path(a.job_file))
    if a.batch:
        print(spec["batch_id"])
        return 0
    if a.shell:
        def q(v):
            return shlex.quote("" if v is None else str(v))
        print(f"JOB_NAME={q(spec['job_name'])}")
        print(f"BATCH_ID={q(spec['batch_id'])}")
        print(f"JOB_MODE={q(spec['mode'])}")
        print(f"TRAIN_CONFIG={q(spec['train_config'])}")
        print(f"JOB_EPOCHS={q('' if spec['epochs'] is None else spec['epochs'])}")
        print(f"JOB_MAX_BATCHES={q('' if spec['max_batches'] is None else spec['max_batches'])}")
        return 0
    if a.tasks:
        for t in spec["tasks"]:
            print("\t".join([t["worker"], t["topology"], t["model"], t["seed"], t["device"],
                             t["resume_from"],
                             "-" if t["epochs"] is None else str(t["epochs"]),
                             "-" if t["max_batches"] is None else str(t["max_batches"])]))
        return 0
    if a.summary:
        print(f"job={spec['job_name']} batch={spec['batch_id']} mode={spec['mode']} "
              f"runs={len(spec['tasks'])}")
        for t in spec["tasks"]:
            print(f"  worker={t['worker']:<18} topology={t['topology']:<10} "
                  f"model={t['model']:<14} seed={t['seed']:<10} device={t['device']} "
                  f"resume_from={t['resume_from']}")
        return 0
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

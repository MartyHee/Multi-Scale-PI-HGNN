#!/usr/bin/env python
"""Freeze / verify the execution identity of one FE v2 batch.

The manifest is the single source of truth for preflight, launch, monitoring,
summary and collection.  It records every effective execution parameter and
the code/data/schema/protocol/ledger/stats hashes, so changing any effective
input invalidates an older preflight PASS (not just the YAML filename).

Modes
  --job <yaml> --write [--expect N]   regenerate from all matching job YAMLs
  --job <yaml> --verify               compare current jobs/effective params
  --job <yaml> --paths                6-col TSV (worker/run/topology/model/seed/kind)
  --job <yaml> --effective            10-col TSV incl epochs/max_batches/resume
  --job <yaml> --shell                bash-evalable batch header
  --job <yaml> --json
  --job <yaml> --sha256
  --batch <id> ...                    same, by batch_id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "qt-fe-v2-batch-manifest-v1"
GRADIENT_MODELS = {"mlp", "hgt_global", "hgt", "rgcn", "gcn", "gat", "graphsage",
                   "gin", "edge_aware_mpnn", "spectral_lift",
                   "hgt_global_off", "hgt_capmatch", "fullfield_mlp", "fullfield_query_mlp",
                   "meshgraphnet", "transolver", "linearno",
                   "meshgraphnet_explore", "transolver_plus_explore", "linearno_explore"}
ML_MODELS = {"ridge", "hist_gb"}
TOPO_ORDER = {"topology1": 0, "topology2": 1, "topology3": 2}
MODEL_ORDER = {"mlp": 0, "hgt_global": 1, "hgt": 2, "rgcn": 3, "gcn": 4, "gat": 5,
               "graphsage": 6, "gin": 7, "ridge": 8, "hist_gb": 9,
               "fullfield_mlp": 10, "hgt_capmatch": 11, "hgt_global_off": 12,
               "meshgraphnet": 13, "transolver": 14, "linearno": 15,
               "meshgraphnet_explore": 17, "transolver_plus_explore": 18, "linearno_explore": 19,
               "fullfield_query_mlp": 16}


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path(native(os.environ.get("PROJECT_ROOT") or HERE.parents[1]))
JOBS_DIR = Path(native(os.environ.get("THREE_TOPO_JOBS_ROOT") or (MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2")))
MANIFEST_DIR = Path(native(os.environ.get("THREE_TOPO_MANIFEST_ROOT") or (JOBS_DIR / "manifests")))
PROCESSED_ROOT = Path(native(os.environ.get("THREE_TOPO_PROCESSED_ROOT")
                             or (MODEL_ROOT / "processed" / "three_topology_fe_v2_9216")))
PACKAGE_ROOT = Path(native(os.environ.get("THREE_TOPO_PACKAGE_ROOT")
                           or (MODEL_ROOT.parent / "raw_data" / "three_topology_fe_v2_9216")))
CONFIG_ROOT = MODEL_ROOT / "configs" / "three_topology_fe_v2"
sys.path.insert(0, str(MODEL_ROOT))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_lf(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def sha_raw(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def hash_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode())


def posix_rel(path: Path) -> str:
    return path.resolve().relative_to(MODEL_ROOT.resolve()).as_posix()


def die(msg: str) -> None:
    print(f"[BATCH MANIFEST ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_job_module():
    sys.path.insert(0, str(HERE))
    import _job_parse  # noqa: PLC0415
    return _job_parse


def load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_suffix(model: str, seed: str, kind: str) -> str:
    return model if model == "ridge" else f"{model}_seed{seed}"


def code_hash() -> str:
    """LF-normalised content hash over the source/config trees that matter."""
    roots = [MODEL_ROOT / "src" / "data" / "three_topology_fe_v2",
             MODEL_ROOT / "src" / "train" / "three_topology_v2",
             MODEL_ROOT / "src" / "models" / "baselines",
             MODEL_ROOT / "src" / "models" / "strong_baselines",
             CONFIG_ROOT, HERE]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            if p.suffix in (".md", ".txt"):
                continue
            if p.name in ("server_config.sh",):
                continue
            files.append(p)
    files.append(MODEL_ROOT / "scripts" / "three_topology_fe_v2" / "train_fe_v2.py")
    rows = []
    for p in sorted(set(files), key=lambda x: posix_rel(x)):
        rows.append(f"{posix_rel(p)}\0{sha_lf(p)}\n")
    return sha256_bytes("".join(rows).encode())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(MODEL_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _protocol_counts() -> dict:
    path = PROCESSED_ROOT / "splits" / "protocol_v1.json"
    if not path.is_file():
        die(f"protocol_v1.json not found: {path}")
    prot = json.loads(path.read_text(encoding="utf-8"))
    # The index is the authoritative topology mapping; roles are cell-level.
    idx_rows = [json.loads(line) for line in
                (PACKAGE_ROOT / "samples.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    by_topo = {}
    for r in idx_rows:
        by_topo.setdefault(r["topology_token"], []).append(r["sample_id"])
    counts = {}
    for topo, sids in by_topo.items():
        counts[topo] = {p: {role: sum(1 for s in sids if prot["roles"][s][p] == role)
                            for role in ("train", "val", "test", "blocked")}
                        for p in "ABC"}
        if not (counts[topo]["A"]["train"] == counts[topo]["B"]["train"]
                == counts[topo]["C"]["train"]):
            die(f"{topo}: A/B/C training cores differ")
    return counts


def _input_identity() -> dict:
    prot = PROCESSED_ROOT / "splits" / "protocol_v1.json"
    ledger = PROCESSED_ROOT / "splits" / "role_ledger_v1.csv"
    split_man = PROCESSED_ROOT / "splits" / "split_manifest.json"
    stats = {}
    for topo in ("topology1", "topology2", "topology3"):
        p = PROCESSED_ROOT / "stats" / f"{topo}.json"
        if not p.is_file():
            die(f"stats missing: {p}")
        stats[topo] = sha_raw(p)
    for p in (prot, ledger, split_man):
        if not p.is_file():
            die(f"frozen split input missing: {p}")
    design, design_meta, geometry, loads = {}, {}, {}, {}
    for topo in ("topology1", "topology2", "topology3"):
        paths = {
            "design": PROCESSED_ROOT / "shared" / f"{topo}_design_members.npz",
            "design_meta": PROCESSED_ROOT / "shared" / f"{topo}_design_cache.json",
            "geometry": PROCESSED_ROOT / "shared" / f"{topo}_geometry.npz",
            "loads": PROCESSED_ROOT / "fields" / f"{topo}_loads.npz",
        }
        for k, p in paths.items():
            if not p.is_file():
                die(f"shared {k} cache missing: {p}; run build_design_cache.py")
        design[topo] = sha_raw(paths["design"])
        design_meta[topo] = sha_raw(paths["design_meta"])
        geometry[topo] = sha_raw(paths["geometry"])
        loads[topo] = sha_raw(paths["loads"])
    return {
        "samples_jsonl_sha256": sha_raw(PACKAGE_ROOT / "samples.jsonl"),
        "dataset_manifest_sha256": sha_raw(PACKAGE_ROOT / "dataset_manifest.json"),
        "feature_schema_sha256": sha_lf(CONFIG_ROOT / "feature_schema.json"),
        "protocol_sha256": sha_raw(prot),
        "role_ledger_sha256": sha_raw(ledger),
        "split_manifest_sha256": sha_raw(split_man),
        "stats_sha256": stats,
        "read_path": "shared_v1",
        "design_cache_sha256": design,
        "design_cache_meta_sha256": design_meta,
        "shared_geometry_sha256": geometry,
        "field_loads_sha256": loads,
        "role_counts": _protocol_counts(),
        "input_dims": {"mesh_node": 15, "beam_element": 18, "plate_element": 9,
                       "mlp_force_input": 33, "ml_table_beam_input": 33},
    }


def _model_cfg(model: str, cfg: dict, seed: str, topology: str) -> dict:
    if model in GRADIENT_MODELS:
        import importlib
        schema = importlib.import_module("src.data.three_topology_fe_v2.schema")
        return schema.task_model_cfg(model, cfg, topology, int(seed))
    mc = dict((cfg.get("ml") or {})[model])
    if model == "hist_gb":
        mc["seed"] = int(seed)
    return mc


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

    configs = {s["train_config"] for _, s in jobs}
    if len(configs) != 1:
        die(f"batch {batch_id}: job YAMLs disagree on train_config: {sorted(configs)}")
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

    tasks, ident = [], {}
    for p, spec in jobs:
        job_rel = posix_rel(p)
        job_sha = sha_lf(p)
        for t in spec["tasks"]:
            kind = "ml" if t["model"] in ML_MODELS else "gradient"
            run = run_suffix(t["model"], t["seed"], kind)
            key = (t["worker"], t["topology"], run)
            if key in ident:
                die(f"duplicate task identity {key} in {ident[key]} and {job_rel}")
            ident[key] = job_rel
            epochs = t["epochs"] if t.get("epochs") is not None else default_epochs
            max_batches = (t["max_batches"] if t.get("max_batches") is not None
                           else default_max_batches)
            tasks.append({
                "worker": t["worker"], "topology": t["topology"], "run": run,
                "model": t["model"], "seed": t["seed"], "device": t["device"],
                "kind": kind, "job_name": spec["job_name"], "job_file": job_rel,
                "job_file_sha256": job_sha,
                "epochs": int(epochs),
                "max_batches": (None if max_batches is None else int(max_batches)),
                "resume_from": (None if t["resume_from"] in (None, "-") else
                                str(t["resume_from"]).replace("\\", "/")),
                "model_config_sha256": hash_json(_model_cfg(t["model"], cfg, t["seed"], t["topology"])),
            })
    if not tasks:
        die(f"batch {batch_id}: no tasks resolved from job YAMLs")
    tasks.sort(key=lambda t: (TOPO_ORDER.get(t["topology"], 9),
                              MODEL_ORDER.get(t["model"], 99),
                              str(t["seed"]), t["worker"]))
    return {
        "schema": SCHEMA, "batch_id": batch_id,
        "train_config": train_config, "train_config_sha256": sha_lf(tc_path),
        "training_effective": {
            "epochs": default_epochs,
            "batch_size": int(training["batch_size"]),
            "val_batch_size": int(training.get("val_batch_size", training["batch_size"])),
            "lr": float(training["lr"]),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "early_stopping_patience": training.get("early_stopping_patience"),
            "min_delta": float(training.get("min_delta", 0.0)),
            "grad_clip": training.get("grad_clip"),
            "max_train_batches_per_epoch": default_max_batches,
            "selection_metric": training.get("selection_metric"),
            "physics_loss_weight": float(training.get("physics_loss_weight", 0.0)),
        },
        "allow_truncation": any(t["max_batches"] is not None for t in tasks),
        "formal_training_authorized": bool((cfg or {}).get("formal_training_authorized", False)),
        "input_identity": _input_identity(),
        "code_sha256": code_hash(), "git_commit": git_commit(),
        "source_jobs": sorted({t["job_file"] for t in tasks}),
        "n_tasks": len(tasks), "tasks": tasks,
    }


def manifest_path(batch_id: str) -> Path:
    return MANIFEST_DIR / f"{batch_id}.json"


def resolve_batch(job_file: Path) -> str:
    return load_job_module().load(job_file)["batch_id"]


def read_manifest(batch_id: str) -> dict:
    p = manifest_path(batch_id)
    if not p.is_file():
        die(f"no frozen manifest for batch {batch_id!r}: {p} (run --write once, then commit it)")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("schema") != SCHEMA:
        die(f"manifest schema {man.get('schema')!r} is not {SCHEMA!r}; regenerate with --write")
    if man.get("batch_id") != batch_id:
        die(f"manifest batch_id mismatch: {man.get('batch_id')} != {batch_id}")
    if not man.get("n_tasks") or not man.get("tasks"):
        die(f"manifest for {batch_id} has no tasks")
    return man


def _task_tuple(t: dict):
    return (t["worker"], t["topology"], t["run"], t["model"], str(t["seed"]),
            t["device"], t["kind"], t["job_file"], t["job_file_sha256"],
            t["epochs"], t["max_batches"], t["resume_from"],
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
              f"sha256 {sha_raw(manifest_path(batch))})")

    man = read_manifest(batch)
    if a.expect is not None and man["n_tasks"] != a.expect:
        die(f"manifest has {man['n_tasks']} tasks, expected {a.expect}")

    if a.verify:
        fresh = scan(batch)
        for key in ("train_config", "train_config_sha256", "training_effective",
                    "allow_truncation", "input_identity", "code_sha256", "n_tasks"):
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
              f"code sha256 {man['code_sha256'][:12]}…")

    if a.paths:
        for t in man["tasks"]:
            print("\t".join([t["worker"], t["run"], t["topology"], t["model"],
                             str(t["seed"]), t["kind"]]))
    if a.effective:
        for t in man["tasks"]:
            print("\t".join([t["worker"], t["run"], t["topology"], t["model"],
                             str(t["seed"]), t["device"], t["kind"], str(t["epochs"]),
                             "-" if t["max_batches"] is None else str(t["max_batches"]),
                             "-" if t["resume_from"] is None else t["resume_from"]]))
    if a.shell:
        import shlex
        def q(v):
            return shlex.quote("" if v is None else str(v))
        print(f"BATCH_ID={q(man['batch_id'])}")
        print(f"TRAIN_CONFIG={q(man['train_config'])}")
        print(f"N_TASKS={man['n_tasks']}")
        print(f"ALLOW_TRUNCATION={1 if man['allow_truncation'] else 0}")
        print(f"MANIFEST_SHA256={q(sha_raw(manifest_path(batch)))}")
    if a.sha256:
        print(sha_raw(manifest_path(batch)))
    if a.json:
        print(json.dumps(man, ensure_ascii=False, indent=2))
    if a.show and not (a.json or a.paths or a.sha256 or a.shell or a.effective):
        print(f"manifest: {manifest_path(batch)}")
        print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

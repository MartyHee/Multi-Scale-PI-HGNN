#!/usr/bin/env python
"""FE v2 training entry: smoke / preflight / formal.

Formal mode is the only server-run mode.  It is explicit and cannot be reached
by an accidental flag: it requires a frozen batch manifest task, a matching
preflight PASS bound to the current code/data/schema/protocol/stats/config, an
atomic new attempt directory, and no train-batch truncation.  The direct Python
entry enforces the same checks as the shell wrapper.

Modes
  --mode smoke      local capped run; no manifest/preflight; explicit --out
  --mode preflight  capped run used by preflight.sh; writes under the preflight
                    root, not a formal identity
  --mode formal     frozen task only; run directory derived from
                    outputs/.../formal/<batch>/<topology>/<worker>/<run>/attempt_NN

Gradient and ML models share one evaluation implementation: the ML models are
wrapped in :class:`MLBatchPredictor` and evaluated by the same
``MultiSliceEvaluator`` / ``engine.validate`` code (three slices, equal-mean
selection, graph-equal reduction, inverse transform, per-slice export).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from functools import partial
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
SERVER_OPS = MODEL_ROOT / "server_ops" / "three_topology_fe_v2"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(SERVER_OPS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator, SliceBundle  # noqa: E402
from src.data.three_topology_fe_v2.ml_adapter import MLBatchPredictor, collect  # noqa: E402
from src.data.three_topology_fe_v2.shared_cache import SharedCache  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

TEMPLATES = ("topology1", "topology2", "topology3")
GRADIENT = set(registry.GRADIENT_MODELS)
ML = set(registry.NON_GRADIENT_MODELS)
PROTOCOL_ID = "protocol_v1"
SCHEMA_ID = "qt-fe-v2-run-summary-v1"


class Refused(RuntimeError):
    def __init__(self, message: str, code: int = 3):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Hashing / atomic files
# ---------------------------------------------------------------------------
def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _run_dir(batch: str, topology: str, worker: str, run: str, attempt: int) -> Path:
    root = Path(os.environ.get("THREE_TOPO_FORMAL_OUTPUT_ROOT")
                or (paths.OUTPUT_ROOT / "formal"))
    return root / batch / topology / worker / run / f"attempt_{attempt:02d}"


def _atomic_new_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise Refused(f"run directory already exists; duplicate start refused: {path}", 3) from exc


def _status(path: Path, state: str, **extra) -> None:
    rec = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update(extra)
    write_json_atomic(path / "STATUS.json", rec)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ---------------------------------------------------------------------------
# Manifest / preflight identity
# ---------------------------------------------------------------------------
def manifest_root() -> Path:
    return Path(os.environ.get("THREE_TOPO_MANIFEST_ROOT")
                or (MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2" / "manifests"))


def preflight_root() -> Path:
    return Path(os.environ.get("THREE_TOPO_PREFLIGHT_ROOT")
                or (paths.OUTPUT_ROOT / "preflight"))


def load_manifest(batch: str) -> dict:
    path = manifest_root() / f"{batch}.json"
    if not path.is_file():
        raise Refused(f"frozen batch manifest missing: {path}", 4)
    man = json.loads(path.read_text(encoding="utf-8"))
    if man.get("schema") != "qt-fe-v2-batch-manifest-v1":
        raise Refused(f"manifest schema {man.get('schema')!r} is not qt-fe-v2-batch-manifest-v1", 4)
    if man.get("batch_id") != batch:
        raise Refused(f"manifest batch_id {man.get('batch_id')!r} != {batch!r}", 4)
    return man


def resolve_task(man: dict, worker: str, topology: str, model: str, seed) -> dict:
    matches = [t for t in man["tasks"]
               if t["worker"] == worker and t["topology"] == topology
               and t["model"] == model and str(t["seed"]) == str(seed)]
    if len(matches) != 1:
        raise Refused(
            f"task identity not uniquely frozen: worker={worker} topology={topology} "
            f"model={model} seed={seed}; matches={len(matches)}", 4)
    return matches[0]


def _check_formal_preflight(man: dict, task: dict, cfg_path: Path, cfg: dict) -> dict:
    import _preflight_gate as gate  # noqa: PLC0415
    try:
        rec = gate.check_pass(MODEL_ROOT, man["batch_id"])
    except gate.GateError as exc:
        raise Refused(str(exc), 4) from exc
    if rec.get("result") != "PASS":
        raise Refused("preflight record is not PASS", 4)
    checked = {tuple(x) for x in rec.get("checked", [])}
    if (task["topology"], task["model"]) not in checked:
        raise Refused(
            f"preflight did not exercise topology/model {task['topology']}/{task['model']}", 4)
    if man["train_config_sha256"] != sha_lf(cfg_path):
        raise Refused("train config changed since the frozen manifest", 4)
    tr = cfg["training"]
    if int(task["epochs"]) != int(tr["epochs"]):
        raise Refused(f"epochs {tr['epochs']} != frozen {task['epochs']}", 4)
    if not man.get("allow_truncation", False) and tr.get("max_train_batches_per_epoch") is not None:
        raise Refused("formal mode refuses train-batch truncation "
                      "(max_train_batches_per_epoch must be null)", 4)
    if task["max_batches"] is not None and not man.get("allow_truncation", False):
        raise Refused("formal mode refuses a frozen max_batches value", 4)
    return rec


def _expected_model_cfg(task: dict, cfg: dict) -> dict:
    model = task["model"]
    if model in GRADIENT:
        return schema.task_model_cfg(model, cfg, task["topology"], int(task["seed"]))
    mc = dict(cfg["ml"][model])
    if model == "hist_gb":
        mc["seed"] = int(task["seed"])
    return mc


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _protocol() -> dict:
    path = paths.PROCESSED_ROOT / "splits" / f"{PROTOCOL_ID}.json"
    if not path.is_file():
        raise Refused(f"frozen protocol missing: {path}", 4)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(topology: str) -> dict:
    return stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{topology}.json")


def _shared_enabled() -> bool:
    return os.environ.get("THREE_TOPO_USE_SHARED", "1") != "0"


def _shared_cache(idx, topology: str) -> SharedCache | None:
    return SharedCache(idx, topology, verify=True) if _shared_enabled() else None


def _cache_hashes(topology: str) -> dict:
    if not _shared_enabled():
        return {"read_path": "legacy_extraction"}
    base = paths.PROCESSED_ROOT
    design = base / "shared" / f"{topology}_design_members.npz"
    design_meta = base / "shared" / f"{topology}_design_cache.json"
    geom = base / "shared" / f"{topology}_geometry.npz"
    loads = base / "fields" / f"{topology}_loads.npz"
    for p in (design, design_meta, geom, loads):
        if not p.is_file():
            raise Refused(f"shared cache missing: {p}", 4)
    return {
        "read_path": "shared_v1",
        "design_cache_sha256": sha_raw(design),
        "design_cache_meta_sha256": sha_raw(design_meta),
        "shared_geometry_sha256": sha_raw(geom),
        "field_loads_sha256": sha_raw(loads),
    }


def _identity_extra(topology: str, model_name: str, seed: int) -> dict:
    split_dir = paths.PROCESSED_ROOT / "splits"
    init_from_seed = model_name in schema.INIT_FROM_RUN_SEED
    out = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha_raw(split_dir / f"{PROTOCOL_ID}.json"),
        "role_ledger_sha256": sha_raw(split_dir / "role_ledger_v1.csv"),
        "split_manifest_sha256": sha_raw(split_dir / "split_manifest.json"),
        "feature_schema_sha256": sha_lf(paths.FEATURE_SCHEMA),
        "topology": topology,
        "init_policy": "run_seed" if init_from_seed else "module_default",
        "init_seed": int(seed) if init_from_seed else None,
    }
    if model_name in ("hgt_global", "hgt_global_off"):
        from src.models.baselines.hgt_global import derive_global_init_seed
        out["global_init_seed"] = derive_global_init_seed(int(seed))
    out.update(_cache_hashes(topology))
    return out


def _data_hashes(idx) -> dict:
    return {"samples_jsonl": idx.samples_sha256,
            "dataset_manifest": index_mod.sha256_file(idx.root / "dataset_manifest.json")}


def _slices(idx, prot, topology: str, n_train: int | None, n_val: int | None):
    train_ids = protocol.ids_for(idx, prot, topology, "A", "train")
    val = {
        "design_val": protocol.ids_for(idx, prot, topology, "A", "val"),
        "load_val": protocol.ids_for(idx, prot, topology, "B", "val"),
        "joint_val": protocol.ids_for(idx, prot, topology, "C", "val"),
    }
    if n_train is not None:
        train_ids = train_ids[:n_train]
    if n_val is not None:
        val = {k: v[:n_val] for k, v in val.items()}
    if not train_ids or any(not v for v in val.values()):
        raise Refused(f"{topology}: empty train or validation slice", 4)
    return train_ids, val


# ---------------------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------------------
def fit_ml(name, cfg, X, Y):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(cfg.get("alpha", 1.0))).fit(X, Y)
    from sklearn.ensemble import HistGradientBoostingRegressor
    seed = cfg.get("seed")
    if seed is None:
        raise Refused("hist_gb requires an explicit seed", 4)
    return [HistGradientBoostingRegressor(
        max_iter=int(cfg.get("max_iter", 200)), learning_rate=float(cfg.get("learning_rate", 0.1)),
        max_depth=cfg.get("max_depth"), early_stopping=False,
        random_state=int(seed)).fit(X, Y[:, c])
        for c in range(Y.shape[1])]


def predict_ml(name, model, X):
    return model.predict(X) if name == "ridge" else np.stack([m.predict(X) for m in model], axis=1)


def _run_gradient(args, cfg, cfg_path, task, out_dir, train_ids, val, st, idx,
                  max_val_batches, resume_from):
    model_name = task["model"]
    seed = int(task["seed"])
    local = json.loads(json.dumps(cfg))
    local["training"]["seed"] = seed
    local["training"]["device"] = args.device
    local["models"][model_name] = schema.task_model_cfg(
        model_name, cfg, task["topology"], seed)
    engine.seed_everything(seed)
    model = registry.build_gradient_model(model_name, local["models"][model_name])
    shared = _shared_cache(idx, task["topology"])
    tr = FeV2Dataset(idx, train_ids, st, dtype=torch.float32, normalize=True,
                     use_shared=shared is not None, shared=shared,
                     topology=task["topology"])
    val_ds = {name: FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                                use_shared=shared is not None, shared=shared,
                                topology=task["topology"])
              for name, ids in val.items()}
    pred_fmt = str(cfg["training"].get("predictions_format", "json"))
    evaluator = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=max_val_batches,
                                    predictions_format=pred_fmt)
    identity = engine.build_identity(
        model_name, seed, st, local, train_ids,
        [s for ids in val.values() for s in ids], _data_hashes(idx),
        PROTOCOL_ID, task["topology"],
        extra=_identity_extra(task["topology"], model_name, seed))
    log = engine.train_fold(model_name, model, tr, SliceBundle(val_ds.values()), st, local,
                            out_dir, identity, resume_from=resume_from,
                            val_evaluator=evaluator)
    if evaluator.last is None:
        raise RuntimeError("three-slice evaluator produced no final result")
    best_sha = log.get("best", {}).get("checkpoint_sha256")
    if log.get("final_eval_checkpoint_sha256") != best_sha:
        raise RuntimeError("final evaluation did not come from the best checkpoint")
    agg = evaluator.write_slice_artifacts(out_dir, run_label=f"best:{best_sha}")
    if abs(agg["selection_metric_equal_mean"] - evaluator.last["selection_metric"]) > 1e-12:
        raise RuntimeError("per-slice equal-mean selection != evaluator selection")
    log["slice_selection"] = evaluator.last["slice_selection"]
    log["slice_metrics"] = evaluator.last["slice_metrics"]
    log["slice_entity_weighted_norm_mse"] = evaluator.last["slice_entity_weighted_norm_mse"]
    log["slice_metrics_file"] = "slice_metrics.json"
    write_json_atomic(out_dir / "run_log.json", log)
    summary = {
        "schema": SCHEMA_ID, "mode": args.mode, "status": "done",
        "batch_id": args.batch_id, "topology": task["topology"], "worker": task["worker"],
        "model": model_name, "seed": seed, "attempt": args.attempt,
        "run_dir": str(out_dir), "train_graphs": len(train_ids),
        "val_graphs": {k: len(v) for k, v in val.items()},
        "selection_metric": evaluator.last["selection_metric"],
        "slice_selection": evaluator.last["slice_selection"],
        "slice_entity_weighted_norm_mse": evaluator.last["slice_entity_weighted_norm_mse"],
        "slice_metrics_file": "slice_metrics.json",
        "final_eval_source": log.get("final_eval_source"),
        "final_eval_checkpoint_sha256": log.get("final_eval_checkpoint_sha256"),
        "best": log.get("best"), "completed_epochs": log.get("completed_epochs"),
        "stop_reason": log.get("stop_reason"),
        "train_actual_updates": log.get("train_actual_updates"),
        "physics_loss_weight": 0.0, "test_constructed": False,
        "protocol": PROTOCOL_ID, "resume": log.get("resumed"),
        "predictions_format": pred_fmt,
        "val_predictions": [f"val_predictions_{k}.{ 'npz' if pred_fmt == 'npz' else 'json'}"
                            for k in val],
    }
    write_json_atomic(out_dir / "summary.json", summary)
    return log.get("pass", {}).get("all") is True, summary


def _run_ml(args, cfg, task, out_dir, train_ids, val, st, idx, max_val_batches):
    model_name = task["model"]
    seed = int(task["seed"]) if task["model"] == "hist_gb" else 0
    shared = _shared_cache(idx, task["topology"])
    tr = FeV2Dataset(idx, train_ids, st, normalize=True,
                     use_shared=shared is not None, shared=shared,
                     topology=task["topology"])
    Xn, Yn, Xb, Yb = collect(tr, train_ids)
    mc = dict(cfg["ml"][model_name])
    if model_name == "hist_gb":
        mc["seed"] = seed
    node = fit_ml(model_name, mc, Xn, Yn)
    beam = fit_ml(model_name, mc, Xb, Yb)
    run_name = model_name if model_name == "ridge" else f"{model_name}_seed{seed}"
    artifact = out_dir / f"ml_{run_name}.pkl"
    tmp = artifact.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"node": node, "beam": beam, "config": mc,
                     "topology": task["topology"], "train_ids_sha256": hash_json(sorted(train_ids))}, fh)
    os.replace(tmp, artifact)
    fit_rec = {
        "schema": "qt-fe-v2-ml-fit-v1",
        "model": model_name, "topology": task["topology"], "seed": seed,
        "config": mc, "train_graphs": len(train_ids),
        "train_ids_sha256": hash_json(sorted(train_ids)),
        "node_rows": int(Xn.shape[0]), "beam_rows": int(Xb.shape[0]),
        "node_features": int(Xn.shape[1]), "beam_features": int(Xb.shape[1]),
        "node_targets": int(Yn.shape[1]), "beam_targets": int(Yb.shape[1]),
        "early_stopping": False, "epochs": None, "best_epoch": None,
        "artifact": artifact.name, "artifact_sha256": sha_raw(artifact),
    }
    write_json_atomic(out_dir / "ml_fit.json", fit_rec)

    predictor = MLBatchPredictor(partial(predict_ml, model_name, node),
                                 partial(predict_ml, model_name, beam))
    val_ds = {name: FeV2Dataset(idx, ids, st, normalize=True,
                                use_shared=shared is not None, shared=shared,
                                topology=task["topology"])
              for name, ids in val.items()}
    pred_fmt = str(cfg["training"].get("predictions_format", "json"))
    evaluator = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=max_val_batches,
                                    predictions_format=pred_fmt)
    ev = evaluator(predictor, int(cfg["training"].get("val_batch_size", 4)),
                   torch.device("cpu"), collect_predictions=True)
    if ev.get("selection_metric") is None:
        raise RuntimeError("ML three-slice evaluation produced no selection metric")
    agg = evaluator.write_slice_artifacts(out_dir, run_label="single_fit")
    if abs(agg["selection_metric_equal_mean"] - ev["selection_metric"]) > 1e-12:
        raise RuntimeError("per-slice equal-mean selection != evaluator selection")
    if pred_fmt == "json":
        write_json_atomic(out_dir / "val_predictions.json", evaluator.last["exported"])
    summary = {
        "schema": SCHEMA_ID, "mode": args.mode, "status": "done",
        "batch_id": args.batch_id, "topology": task["topology"], "worker": task["worker"],
        "model": model_name, "seed": seed, "attempt": args.attempt,
        "run_dir": str(out_dir), "train_graphs": len(train_ids),
        "val_graphs": {k: len(v) for k, v in val.items()},
        "selection_metric": ev["selection_metric"],
        "slice_selection": ev["slice_selection"],
        "slice_entity_weighted_norm_mse": ev["slice_entity_weighted_norm_mse"],
        "slice_metrics_file": "slice_metrics.json",
        "final_eval_source": "single_fit",
        "final_eval_checkpoint_sha256": None,
        "best": None, "completed_epochs": None,
        "stop_reason": "single_fit_no_epochs",
        "train_actual_updates": None,
        "physics_loss_weight": 0.0, "test_constructed": False,
        "protocol": PROTOCOL_ID, "resume": None,
        "fit": fit_rec,
        "predictions_format": pred_fmt,
        "val_predictions": [f"val_predictions_{k}.{ 'npz' if pred_fmt == 'npz' else 'json'}"
                            for k in val],
    }
    write_json_atomic(out_dir / "summary.json", summary)
    return ev.get("selection_metric") is not None, summary


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _common(args):
    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise Refused(f"config not found: {cfg_path}", 4)
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    return cfg, cfg_path, idx


def run_smoke(args) -> int:
    cfg, cfg_path, idx = _common(args)
    prot = _protocol()
    n_train = int(cfg.get("smoke", {}).get("train_samples_per_template", 4))
    n_val = int(cfg.get("smoke", {}).get("val_samples_per_slice", 2))
    train_ids, val = _slices(idx, prot, args.topology, n_train, n_val)
    st = _stats(args.topology)
    out_dir = Path(args.out) / args.topology / args.model / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    task = {"topology": args.topology, "worker": args.worker or "smoke",
            "model": args.model, "seed": args.seed}
    max_val = args.max_val_batches or 1
    if args.model in GRADIENT:
        ok, summary = _run_gradient(args, cfg, cfg_path, task, out_dir, train_ids, val, st,
                                    idx, max_val, None)
    else:
        ok, summary = _run_ml(args, cfg, task, out_dir, train_ids, val, st, idx, max_val)
    print(json.dumps({"mode": "smoke", "pass": ok, "summary": summary}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def run_preflight(args) -> int:
    cfg, cfg_path, idx = _common(args)
    # Preflight uses the formal architecture but a capped budget; it must never
    # start a full-budget run.
    cfg["training"]["epochs"] = 1
    cfg["training"]["max_train_batches_per_epoch"] = 2
    cfg["training"]["val_batch_size"] = 1
    cfg["training"]["early_stopping_patience"] = None
    cfg["training"]["device"] = args.device
    prot = _protocol()
    n_train = args.train_samples
    n_val = args.val_samples
    train_ids, val = _slices(idx, prot, args.topology, n_train, n_val)
    st = _stats(args.topology)
    ts = os.environ.get("THREE_TOPO_PREFLIGHT_TS") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = preflight_root() / args.batch_id / "runs" / ts / args.topology / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    task = {"topology": args.topology, "worker": f"pf_{args.topology}_{args.model}",
            "model": args.model, "seed": args.seed}
    max_val = args.max_val_batches or 1
    if args.model in GRADIENT:
        ok, summary = _run_gradient(args, cfg, cfg_path, task, out_dir, train_ids, val, st,
                                    idx, max_val, None)
    else:
        ok, summary = _run_ml(args, cfg, task, out_dir, train_ids, val, st, idx, max_val)
    print(json.dumps({"mode": "preflight", "pass": ok, "run_dir": str(out_dir),
                      "selection_metric": summary.get("selection_metric")},
                     ensure_ascii=False, indent=2))
    return 0 if ok else 1


def run_formal(args) -> int:
    if args.out:
        raise Refused("formal mode derives its run directory; --out is not allowed", 4)
    if args.attempt < 1:
        raise Refused("attempt must be >= 1", 4)
    if args.attempt > 1 and not args.resume_from:
        raise Refused("a new attempt (>1) requires --resume-from; fresh restarts must use attempt 1", 4)
    if args.resume_from and args.attempt == 1:
        raise Refused("resume must use a new attempt (>=2)", 4)
    man = load_manifest(args.batch_id)
    seed_key = "-" if args.model == "ridge" else args.seed
    task = resolve_task(man, args.worker, args.topology, args.model, seed_key)
    if args.model in ML and task["device"] != "cpu":
        raise Refused("ML models must be frozen as CPU tasks", 4)
    if args.model in GRADIENT and task["device"] != "gpu":
        raise Refused("gradient formal tasks must be frozen as GPU tasks", 4)
    if task["device"] == "gpu" and not str(args.device).lower().startswith("cuda"):
        raise Refused("frozen GPU task requires a cuda device", 4)
    if task["device"] == "cpu" and str(args.device).lower() != "cpu":
        raise Refused("frozen CPU task requires device=cpu", 4)
    if not _shared_enabled():
        raise Refused("formal mode requires the shared read path "
                      "(THREE_TOPO_USE_SHARED=1); legacy extraction read is smoke-only", 4)
    cfg, cfg_path, idx = _common(args)
    _check_formal_preflight(man, task, cfg_path, cfg)
    expected = _expected_model_cfg(task, cfg)
    if hash_json(expected) != task["model_config_sha256"]:
        raise Refused("effective model config differs from the frozen manifest", 4)
    run = task["run"]
    out_dir = _run_dir(args.batch_id, args.topology, args.worker, run, args.attempt)
    _atomic_new_dir(out_dir)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    task_rec = {
        "batch_id": args.batch_id, "topology": args.topology, "worker": args.worker,
        "model": args.model, "seed": task["seed"], "run": run, "attempt": args.attempt,
        "mode": "formal", "device": "cpu" if args.model in ML else "cuda:0",
        "manifest": str(manifest_root() / f"{args.batch_id}.json"),
        "manifest_sha256": sha_raw(manifest_root() / f"{args.batch_id}.json"),
        "train_config": str(cfg_path), "train_config_sha256": sha_lf(cfg_path),
        "preflight_root": str(preflight_root() / args.batch_id),
        "resume_from": (str(Path(args.resume_from).resolve()) if args.resume_from else None),
        "started_at": started, "pid": os.getpid(),
    }
    write_json_atomic(out_dir / "TASK.json", task_rec)
    _status(out_dir, "running", batch_id=args.batch_id, topology=args.topology,
            worker=args.worker, model=args.model, seed=task["seed"], attempt=args.attempt,
            run_mode="resume" if args.resume_from else "new", started_at=started)
    console = (out_dir / "console.log").open("a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, console)
    sys.stderr = Tee(old_err, console)
    try:
        prot = _protocol()
        train_ids, val = _slices(idx, prot, args.topology, None, None)
        st = _stats(args.topology)
        max_val = cfg["training"].get("max_val_batches")
        if args.model in GRADIENT:
            resume = Path(args.resume_from).resolve() if args.resume_from else None
            ok, summary = _run_gradient(args, cfg, cfg_path, task, out_dir, train_ids, val, st,
                                        idx, max_val, resume)
        else:
            ok, summary = _run_ml(args, cfg, task, out_dir, train_ids, val, st, idx, max_val)
        if not ok:
            raise RuntimeError("run pass flag is false")
        _status(out_dir, "done", batch_id=args.batch_id, topology=args.topology,
                worker=args.worker, model=args.model, seed=task["seed"], attempt=args.attempt,
                run_mode="resume" if args.resume_from else "new",
                exit_code=0, selection_metric=summary.get("selection_metric"),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                resume_of=task_rec["resume_from"])
        print(json.dumps({"formal": True, "pass": True, "run_dir": str(out_dir),
                          "selection_metric": summary.get("selection_metric")},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        write_text_atomic(out_dir / "traceback.txt", tb)
        _status(out_dir, "failed", batch_id=args.batch_id, topology=args.topology,
                worker=args.worker, model=args.model, seed=task["seed"], attempt=args.attempt,
                run_mode="resume" if args.resume_from else "new",
                exit_code=(exc.code if isinstance(exc, Refused) else 1),
                error_type=type(exc).__name__, error=str(exc),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                resume_of=task_rec["resume_from"])
        print(f"[formal] FAILED: {exc}", file=sys.stderr)
        return exc.code if isinstance(exc, Refused) else 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        console.close()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "preflight", "formal"], default="smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True, choices=sorted(GRADIENT | ML))
    ap.add_argument("--topology", required=True, choices=list(TEMPLATES))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="smoke/preflight only")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--worker", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--train-samples", type=int, default=4)
    ap.add_argument("--val-samples", type=int, default=2)
    ap.add_argument("--max-val-batches", type=int, default=None)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.config is None:
        name = "smoke_pipeline.yaml" if args.mode == "smoke" else "formal_pipeline.yaml"
        args.config = str(paths.CONFIG_ROOT / name)
    try:
        if args.mode == "smoke":
            if args.out is None:
                args.out = str(paths.OUTPUT_ROOT / "smoke_train")
            return run_smoke(args)
        if args.mode == "preflight":
            if not args.batch_id:
                raise Refused("preflight mode requires --batch-id", 4)
            return run_preflight(args)
        if not args.batch_id or not args.worker:
            raise Refused("formal mode requires --batch-id and --worker", 4)
        return run_formal(args)
    except Refused as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

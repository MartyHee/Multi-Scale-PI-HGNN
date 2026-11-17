#!/usr/bin/env python
"""Unified train/validation entry for three-topology v2 baselines (corrected).

Supports:
  * v2 splits (``--protocol split``) and inner proxy folds (``--protocol inner``);
  * explicit device (cpu / cuda / cuda:N) with error if CUDA unavailable;
  * seeding before model construction (init independent of model run order);
  * identity-checked resume; best-checkpoint final evaluation + export;
  * eval-only (recompute validation metrics from a checkpoint);
  * a per-run traceability manifest + driver stdout/stderr log.

The outer test is never constructed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_v2 import dataset as ds_mod, index, paths, stats as stats_mod  # noqa: E402
from src.train.three_topology_v2 import engine, ml_baselines, registry  # noqa: E402


def load_config(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return json.loads(text)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_version():
    try:
        commit = subprocess.check_output(["git", "-C", str(MODEL_ROOT), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(MODEL_ROOT), "status", "--porcelain"],
                                             text=True).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"git_commit": "unknown", "git_dirty": None}


def _env_info():
    import platform
    import torch_geometric
    import numpy as _np
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
            "numpy": _np.__version__, "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}


def resolve_spec(args, cfg):
    """Return dict(split_id, fold_id, outer_fold, inner_fold, train_ids, val_ids, stats_path)."""
    if args.protocol == "split":
        split_id = args.split or cfg["data"]["split_id"]
        fold_id = args.fold or cfg["data"]["fold_id"]
        with (paths.SPLIT_DIR / f"split_{split_id}.json").open(encoding="utf-8") as fh:
            split = json.load(fh)
        fold = next(f for f in split["folds"] if f["fold_id"] == fold_id)
        stats_path = paths.STATS_DIR / split_id / f"{fold_id}.json"
        return {"split_id": split_id, "fold_id": fold_id, "outer_fold": None, "inner_fold": None,
                "train_ids": list(fold["train_sample_ids"]),
                "val_ids": list(fold["validation_sample_ids"]), "stats_path": stats_path}
    if args.protocol == "inner":
        manifest = json.loads((paths.MANIFEST_DIR / "inner_proxy_folds.json").read_text(encoding="utf-8"))
        outer = next(f for f in manifest["folds"] if f["outer_fold_id"] == args.outer_fold)
        inner = next(i for i in outer["inner_folds"] if i["inner_fold_id"] == args.inner_id)
        stats_path = paths.TRAIN_ROOT.parent / inner["inner_stats_path"]
        return {"split_id": f"inner_{args.outer_fold}", "fold_id": inner["inner_fold_id"],
                "outer_fold": args.outer_fold, "inner_fold": inner["inner_fold_id"],
                "train_ids": list(inner["train_sample_ids"]),
                "val_ids": list(inner["validation_sample_ids"]), "stats_path": stats_path}
    raise ValueError(args.protocol)


def load_stats_checked(stats_path: Path, train_ids):
    with stats_path.open(encoding="utf-8") as fh:
        st = json.load(fh)
    st["_file_sha256"] = _sha(stats_path)
    st["_path"] = str(stats_path)
    if sorted(st.get("train_sample_ids", [])) != sorted(train_ids):
        raise RuntimeError(
            f"stats/train mismatch: stats fit on different train samples than requested "
            f"(stats_n={len(st.get('train_sample_ids', []))}, run_n={len(train_ids)})")
    return st


def build_datasets(idx, train_ids, val_ids, st, root, cfg):
    tr = ds_mod.FoldDataset(index=idx, sample_ids=train_ids, fold_stats=st, root=root,
                            dtype=torch.float32, verify_hash=cfg["data"]["verify_hash"],
                            normalize_labels=True)
    va = ds_mod.FoldDataset(index=idx, sample_ids=val_ids, fold_stats=st, root=root,
                            dtype=torch.float32, verify_hash=cfg["data"]["verify_hash"],
                            normalize_labels=True)
    return tr, va


def run_one(model_name, seed, cfg, idx, spec, root, run_root, device_str, diagnostic,
            resume, eval_only, checkpoint, test_check, out_dir_override=None):
    st = load_stats_checked(spec["stats_path"], spec["train_ids"])
    train_ds, val_ds = build_datasets(idx, spec["train_ids"], spec["val_ids"], st, root, cfg)
    disp_inv = lambda t: stats_mod.inverse_transform_disp(t, st)
    force_inv = lambda t: stats_mod.inverse_transform_force(t, st)

    local_cfg = copy.deepcopy(cfg)
    local_cfg["training"]["seed"] = seed
    local_cfg["training"]["device"] = device_str
    local_cfg["training"]["diagnostic"] = diagnostic

    out_dir = Path(out_dir_override) if out_dir_override else (run_root / model_name / f"seed_{seed}")

    data_hashes = {"samples_jsonl": idx["hashes"]["samples_jsonl"],
                   "dataset_manifest": idx["hashes"]["dataset_manifest"]}

    is_ml = model_name in registry.NON_GRADIENT_MODELS
    if eval_only:
        if is_ml:
            raise RuntimeError("eval-only is for gradient checkpoints (ML run_log carries metrics)")
        model = registry.build_gradient_model(model_name, cfg["models"][model_name])
        dev = engine.resolve_device(device_str)
        log = engine.evaluate_checkpoint(model, val_ds, st, int(cfg["training"].get("val_batch_size",
                                    cfg["training"]["batch_size"])), dev, checkpoint,
                                    out_dir / "eval")
        return log, out_dir

    # seeding BEFORE model construction (init independent of run order)
    engine.seed_everything(seed)

    if is_ml:
        ml_cfg = dict(cfg.get("ml", {}).get(model_name, {})); ml_cfg["seed"] = seed
        log = ml_baselines.run_ml_baseline(model_name, ml_cfg, train_ds, val_ds, st,
                                           disp_inv, force_inv, seed, out_dir)
        log["identity"] = engine.build_identity(
            model_name, seed, st, local_cfg, spec["train_ids"], spec["val_ids"], data_hashes,
            spec["split_id"], spec["fold_id"], spec["outer_fold"], spec["inner_fold"])
        log["stats_path"] = str(spec["stats_path"]); log["stats_sha256"] = st["_file_sha256"]
        (out_dir / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return log, out_dir

    model = registry.build_gradient_model(model_name, cfg["models"][model_name])
    identity = engine.build_identity(
        model_name, seed, st, local_cfg, spec["train_ids"], spec["val_ids"], data_hashes,
        spec["split_id"], spec["fold_id"], spec["outer_fold"], spec["inner_fold"])
    log = engine.train_fold(model_name, model, train_ds, val_ds, st, local_cfg, out_dir,
                            identity, resume_from=resume)
    log["stats_path"] = str(spec["stats_path"]); log["stats_sha256"] = st["_file_sha256"]
    (out_dir / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log, out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "formal_baselines.yaml"))
    ap.add_argument("--models", default="mlp,hgt_global")
    ap.add_argument("--protocol", choices=["split", "inner"], default="split")
    ap.add_argument("--split", default=None)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--outer-fold", default=None)
    ap.add_argument("--inner-id", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ml-repeat-seeds", action="store_true",
                    help="repeat deterministic ML models across seeds (off by default)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-root", default=None,
                    help="exact per-task run directory (used by run_task.sh)")
    ap.add_argument("--require-new-dir", action="store_true",
                    help="fail if --run-root already exists (refuse overwrite)")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--worker-id", default=None)
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(Path(args.config)))
    paths.ensure_derived_dirs()
    root = Path(cfg["data"].get("package_root") or paths.package_root())
    device_str = args.device or cfg["training"].get("device", "cpu")

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_batches is not None:
        cfg["training"]["max_train_batches_per_epoch"] = args.max_batches
    if args.val_batch_size is not None:
        cfg["training"]["val_batch_size"] = args.val_batch_size
    if args.smoke:
        cfg["training"]["epochs"] = args.epochs or 2
        cfg["training"]["max_train_batches_per_epoch"] = args.max_batches or 2

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["training"]["seeds"])
    if args.smoke and args.seeds is None:
        seeds = seeds[:1]
    if args.eval_only and args.seeds is None:
        seeds = seeds[:1]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.resume and (len(models) != 1 or len(seeds) != 1):
        raise SystemExit("--resume requires exactly one --models entry and one seed")

    spec = resolve_spec(args, cfg)
    idx = index.load_index(root=root, strict=True)

    # outer-test seal assertion: test ids never enter train/val
    test_ids = set()
    if args.protocol == "split":
        with (paths.SPLIT_DIR / f"split_{spec['split_id']}.json").open(encoding="utf-8") as fh:
            sp = json.load(fh)
        fold = next(f for f in sp["folds"] if f["fold_id"] == spec["fold_id"])
        test_ids = set(fold["test_sample_ids"])
    test_check = {"test_ids_present": bool(test_ids),
                  "test_disjoint_from_train": not (test_ids & set(spec["train_ids"])),
                  "test_disjoint_from_val": not (test_ids & set(spec["val_ids"]))}
    if test_ids and not (test_check["test_disjoint_from_train"] and test_check["test_disjoint_from_val"]):
        raise RuntimeError(f"outer-test seal violated: {test_check}")

    tag = args.tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.run_root:
        run_root = Path(args.run_root)
        mode = "task"
        if args.require_new_dir:
            run_root.mkdir(parents=True, exist_ok=False)
        else:
            run_root.mkdir(parents=True, exist_ok=True)
    else:
        mode = "eval" if args.eval_only else ("smoke" if args.smoke else ("inner" if args.protocol == "inner" else "formal"))
        run_root = paths.OUTPUT_ROOT / mode / tag
        run_root.mkdir(parents=True, exist_ok=True)

    # driver stdout/stderr log
    driver_path = run_root / "driver.log"
    driver_fh = driver_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, driver_fh); sys.stderr = Tee(old_err, driver_fh)

    device_obj = engine.resolve_device(device_str)  # explicit; raises if cuda unavailable
    run_manifest = {
        "schema": "qt-three-topology-v2-run-manifest-v1",
        "tag": tag, "mode": mode, "protocol": args.protocol,
        "batch_id": args.batch_id, "worker_id": args.worker_id,
        "run_root": str(run_root),
        "split_id": spec["split_id"], "fold_id": spec["fold_id"],
        "outer_fold": spec["outer_fold"], "inner_fold": spec["inner_fold"],
        "train_sample_ids": spec["train_ids"], "val_sample_ids": spec["val_ids"],
        "train_sample_list_sha256": hashlib.sha256("\n".join(sorted(spec["train_ids"])).encode()).hexdigest(),
        "val_sample_list_sha256": hashlib.sha256("\n".join(sorted(spec["val_ids"])).encode()).hexdigest(),
        "data_hashes": idx["hashes"],
        "stats_path": str(spec["stats_path"]), "stats_sha256": _sha(spec["stats_path"]),
        "resolved_config": cfg, "device_requested": device_str,
        "device_info": engine.device_info(device_obj),
        "environment": _env_info(), "code_version": _code_version(),
        "seeds": seeds, "models": models, "outer_test_seal": test_check,
        "outer_test_access": "never constructed in this entry",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    summaries = {}
    failures = {}
    for model_name in models:
        run_seeds = seeds if (model_name not in registry.NON_GRADIENT_MODELS or args.ml_repeat_seeds) else [seeds[0]]
        for seed in run_seeds:
            key = f"{model_name}/seed_{seed}"
            single_task = (len(models) == 1 and len(run_seeds) == 1)
            out_override = run_root if (args.run_root and single_task) else None
            try:
                log, out_dir = run_one(model_name, seed, cfg, idx, spec, root, run_root,
                                       device_str, args.diagnostic,
                                       Path(args.resume) if args.resume else None,
                                       args.eval_only, Path(args.checkpoint) if args.checkpoint else None,
                                       test_check, out_dir_override=out_override)
                if args.eval_only:
                    ok = log.get("selection_metric") is not None
                    sel = log.get("selection_metric")
                else:
                    ok = bool(log.get("pass", {}).get("all", False))
                    sel = log.get("selection_metric",
                                  log.get("final_val", {}).get("selection_metric")
                                  if isinstance(log.get("final_val"), dict) else None)
                summaries[key] = {"pass": ok, "path": str(out_dir), "selection_metric": sel}
            except Exception as exc:  # noqa: BLE001
                failures[key] = repr(exc)
                err_dir = out_override if out_override is not None else (run_root / model_name / f"seed_{seed}")
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "traceback.txt").write_text(
                    f"{model_name} seed={seed}\n{traceback.format_exc()}", encoding="utf-8")
                raise

    summary = {"schema": "qt-three-topology-v2-run-summary-v1", "tag": tag, "mode": mode,
               "batch_id": args.batch_id, "worker_id": args.worker_id, "run_root": str(run_root),
               "split_id": spec["split_id"], "fold_id": spec["fold_id"],
               "counts": {"train": len(spec["train_ids"]), "val": len(spec["val_ids"])},
               "seeds": seeds, "models": summaries, "failures": failures,
               "outer_test_seal": test_check,
               "all_pass": not failures and all(v["pass"] for v in summaries.values())}
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.stdout = old_out; sys.stderr = old_err; driver_fh.close()
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

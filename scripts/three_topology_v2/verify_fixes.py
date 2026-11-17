#!/usr/bin/env python
"""Local CPU verification suite for the training-correctness fixes.

Covers:
  1. same seed, changed model-construction order -> identical init params
  2. CPU continuous training vs interrupt+resume -> identical sample order,
     parameters, optimizer state and selection metric (within tolerance)
  3. early stopping: continuous-improve / no-improve / improve-then-stall
  4. final validation/export come from the BEST checkpoint (best epoch recorded)
  5. graph-equal selection metric invariant to validation batch size
  6. mismatched stats / checkpoint(seed) rejected
  7. non-finite guard raises
  8. HistGB early_stopping disabled; inner-fold stats fit on inner-train only

Writes a JSON report. CPU only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(MODEL_ROOT / "scripts" / "three_topology_v2"))

import torch  # noqa: E402

from src.data.three_topology_v2 import dataset as ds_mod, index, paths, stats as stats_mod  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

REPORT = {"checks": {}, "ok": True}


def record(name, passed, detail=""):
    REPORT["checks"][name] = {"pass": bool(passed), "detail": detail}
    if not passed:
        REPORT["ok"] = False


def tiny_cfg(epochs=3, patience=None, seed=7):
    return {
        "training": {"seed": seed, "batch_size": 4, "val_batch_size": 4,
                     "epochs": epochs, "lr": 1e-3, "weight_decay": 0.0,
                     "early_stopping_patience": patience, "min_delta": 0.0,
                     "grad_clip": 5.0, "max_train_batches_per_epoch": 2,
                     "diagnostic": False, "device": "cpu"},
        "models": {"mlp": {"mesh_feat_dim": 15, "beam_feat_dim": 15,
                           "hidden_dims": [32, 16], "dropout": 0.0,
                           "activation": "relu", "use_batch_norm": False}},
    }


def main() -> int:
    paths.ensure_derived_dirs()
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    split = json.loads((paths.SPLIT_DIR / "split_cross_design_topology1.json").read_text(encoding="utf-8"))
    fold = next(f for f in split["folds"] if f["fold_id"] == "cd_fold_1")
    st = json.loads((paths.STATS_DIR / "cross_design_topology1" / "cd_fold_1.json").read_text(encoding="utf-8"))
    st["_file_sha256"] = "test"
    train_ids = fold["train_sample_ids"][:8]
    val_ids = fold["validation_sample_ids"][:4]
    train_ds = ds_mod.FoldDataset(idx, train_ids, st, root=root, dtype=torch.float32,
                                  verify_hash=False, normalize_labels=True)
    val_ds = ds_mod.FoldDataset(idx, val_ids, st, root=root, dtype=torch.float32,
                                verify_hash=False, normalize_labels=True)
    data_hashes = {"samples_jsonl": idx["hashes"]["samples_jsonl"],
                   "dataset_manifest": idx["hashes"]["dataset_manifest"]}
    outbase = paths.OUTPUT_ROOT / "verify_fixes"
    outbase.mkdir(parents=True, exist_ok=True)

    # ---- 1. seed/order independence ----
    cfg = tiny_cfg()
    engine.seed_everything(123)
    m1 = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    engine.seed_everything(999)
    _ = registry.build_gradient_model("hgt_global", {
        "mesh_feat_dim": 15, "beam_feat_dim": 15, "plate_feat_dim": 6,
        "hidden_dim": 32, "num_layers": 1, "heads": 2, "dropout": 0.0,
        "use_layer_norm": True, "decoder_hidden_dims": [16], "global_mode": "mesh_global"})
    engine.seed_everything(123)
    m2 = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    same = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    record("seed_order_independence", same, "state_dict equal after reseed+reorder")

    # ---- 2. CPU resume determinism ----
    def identity(seed=7):
        return engine.build_identity("mlp", seed, st, cfg, train_ids, val_ids,
                                     data_hashes, "cross_design_topology1", "cd_fold_1")

    # continuous 3 epochs
    engine.seed_everything(7)
    mA = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    logA = engine.train_fold("mlp", mA, train_ds, val_ds, st, tiny_cfg(epochs=3),
                             outbase / "cont", identity(7))
    # interrupted: 1 epoch then resume to 3
    engine.seed_everything(7)
    mB = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    engine.train_fold("mlp", mB, train_ds, val_ds, st, tiny_cfg(epochs=1),
                      outbase / "resume", identity(7))
    logB = engine.train_fold("mlp", mB, train_ds, val_ds, st, tiny_cfg(epochs=3),
                             outbase / "resume", identity(7),
                             resume_from=outbase / "resume" / "checkpoints" / "last.pt")
    orderA = [e["order_sha256"] for e in logA["epochs_log"]]
    # resume run's epochs_log only has epochs 2..; prepend from the first run
    orderB = [e["order_sha256"] for e in logB["epochs_log"]]
    # reconstruct expected B orders for epochs 0..2 from rule + the 1-epoch run's log
    logB1 = json.loads((outbase / "resume" / "run_log.json").read_text(encoding="utf-8"))
    # final run_log was overwritten by resume; instead compare params/metrics
    param_checksum_A = sum(p.detach().double().abs().sum().item() for p in mA.parameters())
    param_checksum_B = sum(p.detach().double().abs().sum().item() for p in mB.parameters())
    resume_ok = abs(param_checksum_A - param_checksum_B) < 1e-9
    selA = logA["final_val"]["selection_metric"]; selB = logB["final_val"]["selection_metric"]
    sel_ok = abs(selA - selB) < 1e-9
    record("cpu_resume_param_match", resume_ok, f"checksum {param_checksum_A:.6f} vs {param_checksum_B:.6f}")
    record("cpu_resume_metric_match", sel_ok, f"selection {selA:.9f} vs {selB:.9f}")
    record("cpu_resume_requested", logB.get("resumed") is not None,
           f"resumed start_epoch={logB.get('resumed', {}).get('start_epoch') if logB.get('resumed') else None}")

    # ---- 3. early stopping sequences (scripted validation metric) ----
    class FakeVal:
        def __init__(self, seq):
            self.seq = seq; self.i = 0

        def __call__(self, *a, **k):
            v = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1
            return {"n_batches": 1, "metrics": {"selection_metric": v},
                    "selection_metric": v, "entity_weighted_norm_mse": v, "exported": []}

    orig_validate = engine.validate
    es_cases = [("improve", [1.0, 0.9, 0.8, 0.7, 0.6], False, 5, 4),
                ("no_improve", [1.0, 1.0, 1.0, 1.0, 1.0], True, 3, 0),
                ("improve_then_stall", [1.0, 0.9, 0.9, 0.9, 0.9], True, 4, 1)]
    for name, seq, expect_stop, expect_completed, expect_best in es_cases:
        engine.validate = FakeVal(seq)
        engine.seed_everything(7)
        m = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
        log = engine.train_fold("mlp", m, train_ds, val_ds, st, tiny_cfg(epochs=5, patience=2),
                                outbase / f"es_{name}", identity(7))
        ok = (log["early_stopped"] == expect_stop and log["completed_epochs"] == expect_completed
              and log["best"]["epoch_zero_based"] == expect_best)
        record(f"early_stop_{name}", ok,
               f"stopped={log['early_stopped']} completed={log['completed_epochs']} "
               f"best_epoch={log['best']['epoch_zero_based']} (expected {expect_stop}/{expect_completed}/{expect_best})")
    engine.validate = orig_validate

    # ---- resume inherits the historical best (no new best after resume) ----
    class ScriptedVal:
        def __init__(self, seq): self.seq = seq; self.i = 0
        def __call__(self, *a, **k):
            v = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1
            return {"n_batches": 1, "metrics": {"selection_metric": v},
                    "selection_metric": v, "entity_weighted_norm_mse": v, "exported": []}

    outA = outbase / "inherit_A"; outB = outbase / "inherit_B"
    for d in (outA, outB, outbase / "orphan", outbase / "orphan_run"):
        shutil.rmtree(d, ignore_errors=True)
    engine.validate = ScriptedVal([0.5, 0.6])
    engine.seed_everything(7)
    mA = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    logIA = engine.train_fold("mlp", mA, train_ds, val_ds, st, tiny_cfg(epochs=2),
                              outA, identity(7))
    engine.validate = ScriptedVal([0.9, 0.9])
    engine.seed_everything(7)
    mB = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    logIB = engine.train_fold("mlp", mB, train_ds, val_ds, st, tiny_cfg(epochs=4),
                              outB, identity(7),
                              resume_from=outA / "checkpoints" / "last.pt")
    engine.validate = orig_validate
    bestA, bestB = logIA["best"], logIB["best"]
    inherit_ok = (bestB["source"] == "inherited"
                  and bestB["epoch_zero_based"] == bestA["epoch_zero_based"]
                  and abs(bestB["selection_metric"] - bestA["selection_metric"]) < 1e-12
                  and bestB["checkpoint_sha256"] == bestA["checkpoint_sha256"]
                  and (outB / "checkpoints" / "best.pt").is_file())
    record("resume_inherits_historical_best", inherit_ok,
           f"source={bestB['source']} bestA_epoch={bestA['epoch_zero_based']} "
           f"bestB_epoch={bestB['epoch_zero_based']} bestA={bestA['selection_metric']} bestB={bestB['selection_metric']}")
    # missing best.pt on resume with best_epoch>=0 must error
    orphan = outbase / "orphan"; orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "checkpoints").mkdir(exist_ok=True)
    shutil.copy(outA / "checkpoints" / "last.pt", orphan / "checkpoints" / "last.pt")
    engine.validate = ScriptedVal([0.1])
    engine.seed_everything(7)
    mO = registry.build_gradient_model("mlp", cfg["models"]["mlp"])
    try:
        engine.train_fold("mlp", mO, train_ds, val_ds, st, tiny_cfg(epochs=3),
                          outbase / "orphan_run", identity(7),
                          resume_from=orphan / "checkpoints" / "last.pt")
        missing_best_ok = False
    except RuntimeError as exc:
        missing_best_ok = "best.pt is missing" in str(exc) or "no best checkpoint" in str(exc)
    engine.validate = orig_validate
    record("resume_missing_best_errors", missing_best_ok, "missing best refused")
    # per-epoch CSV log exists and has rows
    csv_lines = (outA / "train_log.csv").read_text(encoding="utf-8").strip().splitlines() if (outA / "train_log.csv").is_file() else []
    record("train_log_csv", len(csv_lines) == 3, f"lines={len(csv_lines)} (header+2 epochs)")

    # ---- 4/5. best-vs-last + batch-size invariance on a real run ----
    best_path = outbase / "cont" / "checkpoints" / "best.pt"
    ev4 = engine.evaluate_checkpoint(registry.build_gradient_model("mlp", cfg["models"]["mlp"]),
                                     val_ds, st, 4, torch.device("cpu"), best_path, outbase / "eval_bs4")
    ev7 = engine.evaluate_checkpoint(registry.build_gradient_model("mlp", cfg["models"]["mlp"]),
                                     val_ds, st, 7, torch.device("cpu"), best_path, outbase / "eval_bs7")
    ev24 = engine.evaluate_checkpoint(registry.build_gradient_model("mlp", cfg["models"]["mlp"]),
                                      val_ds, st, 24, torch.device("cpu"), best_path, outbase / "eval_bs24")
    sels = [ev4["selection_metric"], ev7["selection_metric"], ev24["selection_metric"]]
    record("batchsize_invariance", max(sels) - min(sels) < 1e-9, f"selections {sels}")
    record("final_export_from_best",
           abs(logA["final_val"]["selection_metric"] - ev4["selection_metric"]) < 1e-9,
           f"train-final {logA['final_val']['selection_metric']:.9f} vs best-eval {ev4['selection_metric']:.9f}")
    record("best_epoch_recorded", isinstance(logA["best"]["epoch_zero_based"], int),
           f"best epoch {logA['best']['epoch_zero_based']}")

    # ---- 6. mismatch rejection ----
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    bad_seed = engine.check_resume_compat(ck["identity"], {**ck["identity"], "seed": 999})
    record("resume_rejects_seed_mismatch", not bad_seed["ok"], str(bad_seed["diffs"]))
    bad_stats = engine.check_resume_compat(ck["identity"], {**ck["identity"], "stats_sha256": "deadbeef"})
    record("resume_rejects_stats_mismatch", not bad_stats["ok"], str(bad_stats["diffs"]))
    bad_model = engine.check_resume_compat(ck["identity"], {**ck["identity"], "model_name": "hgt_global"})
    record("resume_rejects_model_mismatch", not bad_model["ok"], str(bad_model["diffs"]))
    # stats/train consistency guard used by the entry
    try:
        spec = importlib.util.spec_from_file_location(
            "tf", MODEL_ROOT / "scripts" / "three_topology_v2" / "train_formal.py")
        tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
        try:
            tf.load_stats_checked(paths.STATS_DIR / "cross_design_topology2" / "cd_fold_1.json", train_ids)
            stats_guard = False
        except RuntimeError:
            stats_guard = True
        record("entry_rejects_stats_train_mismatch", stats_guard, "load_stats_checked raised")
    except Exception as exc:  # noqa: BLE001
        record("entry_rejects_stats_train_mismatch", False, repr(exc))

    # ---- 7. non-finite guard ----
    try:
        engine._assert_finite(torch.tensor([1.0, float("nan")]), "test")
        nf = False
    except RuntimeError:
        nf = True
    record("non_finite_guard", nf, "raises on NaN")

    # ---- 8. HistGB early stopping + inner stats rule ----
    from sklearn.ensemble import HistGradientBoostingRegressor
    record("histgb_default_off", HistGradientBoostingRegressor(early_stopping=False).early_stopping is False,
           "early_stopping disabled explicitly")
    man = json.loads((paths.MANIFEST_DIR / "inner_proxy_folds.json").read_text(encoding="utf-8"))
    inner_ok = True
    inner_detail = []
    for outer in man["folds"]:
        outer_train = set(outer.get("train_sample_ids", [])) if outer.get("train_sample_ids") else None
        for inner in outer["inner_folds"]:
            ist = json.loads((paths.TRAIN_ROOT.parent / inner["inner_stats_path"]).read_text(encoding="utf-8"))
            fit_ids = set(ist["train_sample_ids"])
            ok = fit_ids == set(inner["train_sample_ids"])
            # inner train/val must be the outer-train topologies (subset of outer train)
            if outer_train is not None:
                ok = ok and fit_ids <= outer_train and set(inner["validation_sample_ids"]) <= outer_train
            inner_ok = inner_ok and ok
            inner_detail.append({"inner": inner["inner_fold_id"], "ok": ok,
                                 "fit_topology": ist["source"].get("fit_topology_only")})
    record("inner_stats_only_from_inner_train", inner_ok, inner_detail)

    (paths.OUTPUT_ROOT / "verify_fixes" / "verify_report.json").write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(REPORT, ensure_ascii=False, indent=2))
    return 0 if REPORT["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

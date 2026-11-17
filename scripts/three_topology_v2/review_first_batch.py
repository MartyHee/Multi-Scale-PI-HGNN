#!/usr/bin/env python
"""Local review of the first-batch server artifacts (topology1 cd_fold_1).

Reads the extracted artifact dir, verifies archive-self consistency and
frozen-input identity, checks training correctness from run evidence, and
recomputes validation metrics from the exported predictions + val labels
(NO new inference; outer test never loaded).

Outputs JSON/CSV/PNG under outputs/three_topology_v2/first_batch_review_20261118/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402
from src.train.three_topology_v2.metrics import (DISP_NAMES, DISP_UNITS, FORCE_NAMES,  # noqa: E402
                                                 FORCE_UNITS)

BATCH = "fb_20261117_01"
SPLIT_ID = "cross_design_topology1"
FOLD_ID = "cd_fold_1"
EXPECTED = [
    ("t1_mlp", "mlp_seed20261117", "mlp", 20261117, "gradient"),
    ("t1_mlp", "mlp_seed20261118", "mlp", 20261118, "gradient"),
    ("t1_mlp", "mlp_seed20261119", "mlp", 20261119, "gradient"),
    ("t1_ridge", "ridge", "ridge", None, "ml"),
    ("t1_histgb", "hist_gb", "hist_gb", None, "ml"),
    ("t2_hgt", "hgt_global_seed20261117", "hgt_global", 20261117, "gradient"),
    ("t3_hgt", "hgt_global_seed20261118", "hgt_global", 20261118, "gradient"),
    ("t4_hgt", "hgt_global_seed20261119", "hgt_global", 20261119, "gradient"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha256sums_verify(root: Path):
    sums = root / "SHA256SUMS"
    listed, bad = {}, []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        rel = rel.lstrip("./")
        listed[rel] = h
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    for rel, h in listed.items():
        p = root / rel
        if not p.is_file():
            bad.append({"file": rel, "issue": "missing"})
        elif sha(p) != h:
            bad.append({"file": rel, "issue": "mismatch"})
    extra = sorted(actual - set(listed))
    return {"listed": len(listed), "actual": len(actual), "bad": bad, "extra": extra,
            "ok": not bad and not extra}


def rmse(a):
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(MODEL_ROOT / "remote_artifacts" / f"first_batch_{BATCH}_review"))
    ap.add_argument("--out-dir", default=str(paths.OUTPUT_ROOT / "first_batch_review_20261118"))
    a = ap.parse_args()
    arch = Path(a.archive_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    local_root = paths.package_root()
    idx = index.load_index(root=local_root, strict=True)
    row_by_id = idx["row_by_id"]
    local_split_file = paths.SPLIT_DIR / f"split_{SPLIT_ID}.json"
    local_split = read_json(local_split_file)
    local_fold = next(f for f in local_split["folds"] if f["fold_id"] == FOLD_ID)
    local_stats_file = paths.STATS_DIR / SPLIT_ID / f"{FOLD_ID}.json"
    local_stats = read_json(local_stats_file)

    arch_split_file = arch / "manifests" / "splits" / f"split_{SPLIT_ID}.json"
    arch_stats_file = arch / "manifests" / "stats" / SPLIT_ID / f"{FOLD_ID}.json"
    arch_split = read_json(arch_split_file)
    arch_fold = next(f for f in arch_split["folds"] if f["fold_id"] == FOLD_ID)
    arch_stats_sha = sha(arch_stats_file)

    tar_path = MODEL_ROOT / "remote_artifacts" / f"first_batch_{BATCH}_artifacts.tar.gz"
    report = {"schema": "qt-first-batch-review-v1", "date": "2026-11-18",
              "archive_dir": str(arch), "batch_id": BATCH,
              "archive_sha256_local": (sha(tar_path) if tar_path.is_file() else None),
              "sha256sums": sha256sums_verify(arch),
              "frozen_input": {
                  "samples_jsonl_sha256": idx["hashes"]["samples_jsonl"],
                  "dataset_manifest_sha256": idx["hashes"]["dataset_manifest"],
                  "local_split_sha256": sha(local_split_file),
                  "archived_split_sha256": sha(arch_split_file),
                  "local_stats_sha256": sha(local_stats_file),
                  "archived_stats_sha256": arch_stats_sha,
                  "split_match": sha(local_split_file) == sha(arch_split_file),
                  "stats_match": sha(local_stats_file) == arch_stats_sha,
              },
              "runs": {}, "problems": []}

    # expected partitions from archived split (what training actually used)
    tr_train = set(arch_fold["train_sample_ids"]); tr_val = set(arch_fold["validation_sample_ids"])
    tr_test = set(arch_fold["test_sample_ids"])
    report["frozen_input"].update({"train_n": len(tr_train), "val_n": len(tr_val), "test_n": len(tr_test)})

    # load val graphs once (val only) for physical targets
    val_graphs = {}
    val_label = {}
    for sid in sorted(tr_val):
        row = row_by_id[sid]
        d = graph_io.load_graph(row, root=local_root, dtype=torch.float64, verify_hash=True)
        val_graphs[sid] = d
        val_label[sid] = (d["mesh_node"].y_disp.numpy(), d["beam_element"].y_force.numpy())
    std_disp = np.array([c["effective_std"] for c in local_stats["columns"]["y_disp"]], dtype=np.float64)
    std_force = np.array([c["effective_std"] for c in local_stats["columns"]["y_force"]], dtype=np.float64)

    def recompute(recs):
        by_id = {r["sample_id"]: r for r in recs}
        if set(by_id) != tr_val:
            return None, {"error": "val prediction ids != fold val ids",
                          "missing": sorted(tr_val - set(by_id))[:5],
                          "extra": sorted(set(by_id) - tr_val)[:5]}
        errs = {"disp": [], "force": [], "gdisp": [], "gforce": []}
        per_graph = {}
        comp_rows = []
        for sid in sorted(tr_val):
            r = by_id[sid]
            pd = np.asarray(r["pred_disp"], dtype=np.float64)
            pf = np.asarray(r["pred_force"], dtype=np.float64)
            td, tf = val_label[sid]
            if pd.shape != td.shape or pf.shape != tf.shape:
                return None, {"error": f"{sid} shape pred/target mismatch {pd.shape}/{td.shape} {pf.shape}/{tf.shape}"}
            ed = pd - td; ef = pf - tf
            errs["disp"].append(ed); errs["force"].append(ef)
            per_graph[sid] = {
                "disp_mse": (ed ** 2).mean(axis=0), "force_mse": (ef ** 2).mean(axis=0),
                "disp_mae": np.abs(ed).mean(axis=0), "force_mae": np.abs(ef).mean(axis=0),
            }
        D = np.concatenate(errs["disp"], axis=0); F = np.concatenate(errs["force"], axis=0)
        # per-component entity-weighted MAE/RMSE
        for kind, E, names, units, std in (("disp", D, DISP_NAMES, DISP_UNITS, std_disp),
                                           ("force", F, FORCE_NAMES, FORCE_UNITS, std_force)):
            for c in range(E.shape[1]):
                comp_rows.append({"task": kind, "component": names[c], "unit": units[c],
                                  "mae": float(np.abs(E[:, c]).mean()),
                                  "rmse": rmse(E[:, c]),
                                  "norm_mae_by_train_std": float(np.abs(E[:, c]).mean() / std[c]),
                                  "mean_abs_target": float(np.abs(np.concatenate(
                                      [val_label[s][0 if kind == "disp" else 1][:, c] for s in sorted(tr_val)])).mean())})
        # graph-equal component-equal normalised MSE (matches engine selection)
        gd = np.stack([per_graph[s]["disp_mse"] for s in sorted(tr_val)])  # (24,6)
        gf = np.stack([per_graph[s]["force_mse"] for s in sorted(tr_val)])  # (24,12)
        sel_disp = float(np.nanmean(gd / (std_disp ** 2)[None, :]))
        sel_force = float(np.nanmean(gf / (std_force ** 2)[None, :]))
        selection = 0.5 * (sel_disp + sel_force)
        ent_disp = float(np.mean(np.mean(D ** 2, axis=0) / (std_disp ** 2)))
        ent_force = float(np.mean(np.mean(F ** 2, axis=0) / (std_force ** 2)))
        per_graph_rows = []
        for sid in sorted(tr_val):
            g = per_graph[sid]
            seld = float(np.mean(g["disp_mse"] / (std_disp ** 2)))
            self_ = float(np.mean(g["force_mse"] / (std_force ** 2)))
            row = row_by_id[sid]
            per_graph_rows.append({"sample_id": sid, "design_id": row["design_id"],
                                   "template_id": row.get("template_id"), "topology": row["topology_token"],
                                   "selection": 0.5 * (seld + self_),
                                   "disp_mae_entity": float(g["disp_mae"].mean()),
                                   "force_mae_entity": float(g["force_mae"].mean())})
        return {"selection": selection, "selection_disp": sel_disp, "selection_force": sel_force,
                "entity_weighted_norm_mse_disp": ent_disp, "entity_weighted_norm_mse_force": ent_force,
                "components": comp_rows, "per_graph": per_graph_rows}, {}

    for worker, run, model, seed, kind in EXPECTED:
        d = arch / "batch" / BATCH / worker / run
        rec = {"worker": worker, "run": run, "model": model, "expected_seed": seed, "kind": kind,
               "dir": str(d), "files": {}, "checks": {}, "problems": []}
        for f in ("STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
                  "console.log", "val_predictions.json", "train_log.csv", "checkpoints/best.pt",
                  "checkpoints/last.pt", f"ml_{model}.pkl"):
            rec["files"][f] = (d / f).is_file()
        if not (d / "run_manifest.json").is_file():
            rec["problems"].append("missing run_manifest")
            report["runs"][f"{worker}/{run}"] = rec
            continue
        man = read_json(d / "run_manifest.json"); status = read_json(d / "STATUS.json") if (d/"STATUS.json").is_file() else {}
        task = read_json(d / "TASK.json") if (d/"TASK.json").is_file() else {}
        log = read_json(d / "run_log.json") if (d/"run_log.json").is_file() else {}
        summary = read_json(d / "summary.json") if (d/"summary.json").is_file() else {}
        preds = read_json(d / "val_predictions.json") if (d/"val_predictions.json").is_file() else []

        # identity checks
        rec["checks"]["status_done"] = status.get("state") == "done" and status.get("exit_code") == 0
        rec["checks"]["batch_worker"] = (man.get("batch_id") == BATCH and man.get("worker_id") == worker)
        rec["checks"]["split_fold"] = (man.get("split_id") == SPLIT_ID and man.get("fold_id") == FOLD_ID)
        rec["checks"]["data_hash_match"] = (man.get("data_hashes", {}).get("samples_jsonl") == idx["hashes"]["samples_jsonl"]
                                            and man.get("data_hashes", {}).get("dataset_manifest") == idx["hashes"]["dataset_manifest"])
        rec["checks"]["stats_hash_match"] = (man.get("stats_sha256") == arch_stats_sha)
        rec["checks"]["train_ids_match"] = set(man.get("train_sample_ids", [])) == tr_train
        rec["checks"]["val_ids_match"] = set(man.get("val_sample_ids", [])) == tr_val
        rec["checks"]["counts_84_24"] = (len(man.get("train_sample_ids", [])) == 84 and len(man.get("val_sample_ids", [])) == 24)
        rec["checks"]["test_not_in_val"] = not (set(man.get("val_sample_ids", [])) & tr_test)
        rec["checks"]["test_not_in_train"] = not (set(man.get("train_sample_ids", [])) & tr_test)
        ident = log.get("identity", {}) if isinstance(log, dict) else {}
        rec["checks"]["identity_seed"] = (man.get("seeds") == [seed]) if kind == "gradient" else True
        rec["checks"]["identity_model"] = (man.get("models") == [model])
        rec["checks"]["no_smoke_batches"] = (ident.get("max_train_batches_per_epoch") in (None,)) if kind == "gradient" else True

        # training evidence
        if kind == "gradient":
            ep = log.get("epochs_log", [])
            rec["checks"]["epochs_completed_60"] = log.get("completed_epochs") == 60 and len(ep) == 60
            rec["checks"]["physics_zero"] = log.get("physics_loss_weight") == 0.0
            rec["checks"]["loss_finite"] = all(math.isfinite(e.get("train_loss_mean", float("nan"))) for e in ep)
            rec["checks"]["selection_finite"] = all(math.isfinite(e.get("selection_metric", float("nan"))) for e in ep)
            rec["checks"]["best_epoch_valid"] = 0 <= log.get("best", {}).get("epoch_zero_based", -1) < log.get("completed_epochs", 0)
            rec["checks"]["best_matches_final"] = abs(log.get("best", {}).get("selection_metric", 1e9)
                                                      - log.get("final_val", {}).get("selection_metric", -1e9)) < 1e-6
            rec["checks"]["last_epoch_matches"] = log.get("last", {}).get("epoch_completed") == log.get("completed_epochs")
            rec["checks"]["pass_all"] = log.get("pass", {}).get("all") is True
            rec["checks"]["train_log_rows"] = (d / "train_log.csv").is_file() and \
                (sum(1 for _ in (d / "train_log.csv").open(encoding="utf-8")) - 1) == log.get("completed_epochs")
            bsha = sha(d / "checkpoints/best.pt") if (d/"checkpoints/best.pt").is_file() else None
            lsha = sha(d / "checkpoints/last.pt") if (d/"checkpoints/last.pt").is_file() else None
            rec["checks"]["best_sha_match"] = bsha == log.get("best", {}).get("checkpoint_sha256")
            rec["checks"]["best_differs_from_last"] = (bsha != lsha)
            rec["best"] = {"epoch_zero_based": log.get("best", {}).get("epoch_zero_based"),
                           "selection": log.get("best", {}).get("selection_metric"),
                           "source": log.get("best", {}).get("source")}
            rec["training"] = {"epochs_requested": log.get("epochs_requested"),
                               "completed": log.get("completed_epochs"),
                               "early_stopped": log.get("early_stopped"),
                               "early_stopping": log.get("early_stopping"),
                               "batch_size": log.get("batch_size"),
                               "n_params": log.get("n_params"),
                               "train_sec": log.get("timing", {}).get("train_sec"),
                               "final_inference_sec": log.get("timing", {}).get("final_inference_sec"),
                               "device": log.get("device_info", {}).get("device"),
                               "gpu_name": log.get("device_info", {}).get("gpu_name")}
        else:
            rec["checks"]["n_train_graphs_84"] = log.get("n_train_graphs") == 84
            rec["checks"]["n_val_graphs_24"] = log.get("n_val_graphs") == 24
            rec["checks"]["serialization"] = bool(log.get("serialization_reload_ok"))
            rec["checks"]["pass_all"] = log.get("pass", {}).get("all") is True
            rec["checks"]["histgb_early_stop_off"] = (log.get("hist_gb_early_stopping") is False) if model == "hist_gb" else True
            rec["training"] = {"deterministic": log.get("deterministic"),
                               "n_train_nodes": log.get("n_train_nodes"),
                               "n_train_beams": log.get("n_train_beams"),
                               "timing": log.get("timing"), "seed": log.get("seed")}
        # expected train entity totals from index (no subsampling)
        n_nodes = sum(int(row_by_id[s]["counts"]["nodes"]) for s in tr_train)
        n_beams = sum(int(row_by_id[s]["counts"]["beams"]) for s in tr_train)
        rec["expected_train_entities"] = {"nodes": n_nodes, "beams": n_beams}
        if kind == "ml":
            rec["checks"]["uses_all_train_entities"] = (log.get("n_train_nodes") == n_nodes and log.get("n_train_beams") == n_beams)

        # predictions + recompute
        valcov = {r["sample_id"] for r in preds} if isinstance(preds, list) else set()
        rec["checks"]["pred_ids_cover_val"] = valcov == tr_val
        rec["checks"]["pred_not_test"] = not (valcov & tr_test)
        # shape check
        shape_ok = isinstance(preds, list) and all(
            len(r.get("pred_disp", [])) == int(val_graphs[r["sample_id"]]["mesh_node"].x.shape[0])
            and len(r.get("pred_force", [])) == int(val_graphs[r["sample_id"]]["beam_element"].x.shape[0])
            and np.isfinite(np.asarray(r["pred_disp"])).all() and np.isfinite(np.asarray(r["pred_force"])).all()
            for r in preds if r.get("sample_id") in val_graphs)
        rec["checks"]["pred_shapes_finite"] = bool(shape_ok)
        metrics, perr = recompute(preds) if isinstance(preds, list) and preds else (None, {"error": "no predictions"})
        if metrics is None:
            rec["problems"].append(f"recompute failed: {perr}")
        else:
            rec["recomputed"] = {k: metrics[k] for k in ("selection", "selection_disp", "selection_force",
                                                         "entity_weighted_norm_mse_disp", "entity_weighted_norm_mse_force")}
            run_sel = (log.get("final_val", {}).get("selection_metric") if kind == "gradient" else log.get("selection_metric"))
            rec["checks"]["recomputed_matches_run"] = abs(metrics["selection"] - run_sel) < 1e-6
            rec["_components"] = metrics["components"]
            rec["_per_graph"] = metrics["per_graph"]
            rec["_D"] = None  # not stored
        rec["checks"]["all_ok"] = all(bool(v) for v in rec["checks"].values())
        if not rec["checks"]["all_ok"]:
            rec["problems"].extend([k for k, v in rec["checks"].items() if not v])
        report["runs"][f"{worker}/{run}"] = rec

    # aggregate seed stats + component/per-graph CSVs
    agg = {}
    by_model = defaultdict(list)
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        if kind == "gradient" and "recomputed" in rec:
            by_model[model].append(rec["recomputed"]["selection"])
    for model, vals in by_model.items():
        vals = [v for v in vals if v is not None]
        agg[model] = {"n": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "min": float(np.min(vals)), "max": float(np.max(vals)), "values": vals}
    report["seed_aggregate"] = agg

    # per-component CSV + seed-mean per component
    comp_rows = []
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        for c in rec.get("_components", []):
            comp_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed, "kind": kind, **c})
    with (out / "per_component_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys())); w.writeheader(); w.writerows(comp_rows)

    per_graph_rows = []
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        for g in rec.get("_per_graph", []):
            per_graph_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed, **g})
    with (out / "per_graph_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_graph_rows[0].keys())); w.writeheader(); w.writerows(per_graph_rows)

    # design/load-field grouping (per model-seed run)
    design_rows = []
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        pg = rec.get("_per_graph", [])
        for key_name, key in (("design_id", "design_id"), ("template_id", "template_id")):
            groups = defaultdict(list)
            for g in pg:
                groups[g[key]].append(g["selection"])
            for gid, vals in sorted(groups.items()):
                design_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed,
                                    "group_by": key_name, "group": gid, "n_graphs": len(vals),
                                    "selection_graph_equal_mean": float(np.mean(vals))})
    with (out / "design_field_group.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(design_rows[0].keys())); w.writeheader(); w.writerows(design_rows)

    # worst validation samples per run
    worst = []
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        pg = sorted(rec.get("_per_graph", []), key=lambda g: -g["selection"])[:5]
        for g in pg:
            worst.append({"run": f"{worker}/{run}", "model": model, "seed": seed, **g})
    with (out / "worst_val_samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(worst[0].keys())); w.writeheader(); w.writerows(worst)

    # magnitude bins (per run, per component; |y| deciles over val rows)
    mag_rows = []
    for worker, run, model, seed, kind in EXPECTED:
        rec = report["runs"].get(f"{worker}/{run}", {})
        if not rec.get("_per_graph"):
            continue
        # recompute errors per component with bins using stored preds? _D not stored -> recompute from preds
        preds = read_json(arch / "batch" / BATCH / worker / run / "val_predictions.json")
        by_id = {r["sample_id"]: r for r in preds}
        for task, names, units, which in (("disp", DISP_NAMES, DISP_UNITS, 0), ("force", FORCE_NAMES, FORCE_UNITS, 1)):
            truths = np.concatenate([val_label[s][which] for s in sorted(tr_val)], axis=0)
            preds_a = np.concatenate([np.asarray(by_id[s]["pred_disp" if which == 0 else "pred_force"], dtype=np.float64)
                                      for s in sorted(tr_val)], axis=0)
            err = np.abs(preds_a - truths)
            for c in range(truths.shape[1]):
                y = np.abs(truths[:, c]); e = err[:, c]
                qs = np.quantile(y, np.linspace(0, 1, 11))
                for b in range(10):
                    lo, hi = qs[b], qs[b + 1]
                    m = (y >= lo) & (y <= hi) if b == 9 else (y >= lo) & (y < hi)
                    if m.sum() == 0:
                        continue
                    mag_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed,
                                     "task": task, "component": names[c], "unit": units[c],
                                     "bin": b, "n": int(m.sum()), "mean_abs_true": float(y[m].mean()),
                                     "mae": float(e[m].mean())})
    with (out / "magnitude_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mag_rows[0].keys())); w.writeheader(); w.writerows(mag_rows)

    # drop non-serialisable internals
    for rec in report["runs"].values():
        rec.pop("_components", None); rec.pop("_per_graph", None); rec.pop("_D", None)
    (out / "review_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # selection curves
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for worker, run, model, seed, kind in EXPECTED:
            if kind != "gradient":
                continue
            ep = read_json(arch / "batch" / BATCH / worker / run / "run_log.json").get("epochs_log", [])
            ax.plot([e["epoch"] for e in ep], [e["selection_metric"] for e in ep],
                    label=f"{model} s{seed}", alpha=0.8)
        ax.set_xlabel("epoch"); ax.set_ylabel("graph-equal normalised MSE (selection)")
        ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title("Validation selection vs epoch (cd_fold_1)")
        fig.tight_layout(); fig.savefig(out / "train_val_curves.png", dpi=140); plt.close(fig)
        # per-component MAE (best run per model family: mlp seed17, hgt seed17, ridge, hist_gb)
        picks = [("t1_mlp/mlp_seed20261117", "mlp"), ("t2_hgt/hgt_global_seed20261117", "hgt_global"),
                 ("t1_ridge/ridge", "ridge"), ("t1_histgb/hist_gb", "hist_gb")]
        for task, names, units in (("disp", DISP_NAMES, DISP_UNITS), ("force", FORCE_NAMES, FORCE_UNITS)):
            fig, ax = plt.subplots(figsize=(9, 4))
            width = 0.2
            xs = np.arange(len(names))
            for j, (rkey, label) in enumerate(picks):
                rows = [r for r in comp_rows if r["run"] == rkey and r["task"] == task]
                vals = [next(r["mae"] for r in rows if r["component"] == n) for n in names]
                ax.bar(xs + (j - 1.5) * width, vals, width, label=label)
            ax.set_xticks(xs); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yscale("log"); ax.set_ylabel(f"MAE ({task} units, per component)")
            ax.legend(fontsize=7); ax.set_title(f"Per-component MAE — {task} (units differ per component)")
            fig.tight_layout(); fig.savefig(out / f"component_mae_{task}.png", dpi=140); plt.close(fig)
        # per-graph selection distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = []
        labels = []
        for rkey, label in [("t1_mlp/mlp_seed20261117", "mlp"), ("t2_hgt/hgt_global_seed20261117", "hgt"),
                            ("t1_ridge/ridge", "ridge"), ("t1_histgb/hist_gb", "hist_gb")]:
            vals = [g["selection"] for g in per_graph_rows if g["run"] == rkey]
            groups.append(vals); labels.append(label)
        ax.boxplot(groups, labels=labels)
        ax.set_ylabel("per-graph selection (normalised MSE)"); ax.set_title("Per-graph validation error distribution")
        fig.tight_layout(); fig.savefig(out / "per_graph_selection.png", dpi=140); plt.close(fig)
        report["plots"] = ["train_val_curves.png", "component_mae_disp.png", "component_mae_force.png",
                           "per_graph_selection.png"]
    except Exception as exc:  # noqa: BLE001
        report["plots_error"] = repr(exc)

    (out / "review_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": all(r.get("checks", {}).get("all_ok") for r in report["runs"].values()),
                      "sha256sums": report["sha256sums"], "frozen_input": report["frozen_input"],
                      "seed_aggregate": report.get("seed_aggregate"),
                      "runs": {k: {"all_ok": v.get("checks", {}).get("all_ok"), "best": v.get("best"),
                                   "recomputed": v.get("recomputed"), "problems": v.get("problems")}
                               for k, v in report["runs"].items()},
                      "out_dir": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

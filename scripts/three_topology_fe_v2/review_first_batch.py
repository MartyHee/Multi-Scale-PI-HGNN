#!/usr/bin/env python
"""Audit and independently recompute the FE v2 first GPU batch.

Reads the extracted artifact, verifies frozen identity and validation-only
predictions, recomputes the three validation slices from exported predictions
and source validation labels, and writes CSVs/plots.  No model inference, no
training, no test/blocked response access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.predictions import read_predictions  # noqa: E402
from src.data.three_topology_fe_v2.shared_cache import SharedCache  # noqa: E402
from src.train.three_topology_v2 import metrics as metrics_mod  # noqa: E402

BATCH = "fv2_first_20261125_01"
TOPOS = ("topology1", "topology2", "topology3")
SLICES = {"design_val": ("A", "val"), "load_val": ("B", "val"), "joint_val": ("C", "val")}
MODEL_ORDER = {"mlp": 0, "hgt_global": 1}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def run_root(ext: Path, row) -> Path:
    return ext / "batch" / row["topology"] / row["worker"] / row["run"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", default=None)
    ap.add_argument("--archive-sha256", default=None)
    a = ap.parse_args()
    ext = Path(a.extracted)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    audit = {"schema": "qt-fe-v2-first-batch-review-v1", "batch_id": BATCH,
             "archive": {}, "frozen": {}, "runs": [], "problems": []}
    if a.archive:
        actual = sha(Path(a.archive))
        audit["archive"] = {"path": str(a.archive), "sha256": actual,
                            "expected": a.archive_sha256,
                            "match": actual == a.archive_sha256 if a.archive_sha256 else None}
        if a.archive_sha256 and actual != a.archive_sha256:
            audit["problems"].append("archive sha256 mismatch")
    audit["sha256sums_entries"] = sum(1 for _ in (ext / "SHA256SUMS").read_text(encoding="utf-8").splitlines())

    local_manifest_path = MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2" / "manifests" / f"{BATCH}.json"
    artifact_manifest_path = ext / "manifests" / "batch_manifest.json"
    am = jload(artifact_manifest_path); lm = jload(local_manifest_path)
    audit["frozen"] = {
        "artifact_manifest_sha256": sha(artifact_manifest_path),
        "local_manifest_sha256": sha(local_manifest_path),
        "manifest_same": sha(artifact_manifest_path) == sha(local_manifest_path),
        "code_sha256": am["code_sha256"], "local_code_sha256": lm["code_sha256"],
        "code_same": am["code_sha256"] == lm["code_sha256"],
        "train_config_sha256": am["train_config_sha256"],
        "train_config_same": am["train_config_sha256"] == lm["train_config_sha256"],
        "input_identity_same": am["input_identity"] == lm["input_identity"],
        "tasks_same": am["tasks"] == lm["tasks"],
        "n_tasks": am["n_tasks"],
        "training_effective": am.get("training_effective"),
        "git_commit_server": am.get("git_commit"),
    }
    for k in ("manifest_same", "code_same", "train_config_same",
              "input_identity_same", "tasks_same"):
        if not audit["frozen"][k]:
            audit["problems"].append(f"frozen identity mismatch: {k}")
    # Artifact config/stats/split hashes vs local.
    pairs = {
        "configs/three_topology_fe_v2/feature_schema.json": MODEL_ROOT / "configs/three_topology_fe_v2/feature_schema.json",
        "configs/three_topology_fe_v2/formal_pipeline.yaml": MODEL_ROOT / "configs/three_topology_fe_v2/formal_pipeline.yaml",
        "processed/splits/protocol_v1.json": paths.PROCESSED_ROOT / "splits/protocol_v1.json",
        "processed/splits/role_ledger_v1.csv": paths.PROCESSED_ROOT / "splits/role_ledger_v1.csv",
        "processed/splits/split_manifest.json": paths.PROCESSED_ROOT / "splits/split_manifest.json",
    }
    for t in TOPOS:
        pairs[f"processed/stats/{t}.json"] = paths.PROCESSED_ROOT / "stats" / f"{t}.json"
    audit["frozen"]["artifact_files_match_local"] = {
        k: sha(ext / k) == sha(v) for k, v in pairs.items()}
    if not all(audit["frozen"]["artifact_files_match_local"].values()):
        audit["problems"].append("artifact config/stats/split hash mismatch vs local")

    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = jload(paths.PROCESSED_ROOT / "splits" / "protocol_v1.json")
    stats = {t: stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{t}.json") for t in TOPOS}
    std = {t: {"y_disp": [float(c["effective_std"]) for c in stats[t]["columns"]["y_disp"]],
               "y_force": [float(c["effective_std"]) for c in stats[t]["columns"]["y_force"]]}
           for t in TOPOS}
    expected_ids = {t: {name: protocol.ids_for(idx, prot, t, p, r)
                        for name, (p, r) in SLICES.items()} for t in TOPOS}
    expected_train = {t: protocol.ids_for(idx, prot, t, "A", "train") for t in TOPOS}
    caches = {t: SharedCache(idx, t) for t in TOPOS}
    label_cache = {}

    def labels(sid):
        if sid not in label_cache:
            raw = caches[idx.row(sid)["topology_token"]].assemble(sid)
            label_cache[sid] = {"node_y": raw["node_y"], "beam_y": raw["beam_y"]}
        return label_cache[sid]

    summary = jload(ext / "summary" / "batch_summary.json")
    rows = summary["rows"]
    if summary.get("overall_ok") is not True or len(rows) != 18:
        audit["problems"].append("batch summary is not 18/18 overall_ok")
    seen = set()
    metrics_rows, sample_rows, curve_rows, component_data = [], [], [], []
    for row in rows:
        key = (row["topology"], row["model"], str(row["seed"]))
        if key in seen:
            audit["problems"].append(f"duplicate run identity {key}")
        seen.add(key)
        d = run_root(ext, row)
        task = jload(d / "TASK.json"); sm = jload(d / "summary.json")
        log = jload(d / "run_log.json"); status = jload(d / "STATUS.json")
        checks = {}
        checks["status_done"] = status.get("state") == "done" and status.get("exit_code") == 0
        checks["task_identity"] = all([
            task.get("batch_id") == BATCH, task.get("topology") == row["topology"],
            task.get("worker") == row["worker"], task.get("model") == row["model"],
            str(task.get("seed")) == str(row["seed"])])
        checks["task_manifest_sha"] = task.get("manifest_sha256") == audit["frozen"]["artifact_manifest_sha256"]
        checks["task_config_sha"] = task.get("train_config_sha256") == am["train_config_sha256"]
        checks["train_graphs"] = sm.get("train_graphs") == 1360
        checks["val_graphs"] = sm.get("val_graphs") == {"design_val": 280, "load_val": 340, "joint_val": 70}
        checks["no_test"] = sm.get("test_constructed") is False and sm.get("protocol") == "protocol_v1"
        checks["physics_zero"] = sm.get("physics_loss_weight") == 0.0
        best_path = d / "checkpoints" / "best.pt"
        checks["best_file"] = best_path.is_file()
        best_sha = sha(best_path) if best_path.is_file() else None
        checks["best_sha"] = best_sha == sm.get("best", {}).get("checkpoint_sha256") == sm.get("final_eval_checkpoint_sha256")
        checks["final_from_best"] = sm.get("final_eval_source") == "best"
        checks["selection_is_best"] = abs(sm.get("selection_metric", 0) - sm.get("best", {}).get("selection_metric", 0)) < 1e-8
        checks["log_pass"] = log.get("pass", {}).get("all") is True
        checks["identity_stats"] = log.get("identity", {}).get("stats_sha256") == sha(paths.PROCESSED_ROOT / "stats" / f"{row['topology']}.json")
        checks["identity_train"] = log.get("identity", {}).get("train_sample_list_sha256") == hash_json(sorted(expected_train[row["topology"]]))
        expected_val = [s for name in SLICES for s in expected_ids[row["topology"]][name]]
        checks["identity_val"] = log.get("identity", {}).get("val_sample_list_sha256") == hash_json(sorted(expected_val))
        checks["identity_data"] = (log.get("identity", {}).get("samples_jsonl_sha256") == am["input_identity"]["samples_jsonl_sha256"]
                                   and log.get("identity", {}).get("dataset_manifest_sha256") == am["input_identity"]["dataset_manifest_sha256"])
        checks["identity_read_path"] = log.get("identity", {}).get("read_path") == "shared_v1"
        checks["identity_cache"] = (log.get("identity", {}).get("design_cache_sha256") == am["input_identity"]["design_cache_sha256"][row["topology"]])
        # train log / early stopping
        with (d / "train_log.csv").open("r", newline="", encoding="utf-8") as fh:
            tl = list(csv.DictReader(fh))
        checks["train_log_rows"] = len(tl) == sm.get("completed_epochs")
        checks["epoch_budget"] = sm.get("completed_epochs", 0) <= am["training_effective"]["epochs"]
        if tl:
            last = tl[-1]
            if sm.get("stop_reason") == "early_stop":
                checks["early_stop_condition"] = int(last["no_improve"]) >= int(am["training_effective"]["early_stopping_patience"])
            else:
                checks["early_stop_condition"] = sm.get("stop_reason") == "epoch_budget_reached"
            checks["best_epoch_in_log"] = any(int(r["epoch"]) == sm["best"]["epoch_zero_based"] for r in tl)
        else:
            checks["early_stop_condition"] = False
            checks["best_epoch_in_log"] = False
        # prediction slice sets and independent recompute
        recomputed = {}
        print(f"audit {row['topology']}/{row['model']}/{row['seed']}", flush=True)
        for name, (p, r) in SLICES.items():
            pred_path = d / f"val_predictions_{name}.npz"
            if not pred_path.is_file():
                pred_path = d / f"val_predictions_{name}.json"
            preds = read_predictions(pred_path)
            ids = [x["sample_id"] for x in preds]
            checks[f"ids_{name}"] = ids == expected_ids[row["topology"]][name]
            if any("test" in prot["roles"][sid].values() for sid in ids):
                checks[f"ids_{name}"] = False
                audit["problems"].append(f"{row['topology']}/{row['model']}/{row['seed']}/{name}: test-role sample")
                continue
            pd, pf, td, tf = [], [], [], []
            gd, gb = [], []
            for gi, rec in enumerate(preds):
                raw = labels(rec["sample_id"])
                if rec["node_count"] != len(raw["node_y"]) or rec["beam_count"] != len(raw["beam_y"]):
                    checks[f"counts_{name}"] = False
                pd.append(np.asarray(rec["pred_disp"], dtype=np.float64))
                pf.append(np.asarray(rec["pred_force"], dtype=np.float64))
                td.append(np.asarray(raw["node_y"], dtype=np.float64))
                tf.append(np.asarray(raw["beam_y"], dtype=np.float64))
                gd.append(np.full(len(raw["node_y"]), gi, dtype=np.int64))
                gb.append(np.full(len(raw["beam_y"]), gi, dtype=np.int64))
            pd, pf, td, tf = map(np.concatenate, (pd, pf, td, tf))
            gd, gb = np.concatenate(gd), np.concatenate(gb)
            topo = np.full(len(gd), TOPOS.index(row["topology"]) + 1, dtype=np.int64)
            topob = np.full(len(gb), TOPOS.index(row["topology"]) + 1, dtype=np.int64)
            acc = metrics_mod.MetricsAccumulator()
            acc.update(pd, td, gd, topo, pf, tf, gb, topob)
            met = acc.finalize(std[row["topology"]])
            stored = jload(d / "slice_metrics.json")["slices"][name]
            checks[f"selection_{name}"] = abs(met["selection_metric"] - stored["selection_metric"]) < 1e-8
            checks[f"entity_{name}"] = abs(met["disp"]["entity_weighted_norm_mse"] - stored["metrics"]["disp"]["entity_weighted_norm_mse"]) < 1e-8
            recomputed[name] = met
            for task in ("disp", "force"):
                component_data.append({
                    "topology": row["topology"], "model": row["model"],
                    "seed": row["seed"], "slice": name, "task": task,
                    "n": int(met[task]["n"]),
                    "mae": list(met[task]["entity_weighted_mae"]),
                    "rmse": list(met[task]["entity_weighted_rmse"]),
                })
            # per-sample selection + physical errors
            for gi, rec in enumerate(preds):
                raw = labels(rec["sample_id"])
                pdi = np.asarray(rec["pred_disp"], dtype=np.float64); tdi = np.asarray(raw["node_y"])
                pfi = np.asarray(rec["pred_force"], dtype=np.float64); tfi = np.asarray(raw["beam_y"])
                sd = np.asarray(std[row["topology"]]["y_disp"]); sf = np.asarray(std[row["topology"]]["y_force"])
                mse_d = ((pdi - tdi) ** 2).mean(axis=0); mse_f = ((pfi - tfi) ** 2).mean(axis=0)
                sel = 0.5 * (float(np.mean(mse_d / sd ** 2)) + float(np.mean(mse_f / sf ** 2)))
                mae_d = np.abs(pdi - tdi).mean(axis=0); mae_f = np.abs(pfi - tfi).mean(axis=0)
                meta = idx.row(rec["sample_id"])
                fm = idx.field_meta(meta["topology_token"], meta["field_id"])
                sample_rows.append({
                    "topology": row["topology"], "model": row["model"], "seed": row["seed"],
                    "slice": name, "sample_id": rec["sample_id"],
                    "design_id": meta["design_id"], "field_id": meta["field_id"],
                    "load_family": fm.get("family"), "selection_per_sample": sel,
                    **{f"disp_mae_{metrics_mod.DISP_NAMES[i]}": float(mae_d[i]) for i in range(6)},
                    **{f"force_mae_{metrics_mod.FORCE_NAMES[i]}": float(mae_f[i]) for i in range(12)},
                })
        total = float(np.mean([recomputed[n]["selection_metric"] for n in SLICES]))
        checks["total_selection"] = abs(total - sm["selection_metric"]) < 1e-8
        if not all(checks.values()):
            audit["problems"].append({"run": key, "failed_checks": [k for k, v in checks.items() if not v]})
        mrow = {
            "topology": row["topology"], "model": row["model"], "seed": row["seed"],
            "selection": sm["selection_metric"],
            "design_val": sm["slice_selection"]["design_val"],
            "load_val": sm["slice_selection"]["load_val"],
            "joint_val": sm["slice_selection"]["joint_val"],
            "entity_weighted_mean": float(np.mean([sm["slice_entity_weighted_norm_mse"][n] for n in SLICES])),
            "completed_epochs": sm["completed_epochs"], "best_epoch_zero_based": sm["best"]["epoch_zero_based"],
            "stop_reason": sm["stop_reason"], "train_actual_updates": sm["train_actual_updates"],
            "train_sec": log.get("timing", {}).get("train_sec"),
            "final_inference_sec": log.get("timing", {}).get("final_inference_sec"),
            "mem_mb_final": log.get("mem_mb_final"), "gpu": log.get("device_info", {}).get("gpu_name"),
            "best_sha256": best_sha, "recomputed_total": total,
        }
        for name in SLICES:
            mrow[f"recomputed_selection_{name}"] = recomputed[name]["selection_metric"]
        metrics_rows.append(mrow)
        audit["runs"].append({"run": list(key), "checks": checks, "run_dir": str(d)})
        for rec in tl:
            curve_rows.append({
                "topology": row["topology"], "model": row["model"], "seed": row["seed"],
                **{k: rec.get(k) for k in ("epoch", "train_loss_mean", "selection_metric",
                                            "best_val", "improved", "no_improve", "train_sec",
                                            "val_sec", "mem_mb", "utc")},
            })

    write_csv(out / "metrics_by_run.csv", metrics_rows)
    write_csv(out / "epoch_curves.csv", curve_rows)
    write_csv(out / "sample_metrics.csv", sample_rows)

    # Summary by template/model (seed mean/std).
    summary_rows, component_rows = [], []
    for t in TOPOS:
        for m in ("mlp", "hgt_global"):
            rs = [r for r in metrics_rows if r["topology"] == t and r["model"] == m]
            if not rs:
                continue
            vals = np.array([r["selection"] for r in rs], dtype=np.float64)
            srow = {"topology": t, "model": m, "n_seeds": len(rs),
                    "selection_mean": float(vals.mean()), "selection_std": float(vals.std(ddof=1)) if len(rs) > 1 else 0.0,
                    "selection_min": float(vals.min()), "selection_max": float(vals.max())}
            for name in SLICES:
                v = np.array([r[name] for r in rs], dtype=np.float64)
                srow[f"{name}_mean"] = float(v.mean()); srow[f"{name}_std"] = float(v.std(ddof=1)) if len(rs) > 1 else 0.0
            for k in ("completed_epochs", "best_epoch_zero_based", "train_sec", "final_inference_sec"):
                v = np.array([r[k] for r in rs], dtype=np.float64)
                srow[f"{k}_mean"] = float(v.mean()); srow[f"{k}_std"] = float(v.std(ddof=1)) if len(rs) > 1 else 0.0
            summary_rows.append(srow)
    write_csv(out / "model_summary.csv", summary_rows)

    # Component summary: physical MAE/RMSE per component, row-weighted over
    # slices and averaged over seeds.
    comp_rows = []
    for t in TOPOS:
        for m in ("mlp", "hgt_global"):
            for task, names, units in (("disp", metrics_mod.DISP_NAMES, metrics_mod.DISP_UNITS),
                                       ("force", metrics_mod.FORCE_NAMES, metrics_mod.FORCE_UNITS)):
                maes, rmses = [], []
                for r in [x for x in rows if x["topology"] == t and x["model"] == m]:
                    d = [c for c in component_data if c["topology"] == t and c["model"] == m
                         and c["seed"] == r["seed"] and c["task"] == task]
                    if not d:
                        continue
                    n = np.array([c["n"] for c in d], dtype=np.float64)
                    mae = np.average(np.array([c["mae"] for c in d], dtype=np.float64),
                                     axis=0, weights=n)
                    rmse = np.average(np.array([c["rmse"] for c in d], dtype=np.float64),
                                      axis=0, weights=n)
                    maes.append(mae); rmses.append(rmse)
                maes = np.asarray(maes); rmses = np.asarray(rmses)
                for i, (n, u) in enumerate(zip(names, units)):
                    comp_rows.append({"topology": t, "model": m, "task": task,
                                      "component": n, "unit": u,
                                      "mae_mean_over_seeds": float(maes[:, i].mean()),
                                      "mae_std_over_seeds": float(maes[:, i].std(ddof=1)) if len(maes) > 1 else 0.0,
                                      "rmse_mean_over_seeds": float(rmses[:, i].mean())})
    write_csv(out / "component_summary.csv", comp_rows)

    # Design / field layer: mean per-sample selection across seeds, per template/model.
    for layer, keyname, outname in (("design_id", "design_id", "design_layer.csv"),
                                    ("field_id", "field_id", "field_layer.csv"),
                                    ("load_family", "load_family", "load_family_layer.csv")):
        rows2 = []
        for t in TOPOS:
            for m in ("mlp", "hgt_global"):
                groups = {}
                for r in sample_rows:
                    if r["topology"] != t or r["model"] != m:
                        continue
                    groups.setdefault(r[keyname], []).append(r["selection_per_sample"])
                for g, vals in sorted(groups.items()):
                    rows2.append({"topology": t, "model": m, keyname: g,
                                  "n_seed_rows": len(vals), "selection_mean": float(np.mean(vals)),
                                  "selection_max": float(np.max(vals))})
        write_csv(out / outname, rows2)

    # Worst samples per run (top 5 by per-sample selection).
    import itertools
    worst = []
    ordered = sorted(sample_rows, key=lambda r: (r["topology"], r["model"], r["seed"]))
    for (t, m, s), group in itertools.groupby(
            ordered, key=lambda r: (r["topology"], r["model"], r["seed"])):
        items = sorted(list(group), key=lambda r: -r["selection_per_sample"])[:5]
        worst.extend(items)
    write_csv(out / "worst_samples.csv", worst)

    # Curves CSV already; plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        curves = pd.read_csv(out / "epoch_curves.csv")
        for t in TOPOS:
            for m in ("mlp", "hgt_global"):
                sub = curves[(curves.topology == t) & (curves.model == m)]
                if sub.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                for seed, g in sub.groupby("seed"):
                    axes[0].plot(g.epoch, g.train_loss_mean, label=str(seed))
                    axes[1].plot(g.epoch, g.selection_metric, label=str(seed))
                axes[0].set_yscale("log"); axes[1].set_yscale("log")
                axes[0].set_title(f"{t}/{m} train loss (normalized)"); axes[0].set_xlabel("epoch")
                axes[1].set_title(f"{t}/{m} val selection (normalized MSE)"); axes[1].set_xlabel("epoch")
                for ax in axes:
                    ax.grid(alpha=0.3); ax.legend(fontsize=8)
                fig.tight_layout()
                for ext in ("png", "pdf"):
                    fig.savefig(out / "plots" / f"curves_{t}_{m}.{ext}", dpi=150)
                plt.close(fig)
        # selection summary bar with std
        ms = pd.read_csv(out / "model_summary.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r.topology}\n{r.model}" for r in ms.itertuples()]
        ax.bar(range(len(ms)), ms.selection_mean, yerr=ms.selection_std, capsize=3)
        ax.set_xticks(range(len(ms)), labels, fontsize=8)
        ax.set_ylabel("three-slice selection (lower better)")
        ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / "plots" / f"selection_summary.{ext}", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        audit["plot_error"] = repr(exc)

    audit["all_checks_ok"] = not audit["problems"]
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_checks_ok": audit["all_checks_ok"], "problems": audit["problems"],
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0 if audit["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

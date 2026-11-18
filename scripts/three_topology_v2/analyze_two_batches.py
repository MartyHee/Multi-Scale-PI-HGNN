#!/usr/bin/env python
"""Joint first/second batch comparison + 60-epoch convergence diagnostics + plots.

Reads only extracted archives, frozen split/stats and validation labels.
Produces the 10-model table, per-component/per-graph tables, convergence
statistics, budget-sensitivity evidence, raw curve CSVs and paper-candidate
figures (PNG+PDF). No inference/training/outer-test scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from scripts.three_topology_v2.analysis_common import (BATCHES, BATCH_OF, GRADIENT,  # noqa: E402
                                                       MODEL_LABEL, MODEL_ORDER, load_batch_runs)

OUT = MODEL_ROOT / "outputs" / "three_topology_v2" / "second_batch_review_20261118"
COMMIT_FIRST = BATCHES["fb_20261117_01"]["commit"]
COMMIT_SECOND = BATCHES["sb_20261118_01"]["commit"]
KEY_FILES = [
    "src/train/three_topology_v2/engine.py",
    "src/train/three_topology_v2/metrics.py",
    "src/train/three_topology_v2/ml_baselines.py",
    "src/data/three_topology_v2/graph_io.py",
    "src/data/three_topology_v2/dataset.py",
    "src/data/three_topology_v2/stats.py",
    "src/data/three_topology_v2/splits.py",
    "src/models/baselines/mlp_baseline.py",
    "src/models/baselines/hgt_global.py",
    "src/models/baselines/hgt_baseline.py",
    "src/models/baselines/hetero_rgcn.py",
    "src/models/baselines/homogeneous_gcn.py",
    "src/models/baselines/homogeneous_gat.py",
    "src/models/baselines/homogeneous_extra.py",
    "src/models/baselines/hetero_to_homo_adapter.py",
    "configs/three_topology_v2/formal_baselines.yaml",
    "scripts/three_topology_v2/train_formal.py",
]


def git(*args):
    return subprocess.check_output(["git", "-C", str(MODEL_ROOT)] + list(args), text=True, encoding="utf-8", errors="replace").strip()


def compare_commits():
    diff = git("diff", "--name-only", f"{COMMIT_FIRST}..{COMMIT_SECOND}", "--",
               "src", "configs/three_topology_v2/formal_baselines.yaml",
               "scripts/three_topology_v2/train_formal.py")
    files = []
    for rel in KEY_FILES:
        try:
            a = git("show", f"{COMMIT_FIRST}:{rel}")
            b = git("show", f"{COMMIT_SECOND}:{rel}")
            files.append({"file": rel, "same": a == b})
        except subprocess.CalledProcessError:
            files.append({"file": rel, "same": False, "error": "not found"})
    return {"commit_first": COMMIT_FIRST, "commit_second": COMMIT_SECOND,
            "src_config_script_diff": diff.splitlines(), "key_files": files,
            "engine_model_metric_same": not diff and all(f["same"] for f in files)}


def agg_model(runs):
    rows = []
    for model in MODEL_ORDER:
        rs = [r for r in runs if r["model"] == model and r.get("metrics")]
        if not rs:
            continue
        vals = np.array([r["metrics"]["selection"] for r in rs], dtype=float)
        rows.append({
            "model": model, "label": MODEL_LABEL[model], "batch_id": BATCH_OF[model],
            "n_seeds": len(vals), "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()),
            "values": [float(v) for v in vals],
            "seeds": [r["seed"] for r in rs],
            "n_params": next((r["log"].get("n_params") for r in rs if r["log"].get("n_params")), None),
            "selection_disp_mean": float(np.mean([r["metrics"]["selection_disp"] for r in rs])),
            "selection_force_mean": float(np.mean([r["metrics"]["selection_force"] for r in rs])),
        })
    return rows


def curve_rows(runs):
    raw, mean = [], []
    by_model = defaultdict(list)
    for r in runs:
        if r["model"] not in GRADIENT:
            continue
        ep = r["log"].get("epochs_log") or []
        if not ep:
            continue
        t = 0.0
        for e in ep:
            t += float(e.get("train_sec") or 0) + float(e.get("val_sec") or 0)
            raw.append({"batch_id": r["batch_id"], "model": r["model"], "seed": r["seed"],
                        "epoch1": int(e["epoch"]) + 1,
                        "train_loss_mean": float(e["train_loss_mean"]),
                        "val_selection": float(e["selection_metric"]),
                        "best_so_far": float(e["best_val"]),
                        "train_sec": float(e.get("train_sec") or 0),
                        "val_sec": float(e.get("val_sec") or 0),
                        "cum_time_sec": t})
        by_model[r["model"]].append(ep)
    for model, eps in by_model.items():
        maxn = max(len(e) for e in eps)
        for i in range(maxn):
            tr = [float(e[i]["train_loss_mean"]) for e in eps if i < len(e)]
            va = [float(e[i]["selection_metric"]) for e in eps if i < len(e)]
            bv = [float(e[i]["best_val"]) for e in eps if i < len(e)]
            mean.append({"model": model, "epoch1": i + 1, "n_seeds": len(va),
                         "train_loss_mean": float(np.mean(tr)),
                         "val_selection_mean": float(np.mean(va)),
                         "val_selection_std": float(np.std(va, ddof=1)) if len(va) > 1 else 0.0,
                         "best_so_far_mean": float(np.mean(bv))})
    return raw, mean


def convergence(runs):
    rows = []
    for r in runs:
        if r["model"] not in GRADIENT:
            continue
        ep = r["log"].get("epochs_log") or []
        if not ep:
            continue
        x = np.array([int(e["epoch"]) for e in ep], dtype=float)
        tr = np.array([float(e["train_loss_mean"]) for e in ep])
        va = np.array([float(e["selection_metric"]) for e in ep])
        bv = np.array([float(e["best_val"]) for e in ep])
        n = len(ep)
        def best_at(k):
            return float(va[:min(k, n)].min()) if n else None
        def slope(y, k):
            if len(y) < 2:
                return None
            yy = y[-k:]
            return float(np.polyfit(np.arange(len(yy)), yy, 1)[0])
        v40, v50, v60 = best_at(40), best_at(50), best_at(60)
        last10 = min(10, n)
        before10 = float(va[:-last10].min()) if n > last10 else float(va[:1].min())
        last10_best = float(va[-last10:].min())
        rel_last10 = (before10 - last10_best) / before10 if before10 else None
        rows.append({
            "batch_id": r["batch_id"], "run": f"{r['worker']}/{r['run']}",
            "model": r["model"], "seed": r["seed"], "completed_epochs": n,
            "early_stopped": bool(r["log"].get("early_stopped")),
            "best_epoch_zero_based": int(r["log"]["best"]["epoch_zero_based"]),
            "best_epoch1": int(r["log"]["best"]["epoch_zero_based"]) + 1,
            "best_selection": float(r["log"]["best"]["selection_metric"]),
            "best_at_40": v40, "best_at_50": v50, "best_at_60": v60,
            "improve_40_to_60_rel": (v40 - v60) / v40 if v40 else None,
            "improve_50_to_60_rel": (v50 - v60) / v50 if v50 else None,
            "last10_val_min": last10_best,
            "last10_improve_rel": rel_last10,
            "val_slope_last10": slope(va, 10), "val_slope_last20": slope(va, 20),
            "train_slope_last10": slope(tr, 10), "train_slope_last20": slope(tr, 20),
            "best_in_last10": (int(r["log"]["best"]["epoch_zero_based"]) >= n - 10),
            "final_val_improved_over_last10_min": bool(abs(float(r["log"]["final_val"]["selection_metric"]) - float(bv[-1])) < 1e-12),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for b in BATCHES:
        rs, ctx = load_batch_runs(b)
        runs.extend(rs)
    runs = [r for r in runs if r.get("metrics")]
    comp = compare_commits()
    models = agg_model(runs)
    raw, mean = curve_rows(runs)
    conv = convergence(runs)
    (out / "comparability.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(name, rows):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    write_csv("model_results.csv", [{k: v for k, v in m.items() if k not in ("values", "seeds")} | {"selection_values": m["values"]} for m in models])
    run_rows = []
    for r in runs:
        run_rows.append({"batch_id": r["batch_id"], "worker": r["worker"], "run": r["run"],
                         "model": r["model"], "seed": r["seed"],
                         "selection": r["metrics"]["selection"],
                         "selection_disp": r["metrics"]["selection_disp"],
                         "selection_force": r["metrics"]["selection_force"],
                         "best_epoch_zero_based": r["log"].get("best", {}).get("epoch_zero_based"),
                         "completed_epochs": r["log"].get("completed_epochs"),
                         "early_stopped": r["log"].get("early_stopped"),
                         "n_params": r["log"].get("n_params"),
                         "device": r["status"].get("device"),
                         "external_cvd": r["status"].get("external_cuda_visible_devices")})
    write_csv("all_runs.csv", run_rows)
    write_csv("curves_raw.csv", raw)
    write_csv("curves_mean.csv", mean)
    write_csv("convergence_summary.csv", conv)
    comp_rows, graph_rows, group_rows = [], [], []
    for r in runs:
        for c in r["metrics"]["components"]:
            comp_rows.append({"batch_id": r["batch_id"], "run": f"{r['worker']}/{r['run']}",
                              "model": r["model"], "seed": r["seed"], **c})
        for g in r["metrics"]["per_graph"]:
            graph_rows.append({"batch_id": r["batch_id"], "run": f"{r['worker']}/{r['run']}",
                               "model": r["model"], "seed": r["seed"], **g})
        groups = defaultdict(list)
        for g in r["metrics"]["per_graph"]:
            for key in ("design_id", "template_id"):
                groups[(key, g[key])].append(g["selection"])
        for (key, val), sels in sorted(groups.items()):
            group_rows.append({"batch_id": r["batch_id"], "run": f"{r['worker']}/{r['run']}",
                               "model": r["model"], "seed": r["seed"], "group_by": key,
                               "group": val, "n_graphs": len(sels),
                               "selection_graph_equal_mean": float(np.mean(sels))})
    write_csv("component_metrics_all.csv", comp_rows)
    write_csv("per_graph_metrics_all.csv", graph_rows)
    write_csv("design_field_group_all.csv", group_rows)

    # model-level convergence + exploratory threshold table
    conv_by_model = defaultdict(list)
    for c in conv:
        conv_by_model[c["model"]].append(c)
    model_conv = []
    for model in MODEL_ORDER:
        rs = conv_by_model.get(model, [])
        if not rs:
            continue
        model_conv.append({
            "model": model, "label": MODEL_LABEL[model], "n_seeds": len(rs),
            "best_epoch1_min": min(c["best_epoch1"] for c in rs),
            "best_epoch1_median": float(np.median([c["best_epoch1"] for c in rs])),
            "best_epoch1_max": max(c["best_epoch1"] for c in rs),
            "seeds_best_in_last10": sum(c["best_in_last10"] for c in rs),
            "median_improve_40_60_rel": float(np.median([c["improve_40_to_60_rel"] for c in rs])),
            "median_improve_50_60_rel": float(np.median([c["improve_50_to_60_rel"] for c in rs])),
            "median_last10_improve_rel": float(np.median([c["last10_improve_rel"] for c in rs])),
            "seeds_last10_val_slope_negative": sum((c["val_slope_last10"] or 0) < 0 for c in rs),
            "seeds_train_slope_down": sum((c["train_slope_last10"] or 0) < 0 for c in rs),
            "early_stopped_seeds": sum(c["early_stopped"] for c in rs),
        })
    write_csv("convergence_model_summary.csv", model_conv)

    # exploratory: first epoch where seed-mean best-so-far is within 2% of its 60-epoch value
    thr_rows = []
    for model in GRADIENT:
        rs = [r for r in runs if r["model"] == model]
        if not rs:
            continue
        maxn = max(len(r["log"]["epochs_log"]) for r in rs)
        curve = []
        for i in range(maxn):
            vals = [float(r["log"]["epochs_log"][i]["best_val"]) for r in rs if i < len(r["log"]["epochs_log"])]
            curve.append(float(np.mean(vals)))
        final = curve[-1]
        row = {"model": model, "label": MODEL_LABEL[model], "final_seed_mean_best": final,
               "rule": "first epoch where seed-mean best-so-far <= (1+tol) * final 60-epoch value; exploratory"}
        for tol in (0.02, 0.05, 0.10):
            target = final * (1 + tol)
            hit = next((i + 1 for i, v in enumerate(curve) if v <= target), None)
            row[f"target_{int(tol*100)}pct"] = target
            row[f"first_epoch_{int(tol*100)}pct"] = hit
            row[f"reached_{int(tol*100)}pct_within_60"] = hit is not None
        thr_rows.append(row)
    write_csv("budget_thresholds_exploratory.csv", thr_rows)

    summary = {"schema": "qt-two-batch-analysis-v1", "date": "2026-11-18",
               "n_runs": len(runs), "models": models, "comparability": comp,
               "convergence_model_summary": model_conv,
               "thresholds_exploratory": thr_rows,
               "aggregation_note": ("train_loss_mean is mean per-batch normalized MSE over the epoch; "
                                    "val selection is graph-equal/task-equal/component-equal normalized MSE. "
                                    "They are different aggregations; their difference is not a clean "
                                    "generalization gap."),
               "time_note": "cum_time_sec sums per-epoch train_sec+val_sec; shared-server concurrency "
                            "means wall-clock is descriptive only, not a controlled speed comparison.",
               "scope_note": "single topology (topology1), single fold (cd_fold_1), validation side; "
                             "three seeds are repeated runs, not independent datasets."}
    (out / "joint_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"mlp": "C0", "gcn": "C1", "graphsage": "C2", "gat": "C3",
                  "gin": "C4", "rgcn": "C5", "hgt": "C6", "hgt_global": "k"}
        gradient_models = [m for m in MODEL_ORDER if m in GRADIENT]
        by_model = defaultdict(list)
        for r in runs:
            if r["model"] in GRADIENT:
                by_model[r["model"]].append(r)

        def small_multiples(value_key, best_key, title, fname, logy=True, ylabel="normalized MSE"):
            fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
            for ax, model in zip(axes.ravel(), gradient_models):
                rs = by_model.get(model, [])
                maxn = max(len(r["log"]["epochs_log"]) for r in rs)
                for r in rs:
                    ep = r["log"]["epochs_log"]
                    x = [e["epoch"] + 1 for e in ep]
                    y = [float(e[value_key]) for e in ep]
                    ax.plot(x, y, color=colors[model], alpha=0.28, lw=0.9)
                xs, ys = [], []
                for i in range(maxn):
                    vals = [float(r["log"]["epochs_log"][i][value_key]) for r in rs if i < len(r["log"]["epochs_log"])]
                    xs.append(i + 1); ys.append(float(np.mean(vals)))
                ax.plot(xs, ys, color=colors[model], lw=1.9, label="seed mean")
                if logy:
                    ax.set_yscale("log")
                ax.set_title(f"{MODEL_LABEL[model]} (n={len(rs)})")
                ax.set_xlabel("epoch (1-based)")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25, lw=0.4)
            fig.suptitle(title)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"{fname}.{ext}", dpi=200)
            plt.close(fig)

        small_multiples("train_loss_mean", None,
                        "Training loss vs epoch (thin: individual seeds; bold: seed mean)",
                        "fig_train_loss", logy=True, ylabel="train loss (mean per-batch normalized MSE)")
        small_multiples("selection_metric", None,
                        "Validation selection vs epoch (thin: seeds; bold: seed mean; lower is better)",
                        "fig_val_selection", logy=True, ylabel="validation selection (normalized MSE)")
        small_multiples("best_val", None,
                        "Best validation selection so far vs epoch (seed mean)",
                        "fig_best_so_far", logy=True, ylabel="best validation selection so far")

        # validation vs cumulative time (seed mean)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for model in gradient_models:
            rs = [r for r in runs if r["model"] == model]
            maxn = max(len(r["log"]["epochs_log"]) for r in rs)
            xs, ys = [], []
            for i in range(maxn):
                times = [sum(float(e.get("train_sec") or 0) + float(e.get("val_sec") or 0)
                             for e in r["log"]["epochs_log"][:i + 1]) / 60.0
                         for r in rs if i < len(r["log"]["epochs_log"])]
                vals = [float(r["log"]["epochs_log"][i]["selection_metric"]) for r in rs if i < len(r["log"]["epochs_log"])]
                if times:
                    xs.append(float(np.mean(times))); ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, color=colors[model], lw=1.4, label=MODEL_LABEL[model])
        ax.set_xlabel("cumulative logged train+val time (minutes; descriptive, shared server)")
        ax.set_ylabel("validation selection (normalized MSE)")
        ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=7, ncol=2)
        ax.set_title("Validation selection vs cumulative logged time (seed mean)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_val_vs_time.{ext}", dpi=200)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        summary["plot_error"] = repr(exc)
        (out / "joint_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"n_runs": len(runs), "models": [
        {k: m[k] for k in ("model", "n_seeds", "mean", "std", "min", "max", "n_params")} for m in models],
        "comparability_same": comp["engine_model_metric_same"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

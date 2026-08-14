# -*- coding: utf-8 -*-
"""
analyze_stage3b_round.py — Stage Paper-3B result analysis.

Reads the Stage-3B runs from
  remote_artifacts/paper_upgrade_stage3b/unpacked/
(auto-discovered by job-name prefix, timestamp suffix ignored) and produces:

  CSVs  : stage3b_artifact_inventory.csv
          stage3b_metrics_raw.csv
          stage3b_coord_vs_random_3seed.csv   (mean±std per seed set + deltas)
          stage3b_second_wave_triage.csv      (hgt_global/edge_aware_mpnn seed42)
          stage3b_decision_rules.csv          (mechanical A/B/C application)
  Plots : stage3b_coord_vs_random_3seed.png
          stage3b_second_wave_triage.png
  Doc   : docs/paper_upgrade_stage3b_round_metrics_table.md

Expected run dirs (job-name prefixes; timestamps auto-stripped):
  paper_ms_hgt_random_anchor_seed2026   (random, seed 2)
  paper_ms_hgt_random_anchor_seed3407   (random, seed 3)
  paper_ms_hgt_gated_seed3407           (coordinate, seed 3)
  paper_hgt_global_mesh_seed42          (second-wave triage)
  paper_edge_aware_mpnn_seed42          (second-wave triage)

Historical references (seed42): HGT-3 0.9765/0.9054/0.9893/0.0676/744279;
MS-HGT coordinate 0.9952/0.9925/0.9928/0.0519/893527.

Usage (from project root, local llm interpreter):
  KMP_DUPLICATE_LIB_OK=TRUE D:/CodeData/software/Anaconda/Anaconda3/envs/llm/python.exe \
      scripts/paper_check/analyze_stage3b_round.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import yaml
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNPACKED = PROJECT_ROOT / "remote_artifacts" / "paper_upgrade_stage3b" / "unpacked"
TS = time.strftime("%Y%m%d%H%M%S")
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "paper_upgrade_stage3b" / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)

# job-name prefixes -> (model, seed, group)
JOB_SPECS = [
    ("paper_ms_hgt_random_anchor_seed2026", "ms_hgt_random_anchor", 2026, "random"),
    ("paper_ms_hgt_random_anchor_seed3407", "ms_hgt_random_anchor", 3407, "random"),
    ("paper_ms_hgt_gated_seed3407",         "ms_hgt",               3407, "coordinate"),
    ("paper_hgt_global_mesh_seed42",         "hgt_global",           42,  "global"),
    ("paper_edge_aware_mpnn_seed42",         "edge_aware_mpnn",      42,  "mpnn"),
]

# Historical refs (seed42, from Stage 2 / Stage 4 / Stage 3A)
REF = {
    "HGT-3":            dict(disp_r2=0.9765, dy_r2=0.9054, force_r2=0.9893,
                             relmae=0.0676, params=744279),
    "MS-HGT coordinate": dict(disp_r2=0.9952, dy_r2=0.9925, force_r2=0.9928,
                              relmae=0.0519, params=893527),
}
# Historical seed-2 (first wave) anchors for the 3-seed comparison
REF_SEED = {
    "coordinate": {2026: dict(disp_r2=0.9945, dy_r2=0.9916, force_r2=0.9928,
                              relmae=0.0541),  # paper_ms_hgt_gated_seed2026
                   },
    "random":     {42: dict(disp_r2=0.9952, dy_r2=0.9907, force_r2=0.9933,
                            relmae=0.0511),   # paper_ms_hgt_random_anchor_seed42
                   },
}

REQUIRED_FILES = ["metrics_summary.json", "config_resolved.yaml", "train_log.csv",
                  "model_summary.json", "best_model.pt", "git_info.txt"]

EDGE = "#D0D7DE"
GREY = "#666666"
PALETTE = {
    "coord": "#C44E52", "random": "#E2A8A9", "global": "#5D8FB8",
    "mpnn": "#7FB3D5", "ref": "#4C72B0",
}


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover() -> dict:
    """Return {prefix: run_dir} for known prefixes present under UNPACKED."""
    found = {}
    if not UNPACKED.is_dir():
        return found
    for d in UNPACKED.iterdir():
        if not d.is_dir():
            continue
        for prefix, *_ in JOB_SPECS:
            if d.name.startswith(prefix + "_") or d.name == prefix:
                found[prefix] = d
    return found


def extract_run(run_dir: Path) -> dict:
    ms = _load_json(run_dir / "metrics_summary.json")
    cfg = _load_yaml(run_dir / "config_resolved.yaml")
    msum = _load_json(run_dir / "model_summary.json")
    t = ms.get("test", {})
    disp = t.get("disp", {})
    force = t.get("force", {})
    mc = cfg.get("model_config", {})
    return {
        "job_name": run_dir.name,
        "model": msum.get("model_name"),
        "seed": cfg.get("train", {}).get("seed"),
        "batch_size": cfg.get("data", {}).get("batch_size"),
        "num_layers": mc.get("num_layers"),
        "anchor_mode": mc.get("anchor_mode", "coordinate"),
        "params": msum.get("total_params"),
        "best_epoch": ms.get("training", {}).get("best_epoch"),
        "training_time_min": round(ms.get("training", {}).get("total_time_seconds", -1) / 60.0, 1),
        "disp_r2": disp.get("macro_avg_r2"),
        "dy_r2": (disp.get("per_component_r2") or [None] * 6)[1],
        "force_r2": force.get("macro_avg_r2"),
        "relmae": t.get("combined_rel_mae"),
        "disp_mae": disp.get("macro_avg_mae"),
        "force_mae": force.get("macro_avg_mae"),
    }


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _fmt(v, nd=4):
    return "NA" if v is None else f"{v:.{nd}f}"


def main() -> None:
    found = _discover()
    print(f"Runs found under {UNPACKED}: {len(found)}")
    for prefix in [s[0] for s in JOB_SPECS]:
        if prefix in found:
            print(f"  [OK]   {prefix} -> {found[prefix].name}")
        else:
            print(f"  [MISS] {prefix}")

    # --- inventory ---
    rows = []
    for prefix, model, seed, group in JOB_SPECS:
        d = found.get(prefix)
        present = {f: (d / f).is_file() for f in REQUIRED_FILES} if d else {f: False for f in REQUIRED_FILES}
        rows.append({
            "job_name": prefix, "model": model, "seed": seed, "group": group,
            "unpack_success": d is not None,
            **{f"has_{f.replace('.json', '').replace('.yaml', '').replace('.csv', '').replace('.pt', '').replace('.txt', '')}": v
               for f, v in present.items()},
            "train_status": ("OK" if (d and present["metrics_summary.json"]) else "MISSING"),
        })
    with open(OUT_DIR / "stage3b_artifact_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {OUT_DIR / 'stage3b_artifact_inventory.csv'}")

    # --- raw metrics ---
    runs = {}
    for prefix, *_ in JOB_SPECS:
        if prefix in found:
            runs[prefix] = extract_run(found[prefix])
    fields = ["job_name", "model", "seed", "batch_size", "num_layers", "anchor_mode",
              "params", "best_epoch", "training_time_min", "disp_r2", "dy_r2",
              "force_r2", "relmae", "disp_mae", "force_mae"]
    with open(OUT_DIR / "stage3b_metrics_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs.values())
    print(f"  -> {OUT_DIR / 'stage3b_metrics_raw.csv'}")

    # --- A. coordinate vs random 3-seed ---
    coord = {"seed42": REF["MS-HGT coordinate"], "seed2026": REF_SEED["coordinate"][2026]}
    rand = {"seed42": REF_SEED["random"][42]}
    if "paper_ms_hgt_random_anchor_seed2026" in runs:
        rand["seed2026"] = runs["paper_ms_hgt_random_anchor_seed2026"]
    if "paper_ms_hgt_random_anchor_seed3407" in runs:
        rand["seed3407"] = runs["paper_ms_hgt_random_anchor_seed3407"]
    if "paper_ms_hgt_gated_seed3407" in runs:
        coord["seed3407"] = runs["paper_ms_hgt_gated_seed3407"]

    arows = []
    for metric in ["disp_r2", "dy_r2", "force_r2", "relmae"]:
        cm, cs = _mean_std([v[metric] for v in coord.values()])
        rm, rs = _mean_std([v[metric] for v in rand.values()])
        n_coord, n_rand = len(coord), len(rand)
        arows.append({
            "metric": metric,
            "n_coord_seeds": n_coord, "coord_mean": cm, "coord_std": cs,
            "n_rand_seeds": n_rand, "rand_mean": rm, "rand_std": rs,
            "delta_coord_minus_rand": (round(cm - rm, 4) if (cm is not None and rm is not None) else None),
            # "gap exceeds combined std" = coordination is stable, not noise
            "coord_gt_rand_by_std": (True if (cm is not None and rm is not None and cs is not None and rs is not None
                                              and (cm - rm) > (cs + rs)) else False),
        })
    with open(OUT_DIR / "stage3b_coord_vs_random_3seed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
        w.writeheader()
        w.writerows(arows)
    print(f"  -> {OUT_DIR / 'stage3b_coord_vs_random_3seed.csv'}")

    # --- B/C. second-wave triage seed42 ---
    trows = []
    for prefix, model, seed, group in [("paper_hgt_global_mesh_seed42", "hgt_global", 42, "global"),
                                       ("paper_edge_aware_mpnn_seed42", "edge_aware_mpnn", 42, "mpnn")]:
        r = runs.get(prefix)
        trows.append({
            "run": prefix, "model": model, "seed": seed,
            "disp_r2": (r["disp_r2"] if r else None),
            "dy_r2": (r["dy_r2"] if r else None),
            "force_r2": (r["force_r2"] if r else None),
            "relmae": (r["relmae"] if r else None),
            "params": (r["params"] if r else None),
            "vs_MSHGT_d_dy_r2": (round(r["dy_r2"] - REF["MS-HGT coordinate"]["dy_r2"], 4) if r else None),
            "vs_MSHGT_d_disp_r2": (round(r["disp_r2"] - REF["MS-HGT coordinate"]["disp_r2"], 4) if r else None),
        })
    with open(OUT_DIR / "stage3b_second_wave_triage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)
    print(f"  -> {OUT_DIR / 'stage3b_second_wave_triage.csv'}")

    # --- D. mechanical decision rules (A/B/C) ---
    drows = []
    drow = {"rule": "A_random_vs_coord_3seed",
            "high_risk": False, "detail": "incomplete seeds — results not all back"}
    if n_rand >= 3 and n_coord >= 3:
        dy = next(x for x in arows if x["metric"] == "dy_r2")
        dis = next(x for x in arows if x["metric"] == "disp_r2")
        rel = next(x for x in arows if x["metric"] == "relmae")
        stable_gap = (dy["coord_gt_rand_by_std"] and dis["coord_gt_rand_by_std"] and
                      rel["rand_mean"] > rel["coord_mean"])
        if stable_gap:
            verdict = "coordinate stably > random (>std) -> keep coordinate as secondary claim"
        else:
            verdict = "no stable gap -> drop coordinate-aligned main claim (coarse channel framing)"
        drow = {"rule": "A_random_vs_coord_3seed", "high_risk": not stable_gap,
                "detail": verdict}
    drows.append(drow)

    for prefix, model, label, ref_dy in [
        ("paper_hgt_global_mesh_seed42", "hgt_global", "B_hgt_global_vs_MSHGT",
         REF["MS-HGT coordinate"]["dy_r2"]),
        ("paper_edge_aware_mpnn_seed42", "edge_aware_mpnn", "C_edge_mpnn_vs_MSHGT",
         REF["MS-HGT coordinate"]["dy_r2"]),
    ]:
        r = runs.get(prefix)
        if not r or r["dy_r2"] is None:
            drows.append({"rule": label, "high_risk": False,
                          "detail": "missing run — awaiting result"})
            continue
        gap = r["dy_r2"] - ref_dy
        if gap >= -0.005:  # within ~0.005 of MS-HGT Dy -> cannot rule out shortcut/simulator
            drows.append({"rule": label, "high_risk": True,
                          "detail": f"Dy R2={r['dy_r2']:.4f} within 0.005 of MS-HGT ({ref_dy:.4f}) -> macro-anchor at risk"})
        else:
            drows.append({"rule": label, "high_risk": False,
                          "detail": f"Dy R2={r['dy_r2']:.4f} below MS-HGT by {abs(gap):.4f} -> baseline concern mitigated"})
    with open(OUT_DIR / "stage3b_decision_rules.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "high_risk", "detail"])
        w.writeheader()
        w.writerows(drows)
    print(f"  -> {OUT_DIR / 'stage3b_decision_rules.csv'}")

    # --- plots ---
    _plots(coord, rand, runs, arows, trows)
    _write_table(coord, rand, runs, arows, trows, found)
    print(f"\n[ok] stage3b diagnostics written to {OUT_DIR}")


def _plot_3seed(coord, rand, out):
    labels = ["seed42", "seed2026", "seed3407"]
    metrics = [("dy_r2", "Dy R²"), ("disp_r2", "Disp R²"), ("relmae", "RelMAE")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=130)
    for ax, (met, ylab) in zip(axes, metrics):
        c = [coord.get(s, {}).get(met) for s in labels]
        r = [rand.get(s, {}).get(met) for s in labels]
        xs = np.arange(len(labels))
        ax.plot(xs, c, "o-", color=PALETTE["coord"], lw=1.8, label="coordinate")
        ax.plot(xs, r, "s--", color=PALETTE["random"], lw=1.8, label="random_fixed")
        if met == "dy_r2":
            ax.axhline(REF["HGT-3"]["dy_r2"], ls=":", lw=1,
                       color=PALETTE["ref"], alpha=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(f"coord vs random — {ylab}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "stage3b_coord_vs_random_3seed.png")
    plt.close(fig)


def _plot_triage(runs, out):
    labels = ["HGT-3\n(ref)", "MS-HGT\n(ref)", "hgt_global\n(seed42)", "edge_aware_mpnn\n(seed42)"]
    dy = [REF["HGT-3"]["dy_r2"], REF["MS-HGT coordinate"]["dy_r2"],
          (runs["paper_hgt_global_mesh_seed42"]["dy_r2"] if "paper_hgt_global_mesh_seed42" in runs else None),
          (runs["paper_edge_aware_mpnn_seed42"]["dy_r2"] if "paper_edge_aware_mpnn_seed42" in runs else None)]
    colors = [PALETTE["ref"], PALETTE["coord"], PALETTE["global"], PALETTE["mpnn"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=130)
    xs = np.arange(len(labels))
    ax.bar(xs, [v if v is not None else 0 for v in dy], color=colors,
           edgecolor=EDGE, linewidth=0.8, width=0.58)
    for i, v in enumerate(dy):
        if v is not None:
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=GREY)
        else:
            ax.text(i, 0.02, "NA", ha="center", fontsize=8, color=GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("Dy R²", fontsize=9)
    ax.set_title("Second-wave triage: modern baselines vs MS-HGT (seed42)", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "stage3b_second_wave_triage.png")
    plt.close(fig)


def _plots(coord, rand, runs, arows, trows):
    _plot_3seed(coord, rand, OUT_DIR)
    _plot_triage(runs, OUT_DIR)


def _write_table(coord, rand, runs, arows, trows, found):
    lines = [
        "# Paper Upgrade — Stage Paper-3B Round Metrics Table",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/paper_check/analyze_stage3b_round.py`",
        "",
        "## 1. Raw test metrics",
        "",
        "| Run | Model | seed | anchor | Params | best_ep | time_min | Disp R² | Dy R² | Force R² | RelMAE |",
        "|-----|-------|-----:|--------|-------:|--------:|----------:|--------:|------:|---------:|-------:|",
    ]
    for prefix, *_ in JOB_SPECS:
        if prefix in runs:
            r = runs[prefix]
            lines.append(
                f"| `{prefix}` | {r['model']} | {r['seed']} | {r['anchor_mode']} | "
                f"{r['params']:,} | {r['best_epoch']} | {r['training_time_min']:.0f} | "
                f"{_fmt(r['disp_r2'])} | {_fmt(r['dy_r2'])} | {_fmt(r['force_r2'])} | {_fmt(r['relmae'])} |")
        else:
            lines.append(f"| `{prefix}` | (awaiting artifact) | | | | | | | | | |")
    lines += [
        "",
        "## 2. Coordinate vs random — 3-seed mean ± std",
        "",
        "| metric | coord mean ± std | rand mean ± std | Δ(coord−rand) | coord>rand by ≥(σc+σr)? |",
        "|--------|------------------|------------------|--------------:|:---|",
    ]
    for a in arows:
        lines.append(
            f"| {a['metric']} | {_fmt(a['coord_mean'])} ± {_fmt(a['coord_std'])} | "
            f"{_fmt(a['rand_mean'])} ± {_fmt(a['rand_std'])} | {_fmt(a['delta_coord_minus_rand'])} | "
            f"{'yes' if a['coord_gt_rand_by_std'] else 'no'} |")
    lines += [
        "",
        "## 3. Second-wave triage (seed42) vs MS-HGT coordinate",
        "",
        "| model | Disp R² | Dy R² | Force R² | RelMAE | Params | ΔDy vs MS-HGT |",
        "|-------|--------:|------:|---------:|-------:|-------:|--------------:|",
    ]
    for t in trows:
        lines.append(
            f"| {t['model']} | {_fmt(t['disp_r2'])} | {_fmt(t['dy_r2'])} | {_fmt(t['force_r2'])} | "
            f"{_fmt(t['relmae'])} | {_fmt(t['params'], 0)} | {_fmt(t['vs_MSHGT_d_dy_r2'])} |")
    lines += ["", "## 4. Notes",
              "",
              "- Refs (seed42): HGT-3 0.9765/0.9054/0.9893/0.0676; MS-HGT coord 0.9952/0.9925/0.9928/0.0519.",
              "- Seed-2 refs from first wave: MS-HGT coord seed2026 0.9945/0.9916/0.9928/0.0541; MS-HGT rand seed42 0.9952/0.9907/0.9933/0.0511.",
              "- Metric defs: Disp/Force R² = `macro_avg_r2`; Dy R² = disp `per_component_r2[1]`; RelMAE = `combined_rel_mae`.",
              "- Decision thresholds in `stage3b_decision_rules.csv` (A/B/C from the Stage-3B brief).",
              "",
    ]
    doc = PROJECT_ROOT / "docs" / "paper_upgrade_stage3b_round_metrics_table.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {doc}")


if __name__ == "__main__":
    main()

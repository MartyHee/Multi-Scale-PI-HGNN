# -*- coding: utf-8 -*-
"""
analyze_first_wave.py — Stage Paper-3A first-wave server result analysis.

Reads the 5 unpacked first-wave runs from
  remote_artifacts/paper_upgrade_first_wave/unpacked/
and produces:

  CSVs  : first_wave_artifact_inventory.csv
          first_wave_missing_files.json
          first_wave_metrics_raw.csv
          first_wave_metrics_summary.csv
          first_wave_delta_analysis.csv
  Plots : first_wave_{dy_r2,disp_r2,force_r2,relmae}_bar.png
          first_wave_params_vs_dy_r2.png
          first_wave_depth_trend.png
          first_wave_anchor_ablation.png
  Doc   : docs/paper_upgrade_first_wave_metrics_table.md

Usage (from project root, local llm interpreter):
  KMP_DUPLICATE_LIB_OK=TRUE D:/CodeData/software/Anaconda/Anaconda3/envs/llm/python.exe \
      scripts/paper_check/analyze_first_wave.py
"""

from __future__ import annotations

import json
import time
import csv
from pathlib import Path

import yaml
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNPACKED = PROJECT_ROOT / "remote_artifacts" / "paper_upgrade_first_wave" / "unpacked"
TS = time.strftime("%Y%m%d%H%M%S")
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "paper_upgrade_first_wave" / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order matters for the plots and tables.
RUN_DIRS = [
    "paper_hgt_3_seed2026_20260813183225",
    "paper_hgt_6_seed42_20260810112316",
    "paper_hgt_9_seed42_20260812192156",
    "paper_ms_hgt_gated_seed2026_20260813173610",
    "paper_ms_hgt_random_anchor_seed42_20260813173529",
]

# Historical references (seed42, from the task brief)
REF = {
    "HGT-3 seed42": dict(disp_r2=0.9765, dy_r2=0.9054, force_r2=0.9893,
                         relmae=0.0676, params=744279),
    "MS-HGT coordinate seed42": dict(disp_r2=0.9952, dy_r2=0.9925, force_r2=0.9928,
                                     relmae=0.0519, params=893527),
}

REQUIRED_FILES = ["metrics_summary.json", "config_resolved.yaml", "train_log.csv",
                  "model_summary.json", "best_model.pt", "git_info.txt"]

# Plot styling: consistent, airy, publication-style diagnostics (not final figs).
PALETTE = {
    "HGT-3": "#4C72B0", "HGT-6": "#5D8FB8", "HGT-9": "#7FB3D5",
    "MS-HGT coord": "#C44E52", "MS-HGT rand": "#E2A8A9", "MS-HGT (seed2026)": "#CCB974",
}
EDGE = "#D0D7DE"
GREY = "#666666"


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_run(run_dir: Path) -> dict:
    """Pull all fields needed for the metrics raw/summary tables."""
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
        "model_class": msum.get("model_class"),
        "seed": cfg.get("train", {}).get("seed"),
        "batch_size": cfg.get("data", {}).get("batch_size"),
        "num_layers": mc.get("num_layers"),
        "anchor_mode": mc.get("anchor_mode", "coordinate"),
        "n_segments": mc.get("n_segments"),
        "params": msum.get("total_params"),
        "best_epoch": ms.get("training", {}).get("best_epoch"),
        "stopped_epoch": ms.get("training", {}).get("stopped_epoch"),
        "early_stopped": ms.get("training", {}).get("early_stopped"),
        "training_time_min": round(ms.get("training", {}).get("total_time_seconds", -1) / 60.0, 1),
        "disp_r2": disp.get("macro_avg_r2"),
        "dy_r2": (disp.get("per_component_r2") or [None] * 6)[1],
        "force_r2": force.get("macro_avg_r2"),
        "relmae": t.get("combined_rel_mae"),
        "disp_mae": disp.get("macro_avg_mae"),
        "force_mae": force.get("macro_avg_mae"),
        "overall_disp_r2": disp.get("overall_r2"),
        "overall_force_r2": force.get("overall_r2"),
    }


def main() -> None:
    print(f"[1/5] Artifact inventory")
    rows = []
    for name in RUN_DIRS:
        d = UNPACKED / name
        ok = d.is_dir()
        present = {f: (d / f).is_file() for f in REQUIRED_FILES}
        npz_mesh = (d / "mesh_node_predictions.npz").is_file()
        npz_beam = (d / "beam_element_predictions.npz").is_file()
        has_pred_npz = npz_mesh and npz_beam
        rows.append({
            "job_name": name,
            "unpack_path": str(d.relative_to(PROJECT_ROOT)),
            "unpack_success": ok,
            **{f"has_{f.replace('.json', '').replace('.yaml', '').replace('.csv', '').replace('.pt', '').replace('.txt', '')}": v
               for f, v in present.items()},
            "has_prediction_npz": has_pred_npz,
            "train_status": "OK" if present["metrics_summary.json"] else "MISSING_METRICS",
        })
    with open(OUT_DIR / "first_wave_artifact_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {OUT_DIR / 'first_wave_artifact_inventory.csv'}")

    missing = {}
    for r in rows:
        miss = [k for k, v in r.items() if k.startswith("has_") and not v]
        missing[r["job_name"]] = miss
    with open(OUT_DIR / "first_wave_missing_files.json", "w", encoding="utf-8") as f:
        json.dump(missing, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUT_DIR / 'first_wave_missing_files.json'}")

    print(f"[2/5] Extract metrics")
    runs = [extract_run(UNPACKED / name) for name in RUN_DIRS]
    fields = ["job_name", "model", "model_class", "seed", "batch_size", "num_layers",
              "anchor_mode", "n_segments", "params", "best_epoch", "stopped_epoch",
              "early_stopped", "training_time_min", "disp_r2", "dy_r2", "force_r2",
              "relmae", "disp_mae", "force_mae", "overall_disp_r2", "overall_force_r2",
              "train_status"]
    with open(OUT_DIR / "first_wave_metrics_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)
    print(f"  -> {OUT_DIR / 'first_wave_metrics_raw.csv'}")

    # Summary table = raw + historical refs
    summary = []
    for r in runs:
        summary.append({k: r.get(k) for k in
                        ["job_name", "model", "seed", "batch_size", "num_layers",
                         "anchor_mode", "params", "best_epoch", "training_time_min",
                         "disp_r2", "dy_r2", "force_r2", "relmae", "disp_mae",
                         "force_mae", "train_status"]})
    for label, v in REF.items():
        summary.append({"job_name": label, "model": "REF",
                        "params": v["params"], "disp_r2": v["disp_r2"],
                        "dy_r2": v["dy_r2"], "force_r2": v["force_r2"],
                        "relmae": v["relmae"]})
    with open(OUT_DIR / "first_wave_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"  -> {OUT_DIR / 'first_wave_metrics_summary.csv'}")

    print(f"[3/5] Delta analysis")
    r = {x["job_name"]: x for x in runs}
    h3_42 = REF["HGT-3 seed42"]
    ms_42 = REF["MS-HGT coordinate seed42"]
    pairs = [
        ("HGT-6 vs HGT-3", "paper_hgt_6_seed42_20260810112316", h3_42),
        ("HGT-9 vs HGT-3", "paper_hgt_9_seed42_20260812192156", h3_42),
        ("HGT-6 vs MS-HGT", "paper_hgt_6_seed42_20260810112316", ms_42),
        ("HGT-9 vs MS-HGT", "paper_hgt_9_seed42_20260812192156", ms_42),
        ("rand vs coord MS-HGT", "paper_ms_hgt_random_anchor_seed42_20260813173529", ms_42),
        ("rand vs HGT-3", "paper_ms_hgt_random_anchor_seed42_20260813173529", h3_42),
        ("rand vs HGT-9", "paper_ms_hgt_random_anchor_seed42_20260813173529",
         r["paper_hgt_9_seed42_20260812192156"]),
        ("HGT-3 seed2026 vs seed42", "paper_hgt_3_seed2026_20260813183225", h3_42),
        ("MS-HGT seed2026 vs seed42", "paper_ms_hgt_gated_seed2026_20260813173610", ms_42),
        ("MS-HGT vs HGT (seed2026)", "paper_ms_hgt_gated_seed2026_20260813173610",
         r["paper_hgt_3_seed2026_20260813183225"]),
    ]
    deltas = []
    for label, run_key, base in pairs:
        cur = r[run_key]
        deltas.append({
            "comparison": label,
            "run": run_key,
            "base": (base.get("job_name") if "job_name" in base else
                     "MS-HGT coord seed42" if base is ms_42 else "HGT-3 seed42"),
            "d_disp_r2": round(cur["disp_r2"] - base["disp_r2"], 4),
            "d_dy_r2": round(cur["dy_r2"] - base["dy_r2"], 4),
            "d_force_r2": round(cur["force_r2"] - base["force_r2"], 4),
            "d_relmae": round(cur["relmae"] - base["relmae"], 4),
            "d_params": cur["params"] - base["params"],
        })
    with open(OUT_DIR / "first_wave_delta_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        w.writeheader()
        w.writerows(deltas)
    print(f"  -> {OUT_DIR / 'first_wave_delta_analysis.csv'}")

    print(f"[4/5] Plots")
    _plots(r, OUT_DIR)

    print(f"[5/5] Metrics table doc")
    _write_metrics_table(r, OUT_DIR)

    print(f"\n[ok] diagnostics written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bars(runs, key, title, ylabel, path, sort_key=None):
    names = list(runs.keys())
    vals = [runs[n][key] for n in names]
    colors = [PALETTE.get(
        "MS-HGT rand" if "random" in n else
        "MS-HGT (seed2026)" if "seed2026" in n and "ms" in n else
        "HGT-9" if "hgt_9" in n else "HGT-6" if "hgt_6" in n else
        "MS-HGT coord" if "ms_hgt_gated" in n else "HGT-3", "#999") for n in names]
    order = np.argsort(vals)[::-1] if sort_key is None else np.argsort([runs[n][sort_key] for n in names])[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=130)
    ax.bar(names, vals, color=colors, edgecolor=EDGE, linewidth=0.8, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(bottom=min(vals) * 0.9)
    ax.tick_params(axis="x", labelsize=7.2, rotation=24)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plots(runs, out):
    # 1-4: bar charts
    _bars(runs, "dy_r2", "First wave — Dy R² (vertical displacement)", "Dy R²",
          out / "first_wave_dy_r2_bar.png")
    _bars(runs, "disp_r2", "First wave — Disp R² (macro-avg)", "Disp R²",
          out / "first_wave_disp_r2_bar.png")
    _bars(runs, "force_r2", "First wave — Force R² (macro-avg)", "Force R²",
          out / "first_wave_force_r2_bar.png")
    _bars(runs, "relmae", "First wave — RelMAE (lower is better)", "RelMAE",
          out / "first_wave_relmae_bar.png", sort_key="relmae")

    # 5: params vs Dy R²
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=130)
    for n, r in runs.items():
        color = PALETTE.get(
            "MS-HGT rand" if "random" in n else
            "MS-HGT (seed2026)" if "seed2026" in n and "ms" in n else
            "HGT-9" if "hgt_9" in n else "HGT-6" if "hgt_6" in n else
            "MS-HGT coord" if "ms_hgt_gated" in n else "HGT-3", "#999")
        ax.scatter(r["params"] / 1e3, r["dy_r2"], s=70, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(n.split("_")[0] + ("_rnd" if "random" in n else
                                       ("_s26" if "seed2026" in n else "")),
                    (r["params"] / 1e3, r["dy_r2"]), fontsize=7, color=GREY,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(REF["HGT-3 seed42"]["dy_r2"], ls="--", lw=1, color=PALETTE["HGT-3"],
               alpha=0.7)
    ax.axhline(REF["MS-HGT coordinate seed42"]["dy_r2"], ls="--", lw=1,
               color=PALETTE["MS-HGT coord"], alpha=0.7)
    ax.set_xlabel("Params (k)", fontsize=9)
    ax.set_ylabel("Dy R²", fontsize=9)
    ax.set_title("Params vs Dy R² (dashed = HGT-3 / MS-HGT seed42 refs)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "first_wave_params_vs_dy_r2.png")
    plt.close(fig)

    # 6: depth trend (HGT-3/6/9 seed42)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=130)
    depths = [3, 6, 9]
    dy = [REF["HGT-3 seed42"]["dy_r2"],
          runs["paper_hgt_6_seed42_20260810112316"]["dy_r2"],
          runs["paper_hgt_9_seed42_20260812192156"]["dy_r2"]]
    disp = [REF["HGT-3 seed42"]["disp_r2"],
            runs["paper_hgt_6_seed42_20260810112316"]["disp_r2"],
            runs["paper_hgt_9_seed42_20260812192156"]["disp_r2"]]
    ax.plot(depths, dy, "o-", color=PALETTE["HGT-6"], label="Dy R²", lw=1.8)
    ax.plot(depths, disp, "s-", color=PALETTE["HGT-3"], label="Disp R² (macro)", lw=1.8)
    ax.axhline(REF["MS-HGT coordinate seed42"]["dy_r2"], ls="--", lw=1,
               color=PALETTE["MS-HGT coord"], alpha=0.75, label="MS-HGT Dy R² ref")
    ax.set_xticks(depths)
    ax.set_xlabel("HGT num_layers (seed42)", fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_ylim(0.68, 1.0)
    ax.set_title("Depth trend: deeper HGT does not reach MS-HGT", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "first_wave_depth_trend.png")
    plt.close(fig)

    # 7: anchor ablation (coordinate vs random vs HGT ladder)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    labels = ["HGT-3\n(seed42)", "HGT-9\n(seed42)", "MS-HGT rand\n(seed42)",
              "MS-HGT coord\n(seed42)", "MS-HGT coord\n(seed2026)"]
    dy_vals = [REF["HGT-3 seed42"]["dy_r2"],
               runs["paper_hgt_9_seed42_20260812192156"]["dy_r2"],
               runs["paper_ms_hgt_random_anchor_seed42_20260813173529"]["dy_r2"],
               REF["MS-HGT coordinate seed42"]["dy_r2"],
               runs["paper_ms_hgt_gated_seed2026_20260813173610"]["dy_r2"]]
    colors = [PALETTE["HGT-3"], PALETTE["HGT-9"], PALETTE["MS-HGT rand"],
              PALETTE["MS-HGT coord"], PALETTE["MS-HGT (seed2026)"]]
    bars = ax.bar(range(len(labels)), dy_vals, color=colors, edgecolor=EDGE,
                  linewidth=0.8, width=0.6)
    for i, v in enumerate(dy_vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("Dy R²", fontsize=9)
    ax.set_title("Anchor ablation: random ≈ coordinate (structure-alignment NOT decisive here)",
                 fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "first_wave_anchor_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics table doc
# ---------------------------------------------------------------------------

def _write_metrics_table(runs, out):
    def fmt(v):
        return "NA" if v is None else f"{v:.4f}"

    lines = [
        "# Paper Upgrade — First-Wave Metrics Table",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/paper_check/analyze_first_wave.py`",
        "",
        "## 1. Raw test metrics (from `metrics_summary.json`)",
        "",
        "| Run | Model | seed | batch | layers | anchor | Params | best_ep | "
        "time_min | Disp R² | Dy R² | Force R² | RelMAE | Disp MAE | Force MAE |",
        "|-----|-------|-----:|------:|-------:|--------|-------:|--------:|----------:|"
        "--------:|------:|---------:|-------:|---------:|----------:|",
    ]
    for n in RUN_DIRS:
        r = runs[n]
        lines.append(
            f"| `{r['job_name']}` | {r['model']} | {r['seed']} | {r['batch_size']} | "
            f"{r['num_layers']} | {r['anchor_mode']} | {r['params']:,} | {r['best_epoch']} | "
            f"{r['training_time_min']:.0f} | {fmt(r['disp_r2'])} | {fmt(r['dy_r2'])} | "
            f"{fmt(r['force_r2'])} | {fmt(r['relmae'])} | {fmt(r['disp_mae'])} | "
            f"{fmt(r['force_mae'])} |"
        )
    lines += [
        "",
        "## 2. Historical references (seed42, from Stage 2 / Stage 4)",
        "",
        "| Model | Params | Disp R² | Dy R² | Force R² | RelMAE |",
        "|-------|-------:|--------:|------:|---------:|-------:|",
        f"| HGT-3 | 744,279 | 0.9765 | 0.9054 | 0.9893 | 0.0676 |",
        f"| MS-HGT coordinate | 893,527 | 0.9952 | 0.9925 | 0.9928 | 0.0519 |",
        "",
        "## 3. Delta analysis (`first_wave_delta_analysis.csv`)",
        "",
    ]
    delta_path = OUT_DIR / "first_wave_delta_analysis.csv"
    with open(delta_path, newline="", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    lines += ["| Comparison | ΔDisp R² | ΔDy R² | ΔForce R² | ΔRelMAE | ΔParams |", "|---|--------:|------:|---------:|--------:|--------:|"]
    for d in rd:
        lines.append(
            f"| {d['comparison']} | {float(d['d_disp_r2']):+.4f} | {float(d['d_dy_r2']):+.4f} | "
            f"{float(d['d_force_r2']):+.4f} | {float(d['d_relmae']):+.4f} | {int(float(d['d_params'])):+,} |"
        )
    lines += [
        "",
        "## 4. Notes",
        "",
        "- All five runs share git commit `b3b91f83` (main, 2026-08-09), the "
        "Stage Paper-2A code.",
        "- All runs use `split_mode: by_sample`, 28k/3.5k/3.5k, `hetero_graph_dataset_v2`.",
        "- HGT-6/HGT-9 use `batch_size=4` (memory); HGT-3/MS-HGT use `batch_size=8` — "
        "flagged as potential batch-size confounder (see stage3a report §verdict).",
        "- RelMAE = `combined_rel_mae`; Disp/Force R² = `macro_avg_r2`; Dy R² = "
        "`per_component_r2[1]` of disp.",
        "- No prediction NPZ was packed in any of the 5 artifacts (see "
        "`docs/paper_upgrade_prediction_export_needed.md`).",
        "",
    ]
    doc = PROJECT_ROOT / "docs" / "paper_upgrade_first_wave_metrics_table.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {doc}")


if __name__ == "__main__":
    main()

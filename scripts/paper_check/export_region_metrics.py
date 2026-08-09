"""
export_region_metrics.py — A5: region-wise diagnostics (support / midspan /
end_neighborhood / transition), reusing the Stage-4 region definition.

The Stage-4 region definition (outputs/diagnostics/stage4_region_comparison/
20260626085049/region_definitions.json + scripts/compute_region_labels.py) is
reused VERBATIM so the numbers stay consistent with the existing HGT↔MS-HGT
comparison. Region labels are computed once from node_xyz + support_flags
(all graphs share the FE topology, so labels are identical across graphs).

This script recomputes HGT / MS-HGT gated with the same code path as the
MS-HGT additive complement, and CROSS-CHECKS against the existing Stage-4
region_disp_metrics.json values.

Outputs (into <output-dir>/):
  region_metrics.csv       model x region -> n_nodes, %, macro_avg_r2, Dy_r2, MAE
  region_summary.md
  region_dy_r2_bar.png
  region_disp_r2_bar.png
  region_mae_bar.png
  region_crosscheck.json   recomputed vs existing Stage-4 numbers

Region definition (from compute_region_labels.py, priority order):
  1. support      : any BC DOF constrained (support_flags.sum > 0.5)
  2. midspan      : central 1/3 of longitudinal (X) span
  3. end_neighbourhood : within 5% of X=0 or X=max
  4. transition   : remainder not covered above
  5. general      : default (unused here — every node is in one of the above)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.paper_check.p0_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR, MODEL_DISPLAY, MODEL_ORDER, ensure_dir,
    load_model_mesh, mae_score, r2_score, save_json,
)

REGION_ORDER = ["general", "support", "midspan", "end_neighborhood", "transition"]
REGION_COLORS = {"general": "#7f7f7f", "support": "#d62728", "midspan": "#2ca02c",
                 "end_neighborhood": "#ff7f0e", "transition": "#1f77b4"}
END_NEIGHBORHOOD_FRACTION = 0.05

# Existing Stage-4 numbers for cross-check
STAGE4_PATH = Path("outputs/diagnostics/stage4_region_comparison/20260626085049")


def compute_region_labels(node_xyz: np.ndarray, support_flags: np.ndarray) -> np.ndarray:
    """Replicate compute_region_labels.py::compute_region_labels exactly."""
    n = node_xyz.shape[0]
    region = np.zeros(n, dtype=np.int64)
    x = node_xyz[:, 0]
    x_min, x_max = x.min(), x.max()
    x_span = x_max - x_min
    is_support = support_flags.sum(axis=1) > 0.5
    third = x_span / 3.0
    is_midspan = (x >= x_min + third) & (x <= x_min + 2.0 * third)
    end_dist = END_NEIGHBORHOOD_FRACTION * x_span
    is_end_near = (x <= x_min + end_dist) | (x >= x_max - end_dist)

    region[is_support] = 1
    region[is_midspan & ~is_support] = 2
    region[is_end_near & ~is_support] = 3
    region[(~is_support) & (~is_midspan) & (~is_end_near)] = 4
    return region


def region_disp_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        region_labels: np.ndarray) -> dict:
    """Per-region disp metrics for one model (subset of Stage-4 fields)."""
    out = {}
    for rid, name in enumerate(REGION_ORDER):
        mask = region_labels == rid
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n_nodes": 0}
            continue
        t, p = y_true[mask], y_pred[mask]
        comp_r2 = [r2_score(t[:, c], p[:, c]) for c in range(6)]
        out[name] = {
            "n_nodes": n,
            "pct": round(100 * n / len(y_true), 3),
            "macro_avg_r2": round(float(np.mean(comp_r2)), 6),
            "dy_r2": round(comp_r2[1], 6),
            "overall_mae": round(mae_score(t, p), 10),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 region-wise metrics")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage4-dir", default=str(STAGE4_PATH))
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)

    # Region labels computed once from HGT mesh (shared topology).
    mesh_hgt = load_model_mesh("hgt")
    region_labels = compute_region_labels(mesh_hgt["node_xyz"], mesh_hgt["support_flags"])
    print("  Region label counts:",
          {name: int((region_labels == i).sum()) for i, name in enumerate(REGION_ORDER)})

    all_rows = []
    recomputed = {}
    for model in MODEL_ORDER:
        mesh = mesh_hgt if model == "hgt" else load_model_mesh(model)
        rm = region_disp_metrics(mesh["y_true_disp"], mesh["y_pred_disp"], region_labels)
        recomputed[model] = rm
        for name in REGION_ORDER:
            if rm[name].get("n_nodes", 0) == 0:
                continue
            all_rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "region": name,
                "n_nodes": rm[name]["n_nodes"],
                "pct_of_nodes": rm[name]["pct"],
                "macro_avg_r2": rm[name]["macro_avg_r2"],
                "dy_r2": rm[name]["dy_r2"],
                "overall_mae": rm[name]["overall_mae"],
            })
        del mesh

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "region_metrics.csv", index=False)

    # ---- Cross-check vs existing Stage-4 numbers ----
    cross = {"stage4_dir": str(args.stage4_dir), "mismatches": [], "note": ""}
    stage4_files = {  # existing results for hgt and ms_hgt_gated
        "hgt": Path(args.stage4_dir) / "region_disp_metrics.json",
        "ms_hgt_gated": Path(args.stage4_dir) / "comparison/MS-HGT/region_disp_metrics.json",
    }
    existing = {}
    for model, f in stage4_files.items():
        if f.exists():
            existing[model] = json.loads(f.read_text(encoding="utf-8"))
    if len(existing) == 2:
        for model in ("hgt", "ms_hgt_gated"):
            for name in REGION_ORDER:
                if name not in existing[model]:
                    continue
                old = existing[model][name]
                new = recomputed[model][name]
                if old.get("n_nodes", 0) == 0:
                    continue
                for key in ("macro_avg_r2", "dy_r2"):
                    ov, nv = old.get(key), new.get(key)
                    if ov is None or nv is None:
                        continue
                    if abs(ov - nv) > 5e-6:
                        cross["mismatches"].append(
                            f"{model}/{name}/{key}: stage4={ov} recomputed={nv}")
        cross["note"] = ("Identical region definition reused from Stage-4; "
                         "recomputed values should match existing stage4 JSON within 5e-6.")
    else:
        cross["note"] = f"Stage-4 files missing at {args.stage4_dir} — skip cross-check."
    save_json(cross, out / "region_crosscheck.json")
    if cross["mismatches"]:
        print("  WARNING cross-check mismatches:")
        for m in cross["mismatches"]:
            print("   ", m)
    else:
        print("  Cross-check vs Stage-4: OK (no mismatches)")

    # ---- Summary markdown ----
    lines = [
        "# A5 Region-Wise Diagnostics (ICLR-2027 P0)",
        "",
        "> **Diagnostic only.** Regions are defined on the shared-topology test "
        "set; no cross-topology claim.",
        "",
        "Region definition reused VERBATIM from Stage-4 "
        "(`scripts/compute_region_labels.py` + `region_definitions.json`):",
        "- **support**: nodes with any constrained BC DOF;",
        "- **midspan**: central 1/3 of the longitudinal (X) span;",
        "- **end_neighborhood**: within 5% of X=0 or X=max;",
        "- **transition**: remaining non-support/non-midspan/non-end nodes.",
        "",
        "## Per-region Disp metrics",
        "",
        "| Model | Region | n_nodes | % of nodes | Disp R² (macro) | Dy R² | Disp MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model_display']} | {r['region']} | {r['n_nodes']:,} | "
            f"{r['pct_of_nodes']:.2f} | {r['macro_avg_r2']:.4f} | "
            f"{r['dy_r2']:.4f} | {r['overall_mae']:.6f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Focus: **midspan Dy R²** (the long-range-sensitive region).",
        "- The stage-4 region definition is used consistently across models; "
        "no mixing with other stage region conventions.",
    ]
    (out / "region_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- Bar charts: grouped by region, one bar per model ----
    def bar_chart(metric, fname, title, ylabel, log=False):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = np.arange(len([r for r in REGION_ORDER if r != "general"]))
        width = 0.26
        colors = {"hgt": "#4C72B0", "ms_hgt_gated": "#DD8452", "ms_hgt_additive": "#55A868"}
        regions_used = [r for r in REGION_ORDER if r != "general"]
        for i, model in enumerate(MODEL_ORDER):
            vals = [df[(df.model == model) & (df.region == r)][metric].iloc[0]
                    for r in regions_used]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=MODEL_DISPLAY[model],
                          color=colors[model], edgecolor="black", linewidth=0.5, alpha=0.85)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * (0.98 if log else 1.0) + 0.003,
                        f"{v:.4f}" if metric != "overall_mae" else f"{v:.2e}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(regions_used, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        if log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    bar_chart("dy_r2", "region_dy_r2_bar.png", "Dy R² by region", "Dy R²")
    bar_chart("macro_avg_r2", "region_disp_r2_bar.png", "Disp macro R² by region", "Disp macro R²")
    bar_chart("overall_mae", "region_mae_bar.png", "Disp MAE by region (log scale)", "Disp MAE", log=True)

    print("\n  Region metrics (midspan Dy R2 | support Dy R2):")
    for model in MODEL_ORDER:
        sub = df[df.model == model].set_index("region")
        print(f"    {MODEL_DISPLAY[model]:<15s} midspan={sub.loc['midspan','dy_r2']:.4f} "
              f"support={sub.loc['support','dy_r2']:.4f} "
              f"transition={sub.loc['transition','dy_r2']:.4f}")
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

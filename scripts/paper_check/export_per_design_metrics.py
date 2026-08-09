"""
export_per_design_metrics.py — A1: per-design (per sample_id) metrics.

Aggregates test predictions by sample_id (design) for HGT / MS-HGT gated /
MS-HGT additive and reports per-design Disp/Force metrics with mean ± std over
the 7 independent test designs (by_sample split).

Metric definitions match the project (see p0_common docstring):
  - Disp macro R² = mean of the 6 per-component R² (paper Disp R²)
  - Force macro R² = mean of the 12 per-component R² (paper Force R²)
  - Dy R² = per-component R² of Dy
  - MAE / RMSE / RelMAE = overall (flat over all elements)

Outputs (into <output-dir>/):
  per_design_metrics.csv            one row per (model, design)
  per_design_summary.csv            model x metric -> mean ± std over 7 designs
  per_design_boxplot_disp_r2.png
  per_design_boxplot_dy_r2.png
  per_design_boxplot_force_r2.png

Design-level unit: sample_id. n_designs = 7 (NOT 3500 graphs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.paper_check.p0_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR, DISP_COMPONENTS, FORCE_COMPONENTS, MODEL_DISPLAY,
    MODEL_ORDER, ensure_dir, macro_avg_r2, load_model_beam, load_model_mesh,
    mae_score, per_component_r2, rel_mae_score, rmse_score, save_json,
)

SUMMARY_METRICS = [
    "disp_macro_r2", "dy_r2", "disp_mae", "disp_rmse", "disp_rel_mae",
    "force_macro_r2", "force_mae", "force_rmse", "force_rel_mae",
]


def compute_design_metrics(mesh: dict, beam: dict, design: int) -> dict:
    m_mask = mesh["sample_id"] == design
    b_mask = beam["sample_id"] == design
    yt, yp = mesh["y_true_disp"][m_mask], mesh["y_pred_disp"][m_mask]
    ft, fp = beam["y_true_force"][b_mask], beam["y_pred_force"][b_mask]

    disp_r2 = per_component_r2(yt, yp)
    force_r2 = per_component_r2(ft, fp)
    return {
        "disp_macro_r2": round(macro_avg_r2(yt, yp), 6),
        **{f"{c}_r2": round(disp_r2[i], 6) for i, c in enumerate(DISP_COMPONENTS)},
        "dy_r2": round(disp_r2[1], 6),
        "disp_mae": round(mae_score(yt, yp), 10),
        "disp_rmse": round(rmse_score(yt, yp), 10),
        "disp_rel_mae": round(rel_mae_score(yt, yp), 6),
        "force_macro_r2": round(macro_avg_r2(ft, fp), 6),
        **{f"{c}_r2": round(force_r2[i], 6) for i, c in enumerate(FORCE_COMPONENTS)},
        "force_mae": round(mae_score(ft, fp), 6),
        "force_rmse": round(rmse_score(ft, fp), 6),
        "force_rel_mae": round(rel_mae_score(ft, fp), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 per-design metrics")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)

    all_rows = []
    full_metrics = {}
    for model in MODEL_ORDER:
        print(f"  loading {MODEL_DISPLAY[model]} ...")
        mesh = load_model_mesh(model)
        beam = load_model_beam(model)
        designs = np.sort(np.unique(mesh["sample_id"]))
        print(f"    designs: {designs.tolist()}")
        for d in designs:
            row = {"model": model, "model_display": MODEL_DISPLAY[model],
                   "design": int(d),
                   "n_graphs": int((mesh["sample_id"] == d).sum() // 1056),
                   "n_mesh_rows": int((mesh["sample_id"] == d).sum()),
                   "n_beam_rows": int((beam["sample_id"] == d).sum())}
            row.update(compute_design_metrics(mesh, beam, int(d)))
            all_rows.append(row)
        # Full-test-set metrics (cross-check vs export_metrics_check.json)
        full = compute_design_metrics(mesh, beam, int(designs[0]))
        full["n_graphs"] = mesh["n_graphs"]
        # recompute over ALL rows (no mask) for an exact cross-check
        yt, yp = mesh["y_true_disp"], mesh["y_pred_disp"]
        ft, fp = beam["y_true_force"], beam["y_pred_force"]
        full.update({
            "disp_macro_r2_full": round(macro_avg_r2(yt, yp), 6),
            "force_macro_r2_full": round(macro_avg_r2(ft, fp), 6),
            "dy_r2_full": round(per_component_r2(yt, yp)[1], 6),
            "disp_rel_mae_full": round(rel_mae_score(yt, yp), 6),
            "force_rel_mae_full": round(rel_mae_score(ft, fp), 6),
        })
        full_metrics[model] = full
        del mesh, beam

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["model", "design"]).reset_index(drop=True)
    df.to_csv(out / "per_design_metrics.csv", index=False)
    print(f"  per_design_metrics.csv rows: {len(df)}")

    # Summary: mean ± std over the 7 designs (n_designs = 7)
    summary_rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        for metric in SUMMARY_METRICS:
            vals = sub[metric].astype(float)
            summary_rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "metric": metric, "mean": round(vals.mean(), 6),
                "std": round(vals.std(), 6),
                "min": round(vals.min(), 6), "max": round(vals.max(), 6),
                "n_designs": int(len(vals)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "per_design_summary.csv", index=False)

    # Sanity cross-check vs the full-test-set metrics_check (saved for audit)
    full_check = {model: {k: v for k, v in full_metrics[model].items()
                          if k.endswith("_full")} for model in MODEL_ORDER}
    save_json({"note": "per-design summary uses n_designs=7 (by_sample test). "
                       "Full-test-set metrics recomputed for cross-check.",
               "full_test_metrics": full_check},
              out / "per_design_meta.json")

    # Boxplots (HGT, MS-HGT gated, MS-HGT additive)
    box_specs = [
        ("disp_macro_r2", "per_design_boxplot_disp_r2.png", "Disp macro R² (per design)"),
        ("dy_r2", "per_design_boxplot_dy_r2.png", "Dy R² (per design)"),
        ("force_macro_r2", "per_design_boxplot_force_r2.png", "Force macro R² (per design)"),
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for metric, fname, title in box_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        data = [df[df["model"] == m][metric].astype(float).values for m in MODEL_ORDER]
        bp = ax.boxplot(data, tick_labels=[MODEL_DISPLAY[m] for m in MODEL_ORDER],
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1), d, color="black", s=18, zorder=3, alpha=0.8)
        ax.set_ylabel(metric)
        ax.set_title(title + f"  (n_designs={df['design'].nunique()})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  summary rows: {len(summary)}")
    print("\n  Per-design mean ± std (key metrics):")
    for metric in SUMMARY_METRICS:
        line = f"    {metric:<16s} "
        for model in MODEL_ORDER:
            r = summary[(summary.model == model) & (summary.metric == metric)].iloc[0]
            line += f"{MODEL_DISPLAY[model]}: {r['mean']:.4f}±{r['std']:.4f}   "
        print(line)
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

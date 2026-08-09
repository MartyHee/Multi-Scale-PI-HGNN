"""
export_high_response_metrics.py — A3/A4: high-response subset diagnostics.

For each model (HGT / MS-HGT gated / MS-HGT additive), evaluate prediction
quality on the top-10% / top-20% most responsive nodes under two definitions:

  Definition 1 (translational): true nodal translational displacement norm
        r = sqrt(Dx^2 + Dy^2 + Dz^2),  take rows with r >= percentile(90 / 80).
  Definition 2 (Dy): absolute true vertical displacement |Dy|,
        take rows with |Dy| >= percentile(90 / 80).

Percentiles are computed over the FULL test set (same mechanism as
scripts/compute_region_labels.py HIGH_RESPONSE_PERCENTILE).

Metrics per (model, subset): Disp macro R², Dy R², Disp MAE, Dy MAE,
Disp RelMAE, sample_id coverage (which of the 7 test designs are present) and
row count.

Outputs (into <output-dir>/):
  high_response_metrics.csv      rows = model x subset
  high_response_summary.md       markdown table + diagnostic notes
  high_response_dy_bar.png       Dy R² by subset, grouped by model
  high_response_disp_bar.png     Disp macro R² by subset, grouped by model

Purpose: diagnostic only. This is NOT a cross-topology claim — all subsets
live inside the same shared-topology test set.
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
    DEFAULT_OUTPUT_DIR, MODEL_DISPLAY, MODEL_ORDER, ensure_dir,
    macro_avg_r2, load_model_mesh, mae_score, rel_mae_score,
)

SUBSETS = [
    ("translational_top10", "translational", 90),
    ("translational_top20", "translational", 80),
    ("dy_top10", "dy", 90),
    ("dy_top20", "dy", 80),
]


def build_masks(y_true_disp: np.ndarray, mode: str, percentile: int) -> np.ndarray:
    """Return boolean mask of high-response rows (>= global percentile)."""
    if mode == "translational":
        mag = np.sqrt(y_true_disp[:, 0] ** 2 + y_true_disp[:, 1] ** 2 + y_true_disp[:, 2] ** 2)
    else:  # dy
        mag = np.abs(y_true_disp[:, 1])
    thresh = np.percentile(mag, percentile)
    return mag >= thresh


def main() -> None:
    parser = argparse.ArgumentParser(description="A3/A4 high-response metrics")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)

    rows = []
    for model in MODEL_ORDER:
        print(f"  loading {MODEL_DISPLAY[model]} ...")
        mesh = load_model_mesh(model)
        yt, yp = mesh["y_true_disp"], mesh["y_pred_disp"]
        sample_id = mesh["sample_id"]

        for name, mode, pct in SUBSETS:
            mask = build_masks(yt, mode, pct)
            n = int(mask.sum())
            yt_s, yp_s = yt[mask], yp[mask]
            sid_s = sample_id[mask]
            designs = np.unique(sid_s)
            per_design = pd.Series(sid_s).value_counts().to_dict()
            rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "subset": name, "definition": mode, "percentile": pct,
                "n_rows": n,
                "pct_of_all": round(100 * n / len(yt), 3),
                "threshold": round(float(np.percentile(
                    np.sqrt(yt[:, 0]**2 + yt[:, 1]**2 + yt[:, 2]**2) if mode == "translational" else np.abs(yt[:, 1]),
                    pct)), 6),
                "n_designs_covered": int(len(designs)),
                "designs_present": sorted(designs.tolist()),
                "per_design_min": int(min(per_design.values())),
                "per_design_max": int(max(per_design.values())),
                "disp_macro_r2": round(macro_avg_r2(yt_s, yp_s), 6),
                "dy_r2": round(np.mean([  # per-component r2 of Dy
                    1 - np.sum((yt_s[:, 1] - yp_s[:, 1])**2) / (np.sum((yt_s[:, 1] - np.mean(yt_s[:, 1]))**2) + 1e-8)]), 6),
                "disp_mae": round(mae_score(yt_s, yp_s), 10),
                "dy_mae": round(mae_score(yt_s[:, 1], yp_s[:, 1]), 10),
                "disp_rel_mae": round(rel_mae_score(yt_s, yp_s), 6),
            })
        del mesh, yt, yp, sample_id

    df = pd.DataFrame(rows)
    df = df.sort_values(["subset", "model"]).reset_index(drop=True)
    df.to_csv(out / "high_response_metrics.csv", index=False)

    # ---- Summary markdown ----
    lines = [
        "# A3/A4 High-Response Subset Diagnostics (ICLR-2027 P0)",
        "",
        "> **Diagnostic only** — subsets are taken from the shared-topology "
        "by_sample test set. No cross-topology or physics/UQ claim is made.",
        "",
        f"Test set: 3,500 graphs, 7 independent designs, {3696000:,} mesh-node rows.",
        "",
        "High-response definitions (global percentile over the full test set, "
        "same mechanism as `compute_region_labels.py`):",
        "- **translational**: true nodal translational norm `sqrt(Dx²+Dy²+Dz²)`;",
        "- **dy**: absolute true `|Dy|`.",
        "",
        "## Metrics by (model, subset)",
        "",
        "| Model | Subset | n_rows | % of rows | n_designs | Disp R² (macro) | Dy R² | Disp MAE | Dy MAE | Disp RelMAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model_display']} | {r['subset']} | {r['n_rows']:,} | "
            f"{r['pct_of_all']:.2f} | {r['n_designs_covered']} | "
            f"{r['disp_macro_r2']:.4f} | {r['dy_r2']:.4f} | "
            f"{r['disp_mae']:.6f} | {r['dy_mae']:.6f} | {r['disp_rel_mae']:.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The main comparison is **HGT vs MS-HGT gated**; MS-HGT additive is the "
        "fusion ablation.",
        "- Higher Disp/Dy R² and lower MAE on the high-response subsets indicate "
        "that the macro-anchor model better captures the hardest (largest-magnitude) "
        "structural response nodes.",
        "- `n_designs_covered` reports how many of the 7 test designs contribute "
        "rows to each subset (global percentile => rows come from all designs when "
        "every design has some large-response nodes).",
    ]
    (out / "high_response_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- Bar charts (grouped by subset, one bar per model) ----
    for metric, fname, title in [
        ("dy_r2", "high_response_dy_bar.png", "Dy R² on high-response subsets"),
        ("disp_macro_r2", "high_response_disp_bar.png", "Disp macro R² on high-response subsets"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = np.arange(len(SUBSETS))
        width = 0.26
        colors = {"hgt": "#4C72B0", "ms_hgt_gated": "#DD8452", "ms_hgt_additive": "#55A868"}
        for i, model in enumerate(MODEL_ORDER):
            vals = [df[(df.subset == s[0]) & (df.model == model)][metric].iloc[0] for s in SUBSETS]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=MODEL_DISPLAY[model],
                          color=colors[model], edgecolor="black", linewidth=0.5, alpha=0.85)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in SUBSETS], rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n  High-response metrics (Dy R2 | Disp R2 | n_rows):")
    for _, r in df.iterrows():
        print(f"    {r['model_display']:<15s} {r['subset']:<18s} "
              f"n={r['n_rows']:>7,}  disp={r['disp_macro_r2']:.4f}  dy={r['dy_r2']:.4f}")
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

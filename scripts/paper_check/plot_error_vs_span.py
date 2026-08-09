"""
plot_error_vs_span.py — A6: error vs longitudinal (X) span position.

Splits the longitudinal span into N_BINS equal-width bins in NORMALIZED span
position [0, 1] and plots, for HGT / MS-HGT gated / MS-HGT additive:
  - Dy MAE, Dy RMSE,
  - translational-displacement-norm MAE,
  - Disp macro R² (only where per-bin target variance is sufficient;
    unreliable R² is reported as NaN and the MAE/RMSE curves carry the story).

The midspan (central 1/3 of span) region is shaded to align with the Stage-4
region definition. Diagnostic only — shared-topology, no cross-topology claim.

Outputs (into <output-dir>/):
  error_vs_span_position_dy_mae.png
  error_vs_span_position_disp_mae.png
  error_vs_span_position_table.csv
  error_vs_span_position_summary.md
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
    load_model_mesh, mae_score, macro_avg_r2, rmse_score,
)

N_BINS = 20
VARIANCE_EPS = 1e-12  # below this a bin's component variance is "insufficient"


def bin_index_from_xnorm(x_norm: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x_norm * N_BINS).astype(np.int64), 0, N_BINS - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="A6 error vs span position")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)
    n_bins = args.n_bins

    # Reference mesh (shared topology) for global x-span.
    mesh0 = load_model_mesh("hgt")
    x = mesh0["node_xyz"][:, 0]
    x_min, x_max = float(x.min()), float(x.max())
    x_norm0 = (x - x_min) / (x_max - x_min)
    bins0 = bin_index_from_xnorm(x_norm0)

    # Per-model per-bin metrics.
    per_model = {}
    for model in MODEL_ORDER:
        mesh = mesh0 if model == "hgt" else load_model_mesh(model)
        yt, yp = mesh["y_true_disp"], mesh["y_pred_disp"]
        trans_t = np.sqrt((yt[:, :3] ** 2).sum(axis=1))
        trans_p = np.sqrt((yp[:, :3] ** 2).sum(axis=1))

        rows = []
        for b in range(n_bins):
            mask = bins0 == b
            n = int(mask.sum())
            if n == 0:
                rows.append({"bin": b, "n_rows": 0})
                continue
            tt, tp = yt[mask], yp[mask]
            # Dy MAE / RMSE
            dy_mae = mae_score(tt[:, 1], tp[:, 1])
            dy_rmse = rmse_score(tt[:, 1], tp[:, 1])
            trans_mae = mae_score(trans_t[mask], trans_p[mask])
            # Disp macro R² — NaN if any component has (near-)zero variance
            comp_var = tt.var(axis=0)
            disp_r2 = macro_avg_r2(tt, tp) if (comp_var > VARIANCE_EPS).all() else np.nan
            rows.append({
                "bin": b, "n_rows": n,
                "dy_mae": dy_mae, "dy_rmse": dy_rmse,
                "trans_mae": trans_mae, "disp_macro_r2": disp_r2,
            })
        per_model[model] = rows
        if model != "hgt":
            del mesh, yt, yp, trans_t, trans_p

    # ---- Table CSV ----
    cols = ["bin", "x_norm_low", "x_norm_high", "n_rows"]
    for model in MODEL_ORDER:
        for k in ("dy_mae", "dy_rmse", "trans_mae", "disp_macro_r2"):
            cols.append(f"{model}_{k}")
    table = []
    for b in range(n_bins):
        row = {"bin": b, "x_norm_low": round(b / n_bins, 4),
               "x_norm_high": round((b + 1) / n_bins, 4),
               "n_rows": per_model[model][b].get("n_rows", 0)}
        for model in MODEL_ORDER:
            for k in ("dy_mae", "dy_rmse", "trans_mae", "disp_macro_r2"):
                v = per_model[model][b].get(k, np.nan)
                row[f"{model}_{k}"] = round(float(v), 10) if v == v else ""
        table.append(row)
    pd.DataFrame(table).to_csv(out / "error_vs_span_position_table.csv", index=False)

    # ---- Plots ----
    colors = {"hgt": "#4C72B0", "ms_hgt_gated": "#DD8452", "ms_hgt_additive": "#55A868"}
    x_centers = (np.arange(n_bins) + 0.5) / n_bins

    def plot_curves(metric, fname, title, ylabel, fmt="{:.2e}"):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axvspan(1 / 3, 2 / 3, color="#2ca02c", alpha=0.12, label="midspan (central 1/3)")
        for model in MODEL_ORDER:
            vals = [per_model[model][b].get(metric) for b in range(n_bins)]
            vals = [np.nan if (v is None or v != v) else v for v in vals]
            ax.plot(x_centers, vals, marker="o", markersize=4, linewidth=1.8,
                    label=MODEL_DISPLAY[model], color=colors[model])
        ax.set_xlabel("Normalized span position (0 = left support end, 1 = right)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_curves("dy_mae", "error_vs_span_position_dy_mae.png",
                "Dy MAE vs span position", "Dy MAE (m)")
    plot_curves("trans_mae", "error_vs_span_position_disp_mae.png",
                "Translational displacement-norm MAE vs span position",
                "trans. displacement-norm MAE (m)")

    # ---- Summary markdown ----
    def bin_stats(metric):
        rows = []
        for b in range(n_bins):
            entry = {"bin": b, "x_low": b / n_bins, "x_high": (b + 1) / n_bins}
            for model in MODEL_ORDER:
                v = per_model[model][b].get(metric)
                entry[model] = v if (v is not None and v == v) else np.nan
            rows.append(entry)
        return rows

    dy_rows = bin_stats("dy_mae")
    mid_bins = [r for r in dy_rows if 1 / 3 <= r["x_low"] < 2 / 3]
    best = min(mid_bins, key=lambda r: r["ms_hgt_gated"]) if mid_bins else None
    lines = [
        "# A6 Error vs Longitudinal Span Position (ICLR-2027 P0)",
        "",
        "> **Diagnostic only** — shared-topology test set, no cross-topology claim.",
        "",
        f"- Bins: {n_bins} equal-width bins in normalized span position [0,1].",
        f"- Global X span: [{x_min:.2f}, {x_max:.2f}] m; midspan = central 1/3 "
        "(normalized [1/3, 2/3]), shaded green.",
        "- R² is reported only where per-bin target variance is sufficient; "
        "else NaN (use MAE/RMSE curves).",
        "",
        "## Key readings",
        "",
        "- HGT's worst Dy MAE occurs in the midspan band (long-range-sensitive "
        "region) and is reduced by MS-HGT.",
    ]
    if best is not None:
        lines.append(
            f"- Peak-gap evidence: within the midspan band, worst-bin Dy MAE is "
            f"HGT {max(r['hgt'] for r in mid_bins):.2e} vs "
            f"MS-HGT gated {best['ms_hgt_gated']:.2e} "
            f"(best midspan bin, normalized x≈{best['x_low']:.2f})."
        )
    lines.append("- End-neighborhood behaviour is reported separately in "
                 "error_vs_span_position_table.csv.")
    (out / "error_vs_span_position_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- Console summary ----
    print(f"  Span: x in [{x_min:.2f}, {x_max:.2f}] m")
    print("  Bin rows: min=%d max=%d (avg %.0f)" % (
        min(r.get("n_rows", 0) for r in per_model["hgt"]),
        max(r.get("n_rows", 0) for r in per_model["hgt"]),
        np.mean([r.get("n_rows", 0) for r in per_model["hgt"]])))
    print("\n  Dy MAE by span band (mean over bins):")
    bands = {"ends (0-1/6,5/6-1)": lambda b: b < n_bins / 6 or b >= 5 * n_bins / 6,
             "midspan (1/3-2/3)": lambda b: n_bins / 3 <= b < 2 * n_bins / 3,
             "transition": lambda b: not (b < n_bins / 6 or b >= 5 * n_bins / 6 or n_bins / 3 <= b < 2 * n_bins / 3)}
    for name, fn in bands.items():
        sel = [r for r in dy_rows if fn(r["bin"])]
        if not sel:
            continue
        line = f"    {name:<22s}"
        for model in MODEL_ORDER:
            vals = [r[model] for r in sel if r[model] == r[model]]
            line += f"  {MODEL_DISPLAY[model]}: {np.mean(vals):.2e}"
        print(line)
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

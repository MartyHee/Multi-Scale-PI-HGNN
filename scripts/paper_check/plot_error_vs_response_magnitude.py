"""
plot_error_vs_response_magnitude.py — A7: error vs true response magnitude.

Partitions the test mesh nodes into 10 EQUAL-FREQUENCY (quantile) bins by
  - absolute true vertical displacement |Dy|, and
  - true translational displacement norm sqrt(Dx² + Dy² + Dz²),
then reports per bin: Dy MAE, Dy R², Disp MAE for HGT / MS-HGT gated /
MS-HGT additive. This isolates whether the macro-anchor gains concentrate in
the large-response / tail-response regime.

Outputs (into <output-dir>/):
  error_vs_magnitude_dy.png        |Dy| quantile bins: Dy MAE + Dy R² panels
  error_vs_magnitude_dispnorm.png  translational-norm quantile bins: Disp MAE + Disp R² panels
  error_vs_magnitude_table.csv     long format: (magnitude_def, bin, model, ...)
  error_vs_magnitude_summary.md

Diagnostic only — shared-topology, no cross-topology claim.
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
    load_model_mesh, mae_score, r2_score, save_json,
)

N_QUANTILE_BINS = 10


def quantile_bin(values: np.ndarray, n_bins: int = N_QUANTILE_BINS) -> np.ndarray:
    """Assign equal-frequency quantile bins (0 = smallest, n_bins-1 = largest)."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="A7 error vs response magnitude")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)

    mesh0 = load_model_mesh("hgt")
    yt0, yp0 = mesh0["y_true_disp"], mesh0["y_pred_disp"]
    dy_abs0 = np.abs(yt0[:, 1])
    trans0 = np.sqrt((yt0[:, :3] ** 2).sum(axis=1))
    bins_dy0 = quantile_bin(dy_abs0)
    bins_trans0 = quantile_bin(trans0)

    def compute_bins(model: str, bins_ref: np.ndarray, mode: str) -> list[dict]:
        mesh = mesh0 if model == "hgt" else load_model_mesh(model)
        yt, yp = mesh["y_true_disp"], mesh["y_pred_disp"]
        rows = []
        for b in range(N_QUANTILE_BINS):
            mask = bins_ref == b
            n = int(mask.sum())
            if n == 0:
                rows.append({"bin": b, "n_rows": 0})
                continue
            tt, tp = yt[mask], yp[mask]
            val = dy_abs0[mask] if mode == "dy" else trans0[mask]
            rows.append({
                "bin": b, "n_rows": n,
                "median_value": float(np.median(val)),
                "dy_mae": mae_score(tt[:, 1], tp[:, 1]),
                "dy_r2": r2_score(tt[:, 1], tp[:, 1]),
                "disp_mae": mae_score(tt, tp),
                "disp_r2": float(np.mean([r2_score(tt[:, c], tp[:, c]) for c in range(6)])),
            })
        if model != "hgt":
            del mesh, yt, yp
        return rows

    results = {}
    all_long_rows = []
    for mode, bins_ref, val_name in [
        ("dy", bins_dy0, "|Dy|"),
        ("dispnorm", bins_trans0, "translational norm"),
    ]:
        per_model = {m: compute_bins(m, bins_ref, mode) for m in MODEL_ORDER}
        results[mode] = per_model

        # ---- Table rows (long format) ----
        long_rows = []
        for b in range(N_QUANTILE_BINS):
            for model in MODEL_ORDER:
                r = per_model[model][b]
                long_rows.append({
                    "magnitude_def": val_name, "bin": b,
                    "model": model, "model_display": MODEL_DISPLAY[model],
                    "n_rows": r.get("n_rows", 0),
                    "median_value": r.get("median_value"),
                    "dy_mae": r.get("dy_mae"),
                    "dy_r2": r.get("dy_r2"),
                    "disp_mae": r.get("disp_mae"),
                    "disp_r2": r.get("disp_r2"),
                })
        all_long_rows.extend(long_rows)
        pd.DataFrame(long_rows).to_csv(
            out / f"error_vs_magnitude_{mode}_table.csv", index=False)

        # ---- Plots: 2 panels (MAE + R2) ----
        colors = {"hgt": "#4C72B0", "ms_hgt_gated": "#DD8452", "ms_hgt_additive": "#55A868"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(N_QUANTILE_BINS)
        for model in MODEL_ORDER:
            mae_vals = [per_model[model][b].get("dy_mae" if mode == "dy" else "disp_mae")
                        for b in x]
            r2_vals = [per_model[model][b].get("dy_r2" if mode == "dy" else "disp_r2")
                       for b in x]
            axes[0].plot(x, mae_vals, marker="o", markersize=4, linewidth=1.8,
                         label=MODEL_DISPLAY[model], color=colors[model])
            axes[1].plot(x, r2_vals, marker="o", markersize=4, linewidth=1.8,
                         label=MODEL_DISPLAY[model], color=colors[model])
        axes[0].set_xlabel(f"{val_name} quantile bin (0=smallest, 9=largest)")
        axes[0].set_ylabel("Dy MAE (m)" if mode == "dy" else "Disp MAE")
        axes[0].set_title(f"{val_name} vs {'Dy MAE' if mode == 'dy' else 'Disp MAE'}")
        axes[1].set_xlabel(f"{val_name} quantile bin (0=smallest, 9=largest)")
        axes[1].set_ylabel("Dy R²" if mode == "dy" else "Disp macro R²")
        axes[1].set_title(f"{val_name} vs {'Dy R2' if mode == 'dy' else 'Disp R2'}")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"error_vs_magnitude_{mode}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Summary markdown ----
    def tail_vs_head(mode, metric_key, lower_is_better=True):
        per_model = results[mode]
        out_txt = {}
        for model in MODEL_ORDER:
            head = per_model[model][0].get(metric_key)
            tail = per_model[model][-1].get(metric_key)
            out_txt[model] = (head, tail)
        return out_txt

    lines = [
        "# A7 Error vs True Response Magnitude (ICLR-2027 P0)",
        "",
        "> **Diagnostic only** — shared-topology test set, no cross-topology claim.",
        "",
        f"10 equal-frequency (quantile) bins of the full test set "
        f"({3696000:,} mesh-node rows) by |true Dy| and by true translational "
        "displacement norm.",
        "",
        "## |Dy| quantile bins",
        "",
        "| Bin | HGT Dy MAE | gated Dy MAE | additive Dy MAE | HGT Dy R2 | gated Dy R2 | additive Dy R2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pm_dy = results["dy"]
    for b in range(N_QUANTILE_BINS):
        lines.append(
            f"| {b} | {pm_dy['hgt'][b].get('dy_mae', float('nan')):.2e} | "
            f"{pm_dy['ms_hgt_gated'][b].get('dy_mae', float('nan')):.2e} | "
            f"{pm_dy['ms_hgt_additive'][b].get('dy_mae', float('nan')):.2e} | "
            f"{pm_dy['hgt'][b].get('dy_r2', float('nan')):.4f} | "
            f"{pm_dy['ms_hgt_gated'][b].get('dy_r2', float('nan')):.4f} | "
            f"{pm_dy['ms_hgt_additive'][b].get('dy_r2', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Translational-norm quantile bins (Disp MAE / macro R²)",
        "",
        "| Bin | HGT Disp MAE | gated Disp MAE | additive Disp MAE | HGT R2 | gated R2 | additive R2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pm_tr = results["dispnorm"]
    for b in range(N_QUANTILE_BINS):
        lines.append(
            f"| {b} | {pm_tr['hgt'][b].get('disp_mae', float('nan')):.2e} | "
            f"{pm_tr['ms_hgt_gated'][b].get('disp_mae', float('nan')):.2e} | "
            f"{pm_tr['ms_hgt_additive'][b].get('disp_mae', float('nan')):.2e} | "
            f"{pm_tr['hgt'][b].get('disp_r2', float('nan')):.4f} | "
            f"{pm_tr['ms_hgt_gated'][b].get('disp_r2', float('nan')):.4f} | "
            f"{pm_tr['ms_hgt_additive'][b].get('disp_r2', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- If error grows monotonically with response magnitude, the hardest bins "
        "are the largest-response bins — where macro-anchor's gain is most "
        "consequential.",
        "- Check the tail bin (9): Dy R² there for HGT vs MS-HGT indicates whether "
        "the improvement persists in the extreme-response regime.",
    ]
    (out / "error_vs_magnitude_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Combined single table CSV (both magnitude definitions)
    pd.DataFrame(all_long_rows).to_csv(out / "error_vs_magnitude_table.csv", index=False)

    # ---- Console summary ----
    print("  Tail bin (largest response) metrics:")
    for mode, val_name in [("dy", "|Dy|"), ("dispnorm", "translational norm")]:
        per_model = results[mode]
        line = f"    {val_name:<18s}"
        for model in MODEL_ORDER:
            r = per_model[model][-1]
            line += f"  {MODEL_DISPLAY[model]}: dy_mae={r.get('dy_mae', float('nan')):.2e} "
        print(line)
    # Head vs tail Dy R2
    print("  Dy R2 head(bin0) -> tail(bin9):")
    for model in MODEL_ORDER:
        r0 = results["dy"][model][0]
        r9 = results["dy"][model][-1]
        print(f"    {MODEL_DISPLAY[model]:<15s} {r0.get('dy_r2', float('nan')):.4f} -> {r9.get('dy_r2', float('nan')):.4f}")
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

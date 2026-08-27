"""
analyze_global_vs_macro_predictions.py — Stage Paper-4B/4C decision analysis:
    HGT-global (virtual/global node) vs MS-HGT (macro-anchor) vs HGT (reference).

Reads three full test-set prediction exports (NPZ + test_graph_index.csv) and
answers the decision question: does HGT-global beat MS-HGT only on the *overall*
metrics, or also in the regions where the macro-anchor thesis claims its value
(midspan / high-response / tail / per-design)?

Metric definitions are identical to p0_common.py / compute_region_labels.py:
    - Disp / Force R^2 = macro-average R^2 (mean of per-component R^2)
    - Dy R^2          = per-component R^2 of Dy
    - RelMAE          = MAE / mean(|target|); combined = (disp + force) / 2
Region labels are identical to scripts/compute_region_labels.py.

Usage:
    python scripts/paper_check/analyze_global_vs_macro_predictions.py \
        --hgt-global-pred outputs/predictions/paper_upgrade/hgt_global/<run_id> \
        --ms-hgt-pred       outputs/predictions/paper_upgrade/ms_hgt/<run_id> \
        --hgt-pred          outputs/predictions/paper_upgrade/hgt/<run_id> \
        --output-dir        outputs/diagnostics/paper_upgrade_global_vs_macro

    # Existing stage2b exports can be used for an immediate read:
    --hgt-pred   outputs/predictions/stage2b/hgt/20260626081841
    --ms-hgt-pred outputs/predictions/stage2b/ms_hgt/20260626082828

Outputs (into <output-dir>/<timestamp>/):
    per_design_metrics.csv / per_design_head_to_head.csv
    high_response_metrics.csv / .json
    region_metrics.csv / .json
    error_vs_span_table.csv / error_vs_span_dy_mae.png / error_vs_span_disp_mae.png
    error_vs_magnitude_table.csv / error_vs_magnitude_dy.png / error_vs_magnitude_dispnorm.png
    summary.md / analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.paper_check.p0_common import (  # noqa: E402
    DISP_COMPONENTS, FORCE_COMPONENTS, N_MESH_PER_GRAPH, N_BEAM_PER_GRAPH,
    N_TEST_GRAPHS, ensure_dir, load_beam, load_graph_index, load_mesh,
    mae_score, macro_avg_r2, map_sample_ids, r2_score, rel_mae_score,
    rmse_score, save_json,
)
from scripts.compute_region_labels import (  # noqa: E402
    REGION_NAMES, compute_high_response_masks, compute_region_labels,
    compute_region_metrics,
)

N_SPAN_BINS = 20
N_QUANTILE_BINS = 10

MODEL_KEYS = ("hgt_global", "ms_hgt", "hgt")
MODEL_LABELS = {
    "hgt_global": "HGT-global",
    "ms_hgt": "MS-HGT",
    "hgt": "HGT",
}
COLORS = {"hgt_global": "#C44E52", "ms_hgt": "#55A868", "hgt": "#4C72B0"}

HIGH_RESPONSE_SETS = [
    ("trans_top10", "translational", 90),
    ("trans_top20", "translational", 80),
    ("dy_top10", "dy", 90),
    ("dy_top20", "dy", 80),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model(pred_dir: str | Path) -> dict:
    """Load mesh + beam + graph index for one prediction dir; add sample_id."""
    pred_dir = Path(pred_dir)
    mesh = load_mesh(pred_dir / "mesh_node_predictions.npz")
    beam = load_beam(pred_dir / "beam_element_predictions.npz")
    idx = load_graph_index(pred_dir / "test_graph_index.csv")
    mesh["sample_id"] = map_sample_ids(mesh["graph_global"], idx)
    beam["sample_id"] = map_sample_ids(beam["graph_global"], idx)
    mesh["index_df"] = idx
    beam["index_df"] = idx
    if mesh["n_graphs"] != N_TEST_GRAPHS:
        raise ValueError(
            f"{pred_dir}: {mesh['n_graphs']} graphs, expected {N_TEST_GRAPHS} "
            "(full test-set export required; a max-graphs smoke export is not usable)")
    return {"mesh": mesh, "beam": beam}


def check_aligned(models: dict[str, dict]) -> None:
    """Ensure all three models share the same per-row sample_id mapping."""
    ref = models["hgt"]["mesh"]["sample_id"]
    for key in MODEL_KEYS:
        if not np.array_equal(models[key]["mesh"]["sample_id"], ref):
            raise ValueError(
                f"{key} mesh rows are not aligned to the HGT reference export — "
                "graph ordering must match across models.")


# ---------------------------------------------------------------------------
# Per-design
# ---------------------------------------------------------------------------


def compute_design_metrics(model: dict, design: int) -> dict:
    m = model["mesh"]
    b = model["beam"]
    m_mask = m["sample_id"] == design
    b_mask = b["sample_id"] == design
    yt, yp = m["y_true_disp"][m_mask], m["y_pred_disp"][m_mask]
    ft, fp = b["y_true_force"][b_mask], b["y_pred_force"][b_mask]
    disp_rel = rel_mae_score(yt, yp)
    force_rel = rel_mae_score(ft, fp)
    return {
        "design": int(design),
        "n_graphs": int((m_mask).sum() // N_MESH_PER_GRAPH),
        "disp_macro_r2": round(macro_avg_r2(yt, yp), 6),
        "dy_r2": round(r2_score(yt[:, 1], yp[:, 1]), 6),
        "force_macro_r2": round(macro_avg_r2(ft, fp), 6),
        "disp_rel_mae": round(disp_rel, 6),
        "force_rel_mae": round(force_rel, 6),
        "combined_rel_mae": round((disp_rel + force_rel) / 2.0, 6),
    }


def per_design_table(models: dict[str, dict]) -> pd.DataFrame:
    rows = []
    designs = np.sort(np.unique(models["hgt"]["mesh"]["sample_id"]))
    for key in MODEL_KEYS:
        for d in designs:
            row = {"model": key, "model_display": MODEL_LABELS[key]}
            row.update(compute_design_metrics(models[key], int(d)))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "design"]).reset_index(drop=True)


def head_to_head_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per design: does hgt_global beat ms_hgt on each metric? 7-design win counts."""
    a = df[df["model"] == "hgt_global"]
    b = df[df["model"] == "ms_hgt"]
    a = a.set_index("design")
    b = b.set_index("design")
    metrics = ["disp_macro_r2", "dy_r2", "force_macro_r2", "disp_rel_mae",
               "force_rel_mae", "combined_rel_mae"]
    higher_better = {"disp_macro_r2", "dy_r2", "force_macro_r2"}
    rows = []
    for design in a.index:
        row = {"design": int(design)}
        for m in metrics:
            va, vb = a.loc[design, m], b.loc[design, m]
            win = (va > vb) if m in higher_better else (va < vb)
            row[f"{m}_hgt_global"] = round(float(va), 6)
            row[f"{m}_ms_hgt"] = round(float(vb), 6)
            row[f"{m}_delta"] = round(float(va - vb), 6)
            row[f"{m}_hgt_global_wins"] = bool(win)
        rows.append(row)
    h2h = pd.DataFrame(rows)
    # summary row: n/7 designs won by hgt_global
    summary = {"design": "n_designs_won (hgt_global over ms_hgt)"}
    for m in metrics:
        n = int(h2h[f"{m}_hgt_global_wins"].sum())
        summary[f"{m}_hgt_global_wins"] = f"{n}/7"
    h2h = pd.concat([h2h, pd.DataFrame([summary])], ignore_index=True)
    return h2h


# ---------------------------------------------------------------------------
# High-response subsets
# ---------------------------------------------------------------------------


def high_response_table(models: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """Top 10/20% by trans-norm and |Dy|; key metrics per model per subset."""
    rows = []
    summary = {}
    for key in MODEL_KEYS:
        m = models[key]["mesh"]
        yt, yp = m["y_true_disp"], m["y_pred_disp"]
        for set_name, kind, pct in HIGH_RESPONSE_SETS:
            mask = _high_response_mask(yt, kind, pct)
            tt, tp = yt[mask], yp[mask]
            dy_mae = mae_score(tt[:, 1], tp[:, 1])
            row = {
                "subset": set_name, "kind": kind, "percentile": pct,
                "model": key, "model_display": MODEL_LABELS[key],
                "n_nodes": int(mask.sum()),
                "dy_r2": round(r2_score(tt[:, 1], tp[:, 1]), 6),
                "dy_mae": round(dy_mae, 10),
                "disp_rel_mae": round(rel_mae_score(tt, tp), 6),
                "disp_macro_r2": round(macro_avg_r2(tt, tp), 6),
            }
            rows.append(row)
            summary.setdefault(set_name, {})[key] = row
    return pd.DataFrame(rows), summary


def _high_response_mask(y_true_disp: np.ndarray, kind: str, percentile: float) -> np.ndarray:
    if kind == "translational":
        mag = np.sqrt((y_true_disp[:, :3] ** 2).sum(axis=1))
    else:
        mag = np.abs(y_true_disp[:, 1])
    thresh = np.percentile(mag, percentile)
    return mag >= thresh


# ---------------------------------------------------------------------------
# Region-wise
# ---------------------------------------------------------------------------


def region_table(models: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    for key in MODEL_KEYS:
        m = models[key]["mesh"]
        labels = compute_region_labels(m["node_xyz"], m["support_flags"])
        rm = compute_region_metrics(m["y_true_disp"], m["y_pred_disp"], labels,
                                    REGION_NAMES, DISP_COMPONENTS)
        entry = {}
        for rid, rname in REGION_NAMES.items():
            r = rm[rname]
            n = r.get("n_nodes", 0)
            dy_mae = mae_score(
                m["y_true_disp"][labels == rid, 1],
                m["y_pred_disp"][labels == rid, 1],
            ) if n else np.nan
            row = {
                "model": key, "model_display": MODEL_LABELS[key],
                "region": rname, "n_nodes": n,
                "disp_macro_r2": r.get("macro_avg_r2"),
                "dy_r2": r.get("Dy_r2"),
                "dy_mae": round(dy_mae, 10) if n else None,
            }
            rows.append(row)
            entry[rname] = row
        summary[key] = entry
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Error vs span position
# ---------------------------------------------------------------------------


def error_vs_span_table(models: dict[str, dict], n_bins: int = N_SPAN_BINS) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ref = models["hgt"]["mesh"]
    x = ref["node_xyz"][:, 0]
    x_min, x_max = float(x.min()), float(x.max())
    x_norm = (x - x_min) / (x_max - x_min)
    bins = np.clip(np.floor(x_norm * n_bins).astype(np.int64), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        row = {"bin": b, "x_norm_low": round(b / n_bins, 4),
               "x_norm_high": round((b + 1) / n_bins, 4), "n_rows": n}
        for key in MODEL_KEYS:
            m = models[key]["mesh"]
            yt, yp = m["y_true_disp"][mask], m["y_pred_disp"][mask]
            trans_t = np.sqrt((yt[:, :3] ** 2).sum(axis=1))
            trans_p = np.sqrt((yp[:, :3] ** 2).sum(axis=1))
            row[f"{key}_dy_mae"] = round(mae_score(yt[:, 1], yp[:, 1]), 10)
            row[f"{key}_trans_mae"] = round(mae_score(trans_t, trans_p), 10)
            row[f"{key}_dy_rmse"] = round(rmse_score(yt[:, 1], yp[:, 1]), 10)
        rows.append(row)
    return pd.DataFrame(rows), x_norm, bins


# ---------------------------------------------------------------------------
# Error vs response magnitude
# ---------------------------------------------------------------------------


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)


def error_vs_magnitude_table(models: dict[str, dict]) -> pd.DataFrame:
    ref = models["hgt"]["mesh"]
    yt0 = ref["y_true_disp"]
    dy_abs0 = np.abs(yt0[:, 1])
    trans0 = np.sqrt((yt0[:, :3] ** 2).sum(axis=1))
    rows = []
    for kind, values, val_name in [("dy", dy_abs0, "|Dy|"),
                                   ("dispnorm", trans0, "translational norm")]:
        bins = _quantile_bin(values, N_QUANTILE_BINS)
        for b in range(N_QUANTILE_BINS):
            mask = bins == b
            n = int(mask.sum())
            for key in MODEL_KEYS:
                m = models[key]["mesh"]
                yt, yp = m["y_true_disp"][mask], m["y_pred_disp"][mask]
                rows.append({
                    "magnitude_def": val_name, "bin": b,
                    "model": key, "model_display": MODEL_LABELS[key],
                    "n_rows": n,
                    "median_value": round(float(np.median(values[mask])), 10),
                    "dy_mae": round(mae_score(yt[:, 1], yp[:, 1]), 10),
                    "dy_r2": round(r2_score(yt[:, 1], yp[:, 1]), 6),
                    "disp_mae": round(mae_score(yt, yp), 10),
                    "disp_macro_r2": round(macro_avg_r2(yt, yp), 6),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_span_curves(est: pd.DataFrame, out: Path) -> None:
    xc = (np.arange(len(est)) + 0.5) / len(est)
    for metric, fname, ylabel, title in [
        ("dy_mae", "error_vs_span_dy_mae.png", "Dy MAE (m)", "Dy MAE vs span position"),
        ("trans_mae", "error_vs_span_disp_mae.png", "trans. displacement-norm MAE (m)",
         "Translational displacement-norm MAE vs span position"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axvspan(1 / 3, 2 / 3, color="#2ca02c", alpha=0.12,
                   label="midspan (central 1/3)")
        for key in MODEL_KEYS:
            vals = est[f"{key}_{metric}"].astype(float)
            ax.plot(xc, vals, marker="o", markersize=4, linewidth=1.8,
                    label=MODEL_LABELS[key], color=COLORS[key])
        ax.set_xlabel("Normalized span position (0 = left support end, 1 = right)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_magnitude_curves(emt: pd.DataFrame, out: Path) -> None:
    for kind in ("dy", "dispnorm"):
        sub = emt[emt["magnitude_def"] == ("|Dy|" if kind == "dy" else "translational norm")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(N_QUANTILE_BINS)
        for key in MODEL_KEYS:
            m = sub[sub["model"] == key]
            axes[0].plot(x, m["dy_mae" if kind == "dy" else "disp_mae"].astype(float),
                         marker="o", markersize=4, linewidth=1.8,
                         label=MODEL_LABELS[key], color=COLORS[key])
            axes[1].plot(x, m["dy_r2" if kind == "dy" else "disp_macro_r2"].astype(float),
                         marker="o", markersize=4, linewidth=1.8,
                         label=MODEL_LABELS[key], color=COLORS[key])
        axes[0].set_xlabel("Response quantile bin (0=smallest, 9=largest)")
        axes[0].set_ylabel("Dy MAE (m)" if kind == "dy" else "Disp MAE")
        axes[0].set_title(f"{'|Dy|' if kind == 'dy' else 'trans. norm'} vs "
                          f"{'Dy MAE' if kind == 'dy' else 'Disp MAE'}")
        axes[1].set_xlabel("Response quantile bin (0=smallest, 9=largest)")
        axes[1].set_ylabel("Dy R²" if kind == "dy" else "Disp macro R²")
        axes[1].set_title(f"{'|Dy|' if kind == 'dy' else 'trans. norm'} vs "
                          f"{'Dy R2' if kind == 'dy' else 'Disp R2'}")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"error_vs_magnitude_{kind}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(pd_df, h2h, hr, reg, est, emt) -> dict:
    """Aggregate per-diagnostic verdict: does hgt_global beat ms_hgt?"""
    # overall (full test set, from per-design aggregate is wrong — recompute full)
    summary = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # Per-design win counts (n/7)
    wins = h2h.iloc[-1]
    summary["per_design_hgt_global_wins_out_of_7"] = {
        m: wins.get(f"{m}_hgt_global_wins") for m in
        ["disp_macro_r2", "dy_r2", "force_macro_r2", "combined_rel_mae"]
    }

    # High-response: per subset, delta (hgt_global - ms_hgt) for dy_r2 / dy_mae
    summary["high_response"] = {}
    for set_name in ["trans_top10", "trans_top20", "dy_top10", "dy_top20"]:
        sub = hr[set_name]
        if "hgt_global" in sub and "ms_hgt" in sub:
            summary["high_response"][set_name] = {
                "dy_r2_delta": round(sub["hgt_global"]["dy_r2"] - sub["ms_hgt"]["dy_r2"], 6),
                "dy_mae_delta": round(sub["hgt_global"]["dy_mae"] - sub["ms_hgt"]["dy_mae"], 10),
                "hgt_global_better_dy_r2": sub["hgt_global"]["dy_r2"] > sub["ms_hgt"]["dy_r2"],
            }

    # Region-wise: midspan key metric
    mid = {key: reg[key]["midspan"] for key in MODEL_KEYS}
    summary["midspan"] = {
        key: {"dy_r2": mid[key]["dy_r2"], "dy_mae": mid[key]["dy_mae"]}
        for key in MODEL_KEYS
    }
    summary["midspan"]["hgt_global_better_dy_r2"] = (
        mid["hgt_global"]["dy_r2"] > mid["ms_hgt"]["dy_r2"]
    )
    summary["midspan"]["hgt_global_better_dy_mae"] = (
        mid["hgt_global"]["dy_mae"] < mid["ms_hgt"]["dy_mae"]
    )

    # Span-position: mean Dy MAE in the midspan band [1/3, 2/3]
    mid_band = est[(est["x_norm_low"] >= 1 / 3) & (est["x_norm_low"] < 2 / 3)]
    summary["span_midspan_band_mean_dy_mae"] = {
        key: round(float(mid_band[f"{key}_dy_mae"].mean()), 10) for key in MODEL_KEYS
    }

    # Response-magnitude tail bin (largest response)
    tail = emt[emt["bin"] == N_QUANTILE_BINS - 1]
    summary["response_tail_bin_dy_mae"] = {}
    for kind in ("|Dy|", "translational norm"):
        sub = tail[tail["magnitude_def"] == kind]
        summary["response_tail_bin_dy_mae"][kind] = {
            key: round(float(sub[sub["model"] == key]["dy_mae"].iloc[0]), 10)
            for key in MODEL_KEYS
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HGT-global vs MS-HGT (macro-anchor) decision analysis on full "
                    "test-set prediction exports.")
    parser.add_argument("--hgt-global-pred", required=True,
                        help="Prediction dir for hgt_global (NPZ + test_graph_index.csv)")
    parser.add_argument("--ms-hgt-pred", required=True,
                        help="Prediction dir for MS-HGT (coordinate gated)")
    parser.add_argument("--hgt-pred", required=True,
                        help="Prediction dir for HGT (reference)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir; a timestamped subdir is created")
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / datetime.now().strftime("%Y%m%d%H%M%S"))
    print(f"Output dir: {out}")

    print("Loading predictions ...")
    models = {
        "hgt_global": load_model(args.hgt_global_pred),
        "ms_hgt": load_model(args.ms_hgt_pred),
        "hgt": load_model(args.hgt_pred),
    }
    check_aligned(models)

    print("Per-design metrics ...")
    df = per_design_table(models)
    df.to_csv(out / "per_design_metrics.csv", index=False)
    h2h = head_to_head_table(df)
    h2h.to_csv(out / "per_design_head_to_head.csv", index=False)
    print(f"  per-design head-to-head (n/7 won by HGT-global): "
          f"{h2h.iloc[-1]['dy_r2_hgt_global_wins']} (Dy R2)")

    print("High-response subsets ...")
    hr_df, hr = high_response_table(models)
    hr_df.to_csv(out / "high_response_metrics.csv", index=False)
    save_json(hr, out / "high_response_metrics.json")

    print("Region-wise ...")
    reg_df, reg = region_table(models)
    reg_df.to_csv(out / "region_metrics.csv", index=False)
    save_json(reg, out / "region_metrics.json")

    print("Error vs span position ...")
    est, _, _ = error_vs_span_table(models)
    est.to_csv(out / "error_vs_span_table.csv", index=False)
    plot_span_curves(est, out)

    print("Error vs response magnitude ...")
    emt = error_vs_magnitude_table(models)
    emt.to_csv(out / "error_vs_magnitude_table.csv", index=False)
    plot_magnitude_curves(emt, out)

    print("Aggregating summary ...")
    summary = build_summary(df, h2h, hr, reg, est, emt)
    save_json(summary, out / "analysis_summary.json")

    # ---- Markdown summary ----
    lines = [
        "# Stage Paper-4B/4C: HGT-global vs MS-HGT decision summary",
        "",
        f"- Generated: {summary['timestamp']}",
        "- Diagnostic only — shared-topology test set (by_sample, 7 designs), "
        "no cross-topology claim.",
        "",
        "## Per-design head-to-head (7 test designs)",
        "",
        "n designs where HGT-global beats MS-HGT (out of 7):",
        "",
        "| Metric | n/7 won by HGT-global |",
        "|---|---|",
    ]
    for m in ["disp_macro_r2", "dy_r2", "force_macro_r2", "combined_rel_mae"]:
        lines.append(f"| {m} | {h2h.iloc[-1].get(f'{m}_hgt_global_wins', 'n/a')} |")

    lines += ["", "## Midspan (long-range-sensitive region)", "",
              "| Model | Dy R² | Dy MAE (m) |",
              "|---|---:|---:|"]
    for key in MODEL_KEYS:
        r = summary["midspan"][key]
        lines.append(f"| {MODEL_LABELS[key]} | {r['dy_r2']:.4f} | {r['dy_mae']:.3e} |")
    lines += [
        "",
        f"- HGT-global better on midspan Dy R²: "
        f"{summary['midspan']['hgt_global_better_dy_r2']}",
        f"- HGT-global better on midspan Dy MAE: "
        f"{summary['midspan']['hgt_global_better_dy_mae']}",
        "",
        "## High-response subsets (HGT-global − MS-HGT)",
        "",
        "| Subset | dy_r2 delta | dy_mae delta | HGT-global better Dy R² |",
        "|---|---:|---:|---|",
    ]
    for set_name in ["trans_top10", "trans_top20", "dy_top10", "dy_top20"]:
        r = summary["high_response"].get(set_name)
        if r:
            lines.append(
                f"| {set_name} | {r['dy_r2_delta']:+.4f} | {r['dy_mae_delta']:+.2e} "
                f"| {r['hgt_global_better_dy_r2']} |")
    lines += ["", "## Midspan band mean Dy MAE (span bins in [1/3, 2/3])", ""]
    for key in MODEL_KEYS:
        lines.append(f"- {MODEL_LABELS[key]}: "
                     f"{summary['span_midspan_band_mean_dy_mae'][key]:.3e}")
    lines += ["", "## Tail response bin (largest 10% by magnitude)", ""]
    for kind in ("|Dy|", "translational norm"):
        row = summary["response_tail_bin_dy_mae"][kind]
        lines.append(f"- {kind}: " + ", ".join(
            f"{MODEL_LABELS[k]}={v:.3e}" for k, v in row.items()))
    lines += [
        "",
        "## Reading",
        "",
        "- If HGT-global wins *everywhere* (overall + midspan + high-response + "
        "tail), the macro-anchor thesis is weakened and HGT-global becomes a "
        "strong baseline (see decision_rules.md Rule A).",
        "- If HGT-global wins overall but MS-HGT retains midspan / high-response / "
        "tail gains, keep the regional advantage (Rule B).",
        "- 7 designs is a small n; treat any 5-2 or 4-3 split as directionally "
        "suggestive, not conclusive. 3-seed stability (seeds 2026/3407) is "
        "required before a paper claim.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nKey readings:")
    print(f"  per-design Dy R2 wins: {h2h.iloc[-1]['dy_r2_hgt_global_wins']}")
    print(f"  midspan Dy R2: HGT-global {summary['midspan']['hgt_global']['dy_r2']:.4f} "
          f"vs MS-HGT {summary['midspan']['ms_hgt']['dy_r2']:.4f}")
    print(f"  outputs -> {out}")


if __name__ == "__main__":
    main()

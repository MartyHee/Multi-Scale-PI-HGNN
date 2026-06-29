"""
analyze_conformal.py — Coverage analysis and reporting for Stage 6 UQ.

Reads compute_conformal.py output and produces:
  - stage6_uq_report.md (full markdown report with tables)
  - coverage_width_plot.png (coverage vs interval width trade-off)
  - component_coverage_bar.png (per-DOF/force-component coverage)
  - region_coverage_bar.png (region-wise coverage)
  - high_response_coverage.png (coverage degradation across response quantiles)

Usage:
    python scripts/analyze_conformal.py
        --conformal-dir outputs/diagnostics/stage6_uq/<timestamp>
        --output-dir outputs/diagnostics/stage6_uq/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISP_COMP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_COMP_NAMES = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(val, decimals: int = 6) -> str:
    """Format a value, handling NaN and string inputs."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return "N/A"
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_pct(val, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val * 100:.{decimals}f}%"
    return str(val)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: Dict,
    split_info: Dict,
    output_dir: Path,
) -> str:
    """Generate full markdown report.

    Returns: report text.
    """
    alpha_list = sorted(results.keys())
    lines = []

    lines.append("# Stage 6 UQ Report — Split Conformal Prediction\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- Section 1: Overview ----
    lines.append("## 1. Overview\n")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Backbone | MS-PI-HGT Full (MSHGTBaseline, 893,527 params) |")
    lines.append(f"| UQ Method | Component-wise split conformal prediction |")
    lines.append(f"| Nonconformity score | Absolute residual |")
    lines.append(f"| Split mode | {split_info.get('split_mode', '?')} |")
    lines.append(f"| Calibration graphs | {split_info.get('n_cal_graphs', '?')} |")
    lines.append(f"| Evaluation graphs | {split_info.get('n_eval_graphs', '?')} |")
    lines.append(f"| Stratified by SampleID | {split_info.get('stratified', '?')} |")
    lines.append(f"| Alpha values | {', '.join(_fmt(a, 2) for a in alpha_list)} |")
    lines.append(f"| Finite-sample correction | True |")
    lines.append("")

    # ---- Section 2: Per-Component Coverage ----
    lines.append("## 2. Marginal Coverage — Displacement\n")
    lines.append("| Component | Alpha | Target | Empirical | Gap | Width | Width/Med|y||")
    lines.append("|-----------|:-----:|:-----:|:---------:|:---:|:----:|:-------:|")

    for alpha in alpha_list:
        r = results[alpha]
        dc = r.get("displacement_coverage", {})
        dw = r.get("displacement_width", {})
        for comp in DISP_COMP_NAMES:
            m = dc.get(comp, {})
            w = dw.get(comp, {})
            emp = _fmt_pct(m.get("empirical_coverage"))
            gap = _fmt_pct(m.get("coverage_gap"), 3)
            width = _fmt(m.get("interval_width"))
            w_rel = _fmt(w.get("width_over_median_abs_y", ""), 2)
            lines.append(
                f"| {comp} | {_fmt(alpha,2)} | {_fmt_pct(m.get('target_coverage'))} "
                f"| {emp} | {gap} | {width} | {w_rel} |"
            )
        # ALL aggregate
        m = dc.get("ALL", {})
        w = dw.get("ALL", {})
        emp = _fmt_pct(m.get("empirical_coverage"))
        gap = _fmt_pct(m.get("coverage_gap"), 3)
        width = _fmt(m.get("interval_width"))
        w_rel = _fmt(w.get("width_over_median_abs_y", ""), 2)
        lines.append(
            f"| **ALL** | {_fmt(alpha,2)} | {_fmt_pct(m.get('target_coverage'))} "
            f"| {emp} | {gap} | {width} | {w_rel} |"
        )
        lines.append("")

    lines.append("## 3. Marginal Coverage — Force\n")
    lines.append("| Component | Alpha | Target | Empirical | Gap | Width | Width/Med|y||")
    lines.append("|-----------|:-----:|:-----:|:---------:|:---:|:----:|:-------:|")

    for alpha in alpha_list:
        r = results[alpha]
        fc = r.get("force_coverage", {})
        fw = r.get("force_width", {})
        for comp in FORCE_COMP_NAMES:
            m = fc.get(comp, {})
            w = fw.get(comp, {})
            emp = _fmt_pct(m.get("empirical_coverage"))
            gap = _fmt_pct(m.get("coverage_gap"), 3)
            width = _fmt(m.get("interval_width"))
            w_rel = _fmt(w.get("width_over_median_abs_y", ""), 2)
            lines.append(
                f"| {comp} | {_fmt(alpha,2)} | {_fmt_pct(m.get('target_coverage'))} "
                f"| {emp} | {gap} | {width} | {w_rel} |"
            )
        m = fc.get("ALL", {})
        w = fw.get("ALL", {})
        emp = _fmt_pct(m.get("empirical_coverage"))
        gap = _fmt_pct(m.get("coverage_gap"), 3)
        width = _fmt(m.get("interval_width"))
        w_rel = _fmt(w.get("width_over_median_abs_y", ""), 2)
        lines.append(
            f"| **ALL** | {_fmt(alpha,2)} | {_fmt_pct(m.get('target_coverage'))} "
            f"| {emp} | {gap} | {width} | {w_rel} |"
        )
        lines.append("")

    # ---- Section 4: Conformal Quantiles ----
    lines.append("## 4. Conformal Quantiles\n")
    for alpha in alpha_list:
        r = results[alpha]
        lines.append(f"### Alpha = {_fmt(alpha, 2)}\n")
        lines.append("**Displacement:**\n")
        lines.append("| Component | n_cal | k | Quantile |")
        lines.append("|-----------|:-----:|:--:|:--------:|")
        for comp, info in r.get("displacement_quantiles", {}).items():
            lines.append(
                f"| {comp} | {info.get('n_cal_points', '?')} | {info.get('k', '?')} "
                f"| {_fmt(info.get('quantile'))} |"
            )
        lines.append("")
        lines.append("**Force:**\n")
        lines.append("| Component | n_cal | k | Quantile |")
        lines.append("|-----------|:-----:|:--:|:--------:|")
        for comp, info in r.get("force_quantiles", {}).items():
            lines.append(
                f"| {comp} | {info.get('n_cal_points', '?')} | {info.get('k', '?')} "
                f"| {_fmt(info.get('quantile'))} |"
            )
        lines.append("")

    # ---- Section 5: Region Coverage ----
    lines.append("## 5. Region-Wise Coverage (Displacement)\n")
    lines.append("Only evaluation-set nodes. Coverage = all 6 DOF simultaneously covered.\n")

    for alpha in alpha_list:
        r = results[alpha]
        rc = r.get("region_coverage", {})
        lines.append(f"### Alpha = {_fmt(alpha, 2)}\n")
        lines.append("| Region | n_nodes | Coverage | Gap | Width |")
        lines.append("|--------|:-------:|:--------:|:---:|:----:|")
        for region_name in ["support", "free"] + [f"x_region_Q{q+1}" for q in range(5)]:
            m = rc.get(region_name, {})
            nn = m.get("n_nodes", 0)
            if nn == 0:
                continue
            cov = _fmt_pct(m.get("empirical_coverage"))
            gap = _fmt_pct(m.get("coverage_gap"), 3)
            width = _fmt(m.get("interval_width"))
            lines.append(f"| {region_name} | {nn} | {cov} | {gap} | {width} |")
        lines.append("")

    # ---- Section 6: High-Response Coverage ----
    lines.append("## 6. High-Response Coverage (Displacement)\n")
    lines.append("Coverage on subsets defined by true displacement magnitude.\n")

    for alpha in alpha_list:
        r = results[alpha]
        hr = r.get("high_response_coverage", {})
        lines.append(f"### Alpha = {_fmt(alpha, 2)}\n")
        lines.append("| Subset | n_nodes | Threshold | Coverage | Gap | P95 Residual |")
        lines.append("|--------|:-------:|:---------:|:--------:|:---:|:------------:|")
        for name, m in hr.items():
            nn = m.get("n_nodes", 0)
            if nn == 0:
                continue
            thr = _fmt(m.get("threshold"))
            cov = _fmt_pct(m.get("empirical_coverage"))
            gap = _fmt_pct(m.get("coverage_gap"), 3)
            p95 = _fmt(m.get("p95_residual"))
            lines.append(f"| {name} | {nn} | {thr} | {cov} | {gap} | {p95} |")
        lines.append("")

    # ---- Section 7: Graph-Level Conformal ----
    lines.append("## 7. Graph-Level Conformal\n")
    lines.append("Component-wise graph-level: per-component max residual per graph, then calibrate across graphs.\n")
    lines.append("Normalized graph-level: residual/scale per component, then max across components per node, then max per graph.\n")

    for alpha in alpha_list:
        r = results[alpha]
        lines.append(f"### Alpha = {_fmt(alpha, 2)}\n")
        for target_key, target_name in [("graph_level_conformal_disp", "Displacement"),
                                         ("graph_level_conformal_force", "Force")]:
            gl = r.get(target_key, {})
            lines.append(f"**{target_name}:**\n")
            lines.append("| Variant | n_cal_graphs | n_eval_graphs | Quantile | Coverage | Gap |")
            lines.append("|---------|:------------:|:-------------:|:--------:|:--------:|:---:|")
            for name, m in gl.items():
                ncg = m.get("n_cal_graphs", "?")
                neg = m.get("n_eval_graphs", "?")
                q = _fmt(m.get("quantile"))
                cov = _fmt_pct(m.get("empirical_coverage"))
                gap = _fmt_pct(m.get("coverage_gap"), 3)
                norm_tag = " (normalized)" if m.get("normalized") else ""
                lines.append(f"| {name}{norm_tag} | {ncg} | {neg} | {q} | {cov} | {gap} |")
            lines.append("")

    # ---- Section 8: Summary ----
    lines.append("## 8. Summary\n")
    lines.append("### Pass/Fail against Success Criteria\n")
    lines.append("| Criterion | Target | Status |")
    lines.append("|-----------|:------:|:------:|")

    for alpha in alpha_list:
        r = results[alpha]
        dc = r.get("displacement_coverage", {}).get("ALL", {})
        fc = r.get("force_coverage", {}).get("ALL", {})
        d_cov = dc.get("empirical_coverage", 0)
        f_cov = fc.get("empirical_coverage", 0)

        for crit_name, actual, target, pass_if_ge in [
            (f"90% disp marginal coverage (α={alpha})",
             d_cov, 0.90, d_cov >= 0.90),
            (f"95% disp marginal coverage (α={alpha})",
             d_cov, 0.95, d_cov >= 0.95),
            (f"90% force marginal coverage (α={alpha})",
             f_cov, 0.90, f_cov >= 0.90),
            (f"95% force marginal coverage (α={alpha})",
             f_cov, 0.95, f_cov >= 0.95),
        ]:
            if alpha == 0.10 and "90%" in crit_name:
                actual_fmt = _fmt_pct(actual, 1)
                target_fmt = _fmt_pct(target, 1)
                status = "✅ PASS" if pass_if_ge else "❌ FAIL"
                lines.append(f"| {crit_name} | ≥ {target_fmt} | {status} ({actual_fmt}) |")
            elif alpha == 0.05 and "95%" in crit_name:
                actual_fmt = _fmt_pct(actual, 1)
                target_fmt = _fmt_pct(target, 1)
                status = "✅ PASS" if pass_if_ge else "❌ FAIL"
                lines.append(f"| {crit_name} | ≥ {target_fmt} | {status} ({actual_fmt}) |")

    lines.append("")

    # Per-DOF gap check
    lines.append("#### Per-DOF Coverage Gaps\n")
    for alpha in alpha_list:
        r = results[alpha]
        lines.append(f"α={_fmt(alpha, 2)}:\n")
        for comp in DISP_COMP_NAMES:
            m = r.get("displacement_coverage", {}).get(comp, {})
            gap = m.get("coverage_gap", 0)
            status = "✅" if gap >= -0.02 else "❌"
            lines.append(f"- {status} {comp}: gap = {_fmt_pct(gap, 2)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        return True
    except ImportError:
        return False


def plot_coverage_width(results: Dict, output_dir: Path) -> bool:
    """Coverage vs interval width trade-off."""
    if not _check_matplotlib():
        return False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target_name, comp_names, cov_key, width_key in [
        (axes[0], "Displacement", DISP_COMP_NAMES,
         "displacement_coverage", "displacement_width"),
        (axes[1], "Force", FORCE_COMP_NAMES,
         "force_coverage", "force_width"),
    ]:
        alpha_vals = sorted(results.keys())
        for comp in comp_names:
            covs = []
            widths = []
            for a in alpha_vals:
                r = results[a]
                m = r.get(cov_key, {}).get(comp, {})
                w = r.get(width_key, {}).get(comp, {})
                covs.append(m.get("empirical_coverage", 0))
                widths.append(w.get("interval_width", 0))
            label = comp.replace("_", " ")
            ax.plot(widths, covs, "o-", label=label)

        ax.axhline(y=1.0 - min(alpha_vals), color="gray", linestyle="--", alpha=0.5,
                   label=f"target (α={min(alpha_vals)})")
        ax.set_xlabel("Interval width")
        ax.set_ylabel("Coverage")
        ax.set_title(f"{target_name} — Coverage vs Width")
        ax.legend(fontsize="x-small", loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_width_plot.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'coverage_width_plot.png'}")
    return True


def plot_component_coverage(results: Dict, output_dir: Path) -> bool:
    """Per-component coverage bar chart."""
    if not _check_matplotlib():
        return False
    import matplotlib.pyplot as plt

    for target_name, comp_names, cov_key in [
        ("Displacement", DISP_COMP_NAMES, "displacement_coverage"),
        ("Force", FORCE_COMP_NAMES, "force_coverage"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, len(comp_names) * 0.6), 5))

        alpha_vals = sorted(results.keys())
        x = np.arange(len(comp_names))
        width = 0.8 / max(len(alpha_vals), 1)

        for i, a in enumerate(alpha_vals):
            r = results[a]
            covs = []
            for comp in comp_names:
                m = r.get(cov_key, {}).get(comp, {})
                covs.append(m.get("empirical_coverage", 0))
            offset = (i - len(alpha_vals) / 2 + 0.5) * width
            ax.bar(x + offset, covs, width, label=f"α={a}")

        ax.axhline(y=1.0 - min(alpha_vals), color="red", linestyle="--",
                   alpha=0.7, label=f"target (α={min(alpha_vals)})")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_names, rotation=45, ha="right")
        ax.set_ylabel("Coverage")
        ax.set_title(f"{target_name} — Per-Component Coverage")
        ax.legend(fontsize="small")
        ax.set_ylim(0.7, 1.02)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = f"component_coverage_bar_{target_name.lower()}.png"
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / fname}")

    return True


def plot_region_coverage(results: Dict, output_dir: Path) -> bool:
    """Region-wise coverage bar chart."""
    if not _check_matplotlib():
        return False
    import matplotlib.pyplot as plt

    for alpha in results:
        r = results[alpha]
        rc = r.get("region_coverage", {})
        regions = []
        covs = []
        for name in ["support", "free"] + [f"x_region_Q{q+1}" for q in range(5)]:
            m = rc.get(name)
            if m and m.get("n_nodes", 0) > 0:
                regions.append(name.replace("x_region_", ""))
                covs.append(m.get("empirical_coverage", 0))

        if not regions:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2ecc71" if c >= 0.9 else "#e74c3c" for c in covs]
        ax.bar(range(len(regions)), covs, color=colors)
        ax.axhline(y=1.0 - alpha, color="red", linestyle="--",
                   alpha=0.7, label=f"target (α={alpha})")
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=30, ha="right")
        ax.set_ylabel("Coverage")
        ax.set_title(f"Region-Wise Coverage (α={alpha})")
        ax.legend()
        ax.set_ylim(0.7, 1.02)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = f"region_coverage_bar_alpha_{alpha}.png"
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / fname}")

    return True


def plot_high_response_coverage(results: Dict, output_dir: Path) -> bool:
    """Coverage degradation across response quantiles."""
    if not _check_matplotlib():
        return False
    import matplotlib.pyplot as plt

    for alpha in results:
        r = results[alpha]
        hr = r.get("high_response_coverage", {})

        # Extract true-response coverage at different thresholds
        subsets = []
        covs = []
        names = ["true_response_low_90pct", "true_response_top_10pct",
                 "true_response_top_5pct", "true_response_top_1pct"]
        labels = ["Low 90%", "Top 10%", "Top 5%", "Top 1%"]

        for name, label in zip(names, labels):
            m = hr.get(name)
            if m and m.get("n_nodes", 0) > 0:
                subsets.append(label)
                covs.append(m.get("empirical_coverage", 0))

        if not subsets:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#2ecc71" if c >= 0.85 else "#e74c3c" for c in covs]
        ax.bar(range(len(subsets)), covs, color=colors)
        ax.axhline(y=1.0 - alpha, color="red", linestyle="--",
                   alpha=0.7, label=f"target (α={alpha})")
        ax.set_xticks(range(len(subsets)))
        ax.set_xticklabels(subsets, rotation=0)
        ax.set_ylabel("Coverage")
        ax.set_title(f"High-Response Coverage Degradation (α={alpha})")
        ax.legend()
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = f"high_response_coverage_alpha_{alpha}.png"
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / fname}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyze conformal prediction results for Stage 6 UQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--conformal-dir", type=str, required=True,
        help="Path to compute_conformal.py output directory"
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as conformal-dir)"
    )
    return p.parse_args(argv)


def main(args: argparse.Namespace):
    print("=" * 60)
    print("STAGE 6 — CONFORMAL ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    conformal_dir = Path(args.conformal_dir)
    if not conformal_dir.exists():
        raise FileNotFoundError(f"Conformal directory not found: {conformal_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else conformal_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConformal dir: {conformal_dir}")
    print(f"Output dir:    {output_dir}")

    # ---- Load data ----
    results_path = conformal_dir / "conformal_results_raw.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"conformal_results_raw.json not found. Run compute_conformal.py first."
        )
    results = _load_json(results_path)

    manifest_path = conformal_dir / "split_manifest.json"
    split_info = _load_json(manifest_path) if manifest_path.exists() else {}

    print(f"  Loaded {len(results)} alpha values: {list(results.keys())}")
    print(f"  Split info: {split_info.get('split_mode', '?')}")

    # Normalise dict keys from string to float (JSON serialises keys as strings)
    results_float = {}
    for k, v in results.items():
        try:
            results_float[float(k)] = v
        except (ValueError, TypeError):
            results_float[k] = v
    results = results_float

    # ---- Generate report ----
    print(f"\nGenerating report...")
    report = generate_report(results, split_info, output_dir)

    report_path = output_dir / "stage6_uq_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # ---- Generate plots ----
    print(f"\nGenerating plots...")
    has_mpl = _check_matplotlib()
    if not has_mpl:
        print(f"  [WARN] matplotlib not available — skipping plots.")

    plot_coverage_width(results, output_dir)
    plot_component_coverage(results, output_dir)
    plot_region_coverage(results, output_dir)
    plot_high_response_coverage(results, output_dir)

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Report: {report_path}")
    n_plots = len(list(output_dir.glob("*_plot.png"))) + len(list(output_dir.glob("*_bar.png")))
    print(f"  Plots:  {n_plots}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args_parsed = parse_args()
    main(args_parsed)

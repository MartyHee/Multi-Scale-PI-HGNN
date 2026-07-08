"""
F6: UQ Coverage vs Half-Width Chart
Two-panel figure:
  Left: Per-DOF Displacement coverage bars with half-width overlay
  Right: Force coverage summary
Panel 3 (optional): Region coverage table visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import save_figure

# === Data from experiment_result_and_figure_audit.md Section 2.4 ===
DOF_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
DOF_COV_90  = [89.59, 89.75, 89.66, 89.85, 89.73, 89.87]  # percentages
DOF_WIDTH_90 = [0.000453, 0.000146, 0.001983, 0.000097, 0.000156, 0.000021]  # half-width (m/rad)

FORCE_ALL_COV_90 = 89.97
FORCE_ALL_WIDTH_90 = 40136.0
FORCE_ALL_COV_95 = 94.97
FORCE_ALL_WIDTH_95 = 69715.0

# Joint coverage (region table)
REGIONS = ["Support", "Free", "Q1 (End)", "Q3 (Midspan)", "Q5 (End)", "High-Resp"]
JOINT_COV = [25.2, 55.1, 49.5, 57.4, 53.1, 28.4]
PER_DOF_COV = [90, 90, 90, 90, 90, 85]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.0))

# === Panel 1: Per-DOF Displacement Coverage (α=0.10) ===
x = np.arange(len(DOF_NAMES))
width = 0.5
bars1 = ax1.bar(x, DOF_COV_90, width, color="#377eb8", edgecolor="white", linewidth=0.5, alpha=0.85)
ax1.axhline(y=90, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, label="Target 90%")
ax1.set_xticks(x)
ax1.set_xticklabels(DOF_NAMES, fontsize=9)
ax1.set_ylabel("Empirical Coverage (%)", fontsize=10)
ax1.set_title("Displacement Per-DOF Coverage (α=0.10)", fontsize=9, fontweight="bold")
ax1.set_ylim(88, 92)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="y", alpha=0.3, linestyle=":")

# Annotate max gap from 90%
for i, (bar, cov) in enumerate(zip(bars1, DOF_COV_90)):
    gap = cov - 90
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
            f"{gap:+.2f}%", ha="center", va="bottom", fontsize=7,
            color="#e41a1c" if abs(gap) > 0.5 else "#4daf4a")

# === Panel 2: Displacement half-width (per-DOF, log-scale to handle wide range) ===
DOF_WIDTH_LABELS = [f"{w:.1e}" for w in DOF_WIDTH_90]
ax2.bar(x, DOF_WIDTH_90, width, color="#ff7f00", edgecolor="white", linewidth=0.5, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(DOF_NAMES, fontsize=9)
ax2.set_ylabel("Avg Half-Width (m / rad)", fontsize=10)
ax2.set_title("Displacement Half-Width (α=0.10)", fontsize=9, fontweight="bold")
ax2.set_yscale("log")
ax2.grid(axis="y", alpha=0.3, linestyle=":")

for i, w in enumerate(DOF_WIDTH_90):
    ax2.text(i, w * 1.5, DOF_WIDTH_LABELS[i], ha="center", va="bottom", fontsize=7)

# === Panel 3: Region Joint Coverage ===
x3 = np.arange(len(REGIONS))
ax3.bar(x3, JOINT_COV, width, color="#984ea3", edgecolor="white", linewidth=0.5, alpha=0.85)
ax3.set_xticks(x3)
ax3.set_xticklabels(REGIONS, fontsize=8, rotation=20, ha="right")
ax3.set_ylabel("Joint 6-DOF Coverage (%)", fontsize=10)
ax3.set_title("Region Joint Coverage (Diagnostic, α=0.10)", fontsize=9, fontweight="bold")
ax3.set_ylim(0, 100)
ax3.axhline(y=90, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="Marginal 90%")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3, linestyle=":")

for i, (bar, cov) in enumerate(zip(ax3.patches, JOINT_COV)):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f"{cov:.1f}%", ha="center", va="bottom", fontsize=7)

fig.suptitle("Conformal Uncertainty Quantification — MS-PI-HGT-Full (test_graph_50_50 split)",
             fontsize=11, fontweight="bold", y=1.03)
fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F6_uq_coverage_halfwidth"
save_figure(fig, out)
plt.close(fig)
print("F6 done.")

"""
F5: Physics Ablation Bar Chart
Two-panel figure:
  Left: Constrained DOF MAE (lower is better)
  Right: Force P95 AE (lower is better)
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import PHYSICS_LABELS_SHORT, PHYSICS_COLORS, save_figure

# === Data (from experiment_result_and_figure_audit.md Section 2.3) ===
VARIANTS = PHYSICS_LABELS_SHORT  # ["Baseline", "BC Only", "Link Only", "Full"]
DOF_MAE  = [0.000242, 0.000188, 0.000277, 0.000148]  # Constrained DOF MAE (m)
P95_AE   = [38000, 39756, 39277, 37917]              # Force P95 AE (N/N·m)

N = len(VARIANTS)
x = np.arange(N)
width = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.8))

# Left panel: Constrained DOF MAE
bars1 = ax1.bar(x, DOF_MAE, width, color=PHYSICS_COLORS, edgecolor="white", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(VARIANTS, fontsize=9, rotation=15, ha="right")
ax1.set_ylabel("Constrained DOF MAE (m)", fontsize=10)
ax1.set_title("BC Constraint Satisfaction", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, linestyle=":")
# Add reduction % annotations
pct_reduction = [0, -22.3, 14.5, -38.8]  # relative to Baseline
for i, (bar, pct) in enumerate(zip(bars1, pct_reduction)):
    h = bar.get_height()
    if pct < 0:
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.000008, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#4daf4a", fontweight="bold")
    elif pct > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.000008, f"+{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#e41a1c")

# Right panel: Force P95 AE
bars2 = ax2.bar(x, P95_AE, width, color=PHYSICS_COLORS, edgecolor="white", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(VARIANTS, fontsize=9, rotation=15, ha="right")
ax2.set_ylabel("Force P95 Absolute Error (N/N·m)", fontsize=10)
ax2.set_title("Force Tail Error", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.3, linestyle=":")

pct_reduction_f = [0, 4.6, 3.4, -0.2]
for i, (bar, pct) in enumerate(zip(bars2, pct_reduction_f)):
    h = bar.get_height()
    if pct < 0:
        ax2.text(bar.get_x() + bar.get_width()/2., h + 200, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#4daf4a", fontweight="bold")
    elif pct > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., h + 200, f"+{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#e41a1c")

fig.suptitle("Physics Regularization Ablation", fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F5_physics_ablation_bar"
save_figure(fig, out)
plt.close(fig)
print("F5 done.")

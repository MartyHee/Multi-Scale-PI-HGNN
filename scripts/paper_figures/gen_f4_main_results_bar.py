"""
F4: Main Results Bar Chart
Grouped bar chart: Disp R², Dy R², Force R² across all 7 models.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import MODEL_ORDER, MODEL_COLORS, save_figure

# === Data (from README.md Main Results Summary, locked) ===
DISP_R2  = [0.8554, 0.8476, 0.8421, 0.9366, 0.9765, 0.9952, 0.9948]
DY_R2    = [0.1833, 0.1778, 0.1649, 0.670,  0.905,  0.993,  0.993]
FORCE_R2 = [0.9824, 0.9696, 0.9632, 0.9878, 0.9893, 0.9928, 0.9933]

N = len(MODEL_ORDER)
x = np.arange(N)
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))

bars1 = ax.bar(x - width, DISP_R2,  width, label="Disp R²", color="#377eb8", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x,        DY_R2,    width, label="Dy R²",    color="#ff7f00", edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + width, FORCE_R2, width, label="Force R²", color="#4daf4a", edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("R² Score", fontsize=11)
ax.set_ylim(0, 1.08)
ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

# Add value labels on bars
def add_labels(bars, fmt=".4f"):
    for bar in bars:
        h = bar.get_height()
        if h < 0.3:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

add_labels(bars1, ".4f")
add_labels(bars2, ".3f")
add_labels(bars3, ".4f")

ax.legend(loc="lower right", fontsize=9, ncol=3)
ax.set_title("Model Comparison on Structural FE Surrogate Prediction", fontsize=11, fontweight="bold")
ax.set_ylabel("R² ↑")
ax.grid(axis="y", alpha=0.3, linestyle=":")

# Separator line between homogeneous and heterogeneous
ax.axvline(x=2.5 + width, color="grey", linewidth=1.5, linestyle="--", alpha=0.6)
ax.text(2.55, 1.04, "Homogeneous → Heterogeneous →", fontsize=8, color="grey", rotation=90, va="top")

fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F4_main_results_bar"
save_figure(fig, out)
plt.close(fig)
print("F4 done.")

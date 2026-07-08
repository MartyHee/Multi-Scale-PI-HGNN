"""
Shared style configuration for ICTAI 2026 paper figures.
Defines colorblind-friendly palette, font sizes, and common settings.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Colorblind-friendly palette (ColorBrewer Set1-inspired, Paul Tol's scheme) ===
CB_COLORS = {
    "MLP":         "#377eb8",   # blue
    "GCN":         "#ff7f00",   # orange
    "GAT":         "#4daf4a",   # green
    "RGCN":        "#f781bf",   # pink
    "HGT":         "#a65628",   # brown
    "MS-HGT":      "#984ea3",   # purple
    "MS-PI-HGT":   "#e41a1c",   # red
    "Baseline":    "#999999",   # grey
    "BC Only":     "#377eb8",   # blue
    "Link Only":   "#ff7f00",   # orange
    "Full":        "#4daf4a",   # green
}

MODEL_ORDER = ["MLP", "GCN", "GAT", "RGCN", "HGT", "MS-HGT", "MS-PI-HGT"]
MODEL_COLORS = [CB_COLORS[m] for m in MODEL_ORDER]

PHYSICS_MODEL_ORDER = ["MS-HGT gated", "MS-PI-HGT-BC", "MS-PI-HGT-Link", "MS-PI-HGT-Full"]
PHYSICS_LABELS_SHORT = ["Baseline", "BC Only", "Link Only", "Full"]
PHYSICS_COLORS = [CB_COLORS["Baseline"], CB_COLORS["BC Only"], CB_COLORS["Link Only"], CB_COLORS["Full"]]

# === Font settings for paper readability ===
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# === Metric display helpers ===
DISP_LABEL = "Disp R²"
DY_LABEL = "Dy R²"
FORCE_LABEL = "Force R²"
RELMAE_LABEL = "RelMAE"

def save_figure(fig, path_stem, dpi=300):
    """Save figure as both PNG and SVG."""
    fig.savefig(f"{path_stem}.png", dpi=dpi, facecolor="white", edgecolor="none")
    fig.savefig(f"{path_stem}.svg", dpi=dpi, facecolor="white", edgecolor="none")
    print(f"  Saved: {path_stem}.png + .svg")

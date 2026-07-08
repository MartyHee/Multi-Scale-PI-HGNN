"""
F1: Overall Framework Pipeline Diagram
End-to-end pipeline: raw FE data → heterogeneous graph → MS-HGT → predictions → UQ
Styled pipeline using matplotlib patches and arrows.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import save_figure

fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

# ========== Pipeline Stages ==========
# Each stage: a rounded rectangle with title + brief content

def draw_stage(ax, x, y, w, h, color, title, lines, title_color="white"):
    """Draw a pipeline stage box."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor="#333333", linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    # Title bar
    title_bar = FancyBboxPatch((x, y + h - 0.55), w, 0.55,
                                boxstyle="round,pad=0.02",
                                facecolor="#222222", edgecolor="none", alpha=0.3)
    ax.add_patch(title_bar)
    ax.text(x + w/2, y + h - 0.08, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        # Replace Unicode chars that may not render in all fonts
        line_clean = line.replace("①", "[1]").replace("②", "[2]")
        ax.text(x + 0.12, y + h - 0.7 - i*0.28, line_clean, ha="left", va="top",
                    fontsize=7, color="white", alpha=0.95)

def draw_arrow(ax, x1, y1, x2, y2, color="#666666", label=""):
    """Draw a connecting arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0,
                               connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7, color=color,
                fontstyle="italic")

# Row positions (y centers)
ROW1 = 5.2  # Top row
ROW2 = 3.4  # Middle row
ROW3 = 1.6  # Bottom row

BOX_W = 2.6
BOX_H = 2.6

# === Row 1: Data Preparation ===
draw_stage(ax, 0.3, ROW1-1.2, BOX_W, BOX_H, "#377eb8",
    "Step 1:\nRaw FE Data",
    ["70 steel truss designs",
     "× 500 load cases",
     "= 35,000 graph instances",
     "",
     "Input tables:",
     "• nodes.csv",
     "• beam_elements.csv",
     "• plate_elements.csv",
     "• nodal_loads.csv",
     "• beam_sections.csv",
     "• rigid_elastic_links.csv",
     "• node_displacement.csv",
     "• beam_results.csv"])

draw_stage(ax, 3.4, ROW1-1.2, BOX_W, BOX_H, "#377eb8",
    "Step 2:\nGraph Construction",
    ["Heterogeneous graph:",
     "",
     "Node types (3):",
     "• mesh_node (1,056)",
     "• beam_element (1,646)",
     "• plate_element (832)",
     "",
     "Edge types (5):",
     "• belongs_to_beam",
     "• belongs_to_plate",
     "• structural_link"])

# === Row 2: Core Model ===
draw_stage(ax, 6.7, ROW2-1.4, 2.4, 2.8, "#984ea3",
    "Step 3:\nTyped Encoder",
    ["Type-specific linear",
     "projection per node type:",
     "",
     "• mesh_node: 15→latent",
     "• beam_element: 11→latent",
     "• plate_element: 6→latent",
     "",
     "HGT-style relation-",
     "specific QKV attention"])

draw_stage(ax, 9.4, ROW2-1.4, 2.4, 2.8, "#4daf4a",
    "Step 4:\nMicro Message\nPassing",
    ["L HGT layers on",
     "heterogeneous graph:",
     "",
     "Relation-specific",
     "message functions:",
     "• 5 edge types ×",
     "  W_msg, W_attn",
     "",
     "Node-wise residual",
     "update per layer"])

draw_stage(ax, 12.1, ROW2-1.4, 2.4, 2.8, "#e41a1c",
    "Step 5:\nMacro Anchor\nPooling",
    ["Stiffness-aware",
     "anchor construction:",
     "",
     "• 12 anchors via",
     "  x-coord clustering",
     "• macro graph: k-NN",
     "• SAGEConv on macro",
     "• gated cross-scale",
     "  fusion back to micro"])

# === Row 3: Output ===
draw_stage(ax, 7.6, ROW3-1.0, 2.6, 2.0, "#ff7f00",
    "Step 6:\nDual Decoder",
    ["Two MLP decoders:",
     "",
     "① mesh_node → 6-DOF",
     "   displacement",
     "",
     "② beam_element → 12-D",
     "   internal force"])

draw_stage(ax, 10.9, ROW3-1.0, 2.6, 2.0, "#a65628",
    "Step 7:\nPhysics Loss",
    ["Regularization:",
     "",
     "• Support BC loss:",
     "  penalize nonzero disp",
     "  at constrained DOFs",
     "",
     "• Link consistency:",
     "  penalize link disp gap"])

draw_stage(ax, 14.2, ROW3-1.0, 1.6, 2.0, "#f781bf",
    "Step 8:\nConformal UQ",
    ["Split conformal:",
     "",
     "• component-wise",
     "• α=0.10 & 0.05",
     "• marginal coverage",
     "• half-width intervals"])

# === Arrows between stages ===
draw_arrow(ax, 2.9, ROW1, 3.4, ROW1)
draw_arrow(ax, 6.0, ROW1, 6.7, ROW2)  # down to row 2
draw_arrow(ax, 9.1, ROW2, 9.4, ROW2)
draw_arrow(ax, 11.8, ROW2, 12.1, ROW2)
# Down to row 3
draw_arrow(ax, 7.6, ROW2-0.3, 7.6, ROW3+0.4, label="mesh_node rep →")
draw_arrow(ax, 10.7, ROW2-0.3, 10.7, ROW3+0.4, label="beam_element rep →")
# Along row 3
draw_arrow(ax, 10.2, ROW3, 10.9, ROW3)
draw_arrow(ax, 13.5, ROW3, 14.2, ROW3)

# === Title ===
ax.text(8, 6.8, "MS-PI-HGT: Multi-Scale Physics-Informed Heterogeneous Graph Transformer — Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(8, 6.5, "End-to-End Framework for Steel Truss FE Surrogate Modeling",
        ha="center", va="center", fontsize=9, color="grey")

# === Encoding info box ===
info_box = FancyBboxPatch((0.3, 0.1), 6.0, 1.0, boxstyle="round,pad=0.08",
                           facecolor="#f0f0f0", edgecolor="#999999", linewidth=1.0)
ax.add_patch(info_box)
ax.text(0.5, 0.85, "Key Design Principles:", fontsize=8, fontweight="bold")
ax.text(0.5, 0.55, "• Heterogeneous typed message passing preserves physical entity differentiation", fontsize=7)
ax.text(0.5, 0.30, "• Multi-scale macro-anchor fusion captures long-range force transfer across the full bridge",
        fontsize=7)
ax.text(3.5, 0.55, "• Physics-informed regularization enforces known structural constraints", fontsize=7)
ax.text(3.5, 0.30, "• Conformal UQ provides distribution-free uncertainty without retraining", fontsize=7)

fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F1_overall_framework_pipeline"
save_figure(fig, out)
plt.close(fig)
print("F1 done.")

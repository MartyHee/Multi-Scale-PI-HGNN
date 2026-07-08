"""
F2: Heterogeneous Graph Schema Illustration
Visual showing 3 node types, 5 edge types with counts, feature dimensions, and targets.
Uses matplotlib with patches for a clean schema diagram.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import save_figure

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# ========== Node Type Boxes ==========
def draw_node_box(ax, x, y, w, h, color, title, details, hatch=""):
    """Draw a styled node type box."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="#333333", linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.25, title, ha="center", va="top", fontsize=11,
            fontweight="bold", color="white")
    for i, line in enumerate(details):
        ax.text(x + 0.15, y + h - 0.6 - i*0.3, line, ha="left", va="top",
                fontsize=7.5, color="white", alpha=0.95)

# --- mesh_node (center-left) ---
draw_node_box(ax, 1.0, 3.0, 3.6, 2.8, "#377eb8", "mesh_node",
    ["Count: 1,056 per graph",
     "Features (15-dim):",
     "  xyz, loads, BC flags",
     "Target (6-dim):",
     "  Dx, Dy, Dz, Rx, Ry, Rz"])

# --- beam_element (upper-right) ---
draw_node_box(ax, 6.5, 4.8, 3.6, 2.2, "#ff7f00", "beam_element",
    ["Count: 1,646 per graph",
     "Features (11-dim):",
     "  section, material, geometry",
     "Target (12-dim):",
     "  Fx_I..Mz_J"])

# --- plate_element (lower-right) ---
draw_node_box(ax, 6.5, 1.2, 3.6, 1.8, "#4daf4a", "plate_element",
    ["Count: 832 per graph",
     "Features (6-dim):",
     "  thickness, material"])

# ========== Edge Types ==========
# Define connection points
mesh_center = (2.8, 4.4)
beam_center = (8.3, 5.9)
plate_center = (8.3, 2.1)

edge_color = "#666666"
edge_style = dict(arrowstyle="->", color=edge_color, lw=1.5, connectionstyle="arc3,rad=0.15")

# belongs_to_beam: mesh_node -> beam_element
ax.annotate("", xy=(8.0, 5.5), xytext=(4.7, 4.8),
            arrowprops=dict(arrowstyle="->", color="#ff7f00", lw=2, connectionstyle="arc3,rad=0.2"))
ax.text(6.2, 5.7, "belongs_to_beam\n(1056 edges)", ha="center", va="bottom",
        fontsize=7.5, color="#ff7f00", fontweight="bold")

# rev_belongs_to_beam: beam_element -> mesh_node
ax.annotate("", xy=(4.7, 4.2), xytext=(8.0, 5.3),
            arrowprops=dict(arrowstyle="->", color="#ff7f00", lw=2, connectionstyle="arc3,rad=0.2"))
ax.text(6.2, 4.3, "rev_belongs_to_beam\n(1646 edges)", ha="center", va="top",
        fontsize=7.5, color="#ff7f00", fontweight="bold")

# belongs_to_plate: mesh_node -> plate_element
ax.annotate("", xy=(8.0, 2.5), xytext=(4.7, 3.5),
            arrowprops=dict(arrowstyle="->", color="#4daf4a", lw=2, connectionstyle="arc3,rad=-0.15"))
ax.text(6.2, 3.5, "belongs_to_plate\n(1056 edges)", ha="center", va="bottom",
        fontsize=7.5, color="#4daf4a", fontweight="bold")

# rev_belongs_to_plate: plate_element -> mesh_node
ax.annotate("", xy=(4.7, 3.8), xytext=(8.0, 1.8),
            arrowprops=dict(arrowstyle="->", color="#4daf4a", lw=2, connectionstyle="arc3,rad=-0.15"))
ax.text(6.2, 2.3, "rev_belongs_to_plate\n(832 edges)", ha="center", va="top",
        fontsize=7.5, color="#4daf4a", fontweight="bold")

# structural_link: mesh_node <-> mesh_node (self-loop style)
# Draw two nodes representing mesh_node self-connection
link_color = "#e41a1c"
# Left node
ax.annotate("", xy=(0.7, 2.5), xytext=(1.0, 2.8),
            arrowprops=dict(arrowstyle="->", color=link_color, lw=2.5,
                           connectionstyle="arc3,rad=-0.8"))
ax.annotate("", xy=(1.0, 3.2), xytext=(0.7, 2.9),
            arrowprops=dict(arrowstyle="->", color=link_color, lw=2.5,
                           connectionstyle="arc3,rad=-0.8"))
ax.text(0.15, 2.65, "structural_link\n(132 directed,\n66 physical)\n10-dim attrs:\nKx..Krz, β, dist",
        ha="left", va="center", fontsize=7, color=link_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=link_color))

# ========== Legend ==========
legend_x, legend_y = 0.5, 0.2
legend_items = [
    ("#377eb8", "mesh_node — displacement"),
    ("#ff7f00", "beam_element — internal force"),
    ("#4daf4a", "plate_element — input only"),
    ("#e41a1c", "structural_link (rigid connection)"),
]
for i, (c, lbl) in enumerate(legend_items):
    ax.add_patch(mpatches.Circle((legend_x + 0.2, legend_y - i*0.3), 0.08,
                                  facecolor=c, edgecolor="#333", linewidth=0.5))
    ax.text(legend_x + 0.4, legend_y - i*0.3, lbl, fontsize=8, va="center")

# Title
ax.text(6, 7.6, "Heterogeneous Graph Schema (per graph instance)",
        ha="center", va="center", fontsize=13, fontweight="bold")

fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F2_heterogeneous_graph_schema"
save_figure(fig, out)
plt.close(fig)
print("F2 done.")

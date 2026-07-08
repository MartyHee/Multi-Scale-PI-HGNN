"""
F3: MS-HGT Macro Anchor Architecture Detail
Focus on the multi-scale contribution: micro HGT → coordinate-based anchor pooling → macro SAGEConv → gated residual fusion.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import save_figure

fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis("off")

def draw_box(ax, x, y, w, h, color, title, lines, title_color="white", fontsize_title=8, fontsize_text=6.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor="#333333", linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.15, title, ha="center", va="top", fontsize=fontsize_title,
            fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        # Replace special chars that may not render in Arial
        line_clean = line.replace("ℝ", "R").replace("∈", "in")
        line_clean = line_clean.replace("∥", "||").replace("⊙", "odot")
        line_clean = line_clean.replace("✓", "[ok]").replace("⁺", "+")
        line_clean = line_clean.replace("⁹", "9").replace("⁻", "-")
        line_clean = line_clean.replace("⁽", "(").replace("⁾", ")")
        ax.text(x + 0.12, y + h - 0.4 - i*0.22, line_clean, ha="left", va="top",
                    fontsize=fontsize_text, color="white", alpha=0.95)

def draw_arrow(ax, x1, y1, x2, y2, color="#666666", label="", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.12
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7, color=color,
                fontstyle="italic")

# ===== Architecture Layout =====

# === Section 1: Micro HGT Layers (left, full height) ===
draw_box(ax, 0.3, 1.2, 3.5, 5.8, "#984ea3",
    "Micro Heterogeneous Message Passing (HGT Backbone)",
    ["L stacked HGTConv layers:",
     "",
     "For each edge type (r):",
     "  h_dst⁽ˡ⁺¹⁾ = Aggregate(",
     "    Attention(src, dst, r)",
     "    × Message(src, dst, r)",
     "    )",
     "",
     "Type-dependent QKV:",
     "  Q = h_dst · W_Q(τ_dst)",
     "  K = h_src · W_K(τ_src)",
     "  V = h_src · W_V(τ_src)",
     "",
     "Message types:",
     "  • belongs_to_beam",
     "  • rev_belongs_to_beam",
     "  • belongs_to_plate",
     "  • rev_belongs_to_plate",
     "  • structural_link",
     "",
     "Output: node representations",
     "  h_mesh, h_beam, h_plate ∈ ℝ^d",
     "",
     "Learnable skip connections",
     "LayerNorm after each HGT block",
     ""],
     fontsize_title=9, fontsize_text=6.5)

# === Section 2: Anchor Construction (center top) ===
draw_box(ax, 4.8, 4.5, 3.0, 2.5, "#e41a1c",
    "Macro Anchor Construction",
    ["From mesh_node coords:",
     "",
     "x-coordinate clustering",
     "→ K=12 anchor nodes",
     "",
     "Anchor features:",
     "  h_anchor[k] = MEAN(",
     "    h_mesh[i] for i∈C_k)",
     "",
     "Macro edge: k-NN in",
     "  coordinate+feature space",
     "  → fully connected chain",
     "",
     "Stiffness-aware variant:",
     "  Weight by beam stiffness"],
     fontsize_title=8, fontsize_text=6.5)

# === Section 3: Macro Chain Graph (center bottom-left) ===
draw_box(ax, 4.8, 1.2, 3.0, 2.5, "#ff7f00",
    "Macro Message Passing",
    ["Macro graph G_macro:",
     "  12 anchor nodes",
     "  undirected k-NN edges",
     "",
     "Processor: 2× SAGEConv",
     "  h_anchor' = SAGEConv(",
     "    h_anchor, G_macro)",
     "",
     "Captures long-range",
     "  force transfer across",
     "  full bridge span",
     "",
     "Output: refined macro",
     "  anchor representations"],
     fontsize_title=8, fontsize_text=6.5)

# === Section 4: Cross-Scale Fusion (right top) ===
draw_box(ax, 8.5, 4.5, 3.2, 2.5, "#4daf4a",
    "Cross-Scale Gated Fusion",
    ["For each mesh node i:",
     "  Find nearest anchor a(i)",
     "",
     "Gate:",
     "  g = σ(W_g · [h_mesh ∥ h_anchor])",
     "",
     "Residual fusion:",
     "  h_mesh' = h_mesh + g ⊙",
     "    (W_f · h_anchor[a(i)])",
     "",
     "Gating controls how much",
     "  macro info flows back",
     "",
     "Comparison tested:",
     "  additive vs gated (gated ✓)"],
     fontsize_title=8, fontsize_text=6.5)

# === Section 5: Dual Decoder (right bottom) ===
draw_box(ax, 8.5, 1.2, 3.2, 2.5, "#a65628",
    "Dual Physical Decoder",
    ["Displacement Decoder:",
     "  h_mesh' → MLP →",
     "  6-DOF (Dx, Dy, Dz,",
     "           Rx, Ry, Rz)",
     "",
     "Force Decoder:",
     "  h_beam → MLP →",
     "  12-D (Fx_I..Mz_J)",
     "",
     "Shared latent from",
     "  hetero encoder",
     "",
     "Separate output heads",
     "  per physical quantity"],
     fontsize_title=8, fontsize_text=6.5)

# === Arrows ===
draw_arrow(ax, 3.8, 5.8, 4.8, 5.8, label="micro mesh → macro anchors")
draw_arrow(ax, 3.8, 2.8, 4.8, 2.8)
draw_arrow(ax, 6.3, 4.5, 6.3, 3.7, label="down to macro processor")
draw_arrow(ax, 7.8, 5.8, 8.5, 5.8, label="macro anchor → cross-scale fusion")
draw_arrow(ax, 7.8, 2.8, 8.5, 2.8, label="micro & macro → decoder")
draw_arrow(ax, 11.7, 5.8, 11.7, 3.7, label="fused rep → decoder")

# === Information callouts ===
# Side note about why multi-scale
note_box = FancyBboxPatch((12.5, 3.5), 1.3, 3.0, boxstyle="round,pad=0.06",
                           facecolor="#f0f0f0", edgecolor="#999999", linewidth=1.0)
ax.add_patch(note_box)
ax.text(13.15, 6.2, "Why Multi-\nScale?", ha="center", va="top", fontsize=8, fontweight="bold")
ax.text(13.15, 5.2, "Standard GNN\nhas limited\nreceptive field\n(~K layers).\n\nLong-range\nforce transfer\nin bridges\nspans the full\nstructure.\n\nMacro anchors\nbridge this gap.",
        ha="center", va="top", fontsize=6.5, color="#333333")

# Title
ax.text(7, 7.2, "MS-HGT Architecture Detail — Multi-Scale Macro Anchor Fusion",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(7, 6.95, "Core method contribution: micro HGT backbone → x-coordinate anchor pooling → macro SAGEConv → gated residual fusion",
        ha="center", va="center", fontsize=8, color="grey")

fig.tight_layout()
out = "D:/※CREC/BiShe/S1/Multi-Scale-PI-HGNN/outputs/figures/ictai2026/F3_ms_hgt_macro_anchor_architecture"
save_figure(fig, out)
plt.close(fig)
print("F3 done.")

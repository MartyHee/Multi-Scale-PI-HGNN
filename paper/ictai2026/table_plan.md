# Table Plan — ICTAI 2026

> **Target:** 5 core tables + 0–1 supplementary tables for 8-page paper.
> Each entry describes table content, data source, and notes on formatting.

---

## Table 1: Dataset Statistics

| Column | Content | Source | Status |
|--------|---------|--------|:------:|
| Total graph instances | 35,000 (70 samples × 500 load cases) | Dataset validation | ✅ |
| Train / Val / Test | 28,000 / 3,500 / 3,500 | by_sample 80/10/10 split | ✅ |
| Mesh nodes per graph | 1,056 | HeteroData schema | ✅ |
| Beam elements per graph | 1,646 | HeteroData schema | ✅ |
| Plate elements per graph | 832 | HeteroData schema | ✅ |
| Edge types | 5 (belongs_to_beam, rev_belongs_to_beam, belongs_to_plate, rev_belongs_to_plate, structural_link) | HeteroData schema | ✅ |
| Structural links per graph | 132 directed (66 physical); constant across all 35K graphs | Dataset validation | ✅ |
| Node feature dimensions | 15 (mesh_node), 11 (beam_element), 6 (plate_element) | schema definition | ✅ |
| Displacement target dim | 6 (Dx, Dy, Dz, Rx, Ry, Rz) | schema definition | ✅ |
| Force target dim | 12 (Fx_I..Mz_J) | schema definition | ✅ |
| Topology | Shared across all 70 design samples | Dataset validation | ✅ |

**Placement:** Section 4.1, after "Dataset and Experimental Setup" text.
**Format:** Two-column (property, value). Compact with no grid lines.
**Space estimate:** 1/4 page.

---

## Table 2: Main Model Comparison (Primary Paper Table)

| Column | Content | Source | Status |
|--------|---------|--------|:------:|
| Method | MLP, GCN, GAT, RGCN, HGT, MS-HGT gated, MS-PI-HGT-Full | Experiment audit §2.1 | ✅ |
| Graph Type | (none / homogeneous / heterogeneous) | Schema definition | ✅ |
| Typed Message | (no / no / no / relation-specific / typed attention / typed attention / typed attention) | Model config | ✅ |
| Multi-Scale | (no / no / no / no / no / macro gated / macro gated) | Model config | ✅ |
| Physics Loss | (no / no / no / no / no / no / BC+Link) | Model config | ✅ |
| Params | 96K / 76K / 77K / 520K / 744K / 894K / 894K | metrics_summary.json | ✅ |
| Disp R² | 0.8554 / 0.8476 / 0.8421 / 0.9366 / 0.9765 / **0.9952** / 0.9948 | metrics_summary.json | ✅ |
| Dy R² | 0.1833 / 0.1778 / 0.1649 / 0.670 / 0.905 / 0.993 / 0.993 | Server artifact extraction | ✅ |
| Force R² | 0.9824 / 0.9696 / 0.9632 / 0.9878 / 0.9893 / 0.9928 / **0.9933** | metrics_summary.json | ✅ |
| RelMAE | 0.0884 / 0.1227 / 0.1361 / 0.0724 / 0.0676 / **0.0519** / 0.0516 | metrics_summary.json | ✅ |

**Placement:** Section 4.2, after baseline comparison text.
**Format:** Landscape or compact table. Bold best values.
**Highlighting:** Add separator line between homogeneous (GCN/GAT) and heterogeneous (RGCN onward).
**Space estimate:** 1/2 page.

**Narrative role:** Show the progressive improvement: MLP (strong baseline) → homogeneous GNN (worse) → typed RGCN (better) → HGT (better) → MS-HGT (dramatic jump) → MS-PI-HGT (physical consistency + tail).

---

## Table 3: Multi-Scale Ablation

| Column | Content | Source | Status |
|--------|---------|--------|:------:|
| Variant | HGT (no macro), MS-HGT additive, MS-HGT gated | Stage 4 result lock | ✅ |
| Fusion type | — / additive / gated residual | Config | ✅ |
| Params | 744K / 844K / 894K | metrics_summary.json | ✅ |
| Disp R² | 0.9765 / 0.9950 / **0.9952** | Stage 4 metrics | ✅ |
| Dy R² | 0.905 / 0.993 / **0.993** | Stage 4 metrics | ✅ |
| Force R² | 0.9893 / 0.9931 / 0.9928 | Stage 4 metrics | ✅ |
| RelMAE | 0.0676 / 0.0531 / **0.0519** | Stage 4 metrics | ✅ |

**Placement:** Section 4.3, after multi-scale ablation text.
**Format:** Compact. Bold best values.
**Space estimate:** 1/4 page.

**Narrative role:** Isolate the macro-anchor module's contribution. Show that both additive and gated fusion dramatically improve over HGT, with gated providing the best overall balance.

---

## Table 4: Physics Regularization Ablation

| Column | Content | Source | Status |
|--------|---------|--------|:------:|
| Variant | MS-HGT gated (base), MS-PI-HGT-BC, MS-PI-HGT-Link, MS-PI-HGT-Full | Stage 5 experiment | ✅ |
| λ_BC | 0 / 0.08 / 0 / 0.08 | Config | ✅ |
| λ_link | 0 / 0 / 0.002 / 0.002 | Config | ✅ |
| Disp R² | 0.9952 / 0.9951 / **0.9952** / 0.9948 | metrics_summary.json | ✅ |
| Force R² | 0.9928 / **0.9934** / **0.9934** / 0.9933 | metrics_summary.json | ✅ |
| RelMAE | **0.0519** / 0.0529 / 0.0515 / 0.0516 | metrics_summary.json | ✅ |
| Constrained DOF MAE ↓ | 0.000242 / 0.000188 / 0.000277 / **0.000148** | Physics diagnostics | ✅ |
| Force P95 AE ↓ | 38,000 / 39,756 / 39,277 / **37,917** | Physics diagnostics | ✅ |

**Placement:** Section 4.4, after physics ablation text. Reference Figure 5.
**Format:** Compact. ↓ indicator for "lower is better" metrics. Bold best.
**Space estimate:** 1/4 page.

**Narrative role:** Show that BC loss improves constraint satisfaction, Link loss has limited convergence, and Full variant achieves best balance. The key claim: physics loss improves physical consistency (39% BC reduction) without degrading accuracy.

---

## Table 5: Conformal UQ Coverage

| Column | Content | Source | Status |
|--------|---------|--------|:------:|
| Domain | Displacement ALL, Displacement Dx..Rz (range), Force ALL, Force per-comp (range) | Stage 6 conformal_summary.json | ✅ |
| 90% Cov | 89.74%, 89.6–89.9%, **89.97%**, 89.94–90.00% | conformal_summary.json | ✅ |
| 90% Avg Half-Width | 0.000476 m/rad, 2.1e-5–2.0e-3, 40,136 N/N·m, range | conformal_summary.json | ✅ |
| 95% Cov | 94.78%, —, **94.97%**, — | conformal_summary.json | ✅ |
| 95% Avg Half-Width | 0.000567 m/rad, —, 69,715 N/N·m, — | conformal_summary.json | ✅ |

**Placement:** Section 4.5, after conformal UQ results text. Reference Figure 6.
**Format:** Compact. Note: half-width = q (conformal quantile), not total width.
**Space estimate:** 1/4 page.

**Narrative role:** Show near-nominal coverage across all 18 components. Distinguish marginal (per-DOF, primary guarantee) from joint (diagnostic only).

---

## Table 6 (Supplementary): Region Coverage — OPTIONAL

| Column | Content | Source |
|--------|---------|--------|
| Region | Support / Free / Q1 (End) / Q3 (Midspan) / Q5 (End) / High-Response | region_labels |
| Nodes count | 28K / 3,668K / 392K / 392K / 392K / 18.5K | region_labels |
| 90% Joint Coverage | 25.2% / 55.1% / 49.5% / 57.4% / 53.1% / 28.4% | conformal_summary.json |
| 90% Per-DOF Coverage | ~90% / ~90% / ~90% / ~90% / ~90% / ~85% | conformal_summary.json |

**Placement:** Supplementary or Section 5 (Discussion). Only if space permits.
**Format:** Compact.
**Space estimate:** 1/6 page if included.

**Narrative role:** Show the gap between marginal (~90%) and joint (25–57%) coverage. Joint coverage for high-response region (28.4%) is lowest → motivates future work on adaptive intervals.

---

## Table Formatting Notes

### Font sizes
- Main tables (Table 2): 8–9pt font, single-spaced
- Compact tables (Table 3–5): 8pt font
- IEEE conference format: tables fit within single column (84mm) or double column (178mm)

### Grid and borders
- IEEE style: minimal grid lines. Use horizontal rules sparingly.
- Use \toprule, \midrule, \bottomrule (booktabs style)

### Abbreviation handling
- Define "RelMAE" = Relative Mean Absolute Error once in caption
- Define "AE" = Absolute Error
- All units in parentheses: (m), (rad), (N), (N·m)

### Color (if allowed)
- Bold text for best results, or shaded cells
- For online PDF: use blue bold for best, but grayscale-printable
- Avoid red-green distinction for colorblind accessibility

---

*Document version: v1.0 / 2026-07-08 / ICTAI Phase 3 Initial Draft*

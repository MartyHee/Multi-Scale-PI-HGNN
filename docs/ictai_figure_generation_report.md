# ICTAI 2026 — Phase 2B: Paper Figure Generation Report

> **Date:** 2026-07-08  
> **Task:** Generate 6 paper figures (F1–F6) for ICTAI 2026 submission
> **Constraint:** No model training, no experiment re-runs, no checkpoint modification

---

## 1. Summary

| Figure | ID | Type | Script | PNG | SVG | Paper-Ready? |
|--------|:--:|------|--------|:---:|:---:|:------------:|
| Overall framework pipeline | F1 | Schematic diagram | `gen_f1_pipeline.py` | ✅ | ✅ | ⚠️ Needs AI polish |
| Heterogeneous graph schema | F2 | Schema diagram | `gen_f2_hetero_schema.py` | ✅ | ✅ | ⚠️ Needs AI polish |
| MS-HGT macro anchor architecture | F3 | Architecture diagram | `gen_f3_architecture.py` | ✅ | ✅ | ⚠️ Needs AI polish |
| Main results bar chart | F4 | Data bar chart | `gen_f4_main_results_bar.py` | ✅ | ✅ | ✅ Paper-ready |
| Physics ablation bar chart | F5 | Data bar chart | `gen_f5_physics_ablation_bar.py` | ✅ | ✅ | ✅ Paper-ready |
| UQ coverage half-width | F6 | Data bar/line chart | `gen_f6_uq_coverage_halfwidth.py` | ✅ | ✅ | ✅ Paper-ready |

**Total:** 6 figures × 2 formats = **12 output files** in `outputs/figures/ictai2026/`

---

## 2. Data Sources

| Figure | Primary Data Source | Secondary Source | Schema Validated? |
|--------|---------------------|:----------------:|:-----------------:|
| F1 | In-script description | `experiment_audit.md` §4.1, `stage4_macro_anchor_design.md` | N/A (schematic) |
| F2 | `experiment_audit.md` §5 (Table 1) | `README.md` | ✅ All counts from canonical v2 schema |
| F3 | `stage4_result_lock.md` | `stage4_macro_anchor_design.md` | N/A (schematic) |
| F4 | `README.md` Main Results Summary | `experiment_audit.md` §2.1 | ✅ Locked from server artifacts |
| F5 | `experiment_audit.md` §2.3 | `outputs/diagnostics/stage5_physics/diagnostics_summary.json` | ✅ Cross-checked with stage5_result_lock.md |
| F6 | `outputs/diagnostics/stage6_uq/test_graph_50_50/20260629082344/conformal_summary.json` | `experiment_audit.md` §2.4 | ✅ Half-width convention locked |

### 2.1 Key Data Integrity Checks

- **Dy R² values (F4):** MLP=0.1833, GCN=0.1778, GAT=0.1649 — exact values from server training artifacts, not estimates
- **Structural link count (F2):** 132 directed edges per graph, verified constant across all 35K graphs (Phase 1.5c)
- **UQ half-width convention (F6):** All displayed values are `q` (half-width), NOT total width. Confirmed from `compute_conformal.py` source code
- **Coverage statements (F6):** Marginal coverage ± gap annotations; no "guaranteed joint coverage" claims

### 2.2 Data Not Found (If Any)

None. All required data for F4–F6 was available from locked server artifacts and diagnostics output. No `TODO_MISSING` markers needed.

---

## 3. Figure Quality Assessment

### 3.1 Paper-Ready Figures (F4, F5, F6)

These 3 data-driven bar charts are directly usable in the ICTAI 2026 paper:

**F4 — Main Results Bar:**
- Font sizes 9–11pt (suitable for single-column width ~8 cm)
- Colorblind-friendly palette (ColorBrewer-inspired blue/orange/green)
- Value annotations on all bars
- Vertical separator between homogeneous and heterogeneous models
- Legend in lower right, ncol=3

**F5 — Physics Ablation Dual Panel:**
- Dual-panel design avoids mixing different scales (m vs N)
- % change annotations relative to baseline for quick interpretation
- Full variant 38.8% BC violation reduction clearly visible
- Colorblind-friendly palette (4 distinct colors)

**F6 — UQ Coverage Three-Panel:**
- Per-DOF coverage with gap annotations from target 90%
- Per-DOF half-width with log scale for wide dynamic range (2.1e-5 to 2.0e-3)
- Region joint coverage (Support 25.2% → High-Response 28.4%)
- All axes use "Avg Half-Width" convention
- Marginal vs joint coverage clearly labeled

### 3.2 Figures Needing AI Polish (F1, F2, F3)

These 3 schematic diagrams are functionally complete but would benefit from professional vector graphics tools (draw.io / Adobe Illustrator / TikZ) for the final submission:

**Current limitations:**
- **F1 Pipeline:** 8 stages rendered as colored boxes with text. Information is complete and correctly sequenced but uses simple matplotlib patches. A professional tool would add better alignment, curved connectors, and visual hierarchy.
- **F2 Schema:** Node type boxes with edge annotations. All counts and dimensions are accurate. A design tool could produce a more elegant layout with better visual separation of edge types.
- **F3 Architecture:** Most content-dense figure (micro HGT → macro anchor → macro processor → cross-scale fusion → dual decoder + "why multi-scale?" callout). Text at 6.5pt is readable at 300 DPI but would be cleaner in vector form.

**Recommendation:** Export the SVG from these figures and import into draw.io / Illustrator for touch-up. The SVG structure preserves the layout; text replacement should be straightforward.

### 3.3 Common Quality Checklist

| Criterion | F1 | F2 | F3 | F4 | F5 | F6 |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| 300 DPI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SVG + PNG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Colorblind-friendly | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Axis labels with units | N/A | N/A | N/A | ✅ R² | ✅ m, N | ✅ %, m, rad |
| Font ≥ 8 pt | ✅ | ✅ | ⚠️ 6.5pt | ✅ 9-11pt | ✅ 9-10pt | ✅ 7-11pt |
| Legend clear | N/A | ✅ | N/A | ✅ | N/A | ✅ |
| No misleading titles | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Half-Width convention | N/A | N/A | N/A | N/A | N/A | ✅ |
| No joint coverage claim | N/A | N/A | N/A | N/A | N/A | ✅ |

---

## 4. New Scripts Created

All scripts in `scripts/paper_figures/`:

| Script | Purpose | Lines | Dependencies |
|--------|---------|:-----:|:------------:|
| `__init__.py` | Package init | 1 | — |
| `fig_style.py` | Shared palette (CB colors), fonts, `save_figure()` helper | 56 | matplotlib, numpy |
| `gen_f1_pipeline.py` | F1 pipeline diagram (8-stage block diagram) | 185 | matplotlib |
| `gen_f2_hetero_schema.py` | F2 graph schema (node/edge types with counts) | 118 | matplotlib |
| `gen_f3_architecture.py` | F3 macro anchor architecture detail | 169 | matplotlib |
| `gen_f4_main_results_bar.py` | F4 grouped bar chart (Disp/Dy/Force R²) | 76 | matplotlib, numpy |
| `gen_f5_physics_ablation_bar.py` | F5 dual-panel physics ablation | 74 | matplotlib, numpy |
| `gen_f6_uq_coverage_halfwidth.py` | F6 three-panel UQ coverage/width | 104 | matplotlib, numpy |

**Total:** 7 new files (+782 lines) in a dedicated `scripts/paper_figures/` package.

---

## 5. Immediate Paper Use

### Can embed directly (with minimal caption editing):
- **F4** → Main results figure (required in nearly all ML/structural surrogate papers)
- **F5** → Physics ablation figure (supports the "why physics loss matters" claim)
- **F6** → UQ figure (supports the "distribution-free uncertainty" claim)

### Need vector touch-up before embedding:
- **F1** → Position as first figure showing the full pipeline
- **F2** → Position in preliminaries or method section explaining graph construction
- **F3** → Position in method section explaining the macro anchor contribution

---

## 6. Figures Not Generated (Scope Clarification)

The following figures from `experiment_audit.md` §4.1 were **not regenerated** per Phase 2B scope (only F1–F6 requested):

| Figure | Priority | Status | Notes |
|--------|:--------:|:------:|-------|
| F7: Region-wise coverage | Medium | Exists (needs cleanup) | Already present in `outputs/diagnostics/stage6_uq/` |
| F8: Region map | Medium | Exists (needs cleanup) | Already present in `outputs/diagnostics/stage4_region_baseline/` |
| F9: Training curves | Medium | Exists (needs cleanup) | Already present in remote artifact dirs |
| F10: High-response coverage | Low | Exists (needs cleanup) | Already present in `outputs/diagnostics/stage6_uq/` |

These existing figures were assessed in `experiment_audit.md` §3.6 (common issues: missing axis labels, small fonts, low resolution). If needed, they can be cleaned up with a similar script-based approach.

---

## 7. Commands to Regenerate All Figures

```bash
# From Multi-Scale-PI-HGNN root:
python scripts/paper_figures/gen_f1_pipeline.py
python scripts/paper_figures/gen_f2_hetero_schema.py
python scripts/paper_figures/gen_f3_architecture.py
python scripts/paper_figures/gen_f4_main_results_bar.py
python scripts/paper_figures/gen_f5_physics_ablation_bar.py
python scripts/paper_figures/gen_f6_uq_coverage_halfwidth.py
```

Or run all at once:
```bash
for f in scripts/paper_figures/gen_f*.py; do python "$f"; done
```

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Phase 2B Figure Generation*

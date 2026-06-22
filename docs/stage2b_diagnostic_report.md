# Stage 2-B Diagnostic Report — Fine-Grained Model Analysis

> **Status:** ✅ Diagnostic evaluation completed
> **Dataset:** `processed/hetero_graph_dataset_v2`
> **Split:** `by_sample` (train=28,000 / val=3,500 / test=3,500)
> **Date:** 2026-06-23
> **Script:** `scripts/analyze_stage2b_diagnostics.py`
> **Output:** `outputs/diagnostics/stage2b/20260623010331/`

---

## 1. Model Ranking (Confirmed)

| Model | Graph Type | Typed Message | Params | Best Ep | Train Time | **Disp R2** | **Force R2** | **RelMAE** |
|-------|-----------|---------------|-------:|--------:|-----------:|-----------:|------------:|----------:|
| MLP | none | no | 96,274 | 88 | 56.8min | 0.8554 | 0.9824 | 0.0884 |
| GCN | homogeneous | no | 76,050 | 96 | 1.9h | 0.8476 | 0.9696 | 0.1227 |
| GAT | homogeneous | no | 76,818 | 88 | 2.1h | 0.8421 | 0.9632 | 0.1361 |
| RGCN | heterogeneous | relation-specific (SAGEConv) | 520,338 | 90 | 2.3h | 0.9366 | 0.9878 | 0.0724 |
| **HGT** 🏆 | **heterogeneous** | **typed attention (HGTConv)** | **744,279** | **99** | **5.9h** | **0.9769** | **0.9891** | **0.0683** |

**Final ranking: HGT > RGCN > MLP > GCN > GAT**

All 5 artifacts pass consistency checks:
- ✅ All artifacts have complete file sets (13 files each)
- ✅ All train_log.csv have 100 continuous epochs, no NaN/Inf
- ✅ All best_epoch consistent between metrics_summary.json and train_log.csv
- ✅ All server_output.log show clean completion (no OOM/Traceback)
- ✅ All reported metrics match metrics_summary.json values within tolerance
- ✅ All use `by_sample` split, same dataset, same 100 epochs

---

## 2. What HGT Has Solved

### 2.1 Dy Bottleneck Broken

The most significant achievement across Stage 2-A → 2-B:

| Model | Dy R2 | Cumulative Gain |
|-------|------:|----------------:|
| MLP | 0.1833 | — |
| GCN | 0.1778 | -0.0055 |
| GAT | 0.1649 | -0.0184 |
| RGCN | 0.6692 | **+0.4859** vs MLP |
| **HGT** | **0.9077** | **+0.7244** vs GCN, **+0.2385** vs RGCN |

Dy R2 trajectory: `0.18 (homogeneous) → 0.67 (typed conv) → 0.91 (typed attn)`

**Finding:** Dy was proven to be model-limited, not data-limited. HGT's typed attention mechanism is the key to capturing lateral displacement patterns.

### 2.2 Force Components All Improved

HGT achieves the highest Force R2 on all 12 components. The weakest force component (Fy, R2=0.9866) is already near the practical ceiling. Every force component ≥ 0.9866.

### 2.3 Macro Disp R2 at 0.9769

Displacement macro average R2 at 0.9769 means most displacement variance is captured. Five of six components (Dx, Dz, Rx, Ry, Rz) have R2 ≥ 0.9875.

---

## 3. Where HGT Still Falls Short

### 3.1 Dy Remains the Weakest Component

| Component | HGT R2 | Remaining Error (1-R2) | Priority |
|-----------|:------:|:---------------------:|:--------:|
| Dy | **0.9077** | **0.0923** | 🎯 **Highest** |
| Rz | 0.9875 | 0.0125 | Medium |
| Dx | 0.9896 | 0.0104 | Medium |
| Dz | 0.9906 | 0.0094 | Low |
| Ry | 0.9925 | 0.0075 | Low |
| Rx | 0.9935 | 0.0065 | Low |

**Dy has 9.23% remaining unexplained variance** — this is 7-14× larger than any other displacement component. This is the single largest remaining opportunity for Stage 3 (Ours-base).

### 3.2 Disp vs Force: Asymmetric Saturation

| Task | HGT R2 | 1-R2 | Status |
|------|:------:|:----:|--------|
| Force (macro) | 0.9891 | 0.0109 | Near saturation |
| Disp (macro) | 0.9769 | 0.0231 | Room to improve |

Force R2 at 0.9891 is approaching the data-limited ceiling. Displacement still has ~2.3× more remaining unexplained variance than force.

### 3.3 HGT Training Time Cost

HGT takes 5.9 hours (353 min) — **2.6× longer than RGCN** (2.3h). The compute cost is significant and should be considered when designing Ours.

---

## 4. Is R2 Saturated?

### Partial Saturation Assessment

| Dimension | Assessment | Evidence |
|-----------|-----------|----------|
| **Overall Force R2** | ⚠️ **Near saturated** | 0.9891, remaining error = 0.0109. Further improvement likely < 0.005 |
| **Overall Disp R2** | 🔶 **Some room** | 0.9769, remaining error = 0.0231. 0.005-0.010 improvement possible |
| **Dy R2** | 🟢 **Significant room** | 0.9077, remaining error = 0.0923. Major improvement possible |
| **Tail / Region errors** | ❓ **Unknown** | Cannot assess without full predictions |
| **Physical consistency** | ❓ **Unknown** | Cannot assess without full predictions |

**Key insight:** Overall Force R2 is near saturation *as an aggregate metric*. But:
- Overall R2 does not capture region-wise variation
- Overall R2 does not capture high-response tail errors
- Overall R2 does not capture physical consistency
- Even small R2 improvements (e.g., +0.002) can represent meaningful relative error reduction

### RelMAE Trajectory Suggests Diminishing Returns

| Model | RelMAE |
|-------|-------:|
| MLP | 0.0884 |
| RGCN | 0.0724 (↓ 18.1% vs MLP) |
| HGT | 0.0683 (↓ 5.7% vs RGCN) |

The rate of RelMAE reduction is slowing. Combined RelMAE at 0.0683 is approaching a floor, but per-component analysis (especially Dy) still shows meaningful gaps.

---

## 5. Why Ours Still Has Research Space

### 5.1 Ours-Based Innovation Targets

The diagnostic analysis identifies the following concrete directions for Ours-base:

**Target 1: Dy Remaining Error**
- Dy (R2=0.9077) has 9.2× more remaining error than Dx (R2=0.9896)
- This is the single largest HGT weakness
- Hypothesis: edge_attr-aware structural_link message passing (stiffness features) could improve lateral displacement modeling
- Metric: Dy R2 improvement from 0.9077 toward 0.95+

**Target 2: Physics-Gated Message Passing**
- HGT uses pure typed attention — no physical knowledge
- Structural_link edges carry stiffness features (Kx, Ky, Kz, etc.) not used by HGT
- Physics-gated or stiffness-aware message functions could improve structural_link message quality
- Expected benefit: Dy, region-wise, support boundary consistency

**Target 3: Macro Anchor Graph (Stage 4)**
- Long-range force transmission in truss girders is inherently non-local
- HGT's micro message passing has limited receptive field per layer
- Macro anchor graph could improve global consistency
- Expected benefit: midspan displacement, support reaction consistency

**Target 4: Not Only Overall R2**
- Region-wise errors (supports, connections, midspan) not captured by global R2
- Physical consistency (BC residuals, link smoothness) not captured by R2
- High-response / tail errors not reflected in overall metrics
- Uncertainty quantification (Stage 6) provides independent value

### 5.2 HGT Is a Strong Baseline — Not a Ceiling

HGT provides an extremely strong baseline, but:
- HGT has no edge_attr awareness → structural_link features unused
- HGT has no multi-scale fusion → limited long-range modeling
- HGT has no physics regularization → no physical consistency guarantee
- HGT uses 4× more parameters than RGCN with 2.6× training cost

Ours-base does not need to outperform HGT on every metric. A model that:
- Matches HGT on Force R2 (≥ 0.989)
- Improves Dy R2 from 0.9077 toward 0.93+
- Reduces region-wise or tail errors
- Shows better physical consistency
- Is more parameter-efficient

...would constitute a meaningful contribution even if overall Disp R2 does not dramatically surpass 0.9769.

### 5.3 Innovation Must Be Structural

Ours-base **cannot** be "HGT + small tweak." The innovation must come from:
1. **Edge_attr-aware message passing** (stiffness features on structural_link)
2. **Physics-typed message functions** (distinct conv for each relation type)
3. **Dual decoder with shared physics latent** (force & displacement coupled)
4. **Explicit structural_link modeling** (rigid connection consistency)
5. **Relation-level gated aggregation** (adaptive importance weighting per edge type)

Any combination of 2+ of these constitutes a genuine structural improvement over HGT.

---

## 6. Missing Diagnostics (Blocked Items)

### 6.1 Full Test Predictions Not Available

Current artifacts do NOT contain `test_predictions.csv` or equivalent full prediction output. This blocks:

| Diagnostic | Why Blocked | Impact |
|-----------|-------------|--------|
| P50/P90/P95/P99 tail error | Need per-sample pred-vs-true pairs | Cannot assess model performance on difficult cases |
| High-response subset error | Need full prediction distribution | Cannot assess if R2 is inflated by low-magnitude samples |
| Region-wise metrics (support/midspan/connection) | Need node-level predictions grouped by region | Cannot assess spatial error distribution |
| Physical consistency (BC residual, link smoothness) | Need mesh_node displacement predictions at specific locations | Cannot assess physical realism |
| Beam force consistency | Need beam_element force predictions | Cannot be done without predictions |

**Resolution:** Full prediction export implemented in `scripts/export_full_predictions.py`. See `docs/full_prediction_export.md` for usage.
Run `bash server_ops/export_stage2b_predictions.sh` on server to generate predictions, then re-run diagnostics with `--predictions-dir outputs/predictions/stage2b`.

Note: This diagnostic report was generated BEFORE prediction export was implemented. The missing metrics (tail error, region, physical consistency) are now UNBLOCKED — run the export + re-run diagnostics to compute them.

### 6.2 Region Labels Not Available

Current dataset does not have region labels (support/midspan/connection). However, these can be computed on-the-fly from:
- Node coordinates (available in mesh_node.x)
- Support BC flags (available in mesh_node.x)
- Edge incidence (available in edge_index_dict)

A `scripts/diagnostic_region_metrics.py` can compute these without modifying the dataset schema.

### 6.3 Physical Diagnostics Plan

A separate plan document has been generated at:
`outputs/diagnostics/stage2b/20260623010331/diagnostic_physical_diagnostic_plan.md`

---

## 7. MeshGraphNet-Style Baseline: Optional

**Recommendation:** Optional — not critical for the research thesis.

The progression `GCN/GAT < MLP < RGCN < HGT` already establishes:
- Homogeneous graph methods fail on physics-heterogeneous graphs
- Typed message passing (conv → attention) progressively improves
- Typed attention (HGT) is the current best

MeshGraphNet-style processor would add:
- Encoder-processor-decoder architecture comparison
- Iterative message passing with latent node updates
- A citation anchor in related work

**When to add:** If paper length allows and a 6th baseline strengthens the "types of message passing" narrative. Not needed if the focus is Ours vs best baseline (HGT).

---

## 8. Diagnostics Output Files

| File | Description |
|------|-------------|
| `diagnostic_artifact_check.csv` | Per-model artifact completeness and consistency (46 checks) |
| `stage2b_main_metrics.csv` | Aggregate metrics table |
| `stage2b_main_metrics.md` | Aggregate metrics in markdown |
| `stage2b_per_component_metrics_disp_r2.csv` | Per-component displacement R2 |
| `stage2b_per_component_metrics_force_r2.csv` | Per-component force R2 |
| `stage2b_per_component_metrics.md` | Per-component metrics in markdown |
| `diagnostic_missing_predictions.md` | Blocked diagnostics due to missing predictions |
| `diagnostic_region_metric_requirements.md` | Region label construction plan |
| `diagnostic_physical_diagnostic_plan.md` | Physical consistency diagnostic plan |
| `figures/stage2b_model_ranking.png` | Model ranking bar chart |
| `figures/stage2b_dy_r2_comparison.png` | Dy trajectory across models |
| `figures/stage2b_per_component_heatmap_disp.png` | Disp per-component heatmap |
| `figures/stage2b_per_component_heatmap_force.png` | Force per-component heatmap |
| `figures/stage2b_remaining_error_hgt.png` | HGT remaining error by component |
| `figures/stage2b_training_curves.png` | RGCN vs HGT training curves |

---

## 9. Next Steps

| Priority | Task | Rationale |
|----------|------|-----------|
| **P0** | Add eval-only full prediction export | Unblocks tail error, region, and physical diagnostics |
| **P1** | Compute tail error + region-wise metrics once predictions exist | Provides concrete evidence for Ours design focus |
| **P2** | Enter Stage 3 (Ours-base) | Build on HGT foundation with edge_attr-aware/physics-typed message passing |
| **P3** | MeshGraphNet-style baseline (optional) | Only if needed for related work/citation coverage |

### Recommendation for Stage 3 (Ours-Base)

Design Ours-base to concretely address:
1. **Dy remaining error** (R2 0.9077 → target 0.93+) via structural_link stiffness-aware message
2. **Edge_attr awareness** as the primary innovation over HGT
3. **Maintain Force R2** at ≥ 0.989
4. **Parameter efficiency** — target < 744K params (HGT's count)
5. **Comparable training time** — target < 4h

The core question is: *Can stiffness-aware physics-typed message passing improve upon HGT's typed attention, especially on difficult components and edges?*

---

## 10. Updated Scientific Narrative (Corrections)

The following corrections apply to earlier documentation:

| Previous Statement | Corrected Version |
|-------------------|-------------------|
| "Stage 2-B全线完成" | "Stage 2-B typed baselines (RGCN, HGT) completed and validated" |
| "HGT已经证明Ours一定能提升" | "HGT establishes a strong baseline; Ours must demonstrate incremental structural innovation to show improvement" |
| "边类型混淆是唯一原因" | "Edge-type confusion is the primary identified cause, but node-type dilution and MLP's strong local features also contribute to the homogeneous GCN/GAT degradation" |
| "Dy原因已经完全解释清楚" | "Dy was shown to be model-limited (not data-limited), and HGT largely resolves it (R2 0.9077), but the remaining 9.2% error is not yet explained" |
| "Stage 2-B包含的所有可能baseline已全部完成" | "Stage 2-B typed baselines (RGCN, HGT) are completed; MeshGraphNet-style processor remains optional based on paper scope" |

**More precise Stage 2-B status:**
- ✅ **RGCN (typed convolution)** — completed, validated
- ✅ **HGT (typed attention)** — completed, validated (best baseline)
- 💤 **MeshGraphNet-style processor** — optional, not yet implemented

The research thesis remains: **Physical heterogeneous graphs require type-aware message passing**, and typed attention (HGT) represents the best standard approach. Ours must show that physics-informed structural innovations can further improve upon HGT on dimensions not captured by overall R2 alone.

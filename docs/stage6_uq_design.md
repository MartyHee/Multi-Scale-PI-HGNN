# Stage 6 UQ Design — Split Conformal Prediction for MS-PI-HGT

**Date:** 2026-06-29
**Status:** Design Spec (Phase 1)
**Backbone:** MS-PI-HGT Full (MSHGTBaseline, 893,527 params)
**Checkpoint:** `outputs/baselines/MS_HGT/20260626170354/best_model.pt` (best_epoch=134)

---

## 1. Overview

Stage 6 applies **split conformal prediction** to provide calibrated prediction intervals for the locked Stage 5 model (MS-PI-HGT Full). The goal is NOT to improve point prediction accuracy, but to quantify the uncertainty of existing predictions in a distribution-free manner.

### 1.1 What Split Conformal Prediction Provides

- **Marginal coverage guarantee:** For any nominal level $1-\alpha$, the conformal interval covers the true value with probability at least $1-\alpha$ *on average* over the test distribution.
- **Distribution-free:** No parametric assumptions on residuals.
- **Post-hoc:** Uses model predictions only — no model retraining required.
- **Computationally cheap:** One forward pass on the calibration set, then quantile computation.

### 1.2 What It Does NOT Provide

- Pointwise/exact coverage per individual prediction.
- Strict epistemic/aleatoric uncertainty decomposition.
- Causal guarantees.
- Guarantees under distribution shift beyond exchangeability.

---

## 2. Nonconformity Score Design

### 2.1 Recommended Primary: Component-Wise Absolute Residual

For each output component $c$ of each target type $t \in \{disp, force\}$:

$$s_{t,c,i} = |y_{i,c}^{\text{true}} - \hat{y}_{i,c}|$$

Where $i$ indexes over all calibration points of the relevant node type.

**Rationale:**
- Intuitive: interval = prediction ± threshold
- Component-wise: handles different DOF scales naturally (D ~1e-4 m vs M ~1e5 N·m)
- No scale parameter estimation needed
- Easily invertible: prediction interval = $[\hat{y} - q, \hat{y} + q]$

### 2.2 Diagnostic Only: Normalized Residual

$$s_{t,c,i}^{\text{norm}} = \frac{|y_{i,c}^{\text{true}} - \hat{y}_{i,c}|}{\sigma_{c}^{\text{train}}}$$

Where $\sigma_{c}^{\text{train}}$ is the training-set standard deviation of component $c$.

**Use case:** Comparing relative uncertainty across components. Not used for main results.

### 2.3 Why NOT Pooled Scores

Pooling all DOF into a single score (e.g., $\|y - \hat{y}\|_2$) would:
- Mix displacement (m) with rotation (rad), force (N) with moment (N·m)
- Produce intervals that are over-covered on some DOF and under-covered on others
- Lose component-level interpretability

**Decision:** Component-wise absolute residual is the primary score. Pooled scores are not used.

---

## 3. Calibration Split Strategy

Three options are available. See [stage6_calibration_split_audit.md](stage6_calibration_split_audit.md) for full audit.

### 3.1 Option A: Val → Test Calibration (Engineering Baseline)

| Set | Split | Graphs | Notes |
|-----|-------|-------:|-------|
| Calibration | existing val | 3,500 | 7 samples × 500 LC |
| Evaluation | existing test | 3,500 | 7 samples × 500 LC |

**Advantage:** Maximum evaluation set size (3,500 graphs), simple to implement.
**Disadvantage:** Val split was used for early stopping — not a strictly independent calibration set. The distribution-free guarantee is weakened.

### 3.2 Option B: Test-Internal Split (Recommended for Primary Result)

| Set | Split | Graphs | Notes |
|-----|-------|-------:|-------|
| Calibration | test graphs (random 50%) | ~1,750 | Unseen during training AND validation |
| Evaluation | test graphs (remaining 50%) | ~1,750 | Unseen during calibration |

**Advantage:** Truly independent calibration set (no training/validation leakage). Strict conformal validity.
**Disadvantage:** Smaller evaluation set; reduced statistical power on tail metrics.

### 3.3 Option B-Sample: Test-Internal Sample-Level Split (Robustness Check)

| Set | Split | Graphs | Notes |
|-----|-------|-------:|-------|
| Calibration | 3 test samples | 1,500 | No LoadCase from these samples in training |
| Evaluation | 4 test samples | 2,000 | No overlap by construction |

**Advantage:** Preserves sample-level independence — no LoadCase shared between calibration and evaluation.
**Disadvantage:** Small calibration set (1,500) may degrade quantile stability.

### 3.4 Recommendation

| Context | Choice |
|---------|--------|
| **Primary paper result** | **Option B** (test-internal, graph-level 50/50) |
| Supplementary | Option A (val calibration, for comparison) |
| Robustness check | Option B-Sample (sample-level) |

---

## 4. Coverage Metrics

### 4.1 Marginal Coverage (Main Result)

Reported separately for each output domain:

**Displacement (mesh_node × 6 DOF):**

| Target Coverage | Empirical Coverage | Avg Interval Width | Median Width | P90 Width |
|-----------------|------------------:|------------------:|------------:|----------:|
| 90% ($\alpha=0.10$) | | | | |
| 95% ($\alpha=0.05$) | | | | |

Also per DOF (6 rows):

| DOF | $\alpha=0.10$ Coverage | $\alpha=0.10$ Width | $\alpha=0.05$ Coverage | $\alpha=0.05$ Width |
|-----|----------------------:|-------------------:|----------------------:|-------------------:|
| Dx | | | | |
| Dy | | | | |
| Dz | | | | |
| Rx | | | | |
| Ry | | | | |
| Rz | | | | |

**Force (beam_element × 12 components):**

Same structure as above, per component (Fx_I..Mz_J).

### 4.2 Region-Wise Coverage

| Region | $\alpha=0.10$ Coverage | $\alpha=0.10$ Width | Coverage Gap |
|--------|----------------------:|-------------------:|-------------:|
| Support | | | |
| Midspan (Q3) | | | |
| End neighborhood (Q1) | | | |
| Transition (Q2/Q4) | | | |
| High-response (top 10% by |y|) | | |
| Low-response (bottom 90%) | | | |

See [stage6_calibration_split_audit.md](stage6_calibration_split_audit.md) for region definitions.

### 4.3 High-Response Coverage

| Subset | $\alpha=0.10$ Coverage | $\alpha=0.10$ Width | P95 Residual |
|--------|----------------------:|-------------------:|------------:|
| All test | | | |
| Top 10% displacement | | | |
| Top 5% displacement | | | |
| Top 1% displacement | | | |

### 4.4 Graph-Level Conformal (Optional Conservative Variant)

For each graph $g$, compute:

$$s_g = \max_{i \in \text{graph}_g} \max_{c} s_{i,c}$$

Then calibrate across graphs.

**Purpose:** Provides simultaneous coverage guarantee at graph level. Useful if the application requires that *every node* in a graph is covered.

**Expected result:** Wider intervals, but stronger guarantee.

---

## 5. Conformal Prediction Algorithm

### 5.1 Standard Split Conformal (for Option A and Option B)

Given calibration set $\mathcal{D}_{\text{cal}} = \{(X_i, Y_i)\}_{i=1}^{n_{\text{cal}}}$ and trained model $\hat{f}$:

1. **Compute nonconformity scores:**
   $$s_i = |Y_i - \hat{f}(X_i)| \quad \text{(component-wise)}$$
   Produces score matrices: $S_{\text{disp}} \in \mathbb{R}^{n_{\text{cal}} \times 6}$, $S_{\text{force}} \in \mathbb{R}^{n_{\text{cal}} \times 12}$

2. **Compute conformal quantile:**
   $$q_{t,c} = \text{Quantile}_{1-\alpha}\left(\{s_{i,c}\}_{i=1}^{n_{\text{cal}}}\right)$$
   Adjusted by finite-sample correction:
   $$q_{t,c} = \text{Quantile}_{1-\alpha + \frac{1}{n_{\text{cal}}}}\left(\{s_{i,c}\}_{i=1}^{n_{\text{cal}}}\right)$$

3. **Build prediction intervals:**
   $$C_{t,c}(X_{\text{new}}) = [\hat{f}(X_{\text{new}})_c - q_{t,c}, \; \hat{f}(X_{\text{new}})_c + q_{t,c}]$$

### 5.2 Graph-Level Conformal

1. Compute graph-wise max score: $s_g = \max_{i \in g} s_i$
2. Calibrate over graphs: $q_{1-\alpha} = \text{Quantile}_{1-\alpha}(\{s_g\})$
3. Apply same $q$ to all nodes in each graph

### 5.3 Handling Calibration Set Size

For Option B (1,750 calibration graphs):
- mesh_node calibration points: ~1056 nodes × 1750 graphs = **~1.85M points**
- beam_element calibration points: ~1646 elements × 1750 graphs = **~2.88M points**
- Both are more than sufficient for stable quantile estimation.

---

## 6. Output File Design

All outputs under:

```
outputs/diagnostics/stage6_uq/<YYYYMMDDHHMMSS>/
```

### 6.1 Core Data Files

| File | Content | Format |
|------|---------|--------|
| `conformal_summary.json` | Top-level metrics (marginal coverage for disp/force at $\alpha=0.10, 0.05$) | JSON |
| `conformal_component_metrics.csv` | Per-DOF and per-force-component coverage + width | CSV |
| `conformal_region_metrics.csv` | Region-wise coverage + width | CSV |
| `conformal_high_response_metrics.csv` | High-response coverage + width | CSV |
| `conformal_graph_level_metrics.csv` | Graph-level conformal results | CSV |

### 6.2 Figures

| File | Content |
|------|---------|
| `coverage_width_plot.png` | Coverage vs interval width trade-off at multiple $\alpha$ |
| `component_coverage_bar.png` | Per-DOF and per-force-component coverage bar chart with nominal line |
| `region_coverage_bar.png` | Region-wise coverage with nominal line |
| `high_response_coverage.png` | Coverage degradation across response quantiles |

### 6.3 Report

| File | Content |
|------|---------|
| `stage6_uq_report.md` | Full report with tables, figures, and analysis |

---

## 7. Next Steps to Code Implementation

Phase 2 code implementation will require:

1. **`scripts/export_full_predictions.py`** — already exists, can reuse for calibration set inference
2. **New script: `scripts/compute_conformal.py`** — compute scores, quantiles, intervals
3. **New script: `scripts/analyze_conformal.py`** — coverage analysis, region metrics, figures

The export script already supports `--output-dir` and saves mesh NPZ + beam NPZ. The conformal computation script will load these NPZs plus the calibration split definition.

---

## 8. References

- Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv:2107.07511.*
- Shafer & Vovk (2008). "A Tutorial on Conformal Prediction." *JMLR, 9*, 371-421.
- Vovk et al. (2005). *Algorithmic Learning in a Random World.* Springer.

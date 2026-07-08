# Stage 6 Experiment Plan — Conformal UQ for MS-PI-HGT

**Date:** 2026-06-29
**Status:** Plan (Phase 1 Design)

---

## 1. Research Questions

| # | Question | How Answered |
|---|----------|-------------|
| RQ1 | Do displacement prediction intervals achieve nominal marginal coverage? | Marginal coverage per DOF across all calibration/evaluation splits |
| RQ2 | Do beam force prediction intervals achieve nominal marginal coverage? | Marginal coverage per force component |
| RQ3 | Do support/midspan/high-response regions show significant undercoverage? | Region-wise conditional coverage |
| RQ4 | Are prediction intervals practically useful (not too wide)? | Width / |y| ratio, absolute width vs physical tolerances |
| RQ5 | Which DOF / region / response level is most uncertain? | Width ranking, coverage gap ranking across all categories |

---

## 2. Method

**Primary method:** Component-wise split conformal prediction with absolute residual nonconformity score.

**Conservative variant:** Graph-level split conformal (graph-wise max score).

**Calibration split:** Test-internal graph-level 50/50 (Option B), supplemented by val-calibration (Option A) and sample-level (Option B-Sample).

| $\alpha$ | Nominal Coverage | Practical Meaning |
|:--------:|:----------------:|-------------------|
| 0.10 | 90% interval | Primary evaluation |
| 0.05 | 95% interval | Supplementary evaluation |

---

## 3. Phases

### Phase 1: Design + Split Audit (CURRENT)

| Deliverable | Status |
|-------------|--------|
| `docs/stage6_uq_design.md` | ✅ Complete |
| `docs/stage6_calibration_split_audit.md` | ✅ Complete |
| `docs/stage6_experiment_plan.md` | ✅ Complete |
| `docs/development_log.md` — Stage 6 entry | ✅ Pending |

### Phase 2: Code Implementation

**Prerequisites:**
- Locked MS-PI-HGT Full checkpoint
- Stage 5 NPZ prediction export
- Calibration split definition

**New scripts:**
1. `scripts/compute_conformal.py` — compute nonconformity scores, quantiles, intervals
2. `scripts/analyze_conformal.py` — compute coverage metrics, region diagnostics, generate figures

**Modifications:**
3. `scripts/export_full_predictions.py` — ensure compatibility with calibration split (if needed)

**Estimated effort:** 1-2 sessions

### Phase 3: Server Execution + Artifact Recovery

**Commands:**
```bash
# Step 1: Export full predictions for test set (if not already done)
conda activate pi_hgnn
python scripts/export_full_predictions.py --model mshgt \
    --run-dir outputs/baselines/MS_HGT/20260626170354 \
    --batch-size 8 --device cuda \
    --output-dir outputs/predictions/stage6_test

# Step 2: Compute conformal intervals
python scripts/compute_conformal.py \
    --predictions-dir outputs/predictions/stage6_test \
    --calibration-split test_cal_eval \
    --alpha 0.10 0.05 \
    --output-dir outputs/diagnostics/stage6_uq

# Step 3: Analyze and visualize
python scripts/analyze_conformal.py \
    --conformal-dir outputs/diagnostics/stage6_uq/<timestamp> \
    --output-dir outputs/diagnostics/stage6_uq/<timestamp>
```

**Estimated server time:** ~60 min (export) + ~30 min (conformal computation + analysis)

### Phase 4: Local Analysis + Paper Write-Up

- Review `stage6_uq_report.md`
- Update paper tables with UQ metrics
- Write Section 7 (Uncertainty Quantification) or equivalent
- Update `docs/development_log.md`

---

## 4. Experiment Matrix

### 4.1 Primary Experiments

| Experiment | Calibration | Evaluation | $\alpha$ | Score | Purpose |
|------------|:-----------:|:----------:|:--------:|------|---------|
| E1 | Test 50% | Test 50% | 0.10, 0.05 | Component absolute residual | **Primary result** |
| E2 | Val | Test | 0.10, 0.05 | Component absolute residual | Engineering baseline comparison |
| E3 | Test 3 samples | Test 4 samples | 0.10, 0.05 | Component absolute residual | Sample-level robustness |

### 4.2 Diagnostic Experiments

| Experiment | Calibration | Evaluation | $\alpha$ | Score | Purpose |
|------------|:-----------:|:----------:|:--------:|------|---------|
| D1 | Test 50% | Test 50% | 0.10 | Normalized residual | Cross-component relative comparison |
| D2 | Test 50% | Test 50% | 0.10 | Graph-wise max | Conservative graph-level guarantee |
| D3 | Test 50% | All test | 0.10 | Component absolute | Region-wise coverage |
| D4 | Test 50% | All test | 0.10 | Component absolute | High-response coverage |

### 4.3 Summary

| Type | Count |
|------|:-----:|
| Primary marginals | 2 (E1, E2) |
| Primary robustness | 1 (E3) |
| Diagnostic | 4 (D1-D4) |
| **Total** | **7** |

---

## 5. Success Criteria

### 5.1 Primary Targets (Must Pass)

| Criterion | Target | Measured By |
|-----------|:------:|-------------|
| 90% interval marginal coverage | ≥ 0.90 | E1, displacement + force |
| 95% interval marginal coverage | ≥ 0.95 | E1, displacement + force |
| Per-DOF coverage gap | ≤ 0.02 (from nominal) | E1, each of 6 DOF |
| High-response coverage gap | ≤ 0.05 (from nominal) | D3, top 10% displacement |
| Support/midspan coverage gap | ≤ 0.05 (from nominal) | D3, region-wise |

### 5.2 Secondary Targets (Should Pass)

| Criterion | Target | Measured By |
|-----------|:------:|-------------|
| Region-wise coverage gap explainable | Qualitative | D3 analysis |
| Graph-level conformal coverage | ≥ nominal | D2 |
| Interval width not prohibitive | Width / |y| ≤ 0.3 (mean) | E1, ratio metric |
| UQ identifies high-risk regions | Yes/No | D4 visualization |

### 5.3 Non-Success (Acceptable Outcomes)

- High-response coverage < nominal by > 0.05 — this is an accepted limitation of split conformal under covariate shift
- Dy coverage gap > 0.02 — consistent with Dy being the hardest component across all stages
- Graph-level intervals being impractically wide — expected, documented as trade-off

---

## 6. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|:----------:|------------|
| Calibration set contaminated by early stopping | Optimistic coverage, reduced rigor | High for Opt A, None for Opt B | Use Option B as primary |
| Small calibration set (1,750 graphs) leads to unstable quantiles | Slightly noisy coverage | Low | 1.75M+ calibration points per DOF is well above typical requirements |
| Within-sample correlations distort calibration | Coverage varies by sample | Medium for Option B-Sample | Compare with Option B graph-level |
| Intervals too wide for engineering use | Low practical value | Low | Report width / |y| ratio; structural engineering tolerances typically ~5-15% |
| Exchangeability violated under high-response shift | Undercoverage on tail nodes | Expected | Document as limitation; consider weighted conformal as future work |
| Implementation complexity | Delays Phase 2-3 | Low | Design complete; existing export pipeline reusable |

---

## 7. Paper Contributions

### 7.1 What Stage 6 Adds to the Paper

| Contribution | Evidence |
|-------------|----------|
| First calibrated UQ for steel truss surrogate model | Conformal coverage tables |
| Region-specific reliability assessment | Region-wise coverage metrics |
| Identification of high-uncertainty regimes | High-response coverage gap |
| Distribution-free guarantee | Split conformal methodology |

### 7.2 Paper Claims (Carefully Scoped)

✅ **Can claim:**
- The model provides calibrated marginal coverage for displacement and force predictions
- Coverage is robust across support and midspan regions
- Split conformal prediction identifies high-response nodes as the most uncertain regime
- Intervals are practically interpretable (e.g., "95% interval covers ±X mm for Dy")

❌ **Cannot claim:**
- Pointwise prediction guarantees
- Strict epistemic/aleatoric decomposition
- Calibration under data distribution shift beyond exchangeability
- Causal uncertainty quantification

### 7.3 UQ Table Design for Paper

**Main UQ table (marginal coverage):**

| Output | Domain | 90% Coverage | 90% Avg Half-Width | 95% Coverage | 95% Avg Half-Width |
|--------|--------|:-----------:|:-------------:|:-----------:|:-------------:|
| Displacement | All DOF | | | | |
| Displacement | Dx | | | | |
| Displacement | Dy | | | | |
| Displacement | Dz | | | | |
| Displacement | Rx | | | | |
| Displacement | Ry | | | | |
| Displacement | Rz | | | | |
| Force | All components | | | | |
| Force | Fx | | | | |
| Force | Fy | | | | |
| Force | Fz | | | | |
| Force | Mx | | | | |
| Force | My | | | | |
| Force | Mz | | | | |

**Region-wise coverage table (supplementary):**

| Region | 90% Coverage | 90% Avg Half-Width | Gap |
|--------|:-----------:|:-------------:|:---:|
| Support | | | |
| Midspan | | | |
| Transition | | | |
| High-response (P90+) | | | |

---

## 8. Timeline

| Phase | Duration | Depends On |
|-------|:--------:|------------|
| Phase 1: Design + Audit | ✅ Done | — |
| Phase 2: Code Implementation | 1 session | Phase 1 |
| Phase 3: Server Execution | ~1.5 hr server | Phase 2 |
| Phase 4: Analysis + Write-up | 1 session | Phase 3 |

**Total estimated effort:** 2-3 local sessions + 1 server session (~1.5 hr)

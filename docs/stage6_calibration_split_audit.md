# Stage 6 Calibration Split Audit

**Date:** 2026-06-29
**Dataset:** `processed/hetero_graph_dataset_v2`
**Split mode:** `by_sample` (existing)

---

## 1. Current Split Structure

The existing `by_sample` split partitions 70 unique SampleIDs into:

| Split | Sample Count | Sample IDs | Graphs (500 LC/sample) |
|-------|:-----------:|------------|----------------------:|
| Train | 56 | 1274, 1406, 1589, 1705, 1867, 1967, 2264, 2576, 2578, 2616, 2847, 3153, 3168, 3305, 3338, 3400, 3415, 3766, 3859, 4375, 4580, 4638, 4708, 4750, 4915, 4958, 5375, 5448, 5499, 5587, 5858, 5916, 6123, 6367, 6445, 6499, 6553, 6557, 6592, 6846, 6960, 7258, 7423, 7664, 7859, 7871, 8002, 8107, 8239, 8256, 8320, 8390, 8468, 8477, 8478, 8567 | 28,000 |
| Val | 7 | 1853, 4644, 5241, 5430, 6039, 6469, 7723 | 3,500 |
| Test | 7 | 1700, 3014, 3059, 3281, 4648, 4761, 5277 | 3,500 |

Each SampleID × LoadCase combination produces exactly one graph with:
- 1,056 mesh_nodes
- 1,646 beam_elements
- 832 plate_elements

**Key invariant:** No SampleID appears in more than one split. No LoadCase leakage between splits (all 500 LoadCases are present in each sample, so there is no LoadCase-level overlap concern).

---

## 2. Option A: Val Calibration → Test Evaluation

### 2.1 Setup

| Role | Source | Graphs | Sample IDs |
|------|--------|------:|------------|
| Calibration | Existing val split | 3,500 | 1853, 4644, 5241, 5430, 6039, 6469, 7723 |
| Evaluation | Existing test split | 3,500 | 1700, 3014, 3059, 3281, 4648, 4761, 5277 |

### 2.2 Calibration Set Size

- mesh_node calibration points: 1,056 × 3,500 = **3,696,000**
- beam_element calibration points: 1,646 × 3,500 = **5,761,000**

Both are more than sufficient for stable quantile estimation.

### 2.3 Validity Concern

The val split was used for **early stopping** during training (patience=80, best_epoch=134). This means:
- Val predictions are not strictly exchangeable with a fresh test sample
- The model was indirectly selected to perform well on val
- Conformal coverage on a val-calibrated test set may be slightly **optimistic** (narrower intervals than warranted)

**Assessment:** The risk is small in practice (early stopping is a mild selection mechanism), but it weakens the formal distribution-free guarantee. For a conference paper where UQ is a secondary contribution, Option A may be acceptable. For a journal submission requiring rigorous calibration, Option B is preferred.

### 2.4 Complexity

**Low.** No data modification needed. The existing `HeteroGraphDataset` can load val as calibration and test as evaluation directly.

---

## 3. Option B: Test-Internal Graph-Level Split

### 3.1 Setup

Partition the 3,500 test graphs randomly (or stratified by SampleID) into calibration and evaluation:

| Role | Allocation | Graphs | Sample IDs |
|------|-----------|-------:|------------|
| Calibration | Random 50% of test graphs | ~1,750 | All 7 test samples (~250 LC each) |
| Evaluation | Remaining 50% of test graphs | ~1,750 | All 7 test samples (~250 LC each) |

### 3.2 Calibration Set Size

- mesh_node calibration points: 1,056 × 1,750 = **~1.85M**
- beam_element calibration points: 1,646 × 1,750 = **~2.88M**

These are **still more than sufficient** — conformal prediction calibrates reliably with as few as 500-1000 calibration points.

### 3.3 Validity

✅ **Clean.** Neither calibration nor evaluation graphs have been used for training or early stopping. The exchangeability assumption holds for test-distribution inference.

### 3.4 Complexity

**Low.** Requires a new index file or split definition for the test-internal partition. Implementation details:
- Create `splits/split_test_cal_eval.json` with `calibration` and `evaluation` keys, each listing graph_ids
- Or generate on-the-fly with a fixed random seed

### 3.5 Graph-Level Stratification Consideration

Simple random split of test graphs may distribute LoadCases unevenly per SampleID. Since all 500 LoadCases per sample are distinct (different loading conditions), this is **not a concern** — there is no covariate shift between calibration and evaluation within the test set.

---

## 4. Option B-Sample: Test-Internal Sample-Level Split

### 4.1 Setup

Partition the 7 test samples by sample ID:

| Role | Sample IDs | Graphs | Notes |
|------|-----------|-------:|-------|
| Calibration | 3 samples of 7 | 1,500 | No overlap with training |
| Evaluation | 4 samples of 7 | 2,000 | No overlap with calibration |

### 4.2 Candidate Partition (Fixed Seed 42)

Using the same seed as the original split:

| Role | Sample IDs | Graphs |
|------|-----------|-------:|
| Calibration | 3014, 4648, 5277 | 1,500 |
| Evaluation | 1700, 3059, 3281, 4761 | 2,000 |

Or alternatively, random stratified assignment. The exact assignment should be fixed before any analysis begins.

### 4.3 Validity

✅ **Cleanest.** Preserves sample-level independence. Calibration and evaluation samples share **no design parameters** (no common SampleID). No LoadCase from a calibration sample appears in evaluation.

### 4.4 Calibration Set Size

- mesh_node calibration points: 1,056 × 1,500 = **~1.58M**
- beam_element calibration points: 1,646 × 1,500 = **~2.47M**

Still sufficient, though quantile estimates at the extreme tail (P95, P99) will have slightly higher variance than Option A.

### 4.5 Limitation: Small Sample Count

With only 3 samples in calibration, there is a risk that within-sample correlations distort score distribution estimates. Specifically:
- Each sample has a fixed structural topology and material configuration
- All 500 LoadCases within a sample share this topology
- If the 3 calibration samples happen to be systematically easier to predict than the 4 evaluation samples, coverage may be under-estimated

**Mitigation:** Compare results with Option B (graph-level, 50/50) to verify consistency.

---

## 5. Cross-Validation for Calibration (Not Recommended)

K-fold cross-conformal prediction is technically possible but **not recommended** for this task:

1. Increased computation (K forward passes over different splits)
2. More complex implementation
3. Weaker theoretical guarantees (no exact finite-sample coverage)
4. Marginal benefit given the huge calibration set sizes available

---

## 6. Region Definitions for Region-Wise Coverage

Using the same x-position quartile bins as Stage 5 diagnostics:

| Region | x-Position Bins | Physical Meaning |
|--------|----------------|------------------|
| Q1 (end) | 0-20% span | End/support neighborhood |
| Q2 | 20-40% span | Transition zone |
| Q3 (midspan) | 40-60% span | Midspan (maximum deflection) |
| Q4 | 60-80% span | Transition zone |
| Q5 (end) | 80-100% span | End/support neighborhood |

And by support status:

| Region | Definition |
|--------|-----------|
| Support | Nodes with any constrained DOF (support_flags != 0) |
| Free | Nodes with no constrained DOF (support_flags == 0) |

And by response magnitude:

| Region | Definition |
|--------|-----------|
| High-response | Top 10% of nodes by absolute displacement magnitude |
| Low-response | Remaining 90% of nodes |

---

## 7. Sample-Level Statistics (Test Samples)

| SampleID | Original Role | Graphs | mesh_node/graph | beam_element/graph | Total mesh points | Total beam points |
|----------|:------------:|:------:|:---------------:|:------------------:|:-----------------:|:-----------------:|
| 1700 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 3014 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 3059 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 3281 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 4648 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 4761 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| 5277 | Test | 500 | 1,056 | 1,646 | 528,000 | 823,000 |
| **Total** | | **3,500** | | | **3,696,000** | **5,761,000** |

---

## 8. Recommendation

| Priority | Option | Validity | Eval Size | Effort | Use Case |
|----------|--------|:--------:|:---------:|:-----:|----------|
| **Primary** 🏆 | **B (graph-level 50/50)** | ✅ Strict | 1,750 | Low | Main paper result |
| Supplementary 1 | B-Sample (sample-level) | ✅ Strictest | 2,000 | Low | Robustness check |
| Supplementary 2 | A (val calibration) | ⚠️ Weakened | 3,500 | None | Engineering baseline |

**Recommended primary:** Option B (graph-level 50/50), which balances strict exchangeability with sufficient evaluation size.

**Decision:** Implement all three options in the conformal analysis script. The main result table uses Option B. Supplementary tables compare all three for robustness.

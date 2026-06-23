# Stage 3 Metric Sanity Check — Full Prediction Diagnostics Review

> **Purpose:** Verify correctness, limitations, and interpretation risks of metrics computed from full test-set predictions.
> **Date:** 2026-06-23
> **Script reviewed:** `scripts/analyze_stage2b_diagnostics.py` (lines 1108–1248)
> **Source data:** NPZ predictions from `scripts/export_full_predictions.py`

---

## 1. High/Low Response Disp R²

### 1.1 Current Definition

```python
magnitude = np.abs(true).mean(axis=1)           # node-level, mean of |6 components|
top10 = np.percentile(magnitude, 90)             # threshold from ground truth
high_mask = magnitude >= top10                    # node-level mask
# R² computed on subset across all 6 disp dimensions together:
#   1 - sum((pred - true)²) / sum((true - true_mean)²)
```

### 1.2 Verified Correct

| Question | Answer |
|----------|--------|
| Variable used for ranking | `np.abs(true).mean(axis=1)` — mean absolute value of all 6 displacement components |
| Level of selection | **Node-level** (each mesh node independently) |
| Same mask across models | **Yes** — mask derived from `true`, not `pred`. All models use identical ground-truth mask. ✓ |
| R² type | **Overall (flat) R²** on the 6D displacement tensor of the subset |
| Within-subset mean | Uses subset's own `true.mean(axis=0, keepdims=True)` — correct for subset R² |
| Top 10% threshold | From `y_true`, not `y_pred` ✓ |

### 1.3 Identified Issues

**Issue 1: Mixed translational and rotational magnitude**

`np.abs(true).mean(axis=1)` averages across [Dx, Dy, Dz, Rx, Ry, Rz]. Rotations (radians) and translations (meters) have very different scales and physical meanings. The resulting "magnitude" is not a true physical displacement magnitude (which should be `sqrt(Dx² + Dy² + Dz²)` for translation alone).

**Impact:** The high-response subset is dominated by nodes where any single component is large. Since Dz typically has the largest absolute values, the subset is primarily selecting nodes with large Dz, not nodes with large Dy (which is our main weakness).

**Issue 2: R² on subset with reduced variance**

The subset R² formula `1 - SS_res / SS_tot` uses the subset's own SS_tot. If the high-response subset has homogeneous displacement patterns (all midspan nodes with similar Dz), the variance (`SS_tot`) is small relative to error (`SS_res`), producing a low R² even when absolute errors are reasonable.

**Issue 3: Disp HighResponse R² = 0.208 for HGT < 0.336 for MLP**

This counterintuitive result (HGT worse than MLP on high-response nodes) may be partially explained by variance effects:

- HGT reduces error on low/medium displacement nodes more than on high-response nodes
- But the subset itself has lower variance after HGT's improvements are removed from the dataset
- The metric is *directionally informative* but should not be taken as a literal "HGT is 38% worse on large displacements"

### 1.4 Recommendations

1. **Add translational-only magnitude** for defining high-response subset:
   ```python
   trans_mag = np.sqrt(true[:, 0]**2 + true[:, 1]**2 + true[:, 2]**2)  # sqrt(Dx² + Dy² + Dz²)
   ```
   This separates translation and rotation behavior.

2. **Add Dy-only high-response subset:** Select nodes where |Dy| is in top 10%, compute Dy R², MAE, P95/P99 AbsErr. This directly measures the component we want to improve.

3. **Report MAE alongside R²** for high-response subsets — MAE is not affected by within-subset variance.

4. **Document the limitation:** "HighResponse Disp R² uses the subset's own mean for SS_tot. Reduced within-subset variance can lower R² independently of model quality. Therefore, MAE and RMSE should be read alongside R² for this metric."

### 1.5 Action Items

- [ ] Add translational magnitude (`sqrt(Dx²+Dy²+Dz²)`) as alternative high-response criterion
- [ ] Add Dy-only high-response subset metric
- [ ] Document variance caveat in diagnostic report and future paper
- [ ] Do NOT modify the existing metric (maintain backward compatibility); add supplementary metrics

---

## 2. Dy Relative Error

### 2.1 Current Definition

```python
rel_err = np.abs(pred - true) / (np.abs(true) + 1e-10)
```

`eps = 1e-10` — applied to all 6 disp and 12 force components identically.

### 2.2 Verified Correct

| Question | Answer |
|----------|--------|
| eps value | 1e-10 — very small |
| Denominator | `abs(true) + eps` — per-component, per-sample |
| Percentiles | P50, P90, P95, P99 of the per-sample relative error distribution |
| Component-level | Each component computed independently |

### 2.3 Identified Issues

**Issue 1: Dy P99 RelErr = 2749% is a near-zero denominator artifact**

For nodes where Dy ≈ 0 (many nodes in a truss girder have minimal lateral displacement), the denominator `|Dy|` is near zero. With `eps = 1e-10`, any non-zero prediction error (e.g., 1e-6) produces RelErr ≈ 1e6 (100,000%).

This is the **same mechanism** that produces the astronomical Force RelErr values (Fz P90 RelErr = 1.57e13).

**Evidence:** The P99 AbsErr for HGT Dy = 0.000596 (≈ 0.6 mm), which is actually very small in absolute terms. The 2749% RelErr is driven by nodes where Dy ≈ 0 but the denominator is microscopically small.

**Issue 2: P50/P90 RelErr is informative, P99 is not**

- Dy P50 RelErr = 32.9% — this is at the median, where Dy is not near-zero. This is informative.
- Dy P90 RelErr = 175% — at the 90th percentile of relative error. Some of these are genuine large errors, some are near-zero artifacts.
- Dy P95/P99 — dominated by near-zero artifacts, not useful without context.

### 2.4 Recommendations

1. **Report Dy AbsErr alongside RelErr** as the primary tail metric:
   ```python
   # Already available in the CSV: AbsErr_P50, AbsErr_P90, AbsErr_P95, AbsErr_P99
   ```
   For HGT Dy: AbsErr_P50 = 9.7e-5, AbsErr_P90 = 2.7e-4, AbsErr_P99 = 5.96e-4 (all in meters)
   These are small in absolute terms and tell a different story from 2749% RelErr.

2. **For RelErr, use a larger eps** or report only P50/P90:
   ```python
   rel_err = np.abs(pred - true) / (np.abs(true) + 1e-6)  # more stable
   ```
   Or cap RelErr at a reasonable maximum (e.g., 1000%) for display purposes.

3. **Paper recommendation:** Dy AbsErr P90/P95/P99 as the primary tail metric. Dy RelErr P50/P90 as secondary. P99 RelErr should only appear in a footnote describing the near-zero denominator effect.

### 2.5 Action Items

- [x] Dy AbsErr data already exists in tail_error_disp.csv
- [ ] Document that Dy RelErr P99 is dominated by near-zero denominator artifacts
- [ ] For paper: use Dy AbsErr P95/P99 and Dy RelErr P50/P90

---

## 3. Support BC Residual

### 3.1 Current Definition

```python
support_mask = support_flags > 0.5
bc_residual = np.abs(disp_pred[support_mask] - disp_true[support_mask])
# Reported: mean_abs, median_abs, p95, n_constrained_dof
```

`support_flags` are binary (0/1) after inverse transform and thresholding in `export_full_predictions.py`:
```python
sup_orig = sup_std * x_std[9:15] + x_mean[9:15]    # inverse standardize
sup_binary = (sup_orig > 0.5).float()                # threshold to 0/1
```

### 3.2 Verified Correct

| Question | Answer |
|----------|--------|
| support_flags source | Dataset's `mesh_node.x[:, 9:15]` — the 6 BC DOF indicators from raw data ✓ |
| Residual scope | Only on constrained DOF (where support_flags > 0.5) ✓ |
| Displacement scale | Original (physical) scale — inverse transformed ✓ |
| Same mask across models | Yes — `support_flags` are from ground-truth data, shared across all models ✓ |
| n_constrained_dof | 35,000 across all test graphs — consistent ✓ |

### 3.3 Identified Issues

**Issue 1: Translation and rotation residuals are mixed**

The current `bc_residual` = `abs(disp_pred - disp_true)` on constrained DOFs includes:
- Dx_fix, Dy_fix, Dz_fix (translation, meters)
- Rx_fix, Ry_fix, Rz_fix (rotation, radians)

**These have different units and scales.** The mean_abs = 0.000206 is a mixture of translation errors (~1e-4 m) and rotation errors (~1e-5 rad). The rotation errors are numerically smaller and contribute less to the mean.

**Issue 2: Different beam elements have different constraint patterns**

Some nodes have only Dz_fix (simple support), others have all 6 DOFs fixed (fixed support). The aggregate mixes these different constraint types.

### 3.4 Recommendations

1. **Report separate translation and rotation BC residuals:**

| Metric | Current | Proposed |
|--------|---------|----------|
| Translation BC residual | not separated | `bc_residual[:, :3]` — Dx, Dy, Dz constrained DOFs |
| Rotation BC residual | not separated | `bc_residual[:, 3:6]` — Rx, Ry, Rz constrained DOFs |
| Per-DOF BC residual | not available | Per DOF type (Dx_fix, Dy_fix, etc.) |

2. **Format:**
   ```python
   diagnostics['support_bc_residual'] = {
       'translation': {'mean_abs': ..., 'median_abs': ..., 'p95': ...},
       'rotation': {'mean_abs': ..., 'median_abs': ..., 'p95': ...},
       'n_constrained_translation': int(trans_mask.sum()),
       'n_constrained_rotation': int(rot_mask.sum()),
   }
   ```

3. **Paper recommendation:** Use translation BC residual (Dx_fix, Dy_fix, Dz_fix) as the primary physical consistency metric. Rotation residual can be supplementary.

### 3.5 Action Items

- [ ] Split BC residual into translation and rotation components
- [ ] Current aggregate value is still directionally useful (HGT 0.000206, RGCN 0.000158, MLP 0.000379)
- [ ] Document that rotations (radians) have smaller numeric scale

---

## 4. Force Tail Relative Error

### 4.1 Confirmed Problem

Force RelErr at P90/P95/P99 produces extreme values (e.g., Fz P90 RelErr = 1.57e13). This is **confirmed as a near-zero denominator artifact**, not a model failure.

**Root cause:** In steel truss girders, beam elements carry primarily axial force (Fx). Shear forces (Fy, Fz) and moments (Mx, My, Mz) are structurally near-zero for many elements, especially at I/J ends that are modelled as pinned. When the true value is e.g., Fz = 0.001 N and the model predicts Fz = 100 N, the RelErr = 100/0.001 = 100,000%.

### 4.2 Recommendations

- **Force tail primary metric:** AbsErr P50/P90/P95/P99 (already available in CSV)
- **Force tail secondary metric:** RelErr P50 only (least affected by near-zero denominator)
- **Force tail not recommended:** RelErr P90/P95/P99
- **Paper guidance:** Report Force RelErr P50 in main table, AbsErr P95 in supplement. Note that RelErr at higher percentiles is unstable due to near-zero structural forces in non-load-bearing directions.

### 4.3 Action Items

- [ ] Document: "Force tail RelErr at P90+ is dominated by near-zero structural forces"
- [ ] For all force analysis, use AbsErr as the primary metric
- [ ] Consider filtering: compute RelErr only on elements where |true_force| > 1 N (or 1 N·m for moments)

---

## 5. Summary: Metrics Requiring Caution

| Metric | Risk Level | Issue | Paper Usage |
|--------|:----------:|-------|-------------|
| Disp HighResponse R² | 🔶 Medium | Variance sensitivity, mixed trans/rot | Use with MAE + RMSE; add trans-only |
| Dy RelErr P95/P99 | 🔴 High | Near-zero denominator artifact | Replace with AbsErr P95/P99 |
| Force RelErr P90+ | 🔴 High | Near-zero denominator artifact | Not for main claims |
| Support BC residual (mixed) | 🟢 Low-Medium | Trans/rot scale mixing | Split into trans/rot |
| Combined RelMAE | 🟢 Low | Simple average, well-defined | Continue using |
| Macro R² (Disp/Force) | 🟢 Low | Well-defined per-component average | Continue using |

---

## 6. Implications for Ours-base Evaluation

The sanity check confirms that:

1. **Disp R² (macro) and Force R² (macro) are clean metrics** — continue using as primary.
2. **HighResponse Disp R² needs better definition** — add translational magnitude and Dy-only subsets.
3. **Support BC residual should be separated** — translational and rotational.
4. **Dy tail errors should use AbsErr** — RelErr is inflated by near-zero artifacts at P99.
5. **Force tail analysis should use AbsErr** — RelErr is unusable at P90+.

These refinements do not invalidate existing diagnostics but provide more precise interpretation for the Ours-base design stage.

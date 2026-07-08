# Phase 3.1: Draft Consistency Fix Report

> **Date:** 2026-07-08
> **Task:** Fix main_draft.md inconsistencies against locked claims, physics_loss.py implementation, and Stage 6 UQ conventions.
> **Status:** All fixes applied.

---

## 1. Files Modified

| File | Changes |
|------|---------|
| `paper/ictai2026/main_draft.md` | **7 edits** — BC loss formula, link loss DOF count, calibration split wording, conformal guarantee, joint coverage paragraph, 3 TODO_VERIFY_CODE resolved |
| `paper/ictai2026/claim_boundary_checklist.md` | **1 edit** — "independently" → "separately" in reviewer Q&A |
| `paper/ictai2026/phase3_1_draft_consistency_fix_report.md` | **NEW** — this report |

---

## 2. Fix Details

### Fix 1: §3.5 Support BC Loss — Formula and Wording

**Before (incorrect):**
```
L_BC = mean (y_hat)^2
"penalizes non-zero predicted displacements at constrained DOFs"
```

**After (correct, verified against physics_loss.py):**
```
L_BC = mean (y_hat - y)^2 over constrained DOFs
"penalizes deviations from prescribed displacement values at constrained DOFs"
```

**Source code verification:** `physics_loss.py` line 121-125:
```python
loss = F.mse_loss(
    pred_disp[:, dof_indices][dof_mask],
    y_disp[:, dof_indices][dof_mask],  # <-- uses y_disp (ground truth), NOT zero
    reduction="mean",
)
```

**Added clarification:** "Since all constrained DOFs in our dataset have prescribed displacement values of zero, this loss effectively penalizes non-zero predictions at supports."

### Fix 2: §3.5 Link Loss — DOF Usage

**Before (incorrect):**
```
L_link = mean sum_{j=1}^{6} (y_hat_u - y_hat_v)^2
"Rigid structural links enforce identical displacements at their endpoint nodes"
```

**After (correct, verified against physics_loss.py):**
```
L_link = mean ||u_hat_u_trans - u_hat_v_trans||_2^2
"penalizes translation-component mismatches between linked node pairs"
"Only the three translational degrees of freedom are constrained; rotational components are excluded"
```

**Source code verification:** `physics_loss.py` line 178:
```python
loss = F.mse_loss(
    pred_disp[src, :3],   # <-- :3 = Dx, Dy, Dz only
    pred_disp[dst, :3],
    reduction="mean",
)
```

### Fix 3: §3.6 Calibration Split Wording

**Before:** "drawn from the validation split, distinct from training data"

**After:** "a held-out calibration set D_cal disjoint from the evaluation set"

**§4.5 updated to:**
"Our primary UQ experiment uses the test_graph_50_50 split, with 1,750 test graphs for calibration and 1,750 test graphs for evaluation (both disjoint from training)."

### Fix 4: §3.6 Conformal Guarantee Wording

**Before:** "|Coverage - (1-alpha)| <= O(1/sqrt(n))"

**After:**
"Under exchangeability between calibration and evaluation examples, split conformal prediction provides finite-sample marginal coverage at the target level (1-alpha), up to the standard quantile discretization effect."

And emphasized: "This guarantee is marginal — it applies to each separately calibrated component individually."

### Fix 5: §4.5 Joint Coverage Paragraph

**Before (contained prohibited language):**
- "independent components"
- "(0.90)^6 ≈ 53%"
- "correlation structure of the displacement components"

**After (clean):**
- "Component-wise split conformal prediction calibrates each output component separately and provides marginal coverage, but it does not guarantee simultaneous coverage over all displacement DOFs."
- "The lower joint coverage therefore reflects the stricter simultaneous-coverage requirement and motivates future work on vector-valued or graph-level conformal calibration."

### Fix 6: TODO_VERIFY_CODE All Resolved

| Location | Item | Status | Source |
|----------|------|--------|--------|
| §3.2 | plate_element feature dimension (6 vs 7) | ✅ **6 confirmed** | `hetero_schema.py` line 70 |
| §3.5 | λ_BC = 0.08 | ✅ **Confirmed** | Stage 5 experiment config |
| §3.5 | λ_link = 0.002 | ✅ **Confirmed** | Stage 5 experiment config |
| §4.1 | batch_size = 8 | ✅ **Confirmed** | Training config |
| §4.1 | seed = 42 | ✅ **Confirmed** | Training config |

### Fix 7: Derivative File — claim_boundary_checklist.md

"independently" → "separately" in reviewer Q&A for consistency with paper wording.

---

## 3. BC Loss: y_disp vs Zero Target

**BC loss uses y_disp (ground truth displacement) as target, NOT hard-coded zero.**

The formula is `MSE(pred_disp, y_disp)` over constrained DOFs. Since all constrained DOFs in this dataset have prescribed displacement values of zero, the practical effect is the same as penalizing non-zero predictions. But the formula is correct either way — the distinction matters only if a future dataset has non-zero prescribed displacements.

**Recommendation:** No further action needed. The updated wording accurately describes the implementation and adds a clarifying sentence about the zero-prescribed-displacement assumption.

---

## 4. Link Loss: 3 Translation DOF vs 6 DOF

**Link loss uses only 3 translational DOFs (Dx, Dy, Dz), NOT all 6.**

The implementation in `physics_loss.py` line 178 uses `pred_disp[:, :3]`, selecting only the first three columns (Dx, Dy, Dz). Rotational components are excluded.

**Recommendation:** No further action needed. The updated wording and formula accurately reflect this.

---

## 5. UQ Joint Coverage: Clean Status

✅ All prohibited language removed from main_draft.md:
- "independent components" — removed
- "(0.90)^6 ≈ 53%" — removed
- "correlation structure" — removed

---

## 6. Conformal Calibration Split: Corrected

✅ §3.6 now uses generic "held-out calibration set" language
✅ §4.5 explicitly states test_graph_50_50 as primary split with 1,750/1,750 split
✅ No longer claims validation calibration

---

## 7. Can Enter ChatGPT Polishing?

✅ **Yes.** All technical inaccuracies fixed:

| Prerequisite | Status |
|-------------|:------:|
| BC loss formula matches code | ✅ |
| Link loss DOF count matches code | ✅ |
| Feature dimensions match schema | ✅ |
| Lambda values verified | ✅ |
| batch_size/seed verified | ✅ |
| Joint coverage — no independent/0.9^6 | ✅ |
| Calibration split — no validation leakage | ✅ |
| Conformal guarantee — no O(1/sqrt(n)) | ✅ |
| Derivative files consistent | ✅ |

---

*Document version: v1.0 / 2026-07-08 / Phase 3.1 Draft Consistency Fix*

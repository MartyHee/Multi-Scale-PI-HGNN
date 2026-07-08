# Phase 3.1b: Mechanical Draft Cleanup Report

> **Date:** 2026-07-08
> **Task:** Remove duplicate content, check TODO_VERIFY_CODE residuals, verify structural cleanliness of main_draft.md.
> **Status:** All items resolved.

---

## 1. Files Modified

| File | Changes |
|------|---------|
| `paper/ictai2026/main_draft.md` | **2 edits** — remove duplicate paragraph (§3.2), remove duplicate heading (§3.6) |
| `paper/ictai2026/phase3_1b_mechanical_cleanup_report.md` | **NEW** — this report |

---

## 2. Fix Details

### Fix 1: §3.2 Duplicate Paragraph Removed

**Location:** After "Code verification" note and before "Figure 2" placeholder.

**Removed (was line 114):**
```
The graph is supervised with two target tensors: $\mathbf{y}_{\text{disp}}
\in \mathbb{R}^{N_m \times 6}$ for node displacements and
$\mathbf{y}_{\text{force}} \in \mathbb{R}^{N_b \times 12}$ for beam-end
forces. All graphs share the same topology (fixed node/edge indices)
across the 70 design samples; only node/edge features and target values
vary with design parameters and loading conditions.
```

This was an exact duplicate of the preceding paragraph (line 112). One copy retained.

### Fix 2: §3.6 Duplicate Heading Removed

**Location:** §3.6 boundary.

**Removed (was line 210):**
```
### 3.6 Component-Wise Split Conformal UQ
```

This was an immediate duplicate of the heading on line 209. One copy retained.

---

## 3. Verification Results

| Check | Method | Status |
|-------|--------|:------:|
| `TODO_VERIFY_CODE` residual | `grep TODO_VERIFY_CODE` | ✅ **0 matches** — none remain |
| Duplicate heading §3.6 | `grep -c "### 3.6 Component-Wise Split Conformal UQ"` | ✅ **1 occurrence** |
| Duplicate paragraph §3.2 | `grep -c "The graph is supervised with two target tensors"` | ✅ **1 occurrence** |
| All `### N.N` headings unique | `grep "^### \d\.\d+"` → visual scan | ✅ **All 18 headings distinct** |

### 3.1 Remaining "Code verification" Annotations

The following `> **Code verification:** ...` comments remain in the draft:

1. **§3.2** (line 110): `hetero_schema.py` confirms `plate_element.feature_dim = 6`, etc.
2. **§3.5** (line 204): `batch_size = 8` and `seed = 42` confirmed.

Per task instruction: *"Code verification 注释可以暂时保留在 markdown draft 中，但后续 LaTeX 正稿需要删除或转入内部注释。"* — these are acceptable at the draft stage.

### 3.2 Writing Notes TODO List

The `## Writing Notes` section (lines 373–399) contains a checklist with 6 strikethrough-completed items and 6 unchecked final-check boxes. This section is excluded from the paper body and is a working document for the authors. No action needed.

---

## 4. Can Enter ChatGPT Polishing?

| Prerequisite | Status |
|-------------|:------:|
| Duplicate paragraphs removed | ✅ |
| Duplicate headings removed | ✅ |
| No TODO_VERIFY_CODE residues | ✅ |
| All section headings unique | ✅ |
| Code verification notes acceptable (draft stage) | ✅ |
| No claim inaccuracies introduced | ✅ (no substantive content modified) |

✅ **Yes.** Mechanical cleanup complete. `main_draft.md` is structurally clean and ready for ChatGPT-based section-by-section polishing and compression.

---

*Document version: v1.0 / 2026-07-08 / Phase 3.1b Mechanical Draft Cleanup*

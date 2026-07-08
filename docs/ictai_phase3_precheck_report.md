# ICTAI 2026 — Phase 3-precheck: Citation Key and Figure Text Cleanup

> **Date:** 2026-07-08  
> **Task:** Cross-check citation keys (BibTeX ↔ skeleton), fix figure text, verify captions  
> **Status:** **All issues resolved. Paper draft ready to begin.**

---

## 1. Citation Key Consistency Check

### Method
Extracted all `\cite{...}` keys from `docs/ictai_related_work_skeleton.md` and matched against all `@` entry keys in `references/ictai_refs.bib`.

### Results

| Source | Total \cite{} calls | Match BibTeX? | Issues |
|--------|:-------------------:|:-------------:|--------|
| Skeleton §1 | 4 | ✅ 4/4 | None |
| Skeleton §2 | 4 | ✅ 4/4 | **2 fixed** |
| Skeleton §3 | 8 | ✅ 8/8 | None |
| Skeleton §4 | 9 | ✅ 9/9 | None |
| Summary table | 23 | ✅ 23/23 | **1 fixed** |
| **Total** | **24 (unique)** | ✅ **All matched** | |

### Issues Found and Fixed

| # | Skeleton Location | Broken Key | Correct Key | Type |
|---|---|---|---|---|
| 1 | §2 body text + citation list | `\cite{Sanchez-Gonzalez2020gns}` | `\cite{sanchezgonzalez2020gns}` | Case + hyphen mismatch |
| 2 | §2 body text | `\cite{fortuanto2022multiscale}` | `\cite{fortunato2022multiscale}` | Typo: missing "o" |
| 3 | Summary table row | `Sanchez-Gonzalez2020gns` (bare) | `sanchezgonzalez2020gns` | Same mismatch as #1 |

### All 24 Unique Cite Keys Verified

```
sacks1989design           ✅   vovk2022algorithmic        ✅
raissi2019pinns           ✅   lei2018distribution        ✅
karniadakis2021piml       ✅   angelopoulos2023gentle     ✅
haghighat2021physicsinformed ✅ zargarbashi2024conformal  ✅
battaglia2016interaction  ✅   zhang2025rrgnn             ✅
sanchezgonzalez2020gns    ✅   dadras2025hybrid           ✅
pfaff2021meshgraphnets    ✅   song2023structgnn          ✅
fortunato2022multiscale   ✅   wurth2024pimgn             ✅
schlichtkrull2018rgcn     ✅   hu2020hgt                  ✅
li2025universalhgcn       ✅   cai2024hgtpanels           ✅
du2026mefgcnn             ✅   ying2018diffpool           ✅
jain2024latticegraphnet   ✅   kovachki2023neuraloperator ✅
```

### Additional Checks

- **BibTeX orphan keys** (entries NOT referenced in skeleton): 31 entries intentionally not cited in skeleton (these are experiment baselines, supplementary, or background references that will be cited in Methods/Experiments sections). Not an issue.
- **BibTeX `others` entries:** `nourian2023truss`, `he2026mmpn`, `peng2026eemgnn`, `cao2025structureattention` — marked as TODO_VERIFY, not used in skeleton. Acceptable.

---

## 2. Figure Text Verification

### F4 — Main Results Bar Chart

| Check | Status |
|-------|:------:|
| Title wording | ✅ **Fixed**: "Main Results Across All Models" → **"Model Comparison on Structural FE Surrogate Prediction"** |
| y-axis label | ✅ R² ↑ |
| Value annotations | ✅ All bars annotated with exact R² values |
| Separator label | "Homogeneous → Heterogeneous →" → clear |
| Manifest caption | ✅ Updated to match new title |

### F5 — Physics Ablation Bar Chart

| Check | Status |
|-------|:------:|
| Title spelling | ✅ **"Physics Regularization Ablation"** — correct |
| Left panel title | "BC Constraint Satisfaction" — clear |
| Right panel title | "Force Tail Error" — clear |
| % change annotations | ✅ -22.3%, +14.5%, -38.8% visible on bars |
| Manifest caption | ✅ Accurate |

### F6 — UQ Coverage vs Half-Width

| Check | Status |
|-------|:------:|
| Panel 1 title | "Displacement Per-DOF Coverage (α=0.10)" ✅ |
| Panel 2 title | "Displacement Half-Width (α=0.10)" ✅ |
| Panel 3 title | ✅ **Fixed**: "Region Joint Coverage (α=0.10)" → **"Region Joint Coverage (Diagnostic, α=0.10)"** |
| Joint coverage labeled as Diagnostic? | ✅ **Fixed** — explicit "Diagnostic" in title |
| Marginal 90% reference line | ✅ Dashed line labeled "Marginal 90%" on Panel 3 |
| Half-Width convention | ✅ All axes say "Half-Width", not "Width" |
| Log scale for half-width | ✅ Wide dynamic range handled (2.1e-5 to 2.0e-3) |
| Subtitle | "Conformal Uncertainty Quantification — MS-PI-HGT-Full" — does not claim joint guarantee |
| Manifest caption | ✅ Updated: explicitly states "diagnostic observation, not a conformal marginal guarantee" |

### Caption Status

| Figure | Caption in manifest | Paper-ready? |
|:------:|:-------------------:|:------------:|
| F1 | Pipeline description | Needs polish |
| F2 | Schema description | Needs polish |
| F3 | Architecture description | Needs polish |
| **F4** | Model comparison caption | ✅ Paper-ready |
| **F5** | Physics ablation caption | ✅ Paper-ready |
| **F6** | UQ caption with diagnostic clarification | ✅ Paper-ready |

---

## 3. Summary of Files Modified

| File | Change | Type |
|------|--------|:----:|
| `docs/ictai_related_work_skeleton.md` | Fixed 3 citation keys: `Sanchez-Gonzalez2020gns`→`sanchezgonzalez2020gns` (2x), `fortuanto2022multiscale`→`fortunato2022multiscale` (1x) | ✅ Fix |
| `scripts/paper_figures/gen_f4_main_results_bar.py` | Title changed to "Model Comparison on Structural FE Surrogate Prediction" | ✅ Fix |
| `scripts/paper_figures/gen_f6_uq_coverage_halfwidth.py` | Panel 3 title: added "(Diagnostic)" to emphasize joint coverage is observed, not guaranteed | ✅ Fix |
| `outputs/figures/ictai2026/F4_main_results_bar.{png,svg}` | Regenerated with new title | ✅ Regenerated |
| `outputs/figures/ictai2026/F6_uq_coverage_halfwidth.{png,svg}` | Regenerated with "(Diagnostic)" annotation | ✅ Regenerated |
| `outputs/figures/ictai2026/figures_manifest.csv` | F4 + F6 captions updated | ✅ Fix |
| `references/ictai_refs.bib` | **No changes needed** — all existing keys were correct | ✅ Verified |
| `docs/ictai_literature_review_inventory.md` | **No changes needed** | ✅ Verified |

---

## 4. Conclusion

**Phase 3-precheck: PASS — All issues resolved.**

| Check | Result |
|-------|:------:|
| Citation keys: BibTeX ↔ skeleton | ✅ All 24 keys match |
| Orphan BibTeX entries | 31 intentionally unused (baselines/supplementary, to be cited elsewhere) |
| F4 title | ✅ Fixed and regenerated |
| F5 title spelling | ✅ Correct |
| F6 diagnostic annotation | ✅ Added "(Diagnostic)" to joint coverage panel |
| F6 joint coverage = observed, not guarantee | ✅ Stated in title + manifest caption |
| Captions in manifest | ✅ F4/F5/F6 have current captions |
| Regenerated figures | ✅ F4 + F6 PNG/SVG regenerated |

### Verdict

**The paper can proceed to initial draft writing (Phase 3).** All cross-reference integrity checks pass, figure text is accurate, and the UQ figure correctly distinguishes marginal coverage (conformal guarantee) from joint coverage (diagnostic observation).

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Phase 3-precheck Report*

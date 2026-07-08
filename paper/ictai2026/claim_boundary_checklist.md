# Claim Boundary Checklist — ICTAI 2026

> This document defines what the paper **can** claim, what it **cannot** claim, and how to phrase each claim safely.
> Every claim listed here has been cross-checked against available experimental evidence.

---

## 1. ✅ SAFE CLAIMS (Directly Supported by Evidence)

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| 1 | "Heterogeneous graph representation is essential for structural FE surrogate modeling — homogeneous GCN/GAT are outperformed by typed models (HGT, RGCN) on the same data." | Disp R²: GCN 0.8476, GAT 0.8421 vs RGCN 0.9366, HGT 0.9765 | 4.2 |
| 2 | "MS-HGT substantially improves over HGT — macro-anchor fusion raises Dy R² from 0.905 to 0.993." | Stage 4 result lock; all metrics locked from server artifacts | 4.3 |
| 3 | "Multi-scale macro-anchor fusion is the primary driver of improvement over local message passing." | MS-HGT additive 0.9950, gated 0.9952 vs HGT 0.9765 | 4.3 |
| 4 | "Physics-informed regularization with BC loss improves physical consistency (reduces constrained DOF MAE by 39%) without degrading overall accuracy." | Stage 5: BC DOF MAE 0.000242 → 0.000148 (Full) | 4.4 |
| 5 | "Component-wise split conformal prediction provides near-nominal marginal coverage — all 18 components within ±0.004 of nominal at 90% level." | Stage 6 UQ: Disp 89.74%, Force 89.97% | 4.5 |
| 6 | "MS-PI-HGT-Full achieves best force tail errors among physics variants (Force P95 AE = 37,917)." | Stage 5 result: P95 37,917 vs 38,000/39,756/39,277 | 4.4 |
| 7 | "MLP is a strong local-feature baseline for this data — its Force R² = 0.9824 shows that local features are highly informative." | Stage 2-A server artifact | 4.2 |
| 8 | "Structural links are constant across all 35K graphs (132 directed edges per graph), serving as static relational structure rather than variable input." | Dataset validation, figures_manifest F2 notes | 3.1 |
| 9 | "The current evaluation uses by-sample split (80/10/10) over 70 design samples, testing generalization to unseen design parameters under the same mesh topology." | README, experiment_audit | 4.1 |

---

## 2. ❌ PROHIBITED CLAIMS (Not Supported)

| # | Prohibited Claim | Why | Penalty if stated |
|---|-----------------|-----|-------------------|
| 1 | "SOTA" or "state-of-the-art" | Only compared against project-internal baselines; no external benchmark | Reviewer rejection — not verifiable |
| 2 | "Cross-topology generalization" | All 70 samples share identical mesh topology | Factual error — contradicts known data |
| 3 | "Full physical equilibrium" | Physics losses are specific penalties, not FEA residuals | Reviewer will detect overclaim |
| 4 | "Energy conservation" or "constitutive law satisfaction" | Not modeled or verified | Misleading domain claim |
| 5 | "Pointwise prediction guarantee" or "simultaneous guarantee" | Conformal coverage is marginal (per-DOF), not joint | Technical error — contradicts CP theory |
| 6 | "First" (unqualified) | Prior work exists in each individual direction | Overclaim — Song 2023, Li 2025, Dadras 2025 exist |
| 7 | "Complete surrogate for all bridge types" | Tested only on steel truss girder | Factual overreach |
| 8 | "1:1 replacement for FE simulation" | Accuracy for edge cases and tails needs further validation | Reviewer skepticism due to domain mismatch |

---

## 3. ⚠️ CAREFUL CLAIMS (Must Be Phrased with Qualification)

| # | Claim | Safe Phrasing | Unsafe Phrasing |
|---|-------|---------------|-----------------|
| 1 | "First combination of XYZ" | "To the best of our knowledge, no existing work combines heterogeneous graph representation, macro-anchor multi-scale fusion, and component-wise conformal UQ for structural FE surrogate modeling of steel truss girders." | "First to propose XYZ" |
| 2 | "Physics-informed loss improves results" | "The proposed physics-informed regularization improves physical consistency (39% reduction in support BC violation) without degrading predictive accuracy." | "Physics-informed loss significantly improves accuracy" (Disp R²: 0.9952 → 0.9948; negligible change) |
| 3 | "Conformal UQ coverage" | "Component-wise split conformal prediction provides near-nominal marginal coverage across all 18 output components." | "Conformal prediction guarantees 90% coverage for all predictions" |
| 4 | "Macro-anchor innovation" | "Our macro-anchor module draws inspiration from multi-scale graph architectures (Fortunato et al., 2022) but differs in using coordinate-based structural anchors with gated residual fusion rather than learned hierarchical pooling." | "Novel macro-anchor mechanism that no one has done before" |
| 5 | "Strong Dy improvement" | "The most challenging displacement component (Dy, vertical deflection) shows the largest relative improvement: Dy R² rises from 0.183 (MLP) / 0.905 (HGT) to 0.993 (MS-HGT), indicating that multi-scale fusion captures critical long-range bending behavior." | "We solved the Dy prediction problem" |
| 6 | "Generalization" | "Results demonstrate generalization to unseen design parameters under a shared topology, as evaluated through a by-sample train/val/test split." | "Generalization to new structures" |

---

## 4. CLAIM PHRASING TEMPLATES

### Abstract

> **Template:** "We propose MS-PI-HGT, a multi-scale physics-informed heterogeneous graph transformer for structural FE surrogate modeling. The framework constructs a heterogeneous graph from FEA data, applies typed attention, and introduces macro-anchor pooling with gated cross-scale fusion. Physics-informed regularization improves physical consistency. Component-wise split conformal prediction provides calibrated prediction intervals. On 35,000 steel truss bridge instances, MS-PI-HGT achieves Disp R²=0.995 and Force R²=0.993, substantially outperforming MLP and homogeneous GNN baselines."

**Safe?** ✅ No SOTA. No cross-topology. No physical equilibrium. No "first." Internal baselines.

### Introduction — Contribution Bullets

> 1. "A heterogeneous graph construction for steel truss FE systems with 3 physical node types and 5 edge types, capturing the multi-entity nature of structural FE models."
> 2. "A multi-scale macro-anchor fusion mechanism that pools micro-node features into structural segments and propagates global information through gated cross-scale connections, substantially improving long-range force transfer (Dy R²: 0.905 → 0.993)."
> 3. "Lightweight physics-informed regularization that improves support boundary condition satisfaction (39% reduction in constrained DOF MAE) without degrading prediction accuracy."
> 4. "Component-wise split conformal prediction providing near-nominal marginal coverage (89.74%/89.97% at 90% nominal) across all 18 output components, with clear distinction between marginal and joint coverage."

**Safe?** ✅ All claims quantifiable and verified by locked experiments.

### Methodology

| Component | Safe Phrasing | Citations |
|-----------|---------------|-----------|
| Hetero graph | "Following the heterogeneous graph paradigm \cite{hu2020hgt, schlichtkrull2018rgcn}..." | hu2020hgt, schlichtkrull2018rgcn |
| Typed attention | "Our backbone adopts the type-dependent attention mechanism of HGT \cite{hu2020hgt}..." | hu2020hgt |
| Macro anchor | "Inspired by hierarchical graph representations \cite{ying2018diffpool, jain2024latticegraphnet} and multi-scale MeshGraphNets \cite{fortunato2022multiscale}, we introduce a coordinate-based macro-anchor pooling..." | ying2018diffpool, jain2024latticegraphnet, fortunato2022multiscale |
| Physics loss | "Our physics-informed regularization builds on the paradigm of PINNs \cite{raissi2019pinns} and physics-informed graph learning \cite{song2023structgnn, wurth2024pimgn}..." | raissi2019pinns, song2023structgnn, wurth2024pimgn |
| Conformal UQ | "We apply split conformal prediction \cite{lei2018distribution, angelopoulos2023gentle} with component-wise nonconformity scores..." | lei2018distribution, angelopoulos2023gentle |

---

## 5. REGION METRICS BOUNDARY

| Metric | Can Report? | Caveat |
|--------|:-----------:|--------|
| Support Disp R² = 0.975 | ✅ | From Stage 5 analysis; consistent region definition |
| Free Disp R² = 0.982 | ✅ | Same |
| High-Response R² = 0.872 (Full) | ✅ | But drop from aggregate 0.995 indicates tail weakness |
| Region definition consistency | ⚠️ | Stage 4 vs Stage 5 region defs differ slightly; use Stage 5 for paper |
| Joint 6-DOF coverage by region | ✅ | Must label as (Diagnostic) — not a conformal guarantee |

---

## 6. LINK LOSS BOUNDARY

| Statement | Allowed? | Correct Wording |
|-----------|:--------:|-----------------|
| "Link loss did not converge during training" | ✅ | "The structural link consistency loss showed limited convergence under the current weighting scheme" |
| "Link loss improves accuracy" | ❌ | It does not: R² changes are negligible |
| "Full variant benefits from link loss" | ⚠️ | "The Full variant, which combines BC and link consistency losses, achieves the best balance of BC constraint satisfaction and force tail error reduction" |

---

## 7. REVIEWER RESPONSE PREPARATION

| Likely Reviewer Question | Prepared Answer |
|--------------------------|-----------------|
| "Why not compare with published methods?" | "Our focus is on systematic ablation within a controlled pipeline. The seven-model comparison (MLP → GCN → GAT → RGCN → HGT → MS-HGT → MS-PI-HGT) isolates the contribution of each design decision. Published methods are re-implemented in the same framework for fair comparison." |
| "Is 70 samples enough?" | "70 design samples × 500 load cases = 35,000 unique FE instances, providing 28,000/3,500/3,500 train/val/test splits. While the topology is shared, the load and design variation yields a diverse dataset." |
| "Why is Dy lower than other components?" | "Vertical deflection (Dy) in steel truss girders depends on complex interactions between bending, shear, and axial deformation across the full span. Local features alone are insufficient — long-range force transfer captured by multi-scale fusion is essential. MS-HGT raises Dy R² from 0.905 to 0.993." |
| "Why is joint coverage so low (25–57%)?" | "Each DOF is calibrated separately for 90% marginal coverage. The probability that all 6 DOF simultaneously fall within their intervals is necessarily lower under marginal calibration. This is a known property of component-wise conformal prediction, not a model deficiency." |
| "What about cross-topology generalization?" | "Not evaluated in this work — all samples share the same mesh topology. Cross-topology generalization is an important direction for future work requiring a multi-topology dataset." |
| "How does this compare to Song et al. (2023) StructGNN-E?" | "StructGNN-E achieves impressive zero-label results on homogeneous bar/truss graphs with physics loss. Our work targets a more complex heterogeneous setting with beam elements, plate elements, and structural links, and additionally provides multi-scale fusion and UQ — capabilities outside StructGNN-E's scope." |

---

## 8. CLAIM VERIFICATION PROTOCOL

Before any submission or public release:

1. Every numerical claim → check against `metrics_summary.json` or locked audit table
2. Every "improves" claim → check significance (is the difference meaningful, not noise?)
3. Every "first" claim → check literature inventory §10 (Novelty Assessment)
4. Every "generalization" claim → check data split and topology
5. Every "guarantee" claim → check CP theory (marginal ≠ joint, distribution-free ≠ assumption-free)
6. Every "physics" claim → check what is actually constrained (BC DOFs only, not full equilibrium)

---

*Document version: v1.0 / 2026-07-08 / ICTAI Phase 3 Initial Draft*

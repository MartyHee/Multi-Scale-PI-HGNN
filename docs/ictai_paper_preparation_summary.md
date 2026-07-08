# ICTAI 2026 — Paper Preparation Summary

> Consolidated readiness assessment, risk analysis, and action plan for ICTAI 2026 submission.  
> **Phase 1** — Project organization, research story, literature inventory, experiment audit.

---

## 1. Paper Readiness Assessment

| Dimension | Score (1-5) | Notes |
|-----------|:-----------:|-------|
| **Research story** | 5/5 | Clear problem chain; strong empirical progression; ICTAI-appropriate framing |
| **Experimental results** | 5/5 | All 13 models trained with complete metrics across all 18 output components |
| **Figures** | 1/5 | No publication-ready figures; 3 critical diagrams need creation from scratch |
| **Literature review** | 2/5 | BibTeX not collected; structural surrogate review not systematic |
| **Reproducibility** | 4/5 | Configs, remote_jobs, artifacts all documented; processed data not public |
| **Paper outline** | 3/5 | Rough outline exists; needs detailed section planning |
| **Abstract** | 4/5 | Skeleton abstract exists (~180 words); needs polishing |

**Overall readiness: 24/35 (69%).** Paper is structurally ready but needs significant figure and literature work.

---

## 2. Strongest Selling Points

1. **Clear empirical progression.** MLP (0.855) → GCN/GAT (0.848/0.842) → RGCN (0.937) → HGT (0.977) → MS-HGT (0.995) tells a coherent story that justifies each design decision.
2. **Multi-scale macro-anchor fusion works dramatically.** Dy R² from 0.905 (HGT) to 0.993 (MS-HGT) is the most impactful single innovation.
3. **Complete UQ pipeline.** Not many surrogate modeling papers include distribution-free conformal prediction with rigorous split diagnostics.
4. **Honest limitations.** Clear statements about shared topology, partial physics loss, marginal vs joint coverage build credibility.
5. **Reproducible infrastructure.** Config-driven, smoke-tested, artifact-tracked — rare in academic ML projects.

---

## 3. Biggest Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|:-----------:|------------|
| **ICTAI reviewers may not appreciate civil engineering context** | High — paper rejected as "too applied" or "too niche" | Medium | Frame as AI tools paper, not civil engineering; highlight method generality |
| **Figures not ready by deadline** | High — paper cannot be submitted without architecture diagram | Medium | Prioritize F1/F2/F3 creation; can reuse existing diagnostic plots with cleanup |
| **Missing structural surrogate literature** | Medium — related work section appears incomplete | High | Dedicated literature search; at least 5-10 relevant references needed |
| **Dy R² for early baselines was estimated in Phase 1; now resolved** | Low — exact values extracted (MLP=0.1833, GCN=0.1778, GAT=0.1649) | Low | Values locked from server artifacts |
| **No benchmark comparison** | Medium — reviewer may ask "compared to what published method?" | Medium | Frame as project-internal ablation; do not claim SOTA |
| **Shared topology limitation** | Medium — for ICTAI, single-topology is acceptable for application paper | Low | Clearly state boundary; do not overclaim |
| **Physics loss impact is modest** | Medium — "physics-informed" may be misleading | Low | Carefully worded claims: "improves physical consistency" not "improves accuracy" |

### Risk Mitigation Strategy

- **For reviewer skepticism:** Lead every section with the problem, not the method. Show that each design choice is motivated by a concrete limitation of the prior approach.
- **For missing literature:** Use Angelopoulos & Bates (2021) for UQ, Raissi et al. (2019) for physics-informed learning, Kipf & Welling (2017) for GNN as foundational references. These are widely recognized and will be accepted by ICTAI reviewers.
- **For figures:** Create a high-quality framework diagram first. A clear overview figure compensates for many other figure quality issues.

---

## 4. Missing Figures & Action Plan

| # | Figure | Priority | Creator | Est. Time | Deadline |
|---|--------|:--------:|---------|:---------:|:--------:|
| F1 | Overall framework pipeline | **Critical** | Draw.io | 3-4 hrs | Phase 2 |
| F2 | Heterogeneous graph schema | **Critical** | Draw.io / matplotlib | 1-2 hrs | Phase 2 |
| F3 | MS-HGT architecture detail | **Critical** | Draw.io | 2-3 hrs | Phase 2 |
| F4 | Main results bar chart | High | matplotlib script | 30 min | Phase 2 |
| F5 | Physics loss ablation bar | High | matplotlib script | 30 min | Phase 2 |
| F6 | UQ coverage-width (cleanup) | High | Existing script fix | 1 hr | Phase 2 |
| F7 | Region coverage plot (cleanup) | Medium | Existing script fix | 1 hr | Phase 2 |
| F8 | Region map (cleanup) | Medium | Existing script fix | 30 min | Phase 2 |
| F9 | Training curves (cleanup) | Medium | Existing script fix | 30 min | Phase 2 |
| F10 | High-response coverage (cleanup) | Low | Existing script fix | 30 min | Phase 2 |

**Phase 2 of ICTAI preparation should be: "Figure Generation and Cleanup."**

---

## 5. Literature Review Gaps

| Topic | Current Status | Action Needed |
|-------|---------------|---------------|
| Structural FE surrogate models | No references collected | Systematic search (Google Scholar: "surrogate model truss bridge graph neural network") |
| GNN for structural engineering | No references collected | Search: "graph neural network structural analysis" |
| Physics-informed ML for structures | Only PINNs (Raissi 2019) | Search: "physics-informed machine learning structural engineering" |
| Conformal UQ for surrogates | Angelopoulos & Bates (2021), Vovk (2005) | Search: "conformal prediction surrogate model" |
| Multi-scale GNN | No specific references | Search: "multi-scale graph neural network" or "hierarchical graph pooling" |
| MeshGraphNets / graph simulators | Pfaff et al. (2021) | ✅ Good reference; ensure cited correctly |

### Recommended Search Queries

```
("surrogate model" OR "emulator") AND ("finite element" OR "structural") AND ("graph neural network")
("graph neural network") AND ("truss" OR "frame" OR "bridge") AND ("surrogate")
("conformal prediction") AND ("finite element" OR "surrogate" OR "structural")
("physics-informed") AND ("graph neural network") AND ("structural")
```

---

## 6. Supplementary Experiments Checklist

| Possible Experiment | Needed? | Reason | Effort |
|--------------------|:-------:|--------|:------:|
| Per-component Dy R² for MLP/GCN/GAT | ✅ Resolved | MLP=0.1833, GCN=0.1778, GAT=0.1649 (from server artifacts) | Done |
| Region-consistent re-analysis across all models | ⚠️ Maybe | Only if region analysis is paper section | Medium |
| Training time for each model in unified format | ⚠️ Maybe | Only if runtime comparison is needed | Low (from train_log.csv) |
| Full by-loadcase split for gap analysis | ❌ No | Not needed for paper story | High |
| Cross-topology test (if data available) | ❌ No | Not in current data scope | N/A |
| Link loss with higher λ (0.01) | ⚠️ Maybe if time permits | Stage 5 result lock suggests this | High (server training) |
| Coverage-width ratio calibration | ⚠️ Maybe | Supplementary UQ analysis | Medium |

**Verdict:** No additional server training is strictly required for paper submission. All core experiments are complete. Only minor metric extraction and figure generation are needed.

---

## 7. Initial Paper Outline (Coarse)

### Suggested Structure (ICTAI 8-page limit)

| Section | Estimated Pages | Status | Notes |
|---------|:--------------:|:------:|-------|
| **1. Introduction** | 1.5 | Skeleton exists | Problem → gap → contribution |
| **2. Related Work** | 1.0 | ❌ Skeleton | Surrogates, GNN for physics, UQ |
| **3. Methodology** | 2.5-3.0 | Design exists | 3.1 Graph, 3.2 Backbone, 3.3 Multi-scale, 3.4 Physics, 3.5 UQ |
| **4. Experiments** | 1.5-2.0 | Data ready | 4.1 Setup, 4.2 Main results, 4.3 Ablation, 4.4 UQ |
| **5. Discussion** | 0.5 | ❌ Outline | Limitations, future work |
| **6. Conclusion** | 0.25 | Skeleton | Summary |
| **References** | — | ❌ Need collection | |
| **Figures & tables** | — | ❌ See audit | 5-6 figures, 5-6 tables |

### Section 3 (Methodology) Subsection Outline

```
3.1 Heterogeneous Graph Construction
  - FE data → HeteroData (3 node types, 5 edge types)
  - Feature encoding (mesh_node, beam_element, plate_element)
  - structural_link as edge, not node
  
3.2 Typed Graph Transformer Backbone
  - HGT-style type-dependent attention
  - Relation-specific message computation
  - Dual decoder: displacement head + force head
  
3.3 Multi-Scale Macro-Anchor Fusion  ← KEY CONTRIBUTION
  - X-coordinate anchor pooling (MacroAnchorPool)
  - Macro-level graph message passing (MacroGNN)
  - Cross-scale gated fusion (CrossScaleFusion)
  
3.4 Physics-Informed Regularization
  - Support boundary condition loss (L_BC)
  - Structural link consistency loss (L_link)
  - Combined objective
  
3.5 Conformal Uncertainty Quantification
  - Component-wise split conformal prediction
  - Finite-sample quantile estimation
  - Coverage metrics (marginal, region-wise, high-response)
```

---

## 8. Information Checklist for ChatGPT/GPT Paper Generation

If passing the project context to an LLM for initial paper draft generation, prepare:

### Required Inputs for GPT

| Item | Prepared? | Location |
|------|:---------:|----------|
| Research story (this doc) | ✅ | `docs/ictai_research_story.md` |
| Literature inventory | ✅ | `docs/method_literature_inventory.md` |
| Experiment audit | ✅ | `docs/experiment_result_and_figure_audit.md` |
| Main metrics table (CSV-ready) | ✅ | `outputs/diagnostics/stage5_physics/paper_main_table.csv` |
| Stage 5 ablation data | ✅ | `outputs/diagnostics/stage5_physics/stage5_ablation_table.csv` |
| Stage 6 UQ summary | ✅ | `outputs/diagnostics/stage6_uq/test_graph_50_50/20260629082344/conformal_summary.json` |
| Per-component metrics | ✅ | All metrics_summary.json files |
| PyG/HGT architecture details | ✅ | `src/models/baselines/` source code |
| Baseline model descriptions | ✅ | CLAUDE.md sections 7.2-7.3 |
| Stage 4 result lock | ✅ | `docs/stage4_result_lock.md` |
| Stage 5 result lock | ✅ | `docs/stage5_result_lock.md` |

### Suggested Prompt Structure for GPT

```
You are writing an ICTAI 2026 conference paper (8 pages, IEEE format) on:

"Multi-Scale Physics-Informed Heterogeneous Graph Transformer for 
Structural Finite Element Surrogate Modeling"

CONTEXT:
1. Project is a steel truss girder FE surrogate modeling using heterogeneous GNN
2. ICTAI track: AI Applications / Tools with AI
3. Key innovation: MS-HGT (typed attention + macro-anchor fusion + physics loss + conformal UQ)
4. All baselines are project-internal comparisons (not against published SOTA)

KEY RESULTS (from paper_main_table.csv):
[full table]

METHOD HIGHLIGHTS:
- HeteroData with 3 node types, 5 edge types
- HGT-style typed attention backbone
- MacroAnchorPool (x-coordinate-based) → MacroGNN → CrossScaleFusion
- L_BC (support BC) + L_link (structural link consistency)
- Component-wise split conformal prediction

PAPER OUTLINE:
[as defined in section 7 above]

PLEASE WRITE:
1. A complete first draft following ICTAI format
2. Focus on the AI/ML methodology framing (not civil engineering)
3. Include section 3 (Methodology) with mathematical formulations
4. Tables and figures marked as placeholders with their data
5. References as placeholder [REF] markers

CONSTRAINTS:
- 8 pages max including references
- IEEE conference format
- No overclaiming ("first", "SOTA", "generalizes to all structures")
- Clearly state: shared topology, marginal coverage, partial physics loss
```

---

## 9. Next Steps (Phase 2)

| # | Task | Priority | Est. Time | 
|---|------|:--------:|:---------:|
| 1 | ~~Extract MLP/GCN/GAT per-component Dy R²~~ ✅ Done (Phase 1.5) | — | — |
| 2 | Create F1: Overall framework diagram | **Critical** | 3-4 hrs |
| 3 | Create F2: Heterogeneous graph schema | **Critical** | 1-2 hrs |
| 4 | Create F3: MS-HGT architecture detail | **Critical** | 2-3 hrs |
| 5 | Generate F4: Main results bar chart (matplotlib) | High | 30 min |
| 6 | Generate F5: Physics loss bar chart | High | 30 min |
| 7 | Clean up F6-F10 with proper formatting | Medium | 2-3 hrs |
| 8 | Systematic literature search | High | 2-3 hrs |
| 9 | Collect BibTeX for all references | Medium | 1 hr |
| 10 | Set up IEEE LaTeX/Word template | Medium | 30 min |
| 11 | Generate GPT draft | Medium | 1 hr |
| 12 | Review and revise draft | High | 4-6 hrs |

**Total Phase 2 estimate:** ~15-20 hours of work (2-3 sessions).

---

## 10. Decision Summary

| Question | Answer |
|----------|--------|
| **Can we submit to ICTAI 2026?** | ✅ **Yes** — the research is complete, results are strong, story is coherent |
| **What is the biggest gap?** | **Figures** — no publication-ready diagrams, 3 critical figures missing |
| **What is the biggest risk?** | Reviewer perception as "civil engineering application" rather than "AI tool" |
| **Are any additional experiments needed?** | **No** — all core experiments are complete |
| **Should we run more server training?** | **No** — experimental results are final and locked |
| **Can paper draft be generated now?** | ✅ **Yes** — all metrics and method details are documented |
| **ICTAI deadline awareness?** | Check ICTAI 2026 CFP for exact deadline; typically ~April-May for Fall conference |

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Preparation Phase 1*

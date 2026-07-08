# ICTAI 2026 — Research Story

**Target venue:** IEEE International Conference on Tools with Artificial Intelligence (ICTAI)  
**CORDIS track (suggested):** AI Applications / Machine Learning Systems / Uncertainty-Aware AI  
**Paper type:** Regular (conference) — target 8 pages

---

## 1. Target Venue Positioning

### Why ICTAI?

ICTAI's scope includes "Tools with Artificial Intelligence" — systems, frameworks, and methodologies where AI is the **tool** for solving practical problems. This fits the project's identity:

| ICTAI Aspect | How This Project Fits |
|--------------|-----------------------|
| **AI as a tool** | A surrogate AI model replaces expensive FE simulations for structural engineering |
| **Machine learning systems** | Reproducible pipeline from data construction → heterogeneous graph learning → uncertainty quantification |
| **Structured data & graph learning** | Heterogeneous graph representation of physical FE systems |
| **Uncertainty-aware AI** | Split conformal prediction with component-wise calibrated intervals |
| **Application-driven** | Real steel truss girder data; all 18 prediction components are physically meaningful |

### What ICTAI Is NOT For

- Do **not** position as a pure civil engineering paper (target: *Engineering Structures*, *Computers & Structures*)
- Do **not** position as a pure graph theory paper (target: *NeurIPS*, *ICLR*)
- Do **not** claim "new SOTA on benchmark" — this is a domain-specific application paper
- Appropriate framing: "AI tool for rapid structural simulation with built-in uncertainty awareness"

---

## 2. One-Sentence Contribution

### Technical Version

> A multi-scale heterogeneous graph transformer with macro-anchor fusion and conformal uncertainty quantification for surrogate modeling of steel truss girder finite element systems.

### Engineering Application Version

> An end-to-end AI framework that predicts node displacements and beam-end internal forces for steel truss girder bridges with calibrated uncertainty intervals, enabling rapid design iteration without full finite element simulation.

### Conference Abstract Version

> We propose MS-PI-HGT, a physics-informed heterogeneous graph transformer that achieves Disp R²=0.995 and Force R²=0.993 on steel truss FE surrogate modeling through typed attention, multi-scale macro-anchor fusion, and support-boundary regularization, with component-wise split conformal prediction providing near-nominal (89.74% / 89.97% at 90% target) marginal coverage across all 18 output components.

---

## 3. Research Motivation (Problem Chain)

```
FE simulation is the standard tool
    ↓ but expensive for iterative tasks
Surrogate model is needed
    ↓ but structural FE data is inherently heterogeneous
MLP ignores graph structure; homogeneous GNN mixes entity types
    ↓ leading to poor Dy and high-response predictions
Typed message passing (RGCN/HGT) significantly improves accuracy
    ↓ but local message passing is insufficient for long-range force transfer
Multi-scale macro-anchor fusion captures global structural behavior
    ↓ but physical consistency is not guaranteed by data alone
Physics-informed regularization improves BC satisfaction and tail errors
    ↓ but no uncertainty quantification for engineering decisions
Split conformal prediction provides calibrated prediction intervals
    ↓
MS-PI-HGT: A complete framework from data to uncertainty-aware surrogate
```

### Core Thesis Statement

> Explicitly modeling the heterogeneous physical structure of FE systems — through typed message passing, multi-scale macro-anchor fusion, physics-informed regularization, and conformal UQ — is essential for building accurate, reliable, and practically useful surrogate models for structural engineering.

---

## 4. Proposed Method Story

### 4.1 Heterogeneous Structural Graph Representation

Structural FE data is naturally a heterogeneous graph. A steel truss girder bridge has:
- **Mesh nodes** (1056 per graph) — spatial locations, loads, support constraints
- **Beam elements** (1646 per graph) — section/ material / geometry properties
- **Plate elements** (832 per graph) — thickness, material
- **Structural links** (132 directed edges, 66 physical rigid connections) — rigid connections between nodes

We construct a 3-node-type, 5-edge-type heterogeneous graph `HeteroData` object for each design-loadcase pair. The graph topology is static across all samples; only node/edge features vary with design parameters and loading conditions.

### 4.2 Typed Graph Transformer Backbone

**Baseline RGCN** uses relation-specific convolution (HeteroConv + SAGEConv), validating that typed message passing drastically outperforms homogeneous GCN/GAT.

**HGT** (Heterogeneous Graph Transformer) further extends this with type-dependent attention: separate linear projections for each node type, relation-specific attention, and type-dependent message aggregation. HGT achieves Dy R² = 0.905, confirming the value of typed attention.

### 4.3 Multi-Scale Macro-Anchor Fusion (MS-HGT)

*This is the first core methodological innovation.*

Local message passing is insufficient for long-range force transfer in large-span structures. We introduce:

1. **Macro anchor pooling** — partition mesh nodes into 12 segments by x-coordinate, aggregate micro-node representations into anchor-level macro features
2. **Macro graph** — bidirectional sequential chain connecting adjacent anchors, processed by 2-layer SAGEConv
3. **Cross-scale gated fusion** — gated residual connections that selectively merge macro representations back into micro-node features at each GNN layer

**Result:** MS-HGT gated improves Dy R² from 0.905 (HGT) to 0.993, and Disp R² from 0.977 to 0.995.

### 4.4 Physics-Informed Regularization

*The second innovation: domain-appropriate physics constraints.*

Two lightweight physics-inspired loss terms that respect current data capabilities:

1. **Support boundary condition loss** ($\mathcal{L}_{BC}$) — penalizes non-zero predicted displacement at constrained DOFs. Reduces constrained DOF MAE by 39%.
2. **Structural link consistency loss** ($\mathcal{L}_{link}$) — penalizes displacement mismatch at RIGID link endpoints.

These losses do *not* claim full equilibrium but improve physical consistency without degrading prediction accuracy. The Full variant achieves the best force tail errors (P95, P99, Max) among all variants.

### 4.5 Post-Hoc Conformal UQ

*The third innovation: distribution-free reliability assessment.*

Component-wise split conformal prediction with absolute residual nonconformity score:
- 18 output components, each calibrated separately (6 displacement + 12 force)
- Finite-sample correction ($k = \lceil (n+1)(1-\alpha) \rceil$)
- Three calibration split strategies (test-internal, sample-level, val→test)
- Coverage metrics: marginal, region-wise (support, midspan, free), high-response, graph-level

**Result:** Near-nominal marginal coverage across all components (max gap = 0.004), robust across split strategies.

---

## 5. Main Claims and Boundaries

### ✅ Can Claim (Supported by Evidence)

1. **Heterogeneous graph representation** is essential — homogeneous GCN/GAT are outperformed by typed models on the same data (Disp R²: 0.848 vs 0.977 for HGT).
2. **MS-HGT substantially improves over HGT** — Dy R²: 0.905 → 0.993, RelMAE: 0.068 → 0.052 on the current by-sample split.
3. **Multi-scale macro-anchor fusion is the primary driver** of improvement — additive variant (0.9950) and gated variant (0.9952) both far exceed HGT (0.9765).
4. **Physics-informed regularization improves physical consistency** without degrading accuracy — BC loss reduces constrained DOF MAE by 39%; Full variant achieves best force tail errors.
5. **Component-wise split conformal prediction provides near-nominal marginal coverage** — all 18 components within 0.004 of nominal at 90% and 95% levels.
6. **The framework is reproducible** with documented pipeline, configs, artifacts, and diagnostics.

### ❌ Cannot Claim (Explicit Boundaries)

1. **Cross-topology generalization** — not tested; all 70 samples share the same mesh topology.
2. **Complete physical equilibrium** — physics losses are specific penalties, not full FEA residual.
3. **Energy conservation or constitutive law satisfaction** — not modeled.
4. **Pointwise or simultaneous UQ guarantee** — coverage is marginal (per-DOF), not joint (all DOF).
5. **Universal superiority over all scientific GNN methods** — comparison is against project-internal baselines.
6. **Plate element stress/force prediction** — no labels available in current data.
7. **Deployment-ready for all bridge types** — tested only on steel truss girder bridges.

### ⚠️ Careful Claims (Qualified)

- "Physics-informed regularization improves **physical consistency**" — correct; do not say "improves accuracy"
- "First calibrated UQ for steel truss surrogate model" — OK only if literature search confirms
- "Split conformal prediction provides **distribution-free** marginal coverage" — yes, per theory

---

## 6. Abstract Skeleton (180 words, ICTAI-style)

> Finite element analysis (FEA) is the standard tool for structural engineering design, but its computational cost makes iterative tasks — design space exploration, uncertainty quantification, optimization — prohibitively expensive. Data-driven surrogate models offer an alternative, yet standard MLP and homogeneous graph neural networks fail to fully exploit the heterogeneous structure of FEA data, which contains diverse physical entities (nodes, beam elements, plate elements, structural links) with different feature types and relational dependencies.
>
> We propose MS-PI-HGT, a multi-scale physics-informed heterogeneous graph transformer for structural FEA surrogate modeling. The framework constructs a heterogeneous graph from FEA data, applies typed attention for relation-specific message passing, and introduces a macro-anchor pooling mechanism with cross-scale gated fusion to capture long-range force transfer. Physics-informed regularization terms — support boundary condition loss and structural link consistency loss — improve physical consistency without degrading predictive accuracy. Finally, component-wise split conformal prediction provides calibrated prediction intervals.
>
> Evaluated on 35,000 steel truss girder bridge FEA instances spanning 70 design samples, MS-PI-HGT achieves Disp R²=0.995, Force R²=0.993, and RelMAE=0.052, substantially outperforming MLP (0.855/0.982/0.088) and homogeneous GNN baselines. Uncertainty intervals achieve near-nominal marginal coverage across all 18 output components. The framework provides a practical AI tool for rapid structural evaluation with built-in reliability assessment.

---

## 7. Paper Outline (Tentative)

1. **Introduction** — Motivation, problem statement, contributions (1.5 pages)
2. **Related Work** — Scientific ML surrogates, graph neural networks for physics simulation, UQ for surrogate models (1 page)
3. **Methodology** (3 pages)
   - 3.1 Heterogeneous Graph Construction
   - 3.2 Typed Graph Transformer Backbone
   - 3.3 Multi-Scale Macro-Anchor Fusion
   - 3.4 Physics-Informed Regularization
   - 3.5 Conformal Uncertainty Quantification
4. **Experiments** (2 pages)
   - 4.1 Dataset and Experimental Setup
   - 4.2 Baseline Comparison (Table 2)
   - 4.3 Multi-Scale Ablation (Table 3)
   - 4.4 Physics Regularization Ablation (Table 4)
   - 4.5 Uncertainty Quantification (Table 5)
   - 4.6 Region-wise and Tail Error Analysis
5. **Discussion** — Limitations, failure modes, practical considerations (0.5 pages)
6. **Conclusion** — Summary, future work (0.5 pages)

---

## 8. Target Contributions for ICTAI Reviewers

| Reviewer Type | What They Care About | What to Emphasize |
|---------------|---------------------|-------------------|
| AI methods | Novelty of approach, soundness of evaluation | MS-HGT macro-anchor, conformal UQ pipeline |
| Applications | Practical impact, real data, reasonable baselines | Real steel truss data, improvement over MLP/GCN/GAT |
| Systems & tools | Reproducibility, code availability, configurability | Config-driven pipeline, remote_jobs, smoke tests |
| Uncertainty | Methodological rigor, calibration quality | Finite-sample correction, component-wise scores, split robustness |

---

## 9. Key Takeaways for Paper Writing

1. **Lead with the problem, not the model.** "FE simulation is expensive → surrogate needed → heterogeneous data → our framework" is more compelling than "here is a graph transformer with modifications."
2. **Reference the baselines correctly.** Show that MLP is strong (local features are informative), homogeneous GCN/GAT fail to use graph structure, and each progressive improvement (typed → multi-scale → physics → UQ) adds demonstrable value.
3. **Do not oversell physics loss.** The honest story (BC improves constraint satisfaction; link loss doesn't converge; Full variant balances all criteria) is more credible than claiming "physics-informed learning guarantees physical consistency."
4. **Split conformal UQ is a strength.** It's simple, rigorous, and practical — exactly what ICTAI values.
5. **Tables and figures must tell a story.** The progression from MLP (0.855) → HGT (0.977) → MS-HGT (0.995) → MS-PI-HGT (0.995 + physics + UQ) is a clear visual narrative.

---

## 10. Plots / Figures Needed

- **Figure 1:** Heterogeneous graph schema illustration (mesh_node, beam_element, plate_element, structural_link)
- **Figure 2:** MS-HGT architecture diagram (micro message passing → macro anchor → cross-scale fusion → dual decoder)
- **Figure 3:** Main results bar chart (Disp R², Force R² across methods)
- **Figure 4:** Physics loss ablation — constrained DOF MAE comparison
- **Figure 5:** Coverage-width plot (UQ calibration)
- **Table 1-6:** As defined in experiment audit

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Preparation Phase 1*

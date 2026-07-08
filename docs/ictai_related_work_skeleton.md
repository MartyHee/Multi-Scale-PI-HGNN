# ICTAI 2026 — Related Work Skeleton

> **Paper:** Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Structural Finite Element Surrogate Modeling  
> **Target:** ICTAI 2026 (8-page conference paper)  
> **Related Work target length:** ~1.2 pages  

---

## Section 1: Surrogate Modeling for Structural Simulation

> **Role:** Establish the engineering motivation and background. Show that FE simulation is expensive, surrogate modeling is a known solution, but existing surrogates have limitations for complex heterogeneous structures.

**Skeleton paragraph:**

> Finite element (FE) analysis is the standard tool for structural design and assessment in civil and mechanical engineering, but its computational cost becomes prohibitive for tasks requiring many evaluations, including design optimization \cite{sacks1989design}, parametric studies, and uncertainty quantification. Surrogate models — also known as metamodels or emulators — replace expensive FE simulations with fast data-driven approximations. Classical approaches include Gaussian process regression (kriging) \cite{sacks1989design} and polynomial chaos expansion, which achieve good accuracy on low-dimensional problems but scale poorly with high-dimensional input spaces. More recently, deep neural network surrogates have demonstrated orders-of-magnitude speedups across diverse physics domains \cite{raissi2019pinns, karniadakis2021piml}. However, most neural surrogates for structural systems rely on either fully-connected architectures \cite{haghighat2021physicsinformed} that discard topological information, or convolutional networks that require regular grid representations poorly suited to unstructured FE meshes. These limitations motivate the use of graph-based representations that natively encode the connectivity and topology of structural FE systems. For steel truss bridge structures — which combine beam elements, plate elements, and rigid links — a graph representation further requires heterogeneous node and edge types to capture distinct physical roles, motivating the approach developed in this work.

**Citation placeholders:**
- `\cite{sacks1989design}` — Classical method: DACE/kriging for computer experiments
- `\cite{raissi2019pinns}` — PINN: deep learning for PDE surrogate
- `\cite{karniadakis2021piml}` — PIML: survey of physics-informed ML
- `\cite{haghighat2021physicsinformed}` — PI deep learning for solid mechanics surrogate

**Key narrative points:**
- FE is expensive → surrogate needed
- Classical surrogates limited in high dimensions
- Deep surrogates fast but lose topological structure
- Graph representation naturally encodes FE topology
- Steel truss bridges specifically need heterogeneous graph → this paper

---

## Section 2: Graph Neural Networks for Physical Systems

> **Role:** Introduce the GNN-based simulation paradigm. Position our work in the lineage of Interaction Networks → GNS → MeshGraphNets. Show that the encoder-processor-decoder template is established, but existing work is typically homogeneous and single-output.

**Skeleton paragraph:**

> Graph neural networks (GNNs) have emerged as a powerful framework for learning physical simulations directly on unstructured representations. Battaglia et al. \cite{battaglia2016interaction} introduced Interaction Networks for object- and relation-centric physical reasoning, establishing that graph-structured computation with learned message functions can predict complex dynamics. Sanchez-Gonzalez et al. \cite{sanchezgonzalez2020gns} proposed the Graph Network-based Simulator (GNS), an encoder-processor-decoder architecture that generalizes across fluids, rigid solids, and deformable materials by iteratively applying learned message passing steps on particle graphs. Pfaff et al. \cite{pfaff2021meshgraphnets} extended this paradigm to mesh-based simulation with MeshGraphNets, achieving 1–2 orders of magnitude speedup over classical solvers on cloth, structural mechanics, and aerodynamics benchmarks. The key architectural insight — encode the physical state into a graph, apply K steps of processor message passing, decode to target quantities — underpins most current mesh-based surrogate models \cite{fortunato2022multiscale}. However, these approaches operate on homogeneous graphs where all nodes represent the same entity type (particles or mesh vertices). Structural FE problems involve multiple physically distinct entity types: mesh nodes that carry displacements, beam and plate elements that carry internal forces, and constraint relationships such as rigid links. A homogeneous graph conflates these distinct physical roles, limiting predictive accuracy. Furthermore, existing simulators are typically designed for single-output tasks (e.g., node displacement or element stress), whereas structural engineering requires simultaneous prediction of both nodal displacements and element-end internal forces for complete structural assessment.

**Citation placeholders:**
- `\cite{battaglia2016interaction}` — Interaction Networks
- `\cite{sanchezgonzalez2020gns}` — GNS (use citation key from BibTeX)
- `\cite{pfaff2021meshgraphnets}` — MeshGraphNets
- Optional: `\cite{fortunato2022multiscale}` — MultiScale MeshGraphNets

**Key narrative points:**
- Establish canonical lineage: Interaction Networks → GNS → MeshGraphNets
- Acknowledge encoder-processor-decoder as established paradigm
- Point out limitations: homogeneous graph, single output
- Position our work: heterogeneous graph + dual decoder as extension

---

## Section 3: Heterogeneous and Multi-Scale Graph Learning for Mechanics

> **Role:** Show that heterogeneous GNNs exist and multi-scale GNNs exist, but their combination for structural mechanics is novel. Cite the closest domain works.

**Skeleton paragraph:**

> Heterogeneous graph neural networks extend GNNs to systems with multiple node and edge types through type-specific parameterization. Schlichtkrull et al. \cite{schlichtkrull2018rgcn} introduced Relational Graph Convolutional Networks (R-GCNs) with relation-specific weight matrices, and Hu et al. \cite{hu2020hgt} proposed the Heterogeneous Graph Transformer (HGT) with type-dependent attention and message functions. Within structural engineering, Li et al. \cite{li2025universalhgcn} demonstrated that heterogeneous graph representations outperform homogeneous ones for nonlinear structural surrogate modeling, achieving high accuracy on automotive body frame analysis. Cai and Jelovica \cite{cai2024hgtpanels} showed similar benefits of heterogeneous over homogeneous modeling for stiffened panel stress prediction using HGT. Du et al. \cite{du2026mefgcnn} proposed a multi-edge feature GCN that simultaneously predicts nodal displacements and member forces on space truss structures, validating the dual-output paradigm. However, these works either target different structural domains (automotive frames, stiffened panels) or use homogeneous graph representations with edge-type features rather than fully heterogeneous node types. A separate line of work addresses long-range physical interactions through multi-scale graph architectures. Ying et al. \cite{ying2018diffpool} introduced differentiable hierarchical pooling for graph-level classification tasks, and Fortunato et al. \cite{fortunato2022multiscale} extended MeshGraphNets with coarse-fine hierarchical message passing. Jain et al. \cite{jain2024latticegraphnet} proposed a two-scale graph neural operator that first predicts reduced beam-scale dynamics and then maps to full-mesh predictions. Our macro-anchor module draws inspiration from these multi-scale approaches but differs in two key aspects: anchors are constructed from structural coordinates with stiffness-aware weighting rather than learned pooling, and a gated residual fusion mechanism controls the information flow from macro to micro representations. This domain-motivated design preserves physical interpretability while capturing long-range force transfer across the full bridge span.

**Citation placeholders:**
- `\cite{schlichtkrull2018rgcn}` — R-GCN
- `\cite{hu2020hgt}` — HGT
- `\cite{li2025universalhgcn}` — Universal HGNN surrogate
- `\cite{cai2024hgtpanels}` — HGT for stiffened panels
- `\cite{du2026mefgcnn}` — MeF-GCNN (dual output)
- `\cite{ying2018diffpool}` — DiffPool (hierarchical pooling)
- `\cite{fortunato2022multiscale}` — MultiScale MeshGraphNets
- `\cite{jain2024latticegraphnet}` — LatticeGraphNet

**Key narrative points:**
- Heterogeneous GNNs exist (R-GCN, HGT) but mostly for general graph tasks
- Structural HGNN works exist (Li 2025, Cai 2024) but target different domains
- Multi-scale GNNs exist (DiffPool, MultiScale MeshGraphNets) but not with structural anchors
- Our combination: stiffness-aware coordinate anchors + gated fusion is novel

---

## Section 4: Physics-Informed Learning and Uncertainty Quantification for Surrogate Models

> **Role:** Cover two contributions in one section (physics loss + conformal UQ). Show that both techniques exist separately, but their combined application to heterogeneous structural surrogate models is new.

**Skeleton paragraph:**

> Physics-informed learning incorporates domain knowledge into neural network training through loss function regularization or architecture constraints. Raissi et al. \cite{raissi2019pinns} introduced Physics-Informed Neural Networks (PINNs), encoding PDE residuals as soft constraints via automatic differentiation. This paradigm naturally extends to graph-structured systems: Song et al. \cite{song2023structgnn} proposed StructGNN-E, embedding bar equilibrium and constitutive equations directly into the GNN loss function to achieve accurate elastic structural analysis without labeled data. Wurth et al. \cite{wurth2024pimgn} combined PINN-style physics loss with MeshGraphNet architecture for time-dependent nonlinear PDEs on arbitrary meshes. Our physics-informed regularization is a lightweight extension of these ideas: we incorporate support boundary condition losses (penalizing non-zero displacement at constrained degrees of freedom) and rigid-link consistency constraints, both of which are directly computable from the available data schema without requiring full global stiffness matrix assembly. For uncertainty quantification, conformal prediction provides distribution-free, finite-sample valid prediction intervals for any pre-trained model. Vovk et al. \cite{vovk2022algorithmic} established the theoretical foundations, and Lei et al. \cite{lei2018distribution} formalized split conformal prediction for regression with marginal coverage guarantees. Angelopoulos and Bates \cite{angelopoulos2023gentle} provided a comprehensive modern treatment. Several recent works have adapted conformal prediction to graph-structured data: Zargarbashi and Bojchevski \cite{zargarbashi2024conformal} established exchangeability conditions for inductive graph node prediction, and Zhang et al. \cite{zhang2025rrgnn} proposed topology-aware Mondrian conformal prediction for region-conditional coverage. For structural engineering, Dadras Eslamlou et al. \cite{dadras2025hybrid} demonstrated CP at 90% confidence for GNN-based truss bridge damage identification. Building on these foundations, we apply component-wise split conformal prediction to obtain marginal prediction intervals for each output degree of freedom, distinguishing between per-DOF marginal coverage (~90% at α=0.10) and joint 6-DOF coverage, which is necessarily lower under marginal calibration.

**Citation placeholders:**
- `\cite{raissi2019pinns}` — PINNs
- `\cite{song2023structgnn}` — StructGNN-E
- `\cite{wurth2024pimgn}` — PI-MGNs
- `\cite{vovk2022algorithmic}` — Algorithmic Learning (CP book)
- `\cite{lei2018distribution}` — Distribution-free CP for regression
- `\cite{angelopoulos2023gentle}` — Gentle intro to CP
- `\cite{zargarbashi2024conformal}` — Conformal inductive GNN
- `\cite{zhang2025rrgnn}` — RR-GNN (topology-aware CP)
- `\cite{dadras2025hybrid}` — Hybrid CP + GNN for damage

**Key narrative points:**
- Physics loss is well-established (PINNs, StructGNN, PI-MGNs)
- Our contribution: lightweight BC + link loss tailored to available data
- Conformal prediction is established (Vovk, Lei, Angelopoulos)
- Graph CP is emerging (Zargarbashi, Zhang)
- Our contribution: component-wise CP for structural surrogate with clear marginal vs. joint distinction

---

## Citation Summary for Related Work Sections

| Section | Citations | Total |
|---------|-----------|:-----:|
| 1. Surrogate Modeling | `sacks1989design`, `raissi2019pinns`, `karniadakis2021piml`, `haghighat2021physicsinformed` | ~4 |
| 2. GNN for Physics | `battaglia2016interaction`, `sanchezgonzalez2020gns`, `pfaff2021meshgraphnets`, `fortunato2022multiscale` | ~4 |
| 3. Hetero + Multi-Scale | `schlichtkrull2018rgcn`, `hu2020hgt`, `li2025universalhgcn`, `cai2024hgtpanels`, `du2026mefgcnn`, `ying2018diffpool`, `jain2024latticegraphnet` | ~7 |
| 4. Physics + UQ | `raissi2019pinns` (shared), `song2023structgnn`, `wurth2024pimgn`, `vovk2022algorithmic`, `lei2018distribution`, `angelopoulos2023gentle`, `zargarbashi2024conformal`, `zhang2025rrgnn`, `dadras2025hybrid` | ~8 |
| **Total unique** | | **~18** |

Note: Some citations appear in multiple sections (`raissi2019pinns` in §1 and §4). Unique count reflects only the first occurrence.

---

## Writing Guidelines for ICTAI

1. **Keep paragraphs self-contained.** Each should motivate a gap, describe prior work, and state our position — enabling selective reading for space-constrained reviewers.

2. **Use active, critical description.** Not "X et al. proposed Y" but "X et al. proposed Y, which achieves Z but requires W, motivating our approach."

3. **Avoid claiming "first" unless certain.** Where strong prior work exists, write "to the best of our knowledge, no existing work combines..." or "while X and Y address individual aspects, their combination for steel truss FE surrogate modeling has not been explored."

4. **Cite baselines in methods/experiments too.** GCN, GAT, R-GCN, HGT should appear in Related Work AND in the Experiments section.

5. **Limit citations to 2–3 per claim.** The skeleton above may be dense; trim to most impactful references in the final version.

6. **Domain differentiation is key.** The Li et al. (2025) HGNN paper is the closest; clearly and respectfully differentiate: different structural domain (auto body vs. steel truss), different graph schema (their design vs. our beam/plate/mesh_node), plus our macro anchor + UQ contributions.

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Phase 2A Related Work Skeleton*

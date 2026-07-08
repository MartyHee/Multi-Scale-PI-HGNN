# Method & Literature Inventory

> Mapping of technical components to source code, reference papers, and GitHub/library sources.  
> Prepared for ICTAI 2026 paper writing — references marked with `TODO_VERIFY` need BibTeX confirmation.

---

## 1. Graph Construction for Structural FE Data

### Source Code

| Component | File(s) | Description |
|-----------|---------|-------------|
| Heterogeneous graph builder | `src/data/build_hetero_graph_dataset.py` | Reads CSV tables → constructs PyG `HeteroData` |
| Dataset loader | `src/data/hetero_graph_dataset.py` | `HeteroGraphDataset(processed_dir)` |
| Schema definition | `src/data/hetero_schema.py` | Node types, edge types, feature/target fields |
| Data transforms | `src/data/hetero_transforms.py` | Feature standardization, graph transforms |
| Split | `src/data/hetero_split.py` | `by_sample` / `by_loadcase` split strategies |

### Method Description

The structural FE data (steel truss girder bridge) is represented as a heterogeneous graph with:
- **3 node types:** `mesh_node` (1056, 15-dim features), `beam_element` (1646, 11-dim), `plate_element` (832, 6-dim)
- **5 edge types:** `belongs_to_beam`, `rev_belongs_to_beam`, `belongs_to_plate`, `rev_belongs_to_plate`, `structural_link`
- `structural_link` edges have 10-dim edge attributes (Kx, Ky, Kz, Krx, Kry, Krz, BetaAngle, DistanceRatio, ElasticLinkType, is_rigid)

The rigid_elastic_links.csv data is explicitly modeled as `mesh_node → structural_link → mesh_node` interaction edges — NOT as a separate node type.

### Reference Papers

| Reference | Role in Paper | Status |
|-----------|---------------|--------|
| **Heterogeneous graph representation** — the concept of modeling structured data as heterogeneous graphs with typed nodes and edges is well-established (see HGT, RGCN below) | Foundational concept | N/A |
| PyTorch Geometric `HeteroData` — implicit reference for implementation | Implementation framework | Will cite PyG |

### GitHub / Library Sources

| Source | Usage |
|--------|-------|
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) — `HeteroData`, `to_hetero()` | Core data structure |
| `torch_geometric.data.HeteroData` | Subclassed directly |

### Originality

This is an **engineering adaptation** of the standard heterogeneous graph formulation to the steel truss FE domain. The schema design (which entity types become which node types, how structural_link is modeled as an edge) is project-specific.

---

## 2. Baseline Models

### 2.1 MLP Baseline

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/mlp_baseline.py` |
| Class | `MLPBaseline` |
| Conv layers | None — per-edge MLP on concatenated features |
| Params | 96,274 |
| Paper reference | Standard MLP; no specific citation needed other than good ML practices |

### 2.2 Homogeneous GCN

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/homogeneous_gcn.py` |
| Class | `HomogeneousGCN` |
| Conv layer | `GCNConv` |
| Params | 76,050 (Stage 2-A) |
| Reference paper | **Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017** |
| PyG source | `torch_geometric.nn.GCNConv` |

### 2.3 Homogeneous GAT

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/homogeneous_gat.py` |
| Class | `HomogeneousGAT` |
| Conv layer | `GATConv` |
| Params | 76,818 |
| Reference paper | **Veličković et al., "Graph Attention Networks", ICLR 2018** |
| PyG source | `torch_geometric.nn.GATConv` |

### 2.4 RGCN / HeteroConv

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/hetero_rgcn.py` |
| Class | `HeteroRGCNBaseline` |
| Conv layer | `HeteroConv` + `SAGEConv` (relation-specific) |
| Params | 520,338 |
| Reference paper | **Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018** |
| PyG source | `torch_geometric.nn.HeteroConv`, `torch_geometric.nn.SAGEConv` |

### 2.5 HGT Baseline

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/hgt_baseline.py` |
| Class | `HGTBaseline` |
| Conv layer | Custom HGT-style typed attention |
| Params | 744,279 |
| Reference paper | **Hu et al., "Heterogeneous Graph Transformer", WWW 2020** |
| Notes | Implementation follows the HGT paper design: type-dependent QKV projections, relation-specific attention, adaptive message function |
| PyG source | `torch_geometric.nn.HGTConv` is a reference; our implementation uses custom typed attention |

### 2.6 Ours-Base (Edge-Attribute Aware)

| Property | Detail |
|----------|--------|
| File | `src/models/baselines/ours_base.py` |
| Classes | `OursBaseline`, `OursBaselineV2` |
| Key modules | `StructuralLinkConv` (edge-attr aware message passing), `PhysicsGate` |
| Params | 523,029 (v2) |
| Reference | HGT backbone + edge-attribute-conditioned modifications |
| Status | Completed; test metrics available |

### Key Paper References for Baselines Section

| Reference | Citation | Status |
|-----------|----------|--------|
| GCN | Kipf & Welling, ICLR 2017 | ✅ Confirmed |
| GAT | Veličković et al., ICLR 2018 | ✅ Confirmed |
| R-GCN | Schlichtkrull et al., ESWC 2018 | ✅ Confirmed |
| HGT | Hu et al., WWW 2020 | ✅ Confirmed |
| PyG docs | Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", 2019 | ✅ Confirmed |
| GraphSAGE | Hamilton et al., NeurIPS 2017 | ✅ Confirmed (used in RGCN conv) |

---

## 3. Multi-Scale Modeling (Macro Anchor)

### Source Code

| Component | File | Class/Function |
|-----------|------|----------------|
| Macro anchor pooling | `src/models/baselines/ms_hgt.py` | `MacroAnchorPool` |
| Macro GNN | `src/models/baselines/ms_hgt.py` | `MacroGNN` |
| Cross-scale fusion | `src/models/baselines/ms_hgt.py` | `CrossScaleFusion` |
| MS-HGT full model | `src/models/baselines/ms_hgt.py` | `MSHGTBaseline` |

### Method Description

**MacroAnchorPool** partitions the 1056 mesh nodes into 12 segments by x-coordinate percentile bins, then aggregates node representations within each segment (mean pooling) to produce 12 anchor representations. This is inspired by:

- **Graph pooling / clustering**: anchor nodes serve as learnable macro-level entities
- **GraphSAGE-style aggregation**: within-segment mean pooling
- **Stiffness-aware design**: x-coordinate-based segmentation reflects the physical intuition that structural behavior varies along the girder span

**MacroGNN**: A 2-layer SAGEConv on a bidirectional chain graph connecting adjacent anchors, allowing macro-level message passing across the span.

**CrossScaleFusion**: Gated residual fusion (or additive fusion for ablation) that merges macro anchor representations back into micro-node features:
- Gate network: linear layer → sigmoid activation
- Per-layer fusion: applies after each micro-GNN layer
- **Reference:** gated fusion is common in multi-scale architectures (e.g., U-Net skip connections, multi-scale GNNs)

### Reference Papers

| Reference | Role | Status |
|-----------|------|--------|
| **GraphSAGE** — Hamilton et al., NeurIPS 2017 | Macro GNN backbone (SAGEConv) | ✅ Confirmed |
| **Hierarchical graph pooling** — Ying et al., "Hierarchical Graph Representation Learning with Differentiable Pooling", NeurIPS 2018 | Related concept (differentiable pooling) | **TODO_VERIFY** — not directly used; conceptual reference |
| **Multi-scale GNN** — clusters / anchoring / coarse-graining is standard in graph learning literature | General concept | N/A — will cite representative works |

### Originality Assessment

| Aspect | Originality |
|--------|-------------|
| X-coordinate based anchor assignment | **Project-specific** — domain-motivated partitioning |
| Bidirectional chain macro graph | Standard design choice; no specific paper |
| Gated residual cross-scale fusion | **Combination is project-specific**; individual components (gating, residual fusion) are standard |
| Overall MS-HGT architecture | **Novel combination** of HGT backbone + macro-anchor + cross-scale fusion for FE surrogate modeling |

### Citation Strategy

- Cite GraphSAGE for the macro GNN backbone
- Cite hierarchical/multi-scale GNN literature as related work (not as direct baselines)
- The x-coordinate anchor assignment is a domain-specific design and does not need external citation

---

## 4. Physics-Informed Learning

### Source Code

| Component | File | Class/Function |
|-----------|------|----------------|
| Physics loss module | `src/losses/physics_loss.py` | `BCLoss`, `StructuralLinkConsistencyLoss`, `PhysicsRegularizer` |
| Training with physics loss | `src/trainers/baseline_trainer.py` | Incorporates `lambda_bc`, `lambda_link` |
| Loss scale diagnostics | `scripts/inspect_physics_loss_scale.py` | Pre-training loss scale analysis |

### Method Description

Two physics-inspired loss terms:

1. **Support Boundary Condition Loss** ($\mathcal{L}_{BC}$):  
   $`\mathcal{L}_{BC} = \frac{1}{N_{support}} \sum_{i \in support} \| \hat{y}_{i, constrained} \|_2^2`$  
   Penalizes non-zero predicted displacement at DOFs that are physically constrained to zero.  
   Uses the `Dx_fix...Rz_fix` flags in `mesh_node.x[:, 9:15]` to identify constrained DOFs.

2. **Structural Link Consistency Loss** ($\mathcal{L}_{link}$):  
   $`\mathcal{L}_{link} = \frac{1}{N_{links}} \sum_{(i,j) \in links} \| \hat{y}_i - \hat{y}_j \|_2^2`$  
   Penalizes displacement mismatch at the two endpoints of a rigid link, enforcing that rigidly connected nodes should have the same displacement.

### Reference Papers

| Reference | Role | Status |
|-----------|------|--------|
| **Physics-Informed Neural Networks (PINNs)** — Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", JCP 2019 | Foundational concept (physics in loss functions) | ✅ Confirmed |
| **MeshGraphNets** — Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021 | Related: graph-based physical simulators with learned physics | ✅ Confirmed |
| **Graph-based PDE learning** — general field of physics-constrained ML | Context | N/A |
| **Structural link consistency** — no specific paper; project-specific | Project-specific design | N/A |

### Originality Assessment

| Aspect | Originality |
|--------|-------------|
| Support BC loss ($\mathcal{L}_{BC}$) | **Engineering adaptation** — standard idea of penalizing BC violations, applied to structural FE surrogate |
| Structural link consistency ($\mathcal{L}_{link}$) | **Project-specific** — motivated by the rigid link physical model |
| Combined BC + Link regularization | **Project-specific combination** |
| Overall physics loss framework | **Novel for the structural FE surrogate domain** but borrows from established PINN concepts |

### Citation Strategy

- Cite Raissi et al. (PINNs) as the foundational physics-informed ML reference
- Cite Pfaff et al. (MeshGraphNets) as graph-based physical simulation reference
- Position the BC/link losses as lightweight, domain-appropriate physics constraints
- Do **not** claim novel physics loss formulation; claim novel application domain

---

## 5. Uncertainty Quantification

### Source Code

| Component | File | Function |
|-----------|------|----------|
| Conformal prediction | `scripts/compute_conformal.py` | `compute_conformal_quantile()`, `build_split()` |
| Coverage analysis | `scripts/analyze_conformal.py` | Coverage metrics, region diagnostics, plotting |
| Design doc | `docs/stage6_uq_design.md` | Full methodological specification |

### Method Description

**Component-wise split conformal prediction:**

1. **Calibration:** Given calibration set $`\{(X_i, y_i)\}_{i=1}^n`$, compute nonconformity scores $`R_i = |\hat{y}_i - y_i|`$ (absolute residual)
2. **Quantile:** $`\hat{q} = \text{Quantile}(R_1, ..., R_n; \frac{\lceil (n+1)(1-\alpha) \rceil}{n})`$
3. **Prediction interval:** $`C(X_{new}) = [\hat{y}_{new} - \hat{q}, \hat{y}_{new} + \hat{q}]`$

Properties:
- Distribution-free marginal coverage guarantee
- Finite-sample validity (not asymptotic)
- Each of the 18 output components is calibrated separately with its own nonconformity score
- Three calibration split strategies for robustness

### Reference Papers

| Reference | Role | Status |
|-----------|------|--------|
| **Split conformal prediction** — Vovk et al., "Algorithmic Learning in a Random World", 2005 | Foundational theory | ✅ Confirmed |
| **Tutorial** — Shafer & Vovk, "A Tutorial on Conformal Prediction", JMLR 2008 | Algorithm reference | ✅ Confirmed |
| **Modern review** — Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 2021 | Accessible reference for paper | ✅ Confirmed |
| **Conformal prediction for regression** — Lei et al., "Distribution-Free Predictive Inference For Regression", JASA 2018 | Regression-specific theory | **TODO_VERIFY** — check exact citation |
| **Split conformal algorithm** — Papadopoulos et al., "Inductive Conformal Prediction: Theory and Application to Neural Networks", 2008 | Split method origin | **TODO_VERIFY** |

### GitHub / Library Sources

| Source | Usage |
|--------|-------|
| No external conformal library used | Implementation is from-scratch based on paper algorithms |
| `numpy`, `scipy` | Numerical computation |

### Originality Assessment

| Aspect | Originality |
|--------|-------------|
| Component-wise split conformal | **Standard method** — no novelty claimed |
| Three-callibration-split robustness check | **Good experimental practice** — not method novelty |
| Region/high-response coverage diagnostics | **Application-specific evaluation** |
| Graph-level conformal diagnostic | **Diagnostic variant** — not a method contribution |
| Application to structural FE surrogate | **Novel application domain** |

### Citation Strategy

- Cite Vovk et al. (2005) and Angelopoulos & Bates (2021) as primary references
- Position as standard methodology applied to a new domain
- Do **not** claim methodological novelty in UQ

---

## 6. Engineering Surrogate Modeling Literature

### Candidate References for Related Work Section

The following are potential references for the "related work" section on surrogate modeling and scientific ML:

| # | Reference | Topic | Status |
|---|-----------|-------|--------|
| 1 | **Bishop**, "Pattern Recognition and Machine Learning", 2006 — Chapter on Gaussian Processes for regression | Classic reference for surrogate modeling | ✅ Confirmed |
| 2 | **Kriging / Gaussian process regression** — Sacks et al., "Design and Analysis of Computer Experiments", Statistical Science 1989 | Classic surrogate modeling | **TODO_VERIFY** exact citation |
| 3 | **Neural network surrogates** — Haghighat et al., "A deep learning-based surrogate model for structural analysis", 2020 | Deep surrogate for structural analysis | **TODO_VERIFY** exact citation |
| 4 | **Graph network simulators** — Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks", ICML 2020 | Graph-based physical simulation | ✅ Confirmed |
| 5 | **MeshGraphNets** — Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021 | Mesh-based graph simulation | ✅ Confirmed |
| 6 | **GNN for FEM surrogate** — Beltran-Pulido et al., "A physics-informed graph neural network for structural response prediction", 2022 | GNN + FEM surrogate | **TODO_VERIFY** exact citation |
| 7 | **Learnable physical simulators** — Battaglia et al., "Interaction Networks for Learning about Objects and Relations", NeurIPS 2016 | Early graph-based physical reasoning | ✅ Confirmed |

### Domain-Specific References (Structural Engineering)

| # | Reference | Topic | Status |
|---|-----------|-------|--------|
| 1 | Steel truss / FE analysis standard textbooks | Standard domain references | N/A — cite where appropriate |
| 2 | Existing surrogate models for truss structures | Prior work in same domain | **TODO_VERIFY** — literature search needed |

### Notes on Literature Search

- The related work section should cover: (a) scientific ML surrogates, (b) graph neural networks for physics, (c) GNNs for structural/materials problems, (d) UQ for surrogates
- A systematic literature search is **recommended** before finalizing the references
- The above lists are **candidate references** — some may be replaced with more relevant papers
- The project's main novelty claim does **not** rely on these references being exhaustive

---

## 7. Summary: Innovation Attribution

| Module | Innovation Level | Paper Can Claim |
|--------|:----------------:|-----------------|
| Heterogeneous graph construction | Adaptation | "We formulate steel truss FE data as a heterogeneous graph with 3 node types and 5 edge types, explicitly modeling structural links as interaction edges." |
| HGT backbone | Standard + adaptation | "We adapt the Heterogeneous Graph Transformer for structural FE node and edge prediction tasks." |
| Macro-anchor pooling | Novel combination | "We introduce a stiffness-aware macro-anchor pooling mechanism that partitions the structure along its span and aggregates local representations into macro-level features." |
| Cross-scale gated fusion | Novel combination | "A gated residual fusion mechanism selectively merges macro context into micro representations at each GNN layer, enabling long-range force transfer." |
| BC + Link physics loss | Adaptation | "We incorporate lightweight physics-informed regularization: support boundary condition penalty and rigid structural link consistency." |
| Split conformal UQ | Standard application | "We apply component-wise split conformal prediction to provide calibrated uncertainty intervals for the surrogate model's predictions." |
| **Overall framework** | **Novel integration** | "To the best of our knowledge, this is the first unified framework integrating heterogeneous graph construction, typed attention, multi-scale macro-anchor fusion, physics-informed regularization, and conformal UQ for structural FE surrogate modeling." |

---

## 8. To-Verify Checklist

| Item | Action | Priority |
|------|--------|:--------:|
| GCN citation details (Kipf & Welling, ICLR 2017) | Verify exact proceedings | Low |
| GAT citation details (Veličković et al., ICLR 2018) | Verify exact proceedings | Low |
| R-GCN citation (Schlichtkrull et al., ESWC 2018) | Verify exact proceedings | Low |
| HGT citation (Hu et al., WWW 2020) | Verify exact proceedings | Low |
| PyG citation (Fey & Lenssen, 2019) | Check arXiv update | Low |
| Hierarchical graph pooling papers | Identify best reference for multi-scale section | Medium |
| MeshGraphNets (Pfaff et al., ICLR 2021) | Verify proceedings | Low |
| PINNs (Raissi et al., JCP 2019) | Verify DOI | Low |
| Split conformal (Vovk et al., 2005) | Verify book citation format | Low |
| Angelopoulos & Bates (2021) | Check arXiv version | Low |
| Existing structural surrogate literature | Systematic search recommended | **High** |
| Sanchez-Gonzalez et al. (ICML 2020) | Verify proceedings | Low |
| Bhatt et al., "CNN-based surrogate for truss optimization", 2020 | **TODO_VERIFY** if exists | Medium |

---

*Document version: v1.0 / 2026-07-08 / ICTAI 2026 Preparation Phase 1*

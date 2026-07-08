# ICTAI 2026 — Literature Review Inventory

> Systematic literature search for MS-PI-HGT paper references.
> **Date:** 2026-07-08
> **Phase 2A:** 7-direction systematic search with multi-agent parallel retrieval

---

## 1. Search Summary

| Direction | Queries | Retrieved | Recommended for paper |
|-----------|:-------:|:---------:|:--------------------:|
| 1. Structural FE surrogate modeling | 5 | ~18 | 6 |
| 2. GNN for physical simulation | 5 | ~10 | 5 |
| 3. GNN for structural engineering / mechanics | 5 | ~12 | 6 |
| 4. Heterogeneous GNN (foundational methods) | 5 | ~8 | 6 |
| 5. Multi-scale / hierarchical GNN | 6 | ~14 | 3 |
| 6. Physics-informed learning | 5 | ~16 | 4 |
| 7. Conformal prediction / UQ | 7 | ~18 | 7 |
| **Total** | **38** | **~96** | **~37** |

**Final recommended citation set:** **20–25** entries for ICTAI 8-page paper.

---

## 2. Foundational Method References

| Citation Key | Title | Authors / Venue / Year | Topic | Role in Our Paper | Must Cite? | BibTeX | Notes |
|---|---|---|---|---|---|---|---|
| `kipf2017semi` | Semi-Supervised Classification with GCN | Kipf & Welling, ICLR 2017 | GNN foundation | GCN baseline (Stage 2-A) | **Yes** | ✅ | Standard GCN convolution |
| `velickovic2018graph` | Graph Attention Networks | Veličković et al., ICLR 2018 | GNN attention | GAT baseline (Stage 2-A) | **Yes** | ✅ | Self-attention on graphs |
| `hamilton2017inductive` | Inductive Representation Learning on Large Graphs | Hamilton et al., NeurIPS 2017 | Neighborhood sampling | GraphSAGE neighbor aggregation | **Yes** | ✅ | Inductive graph learning |
| `schlichtkrull2018rgcn` | Modeling Relational Data with GCNs | Schlichtkrull et al., ESWC 2018 | Relational GNN | RGCN baseline (Stage 2-B) | **Yes** | ✅ | Relation-specific weights |
| `hu2020hgt` | Heterogeneous Graph Transformer | Hu et al., WWW 2020 | Hetero attention | HGT baseline (Stage 2-B) | **Yes** | ✅ | Type-dependent QKV |
| `wang2019han` | Heterogeneous Graph Attention Network | Wang et al., WWW 2019 | Hetero attention | Optional hetero reference | Optional | ✅ | Meta-path based attention |
| `gilmer2017mpnn` | Neural Message Passing for Quantum Chemistry | Gilmer et al., ICML 2017 | MPNN framework | Message passing formalism | **Yes** | ✅ | Unifying MPNN framework |
| `fey2019pytorch` | Fast Graph Representation Learning with PyTorch Geometric | Fey & Lenssen, ICLR-W 2019 | Software | Implementation credit | Optional | ✅ | PyG framework |

---

## 3. Graph Network Simulators (Physical Simulation)

| Citation Key | Title | Authors / Venue / Year | Graph Type | Physical System | Why Relevant | Limitation relative to our task |
|---|---|---|---|---|---|---|
| `battaglia2016interaction` | Interaction Networks | Battaglia et al., NeurIPS 2016 | Object graph | n-body, rigid-body, non-rigid dynamics | Foundational physics GNN; object-relation paradigm | Homogeneous objects only, no structured FE mesh |
| `sanchezgonzalez2018gnp` | Graph Networks as Learnable Physics Engines | Sanchez-Gonzalez et al., ICML 2018 | Object/relation graph | 8 physical systems, robotics | GN-based physics engine with system identification | Homogeneous graph, no structural mechanics |
| `sanchezgonzalez2020gns` | Learning to Simulate Complex Physics with Graph Networks | Sanchez-Gonzalez et al., ICML 2020 | Particle graph | Fluids, rigid solids, deformables | Canonical GNS encoder-processor-decoder | Dynamic particle graph vs. static FE mesh |
| `pfaff2021meshgraphnets` | Learning Mesh-Based Simulation with Graph Networks | Pfaff et al., ICLR 2021 | Mesh graph | Cloth, structural mechanics, aerodynamics | Closest architectural template; mesh-based FE surrogate | Homogeneous mesh; no beam/plate typed elements, no dual decoder |
| `fortunato2022multiscale` | MultiScale MeshGraphNets | Fortunato et al., arXiv 2022 | Hierarchical mesh | Fluid dynamics, cloth | Coarse-fine hierarchical MP; inspiration for macro anchor graph | Not structural-specific, no physics constraints |
| `battaglia2018relational` | Relational inductive biases, deep learning, and graph networks | Battaglia et al., arXiv 2018 | Graph Networks (unified) | Various | Formal GN block framework; theoretical grounding | Position paper, not domain-specific |

---

## 4. Structural / Mechanics GNN (Domain-Specific)

| Citation Key | Title | Structural Task | Method | Simulated System | Similarity to Our Work | Difference from Our Work | Must Cite? |
|---|---|---|---|---|---|---|---|
| `song2023structgnn` | Elastic structural analysis based on GNN without labeled data | Bar/frame displacement + internal force | StructGNN-E with equilibrium + constitutive loss | 2D/3D frames and trusses | GNN + physics-informed loss for truss/frame structures; no labeled data needed | Homogeneous bar graph; no hetero types (beam/plate/mesh); no plate elements | **Yes** |
| `parisi2024mism` | Mechanics-informed GNN for structural analysis | Truss deformation prediction | MISM: topology-aware message passing | 2D/3D trusses | GNN for truss surrogate; mechanics-informed design | Homogeneous truss member graph; displacement only; no dual decoder | **Yes** |
| `li2025universalhgcn` | Universal surrogate modeling based on HGNN for nonlinear analysis | Nonlinear FE response prediction | Heterogeneous GNN + physics loss | Car body frame, roof structures | Hetero graph for structural surrogate; physics-constrained loss | Car body not steel truss; no macro anchor graph; no conformal UQ | **Yes** |
| `du2026mefgcnn` | MeF-GCNN: spatial graph conv. for space-truss displacement and axial-force prediction | Truss displacement + axial force | Multi-edge feature GCN | Space trusses | Dual output (displacement + force) like our work; multi-edge feature | Homogeneous graph; no beam/plate distinction; no macro anchor | **Yes** |
| `deshpande2024magneto` | MAgNET: a graph U-Net for mesh-based simulations | Hyperelasticity stress | Graph U-Net with MAg layers | Non-linear FEM (hyperelastic) | Hierarchical graph pooling/unpooling for mesh simulations | No typed message passing; not structural-specific | Should |
| `cai2024hgtpanels` | Heterogeneous graph representation of stiffened panels | Stiffened panel stress | HGT-based heterogeneous GNN | Ship stiffened panels | Supports heterogeneous > homogeneous for structural problems | Panels not truss bridges; different physics | Should |
| `li2025structuregraph` | StructureGraph: universal performance evaluation via HGNN | Structural performance evaluation | HGNN with automatic architecture search | Various structures | Extends hetero graph paradigm to structural evaluation | Performance classification vs. response prediction | Optional |
| `he2026mmpn` | Mechanics-inspired message passing for bending structures | Bending structure response | Stiffness-embedded message passing | Beam bending | Encodes stiffness in MP; closely related to physics-typed message | Bending structures only; no plate elements | Should |
| `peng2026eemgnn` | EEM-GNN: physics-aware edge-centric GNN for reticulated shells | Shell displacement + force | Edge-centric GNN isomorphic to direct stiffness | Reticulated shells | Edge-centric design aligned with FE direct stiffness method | Shell-specific; no conformal UQ | Should |

---

## 5. Multi-Scale / Hierarchical Graph References

| Citation Key | Title | Graph Pooling Method | Why Relevant | Limitation vs. Our Approach |
|---|---|---|---|---|
| `ying2018diffpool` | Hierarchical Graph Representation Learning with Differentiable Pooling | Soft cluster assignment pooling | Foundation for hierarchical graph representation; conceptual precursor to macro anchor | Learned pooling vs. physics-motivated coordinate anchor; no cross-scale fusion back to nodes |
| `gao2019graphunets` | Graph U-Nets | gPool/gUnpool top-k selection | Pool/unpool encoder-decoder; parallels our cross-scale gated fusion | Node-drop pooling loses structural continuity; no physical motivation |
| `lee2019sagpool` | Self-Attention Graph Pooling | Topological self-attention | Topology-aware lternative to gPool; attention-based node scoring | Also node-drop; not design for meshing structures |
| `jain2024latticegraphnet` | LatticeGraphNet: Two-scale GNO for lattice structures | Coarse beam → fine tetrahedral mesh | Two-scale architecture parallel to macro-micro; lattice = truss | Different structural domain; no typed message passing |

---

## 6. Physics-Informed Learning References

| Citation Key | Title | UQ Method / Approach | How Used in Our Paper | Claim Supported |
|---|---|---|---|---|
| `raissi2019pinns` | Physics-informed neural networks | PDE residual via AD | Foundation for physics-constrained loss in Stage 5 | Physics loss can regularize NN training |
| `karniadakis2021piml` | Physics-informed machine learning | Survey of PI methods | High-level positioning; taxonomy of PI strategies | Physics integration is active research area |
| `wurth2024pimgn` | Physics-informed MeshGraphNets (PI-MGNs) | PDE residual on mesh GNN | Direct template for combining physics loss with mesh GNN architecture | Physics + mesh GNN is feasible |
| `song2023structgnn` | Elastic structural analysis based on GNN without labeled data | Equilibrium + constitutive loss | Direct domain competitor; equilibrium loss template for Stage 5 | Bar/frame physics loss works in structural GNN |
| `chen2024piers` | PIERS: physics-informed edge recurrent simulator | PDE residual + edge update | Architecture reference for physics-informed MP on meshes | Physics-informed edge updates improve long-term prediction |
| `peng2022pigl` | Physics-Informed Graph Learning | Survey of PIGL methods | Literature positioning; conceptual framework | Physics + graph learning has growing literature |

---

## 7. Conformal Prediction / UQ References

| Citation Key | Title | UQ Method | How Used in Our Paper | Claim Supported |
|---|---|---|---|---|
| `vovk2022algorithmic` | Algorithmic Learning in a Random World | Full/inductive conformal prediction | Theoretical foundation for all CP methods | Exchangeability suffices for finite-sample coverage |
| `shafer2008tutorial` | A Tutorial on Conformal Prediction | Online conformal prediction | Standard reference for CP basics | CP produces valid prediction sets |
| `angelopoulos2023gentle` | A Gentle Introduction to Conformal Prediction | Split CP, CQR, weighted CP | Main method reference for our split CP implementation | Distribution-free marginal coverage guarantee |
| `lei2018distribution` | Distribution-Free Predictive Inference for Regression | Split CP, jackknife, cross-CP | Regression-specific CP theory | Split CP yields valid regression intervals |
| `romano2019cqr` | Conformalized Quantile Regression | CQR | Reference for adaptive-width intervals | Adaptive intervals for heteroscedastic data |
| `barber2021jackknife` | Predictive Inference with the Jackknife+ | Jackknife+ / CV+ | Alternative to split CP with better data efficiency | 1-2alpha coverage guarantee |
| `papadopoulos2002inductive` | Inductive Confidence Machines for Regression | Inductive (split) CP | Historical foundation for split CP approach | Split CP is computationally efficient |
| `zargarbashi2024conformal` | Conformal Inductive Graph Neural Networks | NodeEx/EdgeEx CP for graphs | Graph-specific exchangeability theory | CP on graph data requires exchangeability handling |
| `zhang2025rrgnn` | Residual Reweighted CP for Graph Neural Networks | Graph-Structured Mondrian CP | Region-aware coverage for structural zones | Topology-aware CP reduces interval width |
| `dadras2025hybrid` | Hybrid Data-Physics with Conformal GNN for Damage ID | CP at 90% confidence | Most directly relevant structural CP application | CP + GNN works on truss bridges |
| `gopakumar2025cppre` | Calibrated Physics-Informed UQ | CP with PDE residual scores | Physics-informed nonconformity scores | Residuals as informative nonconformity measure |
| `gopakumar2024cpsurrogate` | Uncertainty Quantification of Surrogate Models using CP | Model-agnostic CP | CP for surrogate models directly | CP works for PDE surrogate prediction |

---

## 8. Final Recommended Reference Set (20–25 for ICTAI)

### Core Method References (Must Cite — 10)

| # | Citation Key | Paper | Section |
|---|--------------|-------|---------|
| 1 | `kipf2017semi` | GCN | Methods (baseline) |
| 2 | `velickovic2018graph` | GAT | Methods (baseline) |
| 3 | `schlichtkrull2018rgcn` | R-GCN | Methods (baseline) |
| 4 | `hu2020hgt` | HGT | Methods (baseline + backbone) |
| 5 | `gilmer2017mpnn` | MPNN | Methods (message passing formalism) |
| 6 | `battaglia2016interaction` | Interaction Networks | Related Work |
| 7 | `sanchezgonzalez2020gns` | GNS | Related Work |
| 8 | `pfaff2021meshgraphnets` | MeshGraphNets | Related Work / Methods |
| 9 | `raissi2019pinns` | PINNs | Related Work / Methods (physics loss) |
| 10 | `angelopoulos2023gentle` | Gentle Intro to CP | Methods (conformal UQ) |

### Related Work References (Should Cite — 8)

| # | Citation Key | Paper | Section |
|---|--------------|-------|---------|
| 11 | `song2023structgnn` | StructGNN-E | Related Work (structural GNN competitor) |
| 12 | `parisi2024mism` | MISM | Related Work (truss GNN) |
| 13 | `li2025universalhgcn` | Universal HGNN surrogate | Related Work (closest hetero approach) |
| 14 | `du2026mefgcnn` | MeF-GCNN | Related Work (dual output + truss) |
| 15 | `ying2018diffpool` | DiffPool | Related Work (multi-scale context) |
| 16 | `vovk2022algorithmic` | Algorithmic Learning | Methods (CP foundation) |
| 17 | `lei2018distribution` | Distribution-Free Prediction | Methods (split CP theory) |
| 18 | `karniadakis2021piml` | Physics-informed ML | Related Work (positioning) |

### Optional / Supplementary (4–6)

| # | Citation Key | Paper | Section |
|---|--------------|-------|---------|
| 19 | `hamilton2017inductive` | GraphSAGE | Methods (context) |
| 20 | `wurth2024pimgn` | PI-MGNs | Related Work (physics + GNN) |
| 21 | `fortunato2022multiscale` | MultiScale MeshGraphNets | Related Work (multi-scale) |
| 22 | `zargarbashi2024conformal` | Conformal Inductive GNN | Related Work (graph CP) |
| 23 | `dadras2025hybrid` | Hybrid CP + GNN for damage | Related Work (struct. eng. CP) |
| 24 | `battaglia2018relational` | Relational inductive biases | Related Work (GN theory) |
| 25 | `fey2019pytorch` | PyTorch Geometric | Implementation |
| 26 | `hamilton2017inductive` | GraphSAGE | Methods (context) |

---

## 9. TODO_VERIFY References

| Citation Key | Issue | Priority | Resolution |
|---|---|---|---|
| `li2025universalhgcn` | Authors "Li, Wang, Hou" — may need verification of exact author order and volume/pages | High | Check CMAME DOI S0045782525000659 |
| `du2026mefgcnn` | Authors from SSRN pre-print; verify published version in J. Building Engineering | Medium | Check final journal version |
| `cai2024hgtpanels` | Year may be 2024 or 2025; exact volume/pages not confirmed | Medium | Check Computers & Structures |
| `he2026mmpn` | Pre-print status; author list incomplete | Low | Check published version |
| `peng2026eemgnn` | Pre-print status; author list incomplete | Low | Check Engineering Structures |
| `gopakumar2024cpsurrogate` | Citation has "others" placeholder; verify exact author list | Low | Check arXiv:2408.09881 |
| `meshgnn_2026` (arXiv:2606.08287) | Exact authors not verified | Low | Check arXiv listing |

---

## 10. Novelty Assessment

After systematic search, **no existing work combines all three elements** of our approach:

1. **Heterogeneous physical graph** with mesh_node, beam_element, plate_element, and structural_link types — closest is Li et al. (2025) HGNN but on automotive body frames, not steel truss girders with this specific element classification
2. **Macro-anchor multi-scale fusion** for long-range force transfer on bridge-span structures — closest is Fortunato et al. (2022) MultiScale MeshGraphNets but without stiffness-aware structural anchoring
3. **Conformal UQ on structural graph surrogate predictions** with component-wise marginal coverage — closest is Dadras et al. (2025) but for damage identification, not displacement + force prediction

**Key differentiation claims:**
- Dual displacement + internal force decoder on heterogeneous structural graph
- Coordinate-based stiffness-aware macro anchors for long-range structural mechanics
- Physics-informed regularization (BC constraint + link consistency) designed for available data capabilities
- Component-wise split conformal prediction for structural surrogate model outputs

**Risk if reviewer asks about:**
- Song et al. (2023) StructGNN-E: Also structural GNN + physics loss, but homogeneous bar graph, no dual decoder, no UQ
- Li et al. (2025) HGNN: Also hetero graph + physics loss, but different domain (auto body), no macro anchor, no UQ
- Du et al. (2026) MeF-GCNN: Also dual output (disp+force), but homogeneous graph, no physics loss, no UQ, no multi-scale

**Response strategy:** Acknowledge each as closely related but clearly differentiate along the hetero schema → multi-scale → dual decoder → physics loss → UQ innovation chain.

---

## 11. Claim Adjustment Assessment

| Existing Claim | Adjust? | Reasoning |
|----------------|:-------:|-----------|
| "Novel heterogeneous graph schema for steel truss FE" | **No change** | No existing work uses beam/plate/mesh_node + structural_link for steel truss |
| "First to combine macro anchor with structural GNN" | **Minor soften** | Fortunato et al. (2022) did multi-scale MeshGraphNets, but not stiffness-aware or for truss structures |
| "Novel physics-informed regularization" | **Acknowledge prior** | Song et al. (2023) also uses physics loss; our contribution is BC + link loss specific to our data capabilities |
| "First UQ for structural GNN surrogate" | **Soften** | Dadras et al. (2025) did CP for structural damage ID; but not for joint disp+force prediction |
| "Overall SOTA" | **Remove** | Not universally verifiable; focus on specific axes (Dy R², region error, coverage) |

**Overall assessment:** Claims are defensible if properly scoped to:
- Steel truss girder domain
- Heterogeneous physical graph with 3 node + 5 edge types
- Dual displacement + force decoder
- Macro anchor with gated cross-scale fusion
- Component-wise conformal UQ

No fundamental novelty threat from any single prior work.

---

*Document version: v2.0 / 2026-07-08 / ICTAI 2026 Phase 2A Systematic Literature Search*

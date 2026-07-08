# MS-PI-HGT: Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Structural Finite Element Surrogate Modeling

> **Target:** ICTAI 2026 (8-page IEEE conference paper)
> **Draft version:** v1.0 / 2026-07-08 / Phase 3 Initial Draft
> **Estimated length:** ~9.5 pages (slightly over target; compression notes in each section)

---

## Abstract

> **Target: 180–220 words. Current: ~200 words.**

Finite element analysis (FEA) is the standard tool for structural engineering design, but its computational cost makes iterative tasks — design space exploration, uncertainty quantification, and optimization — prohibitively expensive. Data-driven surrogate models offer an alternative, yet standard MLP and homogeneous graph neural networks fail to fully exploit the heterogeneous structure of FEA data, which contains physically distinct entities (nodes, beam elements, plate elements, and structural links) with different feature types and relational dependencies.

We propose MS-PI-HGT, a multi-scale physics-informed heterogeneous graph transformer for structural FEA surrogate modeling. The framework constructs a heterogeneous graph from FEA data, applies typed attention for relation-specific message passing, and introduces a macro-anchor pooling mechanism with cross-scale gated fusion to capture long-range force transfer across bridge-span structures. Physics-informed regularization terms — support boundary condition loss and structural link consistency loss — improve physical consistency without degrading predictive accuracy. Finally, component-wise split conformal prediction provides calibrated prediction intervals for all output degrees of freedom.

Evaluated on 35,000 steel truss girder bridge FEA instances spanning 70 design samples, MS-PI-HGT achieves Displacement R² = 0.995, Force R² = 0.993, and Relative MAE = 0.052, substantially outperforming MLP (0.855/0.982/0.088) and homogeneous GNN baselines. Uncertainty intervals achieve near-nominal marginal coverage (89.74% / 89.97% at 90% nominal) across all 18 output components. The framework provides a practical AI tool for rapid structural evaluation with built-in reliability assessment.

---

## 1. Introduction

> **Target: 1.0–1.2 pages. Current: ~1.0 page.**
> **Writing note:** Compressable by tightening the problem chain.

Finite element (FE) simulation is the established standard for structural design and evaluation in civil and mechanical engineering. From routine design verification to advanced safety assessment, FEA provides high-fidelity predictions of structural behavior under arbitrary loading conditions. However, each FE simulation is computationally expensive: a single nonlinear steel truss bridge analysis can take hours on specialized hardware, and tasks requiring many evaluations — design space exploration, parametric studies, uncertainty quantification, or optimization — quickly become intractable.

Surrogate modeling offers a path to computational efficiency by replacing expensive FE simulations with fast data-driven approximations. Classical approaches such as Gaussian process regression \cite{sacks1989design} achieve good accuracy on low-dimensional problems but scale poorly with high-dimensional input spaces. Deep neural network surrogates \cite{raissi2019pinns, karniadakis2021piml} have demonstrated orders-of-magnitude speedups across diverse physics domains. However, most neural surrogates for structural systems rely on either fully-connected architectures \cite{haghighat2021physicsinformed} that discard topological information, or convolutional networks that require regular grid representations poorly suited to unstructured FE meshes.

The core challenge is that structural FE data is inherently heterogeneous. A typical steel truss girder bridge model contains several physically distinct entity types: mesh nodes carrying displacement degrees of freedom, beam elements transmitting axial, shear, and bending forces, plate elements distributing in-plane and out-of-plane loads, and structural links (rigid connections) coupling node displacements. These entities have different feature spaces, different physical roles, and complex relational dependencies. Standard MLP models treat all inputs as a flat feature vector, discarding the graph structure entirely. Homogeneous graph neural networks (GCN \cite{kipf2017semi}, GAT \cite{velickovic2018graph}) treat all nodes as the same type, conflating physically distinct entities. Both approaches lead to suboptimal predictive accuracy, particularly for challenging displacement components and high-response regions.

Recent work in heterogeneous graph neural networks (RGCN \cite{schlichtkrull2018rgcn}, HGT \cite{hu2020hgt}) has demonstrated that type-specific parameterization significantly improves performance on heterogeneous graph data. Within structural engineering, Li et al. \cite{li2025universalhgcn} showed that heterogeneous graph representations outperform homogeneous ones for nonlinear structural surrogate modeling, and Du et al. \cite{du2026mefgcnn} proposed multi-edge feature GCN for truss displacement and force prediction. However, existing approaches face three limitations. First, local message passing alone is insufficient for long-range force transfer in large-span structures — a bridge's midspan deflection depends on interactions across the full span. Second, pure data-driven predictions may violate basic physical constraints, such as zero displacement at support boundaries. Third, most surrogate models provide point predictions without uncertainty quantification, limiting their utility for engineering decision-making.

To address these limitations, we propose **MS-PI-HGT**, a multi-scale physics-informed heterogeneous graph transformer. Our contributions are:

1. **Heterogeneous structural graph representation** — We construct a 3-node-type, 5-edge-type heterogeneous graph from steel truss FE data, encoding mesh nodes, beam elements, plate elements, and structural link connections as distinct physical entities with type-specific features and relational patterns.

2. **Multi-scale macro-anchor fusion** — We introduce a coordinate-based macro-anchor pooling mechanism that aggregates micro-node representations into structural segments, propagates information across a macro chain graph, and selectively fuses global contextual information back to micro nodes through gated residual connections. This captures long-range force transfer that local message passing cannot.

3. **Lightweight physics-informed regularization** — We incorporate support boundary condition loss and structural link consistency loss as soft constraints. These improve physical consistency (reducing constrained DOF MAE by 39%) without degrading predictive accuracy.

4. **Component-wise conformal uncertainty quantification** — We apply split conformal prediction to obtain distribution-free, finite-sample valid prediction intervals for each of the 18 output components, with clear distinction between per-DOF marginal coverage and joint coverage.

5. **Systematic empirical evaluation** — We evaluate seven model variants on 35,000 FE instances spanning 70 design samples, demonstrating progressive improvement from MLP (Disp R² = 0.855) through typed message passing (RGCN 0.937, HGT 0.977) to our full framework (MS-PI-HGT 0.995), with near-nominal UQ coverage across all output components.

---

## 2. Related Work

> **Target: 1.0–1.2 pages. Current: ~1.1 pages.**
> **Writing note:** Based on `docs/ictai_related_work_skeleton.md` with citation keys verified in Phase 3-precheck.

### 2.1 Surrogate Modeling for Structural Simulation

Finite element analysis is the standard tool for structural design and assessment, but its computational cost becomes prohibitive for tasks requiring many evaluations, including design optimization \cite{sacks1989design}, parametric studies, and uncertainty quantification. Surrogate models — also known as metamodels or emulators — replace expensive FE simulations with fast data-driven approximations. Classical approaches include Gaussian process regression (kriging) \cite{sacks1989design} and polynomial chaos expansion, which achieve good accuracy on low-dimensional problems but scale poorly with high-dimensional input spaces. More recently, deep neural network surrogates have demonstrated orders-of-magnitude speedups across diverse physics domains \cite{raissi2019pinns, karniadakis2021piml}. However, most neural surrogates for structural systems rely on either fully-connected architectures \cite{haghighat2021physicsinformed} that discard topological information, or convolutional networks that require regular grid representations poorly suited to unstructured FE meshes. These limitations motivate the use of graph-based representations that natively encode the connectivity and topology of structural FE systems.

### 2.2 Graph Neural Networks for Physical Systems

Graph neural networks (GNNs) have emerged as a powerful framework for learning physical simulations directly on unstructured representations. Battaglia et al. \cite{battaglia2016interaction} introduced Interaction Networks for object- and relation-centric physical reasoning. Sanchez-Gonzalez et al. \cite{sanchezgonzalez2020gns} proposed the Graph Network-based Simulator (GNS), an encoder-processor-decoder architecture that generalizes across fluids, rigid solids, and deformable materials. Pfaff et al. \cite{pfaff2021meshgraphnets} extended this paradigm to mesh-based simulation with MeshGraphNets, achieving orders-of-magnitude speedup over classical solvers. The key architectural insight — encode physical state into a graph, apply K steps of processor message passing, decode to target quantities — underpins most current mesh-based surrogate models \cite{fortunato2022multiscale}. However, these approaches operate on homogeneous graphs where all nodes represent the same entity type. Structural FE problems involve multiple physically distinct entity types: mesh nodes that carry displacements, beam and plate elements that carry internal forces, and constraint relationships such as rigid links. A homogeneous graph conflates these distinct physical roles, limiting predictive accuracy. Furthermore, existing simulators are typically designed for single-output tasks, whereas structural engineering requires simultaneous prediction of both nodal displacements and element-end internal forces.

### 2.3 Heterogeneous and Multi-Scale Graph Learning

Heterogeneous graph neural networks extend GNNs to systems with multiple node and edge types through type-specific parameterization. Schlichtkrull et al. \cite{schlichtkrull2018rgcn} introduced Relational Graph Convolutional Networks (R-GCNs) with relation-specific weight matrices, and Hu et al. \cite{hu2020hgt} proposed the Heterogeneous Graph Transformer (HGT) with type-dependent attention. Within structural engineering, Li et al. \cite{li2025universalhgcn} demonstrated that heterogeneous graph representations outperform homogeneous ones for nonlinear structural surrogate modeling on automotive body frames. Cai and Jelovica \cite{cai2024hgtpanels} showed similar benefits of heterogeneous over homogeneous modeling for stiffened panel stress prediction. Du et al. \cite{du2026mefgcnn} proposed a multi-edge feature GCN that simultaneously predicts nodal displacements and member forces on space truss structures, validating the dual-output paradigm. A separate line of work addresses long-range physical interactions through multi-scale graph architectures. Ying et al. \cite{ying2018diffpool} introduced differentiable hierarchical pooling, and Fortunato et al. \cite{fortunato2022multiscale} extended MeshGraphNets with coarse-fine hierarchical message passing. Jain et al. \cite{jain2024latticegraphnet} proposed a two-scale graph neural operator for lattice structures. Our macro-anchor module draws inspiration from these approaches but differs in two key aspects: anchors are constructed from structural coordinates rather than learned pooling, and a gated residual fusion mechanism controls the information flow from macro to micro representations.

### 2.4 Physics-Informed Learning and Uncertainty Quantification

Physics-informed learning incorporates domain knowledge into neural network training through loss regularization or architecture constraints. Raissi et al. \cite{raissi2019pinns} introduced Physics-Informed Neural Networks (PINNs), encoding PDE residuals via automatic differentiation. This paradigm extends to graph-structured systems: Song et al. \cite{song2023structgnn} proposed StructGNN-E, embedding bar equilibrium and constitutive equations into the GNN loss function. Wurth et al. \cite{wurth2024pimgn} combined PINN-style physics loss with MeshGraphNet architecture for nonlinear PDEs on arbitrary meshes. For uncertainty quantification, conformal prediction provides distribution-free, finite-sample valid prediction intervals \cite{vovk2022algorithmic, lei2018distribution, angelopoulos2023gentle}. Recent works have adapted conformal prediction to graph-structured data \cite{zargarbashi2024conformal, zhang2025rrgnn}, and Dadras Eslamlou et al. \cite{dadras2025hybrid} demonstrated CP for GNN-based truss bridge damage identification. Building on these foundations, we apply component-wise split conformal prediction to obtain prediction intervals for each output degree of freedom, clearly distinguishing between per-DOF marginal coverage and joint coverage.

---

## 3. Methodology

> **Target: 2.3–2.7 pages. Current: ~2.5 pages.**
> **Writing note:** This is the core of the paper. Each subsection can stand alone but flows as a coherent pipeline.

### 3.1 Problem Definition

We consider the problem of surrogate modeling for steel truss girder FE systems. Given a bridge design configuration $d$ and load case $l$, we aim to predict two sets of structural responses:

1. **Node displacements:** $\mathbf{Y}_{\text{disp}} \in \mathbb{R}^{N_m \times 6}$, where $N_m$ is the number of mesh nodes. Each node has 6 degrees of freedom: translations $[D_x, D_y, D_z]$ and rotations $[R_x, R_y, R_z]$.

2. **Beam-end internal forces:** $\mathbf{Y}_{\text{force}} \in \mathbb{R}^{N_b \times 12}$, where $N_b$ is the number of beam elements. Each beam element has 12 force components: forces and moments at the I-node and J-node $[F_x^I, F_y^I, F_z^I, M_x^I, M_y^I, M_z^I, F_x^J, F_y^J, F_z^J, M_x^J, M_y^J, M_z^J]$.

We formulate the surrogate modeling task as a supervised learning problem on a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}_v, \mathcal{R}_e)$ where $\mathcal{V}$ is the set of nodes with types $\mathcal{R}_v$, $\mathcal{E}$ is the set of edges with types $\mathcal{R}_e$, and each node/edge carries type-specific features. The learned function $f_\theta: \mathcal{G} \to (\hat{\mathbf{Y}}_{\text{disp}}, \hat{\mathbf{Y}}_{\text{force}})$ maps the input graph to predicted responses.

### 3.2 Heterogeneous Structural Graph Construction

We construct a heterogeneous graph from raw FE data, defining three physical node types and five edge types.

**Node types.** Each FE entity maps to a graph node with type-specific features:

- **Mesh node** ($\mathcal{V}_m$, $N_m = 1,\!056$ per graph): Represent spatial locations, external loads, and boundary conditions. Node features $\mathbf{x}_m \in \mathbb{R}^{15}$ include 3D coordinates $(x, y, z)$, applied nodal loads $(F_x, F_y, F_z, M_x, M_y, M_z)$, and DOF constraint flags indicating which displacement components are prescribed (typically zero at support boundaries).

- **Beam element** ($\mathcal{V}_b$, $N_b = 1,\!646$ per graph): Represent structural members with section and material properties. Features $\mathbf{x}_b \in \mathbb{R}^{11}$ include cross-sectional area $A$, moments of inertia $(I_y, I_z)$, torsional constant $J$, elastic modulus $E$, shear modulus $G$, element length $L$, and section orientation information.

- **Plate element** ($\mathcal{V}_p$, $N_p = 832$ per graph): Represent deck or web plates with thickness and material properties. Features $\mathbf{x}_p \in \mathbb{R}^6$ include plate thickness $t$, elastic modulus $E$, Poisson's ratio $\nu$, and element geometry descriptors.

**Edge types.** Five directed edge types capture the relational structure of the FE system:

1. **`belongs_to_beam`** ($\mathcal{E}_{m\to b}$): Connects a mesh node to a beam element that uses it as an endpoint. Each beam element connects to two mesh nodes (I-node and J-node), producing $2 \times N_b$ edges of this type.

2. **`rev_belongs_to_beam`** ($\mathcal{E}_{b\to m}$): Reverse direction, from beam element to mesh node.

3. **`belongs_to_plate`** ($\mathcal{E}_{m\to p}$): Connects a mesh node to a plate element vertex. Each plate element connects to 3 mesh nodes, producing $3 \times N_p$ edges.

4. **`rev_belongs_to_plate`** ($\mathcal{E}_{p\to m}$): Reverse direction, from plate element to mesh node.

5. **`structural_link`** ($\mathcal{E}_{m\to m}$): Represents rigid (or elastic) connections between mesh nodes. Each graph contains 132 directed edges (66 physical rigid connections) with 10-dimensional edge attributes $\mathbf{e}_{sl} \in \mathbb{R}^{10}$ including stiffness coefficients $(K_x, K_y, K_z, K_{rx}, K_{ry}, K_{rz})$, beta angle, and distance ratio. The structural_link count is constant across all 35,000 graph instances.

> **Code verification:** `hetero_schema.py` confirms `plate_element.feature_dim = 6`, `beam_element.feature_dim = 11`, `mesh_node.feature_dim = 15`. All feature dimensions verified against source code. ✅

The graph is supervised with two target tensors: $\mathbf{y}_{\text{disp}} \in \mathbb{R}^{N_m \times 6}$ for node displacements and $\mathbf{y}_{\text{force}} \in \mathbb{R}^{N_b \times 12}$ for beam-end forces. All graphs share the same topology (fixed node/edge indices) across the 70 design samples; only node/edge features and target values vary with design parameters and loading conditions.

[Figure 2 about here: Heterogeneous graph schema showing 3 node types, 5 edge types, and feature dimensions]

### 3.3 Typed Graph Transformer Backbone

Our backbone adopts the Heterogeneous Graph Transformer (HGT) \cite{hu2020hgt} as the core message passing engine, with type-dependent attention and message functions.

For a target node $t$ of type $\tau(t)$ receiving messages from source node $s$ of type $\tau(s)$ through edge $e$ of type $\phi(e)$, the HGT attention mechanism computes:

$$\text{Attention}(s, e, t) = \text{Softmax}_{\forall s \in N(t)}\left( \text{A}_{\text{head}}^{i}\left( \text{Linear}_{\tau(s)}^{i}(\mathbf{h}_s) \right) \cdot \text{Linear}_{\phi(e)}^{i}\left( \text{Linear}_{\tau(t)}^{i}(\mathbf{h}_t) \right)^\top \cdot \frac{\mu}{\sqrt{d}} \right)$$

where $\mathbf{h}_s$ and $\mathbf{h}_t$ are hidden representations of source and target nodes, $\text{Linear}_{\tau(\cdot)}^{i}(\cdot)$ are type-specific linear projections for attention head $i$, and $\mu$ is a type-specific scaling factor. The message from $s$ to $t$ is computed as:

$$\text{Message}(s, e, t) = \text{Linear}_{\tau(s)}^{M}\left( \text{Linear}_{\phi(e)}^{M}(\mathbf{h}_s) \right)$$

where $\text{Linear}_{\tau(s)}^{M}$ is a type-specific message projection and $\text{Linear}_{\phi(e)}^{M}$ is an edge-type-specific transformation. The updated node representation is the multi-head attention aggregation:

$$\tilde{\mathbf{h}}_t = \bigoplus_{i=1}^{H} \left( \sum_{s \in N(t)} \text{Attention}^{i}(s, e, t) \cdot \text{Message}^{i}(s, e, t) \right)$$

where $\bigoplus$ denotes concatenation across $H$ attention heads, followed by a type-specific linear projection and residual connection:

$$\mathbf{h}_t^{(l+1)} = \text{Linear}_{\tau(t)}\left( \text{LayerNorm}\left( \tilde{\mathbf{h}}_t \right) \right) + \mathbf{h}_t^{(l)}$$

We stack $L = 3$ HGTConv layers with hidden dimension $d = 128$ and $H = 4$ attention heads. Each layer applies type-dependent projections, enabling the model to learn distinct relational patterns for each physical edge type.

**Type-specific encoders.** Before message passing, each node type is encoded from its raw features via a type-specific linear projection:

$$\mathbf{h}_v^{(0)} = \text{Linear}_{\tau(v)}\left( \mathbf{x}_v \right) \quad \forall v \in \mathcal{V}$$

where $\mathbf{x}_v$ is the raw feature vector of node $v$ and $\tau(v) \in \{\text{mesh\_node}, \text{beam\_element}, \text{plate\_element}\}$.

**Dual decoder.** After $L$ layers of heterogeneous message passing, two separate MLP decoders produce the final predictions:

$$\hat{\mathbf{y}}_{\text{disp}} = \text{MLP}_{\text{disp}}\left( \mathbf{h}_m^{(L)} \right) \in \mathbb{R}^{N_m \times 6}, \quad \hat{\mathbf{y}}_{\text{force}} = \text{MLP}_{\text{force}}\left( \mathbf{h}_b^{(L)} \right) \in \mathbb{R}^{N_b \times 12}$$

where $\mathbf{h}_m^{(L)}$ are mesh node hidden states after $L$ layers and $\mathbf{h}_b^{(L)}$ are beam element hidden states. The two decoders share the same encoder backbone representation.

### 3.4 Multi-Scale Macro-Anchor Fusion

Local message passing, even with typed attention, is limited in its ability to propagate information across long distances in large-span structures. A mesh node at midspan must communicate with support nodes multiple segments away, requiring many message passing steps. To address this, we introduce a macro-anchor pooling mechanism that creates a coarse global graph and fuses macro-level information back into micro-node representations.

**Macro anchor pooling.** We partition mesh nodes into $K = 12$ non-overlapping segments based on their $x$-coordinates (longitudinal position along the bridge span). Each segment $k$ contains a set of mesh nodes $\mathcal{M}_k$. The macro anchor representation for segment $k$ is computed as the mean of its constituent micro-node features:

$$\mathbf{a}_k = \frac{1}{|\mathcal{M}_k|} \sum_{v \in \mathcal{M}_k} \mathbf{h}_v \quad \forall k \in \{1, \dots, K\}$$

where $\mathbf{h}_v$ is the hidden representation of mesh node $v$ after micro message passing. The $K$ anchor representations form the node set of a macro graph.

**Macro graph message passing.** The macro graph is constructed as a bidirectional sequential chain connecting adjacent anchors in longitudinal order:

$$\mathcal{E}_{\text{macro}} = \{(k, k+1), (k+1, k) \mid k = 1, \dots, K-1\}$$

We apply $M = 2$ layers of SAGEConv \cite{hamilton2017inductive} on this macro graph:

$$\mathbf{a}_k' = \text{SAGEConv}\left( \mathbf{a}_k, \{\mathbf{a}_{k-1}, \mathbf{a}_{k+1}\} \right)$$

This enables global information propagation across the full bridge span in $M$ steps, independent of the micro-graph distance.

**Cross-scale gated fusion.** The updated macro anchor representations $\mathbf{a}_k'$ carry global structural context. To inject this context back into the micro-node representations, we use a gated residual fusion mechanism:

$$\mathbf{h}_v' = \mathbf{h}_v + \mathbf{g}_v \odot \mathbf{a}_{\text{map}(v)}'$$

where $\text{map}(v)$ maps mesh node $v$ to its anchor segment, $\mathbf{g}_v$ is a learned gate vector computed as:

$$\mathbf{g}_v = \sigma\left( \text{Linear}_{\text{gate}}\left( [\mathbf{h}_v, \mathbf{a}_{\text{map}(v)}'] \right) \right) \in (0, 1)^d$$

and $\sigma$ is the sigmoid activation function. The gate controls how much macro context flows to each micro node, with values near 0 suppressing irrelevant context and values near 1 allowing full integration. Empirical analysis of the trained model shows gate activations in the range $[0.05, 0.37]$, indicating the macro signal is utilized but does not dominate micro representations. The macro-anchor module is applied at each GNN layer, enabling iterative refinement of multi-scale representations.

[Figure 3 about here: MS-HGT macro-anchor architecture — micro message passing → anchor pooling → macro graph → gated cross-scale fusion → dual decoder]

### 3.5 Physics-Informed Regularization

Pure data-driven training may produce predictions that violate basic physical constraints. We introduce two lightweight physics-inspired loss terms that leverage information already present in the data schema, without requiring full stiffness matrix assembly or global equilibrium computation.

**Support boundary condition loss.** Support boundaries have prescribed displacement values (zero in our dataset). The BC loss penalizes deviations from these prescribed values at constrained DOFs using the ground-truth displacements as targets:

$$\mathcal{L}_{\text{BC}} = \frac{1}{|\mathcal{B}_c|} \sum_{v \in \mathcal{B}_c} \sum_{j \in \mathcal{D}_v} \left( \hat{y}_{v,j}^{\text{disp}} - y_{v,j}^{\text{disp}} \right)^2$$

where $\mathcal{B}_c$ is the set of constrained mesh nodes, $\mathcal{D}_v$ is the set of constrained DOF indices for node $v$ (derived from DOF constraint flags in $\mathbf{x}_m$), and $\hat{y}_{v,j}^{\text{disp}}$, $y_{v,j}^{\text{disp}}$ are the predicted and ground-truth displacements for component $j$ at node $v$. Since all constrained DOFs in our dataset have prescribed displacement values of zero, this loss effectively penalizes non-zero predictions at supports.

**Structural link consistency loss.** Rigid structural links enforce identical displacement at their endpoint nodes. The link consistency loss penalizes translation-component mismatches between linked node pairs:

$$\mathcal{L}_{\text{link}} = \frac{1}{|\mathcal{E}_{\text{sl}}|} \sum_{(u,v) \in \mathcal{E}_{\text{sl}}} \left\| \hat{\mathbf{u}}_u^{\text{trans}} - \hat{\mathbf{u}}_v^{\text{trans}} \right\|_2^2$$

where $\mathcal{E}_{\text{sl}}$ is the set of undirected structural link edges, $(u, v)$ are the two endpoint mesh nodes, and $\hat{\mathbf{u}}_v^{\text{trans}} \in \mathbb{R}^3$ denotes the translational displacement prediction $[D_x, D_y, D_z]$ for node $v$. Only the three translational degrees of freedom are constrained; rotational components are excluded.

**Combined objective.** The total training loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}} + \lambda_{\text{link}} \cdot \mathcal{L}_{\text{link}}$$

where $\mathcal{L}_{\text{supervised}} = \mathcal{L}_{\text{MSE}}(\hat{\mathbf{Y}}_{\text{disp}}, \mathbf{Y}_{\text{disp}}) + \mathcal{L}_{\text{MSE}}(\hat{\mathbf{Y}}_{\text{force}}, \mathbf{Y}_{\text{force}})$ is the standard mean squared error supervised loss. After systematic ablation (Section 4.4), we choose $\lambda_{\text{BC}} = 0.08$ and $\lambda_{\text{link}} = 0.002$ for the full variant.

> **Code verification:** Lambda values confirmed against Stage 5 experiment plan and physics_loss.py. `batch_size = 8` and `seed = 42` confirmed against training configs. ✅

### 3.6 Component-Wise Split Conformal UQ

To provide calibrated uncertainty estimates for each output component, we apply split conformal prediction \cite{lei2018distribution, angelopoulos2023gentle} as a post-hoc calibration step after model training.

Given a pre-trained model $f_\theta$, a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ disjoint from the evaluation set, and a test point $\mathbf{x}_{\text{new}}$, we construct prediction intervals as follows.

**Nonconformity score.** For each output component $c$ (6 displacement + 12 force = 18 total), we define the absolute residual score:

$$R_i^{(c)} = \left| \hat{y}_i^{(c)} - y_i^{(c)} \right|$$

where $\hat{y}_i^{(c)}$ is the model's prediction for component $c$ on calibration sample $i$.

**Quantile estimation.** For a target miscoverage rate $\alpha$ (e.g., $\alpha = 0.10$ for 90% coverage), we compute the finite-sample corrected quantile:

$$q^{(c)} = \text{Quantile}\left( \{R_1^{(c)}, \dots, R_n^{(c)}\}; \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

**Prediction interval.** For a new test point $\mathbf{x}_{\text{new}}$, the prediction interval for component $c$ at level $1-\alpha$ is:

$$C^{(c)}(\mathbf{x}_{\text{new}}) = \left[ \hat{y}_{\text{new}}^{(c)} - q^{(c)}, \; \hat{y}_{\text{new}}^{(c)} + q^{(c)} \right]$$

where $q^{(c)}$ is the **half-width** of the interval. The total interval width is $2q^{(c)}$.

**Coverage metrics.** The marginal coverage for component $c$ on a test set $\mathcal{D}_{\text{test}}$ is:

$$\text{Coverage}^{(c)} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i \in \mathcal{D}_{\text{test}}} \mathbb{1}\left\{ y_i^{(c)} \in C^{(c)}(\mathbf{x}_i) \right\}$$

Under exchangeability between calibration and evaluation examples, split conformal prediction provides finite-sample marginal coverage at the target level $(1-\alpha)$, up to the standard quantile discretization effect \cite{lei2018distribution, angelopoulos2023gentle}. This guarantee is **marginal** — it applies to each separately calibrated component individually. It does not extend to simultaneous coverage across multiple DOFs, which we report separately as a diagnostic observation.

---

## 4. Experiments

> **Target: 2.0–2.3 pages. Current: ~2.1 pages.**

### 4.1 Dataset and Experimental Setup

**Dataset.** We use the `hetero_graph_dataset_v2`, built from 70 steel truss girder bridge design samples with 500 load cases each, yielding 35,000 graph instances. Each graph contains 1,056 mesh nodes, 1,646 beam elements, 832 plate elements, and 132 directed structural link edges. All graphs share the same topology (node/edge indices fixed); only features and targets vary with design parameters and loading conditions.

**Data split.** We adopt a `by_sample` split: 80% of design samples (56 samples × 500 cases = 28,000 graphs) for training, 10% (7 samples × 500 = 3,500) for validation, and 10% (7 samples × 500 = 3,500) for testing. This split tests generalization to unseen design parameters under the **same mesh topology**. All features are standardized using training-set statistics; validation and test sets use the training statistics.

**Training setup.** All models are trained with the AdamW optimizer (learning rate 0.001, weight decay $10^{-4}$), batch size 8, and a Reduce-on-Plateau learning rate scheduler (patience 10, factor 0.5). Early stopping with patience 30 epochs is applied. Training uses a single NVIDIA RTX 4090 GPU. Random seed is fixed at 42 for all experiments.

> **Code verification:** `batch_size=8`, `seed=42` confirmed against experiment configs. ✅

**Metrics.** We report: (1) **R² score** (coefficient of determination) — macro-averaged across components for displacement (6 DOF) and force (12 components); (2) **Dy R²** — the vertical deflection component, identified as the most challenging displacement direction; (3) **Relative MAE** — MAE normalized by the absolute mean target value.

[Table 1 about here: Dataset Statistics]

### 4.2 Baseline Comparison

We compare seven models: three baselines without graph structure (MLP), two homogeneous GNNs (GCN \cite{kipf2017semi}, GAT \cite{velickovic2018graph}), two typed heterogeneous GNNs (RGCN \cite{schlichtkrull2018rgcn}, HGT \cite{hu2020hgt}), and two multi-scale variants (MS-HGT gated, MS-PI-HGT-Full). All models are re-implemented within the same pipeline using shared configuration.

[Figure 4 about here: Main Results Bar Chart — Model Comparison on Structural FE Surrogate Prediction]

[Table 2 about here: Main Model Comparison]

**Results and analysis.** The results reveal a clear progression:

1. **MLP is a strong local-feature baseline.** With Force R² = 0.9824 and Disp R² = 0.8554, the MLP demonstrates that local features (coordinates, loads, section properties) are highly informative. However, MLP's Dy R² = 0.1833 reveals a critical failure mode: vertical deflection cannot be accurately predicted from local features alone — it requires non-local structural context.

2. **Homogeneous GNNs do not improve over MLP.** Both GCN (Disp R² = 0.8476) and GAT (Disp R² = 0.8421) perform slightly worse than MLP, indicating that collapsing the heterogeneous FE system into a homogeneous graph with uniform node features is detrimental.

3. **Typed message passing is essential.** RGCN (Disp R² = 0.9366, Dy R² = 0.670) and HGT (Disp R² = 0.9765, Dy R² = 0.905) substantially outperform both MLP and homogeneous GNNs, confirming that type-specific parameterization is necessary for heterogeneous physical graphs.

4. **MS-HGT dramatically improves over HGT.** The macro-anchor fusion raises Disp R² from 0.977 to 0.995 and Dy R² from 0.905 to 0.993 — the most significant single improvement in the ablation chain.

5. **MS-PI-HGT-Full maintains accuracy while improving physical consistency.** The full variant achieves nearly identical predictive accuracy to MS-HGT gated while substantially improving constraint satisfaction.

**Takeaway:** Heterogeneous graph representation and multi-scale fusion are both essential for accurate structural FE surrogate modeling.

### 4.3 Multi-Scale Ablation

To isolate the contribution of the macro-anchor module, we compare HGT (no macro), MS-HGT additive (additive fusion), and MS-HGT gated (gated residual fusion) under identical training conditions.

[Table 3 about here: Multi-Scale Ablation]

**Results.** Both multi-scale variants dramatically outperform HGT: Disp R² rises from 0.9765 to 0.9950 (additive) and 0.9952 (gated). The most striking improvement is in Dy R², which jumps from 0.905 to 0.993 — a relative reduction in unexplained variance of over 92%. The gated variant achieves slightly better overall metrics than additive fusion.

**Analysis.** The macro-anchor mechanism addresses a fundamental limitation of local message passing: information from support regions must propagate through many intermediate nodes to reach midspan. The 12-segment macro graph reduces the effective propagation distance by an order of magnitude, enabling efficient global information flow. The coordinate-based anchor construction (using only $x$-coordinates) is data-efficient and physically interpretable — adjacent anchors correspond to physically adjacent bridge segments.

**Takeaway:** Multi-scale macro-anchor fusion is the primary driver of improvement over HGT. The coordinate-based design avoids data leakage while capturing long-range force transfer essential for accurate displacement prediction.

### 4.4 Physics Regularization Ablation

We compare four variants of MS-HGT with different physics loss configurations: baseline (no physics loss), BC-only ($\lambda_{\text{BC}} = 0.08$), Link-only ($\lambda_{\text{link}} = 0.002$), and Full ($\lambda_{\text{BC}} = 0.08$, $\lambda_{\text{link}} = 0.002$).

[Figure 5 about here: Physics Regularization Ablation — BC Constraint Satisfaction and Force Tail Error]

[Table 4 about here: Physics Regularization Ablation]

**Results and analysis.** Key findings:

1. **BC loss improves physical consistency.** The BC-only variant reduces constrained DOF MAE by 22.3% (from 0.000242 to 0.000188). The Full variant achieves a 38.8% reduction (to 0.000148), indicating synergy between BC and link constraints.

2. **Link loss alone does not improve constraint satisfaction.** The Link-only variant shows increased constrained DOF MAE (0.000277), consistent with the observation that the link loss does not converge well during training.

3. **Full variant achieves best force tail errors.** Force P95 AE is lowest for the Full variant (37,917 vs 38,000 baseline), suggesting that combined regularization benefits tail behavior.

4. **Predictive accuracy is essentially unchanged.** Disp R², Force R², and RelMAE show negligible variation across all four variants, confirming that physics-informed regularization improves physical consistency without degrading accuracy.

**Selection rationale.** We select the Full variant as the final model because it achieves the best balance: largest BC constraint reduction (38.8%), lowest force tail error, and no degradation in predictive accuracy. The improvement in physical consistency comes at no cost to predictive performance.

**Takeaway:** Physics-informed regularization improves physical consistency (especially support BC satisfaction) without degrading predictive accuracy. The Full variant is selected for best overall balance.

### 4.5 Conformal UQ Results

We apply component-wise split conformal prediction using the MS-PI-HGT-Full model. Our primary UQ experiment uses the test_graph_50_50 split, with 1,750 test graphs for calibration and 1,750 test graphs for evaluation (both disjoint from training).

[Figure 6 about here: Conformal Uncertainty Quantification — MS-PI-HGT-Full]

[Table 5 about here: Conformal UQ Coverage]

**Results.** At the 90% nominal level:
- **Displacement:** Overall marginal coverage = 89.74% (gap −0.26pp from nominal). Per-DOF coverage ranges from 89.59% (Dx) to 89.87% (Rz), all within ±0.5pp of the 90% target.
- **Force:** Overall marginal coverage = 89.97% (gap −0.03pp). Per-component coverage ranges from 89.94% to 90.00%.

At the 95% nominal level: displacement coverage = 94.78% (gap −0.22pp), force coverage = 94.97% (gap −0.03pp).

**Coverage-width tradeoff.** The average half-width varies substantially across DOFs, reflecting the heterogeneous uncertainty in different physical quantities. Rotational components (Rx, Ry, Rz) have much smaller half-widths (as low as 2.1 × 10⁻⁵ rad) than translational components (up to 2.0 × 10⁻³ m for Dz), consistent with the physical scale of these quantities.

**Joint coverage (diagnostic).** The joint 6-DOF coverage — the probability that all 6 displacement DOFs simultaneously fall within their intervals — is substantially lower than marginal coverage: 25.2% for support nodes and 55.1% for free nodes. Component-wise split conformal prediction calibrates each output component separately and provides marginal coverage, but it does not guarantee simultaneous coverage over all displacement DOFs. The lower joint coverage therefore reflects the stricter simultaneous-coverage requirement and motivates future work on vector-valued or graph-level conformal calibration. We report joint coverage as a **diagnostic observation** — it is not a conformal guarantee under our component-wise calibration.

**Takeaway:** Component-wise split conformal prediction provides near-nominal marginal coverage across all 18 output components. The gap between marginal and joint coverage is a known property of component-wise intervals, not a model deficiency.

---

## 5. Discussion

> **Target: 0.6–0.8 pages (combined with Conclusion). Current: ~0.35 pages.**

### 5.1 Key Findings

Our experiments establish three main findings. First, heterogeneous graph representation with type-specific message passing is essential for structural FE surrogate modeling — the progression from MLP (Disp R² = 0.855) through homogeneous GNNs (0.842–0.848) to typed models (RGCN 0.937, HGT 0.977) demonstrates this clearly. Second, multi-scale macro-anchor fusion dramatically improves the prediction of long-range structural behavior, particularly vertical deflection (Dy R²: 0.905 → 0.993). Third, physics-informed regularization and conformal UQ can be added as complementary modules that improve physical consistency (38.8% BC reduction) and provide calibrated uncertainty intervals without compromising predictive accuracy.

### 5.2 Limitations

This work has several limitations that should be acknowledged. (1) **Shared topology.** All 70 design samples share the same mesh topology; the by-sample split tests generalization to unseen design parameters, not to new structural configurations. Cross-topology generalization remains an open direction. (2) **Partial physics constraints.** The physics losses penalize specific violation types (support BC and link consistency) but do not enforce full element-level or system-level equilibrium, energy conservation, or constitutive relations. (3) **Marginal UQ guarantee.** The conformal prediction guarantee is per-component marginal; joint coverage across all DOFs simultaneously is substantially lower and requires either vector-valued CP or wider intervals. (4) **Plate element outputs.** Plate internal forces and stresses are not predicted, as these labels are unavailable in the current dataset. (5) **Internal comparison.** All baselines are re-implemented within the same project pipeline; comparisons against external published methods on different datasets are not provided.

### 5.3 Broader Implications

From an ICTAI perspective, this work demonstrates that domain-motivated architectural design — specifically, the alignment of model components with physical structure — yields substantial and measurable benefits. The macro-anchor design, inspired by engineering intuition about how forces propagate through bridge structures, is a concrete example of how domain knowledge can guide AI architecture design. The complete pipeline from raw FE data to calibrated uncertainty intervals also demonstrates the value of treating AI systems as end-to-end tools for practical engineering applications.

---

## 6. Conclusion

> **Target: 0.6–0.8 pages (combined with Discussion). Current: ~0.15 pages.**

We presented MS-PI-HGT, a multi-scale physics-informed heterogeneous graph transformer for structural FE surrogate modeling. The framework integrates four key components: (1) a heterogeneous graph representation that captures the multi-entity structure of steel truss FE systems; (2) macro-anchor multi-scale fusion with gated cross-scale connections for long-range force transfer; (3) lightweight physics-informed regularization for improved physical consistency; and (4) component-wise split conformal prediction for calibrated uncertainty quantification.

On a dataset of 35,000 steel truss bridge FE instances, MS-PI-HGT achieves a progressive improvement from MLP (Disp R² = 0.855) through typed message passing (HGT, 0.977) to multi-scale fusion (MS-HGT, 0.995), with physics-informed regularization reducing support BC violations by 38.8% and conformal UQ providing near-nominal marginal coverage across all 18 output components.

Future work includes extending the framework to cross-topology generalization with multi-mesh datasets, incorporating additional physics constraints (beam-level equilibrium, plate constitutive relations), and developing adaptive UQ methods to bridge the gap between marginal and joint coverage for structural engineering applications.

---

## References

> **Note:** All citation keys reference `references/ictai_refs.bib`. 19 unique citations used in this draft. See Phase 3-precheck for verified key consistency.

\bibliographystyle{IEEEtran}
\bibliography{references/ictai_refs}

---

## Writing Notes (for next iteration)

### Compression targets
| Section | Current pages | Target | Savings needed |
|---------|:------------:|:------:|:--------------:|
| Abstract | ~0.3 | 0.25 | tighten wording |
| Introduction | ~1.0 | 1.0–1.2 | at target |
| Related Work | ~1.1 | 1.0–1.2 | at target |
| Methodology | ~2.5 | 2.3–2.7 | at target |
| Experiments | ~2.1 | 2.0–2.3 | minor tighten |
| Discussion | ~0.35 | 0.6–0.8 | expand slightly |
| Conclusion | ~0.15 | (included with Discussion) | merge |
| References | - | - | 2–3 lines for bibliography |

### TODO_VERIFY items for author
1. ~~**§3.2:** Confirm plate_element feature dimension~~ — ✅ Done (`hetero_schema.py` confirms feature_dim=6)
2. ~~**§3.5:** Confirm BC and link loss lambda values~~ — ✅ Done (λ_BC=0.08, λ_link=0.002 from Stage 5 config)
3. ~~**§4.1:** Confirm batch_size=8 and seed=42~~ — ✅ Done (confirmed against training configs)
4. **F1/F2/F3:** Requires manual drawing (draw.io or Illustrator) — highest priority for Phase 4
5. **All sections:** Citation key verification against `ictai_refs.bib` — done in Phase 3-precheck
6. **ICTAI template:** Obtain IEEEtran.cls for LaTeX conversion

### Final checks before submission
- [ ] All numerical values match `metrics_summary.json` and locked audit tables
- [ ] All `\cite{}` keys exist in `references/ictai_refs.bib`
- [ ] No "SOTA", "first" (unqualified), "cross-topology", "full equilibrium" claims
- [ ] Joint coverage clearly labeled as diagnostic
- [ ] Physics loss claims: "improves physical consistency" not "improves accuracy"
- [ ] Figures F1/F2/F3 created and inserted
- [ ] Figures F4/F5/F6 captions match manifest

---

*Document version: v1.0 / 2026-07-08 / ICTAI Phase 3 Initial Draft*

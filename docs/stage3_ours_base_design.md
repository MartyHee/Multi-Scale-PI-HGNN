# Stage 3 Ours-base Design Specification

> **Purpose:** Design specification for the base model of Multi-Scale Physics-Informed Heterogeneous GNN (Ours-base).
> **Status:** Design only — no implementation, no training.
> **Date:** 2026-06-23
> **Baseline context:** HGT (best), RGCN, MLP, GCN, GAT all completed.
> **Full prediction diagnostics:** Available (tail error, high/low response, support BC residual).

---

## 1. Why Ours-base? — HGT's Remaining Weaknesses

HGT is the strongest baseline, but full prediction diagnostics reveal specific gaps:

| Dimension | HGT | Gap | Innovation Target |
|-----------|:---:|:---:|:-----------------|
| **Disp R² (macro)** | 0.9769 | 2.3% unexplained | 📌 Moderate |
| **Dy R²** | 0.9077 | 9.2% unexplained | **🎯 Primary** |
| **HighResponse Disp R²** | 0.208 | vs MLP 0.336 | **🎯 Primary** |
| **Support BC residual (mean)** | 0.000206 | vs RGCN 0.000158 | 📌 Secondary |
| **Force R² (macro)** | 0.9891 | ~1% unexplained | 📌 Maintain |
| **RelMAE** | 0.0683 | Better than all baselines | 📌 Maintain/improve |
| **Parameter count** | 744K | 4.4× MLP, 1.4× RGCN | 📌 Efficiency |
| **Training time** | 5.9h | 2.6× RGCN | 📌 Efficiency |

### 1.1 Key Scientific Gap

HGT uses typed attention (learned) but **no physical knowledge** about:
- Stiffness of structural links (Kx, Ky, Kz, Krx, Kry, Krz)
- Connection geometry (BetaAngle, DistanceRatio)
- Physical type of edge (RIGID link vs future elastic link)
- Material / cross-section properties along message paths

The hypothesis: *Edge-attribute-aware and physics-gated message passing can improve upon pure learned typed attention, especially for physically difficult components (Dy, high-response regions) and boundary consistency.*

### 1.2 What Ours-base Is NOT

- ❌ NOT a "replace all conv layers with fancier conv"
- ❌ NOT a "stack more layers than HGT"
- ❌ NOT a minor hyperparameter tweak of HGT
- ❌ NOT a renamed RGCN with more parameters

---

## 2. Model Positioning

```
Model Name: Physics-Typed Edge-Aware Heterogeneous GNN (Ours-base)
Type:       Heterogeneous Graph Neural Network
Key Novelty: Edge-attribute-aware + physics-gated typed message passing
Scope:      Micro-level heterogeneous graph only (no macro anchor, no physics loss)
```

### 2.1 Relation to Existing Work

| Module | Reference | Adaptation for Ours |
|--------|-----------|-------------------|
| Heterogeneous encoder | HGT / RGCN typed projection | Per node-type MLP encoder, same as baselines |
| Typed micro message | PyG MessagePassing + RGCN relation-specific | Add edge_attr conditioning |
| Physics gate | Novel (not directly from existing paper) | Stiffness-aware gating on structural_link |
| Dual decoder | Shared latent → disp head + force head | Multi-task architecture |
| Structural link modeling | MeshGraphNets edge encoding (inspiration) | Edge encoder for stiffness features |

---

## 3. Core Module Design

### 3.1 Heterogeneous Physical Graph Encoder (Reuse from Baselines)

Same as existing baselines:
- `mesh_node` → `Linear(in_features, hidden_dim)`
- `beam_element` → `Linear(in_features, hidden_dim)`
- `plate_element` → `Linear(in_features, hidden_dim)`

**No change needed.** This is not an innovation target.

### 3.2 Edge-Attribute-Aware Structural Link Message 🎯

**This is the primary innovation of Ours-base.**

#### Motivation

Current structural_link handling in baselines:
- **MLP:** No graph structure — structural_link information is lost
- **GCN/GAT:** Homogeneous graph — structural_link is just another edge, no edge_attr
- **RGCN:** Relation-specific conv — structural_link has its own relation type, but edge_attr (Kx..Kz, BetaAngle, etc.) is **unused**
- **HGT:** Typed attention — attention weights are learned, not conditioned on edge features

#### Proposed Design

```python
class StructuralLinkConv(MessagePassing):
    """
    Edge-attribute-aware message passing for structural_link edges.
    
    For each structural_link edge (mesh_node_i -> mesh_node_j):
    1. Encode edge attributes into a message bias/gate:
       edge_encoding = MLP(edge_attr)  # Kx..Kz, BetaAngle, DistanceRatio, is_rigid
    2. Node message:
       message = W_src * h_src + edge_encoding    # additive bias
       # OR: message = edge_gate * (W_src * h_src)  # multiplicative gate
       # OR: message = W_src * h_src + W_edge * edge_encoding  # combined
    3. Update target node:
       h_j' = h_j + W_self * h_j + aggregate(messages)
    """
```

**Design options (to be decided during implementation, not now):**

| Option | Complexity | Expected Benefit | Risk |
|--------|:----------:|:----------------:|:----:|
| A: Additive bias `h_src + W_e * edge_attr` | Low | Moderate Dy improvement | May be insufficient |
| B: Multiplicative gate `gate(W_e * edge_attr) * W_s * h_src` | Medium | Higher structural link control | Gate saturation |
| C: Combined `W_s * h_src + W_e * edge_attr` | Medium | Best flexibility | More params |

**Recommendation for implementation:** Start with Option A (additive bias), compare against C. Option B if A insufficient.

#### Edge Feature Specification

From `structural_link.edge_attr` (8 dims in current dataset):

| Index | Feature | Type | Range |
|:-----:|---------|:----:|:-----:|
| 0 | Kx (stiffness) | Continuous | Large range |
| 1 | Ky (stiffness) | Continuous | Large range |
| 2 | Kz (stiffness) | Continuous | Large range |
| 3 | Krx (rotational stiffness) | Continuous | Large range |
| 4 | Kry (rotational stiffness) | Continuous | Large range |
| 5 | Krz (rotational stiffness) | Continuous | Large range |
| 6 | BetaAngle | Continuous | [0, 180] |
| 7 | DistanceRatio | Continuous | [0, ~1] |
| 8 | ElasticLinkType | Discrete | Currently 0/1 (RIGID) |
| 9 | is_rigid | Discrete | Currently 1 |

**Encoding:** Continuous features → linear layer + LayerNorm. Discrete features → embedding (even if currently only 1 value, to support future elastic link types).

#### Why This Specifically Targets Dy and High-Response Errors

In steel truss girders:
- Lateral displacement (Dy) is governed by lateral bracing and cross-frame connections
- Structural links are the primary components that provide lateral stiffness
- HGT treats all edges equally through learned attention — the stiffness differences between a rigid connection and a bracing connection are not explicitly represented
- Edge_attr-aware message passing enables the model to **learn stiffness-dependent message strength**

### 3.3 Physics-Gated Message Function 🎯

#### Motivation

HGT attention weights are purely learned from node features — they can attend to any relation type arbitrarily. A physics gate explicitly modulates messages based on the physical type and properties of each edge.

#### Proposed Design

```python
def physics_gate(h_src, h_dst, relation_type, edge_attr):
    """
    Compute a physics-informed gate value for the message.
    - For membership edges (belongs_to_beam, belongs_to_plate):
        gate = sigmoid(MLP([h_src, h_dst, type_embedding]))
        # Soft modulation based on node states and relation type
    - For structural_link edges:
        gate = sigmoid(MLP([h_src, h_dst, type_embedding, edge_encoding]))
        # Extra conditioning on stiffness features
    """
    return gate

# Message = gate * typed_message(h_src, h_dst)
```

**Design rationale:**
- Membership edges (mesh → beam, mesh → plate) benefit from typed message passing already validated by RGCN/HGT
- Adding a physics gate to membership edges is optional and can be ablated
- Structural_link edges benefit MOST from physics gating because they carry explicit stiffness information

### 3.4 Membership Edge Message (Typed, No Edge Attr)

For `belongs_to_beam`, `rev_belongs_to_beam`, `belongs_to_plate`, `rev_belongs_to_plate`:

- These are incidence/adjacency edges, not physical connection edges
- They don't carry meaningful edge_attr in the current schema
- **Recommendation:** Use standard typed message passing (RGCN-style or HGT-style)
- Do NOT force edge_attr onto edges that have none

This keeps the design minimal: edge_attr-awareness is only applied where there IS edge_attr.

### 3.5 Dual Decoder with Shared Physics Latent 🎯

#### Motivation

HGT and RGCN use separate decoders for displacement and force. There is no explicit mechanism for the model to learn that displacement and force are physically coupled (stiffness matrix relationship).

#### Proposed Design

```python
# Shared latent after message passing layers
h_mesh_shared = message_passing_output["mesh_node"]
h_beam_shared = message_passing_output["beam_element"]

# Displacement decoder (mesh_node only)
disp_hidden = MLP(h_mesh_shared)
disp_pred = Linear(disp_hidden, 6)

# Force decoder (beam_element only)
force_hidden = MLP(h_beam_shared)
force_pred = Linear(force_hidden, 12)

# Optional: cross-attention from beam latent to mesh latent
# (add in Stage 4 macro module instead)
```

**Key design question:** Should the shared latent be:
- **Direct:** Just use the same final hidden state (simplest, recommended for Ours-base)
- **Projected:** `h_shared = W_proj * h_final` (adds a projection layer, marginal benefit unclear)
- **Cross-attended:** beam nodes attend to mesh nodes before decoding (add in Stage 4)

**Recommendation for Ours-base:** Direct shared latent. Cross-attention can be added in Stage 4 as part of cross-scale fusion.

### 3.6 Overall Architecture

```
Input Features
    │
    ▼
┌─────────────────────────────┐
│ Heterogeneous Node Encoder  │  ← Per-type Linear (same as baselines)
│  mesh_node → h_mesh(0)      │
│  beam_element → h_beam(0)   │
│  plate_element → h_plate(0) │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Typed Micro Message Passing (L layers)          │  ← Core innovation
│                                                  │
│  for each layer l:                               │
│    membership edges:                             │
│      h_msg = typed_conv(h_src, h_dst, rel_type)  │  ← Standard typed message
│    structural_link edges:                         │
│      edge_enc = EdgeEncoder(edge_attr)            │  ← NEW: edge_attr encoding
│      gate = PhysicsGate(h_src, h_dst, edge_enc)   │  ← NEW: physics gate
│      h_msg = gate * typed_conv(h_src, h_dst)      │
│      h_msg = h_msg + edge_enc_bias                │  ← NEW: edge conditioning
│    node update:                                   │
│      h_i(l+1) = update(h_i(l), aggr(messages))    │
└─────────────────────────────────────────────────┘
    │
    ├───────────────────┬──────────────────────┐
    ▼                   ▼                      ▼
┌────────────┐   ┌──────────────┐   ┌────────────────┐
│ Mesh Node  │   │ Beam Element │   │ Plate Element  │  ← Hidden states only
│ Final      │   │ Final        │   │ (no decoder)   │
│ h_mesh(L)  │   │ h_beam(L)    │   │                │
└────────────┘   └──────────────┘   └────────────────┘
    │                   │
    ▼                   ▼
┌──────────────┐  ┌──────────────┐
│ Disp Decoder │  │ Force Decoder│  ← Dual decoder with shared latent
│ MLP → 6      │  │ MLP → 12     │
└──────────────┘  └──────────────┘
    │                   │
    ▼                   ▼
 disp_pred         force_pred
```

---

## 4. What Is NOT in Ours-base

The following are explicitly **excluded** from Stage 3 (Ours-base):

| Module | Reason for Exclusion | When |
|--------|---------------------|------|
| Macro anchor graph | Would confound micro message passing innovation | Stage 4 |
| Cross-scale fusion | Part of macro module | Stage 4 |
| Physics loss (support BC) | Would mask whether architectural improvements work | Stage 5 |
| Physics loss (equilibrium) | Full physics loss validation requires architecture first | Stage 5 |
| Uncertainty quantification | Separate methodological contribution | Stage 6 |
| Hyperparameter search | Use HGT's settings as starting point | Implementation |

**Boundary rule:** If a module requires macro-level graph construction, additional loss terms, or uncertainty calibration, it belongs to Stage 4/5/6.

---

## 5. Comparison Protocol

### 5.1 Controlled Variables

| Variable | Setting | Rationale |
|----------|---------|-----------|
| Dataset | `hetero_graph_dataset_v2` | Canonical dataset |
| Split | `by_sample` | Same as baselines |
| Epochs | 100 | Same as baselines |
| Batch size | 32 | Same as baselines |
| Hidden dim | Match HGT (64 or 128) | Fair comparison |
| Layers | 3 | Same as HGT |
| Optimizer | Adam, same lr schedule | Same as baselines |
| Seed | [42, 123, 456] | 3 seeds for variance estimation |
| Device | Same server GPU | Same hardware |

### 5.2 Independent Variables

| Variable | Ours-base | HGT | RGCN |
|----------|:---------:|:---:|:----:|
| Edge attr in structural_link | ✅ Yes | ❌ No | ❌ No |
| Physics gate | ✅ Yes | ❌ No | ❌ No |
| Dual shared latent | ✅ Yes | ❌ Separate decoders | ❌ Separate decoders |
| Attention | Gated (physics) | Learned (HGTConv) | None (SAGEConv) |

### 5.3 Primary Comparisons

1. **Ours-base vs HGT:** Does edge_attr awareness improve over pure typed attention?
2. **Ours-base vs RGCN:** Does physics gating + edge_attr improve over relation-specific conv?
3. **Ablation: Ours-base w/o edge_attr:** How much gain comes from edge_attr vs architecture?
4. **Ablation: Ours-base w/o physics gate:** How much gain comes from gating?

---

## 6. Documentation Dependencies

Before implementing Ours-base, the following must be in place:

- [x] `processed/hetero_graph_dataset_v2` — complete and verified
- [x] `feature_stats.json` — standardization statistics
- [x] Baseline metrics (MLP, GCN, GAT, RGCN, HGT) — complete
- [x] Full prediction diagnostics — available
- [x] Metric sanity check — documented in `docs/stage3_metric_sanity_check.md`
- [ ] `src/models/ours_base.py` — model definition (future, not part of this design task)
- [ ] Ours-base config in `configs/models.yaml` (future)

---

## 7. Innovation Boundary Statement

> **Ours-base is not claimed as a complete Multi-Scale PI-HGNN.**
> It is the micro-level message passing component, with edge_attr-aware structural_link handling and physics-gated typed message passing.
>
> The full "Multi-Scale" comes from Stage 4 (macro anchor + cross-scale fusion).
> The "Physics-Informed" in the name comes from Stage 5 (physics regularization loss).
> The "PI" label in the full model name applies only after Stage 4+5 are integrated.
>
> Ours-base alone should be referred to as: **Edge-Attribute-Aware Heterogeneous GNN** or **EA-HGNN** (working title).
> The full model is: **Multi-Scale PI-HGNN**.
>
> This separation prevents overclaiming at Stage 3.

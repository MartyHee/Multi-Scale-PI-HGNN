# Stage 2-A Baseline Results Draft

> **Status:** In progress — MLP completed, GCN completed, GAT: config ready (awaiting server execution).
> **Dataset:** `processed/hetero_graph_dataset_v2`
> **Split:** `by_sample` (train=28,000 / val=3,500 / test=3,500)
> **Last updated:** 2026-06-21

## Results Table

| Method | Graph Type | Typed Message | Multi-scale | Physics Loss | Params | Best Epoch | Train Time | Test Disp R² | Test Force R² | Test Disp MAE | Test Force MAE | Combined RelMAE | Run Dir | Artifact | Status |
|--------|-----------|---------------|-------------|--------------|-------|-----------|------------|-------------|--------------|--------------|---------------|----------------|---------|----------|--------|
| MLP | none | no | no | no | 96,274 | 88 | 3412.9s (56.9min) | 0.8554 | 0.9824 | 0.000208 | 15972.99 | 0.0884 | `outputs/baselines/MLP/20260620051300/` | `server_mlp_full_20260620060955.tar.gz` | ✅ completed |
| GCN | homogeneous | no | no | no | 76,050 | 96 | 6891.3s (114.9min) | 0.8476 | 0.9696 | 0.000274 | 22604.84 | 0.1227 | `outputs/baselines/GCN/20260620123654/` | `server_gcn_full_20260620143146.tar.gz` | ✅ completed |
| GAT | homogeneous | no | no | no | 76,818 (expected) | — | — | — | — | — | — | — | — | `server_gat_full_<timestamp>.tar.gz` | 🟡 running-ready |

## MLP Details

- **Model class:** `MLPBaseline`
- **Input:** mesh_node.x (15-dim) + beam_element.x (11-dim) with scatter_mean endpoint aggregation
- **Architecture:** [256, 128, 64] hidden dims, ReLU, BatchNorm, dropout=0.1
- **Total time:** 3412.9s (56.9 min)
- **Device:** 8× RTX 4090
- **Num workers:** 4
- **Batch size:** 32
- **Best epoch:** 88 (no early stopping, ran 100 epochs)
- **Artifact:** Verified in `docs/stage2a_mlp_artifact_check.md`

## GCN Details

- **Model class:** `HomogeneousGCN`
- **Input:** HeteroToHomoAdapter → 3× GCNConv(hidden=128) → DispDecoder + ForceDecoder
- **Architecture:** 3 GCNConv layers, hidden_dim=128, dropout=0.2, BatchNorm, type_embed=True
- **Total time:** 6891.3s (114.9 min)
- **Device:** 8× RTX 4090
- **Num workers:** 4
- **Batch size:** 16
- **Best epoch:** 96 (val_loss); Disp peaked at epoch 100
- **Artifact:** Verified (13 files, 3928 KB)

## MLP vs GCN Comparison

| Metric | MLP | GCN | Δ |
|--------|-----|-----|---|
| Test Disp R² (macro) | **0.8554** | 0.8476 | -0.0078 |
| Test Force R² (macro) | **0.9824** | 0.9696 | -0.0128 |
| Test Disp MAE | **0.000208** | 0.000274 | +0.000066 |
| Test Force MAE | **15972.99** | 22604.84 | +6631 |
| Combined RelMAE | **0.0884** | 0.1227 | +0.0343 |
| Params | 96,274 | 76,050 | -20,224 |
| Train Time | **3412.9s** | 6891.3s | +3478s (2×) |

**Key finding: Homogeneous GCN underperforms the non-graph MLP across all metrics.**

### Per-Component Analysis

**Displacement R²:**
| Component | MLP | GCN | Δ |
|-----------|-----|-----|---|
| Dx | 0.9912 | 0.9829 | -0.0082 |
| **Dy** | **0.1833** | **0.1778** | **-0.0055** |
| Dz | 0.9918 | 0.9832 | -0.0085 |
| Rx | 0.9931 | 0.9855 | -0.0076 |
| Ry | 0.9882 | 0.9817 | -0.0065 |
| Rz | 0.9850 | 0.9746 | -0.0104 |

**Force R²:**
| Component | MLP | GCN | Δ |
|-----------|-----|-----|---|
| Fx_I/J | 0.9935 | 0.9793 | -0.0141 |
| Fy_I/J | 0.9832 | 0.9686 | -0.0146 |
| Fz_I/J | 0.9673 | 0.9680 | +0.0008 |
| Mx_I/J | 0.9879 | 0.9763 | -0.0116 |
| My_I/J | 0.9744 | 0.9632 | -0.0112 |
| Mz_I/J | 0.9883 | 0.9623 | -0.0260 |

### Key Diagnostic Observations

1. **Dy is uniformly hard** — Both MLP (0.1833) and GCN (0.1778) fail on Dy (lateral Y-displacement). This is likely a **data characteristic**: Y-direction displacement in a truss bridge is minimal (lateral stiffness is high), so the signal-to-noise ratio is very low. Both models default near the mean.

2. **GCN degrades all other components** — For displacement (Dx, Dz, Rx, Ry, Rz: -0.006~-0.010) and force (Fx, Fy, Mx, My, Mz: -0.01~-0.026), GCN is consistently worse.

3. **Fz is essentially unchanged** (+0.0008) — The vertical force (Fz) is the dominant force component and least affected by the homogeneous conversion.

4. **Training time doubles** — GCN (114.9 min) takes exactly 2× MLP (56.9 min), as expected from message passing overhead.

### Root Cause Analysis

The homogeneous GCN's underperformance is attributable to:

1. **Edge type information loss**: `HeteroToHomoAdapter` merges 5 distinct edge types (belongs_to_beam, rev_belongs_to_beam, belongs_to_plate, rev_belongs_to_plate, structural_link) into a single edge_index. GCNConv treats all edges identically — a structural_link (mesh↔mesh) carries the same weight as a belongs_to_beam (mesh→beam). This "edge type confusion" degrades message quality.

2. **Node type dilution**: 3 node types (mesh 15-dim, beam 11-dim, plate 6-dim) are projected into a shared 128-dim space with 3-dim type embeddings. The physical node-type identity is compressed into 3 dimensions, and plate nodes (832, no supervision) participate in message passing as equals with supervised nodes.

3. **Oversmoothing risk**: With 3534 nodes and 3 GCNConv layers, isotropic message passing may smooth out directional force patterns specific to each node type.

**This result is expected and informative:** It validates the research motivation for typed message passing (Stage 2-B: RGCN/HGT) and physics-aware micro message passing (Stage 3: Ours-base). Homogeneous GCN is not the right tool for physics-heterogeneous graphs.

## Notes

- Stage 2-A will be **complete** when MLP + GCN + GAT all have verified artifact results.
- After Stage 2-A, proceed to Stage 2-B (RGCN, HGT, MeshGraphNet) or directly to Stage 3 (Ours-base) as per project plan.
- "Typed Message = no" means these baselines do not use relation-specific message passing — this is the key distinction from Stage 2-B and Stage 3 models.

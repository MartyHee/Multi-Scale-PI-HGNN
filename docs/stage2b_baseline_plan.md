# Stage 2-B Baseline Plan — Typed Message Passing Baselines

> **Status:** ✅ **RGCN completed** — next: HGT  
> **Dataset:** `processed/hetero_graph_dataset_v2`  
> **Split:** `by_sample` (train=28,000 / val=3,500 / test=3,500)  
> **Last updated:** 2026-06-21

## 1. Stage 2-B Goal

Evaluate whether **typed/relation-specific message passing** recovers the performance gap observed in Stage 2-A, where homogeneous GCN/GAT degraded below the non-graph MLP.

## 2. Motivation

Stage 2-A established:
| Method | Graph Type | Disp R² | Force R² | RelMAE |
|--------|-----------|--------:|---------:|-------:|
| MLP | none | **0.8554** | **0.9824** | **0.0884** |
| GCN | homogeneous | 0.8476 | 0.9696 | 0.1227 |
| GAT | homogeneous | 0.8421 | 0.9632 | 0.1361 |

**Monotonic degradation: MLP > GCN > GAT.**

The root cause analysis pointed to three factors:

1. **Edge type information loss** (primary): `HeteroToHomoAdapter` merges 5 physically distinct edge types into one `edge_index`. GCNConv/GATConv cannot distinguish `belongs_to_beam` from `structural_link`.
2. **Node type dilution** (secondary): 3 node types projected into shared space with only a 3-dim type embedding.
3. **MLP's physically meaningful aggregation**: `scatter_mean` over `belongs_to_beam` edges encodes the exact physical connectivity for force prediction.

**Stage 2-B tests Hypothesis 1 directly**: if edge-type confusion is the primary cause, then preserving edge-type identity via typed message passing should partially or fully recover the gap.

## 3. Why RGCN first

**RGCN / HeteroConv** is the simplest typed-relation baseline:

- Each edge type has independent convolution weights → directly tests "does preserving edge type matter"
- Uses PyG's `HeteroConv` dispatcher + `SAGEConv` per edge type
- No attention, no edge_attr, no physics gating → pure typed message passing
- Quick to implement, quick to train
- If RGCN recovers → strong evidence for Hypothesis 1
- If RGCN does not recover → need to look deeper (maybe node type dilution is more important, or the graph structure itself is limiting)

## 4. RGCN vs GCN/GAT — Key Differences

| Aspect | GCN / GAT | RGCN (HeteroConv) |
|--------|-----------|-------------------|
| Edge types | Merged → 1 `edge_index` | 5 separate `edge_index` dicts |
| Message weights | Shared across all edges | Per-edge-type independent weights |
| Node types | Projected + type embed | Projected (heterogeneous native) |
| plate_element | Participates in message passing | Participates in message passing |
| structural_link | Merged with membership edges | Separate convolution weights |
| Decoder | Same MLPHead | Same MLPHead |

## 5. Model Architecture (HeteroRGCNBaseline)

```
Input: HeteroDataBatch
  │
  ├─ mesh_encoder: Linear(15 → 128)
  ├─ beam_encoder: Linear(11 → 128)
  └─ plate_encoder: Linear(6 → 128)
  │
  └─ Layer 1: HeteroConv({
       belongs_to_beam:        SAGEConv(128→128),
       rev_belongs_to_beam:    SAGEConv(128→128),
       belongs_to_plate:       SAGEConv(128→128),
       rev_belongs_to_plate:   SAGEConv(128→128),
       structural_link:        SAGEConv(128→128),
     }, aggr='sum') → ReLU → Dropout → LayerNorm
  │
  └─ Layer 2: (same structure)
  │
  └─ Layer 3: (same structure)
  │
  ├─ disp_decoder: MLPHead(128 → 6)    [mesh_node]
  └─ force_decoder: MLPHead(128 → 12)  [beam_element]
```

**Params:** 520,338 (vs GCN 76K, GAT 77K, MLP 96K) — larger due to 5× per-edge-type SAGEConv weights.

## 6. Input/Output Schema

Matches `hetero_graph_dataset_v2` exactly:
- **Nodes:** mesh_node (15-dim), beam_element (11-dim), plate_element (6-dim)
- **Edges:** 5 canonical types (see §5)
- **Output:** pred_disp (M×6), pred_force (B×12)
- **plate_element:** unlabeled, participates in message passing only
- **structural_link:** no edge-level label, participates as standard edge

## 7. What is NOT included

This baseline explicitly omits:

| Component | Reason |
|-----------|--------|
| Edge-attribute-aware gating | Reserved for Ours (physics-gated message) |
| Physics loss / BC loss | Reserved for Stage 5 |
| Macro anchor graph | Reserved for Stage 4 |
| Cross-scale fusion | Reserved for Stage 4 |
| Uncertainty quantification | Reserved for Stage 6 |
| Node-type-specific conv depth | All types use same SAGEConv structure |
| Attention mechanism | RGCN is relation-specific convolution, not attention |

## 8. Success Criteria

| Criterion | Target | Interpretation |
|-----------|--------|----------------|
| RGCN Disp R² > GCN (0.8476) | ✅ | Typed message recovers displacement |
| RGCN Force R² > GCN (0.9696) | ✅ | Typed message recovers force |
| RGCN RelMAE < GCN (0.1227) | ✅ | Overall error reduces |
| RGCN approaches MLP (0.8554/0.9824) | 🎯 | Typed message nearly closes the gap |
| Dy component improves | 🔍 | Even lateral displacement benefits from typed message |
| Force R² stays > 0.97 | ✅ | Force prediction remains reliable |

## 9. Expected Recovery Trajectory

```
MLP  (0.8554, 0.9824)  ← non-graph upper bound
  → GCN  (0.8476, 0.9696)  ← homogeneous, degraded by type confusion
  → GAT  (0.8421, 0.9632)  ← further degraded by noisy attention
  → RGCN (???, ???)         ← typed message, expected recovery
  → HGT  (???, ???)         ← typed attention, expected further recovery
```

If RGCN beats MLP on force R² (which MLP already gets at 0.9824), then typed message passing is essential for this task. If it does not, the hypothesis needs refinement.

## 10. RGCN Results (2026-06-21)

### 10.1 Summary

| Metric | MLP | GCN | GAT | **RGCN** | RGCN Δ vs MLP |
|--------|-----|-----|-----|----------|---------------|
| Test Disp R² | 0.8554 | 0.8476 | 0.8421 | **0.9366** | **+0.0812** |
| Test Force R² | 0.9824 | 0.9696 | 0.9632 | **0.9878** | **+0.0054** |
| Combined RelMAE | 0.0884 | 0.1227 | 0.1361 | **0.0724** | **-0.0160** |

### 10.2 Success criteria assessment

| Criterion | Target | Actual | Outcome |
|-----------|--------|--------|---------|
| RGCN Disp R² > GCN (0.8476) | ✅ | 0.9366 | **+0.0890** ✅ |
| RGCN Force R² > GCN (0.9696) | ✅ | 0.9878 | **+0.0182** ✅ |
| RGCN Disp R² > MLP (0.8554) | 🎯 | 0.9366 | **+0.0812** ✅ |
| RGCN Force R² > MLP (0.9824) | 🎯 | 0.9878 | **+0.0054** ✅ |
| RGCN RelMAE < GCN (0.1227) | ✅ | 0.0724 | **-0.0503** ✅ |
| Dy component improved | 🔍 | 0.18 → **0.67** | **+0.49** ✅ |
| Force R² stays > 0.97 | ✅ | **0.9878** | ✅ |

All criteria met. Hypothesis confirmed: **typed message passing fully recovers and exceeds the Stage 2-A homogeneous performance gap.**

### 10.3 Key finding

The Dy bottleneck (R²≈0.18 across all homogeneous models) was broken by RGCN (R²=0.67). This proves it was a **model limitation, not a data limitation**. `structural_link` edges carrying lateral stiffness information require type-aware processing to be useful.

## 11. Next Steps

| Step | Model | Status |
|------|-------|--------|
| 1 | **RGCN (HeteroConv + SAGEConv)** | ✅ **Disp R² 0.9366, Force R² 0.9878** |
| 2 | **HGT (typed attention, HGTConv)** | ✅ **Disp R² 0.9769, Force R² 0.9891** |
| 3 | MeshGraphNet-style baseline | 💤 Optional — not critical to research thesis |

**Recommendation:** Stage 2-B typed baselines (RGCN, HGT) completed. Enter **Stage 3 (Ours-base)** next. MeshGraphNet-style processor remains optional. The comparison chain is now fully established:

```
Homogeneous (GCN/GAT) < Non-graph (MLP) < Typed Conv (RGCN) < Typed Attn (HGT)
```

Ours-base should build on HGT's typed attention foundation while introducing physics-gated message passing as the key innovation over standard HGT.

Note: MeshGraphNet-style processor is less critical because the core research questions have been answered by the RGCN → HGT progression. Consider it only if needed for the MeshGraphNet citation in related work.

## 11. If RGCN Still Underperforms

If RGCN does not recover to at least GCN level, investigate:

1. **HeteroConv relation aggregation**: Check if `aggr='sum'` is appropriate vs `'mean'`
2. **Reverse edges**: Ensure `rev_belongs_to_beam` and `rev_belongs_to_plate` are active
3. **plate_element participation**: Verify plate features provide useful signal (not noise)
4. **structural_link sparsity**: Only 132 edges per graph — may need higher weight or separate treatment
5. **Decoder input correctness**: Verify `x_dict['mesh_node']` and `x_dict['beam_element']` have correct gradients
6. **Edge_attr**: Would edge-attribute-aware gating (e.g., physical stiffness features on edges) help?

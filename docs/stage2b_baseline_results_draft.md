# Stage 2-B Baseline Results Draft — Typed Message Passing Baselines

> **Status:** ✅ RGCN completed — 🏗️ HGT running-ready (server training pending)
> **Dataset:** `processed/hetero_graph_dataset_v2`
> **Split:** `by_sample` (train=28,000 / val=3,500 / test=3,500)
> **Last updated:** 2026-06-22

## Results Table

| Method | Graph Type | Typed Message | Edge Attr Aware | Params | Disp R² | Force R² | RelMAE | Status |
|--------|-----------|---------------|----------------|-------:|--------:|---------:|-------:|--------|
| MLP | none | no | no | 96,274 | 0.8554 | 0.9824 | 0.0884 | ✅ completed |
| GCN | homogeneous | no | no | 76,050 | 0.8476 | 0.9696 | 0.1227 | ✅ completed |
| GAT | homogeneous | no | no | 76,818 | 0.8421 | 0.9632 | 0.1361 | ✅ completed |
| **RGCN** | **heterogeneous** | **relation-specific (SAGEConv)** | **no** | **520,338** | **0.9366** | **0.9878** | **0.0724** | ✅ **completed** |
| **HGT** | **heterogeneous** | **typed attention (HGTConv)** | **no** | **744,279** | **TBD** | **TBD** | **TBD** | 🔄 **running-ready** |

## RGCN Result (completed)

| Detail | Value |
|--------|-------|
| Model class | `HeteroRGCNBaseline` |
| Conv type | `HeteroConv` + per-edge-type `SAGEConv` |
| Params | 520,338 |
| Best epoch | 90 |
| Train time | 8142.3s (135.7 min) |
| Artifact | 13 files, 12048 KB, exit code 0 |
| Edge attr aware? | **No** — pure typed convolution |

### Key findings

1. **RGCN > MLP > GCN > GAT** — typed message passing is essential
2. **Dy breakthrough**: 0.1833 → 0.6692 (+0.49), proving Dy was model-limited, not data-limited
3. **All 12 force components ≥ 0.9848**, best across all baselines

## HGT Baseline (pending)

| Detail | Value |
|--------|-------|
| Model class | `HGTBaseline` |
| Conv type | `HGTConv` (typed multi-head attention, 4 heads) |
| Params | 744,279 |
| Job file | `remote_jobs/server_hgt_full.yaml` |
| Batch size | 8 (conservative) |
| Edge attr aware? | **No** — pure typed attention baseline |

### Research question

Does typed attention (HGT) outperform typed convolution (RGCN) on physics-heterogeneous graphs?

### Possible outcomes

| Scenario | Interpretation |
|----------|----------------|
| HGT > RGCN | Typed attention provides additional benefit on complex heterogeneous graphs |
| HGT ≈ RGCN | Typed convolution and typed attention are roughly equivalent for this task |
| HGT < RGCN | Attention overhead without sufficient data or type granularity may hurt |

# Stage 2-A Baseline Results Draft

> **Status:** In progress — MLP completed, GCN/GAT pending.
> **Dataset:** `processed/hetero_graph_dataset_v2`
> **Split:** `by_sample` (train=28,000 / val=3,500 / test=3,500)
> **Last updated:** 2026-06-20

## Results Table

| Method | Graph Type | Typed Message | Multi-scale | Physics Loss | Params | Best Epoch | Train Time | Test Disp R² | Test Force R² | Test Disp MAE | Test Force MAE | Combined RelMAE | Run Dir | Artifact | Status |
|--------|-----------|---------------|-------------|--------------|-------|-----------|------------|-------------|--------------|--------------|---------------|----------------|---------|----------|--------|
| MLP | none | no | no | no | 96,274 | 88 | 3412.9s (56.9min) | 0.8554 | 0.9824 | 0.000208 | 15972.99 | 0.0884 | `outputs/baselines/MLP/20260620051300/` | `remote_artifacts/server_mlp_full_20260620060955.tar.gz` | ✅ completed |
| GCN | homogeneous | no | no | no | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| GAT | homogeneous | no | no | no | — | — | — | — | — | — | — | — | — | — | ⏳ pending |

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

## Notes

- Stage 2-A will be **complete** when MLP + GCN + GAT all have verified artifact results.
- After Stage 2-A, proceed to Stage 2-B (RGCN, HGT, MeshGraphNet) or directly to Stage 3 (Ours-base) as per project plan.
- "Typed Message = no" means these baselines do not use relation-specific message passing — this is the key distinction from Stage 2-B and Stage 3 models.

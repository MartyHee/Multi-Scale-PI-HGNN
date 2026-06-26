# Stage 5: MS-PI-HGT — 实验计划

> 版本 v1.0 — 基于 Stage 4 Result Lock 后的 Stage 5 实验设计

---

## 1. 实验矩阵

### 1.1 主对照实验

| # | 模型 | Backbone | BC loss | Link loss | Purpose |
|:-:|:----|:---------|:-------:|:---------:|---------|
| 0 | MS-HGT gated | MS-HGT | ✗ | ✗ | Stage 4 best (baseline) |
| 1 | **MS-PI-HGT-BC** | MS-HGT | ✅ λ=0.08 | ✗ | BC loss 单独贡献 |
| 2 | **MS-PI-HGT-Link** | MS-HGT | ✗ | ✅ λ=0.002 | Link consistency 单独贡献 |
| 3 | **MS-PI-HGT** (full) | MS-HGT | ✅ λ=0.08 | ✅ λ=0.002 | Stage 5 完整模型 |

### 1.2 不作为主实验

| 项目 | 否决理由 |
|:-----|:---------|
| HGT + physics loss | HGT 已不是推荐 backbone，且无宏观模块 |
| Ours v1/v2 + physics loss | Backbone 不可比 |
| 多 seed 实验 | Stage 5 应先验证物理 loss 效果，再考虑稳定性 |
| 大规模超参搜索 | 先固定 λ 做一次验证实验 |
| Approximate beam equilibrium | 符号约定待确认，不做第一版 |
| Multi-task consistency | 数据不充分，做 future work |
| UQ | Stage 6 专属 |
| 新数据集构建 | 不必要 |

### 1.3 不纳入消融

| 项目 | 理由 |
|:-----|:------|
| BC loss 使用 y_disp 而不是 0 | 物理意义上是 BC，但仍以 label 为准——不是"消融"，是正确做法 |
| Link loss 使用 translation vs rotation | 第一版只做 translation，rotation 是 optional |

---

## 2. 统一实验配置

### 2.1 固定参数（与 Stage 4 相同）

| 参数 | 值 |
|:-----|:---:|
| 数据集 | `processed/hetero_graph_dataset_v2` |
| Split | `by_sample` (train=28,000 / val=3,500 / test=3,500) |
| Epochs | **200** |
| Batch size | 8 |
| Device | cuda |
| Seed | 42 |
| Eearly stop patience | **50** |
| LR | 0.001 → ReduceLROnPlateau (patience=15, factor=0.5) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Supervised loss | ℒ_disp + ℒ_force (MSE, λ=1.0 each) |

### 2.2 新增 loss 参数

| 参数 | 默认值 | 调优范围 |
|:-----|:------:|:--------:|
| `lambda_bc` | **0.08** 🆕 | [0.03, 0.16] |
| `lambda_link` | **0.002** 🆕 | [0.0007, 0.004] |
| `bc_dofs` | "translation" | ["translation", "translation+rotation"] |
| `link_dofs` | "translation" | ["translation", "translation+rotation"] |

### 2.3 实现方式

- Physics loss 作为 `train_baseline.py` 中的可选 loss 分支
- 不是单独的新训练脚本
- 通过 CLI flags 控制：`--lambda-bc 0.05 --lambda-link 0.005`
- BC loss 需要 `support_flags` → 确保 `batch["mesh_node"]` 包含该字段
- Link loss 需要 `structural_link` edge_index → 确保 batch 包含该类边

---

## 3. 实施前提

### 3.1 必须先做：Loss Scale Dry-Run

**文件：** `scripts/inspect_physics_loss_scale.py`

在执行任何训练前，运行 dry-run 确认：

- BC loss 的数值尺度（与 supervised loss 比较）
- Link loss 的数值尺度
- 非零 support_mask 的 batch 覆盖比例
- 非零 structural_link edge 的 batch 覆盖比例

### 3.2 必须先确认的数据细节

- `batch["mesh_node"].support_flags` 是否存在（NodeType.SUPPORT_FLAGS）
- support_flags 的阈值（> 0.5 与 == 1.0）
- structural_link edge_index 是否包含双向边
- dry-run 后 lambda 是否需要调整

---

## 4. 成功标准

### 4.1 Primary targets

| # | 指标 | 当前 (MS-HGT gated) | 目标 | 判断标准 |
|:-:|:-----|:-------------------:|:----:|:--------:|
| P1 | **Translation BC MAE** | 0.000242 | **< 0.000242** | 物理 loss 的基线目标 |
| P2 | **Translation BC MAE vs HGT** | 0.000242 | **≤ 0.000179** | 追上 HGT（理想接近 additive 0.000171） |
| P3 | **Disp R²** | 0.9952 | **≥ 0.994** | 不低于 gated 超过 0.001 |
| P4 | **Dy R²** | 0.9925 | **≥ 0.9905** | 不低于 gated 超过 0.002 |
| P5 | **RelMAE** | 0.0519 | **≤ 0.054** | 不显著恶化 |
| P6 | **Midspan Dy R²** | 0.9932 | **> 0.990** | 保持宏观效果 |

### 4.2 Secondary targets

| # | 指标 | 说明 |
|:-:|:-----|:------|
| S1 | **Structural link consistency residual** | 两端位移差下降 |
| S2 | **Force R²** | ≥ 0.992 |
| S3 | **P95 / P99 tail error** | 不恶化（各区域） |
| S4 | **Training curve stability** | 无 loss 权重导致过拟合或震荡 |
| S5 | **BC region per-component MAE** | Dx/Dy/Dz 均下降，尤其 Dz（当前 0.000492） |

### 4.3 如果 physics loss 无效 (contingency)

| 可能原因 | 缓解措施 |
|:---------|:---------|
| λ_bc 太小 | 增大 λ_bc（0.05 → 0.1） |
| λ_link 太小 | 增大 λ_link（0.005 → 0.01） |
| λ 太大压倒 supervised loss | 减小 λ 或 warm-up |
| BC loss 和 original loss 目标冲突 | 分析梯度冲突，考虑 dynamic weight |
| Link loss 数值几乎为零 | 检查 structural_link 边数量和两端位移差分布 |

---

## 5. 结果汇报模板

### 5.1 主表

```
| 模型 | BC loss | Link loss | BC MAE↓ | Disp R² | Dy R² | Force R² | RelMAE |
|:-----|:-------:|:---------:|:-------:|:-------:|:-----:|:--------:|:------:|
| MS-HGT gated | ✗ | ✗ | 0.000242 | 0.9952 | 0.9925 | 0.9928 | 0.0519 |
| MS-PI-HGT-BC | ✅ | ✗ | 0.0001XX | 0.99XX | 0.99XX | 0.99XX | 0.05XX |
| MS-PI-HGT-Link | ✗ | ✅ | 0.0001XX | 0.99XX | 0.99XX | 0.99XX | 0.05XX |
| MS-PI-HGT | ✅ | ✅ | 0.0001XX | 0.99XX | 0.99XX | 0.99XX | 0.05XX |
```

### 5.2 Per-DOF BC Residual

```
| DOF | HGT baseline | MS-HGT gated | MS-PI-HGT (best) | Δ |
|:---:|:-----------:|:------------:|:----------------:|:-:|
| Dx  | 0.000231    | 0.000213     |                  |   |
| Dy  | 0.000014    | 0.000007     |                  |   |
| Dz  | 0.000319    | 0.000492     |                  |   |
```

### 5.3 Link Consistency

```
| Metric | MS-HGT gated | MS-PI-HGT-Link | MS-PI-HGT (full) |
|:-------|:------------:|:--------------:|:----------------:|
| Mean link trans residual | | | |
| Max link trans residual | | | |
```

### 5.4 Conclusion

1. BC loss 是否有效降低 BC residual？
2. Link loss 是否改善连接区域一致性？
3. 联合训练是否比单独 loss 更优？
4. 物理损失是否降低了非 BC/非连接区域精度？
5. 下一步：进入 Stage 6 (UQ) 还是需要先迭代 physics loss？

---

## 6. 不纳入 Stage 5 的项目

| 项目 | 理由 |
|:-----|:------|
| ⛔ 完整有限元平衡 | 当前数据不可支持 |
| ⛔ 能量损失 | 当前数据不可支持 |
| ⛔ 本构一致性 | 当前数据不可支持 |
| ⛔ 板单元物理约束 | 当前数据不可支持 |
| ⛔ 连接件内力监督 | 无监督标签 |
| ⛔ Uncertainty quantification | Stage 6 专属 |
| ⛔ 数据集修改 | 不修改 `hetero_graph_dataset_v2` |
| ⛔ Macro anchor 结构感知 | Stage 4 v1 限制，非 Stage 5 内容 |

---

## 7. 文档版本

| 版本 | 日期 | 作者 | 修改说明 |
|:----:|:----:|:----:|:---------|
| v1.0 | 2026-06-26 | Claude Code | 初版 — Stage 5 实验计划，基于 Stage 4 Result Lock |

---

*本文档基于 Stage 4 Result Lock 和 Stage 5 Physics Loss Design 制定，是 Stage 5 实施的路线图。*

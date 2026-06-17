# Stage 2-A Baseline Suite Report

> **更新**：2026-06-17
> **阶段**：Stage 2-A — Baseline Model Suite（第一批：MLP、Homogeneous GCN、Homogeneous GAT）

---

## 1. 本阶段目标

搭建面向 `hetero_graph_dataset_v2` 的统一 baseline 训练与评估框架，实现并验证以下 3 个 baseline：

| Method | Graph Type | Typed Message | Multi-scale | Physics Loss | 说明 |
|--------|-----------|---------------|-------------|--------------|------|
| MLP | none | no | no | no | 非图 baseline |
| Homogeneous GCN | homogeneous | no | no | no | 同质图卷积 baseline |
| Homogeneous GAT | homogeneous | no (普通注意力) | no | no | 同质图注意力 baseline |

**本阶段不实现**：RGCN、HGT/HAN、MeshGraphNet-style、Ours-base、macro anchor、physics loss、uncertainty。

---

## 2. 数据集版本

仅使用 `processed/hetero_graph_dataset_v2`（canonical schema，3 节点类型 + 5 边类型）。

- **split_mode**：默认 `by_sample`（56/7/7 samples → 28000/3500/3500 图）
- **标准化**：`HeteroFeatureScaler`（Welford 在线算法，train-only stats）
- **标签**：`mesh_node.y_disp`（6 维位移）+ `beam_element.y_force`（12 维梁端内力）

---

## 3. 模型列表与结构说明

### 3.1 MLPBaseline

| 属性 | 说明 |
|------|------|
| 参数量 | **96,274** |
| 位移头 | `mesh_node.x`（15-dim）→ `[256, 128, 64]` MLP → 6-dim |
| 内力头 | `beam_element.x`（11-dim）⊕ mean endpoint mesh features（15-dim）→ 26-dim → `[256, 128, 64]` MLP → 12-dim |
| 端点特征 | 通过 `belongs_to_beam` edge_index 获取每个 beam 的两个端点 mesh_node 特征，`scatter_mean` 聚合 |
| 未参与 | `plate_element`、`structural_link` |

### 3.2 HomogeneousGCN

| 属性 | 说明 |
|------|------|
| 参数量 | **76,050** |
| 异构→同质 | `HeteroToHomoAdapter`：type-specific 线性投影 + type embedding → 统一 hidden_dim（128） |
| 边合并 | 全部 5 种边类型合并为单一 edge_index（含 per-type offset） |
| 图卷积 | 3 层 `GCNConv`（PyG 官方算子），hidden_dim=128 |
| 解码 | 按 type mask 取回 mesh_node / beam_element hidden states → `MLPHead`（[64, 32]） |
| 参考 | Kipf & Welling ICLR 2017；PyG `GCNConv` |

### 3.3 HomogeneousGAT

| 属性 | 说明 |
|------|------|
| 参数量 | **76,818** |
| 异构→同质 | 同 `HomogeneousGCN`（共享 `HeteroToHomoAdapter`） |
| 图注意力 | 3 层 `GATConv`（PyG 官方算子），第 1-2 层 4 头注意力，末层 1 头 |
| 注意力性质 | **普通节点级注意力**，不含 relation-type specific attention |
| 参考 | Veličković et al. ICLR 2018；PyG `GATConv` |

### 3.4 HeteroToHomoAdapter 设计

```
mesh_node.x (M, 15)  → Linear(15, 128) → + type_embed[0] → h_mesh (M, 128)
beam_element.x (B, 11) → Linear(11, 128) → + type_embed[1] → h_beam (B, 128)
plate_element.x (P, 6) → Linear(6, 128)  → + type_embed[2] → h_plate (P, 128)

h_all = concat([h_mesh, h_beam, h_plate])  # (M+B+P, 128)

Edge index offset:
  belongs_to_beam:       mesh → beam + M
  rev_belongs_to_beam:   beam + M → mesh
  belongs_to_plate:      mesh → plate + M + B
  rev_belongs_to_plate:  plate + M + B → mesh
  structural_link:       mesh → mesh (no offset)
  → edge_index_homo: (2, total_edges)
```

> **注意**：本 adapter 仅为 homogeneous baseline 服务，relation type 被完全忽略。后续 Ours 将使用 relation-specific message passing。

---

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| epochs | 100 |
| batch_size（MLP） | 64 |
| batch_size（GCN/GAT） | 8 |
| optimizer | AdamW（lr=1e-3, weight_decay=1e-4） |
| scheduler | ReduceLROnPlateau（patience=10, factor=0.5） |
| early stopping patience | 30 |
| loss | MSE（lambda_disp=1.0, lambda_force=1.0） |
| loss 计算空间 | 标准化空间（HeteroFeatureScaler 输出） |
| metrics 计算空间 | 原始物理空间（inverse transform） |
| 训练设备 | NVIDIA RTX 4070 Laptop GPU |

---

## 5. Smoke Test 结果

| 检查项 | MLP | GCN | GAT |
|--------|:---:|:---:|:---:|
| 前向传播 | ✅ | ✅ | ✅ |
| 反向传播 | ✅ | ✅ | ✅ |
| loss 下降 | ✅ (2.19→1.67) | ✅ (2.21→1.93) | ✅ (2.30→2.02) |
| 验证指标计算 | ✅ | ✅ | ✅ |
| test 评估 | ✅ | ✅ | ✅ |
| 模型保存 | ✅ | ✅ | ✅ |
| CSV 日志 | ✅ | ✅ | ✅ |
| 参数量 | 96,274 | 76,050 | 76,818 |
| 训练输出 shape | (M,6)+(B,12) | (M,6)+(B,12) | (M,6)+(B,12) |

**结论**：所有 3 个 baseline 的 forward/loss/backward/validation/test pipeline 均正确。

---

## 6. 正式训练结果

### 6.1 训练进度

| Model | Batch Size | Steps/Epoch | Epoch Time | Epochs Completed | Status |
|-------|-----------|-------------|-----------|-----------------|--------|
| MLP | 64 | 438 | ~300s | 正在运行 | 🟡 |
| GCN | 8 | 3500 | ~356s | 正在运行 | 🟡 |
| GAT | 8 | 3500 | ~380s | 正在运行 | 🟡 |

### 6.2 初始指标（Epoch 1）

| Model | Train Loss | Val Loss | D-R² | F-R² | D-MAE | F-MAE |
|-------|-----------:|---------:|-----:|-----:|------:|------:|
| **MLP** | 0.6621 | 0.3391 | 0.8218 | 0.8380 | 0.0003 | 43420.99 |
| **GCN** | 0.5027 | 0.2028 | 0.8413 | 0.9550 | 0.0003 | 28346.12 |
| **GAT** | ~0.80* | ~0.30* | ~0.80* | ~0.90* | - | - |

> *GAT 仍在 Epoch 1，数值为 batch 级估计。

### 6.3 最终指标（待训练完成后更新）

| Method | Graph Type | Typed Message | Multi-scale | Physics Loss | Disp R² | Force R² | RelMAE | Params | Time |
|--------|-----------|---------------|-------------|--------------|--------:|---------:|-------:|-------:|-----:|
| **MLP** | none | no | no | no | TBD | TBD | TBD | 96,274 | TBD |
| **GCN** | homogeneous | no | no | no | TBD | TBD | TBD | 76,050 | TBD |
| **GAT** | homogeneous | no (注意力) | no | no | TBD | TBD | TBD | 76,818 | TBD |

---

## 7. 新增/修改文件

### 7.1 新增文件

| 文件 | 说明 |
|------|------|
| `configs/train_baseline.yaml` | Baseline 训练配置 |
| `configs/models_baseline.yaml` | Baseline 模型配置 |
| `src/models/baselines/__init__.py` | 包入口 |
| `src/models/baselines/decoders.py` | `MLPHead` 通用回归头 |
| `src/models/baselines/hetero_to_homo_adapter.py` | 异构→同质适配器 |
| `src/models/baselines/mlp_baseline.py` | MLP baseline |
| `src/models/baselines/homogeneous_gcn.py` | Homogeneous GCN |
| `src/models/baselines/homogeneous_gat.py` | Homogeneous GAT |
| `src/trainers/losses.py` | 组合损失函数 |
| `src/trainers/baseline_trainer.py` | Baseline 统一训练器 |
| `train_baseline.py` | 统一训练入口 |
| `docs/stage2a_baseline_suite_report.md` | **本报告** |

### 7.2 修改文件

| 文件 | 操作 |
|------|------|
| `docs/development_log.md` | ✅ 追加本条记录 |

---

## 8. 框架复用与设计原则

### 8.1 复用现有模块

- `HeteroGraphDataset` — 直接复用
- `HeteroFeatureScaler` — 直接复用
- `EarlyStopping` — 直接复用
- `CSVLogger`、`plot_loss_curve`、`plot_metric_curve` — 直接复用

### 8.2 新框架设计亮点

- **统一 `BaselineTrainer`**：支持 dual-output 模型（pred_disp, pred_force），自动处理 inverse transform
- **`CombinedLoss`**：lambda_disp * loss_disp + lambda_force * loss_force
- **可扩展模型注册**：`build_model()` + `MODEL_NAMES_MAP`，新增 baseline 只需添加映射
- **CLI 参数覆盖**：`--batch-size`、`--epochs`、`--split-mode`、`--max-graphs`（smoke test）
- **`--summarise-only`**：可从已有实验目录生成阶段汇总表

---

## 9. 当前问题

1. **Force MAE 数值偏大**：F-MAE ~43420（MLP, epoch 1），主要因为力/力矩量纲差异大（N vs N·m），`RelMAE` 更有参考价值。
2. **GCN/GAT batch_size 受限**：同质图合并后节点数增加（3534/graph），batch_size=8 为 RTX 4070 的安全值。
3. **数据加载速度**：`num_workers=0`（Windows 兼容），28,000 图加载至内存需一定时间。

---

## 10. 是否可以进入 Stage 2-B

| 条件 | 状态 |
|------|:----:|
| MLP baseline 实现并通过 smoke test | ✅ |
| Homogeneous GCN baseline 实现并通过 smoke test | ✅ |
| Homogeneous GAT baseline 实现并通过 smoke test | ✅ |
| 统一训练框架可用 | ✅ |
| 输出目录结构符合规范 | ✅ |
| 存在汇总报告 | ✅ |
| 正式训练结果可用 | 🟡（训练中） |

**当前判断**：可以进入 Stage 2-B（RGCN/HeteroConv、HGT/HAN、MeshGraphNet-style baseline），但建议等 Stage 2-A 正式训练完成后确认最终指标再做决策。

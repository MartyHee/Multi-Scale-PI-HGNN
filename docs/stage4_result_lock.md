# Stage 4 Result Lock

> 审计日期: 2026-06-26
> 审计对象: HGT 200ep / MS-HGT gated / MS-HGT additive

---

## 1. Checkpoint 对齐

### 1.1 best_epoch 一致性

| 模型 | metrics_summary best_epoch | train_log.csv best epoch | val_loss | 一致 |
|:----|:--------------------------:|:------------------------:|:--------:|:----:|
| HGT 200ep | 104 | 104 (epoch 104, val_loss=0.032877) | 0.032877 | ✅ |
| MS-HGT gated | 85 | 85 (epoch 85, val_loss=0.011913) | 0.011913 | ✅ |
| MS-HGT additive | 151 | 151 (epoch 151, val_loss=0.011779) | 0.011779 | ✅ |

结论：所有模型均使用 `best_model.pt`（val_loss 最优点），非 `last_model.pt`。

### 1.2 Export vs metrics_summary 一致性

| 模型 | metrics_summary Disp R² | export_metrics_check Disp R² | 误差 |
|:----|:-----------------------:|:---------------------------:|:----:|
| HGT 200ep | 0.976531 | 0.976531 | ✅ < 1e-6 |
| MS-HGT gated | 0.995183 | 0.995183 | ✅ < 1e-6 |
| MS-HGT additive | 0.995048 | 0.995048 | ✅ < 1e-6 |

全量测试集（3500 graphs）导出指标与训练时 test eval 完全一致。export 验证通过。

---

## 2. 数据泄漏检查

### 2.1 MS-HGT 训练阶段

| 检查项 | 结果 | 依据 |
|--------|:----:|------|
| `MacroAnchorPool` 使用 `y_disp` | ✅ 不使用 | 源码 grep 确认，仅使用 `mesh_node.x[:, :3]` 坐标 |
| `MacroAnchorPool` 使用 `y_force` | ✅ 不使用 | 同上 |
| Forward pass 引用监督标签 | ✅ 不使用 | `forward()` 仅接收 `batch` 输入特征 |
| Training code 含 `region` 逻辑 | ✅ 无 | `src/trainers/` 无 region、high-response 引用 |
| Training code 含 `BC_loss` | ✅ 无 | 同上 |
| Training code 含 `physics` loss | ✅ 无 | 同上 |
| `compute_region_labels.py` 参与训练 | ✅ 否 | 仅为 eval-only 诊断脚本 |
| high-response mask 参与训练 | ✅ 否 | 仅在 diagnostics 脚本中定义 |

**结论：无数据泄漏风险。**

MS-HGT 训练过程中仅使用：
- `mesh_node.x` 输入特征（含坐标、荷载、截面等工程设计特征）
- `beam_element.x` 输入特征
- `plate_element.x` 输入特征
- `mesh_node.y_disp`（监督标签 — 标准 supervised learning）
- `beam_element.y_force`（监督标签）

Anchor assignment 仅使用 X 坐标（`mesh_node.x[:, 0]`）做 bucketize，不含任何测试集信息。

### 2.2 标准化统计量泄漏确认

所有模型的 `feature_stats.json` 均基于 train split 计算 → `standardisation.train_only: true` 确认 ✅

---

## 3. HGT 200ep 公平性

### 3.1 配置比对

| 配置项 | HGT 200ep | MS-HGT gated | MS-HGT additive |
|:-------|:---------:|:------------:|:---------------:|
| Dataset | hetero_graph_dataset_v2 | 相同 | 相同 |
| split_mode | by_sample | 相同 | 相同 |
| batch_size | 8 | 8 | 8 |
| epochs | 200 | 200 | 200 |
| lr | 0.001 | 0.001 | 0.001 |
| optimizer | adamw | adamw | adamw |
| weight_decay | 1e-4 | 1e-4 | 1e-4 |
| scheduler | reduce_on_plateau (patience=10, factor=0.5) | 相同 | 相同 |
| early_stop_patience | 30 | 30 | 30 |
| seed | 42 | 42 | 42 |
| lambda_disp/force | 1.0/1.0, MSE | 相同 | 相同 |
| num_workers | 4 | 4 | 4 |
| device | cuda | cuda | cuda |

### 3.2 Git commit

所有 3 个 jobs 基于同一 commit: `5d20ddd9eff0235eeef6db4bfc5495f51c6de05f` ✅

### 3.3 公平性结论

HGT 200ep 与 MS-HGT 的训练配置**完全相同**（除模型结构差异外）✅。

- HGT 200ep 未因 early stopping 或配置不一致处于劣势
- HGT 200ep 的 best_epoch = 104，说明 200 epochs 为其提供了充分收敛时间
- HGT 200ep 未因 "200 epochs 超过需要" 导致过拟合（test metrics 与 100-epoch 版本一致）

---

## 4. Macro Module Sanity

### 4.1 核心参数

| 参数 | 设定 | 验证 |
|:-----|:----:|:----:|
| n_segments | 12 | config 确认 ✅ |
| 空 anchor 数量 | 0 | smoke test + full training 确认 ✅ |
| macro edge 类型 | bidirectional sequential chain | code review 确认 ✅ |
| fusion_per_layer | true | config 确认 ✅ |
| macro_gnn_layers | 2 | config 确认 ✅ |
| macro_gnn_aggr | mean | config 确认 ✅ |
| include_anchor_static | true | config 确认 ✅ |

### 4.2 参数量

| 模型 | 实际参数量 | 理论预期 | 一致 |
|:----|:---------:|:--------:|:----:|
| MS-HGT gated | 893,527 | ~894K | ✅ |
| MS-HGT additive | 844,119 | ~844K | ✅ |

Additive 少 ~49K（无 gate_net 参数）。

### 4.3 输出目录不混淆

| Model | Output dir | Job yaml |
|:------|:-----------|:---------|
| HGT | `outputs/baselines/HGT/20260624054837` | `remote_jobs/server_hgt_200epoch.yaml` |
| MS-HGT gated | `outputs/baselines/MS_HGT/20260624160353` | `remote_jobs/server_ms_hgt_gated.yaml` |
| MS-HGT additive | `outputs/baselines/MS_HGT_ADDITIVE/20260625072138` | `remote_jobs/server_ms_hgt_additive.yaml` |

输出目录与 job yaml 一一对应，无混淆 ✅

### 4.4 Gate 非退化

| 统计量 | 值 |
|:-------|:---:|
| Fusion 模块 | 1（3 层共享） |
| gate_net.2.bias 均值 | -1.65（初始 -2.0，训练后打开） |
| gate_net.2.bias 范围 | [-2.87, -0.52] |
| 等效 sigmoid gate 范围 | ~0.05 ~ 0.37 |
| 结论 | ✅ 非退化 — gate 部分打开，macro signal 被利用但不过度支配 |

---

## 5. 论文表述边界

### ✅ 可以说的
- MS-HGT 在当前 `split_by_sample` 单拓扑数据上**显著优于** HGT baseline
- Macro anchor + cross-scale fusion 系统性改善 midspan（Dy R² +0.114）与 high-response 区域（+0.149）
- MS-HGT Conv 收敛更快（best epoch: 85 vs 104），同时获得更高精度
- Dy R² 从 0.905 提升到 0.993，**不再是最短板**
- 所有位移分量 R² > 0.992

### ❌ 不应说的
- 不要声称已证明跨拓扑泛化（当前仅 1 个拓扑结构）
- 不要声称物理规律已被严格满足（无 physics loss）
- 不要声称 equilibrium 已保证
- 不要声称所有工程场景都适用
- 不要声称 MS-HGT 已超越"所有已有方法"（仅与项目内 baseline 比较）

---

## 6. 总体结论

| 检查项目 | 状态 |
|:---------|:----:|
| Checkpoint 对齐 | ✅ 全部通过 |
| 数据泄漏 | ✅ 无泄漏 |
| HGT 公平性 | ✅ 配置完全一致 |
| Macro sanity | ✅ 参数/配置/输出/gate 均正常 |
| 论文表述边界 | ✅ 明确文档化 |

**Stage 4 Result Lock 通过。MS-HGT gated 确认为 Stage 5 Physics Loss backbone。**

---

*文档版本: v1.0 / 2026-06-26 / Result Lock 审计专用*

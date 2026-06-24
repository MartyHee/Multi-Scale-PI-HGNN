# Stage 4: MS-HGT — 实验计划

## 版本 v2.0

> **注意:** v1.0 中的 "MS-HGT w/o feedback" (实验 #2) 已被移除。如果 macro GNN 的输出不回流到 micro 节点、也无独立 auxiliary head，则 macro 子图自身无法产生可比较的预测输出，因此不能作为有效性能对照。macro 模块必须通过 cross-scale fusion 才能影响 micro 级别预测。

---

## 1. 实验矩阵

### 1.1 主对照实验

| # | 模型 | Macro | Macro GNN | Fusion | Purpose |
|:-:|:----:|:-----:|:---------:|:------:|---------|
| 1 | **HGT** (baseline) | **无** | — | — | 当前 best baseline (Disp=0.977, Dy=0.908) |
| 2 | **MS-HGT additive** | ✅ Geometric | SAGEConv 2层 | Additive fusion | 简单融合是否有效 |
| 3 | **MS-HGT gated** | ✅ Geometric | SAGEConv 2层 | **Gated residual fusion** | 推荐的完整 MS-HGT |

### 1.2 Ablation 实验

| # | 问题 | Variant | 对照 |
|:-:|------|---------|:----:|
| A1 | MacroGNN 深度重要性 | num_layers=1 vs 2 vs 3 | #3 |
| A2 | Anchor 数量影响 | n_segments=8, 10, 12, 16 | #3 |
| A3 | 融合位置 | 仅最后一层 vs 每层融合 | #3 |

**Ablation 执行优先级：** A2 > A3 > A1。A2（anchor 数量）直接影响设计决策，A3（融合位置）影响架构复杂度。

### 1.3 不纳入主实验的项目

| 项目 | 否决理由 |
|------|----------|
| Ours v1/v2 + macro | Backbone 已不如 RGCN |
| RGCN + macro | HGT 作为 backbone 严格更优 |
| **MS-HGT w/o feedback** | **Macro 不回流则无法影响 micro 预测，不能作为性能对照** |
| Structure-aware anchor | 先验证 geometric 有效再迭代 |
| Learned pooling anchor | 否决，见设计文档 |
| Physics loss | Stage 5 专属 |
| UQ | Stage 6 专属 |

---

## 2. 统一实验配置

### 2.1 固定参数

所有实验共享：

| 参数 | 值 |
|------|:---:|
| 数据集 | `processed/hetero_graph_dataset_v2` |
| Split | `by_sample` (train=28000, val=3500, test=3500) |
| Epochs | **200** (HGT 在 99 epoch 仍在改善，增加 epochs 为 macro 留出收敛时间) |
| Batch size | 8 |
| Device | cuda |
| Seed | 42 |
| Early stop patience | **50** (macro 可能需要更长收敛) |
| LR | 0.001 → ReduceLROnPlateau (patience=15, factor=0.5) |
| 优化器 | AdamW (wd=1e-4) |
| Loss | λ_disp=1.0, λ_force=1.0 (与 Stage 2/3 一致) |

### 2.2 MS-HGT 超参数

| 参数 | 默认值 | 调优范围 |
|:----:|:------:|:--------:|
| n_segments | 12 | [8, 10, 12, 16] |
| macro_hidden_dim | 128 (与 HGT 同) | 固定 |
| macro_gnn_layers | 2 | [1, 2, 3] |
| macro_gnn_aggr | mean | 固定 |
| fusion_method | gated_residual | [additive, gated_residual] |
| fusion_per_layer | true | [true, false] |
| macro_edge_type | sequential | 固定 |

### 2.3 Server 端配置 (remote_jobs 模板)

```yaml
model: "ms_hgt"
dataset: "processed/hetero_graph_dataset_v2"
split_mode: "by_sample"
epochs: 200
batch_size: 8
device: "cuda"
num_workers: 4
run_name: "server_ms_hgt_gated"
# model params passed via CLI or model config
```

---

## 3. 成功标准

### 3.1 HGT region-wise baseline (实测)

基于 `scripts/compute_region_labels.py` 在 HGT 测试集上的实际测量：

| 区域 | 节点数 | 占比 | Disp R² (macro) | Dy R² | MAE | Dy P95 AE |
|:----:|:------:|:----:|:---------------:|:-----:|:---:|:---------:|
| support | 28,000 | 0.8% | 0.9673 | 0.9764 | 0.000142 | 0.000121 |
| midspan | 1,232,000 | 33.3% | **0.9426** | **0.8821** | 0.000206 | 0.000380 |
| end_neighborhood | 420,000 | 11.4% | 0.9697 | 0.9382 | 0.000176 | 0.000327 |
| transition | 2,016,000 | 54.5% | 0.9774 | 0.9137 | 0.000168 | 0.000330 |

**高响应子集 (top 10%):**

| 子集 | N | Macro R² | Dy R² | 主要分布区域 |
|:---:|:-:|:--------:|:-----:|:-----------:|
| Translational top 10% | 369,602 | **0.6455** | 0.8471 | 99.3% 在 midspan |
| Dy top 10% | 369,600 | 0.9791 | 0.9381 | 31% midspan, 56% transition |

**关键发现：**
1. **Midspan 是系统性的弱项** — Disp R² 0.9426 低于整体 0.9769，Dy R² 0.8821 更是显著低于整体 0.9077
2. **高响应节点的 translational 预测是灾难性的** — 宏观 R² 仅 0.6455，且 99.3% 高响应节点集中在 midspan
3. **Dz 在 midspan 高响应集中区域 R² 为负** (-0.66)，说明 HGT 对大位移 Dz 预测接近失效
4. 支撑区（support）整体预测良好，BC residual 中 Dx 和 Dz 仍有小幅残差（~0.0003）

### 3.2 HGT baseline updated (实测)

基于 export_full_predictions 反标准化后验证：

| 指标 | HGT (本文) |
|:----:|:----------:|
| **Disp R² (macro avg)** | 0.976891 |
| **Dy R²** | 0.907669 |
| **Force R² (macro avg)** | 0.989136 |
| **Combined RelMAE** | 0.068337 |
| **Total params** | 744,279 |

### 3.3 Primary targets

| 指标 | HGT baseline | MS-HGT 目标 | 判断标准 |
|:----:|:-----------:|:----------:|:--------:|
| **Disp R² (macro avg)** | 0.9769 | **≥ 0.976** (不低于 baseline) | Δ ≥ -0.002 算持平 |
| **Dy R²** | 0.9077 | **> 0.910** (至少无退化) | Δ ≥ +0.005 算改善 |
| **Combined RelMAE** | 0.0683 | **< 0.068** | 下降低于 0.068 |
| **Force R²** | 0.9891 | **≥ 0.988** | 不显著下降 |
| **Midspan Disp R²** | **0.9426 (实测 baseline)** | **> 0.950** | Macro 的最关键目标 |
| **Midspan Dy R²** | **0.8821 (实测 baseline)** | **> 0.900** | Midspan 垂直位移关键在于 |
| **High-response trans R²** | **0.6455 (实测 baseline)** | **> 0.750** | 尾部误差改善 |
| **Support region error** | 0.000206 (实测) | **下降** | BC residual 改善 |

### 3.2 Secondary targets

| 指标 | 说明 |
|:----:|------|
| Training time ≤ HGT × 1.2 | Macro 模块不应显著增加训练时间 |
| Parameters ≤ 1.0M | 增量 ≤ 250K params |
| Gate stats 检查 | gate 激活分布 = 非退化（不全 0 / 全 1） |
| Anchor-wise 负载均衡 | 无空 anchor |
| 训练曲线稳定 | macro 模块不引入额外过拟合（对比 HGT val loss 曲线） |

### 3.3 If macro fails (contingency)

如果 MS-HGT gated 未达到上述任一 primary target，需要分析：

| 可能原因 | 缓解措施 |
|----------|----------|
| Anchor 数量不合适 | 调 n_segments ablation (A2) |
| Fusion 太弱 | 切换 additive (A3) |
| Macro GNN 深度不足 | 调 num_layers ablation (A1) |
| 所有 anchor 方案均无改善 | **重新评估研究假说** — macro anchor 在单一拓扑下可能不需要 |
| Gate 始终 ≈ 0 | macro signal 不被利用 → 需改进融合设计 |

---

## 4. 区域评估方案

### 4.1 Region label 构造

已实现 `scripts/compute_region_labels.py`，基于 NPZ 预测文件中保存的 `node_xyz`（坐标）和 `support_flags`（BC 标志）动态构造，不修改原始数据。

**Region 定义（优先级从高到低）：**

| ID | 名称 | 条件 | 实际节点数 |
|:--:|:----:|------|:----------:|
| 1 | **support** | 任一 BC DOF > 0.5 | 28,000 (0.8%) |
| 2 | **midspan** | X 位于中央 1/3 范围，非 support | 1,232,000 (33.3%) |
| 3 | **end_neighborhood** | 距离 X=0 或 X=max 在 5% 跨度内，非 support | 420,000 (11.4%) |
| 4 | **transition** | 上述条件之外 | 2,016,000 (54.5%) |
| 0 | **general** | 保留（当前数据中无匹配） | 0 (0%) |

**高相应子集（与 region 可重叠）：**

| 子集 | 条件 | 实际节点数 |
|:----:|------|:----------:|
| translational_high | sqrt(Dx²+Dy²+Dz²) 全局 top 10% | 369,602 (10.0%) |
| dy_high | |Dy| 全局 top 10% | 369,600 (10.0%) |

### 4.2 实现文件

`scripts/compute_region_labels.py` — 独立脚本，仅需 NPZ 文件，无需 dataset 或模型：

```bash
# HGT baseline region diagnostics
python scripts/compute_region_labels.py \\
    --pred-dir outputs/predictions/stage2b/hgt/<timestamp> \\
    --model-name HGT \\
    --output-dir outputs/diagnostics/stage4_region_baseline

# 对比模式（HGT vs future MS-HGT）
python scripts/compute_region_labels.py \\
    --pred-dir outputs/predictions/stage2b/hgt/<timestamp> \\
    --pred-dir-2 outputs/predictions/stage4/ms_hgt/<timestamp> \\
    --model-name HGT --model-name-2 MS-HGT \\
    --output-dir outputs/diagnostics/stage4_region_baseline
```

输出文件：

| 文件 | 内容 |
|------|------|
| `region_baseline_metrics.json` | 所有计算结果汇总 |
| `region_disp_metrics.json` | 分区域 Disp 详细指标 |
| `support_bc_residual.json` | BC 约束残差（Dx/Dy/Dz 的 MAE） |
| `high_response_metrics.json` | 高响应子集指标与区域分布 |
| `region_map.png` | 区域分配空间可视化（侧视+俯视） |
| `region_disp_r2_bar.png` | 分区域 Disp R² 柱状图 |
| `region_dy_r2_bar.png` | 分区域 Dy R² 柱状图 |
| `region_baseline_report.txt` | 人类可读摘要报告 |

### 4.3 实现状态

| 阶段 | 状态 |
|:----:|:----:|
| 设计 region label 方案 | ✅ 已完成 |
| 实现 `scripts/compute_region_labels.py` | ✅ 已完成 |
| 运行 HGT region-wise baseline 测量 | ✅ 已完成 |
| MS-HGT full training 后对比 | ⏳ 待 MS-HGT 训练完成 |

---

## 5. 实现时间线

### Phase 1：Design ✅（已完成）

- [x] Stage 4 design spec (stage4_macro_anchor_design.md)
- [x] Stage 4 experiment plan (stage4_experiment_plan.md)
- [x] Region label 脚本与 HGT baseline 测量 (scripts/compute_region_labels.py)
- [x] 更新 development_log.md

### Phase 2：Model Implementation（下一任务）

预期文件修改：

| 文件 | 操作 | 预计行数 |
|------|:----:|:--------:|
| `src/models/baselines/ms_hgt.py` | **NEW** | ~350 |
| `src/models/baselines/__init__.py` | MODIFY | +2 lines |
| `configs/models_baseline.yaml` | MODIFY | ~20 lines |
| `train_baseline.py` | MODIFY | ~15 lines |
| `scripts/export_full_predictions.py` | MODIFY | ~10 lines |

### Phase 3：Smoke Test

- 本地 2 graphs × 1 epoch
- 检查 forward shapes, loss, backward, no NaN/Inf, checkpoint

### Phase 4：Server Full Training（用户执行）

- `remote_jobs/server_ms_hgt_gated.yaml` (primary, 实验 #3)
- `remote_jobs/server_ms_hgt_additive.yaml` (ablation, 实验 #2)
- `remote_jobs/server_hgt_200epoch.yaml` (HGT re-run with 200 epochs)
- ~~`remote_jobs/server_ms_hgt_nofeedback.yaml`~~ **已移除**（w/o feedback 不能作为性能对照）

### Phase 5：Analysis

- 主指标对比 (Disp/Force R², Dy, RelMAE)
- Region-wise metrics（复用 `scripts/compute_region_labels.py`）
- High-response tail error
- Gate stats 验证
- 汇总到实验报告文档

---

## 6. 最终汇报模板

MS-HGT full training 完成后应汇报：

```
## MS-HGT Full Training 结果

### 实验对比
| 指标 | HGT baseline | MS-HGT additive | MS-HGT gated | Δ (gated) |
|------|:----------:|:---------------:|:------------:|:---------:|

### Per-component Disp R²
| Component | HGT | MS-HGT | Δ |
|:---------:|:---:|:------:|:-:|

### Region-wise Disp R²（实测 HGT baseline）
| Region | HGT baseline | MS-HGT additive | MS-HGT gated | Δ (gated) |
|:------:|:-----------:|:---------------:|:------------:|:---------:|
| support | 0.9673 | | | |
| midspan | 0.9426 | | | |
| end_neighborhood | 0.9697 | | | |
| transition | 0.9774 | | | |

### High-Response Subset Disp R²（实测 HGT baseline）
| Subset | HGT baseline | MS-HGT gated | Δ |
|:------:|:-----------:|:-------------:|:-:|
| translational top 10% | 0.6455 | | |
| Dy top 10% | 0.9791 | | |

### Conclusion
1. Macro anchor 是否有效？midspan Dy 是否改善？
2. 哪种 fusion 最优（additive vs gated）？
3. 高响应尾部误差是否有显著改善？
4. 下一步：进入 Stage 5 (physics loss) 还是需要先优化 macro？
```

---

## 7. 文档版本

| 版本 | 日期 | 作者 | 修改说明 |
|:----:|:----:|:----:|----------|
| v1.0 | 2026-06-24 | Claude Code | 初版 — 实验计划 |
| v2.0 | 2026-06-24 | Claude Code | 移除 w/o feedback 实验；添加实测 HGT region-wise baseline；更新 region 标签实现细节 |

---

*本文档为 Stage 4 实验计划，定义了实验矩阵、成功标准、区域评估方案、实现时间线和汇报模板。请先审阅设计规格书 (`stage4_macro_anchor_design.md`)，确认后进入实现阶段。*

# Stage 5: Physics Loss Design

> 版本 v1.0 — 设计规格书（非实现文档）

---

## 1. 命名建议

**论文主线命名：MS-PI-HGT**

全称：*Multi-Scale Physics-Informed Heterogeneous Graph Transformer for Steel Truss Girder Surrogate Modeling*

- "MS" = Multi-Scale（macro anchor + micro message passing）
- "PI" = Physics-Informed（当前 Stage 5 加入 physics-informed loss）
- "HGT" = Heterogeneous Graph Transformer（backbone）

| 阶段 | 模型命名 | 论文含义 |
|:----|:---------|:---------|
| Stage 4 | MS-HGT | 多尺度异构图 transformer（无物理约束） |
| Stage 5 | **MS-PI-HGT** | 多尺度物理信息异构图 transformer |
| Stage 6 | MS-PI-HGT + UQ | 含不确定性量化的完整模型 |

不建议使用 "MS-HGT + Physics Loss" 作为论文命名，后者更适合实验表格内标记。

---

## 2. Backbone：MS-HGT gated

| 属性 | 值 |
|:-----|:----|
| 模型 | MSHGTBaseline |
| Fusion | gated_residual（收敛更快，midspan 略优） |
| checkpoint | `outputs/baselines/MS_HGT/20260624160353/best_model.pt` |
| 参数量 | 893,527 |
| 需修改模型 | **否** — physics loss 仅作为 loss 项叠加，不改变模型结构 |

---

## 3. 候选 Physics Loss

### 3.1 Support BC Loss（第一版必做）

#### 动机
MS-HGT gated 的 Translation BC MAE = 0.000242，高于 HGT 的 0.000179。BC 约束是结构分析中最基本的物理边界条件，正则化 BC 区域可以改善支座附近预测的物理一致性。

#### 定义

给定 batch 中所有 `support_flags > 0.5` 的 mesh_node，取约束自由度上的预测位移与真实位移差：

```
ℒ_BC = mean( (pred_disp[support_mask, constrained_dofs] - y_disp[support_mask, constrained_dofs])² )
```

其中 `support_flags` 为 `(N, 6)` 的 one-hot 约束标志（Dx, Dy, Dz, Rx, Ry, Rz），`> 0.5` 表示该自由度被约束。

#### 细节

- Translation 与 rotation **分开记录**，不混合平均
- TensorBoard 中分别记录 `loss/bc_trans` 和 `loss/bc_rot`
- 第一版**仅使用 translation BC loss**（Dx, Dy, Dz）
- 以 `y_disp` 为 ground truth，**不强行约束到 0**（真实约束位移可能非 0，取决于坐标系与处理）
- 如果某 batch 中 support 节点极少（`sum(support_mask) = 0`），设 `ℒ_BC = 0` 并记录 warning

#### 预期效果

- Translation BC MAE ≤ MS-HGT gated 0.000242
- Translation BC MAE ≤ HGT 0.000179（理想接近 additive 的 0.000171）
- 不影响 midspan 等非 BC 区域的预测精度

---

### 3.2 Structural Link Translation Consistency Loss（第一版必做）

#### 动机

当前数据中所有 `structural_link` 为 **RIGID**。刚性连接要求两端节点的位移一致，但当前模型没有显式利用这一物理约束。加入 link consistency loss 可以在不改变模型结构的前提下，增强连接区域的预测一致性。

#### 定义

给定 `structural_link` 边 `(i, j)`，对两端 mesh_node 的 translational displacement 作差：

```
ℒ_link = mean( || pred_disp[i, :3] - pred_disp[j, :3] ||² )
```

其中网格中每条 `structural_link` 边参与计算，方向无关（i ↔ j 对称）。

#### 细节

- 第一版仅约束 translation 3 DOF（Dx, Dy, Dz）
- rotation 3 DOF（Rx, Ry, Rz）作为 optional 保留，暂不加入
- **在 batch 层面计算**：从 batch 的 `structural_link` edge_index 中提取所有 (i, j) 对
- 如果某 batch 中 `structural_link` 边数量为 0，设 `ℒ_link = 0` 并记录 warning
- 注意：当前 edge_index 是双向的（one per direction），需确保只计算一次或双向都计算（结果一致）
- TensorBoard 中记录 `loss/link_trans`

#### 预期效果

- Structural link 两端节点位移差下降
- 间接改善连接区预测平滑性
- 不影响无关区域的预测

---

### 3.3 Approximate Beam Equilibrium Loss（第一版不建议实现）

#### 设计思路（参考）

对每个梁单元，预测内力应满足：

```
F_I + F_J ≈ 0   （轴向平衡）
M_I + M_J + F × L ≈ 0   （力矩平衡）
```

但由于当前数据的约束条件、荷载分布、坐标系等细节，不能简单地将"内力绝对平衡"作为损失。需要更仔细的设计。

#### 为什么第一版不做

- 梁端内力是反符号还是同号取决于输出定义（当前输出的 12 维向量中 I 端和 J 端力的正负号约定需要确认）
- 当前数据不支持精确的分布荷载积分
- 如果不小心引入错误符号约定，反而会恶化预测
- 该 Loss 需要先通过数据分析验证符号约定和平衡程度

#### 状态

- **写设计，不实现**
- 作为 Stage 5 的 optional 扩展，或在论文中作为 "future work"
- 如果引入，只称 "approximate beam consistency"，不称 "complete equilibrium"

---

### 3.4 Multi-Task Consistency Regularization（第一版不建议实现）

#### 设计思路（参考）

位移与内力的关系（通过刚度矩阵）是结构分析的核心，但当前数据：
- 没有完整的刚度矩阵
- 没有板单元内力监督
- 没有连接件内力监督

因此，"位移-力的交叉一致性"在当前数据能力下**不充分**，强行引入可能引入系统性偏差。

#### 状态

- **仅记录设计思路**
- 不作为 Stage 5 第一版必做
- 可作为后续补充实验

---

## 4. Loss 组合与总损失

### 4.1 总损失形式

```
ℒ_total = ℒ_supervised + λ_bc × ℒ_BC + λ_link × ℒ_link
```

其中 `ℒ_supervised` 保持 Stage 2-4 的原有定义：

```
ℒ_supervised = ℒ_disp + ℒ_force
= MSE(pred_disp, y_disp) + MSE(pred_force, y_force)
```

### 4.2 Loss 权重建议（初次）

从 loss scale dry-run 获取实际数值后调整，初始建议：

| Loss | 初始 λ | 调优范围 | 来源 |
|:-----|:------:|:--------:|:----|
| BC translation | **0.08** 🆕 | [0.03, 0.16] | Loss scale dry-run: BC/supervised ratio=32% |
| Link translation | **0.002** 🆕 | [0.0007, 0.004] | Loss scale dry-run: Link/supervised ratio=1458% |

### 4.3 第一版实验组合

| 实验 | BC loss | Link loss | 目的 |
|:----|:-------:|:---------:|:-----|
| BC-only | ✅ λ=0.08 | — | BC loss 单独贡献 |
| Link-only | — | ✅ λ=0.002 | link consistency 单独贡献 |
| BC+Link | ✅ λ=0.08 | ✅ λ=0.002 | Stage 5 完整推荐 |

---

## 5. 实现前必需步骤

### 5.1 先实现 loss scale dry-run 脚本

**文件：** `scripts/inspect_physics_loss_scale.py`

功能：

```
1. 加载 MS-HGT gated best_model.pt
2. 取少量 train batch（~4 graphs）+ val batch（~4 graphs）
3. 计算：
   - supervised disp_loss（数值 + 尺度）
   - supervised force_loss（数值 + 尺度）
   - BC translation loss（数值 + 尺度 + 非零 support mask 统计）
   - Link translation loss（数值 + 尺度 + 非零 edge count 统计）
4. 输出 loss scale 对比表
5. 推荐 λ_bc / λ_link 初始值
```

### 5.2 必须确认的符号和假设

在实现 BC loss 前，确认：

- `batch["mesh_node"].support_flags` 的存在与 shape
- `support_flags` 的阈值（>0.5 还是 ==1.0）
- 约束 DOF 的索引顺序与 `y_disp` 一致

在实现 Link loss 前，确认：

- `batch[("mesh_node", "structural_link", "mesh_node")].edge_index` 的 shape 与含义
- 边是否为双向（edge_index 每列一对）
- 刚性连接两端节点是否应在所有 3 个 translation DOF 上一致

### 5.3 不修改模型结构

- Physics loss 不应要求模型修改 forward pass 或添加新 head
- Physics loss 应作为 loss function 的额外项，与 `loss_fn_disp` / `loss_fn_force` 并列
- 不修改 MSHGTBaseline 类
- 不修改 MacroAnchorPool / MacroGNN / CrossScaleFusion

---

## 6. 模型结构不变性

Stage 5 不应改变以下内容：

| 项目 | 不变 | 理由 |
|:-----|:----:|:-----|
| MSHGTBaseline forward | ✅ | Physics loss 不依赖模型结构变更 |
| MacroAnchorPool | ✅ | 无新信息需要 anchor 捕获 |
| MacroGNN | ✅ | 无新信息需要传递 |
| CrossScaleFusion | ✅ | Fusion 机制不需改变 |
| Decoders | ✅ | 输出维度不变 |
| 数据集 | ✅ | `hetero_graph_dataset_v2` 不变 |
| Split | ✅ | `by_sample` 不变 |
| Training loop 主体 | ✅ | 仅 loss 计算增加分支 |

---

## 7. 论文表述边界（Stage 5）

### ✅ 可以说
- 引入 **支持边界条件约束**（support boundary condition loss）作为物理正则
- 引入 **刚性连接一致性正则**（rigid link consistency regularization）改善连接区域预测一致性
- Physics-informed loss 在保持精度的同时降低物理残差
- 物理约束提升模型在约束区域的预测可信度

### ❌ 不应说
- 不要声称已实现完整有限元平衡
- 不要声称已满足所有物理守恒律
- 不要声称本构关系已被强制执行
- 不要声称模型可以外推到未见拓扑或极端荷载

---

## 8. 数据能力边界（再次确认）

| 物理约束 | 当前数据能力 | Stage 5 可实现 |
|:---------|:-----------:|:--------------:|
| 支座 BC 约束 | ✅ support_flags + y_disp | ✅ BC loss |
| 刚性连接一致性 | ✅ structural_link edge | ✅ Link loss |
| 梁平衡 | ⚠️ 需确认符号约定 | ❌ 第一版不做 |
| 板单元内力 | ❌ 无监督标签 | ❌ |
| 连接件内力 | ❌ 无监督标签 | ❌ |
| 完整刚度矩阵 | ❌ 不可恢复 | ❌ |
| 本构一致性 | ❌ | ❌ |
| 弱形式能量 | ❌ | ❌ |

---

## 9. 实施路径

```
Phase 1: Loss scale dry-run（scripts/inspect_physics_loss_scale.py）
   → 确认 BC loss / Link loss 数值尺度
   → 推荐 λ_bc / λ_link

Phase 2: 实现 BC loss（src/losses/ 或 inline in trainer）
   → 验证 BC MAE 下降
   → 确保不影响非 BC 区域精度

Phase 3: 实现 Link loss
   → 验证 link consistency residual 下降

Phase 4: BC + Link 联合训练
   → Stage 5 完整实验

Phase 5（optional）: Approximate beam consistency
   → 先数据分析确认符号约定
   → 再考虑是否实现
```

---

*文档版本: v1.0 / 2026-06-26 / Stage 5 Physics Loss 设计规格书*

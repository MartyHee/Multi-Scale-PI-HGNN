# 数据结构说明 — 面向 Multi-Scale PI-HGNN 模型开发

> **更新**：2026-06-16 · v2 schema 修正

## 数据位置

- **训练数据（主线）**：`D:\※CREC\BiShe\S1\raw_data\GraphTrainingData2\`
- **旧数据（已停止使用）**：`D:\※CREC\BiShe\S1\raw_data\GraphTrainingData\`（43 样本，仅 5 表）
- **详细分析**：`docs/graph_training_data2_analysis_report.md`
- **Schema 修正报告**：`docs/hetero_dataset_schema_fix_report.md`

---

## 当前主线数据使用规范

### 数据版本

| 版本 | 数据集 | 状态 | 说明 |
|------|--------|------|------|
| v1 (已废弃) | `processed/hetero_graph_dataset_v1` | ❌ 已废弃 | 含 link_element 孤立节点 |
| **v2 (当前主线)** | **`processed/hetero_graph_dataset_v2`** | ✅ **当前主线** | 修正后 canonical schema |

正式训练数据只使用 **GraphTrainingData2**。旧数据 GraphTrainingData 不再进入主线。

---

## 正式节点类型（3 类）

| 节点类型 | 来源 | 数量 | 输入特征 | 监督标签 |
|----------|------|------|----------|----------|
| `mesh_node` | nodes + nodal_loads + general_supports | 1056 | 15 维 [X,Y,Z,Fx..Mz,Dx_fix..Rz_fix] | ✅ **位移 [Dx..Rz] 6 维** |
| `beam_element` | beam_elements + beam_sections + materials | 1646 | 11 维 [Area,Ix,Iy,Iz,E,ν,γ,length,cosX,cosY,cosZ] | ✅ **内力 [Fx_I..Mz_J] 12 维** |
| `plate_element` | plate_elements + thicknesses + materials | 832 | 6 维 [Thickness,E,ν,γ,BetaAngle,PlateType] | ❌ 板内力/应力缺失 |

> **重要**：`link_element` **不是正式节点类型**。`rigid_elastic_links.csv` 被建模为 mesh_node ↔ mesh_node 的 interaction edge（structural_link）。

---

## 正式边类型（5 类）

| 边类型 (source, relation, target) | 数量 | 边特征 | 说明 |
|-----------------------------------|------|--------|------|
| mesh_node → **belongs_to_beam** → beam_element | 3292 | endpoint_type (I/J) | 梁单元 membership |
| beam_element → **rev_belongs_to_beam** → mesh_node | 3292 | endpoint_type (I/J) | 反向边（带 edge_attr）|
| mesh_node → **belongs_to_plate** → plate_element | 3328 | corner_type (I/J/K/L) | 板单元 membership |
| plate_element → **rev_belongs_to_plate** → mesh_node | 3328 | corner_type (I/J/K/L) | 反向边（带 edge_attr）|
| mesh_node → **structural_link** → mesh_node | 132 | 10 维刚度+类型+is_rigid | Interaction edge |

### structural_link 边特征字段（10 维）

| 索引 | 字段 | 说明 |
|------|------|------|
| 0-5 | Kx, Ky, Kz, Krx, Kry, Krz | 六自由度刚度 |
| 6 | BetaAngle | β 角 |
| 7 | DistanceRatio | 距离比 |
| 8 | ElasticLinkType | 连接类型编码（当前全部为 1=RIGID）|
| 9 | is_rigid | 是否为刚性连接（当前全部为 1.0）|

---

## `rigid_elastic_links.csv` 的正式图角色

- **Interaction edge**（开题报告中的 Interaction Link Edge）
- **边类型**：`('mesh_node', 'structural_link', 'mesh_node')`
- **不是** link_element 节点
- 当前数据全部为 **RIGID**（K=1e18）
- 未来若有 elastic link，仍通过同一边类型表达，通过 edge_attr 中的 ElasticLinkType 和 is_rigid 区分
- 当前无连接件内力标签

---

## 当前可用标签

| 标签 | 维度 | 每工况记录数 | 可用性 |
|------|------|-------------|--------|
| 梁端内力 (Fx/Fy/Fz/Mx/My/Mz) × I/J | 12 | 3292 | ✅ |
| 节点位移 (Dx/Dy/Dz/Rx/Ry/Rz) | 6 | 1056 | ✅ |

---

## 当前缺失标签

| 缺失项 | 说明 | 备注 |
|--------|------|------|
| plate_element 内力/应力 | 板单元无监督标签 | 需与数据方确认 |
| structural_link 连接件内力 | 连接件无监督标签 | 若补充，优先作为 edge-level label |
| 支座反力 | 无法构建节点平衡损失 | 低优先级 |

---

## 未来标签扩展原则

如果未来补充连接件内力数据：
- 优先作为 **structural_link edge-level label**（`structural_link.y`）
- 不应新增 `link_element` 节点类型
- 不应逆转 v2 schema 的修正

---

## 当前可用特征统计

| 特征作用域 | 维度 | 统计来源 |
|-----------|------|---------|
| mesh_node.x | 15 | train split only (Welford 在线算法) |
| beam_element.x | 11 | train split only |
| plate_element.x | 6 | train split only |
| belongs_to_beam.edge_attr | 1 | train split only |
| rev_belongs_to_beam.edge_attr | 1 | train split only |
| belongs_to_plate.edge_attr | 1 | train split only |
| rev_belongs_to_plate.edge_attr | 1 | train split only |
| structural_link.edge_attr | 10 | train split only |
| mesh_node.y_disp | 6 | train split only |
| beam_element.y_force | 12 | train split only |

---

## 重要说明

1. **所有 70 个样本拓扑完全一致**（同一结构，变截面/变荷载）
2. 新增 6 张表跨样本完全一致（materials, plate_elements, thicknesses, general_supports, rigid_elastic_links）
3. `rigid_elastic_links` 全部为 **刚性连接**（RIGID, K=1e18），无弹性连接
4. **旧数据 GraphTrainingData 不再使用**
5. **link_element 不是正式节点类型** — 已从 v2 schema 中移除
6. 研究计划参考 `D:\※CREC\BiShe\S1\CLAUDE.md`，不要脱离主线研究任务

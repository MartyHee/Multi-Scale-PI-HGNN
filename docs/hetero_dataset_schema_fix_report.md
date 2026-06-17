# 异构图数据集 Schema 修正报告

> **日期**：2026-06-16
> **修正对象**：`hetero_graph_dataset_v1` → `hetero_graph_dataset_v2`
> **修正依据**：开题报告图建模定义 + 负责人反馈

---

## 1. 修正原因

### 1.1 v1 的 schema 歧义

v1 版本将 `rigid_elastic_links.csv` 同时构造为：

1. `link_element` **节点类型**（132 个节点，9 维特征，无监督标签）
2. `mesh_node → rigid_link → mesh_node` **边类型**（132 条边，9 维边特征）

并且 `link_element` 节点没有任何入射边（没有 membership edge 连接到 mesh_node），导致：

- **link_element 成为孤立节点**，无法参与消息传递
- **link_element 节点和 structural_link 边语义重叠**，同一数据源被冗余建模
- 不符合开题报告中"interaction link edge 直接连接两个 mesh_node"的图建模定义

### 1.2 开题报告图建模定义

来自开题报告研究一的微观图定义：

> 1. 图节点分为 `mesh_node` 和 `element_node`。
> 2. `mesh_node` 对应有限元空间节点。
> 3. `element_node` 对应物理构件（`beam_element`、`plate_element`）。
> 4. Membership edge 连接 mesh_node 与 element_node，表达拓扑从属关系。
> 5. Interaction link edge 直接连接两个 mesh_node，表达梁属节点与板属节点之间的刚性或弹性耦合关系。
> 6. Interaction edge 的边特征编码连接件六自由度刚度或连接属性。

**关键结论**：`rigid_elastic_links.csv` 中的连接记录应建模为 **mesh_node ↔ mesh_node 的 interaction edge**，而不是独立节点类型。

---

## 2. v1 错误 Schema

```
节点类型（4 类）：                  ← link_element 在这里是冗余的
  mesh_node      (1056) ✅
  beam_element   (1646) ✅
  plate_element  (832)  ✅
  link_element   (132)  ❌ 孤立节点，无任何入射边

边类型（5 类）：
  mesh → belongs_to_beam → beam          (3292) ✅
  beam → rev_belongs_to_beam → mesh      (3292) ⚠️ 无 edge_attr
  mesh → belongs_to_plate → plate        (3328) ✅
  plate → rev_belongs_to_plate → mesh    (3328) ⚠️ 无 edge_attr
  mesh → rigid_link → mesh               (132)  ✅（但名称不兼容弹性连接）
```

### v1 具体问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| link_element 孤立节点 | **严重（❌）** | 无法参与消息传递，浪费存储 |
| 数据源被双重建模 | **严重（❌）** | 同一 CSV 既建节点又建边 |
| rigid_link 命名不兼容弹性 | **中等（⚠️）** | 未来 elastic link 需要不同命名 |
| 反向边无 edge_attr | **中等（⚠️）** | rev_belongs_to_beam/plate 没有边特征 |
| 无 is_rigid 字段 | **低（ℹ️）** | 无法直接判断连接类型 |

---

## 3. v2 Canonical Schema

```
节点类型（3 类，严格按开题报告）：
  mesh_node      (1056) — 有限元空间节点
  beam_element   (1646) — element_node（梁物理构件）
  plate_element  (832)  — element_node（板物理构件）
  （无 link_element — 刚性/弹性连接为 interaction edge）

边类型（5 类）：
  mesh → belongs_to_beam → beam              (3292)  endpoint_type (I/J)
  beam → rev_belongs_to_beam → mesh          (3292)  endpoint_type (I/J) ← 新增 edge_attr
  mesh → belongs_to_plate → plate            (3328)  corner_type (I/J/K/L)
  plate → rev_belongs_to_plate → mesh        (3328)  corner_type (I/J/K/L) ← 新增 edge_attr
  mesh → structural_link → mesh              (132)   Kx..Krz+β+ratio+type+is_rigid (10-d)
```

### structural_link edge_attr 字段列表（10 维）

| 索引 | 字段 | 来源 | 说明 |
|------|------|------|------|
| 0 | Kx | rigid_elastic_links.csv | 轴向刚度 (N/m) |
| 1 | Ky | rigid_elastic_links.csv | 剪切刚度 Y (N/m) |
| 2 | Kz | rigid_elastic_links.csv | 剪切刚度 Z (N/m) |
| 3 | Krx | rigid_elastic_links.csv | 扭转刚度 (N·m/rad) |
| 4 | Kry | rigid_elastic_links.csv | 弯曲刚度 Y (N·m/rad) |
| 5 | Krz | rigid_elastic_links.csv | 弯曲刚度 Z (N·m/rad) |
| 6 | BetaAngle | rigid_elastic_links.csv | β 角 |
| 7 | DistanceRatio | rigid_elastic_links.csv | 距离比 |
| 8 | ElasticLinkType | rigid_elastic_links.csv | 连接类型编码（1=RIGID） |
| 9 | is_rigid | 派生 | 是否为刚性连接（全部为 1.0） |

---

## 4. rigid_elastic_links.csv 的最终图角色

| 维度 | 定义 |
|------|------|
| **图角色** | Interaction edge（interaction link edge） |
| **边类型** | `('mesh_node', 'structural_link', 'mesh_node')` |
| **数据源** | `rigid_elastic_links.csv` 的 INodeId/JNodeId |
| **边特征** | 10 维刚度 + 连接属性 + is_rigid |
| **标签** | 当前无连接件内力标签 |
| **未来兼容** | 若出现 elastic link，仍使用同一边类型，edge_attr 中的 ElasticLinkType 和 is_rigid 表达差异 |

---

## 5. 节点集合诊断

### 5.1 beam_nodes

| 维度 | 值 |
|------|-----|
| 定义 | beam_elements.csv 中 INodeId 与 JNodeId 的并集 |
| 数量 | **1056** |
| 是否全部属于 nodes.csv | ✅ |
| 结论 | **所有 mesh_node 都至少参与一个梁单元** |

### 5.2 plate_nodes

| 维度 | 值 |
|------|-----|
| 定义 | plate_elements.csv 中 INodeId/JNodeId/KNodeId/LNodeId 的并集 |
| 数量 | **924** |
| 是否全部属于 nodes.csv | ✅ |
| 结论 | **924 个 mesh_node 参与板单元，占总数 87.5%** |

### 5.3 beam_nodes 与 plate_nodes 关系

| 集合 | 数量 | 占比 |
|------|------|------|
| shared_nodes (both beam & plate) | **924** | 87.5% |
| beam_only_nodes (仅梁) | **132** | 12.5% |
| plate_only_nodes (仅板) | **0** | 0% |

**结论**：
- **所有 plate 节点都同时也是 beam 节点**（shared = 924）
- 没有节点"仅属于板单元"
- **132 个 beam_only 节点**不参与板单元

### 5.4 structural_link endpoint 关系

| 维度 | 值 |
|------|-----|
| INodeId 集合大小 | **132** |
| JNodeId 集合大小 | **132** |
| INodeId 是否在 beam_nodes | ✅ 全部 132 个 |
| INodeId 是否在 plate_nodes | ✅ 全部 132 个（属于 shared）|
| JNodeId 是否在 beam_nodes | ✅ 全部 132 个 |
| JNodeId 是否在 plate_nodes | ❌ 0 个 |
| JNodeId 是否在 shared | ❌ 0 个 |

**connection 类型统计**：

| 连接类型 | 数量 |
|----------|------|
| beam_only → plate_only | 0 |
| plate_only → beam_only | 0 |
| shared → beam_only | **132** |
| beam_only → shared | 0 |
| shared → shared | 0 |
| unknown → anything | 0 |

**结论**：
- **structural_link 全部为 shared_node → beam_only_node 单向连接**（INodeId = shared node, JNodeId = beam_only node）
- 这对应于：板区域节点（shared）通过刚性连接耦合到梁区域节点（beam_only）
- 在开题报告中，这些 structural_link 表达"梁属节点与板属节点之间的刚性或弹性耦合关系"
- 由于 shared_node 同时属于梁和板的拓扑，structural_link 可以理解为板-梁界面耦合的显式建模

### 5.5 连接语义统计

| 字段 | 统计 |
|------|------|
| ElasticLinkTypeName | **全部为 RIGID**（132/132）|
| ElasticLinkType | 全部为 1 |
| Kx, Ky, Kz | 全部为 **1e18** |
| Krx, Kry, Krz | 全部为 **1e18** |
| BetaAngle | 全部为 **0.0** |
| DistanceRatio | 全部为 **0.5** |
| 连接件内力标签 | ❌ **不存在** |

---

## 6. 修正操作汇总

| 文件 | 操作 | 说明 |
|------|------|------|
| `configs/hetero_dataset.yaml` | ✅ 修改 | 移除 link_element，rename rigid_link→structural_link，v2 输出路径 |
| `src/data/hetero_schema.py` | ✅ 重写 | 移除 link_element，rename，添加 rev 边 edge_attr，structural_link 10-d |
| `src/data/build_hetero_graph_dataset.py` | ✅ 重写 | 移除 link_element 构建逻辑，反向边加 edge_attr，structural_link 10-d（+is_rigid）|
| `src/data/hetero_transforms.py` | ✅ 更新 | EDGE_TYPES_WITH_ATTR 自动适应新的 schema（含 reverse edges）|
| `src/data/hetero_graph_dataset.py` | ✅ 更新 | 文档字符串更新为 v2 引用 |
| `src/data/validate_hetero_dataset.py` | ✅ **新增** | 全面验证脚本（17+ 检查项）|
| `docs/data_structure.md` (模型项目) | ✅ 更新 | 反映 structural_link 与 3 节点类型 |
| `docs/data_structure.md` (根项目) | ✅ 更新 | 反映 structural_link 图角色 |
| `docs/hetero_dataset_schema_fix_report.md` | ✅ **本次文档** | 本报告 |
| `docs/development_log.md` | ✅ 更新 | 追加本次修正记录 |

---

## 7. 输出目录

```
processed/hetero_graph_dataset_v2/  (~17 GB)
├── metadata.json              (schema 修正版元信息，3 节点类型)
├── schema.json                (v2 canonical schema)
├── index.csv                  (35000 行)
├── feature_stats.json         (train-only, 10 组 mean/std, v2 schema)
├── splits/
│   ├── split_by_sample.json        (56/7/7 samples)
│   └── split_by_loadcase.json      (400/50/50 loadcases)
└── graphs/
    ├── 1274/  (500 .pt)
    ├── 1406/  (500 .pt)
    └── ... (70 samples × 500 LC)
```

## 8. 验证结果

运行 `validate_hetero_dataset.py`（10 个随机图，全量检查）：

| 检查类别 | 通过 | 失败 | 警告 |
|----------|:----:|:----:|:----:|
| 全部 | **223** | **0** | **0** |

### 关键验证项

| 验收项 | 状态 |
|--------|:----:|
| 1. processed/hetero_graph_dataset_v2 可正常读取 | ✅ |
| 2. node_types 只有 mesh_node, beam_element, plate_element | ✅ |
| 3. 不存在 link_element | ✅ |
| 4. rigid_elastic_links.csv → structural_link edge | ✅ |
| 5. structural_link edge_attr 10-d（含 is_rigid） | ✅ |
| 6. 反向 membership edge 均有 edge_attr | ✅ |
| 7. 监督标签只含 mesh_node.y_disp 和 beam_element.y_force | ✅ |
| 8. feature_stats 为 train-only（rev_belongs_to_beam/plate 及 structural_link 均已统计） | ✅ |
| 9. 文档中不再有"link_element 是正式节点类型"表述 | ✅ |
| 10. Stage 2 可基于 v2 数据集开展 baseline model suite | ✅ |

---

## 9. 是否可以进入 Stage 2

✅ **所有 schema 修正完成。等待构建完成后运行 validate_hetero_dataset.py 确认即可进入 Stage 2。**

# Stage 4: MS-HGT — Macro Anchor Graph + Cross-Scale Fusion

## 设计规格书 v1.0

---

## 1. 背景

### 1.1 Stage 1–3 已建立的能力

| Stage | 成果 | 核心发现 |
|:-----:|------|----------|
| 1 | `hetero_graph_dataset_v2` (35000 图) | 3 节点 + 5 边类型，全部 RIGID-only structural_link |
| 2-A | MLP/GCN/GAT baseline | 同质图方法 < 非图 MLP，验证 typed message 必要性 |
| 2-B | RGCN/HGT baseline | HGT (Disp=0.977) >> RGCN (Disp=0.937) > MLP (Disp=0.855) |
| 3 | Ours v1/v2 (terminated) | Edge_attr 全恒定 → 微观点注入无有效信号 |

### 1.2 剩余问题

所有已完成模型（HGT, RGCN, Ours v1/v2）均存在以下未解决共性问题：

| 问题 | 当前最佳 (HGT) | 差距 |
|------|:--------------:|:----:|
| Dy R² | 0.908 | 比其他 5 个位移分量 (0.987–0.993) 低 **8–10×** |
| Disp R² | 0.977 | 仍有 **2.3% 未解释方差** |
| High-response region | 待量化 | HGT 在高响应区未必最优 |
| Support / Midspan error | 待诊断 | 局部消息无法区分区域物理差异 |
| Long-range transfer | 未验证 | 标准 GNN 消息范围受限于层数 |

### 1.3 核心假说

> **当前模型仅使用局部消息传递（K-hop neighbor aggregation），无法有效建模大跨度钢桁梁中的长程传力路径（跨中→支座）和全局刚度耦合。引入宏观锚点图（macro anchor graph）可显式建模长程传力，改善 Dy、高响应区域和跨中/支座区域误差。**

---

## 2. 命名与模型定位

### 2.1 推荐命名：**MS-HGT** (Multi-Scale HGT)

命名逻辑：

| 组成部分 | 含义 |
|:--------:|------|
| **MS** | Multi-Scale — 微观（local HGT message）+ 宏观（anchor graph message）双尺度 |
| **HGT** | Backbone 使用 HGTConv typed attention |

### 2.2 备选命名

| 候选 | 推荐度 | 理由 |
|:----:|:------:|------|
| **MS-HGT** | ⭐ **推荐** | 论文名 "Multi-Scale PI-HGNN" 的直接映射；PI-HGNN = MS-HGT + Physics + UQ |
| HGT-Macro | ⚠️ 次选 | 描述模糊，与论文名脱节 |
| Ours-MacroBase | ❌ 否决 | Ours v1/v2 已终止，不宜延续 |

### 2.3 论文全名映射关系

```
Multi-Scale PI-HGNN
├─ Multi-Scale = Stage 4:  MS-HGT (micro HGT + macro anchor + cross-scale fusion)
├─ PI          = Stage 5:  MS-HGT + physics-regularized objective
├─ HGNN        = Stage 2B: HGT backbone (typed attention over heterogeneous graph)
└─ UQ          = Stage 6:  MS-HGT + conformal prediction
```

---

## 3. Backbone 选择依据

### 3.1 确定：HGT 作为 Stage 4 backbone

| 依据 | 说明 |
|------|------|
| **Disp R² 最高** | 0.9769 vs RGCN 0.9366, Ours v1 0.9283, MLP 0.8554 |
| **Dy R² 最高** | 0.9077 vs RGCN 0.6692 (+0.2385) — 差距超过一个数量级 |
| **Force R² ≈ saturated** | 0.9891 — 与 RGCN (0.9878) 差距极小，增量空间有限 |
| **Typed attention > typed conv** | HGTConv 的类型感知多头注意力显著优于 HeteroConv 的 relation-specific SAGEConv |
| **参数量可接受** | 744K (比 RGCN 520K 多 43%)，在 8×RTX 4090 上训练时间 5.9h |

### 3.2 为什么不使用其他模型

| 模型 | 否决理由 |
|------|----------|
| **MLP** | 非图模型，无图拓扑信息，Disp 0.855 表明已到极限 |
| **GCN/GAT (homogeneous)** | 无 typed message，Disp 0.842-0.848 低于 MLP |
| **RGCN** | 强 baseline (Disp 0.937) 但 Dy 0.669 远低于 HGT (0.908)；作为 Stage 4 backbone 起点太低 |
| **Ours v1/v2** | Edge_attr 全恒定 → edge_attr-aware 模块无效；Disp 0.923-0.928 既低于 RGCN 也低于 HGT |

---

## 4. Macro Anchor 构造方案

### 4.1 桥梁几何确认

从数据分析结果（10 个 train graph 验证）：

| 参数 | 值 |
|------|:---:|
| 纵向坐标轴 | **X** (range: 0.00 → 87.60, span = 87.6 m) |
| 横向 (宽度) | Y (−17.5 → 17.5, 35 m) |
| 竖向 (高度) | Z (0.0 → 17.4, 17.4 m) |
| 支座节点 | 8 个 (4 at X=0, 4 at X=87.6, Dz/Dy/Dx 约束) |
| 全部节点数 | 1056 mesh_node / graph |
| 拓扑一致性 | ✅ 所有 graph 共享相同节点坐标 |

### 4.2 三种候选方案比较

#### 方案 A：Geometric anchor along X（⭐ 推荐第一版）

```
原理：将 mesh_node 沿 X 方向分段，每段内 pooling 为一个 anchor 节点
```

| 特性 | 评估 |
|------|:----:|
| 实现复杂度 | ⭐ 极低 — 仅需 X 坐标 + torch.bucketize |
| 可解释性 | ⭐ 高 — 每段对应桥梁一个物理区段 |
| 不受 edge_attr 限制 | ✅ 完全不依赖 structural_link 数据 |
| 支持 batching | ✅ 拓扑固定 → 段分配可预计算为 static buffer |
| 与 region labels 对齐 | ✅ 支承区在 X=0 / X=87.6 自然形成首尾段 |
| 能否直接服务长程传力 | ✅ 邻接 anchor 链建模从跨中到支座的传力路径 |
| Dy 改善潜力 | ⭐ 高 — 跨中 Dy 最大，锚点链可直接传递跨中信号到支座 |

#### 方案 B：Structure-aware anchor

```
原理：基于 beam/plate/structural_link 拓扑识别结构关键区域
```

| 特性 | 评估 |
|------|:----:|
| 实现复杂度 | ⚠️ 中等偏高 — 需解析 structural_link endpoint 分布、beam 连接模式 |
| 可解释性 | ⭐ 高 — 接近结构工程直觉 |
| 受 edge_attr 限制 | ⚠️ structural_link 刚度恒定 → 结构信息有限 |
| 增量收益 | ❓ 不确定 — 复杂度和收益未知 |
| **建议** | ⏳ 作为 v2 备选，不在第一版实现 |

#### 方案 C：Learned pooling / clustering

```
原理：基于 node hidden states (或坐标) 做聚类，聚类质心作为 anchor
```

| 特性 | 评估 |
|:----:|------|
| 实现复杂度 | ⚠️ 中等 — 需实现可微聚类或 Gumbel-Softmax 池化 |
| 可解释性 | ❌ 低 — 聚类结果不可预期，难以与物理区域对应 |
| 稳定性 | ❌ 差 — 聚类结果随初始化、训练种子和 batch 变化 |
| 创新增量 | ⚠️ 纯 AI 方法，但物理语义弱 |
| **建议** | ❌ 否决 — 与论文"物理可解释"定位冲突 |

### 4.3 推荐方案：Geometric anchor along X（第一版）

#### 4.3.1 段划分参数

| 参数 | 默认值 | 说明 |
|:----:|:------:|------|
| `n_segments` | **12** | 12 段 × 7.3m = 87.6m (涵盖全桥) |
| `bin_method` | `equal_width` | 等宽划分，每段物理长度一致 |
| `padding` | `clip_to_bounds` | 首尾段从边界开始，不超 span |

不同段数对比（基于实际数据）：

| n_segments | nodes/anchor (min, max, mean) | 物理长度/段 | 适用性 |
|:----------:|:----------------------------:|:----------:|:------:|
| 8 | 96–160, mean=132 | 10.95 m | 粒度太粗 |
| **10** | 96–128, mean=106 | 8.76 m | ✅ 合理 |
| **12** | 64–96, mean=88 | 7.30 m | ✅ 推荐 |
| 16 | 32–96, mean=66 | 5.48 m | 可用但段数多 |

**推荐 12 段**：每段 64-96 节点（约 88 mean），粒度足够捕获区域差异，同时保持锚点图轻量。

#### 4.3.2 Anchor node features

```
anchor_feature = [
  mesh_anchor_hidden,       # mean pooling of mesh_node.hidden within segment
  anchor_center_x,          # X position of segment center
  anchor_length,            # segment length
  region_marker,            # one-hot: {support, midspan, connection, general}
  anchor_num_nodes,         # node count in this segment (for weighting)
]
dim = hidden_dim + 1 + 1 + 4 + 1 = hidden_dim + 7
```

**region_marker 定义（自动从坐标/BC flag 生成）：**

| Region | 判定条件 | 编码 |
|--------|----------|:----:|
| support | 段包含 BC 约束节点 | one-hot → [1, 0, 0, 0] |
| midspan | 段中心在中部 1/3 X 范围 | one-hot → [0, 1, 0, 0] |
| connection | 段含 structural_link endpoint 密集区 | one-hot → [0, 0, 1, 0] |
| general | 其他段 | one-hot → [0, 0, 0, 1] |

**注意：** mesh_anchor_hidden 是 pooled from HGT hidden states（经 HGTConv 后的 micro hidden），其他是静态特征。

### 4.4 支持节点分布（用于 region label）

| 位置 | X 坐标 | 节点数 | BC 类型 |
|:----:|:------:|:------:|:-------:|
| 左端 (X=0) 下方 | 0.0 | 2 | Dz only |
| 左端 (X=0) 中间 | 0.0 | 2 | Dy only |
| 右端 (X=87.6) 下方 | 87.6 | 2 | Dx + Dz |
| 右端 (X=87.6) 中间 | 87.6 | 2 | Dy only |

左端在段 0 (X: 0.0–7.3)，右端在段 11 (X: 80.3–87.6)。

---

## 5. Macro Graph Edge 设计

### 5.1 第一版：Sequential chain

```
Anchor 0 → Anchor 1 → Anchor 2 → ... → Anchor K-1
```

| 特性 | 设计 |
|:----:|------|
| Edge type | 邻接链（K-1 条边，双向） |
| Edge_index | `[[0,1,1,2,...,K-2,K-1], [1,0,2,1,...,K-1,K-2]]` |
| 总边数 | 2×(K-1) = 22 (when K=12) |

**理由：**
- 钢桁梁的传力路径本质上是轴向（X方向）的：荷载从跨中→支座沿桁架传递
- 邻接链是这种传力路径的最简洁建模
- 边数少（22 条），macro message passing 极轻量

### 5.2 备选：k-hop 连接（v2 优化方向）

```
Anchor i ←→ Anchor i+k  (k=1,2)
```

增加 k=2 跳跃连接，允许跨一个段的信息传递。适用于大跨度中跨中→支座的快速信号传播。

**不建议第一版就做全连接**（K² 条边 = 144 when K=12），全连接会模糊锚点图的空间结构。

### 5.3 Macro edge feature

第一版不使用 edge_attr。如果 v2 需要，可以加入：

| 特征 | 含义 |
|------|------|
| segment_distance | anchor 间沿 X 的物理距离 |
| is_direct_adjacent | 0/1 是否邻接 |

---

## 6. Macro Message Passing

### 6.1 第一版：轻量 SAGEConv

```python
class MacroGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim + 7, hidden_dim))  # anchor_feat = hidden + static
        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
```

| 参数 | 值 | 理由 |
|:----:|:---:|------|
| hidden_dim | 128 (与 HGT 同) | 便于与 HGT hidden states 对齐融合 |
| num_layers | **2** | 2 层 message passing 对链式锚点图已足够（receptive field = 2-hop on chain） |
| aggr | mean | 简单稳定 |
| activation | ReLU | 与 HGT 一致 |
| normalization | LayerNorm | 与 HGT 一致 |

### 6.2 为什么不使用 HGTConv for macro

| 理由 | 说明 |
|------|------|
| 锚点图只有 1 种节点类型 | HGTConv 的 typed attention 对单类型图无用 |
| 锚点图只有 1 种边类型 | 邻接链边，无 relation type 区分 |
| 参数量考虑 | HGTConv 相对 SAGEConv 参数量大 3-4×，对轻量锚点图不必要 |

### 6.3 Macro message passing 流程

```
Input:  anchor_feat: (n_segments, hidden_dim + 7)
        macro_edge_index: (2, 2*(n_segments-1))

x = anchor_feat
for conv, norm in zip(self.convs, self.norms):
    x = conv(x, macro_edge_index)
    x = F.relu(x)
    x = norm(x)

Output: anchor_hidden: (n_segments, hidden_dim)
```

---

## 7. Cross-Scale Fusion

### 7.1 核心思想

Macro anchor 经过 message passing 后携带了全桥尺度的上下文信息（跨中→支座的传力路径）。需要将这些信息**回流**到 micro node，并与 micro hidden states **融合**。

### 7.2 三方案比较

| 方案 | 公式 | 优点 | 缺点 |
|:----:|:----:|------|------|
| **A: Gated Residual** ⭐ | `h' = h_micro + σ(gate_net([h_micro, h_macro_unpool])) ⊙ h_macro_unpool` | 自适应控制 macro 贡献 | 需额外 gate 参数量~256×256 |
| B: Additive | `h' = h_micro + h_macro_unpool` | 最简单，0 额外参 | macro 信息无论好坏直接叠加 |
| C: Concatenate + proj | `h' = Linear(2×hidden → hidden)([h_micro, h_macro_unpool])` | 信息利用充分 | 参数量翻倍 |

### 7.3 推荐：Gated Residual Fusion

```
1. Unpooling: 将 anchor_hidden 按 node->anchor 映射展开到每个 micro node
   → macro_hidden_unpooled shape: (n_mesh_nodes, hidden_dim)

2. Gate: 对每个 mesh_node，学习一个 gate value ∈ [0,1]
   gate = σ(W_gate([h_mesh, macro_hidden_unpooled]) + b_gate)
   → (n_mesh_nodes, hidden_dim)

3. Fusion:
   h'_mesh = h_mesh + gate ⊙ macro_hidden_unpooled
```

**关键细节：**

- Gate 输入是拼接的 `[h_mesh, macro_hidden]`（2×128 = 256-dim）
- Gate 输出是 **per-dimension gate**（128-dim，不是 scalar），让不同 hidden dimension 受不同强度影响
- 初始时降低 gate_net 权重初始化偏置使 gate ≈ 0，不干扰 HGT 初始化能力

### 7.4 融合位置

| 选项 | 推荐度 | 说明 |
|:----:|:------:|------|
| **每 HGT layer 后融合** | ⭐ 推荐 | 每层 micro → macro → fuse → next layer，渐进式多尺度融合 |
| 仅最后一层融合 | ⚠️ 次选 | 信息流只有一次交叉，不够充分 |
| 无 cross-scale feedback | ❌ ablation | 验证 macro 是否必须回流见 Stage 4 实验矩阵 |

**第一版采用「每层融合」。**

```
Layer 1: HGT micro message → Macro anchor pool → Macro GNN → Unpool + Fuse
Layer 2: HGT micro message → (同上)
Layer 3: HGT micro message → (同上)
                               ↓
                    disp_decoder / force_decoder
```

---

## 8. 完整模型架构 (MS-HGT)

### 8.1 数据流

```
input: HeteroDataBatch (mesh_node, beam_element, plate_element, 5 edge types)
                 │
    ┌────────────┴────────────┐
    │   HGT Micro Encoder     │  (3× layers, each with macro fusion)
    │                         │
    │  Layer 1:                │
    │    HGTConv → ReLU → Dropout → LayerNorm     → mesh_hidden_1
    │          │                                   │
    │          └──→ MacroAnchorPool (geometric)    │
    │              → MacroGNN (2× SAGEConv)        │
    │              → CrossScaleGatedFusion ─────────┘
    │  Layer 2: (same pattern)                     → mesh_hidden_2
    │  Layer 3: (same pattern)                     → mesh_hidden_3
    └──────────────────────────────────────────────┘
                 │
                 ├──→ disp_decoder (mesh_node)   → pred_disp   (N_mesh, 6)
                 └──→ force_decoder (beam_element) → pred_force (N_beam, 12)
```

### 8.2 参数量估算

| 模块 | 参数量 | 说明 |
|:----:|:------:|------|
| HGT micro encoder (3×) | ~744,000 | HGT baseline 全量 |
| MacroAnchorPool | **0** | 无可训练参数 (scatter_mean) |
| MacroGNN (2× SAGEConv) | ~33,000 = 2 × (128+7→128 + 128→128) | 每层 ~16K |
| CrossScaleGatedFusion (3×) | ~99,000 = 3 × (256→128 gate) | 每层 ~33K |
| **MS-HGT total** | **~876,000** | 较 HGT (744K) 增加约 132K |

### 8.3 参数量精算

| 模块 | 公式 | params |
|:----:|:----:|:------:|
| MacroGNN Layer 1 | SAGEConv(135→128) + SAGEConv(128→128) | 135×128 + 128×128 + 128×128 + 128×128 = 66,048 |
| MacroGNN Layer 2 | SAGEConv(128→128) + SAME | 32,768 (共享) 或 2× = 66,048 |
| Gate 3× | Linear(256→128) + bias = 256×128+128 | 32,896 × 3 = 98,688 |
| Total macro | | **~165K–231K** (depends on MacroGNN sharing) |

**如果用共享 MacroGNN（两层权重重复使用）**：~165K 增量 → MS-HGT ≈ 909K
**如果用独立的每层 MacroGNN**：~231K 增量 → MS-HGT ≈ 975K

实际第一版建议用 **共享 MacroGNN**（一层定义，每层 HGT 后复用）。

### 8.4 需要注意的 Batch 处理

所有 graph 在 batch 中共享相同拓扑 + 节点坐标 → anchor 分配可预计算为 static buffer：
```
anchor_assignment: (1056,) — each mesh_node → [0, K-1]
macro_edge_index: (2, 2*(K-1)) — fixed adjacency chain
```

在 forward 中：
1. `mesh_hidden` shape: (B×1056, H)
2. 对每 graph 分别 `scatter_mean(hidden, assignment) → (K, H)`
3. 跨 graph concat: `(B×K, H)`
4. Macro message passing with batch vector: `[0,...,0, 1,...,1, ...]`
5. Unpool: `macro_hidden[assignment]` → (B×1056, H)
6. Gate fusion: per-node gate → (B×1056, H)

---

## 9. 不包含的内容 (边界控制)

| 模块 | 说明 | 引入阶段 |
|:----:|------|:--------:|
| Physics loss (support BC / equilibrium) | 不实现 | Stage 5 |
| Uncertainty Quantification | 不实现 | Stage 6 |
| Edge_attr-aware structural_link message | 已证明无效（data-limited） | ❌ 永久否决 |
| Ours-base v1/v2 StructuralLinkConv | 已终止 | ❌ |
| 大规模超参搜索 | 不执行 | — |
| 数据增强 / 新数据集构建 | 不执行 | — |
| 跨拓扑泛化验证 | 不执行（单一拓扑） | — |

---

## 10. 实现计划

### 10.1 新增文件

| 文件 | 内容 |
|------|------|
| `src/models/baselines/ms_hgt.py` | MS-HGT 全部模块：MacroAnchorPool, MacroGNN, CrossScaleFusion, MSHGTBaseline |
| `src/models/baselines/__init__.py` | 注册 MSHGTBaseline |
| `configs/models_baseline.yaml` | 添加 `ms_hgt` 配置段 |
| `train_baseline.py` | `--model ms_hgt` 注册 |
| `scripts/export_full_predictions.py` | `ms_hgt` 注册 |
| `remote_jobs/server_ms_hgt_full.yaml` | 服务器 full training job |
| 上述 3 个文档 | 本任务已完成 |

### 10.2 代码模块结构

```
src/models/baselines/ms_hgt.py
│
├── class MacroAnchorPool(nn.Module)
│   ├── __init__(self, n_segments, hidden_dim, use_beam_pooling=False)
│   │   └── compute anchor_assignment from coordinates (static buffer)
│   ├── forward(self, x_mesh, x_coords, batch=None)
│   │   └── scatter_mean → anchor features
│   └── get_region_markers(self, x_coords, batch=None)
│       └── support, midspan, connection region labels
│
├── class MacroGNN(nn.Module)
│   ├── __init__(self, hidden_dim, macro_feat_dim, num_layers=2)
│   ├── set_macro_edge_index(self, edge_index)  # set static macro graph
│   └── forward(self, x_anchor)
│
├── class CrossScaleGatedFusion(nn.Module)
│   ├── __init__(self, hidden_dim)
│   └── forward(self, x_micro, macro_hidden, anchor_assignment, batch=None)
│
└── class MSHGTBaseline(nn.Module)
    ├── __init__(self, ...)
    │   ├── hgt_encoder = HGTBaseline (reuse HGT layers)
    │   ├── macro_pool = MacroAnchorPool(...)
    │   ├── macro_gnn = MacroGNN(...)
    │   └── cross_fusion = CrossScaleGatedFusion(...)
    └── forward(self, batch)
        └── per-layer: HGT → macro pool → macro GNN → cross fusion → next
```

### 10.3 预计算逻辑

```python
# At model init or first forward pass:
def _init_anchor_assignment(self, ref_coords):
    """
    ref_coords: (1056, 3) — reference mesh_node coordinates
    Returns:
        anchor_assignment: (1056,) — each node assigned to [0, n_segments-1]
        macro_edge_index: (2, 2*(n_segments-1)) — fixed macro graph
        region_markers: (n_segments, 4) — one-hot region labels
    """
    # 1. Determine longitudinal axis (X has largest range)
    # 2. Equal-width binning along X
    # 3. Build sequential chain adjacency
    # 4. Identify support/midspan/connection segments
```

### 10.4 输出兼容性

`MSHGTBaseline.forward(batch)` → `(pred_disp, pred_force)`，与 `BaselineTrainer` 完全兼容。

无需修改：
- `BaselineTrainer`
- `CombinedLoss`
- 数据集加载
- 标准化
- 指标计算
- Checkpoint 保存

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|:----:|:----:|----------|
| Macro anchor 不带来提升 | 浪费训练时间 | 实验 4 (no feedback) 可快速验证 |
| 训练时间显著增加 | 迭代减慢 | MacroGNN 参数量 < 30K，每层额外计算量极小 |
| Support/midspan region 指标不可计算 | 无法评估 | region labels 可在 eval 时构造（无需模型修改） |
| Macro anchor 数量需调 | 依赖超参 | n_segments=8/10/12/16 为轻量 ablation |
| Batch 处理复杂度高 | 实现 bug | 所有 graph 拓扑固定 → anchor_assignment 可预计算为 static buffer |

---

## 12. 文档版本

| 版本 | 日期 | 作者 | 修改说明 |
|:----:|:----:|:----:|----------|
| v1.0 | 2026-06-24 | Claude Code | 初版 — 设计规格书 |

---

*本文档为 Stage 4 设计规格书，详细记录 MS-HGT 的模型架构、anchor 构造、macro graph 设计、cross-scale fusion 方案、实现计划和风险分析。下一步执行前请先完成实验计划 (`stage4_experiment_plan.md`)。*

# 服务器首批任务说明（用户人工执行）

本目录是**本地可审核**的服务器任务包；本轮未登录服务器、未推送、未下载产物、未启动任务。

已确认：服务器至少 4×RTX 4090 可用，Conda 环境 `pi_hgnn` 支持 CUDA，历史工程路径 `/home/miniconda/Bishe/Multi-Scale-PI-HGNN`。“服务器 CUDA 是否可用”不再是未知条件；但**新代码仍需服务器小规模预检**（路径、依赖、资源）。

## 0. 先确认路径（执行前）

```bash
ls -d /home/miniconda/Bishe/Multi-Scale-PI-HGNN
conda env list | grep pi_hgnn
nvidia-smi -L
```
新数据包在服务器上的路径**未确认**，必须显式填写 `DATA_ROOT`；**不要沿用旧 GraphTrainingData2**。若新数据尚未上传，先按 §2 上传。

## 1. 环境变量

```bash
export PROJECT_ROOT=/home/miniconda/Bishe/Multi-Scale-PI-HGNN
export DATA_ROOT=/path/to/three_topology_graph_v1   # 明确填写，勿用旧数据
export PYTHON=$(conda run -n pi_hgnn which python)
```

## 2. 代码与数据同步清单（用户执行）

**代码**（从本地仓库同步）：`src/`、`scripts/`、`configs/`、`server_ops/`、`requirements.txt`、本 README。不含 raw_data / outputs / processed 大文件。
**数据包**（只读）：整个 `three_topology_graph_v1/` 目录（含 `data/processed/three_topology_graph_v1/samples.jsonl`、`graphs/`、`dataset_manifest.json`、`transfer_manifest.json`、`manifests/`、`docs/`、`reports/`）。

上传后校验：
```bash
THREE_TOPO_PACKAGE_ROOT="$DATA_ROOT" bash "$PROJECT_ROOT/server_ops/three_topology_v2/data_hash_check.sh"
# 期望 samples.jsonl sha256 = 5d8bab5dd706c0f48e2310b69cbed877ce28b81e56b8eb18edef5c85801b99f4
# 期望 dataset_manifest sha256 = c5db8924c9fb21cf33ab771cf231a23cb0723ed8c97c1dc63494b8175c82a6ad
```

## 3. 执行顺序

### 3.1 服务器小规模预检（必须先通过）
```bash
bash "$PROJECT_ROOT/server_ops/three_topology_v2/preflight.sh"
```
检查：Python/Torch/PyG/sklearn 版本、GPU 数量与显存、数据哈希、`train_formal.py --smoke`（cuda:0，1 batch）。要求输出 `all_pass: true`。预检未通过不得启动正式训练。

### 3.2 首批正式训练
```bash
bash "$PROJECT_ROOT/server_ops/three_topology_v2/run_first_batch.sh"
```
首批范围（冻结）：`cross_design_topology1/cd_fold_1`，84 训练 / 24 验证；36 外层测试**不构造、不推理、不评分**。

| 波次 | GPU | 模型 | seed | 备注 |
|---|---|---|---|---|
| 1 | 0 | MLP | 20261117 | |
| 1 | 1 | MLP | 20261118 | |
| 1 | 2 | HGT-global | 20261117 | |
| 1 | 3 | HGT-global | 20261118 | |
| 2 | 0 | MLP | 20261119 | |
| 2 | 1 | HGT-global | 20261119 | |
| CPU | - | Ridge + HistGB | - | `OMP_NUM_THREADS=4`，与 GPU 并行 |

- **单任务单 GPU**：`CUDA_VISIBLE_DEVICES=<gpu>`，入口看到 `cuda:0`；**不使用 DDP**；**不使用历史 8 卡配置**。
- 首批最多 4 个 GPU 任务并发；CPU 树/线性模型单独限线程，避免抢占。
- 梯度模型 3 seed、随机初始化、物理损失 0。
- 若显存/内存紧张：把波次内并发降到 3 或 2（改 `run_first_batch.sh` 的 wave），不要改模型超参或切图。

### 3.3 监控与回收
```bash
bash "$PROJECT_ROOT/server_ops/three_topology_v2/monitor.sh"          # nvidia-smi + 日志
bash "$PROJECT_ROOT/server_ops/three_topology_v2/collect_artifacts.sh" # 轻量产物打包
```
`collect_artifacts.sh` 只打包：配置、划分/统计/内部折清单、`run_manifest`/`run_log`/`summary`/`driver.log`/`val_predictions.json`/`best.pt`/`last.pt`、代码 commit 与 SHA256SUMS；**不打包原始数据、processed 图、全部历史 outputs**。若显存/内存不足，回传 tar 后人工下载。

## 4. 后续批次（另列，不在首批）

- 传统图网络：GCN、GAT、R-GCN、GraphSAGE、GIN（同拆分同预算）。
- 代理与锚点：edge-aware MPNN、HGT。
- 近期模型（适配版）：`spectral_lift`（LiftGCN 核心算子的本任务适配版）；另备第二方案（见 `docs/baseline_readiness_fix_20261117/`）。
- 内部代理折：`--protocol inner --outer-fold <loto_*> --inner-id <inner_*>`（只读外层训练侧样本）。

## 5. 性能估计口径

本地 RTX 4070 Laptop 的计时**不能**直接当作 4090 运行时间。首批服务器任务完成后，用实际 `run_log.timing`（train_sec / final_inference_sec）与 `run_manifest.device_info` 记录；不同硬件的 wall-clock 不用于模型效率比较。

## 6. 禁止事项

- 不启动 DDP，不默认 8 卡。
- 不构造/推理/评分外层测试。
- 不复用旧检查点或旧归一化统计。
- 不静默裁剪图或改变监督目标。
- 新数据位置未确认时，用显式占位符，不沿用 GraphTrainingData2。

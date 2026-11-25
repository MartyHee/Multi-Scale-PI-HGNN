# FE v2 服务器运维（模型仓库自足版，2026-11-25）

本目录是服务器所需脚本、配置模板和操作说明的唯一云端来源。**唯一云端同步仓库是 `Multi-Scale-PI-HGNN`；`S1` 目录仅本地报告，不参与 Git 同步。**

服务器路径（已上传，不重传整包）：

```text
/home/miniconda/Bishe/raw_data/three_topology_fe_v2_9216
/home/miniconda/Bishe/Multi-Scale-PI-HGNN/processed/three_topology_fe_v2_9216
/home/miniconda/Bishe/Multi-Scale-PI-HGNN
/home/miniconda/miniconda3/envs/pi_hgnn/bin/python
```

## 1. 新增共享读取缓存（必须上传）

本轮新增 3 个设计成员缓存 + 元数据（约 1.03 MB）：

```text
processed/three_topology_fe_v2_9216/shared/topology1_design_members.npz
processed/three_topology_fe_v2_9216/shared/topology2_design_members.npz
processed/three_topology_fe_v2_9216/shared/topology3_design_members.npz
processed/three_topology_fe_v2_9216/shared/topology1_design_cache.json
processed/three_topology_fe_v2_9216/shared/topology2_design_cache.json
processed/three_topology_fe_v2_9216/shared/topology3_design_cache.json
processed/three_topology_fe_v2_9216/shared/design_cache_manifest.json
```

SHA256：

| 文件 | SHA256 |
|---|---|
| T1 npz | `164375f07bf02a0eb50be59704926f3e3a17cf22c10e75c897c41bced24a8359` |
| T2 npz | `8aef61998776204017f5433a64a7ea5ed083549b0df6fa20c79be78b94deb605` |
| T3 npz | `89db449c26921d2d3449209546373a53683057acbc6b0e0a021b0ee64f4a0f97` |
| T1 meta | `984689ff5d6419ed192d873a055abff9d6d892434cf6e728403c6a19f255a2c7` |
| T2 meta | `c5e8931cffa8042ecca6df02fe5f4005baa6b6e2cc1113e4521af0290c9adaa0` |
| T3 meta | `cdcb20d256844a71d659ce77e4e307cd098cfe7303ce68ee220cd0b712bb8573` |
| 总清单 | `2efb950a114868223efb1a6d5ce8adf1e1a28d01ca21b5f74121aceac53e7746` |

已有 `shared/*_geometry.npz` 与 `fields/*_loads.npz` 不重传。缓存上限：设计成员全 48/模板（内存约 19–41 MB）、场荷载全 64/模板（约 3–15 MB）、响应分片 LRU 32/进程（`SHARD_CACHE`）；无磁盘缓存。若缺失/哈希不符，预检和正式模式会失败（不会静默回退到慢路径）；smoke 可用 `THREE_TOPO_USE_SHARED=0`。

## 2. 一次配置（只在不存在时复制）

PowerShell：

```powershell
Set-Location /home/miniconda/Bishe/Multi-Scale-PI-HGNN   # 或对应 Windows 路径
$cfg = "server_ops/three_topology_fe_v2/server_config.sh"
if (-not (Test-Path $cfg)) {
  Copy-Item "server_ops/three_topology_fe_v2/server_config.example.sh" $cfg
}
# 编辑 PACKAGE_ROOT / PROCESSED_ROOT / PYTHON / THREADS / USE_SHARED / SHARD_CACHE
```

Linux：

```bash
cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN
[ -f server_ops/three_topology_fe_v2/server_config.sh ] || \
  cp server_ops/three_topology_fe_v2/server_config.example.sh \
     server_ops/three_topology_fe_v2/server_config.sh
```

`server_config.sh` 已被 `.gitignore` 忽略；不要覆盖用户已有配置。GPU 不写入配置，启动时用 `CUDA_VISIBLE_DEVICES=<card>`。

## 3. 预检

```bash
cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN
CUDA_VISIBLE_DEVICES=0 bash server_ops/three_topology_fe_v2/preflight.sh \
  remote_jobs/three_topology_fe_v2/t1_mlp.yaml
CUDA_VISIBLE_DEVICES=-1 bash server_ops/three_topology_fe_v2/preflight.sh \
  remote_jobs/three_topology_fe_v2/ml_cpu.yaml
```

PASS 绑定 manifest、代码、config、schema、protocol、ledger、split-manifest、三份 stats、共享缓存的 design/geometry/loads 哈希。任何变化使旧 PASS 失效。

## 4. 有上限吞吐试跑（正式前）

```bash
CUDA_VISIBLE_DEVICES=0 bash server_ops/three_topology_fe_v2/run_benchmark.sh all cuda:0
```

- 每个模板最多 16 训练样本、每验证切片 4 样本、4 个 fwd/bwd step；无 200 epoch。
- 独立目录 `outputs/three_topology_fe_v2/benchmark/<timestamp>/`，不占正式 run、不进入 18 任务统计。
- 只读 A/B/C 训练与验证角色；不读 test/blocked。
- 输出冷/热/随机读取、batch 装配、前后向、三切片验证、RSS、显存、缓存命中；不据此调参。

## 5. 六卡正式启动（日志追加，不截断）

```bash
TMUX=/home/miniconda/miniconda3/envs/tmux_tools/bin/tmux
cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN
mkdir -p logs/three_topology_fe_v2/fv2_first_20261125_01

$TMUX new -d -s fv2_t1_mlp 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=0 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t1_mlp.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t1_mlp.out 2>&1'
$TMUX new -d -s fv2_t1_hgt 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=1 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t1_hgt_global.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t1_hgt.out 2>&1'
$TMUX new -d -s fv2_t2_mlp 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=2 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t2_mlp.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t2_mlp.out 2>&1'
$TMUX new -d -s fv2_t2_hgt 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=3 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t2_hgt_global.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t2_hgt.out 2>&1'
$TMUX new -d -s fv2_t3_mlp 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=4 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t3_mlp.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t3_mlp.out 2>&1'
$TMUX new -d -s fv2_t3_hgt 'cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN && CUDA_VISIBLE_DEVICES=5 bash server_ops/three_topology_fe_v2/run_job.sh remote_jobs/three_topology_fe_v2/t3_hgt_global.yaml >> logs/three_topology_fe_v2/fv2_first_20261125_01/t3_hgt.out 2>&1'
```

- 日志用 `>>` 追加；每个 run 内部另有唯一时间戳 `console.log`。重复启动若 run 目录已存在会拒绝，不会先截断外层日志。
- 单任务单卡、无 DDP；程序内 `cuda:0`；不覆盖/扩大调用方 `CUDA_VISIBLE_DEVICES`。
- CPU 批单独启动（GPU 批完成后）：`CUDA_VISIBLE_DEVICES=-1 bash ... run_job.sh .../ml_cpu.yaml >> ... 2>&1`。

## 6. 监控/验收/回收

```bash
bash server_ops/three_topology_fe_v2/monitor.sh fv2_first_20261125_01
python -B server_ops/three_topology_fe_v2/summarize_batch.py --batch fv2_first_20261125_01 --json
bash server_ops/three_topology_fe_v2/collect_artifacts.sh fv2_first_20261125_01
```

只有 18/18（GPU 批）或 6/6（CPU 批）都 done、身份一致、文件齐全才 `overall_ok=true`。不完整批仍可打包诊断，但退出 1 并标注 NOT COMPLETE。

## 7. Resume / restart 语义

- **Resume**：仅梯度任务；必须从原 checkpoint 恢复，目标 attempt = 原 attempt+1，新 worker 目录；实际新增 epoch；身份（模型/模板/seed/数据/schema/协议/统计/共享缓存/有效配置）不符即失败。示例见 `resume_template.yaml`。
- **Restart（从头重跑）**：本轮没有独立 restart 实现。不能通过 `attempt>1` 无 `--resume-from` 从头开始；若要重跑，需新建任务身份/批次并重新冻结 manifest、重新预检。不要用 resume 命令冒充重拟合。
- **ML（Ridge/HistGB）**：无 epoch resume；失败重跑需新任务身份/批次，不能借用梯度 resume 命令。

## 8. 不做

- 不复制本地 S1 报告到服务器；服务器说明以本文件为准。
- 不提交 `server_config.sh`、`processed/`、`outputs/`、`logs/`、checkpoint、pickle、归档。
- 不读/评分 test/blocked；不改冻结划分/统计。

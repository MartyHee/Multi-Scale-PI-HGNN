# 服务器任务（run_job.sh + YAML，无需 export）

> 2026-11-18：首批 8 个运行已完成；第二批 `sb_20261118_01`（6 模型 × 3 seed）使用同一接口，任务清单见 `remote_jobs/three_topology_v2/manifests/sb_20261118_01.json`。预检/监控/汇总/回收已改为**冻结清单驱动**；详细命令见 `S1/docs/second_batch_preparation_20261118/server_ops.md`。

新接口：
```bash
CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/<job>.yaml
```
- 服务器公共设置只在 `server_config.sh`（DATA_ROOT / PYTHON / THREADS）；工程路径按脚本位置自动定位。
- 任务模型、seed、批次、协议、job 名称由 YAML 明确指定。
- **GPU 由命令行的 `CUDA_VISIBLE_DEVICES` 选择**；调用链不覆盖/清空/扩大；程序内用 `cuda:0`。CPU 子任务只对其自身子进程隐藏 CUDA。
- 复用 `run_task.sh`：原子目录、重复启动拒绝、STATUS、逐轮 `train_log.csv`、显式 resume、best 继承。

## 1. 一次配置
```bash
cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN
if [ ! -f server_ops/three_topology_v2/server_config.sh ]; then
  cp server_ops/three_topology_v2/server_config.example.sh server_ops/three_topology_v2/server_config.sh
fi
# 编辑 DATA_ROOT / PYTHON / THREADS（server_config.sh 已 gitignore，git pull 不覆盖）
# 已知：PYTHON=/home/miniconda/miniconda3/envs/pi_hgnn/bin/python
#       DATA_ROOT=/home/miniconda/Bishe/data/three_topology_graph_v1
bash server_ops/three_topology_v2/check_config.sh remote_jobs/three_topology_v2/mlp.yaml
# 确认 job YAML 与冻结批次清单一致
python server_ops/three_topology_v2/_batch_manifest.py --job remote_jobs/three_topology_v2/mlp.yaml --verify
```

## 2. 依赖检查 / 补装（按检查结果）
```bash
CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/check_config.sh remote_jobs/three_topology_v2/mlp.yaml --deps
```
必需：torch、torch_geometric、torch_scatter、numpy、scikit-learn、PyYAML。**服务器当前清单缺 scikit-learn**，是否已补以检查为准；仅缺失时最小安装（版本与现有 torch 匹配）。

## 3. 预检（限定调用方 GPU，写本批次 PASS）
```bash
CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/preflight.sh remote_jobs/three_topology_v2/mlp.yaml
```
依赖 + 数据哈希 + 本批**所有梯度模型**各 1-batch CUDA 训练/验证/best 导出；只接受仓库内注册 YAML。PASS 绑定 samples/manifest/split/stats/train_config/**v2 batch manifest**/code commit；缺失或任一身份漂移时 `run_job.sh` 退出 4 并提示重跑。

## 4. 四个终端各一条（共同 batch_id 已在 YAML 中一致）
```bash
CUDA_VISIBLE_DEVICES=<gpuA> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/mlp.yaml
CUDA_VISIBLE_DEVICES=<gpuB> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/hgt_global_seed20261117.yaml
CUDA_VISIBLE_DEVICES=<gpuC> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/hgt_global_seed20261118.yaml
CUDA_VISIBLE_DEVICES=<gpuD> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/hgt_global_seed20261119.yaml
```
`<gpuX>` 填四张真实空闲卡编号。每个 (worker, model, seed) 独立运行目录；重复启动拒绝；外层测试不构造。

## 5. 监控 / 汇总
```bash
bash server_ops/three_topology_v2/monitor.sh remote_jobs/three_topology_v2/mlp.yaml
# 机器可读：python server_ops/three_topology_v2/summarize_batch.py --batch <batch_id> --json
```
- 表格列：模型、seed、状态、已完成轮数/上限、当前验证指标、最佳验证指标、最近更新时间；`MONITOR_JSON=1` 输出 JSON。
- 状态区分等待启动/训练中/已完成/已完成(早停)/失败/异常；训练中无 summary 不算失败；完成验收仍严格检查身份与文件。

## 6. 失败与恢复（YAML 显式 resume）
失败：`STATUS.json state=failed`、`traceback.txt`、`console.log`。恢复示例（新 worker 目录，避免覆盖原 run）：
```bash
cat > /tmp/resume.yaml <<'YAML'
job_name: hgt_resume
batch_id: <同一 batch_id>
protocol: split
split_id: cross_design_topology1
fold_id: cd_fold_1
train_config: configs/three_topology_v2/formal_baselines.yaml
tasks:
  - model: hgt_global
    seeds: [20261117]
    device: gpu
    worker: t2_hgt_resume
    resume_from: /home/miniconda/Bishe/Multi-Scale-PI-HGNN/outputs/three_topology_v2/first_batch/<batch_id>/t2_hgt/hgt_global_seed20261117/checkpoints/last.pt
YAML
CUDA_VISIBLE_DEVICES=<gpuB> bash server_ops/three_topology_v2/run_job.sh /tmp/resume.yaml
```
- 只允许**仓库外** resume YAML；`_resume_check.py` 校验 batch/protocol/split/fold、原任务唯一性、checkpoint identity、新 worker 与目标目录。
- 原失败任务保持只读；恢复 run 是额外目录，`summarize` 仍把原任务计为 failed，**不自动替代、不宣称 18/18**。
- 记录 `resume_checks/<batch>/<time>_<job>.json/.tsv`，新 run 的 TASK/STATUS 含 `resume_of_worker`/`resume_of_run`。
- 身份校验；无新最佳导出历史 best；缺 best 报错。

## 7. 打包
```bash
bash server_ops/three_topology_v2/collect_artifacts.sh remote_jobs/three_topology_v2/mlp.yaml
```
默认输出到 `$PROJECT_ROOT/remote_artifacts/batch_<batch_id>_artifacts.tar.gz`（自动建父目录）；`OUT_TAR=<path>` 可覆盖；**目标已存在时拒绝覆盖**（退出 3）。只打包冻结清单中的任务，包内含 `manifests/batch_manifest.json` 与 `TASKS.tsv`；额外 run 目录单列 `EXTRA_RUN_DIRS.txt`；`SHA256SUMS` 排除自身；done 任务缺必要文件即失败（退出非 0）。`remote_artifacts/` 已 gitignore。

## 8. terminal1..4 兼容包装
`terminal1..4.sh` 仅作为**兼容包装**，内部 `exec run_job.sh <对应 yaml>`，要求调用方已设置 `CUDA_VISIBLE_DEVICES`。它们与 run_job 共用同一 YAML/config，不会产生两套结果。**推荐直接用 run_job.sh**。

## 说明
- 不调用旧 `run_job.sh`（旧 `server_ops/run_job.sh`）、`train_baseline.py`、旧数据检查/统计。
- 首批：拓扑一 `cd_fold_1`，84 训练 / 24 验证，物理损失 0；外层 36 测试不构造、不推理、不评分。

# 服务器首批任务（run_job.sh + YAML，无需 export）

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
cp server_ops/three_topology_v2/server_config.example.sh server_ops/three_topology_v2/server_config.sh
# 编辑 DATA_ROOT / PYTHON / THREADS（server_config.sh 已 gitignore，git pull 不覆盖）
# 已知：PYTHON=/home/miniconda/miniconda3/envs/pi_hgnn/bin/python
#       DATA_ROOT=/home/miniconda/Bishe/data/three_topology_graph_v1
bash server_ops/three_topology_v2/check_config.sh remote_jobs/three_topology_v2/mlp.yaml
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
依赖 + 数据哈希 + MLP/HGT 1-batch CUDA 训练/验证/best 导出；写 `outputs/three_topology_v2/preflight/<batch>/PREFLIGHT_PASS.json`。缺失时 `run_job.sh` 退出 4 并提示该命令。

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
# 或直接：python server_ops/three_topology_v2/summarize_batch.py --batch-dir outputs/three_topology_v2/first_batch/<batch_id> --json
```
汇总不强制要求 GPU。

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
身份校验；无新最佳导出历史 best；缺 best 报错。

## 7. 打包
```bash
bash server_ops/three_topology_v2/collect_artifacts.sh remote_jobs/three_topology_v2/mlp.yaml
```
只打包本批次；`SHA256SUMS` 排除自身；done 任务缺必要文件即失败。

## 8. terminal1..4 兼容包装
`terminal1..4.sh` 仅作为**兼容包装**，内部 `exec run_job.sh <对应 yaml>`，要求调用方已设置 `CUDA_VISIBLE_DEVICES`。它们与 run_job 共用同一 YAML/config，不会产生两套结果。**推荐直接用 run_job.sh**。

## 说明
- 不调用旧 `run_job.sh`（旧 `server_ops/run_job.sh`）、`train_baseline.py`、旧数据检查/统计。
- 首批：拓扑一 `cd_fold_1`，84 训练 / 24 验证，物理损失 0；外层 36 测试不构造、不推理、不评分。

# Server Smoke Test & Artifact Packaging Verification

## Purpose

Run a minimal training job on the GPU server to verify that:

1. The conda environment is fully operational (torch, torch_geometric, CUDA)
2. The `processed/hetero_graph_dataset_v2` dataset loads correctly on the server
3. Training runs on GPU (forward + backward + loss computation)
4. Checkpoints (`best_model.pt`, `last_model.pt`, `last_checkpoint.pt`) are saved
5. `train_log.csv` and `metrics_summary.json` are generated
6. Artifact packaging (`package_results.sh`) produces the expected archive
7. The `run_job.sh` launcher captures logs correctly

**This is NOT a formal experiment.** Results from the smoke test should be discarded before formal training.

---

## Server Execution Steps

### Prerequisites

- Conda environment `pi_hgnn` is created and activated
- `processed/hetero_graph_dataset_v2` is copied to the server
- This commit's code is checked out

### Step-by-Step

```bash
# 1. Enter project directory
cd /path/to/Multi-Scale-PI-HGNN

# 2. Pull latest code
git fetch
git checkout <branch>
git pull

# 3. Activate environment
conda activate pi_hgnn

# 4. Verify environment
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi

# 5. Check dataset
bash server_ops/check_dataset.sh

# 6. Run smoke test
bash server_ops/run_job.sh remote_jobs/server_smoke_mlp.yaml

# 7. Monitor training (in another terminal, or after job starts)
tail -f logs/remote/server_smoke_mlp_<timestamp>.log

# 8. Check GPU usage during training
watch -n 2 nvidia-smi
```

### Expected Output

After completion, the following should exist:

```
outputs/baselines/MLP/<timestamp>/
├── config_resolved.yaml
├── model_summary.json
├── best_model.pt           (50-500 KB)
├── last_model.pt
├── last_checkpoint.pt      (includes optimizer & config)
├── train_log.csv           (2 rows + header)
├── metrics_summary.json
├── loss_curve.png
├── metric_curve_disp.png
├── metric_curve_force.png
└── test_predictions_sample.csv

logs/remote/server_smoke_mlp_<timestamp>.log    (training stdout/stderr)

remote_artifacts/server_smoke_mlp_<timestamp>.tar.gz  (packaged artifact)
```

### Artifact Contents

The packaged artifact should contain:

| File | Description |
|------|-------------|
| `config_resolved.yaml` | Resolved training config |
| `model_summary.json` | Model architecture & param count |
| `train_log.csv` | Per-epoch metrics |
| `metrics_summary.json` | Test-set metrics |
| `loss_curve.png` | Train/val loss curve |
| `metric_curve_disp.png` | Validation displacement MAE curve |
| `metric_curve_force.png` | Validation force MAE curve |
| `best_model.pt` | Best checkpoint by val loss |
| `server_smoke_mlp.yaml` | Job config copy |
| `git_info.txt` | Git branch, commit, timestamp |
| `server_output.log` | Full stdout/stderr log |

### Verification Checklist

- [ ] `check_dataset.sh` reports no FAIL
- [ ] Training completes (exit code 0)
- [ ] 2 epochs finished with decreasing loss
- [ ] `train_log.csv` has 2 data rows
- [ ] `best_model.pt` exists and is non-empty
- [ ] `last_checkpoint.pt` exists
- [ ] `metrics_summary.json` contains disp + force metrics
- [ ] `logs/remote/` has the full log file
- [ ] `remote_artifacts/` has the `.tar.gz` archive
- [ ] `best_model.pt` can be loaded in Python: `torch.load("best_model.pt")`

### If Something Fails

1. **Dataset not found**: Check `processed/hetero_graph_dataset_v2` is copied
2. **CUDA out of memory**: Reduce `batch_size` or `max_graphs` in `remote_jobs/server_smoke_mlp.yaml`
3. **torch-scatter import error**: Reinstall torch-scatter matching torch version
4. **Other import errors**: Run `pip install -r requirements.txt` and verify packages
5. **If nothing works**: Copy the full log file from `logs/remote/` back to local for analysis

```bash
# To retrieve logs if run_dir is known:
ls -lh logs/remote/
ls -lh outputs/baselines/MLP/
bash server_ops/package_results.sh outputs/MLP/<timestamp> server_smoke_mlp
```

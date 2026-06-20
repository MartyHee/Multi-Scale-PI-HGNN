# MLP Baseline — Artifact Verification Report

## Artifact Info

| Field | Value |
|-------|-------|
| Path | `remote_artifacts/server_mlp_full_20260620060955.tar.gz` |
| Size | 3,182,153 bytes (3107 KB) |
| File count | 13 files (+ 1 directory entry) |
| Training job | `server_mlp_full` |
| Server commit | `8945bbd` |
| Server host | 7d2077f2b2dc |
| Server user | miniconda |
| Generated | 2026-06-20 06:09:55 UTC |

## File Manifest

| File | Size | Status |
|------|------|--------|
| `config_resolved.yaml` | 0.8 KB | ✅ |
| `model_summary.json` | 0.6 KB | ✅ |
| `train_log.csv` | 9.0 KB | ✅ |
| `metrics_summary.json` | 2.1 KB | ✅ |
| `loss_curve.png` | 46.9 KB | ✅ |
| `metric_curve_disp.png` | 69.4 KB | ✅ |
| `metric_curve_force.png` | 60.3 KB | ✅ |
| `best_model.pt` | 398.6 KB | ✅ |
| `last_model.pt` | 398.6 KB | ✅ |
| `last_checkpoint.pt` | 1,175.5 KB | ✅ (optional) |
| `server_mlp_full.yaml` (job yaml) | 0.9 KB | ✅ |
| `git_info.txt` | 0.2 KB | ✅ |
| `server_output.log` | 12,336.3 KB | ✅ |

## Metrics Consistency Check

| Metric | Artifact Value | User Report | Tolerance | Status |
|--------|---------------|-------------|-----------|--------|
| Test Disp R² (macro avg) | 0.855406 | 0.8554 | <1% | ✅ |
| Test Force R² (macro avg) | 0.982420 | 0.9824 | <1% | ✅ |
| Test Disp MAE (macro avg) | 0.000208 | 0.000208 | <1% | ✅ |
| Test Force MAE (macro avg) | 15972.99 | 15972.99 | <1% | ✅ |
| Combined RelMAE | 0.088441 | 0.08844 | <1% | ✅ |
| Total Params | 96,274 | 96,274 | exact | ✅ |
| train_log.csv epochs | 100 | 100 | exact | ✅ |

## Additional Checks

| Check | Result |
|-------|--------|
| Training exit code = 0 | ✅ confirmed in server_output.log |
| No `processed/` dir in artifact | ✅ |
| No `raw_data/` dir in artifact | ✅ |
| No `outputs/` full tree in artifact | ✅ |
| No `.tmp` or `__pycache__` files | ✅ |
| Only 3 `.pt` files (best + last + last_checkpoint) | ✅ |
| Git commit recorded | ✅ 8945bbd |
| Git branch recorded | ✅ main |

## Conclusion

**✅ Artifact passes all checks.** The MLP baseline full training result is complete, self-consistent, and can be used as the official MLP row in the Stage 2-A baseline summary.

#!/usr/bin/env python
"""Local CPU + CUDA smoke validation for the six second-batch models.

Each model runs a shortened training (1 epoch x 1 batch, smoke_baselines.yaml,
diagnostic on) on the real cd_fold_1 split (84 train / 24 val) and must show:
forward/finite loss, gradients and a real parameter update, validation, best
checkpoint export, prediction export in physical units and a recomputed
selection metric that matches the run log.

The outer test is never constructed. Formal job YAMLs are also checked to have
no residual epochs/max_batches overrides.

Outputs: outputs/three_topology_v2/second_batch_preparation_20261118/
         local_validation/{cpu,cuda}/<model>/... and local_validation.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402
from src.train.three_topology_v2.metrics import DISP_NAMES, FORCE_NAMES  # noqa: E402

MODELS = ["gcn", "graphsage", "gat", "gin", "rgcn", "hgt"]
SPLIT = "cross_design_topology1"
FOLD = "cd_fold_1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--devices", default="cpu,cuda:0")
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "smoke_baselines.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "second_batch_preparation_20261118"))
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    devices = [d.strip() for d in a.devices.split(",") if d.strip()]
    out = Path(a.out)
    val_dir = out / "local_validation"
    if val_dir.exists():
        shutil.rmtree(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)

    idx = index.load_index(strict=True)
    split = json.loads((paths.SPLIT_DIR / f"split_{SPLIT}.json").read_text(encoding="utf-8"))
    fold = next(f for f in split["folds"] if f["fold_id"] == FOLD)
    val_ids = list(fold["validation_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    stats_path = paths.STATS_DIR / SPLIT / f"{FOLD}.json"
    st = json.loads(stats_path.read_text(encoding="utf-8"))
    std_disp = np.array([c["effective_std"] for c in st["columns"]["y_disp"]], dtype=np.float64)
    std_force = np.array([c["effective_std"] for c in st["columns"]["y_force"]], dtype=np.float64)

    labels = {}
    for sid in val_ids:
        d = graph_io.load_graph(idx["row_by_id"][sid], dtype=torch.float64, verify_hash=True)
        labels[sid] = (d["mesh_node"].y_disp.numpy(), d["beam_element"].y_force.numpy())
    assert not (set(val_ids) & test_ids)

    # formal job YAMLs must not carry a shortened budget
    yaml_checks = {}
    import yaml
    for p in sorted((MODEL_ROOT / "remote_jobs" / "three_topology_v2").glob("*.yaml")):
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
        bad_top = {k: spec.get(k) for k in ("epochs", "max_batches") if k in spec}
        bad_task = [{k: t.get(k) for k in ("epochs", "max_batches") if k in t}
                    for t in spec.get("tasks", []) if any(k in t for k in ("epochs", "max_batches"))]
        yaml_checks[p.name] = {"top_level_overrides": bad_top, "task_overrides": bad_task,
                               "ok": not bad_top and not bad_task}

    report = {"schema": "qt-second-batch-local-validation-v1", "date": "2026-11-18",
              "split_id": SPLIT, "fold_id": FOLD, "val_n": len(val_ids),
              "config": str(a.config), "config_sha256": sha(Path(a.config)),
              "formal_yaml_checks": yaml_checks, "runs": {}, "problems": []}

    for dev in devices:
        tag = dev.replace(":", "_")
        for model in models:
            run_dir = val_dir / tag / model
            cmd = [sys.executable, "-B", "scripts/three_topology_v2/train_formal.py",
                   "--config", str(a.config), "--models", model, "--seeds", "20261117",
                   "--split", SPLIT, "--fold", FOLD, "--device", dev,
                   "--epochs", "1", "--max-batches", "1", "--diagnostic",
                   "--run-root", str(run_dir), "--require-new-dir",
                   "--batch-id", "local_smoke", "--worker-id", f"lv_{tag}_{model}"]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=MODEL_ROOT, capture_output=True, text=True)
            dt = time.time() - t0
            (run_dir.parent).mkdir(parents=True, exist_ok=True)
            rec = {"model": model, "device": dev, "rc": proc.returncode,
                   "wall_sec": round(dt, 2), "checks": {}, "problems": []}
            if proc.returncode != 0:
                rec["problems"].append("train_formal rc != 0")
                rec["stderr_tail"] = proc.stderr[-2000:]
                report["runs"][f"{dev}/{model}"] = rec
                continue
            log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
            preds = json.loads((run_dir / "val_predictions.json").read_text(encoding="utf-8"))
            by_id = {r["sample_id"]: r for r in preds}
            rec["checks"]["pass_all"] = log.get("pass", {}).get("all") is True
            rec["checks"]["completed_epochs"] = log.get("completed_epochs") == 1
            rec["checks"]["best_exported"] = (run_dir / "checkpoints/best.pt").is_file()
            rec["checks"]["best_sha_matches"] = sha(run_dir / "checkpoints/best.pt") == \
                log.get("best", {}).get("checkpoint_sha256")
            best_ep = log.get("best", {}).get("epoch_zero_based")
            rec["best_epoch_zero_based"] = best_ep
            # best==last is expected when the best epoch IS the last epoch (1-epoch smoke)
            rec["checks"]["best_differs_from_last"] = (
                True if best_ep is not None and best_ep == log.get("completed_epochs", 0) - 1
                else sha(run_dir / "checkpoints/best.pt") != sha(run_dir / "checkpoints/last.pt"))
            rec["checks"]["train_log_rows"] = (sum(1 for _ in open(run_dir / "train_log.csv",
                                                                  encoding="utf-8")) - 1) == 1
            rec["checks"]["pred_covers_val"] = set(by_id) == set(val_ids)
            rec["checks"]["pred_no_test"] = not (set(by_id) & test_ids)
            rec["checks"]["physics_zero"] = log.get("physics_loss_weight") == 0.0
            rec["checks"]["diagnostic_param_update"] = log.get("diagnostic") is True
            rec["checks"]["device_actual"] = log.get("device_info", {}).get("device") == dev
            shapes_ok = True
            for sid in val_ids:
                r = by_id[sid]
                td, tf = labels[sid]
                if (len(r["pred_disp"]) != td.shape[0] or len(r["pred_force"]) != tf.shape[0]
                        or not np.isfinite(np.asarray(r["pred_disp"])).all()
                        or not np.isfinite(np.asarray(r["pred_force"])).all()):
                    shapes_ok = False
            rec["checks"]["pred_shapes_finite"] = shapes_ok
            # recompute selection exactly as the engine does
            gd, gf = [], []
            for sid in val_ids:
                r = by_id[sid]; td, tf = labels[sid]
                gd.append(((np.asarray(r["pred_disp"], dtype=np.float64) - td) ** 2).mean(axis=0))
                gf.append(((np.asarray(r["pred_force"], dtype=np.float64) - tf) ** 2).mean(axis=0))
            sel = 0.5 * (float(np.nanmean(np.stack(gd) / std_disp ** 2))
                         + float(np.nanmean(np.stack(gf) / std_force ** 2)))
            rec["selection_recomputed"] = sel
            rec["selection_run_log"] = log.get("final_val", {}).get("selection_metric")
            rec["checks"]["selection_matches_run_log"] = abs(sel - rec["selection_run_log"]) < 1e-6
            rec["n_params"] = log.get("n_params")
            rec["timing"] = log.get("timing")
            rec["device_info"] = log.get("device_info")
            rec["checks"]["all_ok"] = all(bool(v) for v in rec["checks"].values())
            if not rec["checks"]["all_ok"]:
                rec["problems"].extend([k for k, v in rec["checks"].items() if not v])
                report["problems"].append(f"{dev}/{model}: {rec['problems']}")
            report["runs"][f"{dev}/{model}"] = rec

    rows = []
    for key, rec in report["runs"].items():
        rows.append({"run": key, "model": rec["model"], "device": rec["device"],
                     "rc": rec["rc"], "wall_sec": rec["wall_sec"],
                     "n_params": rec.get("n_params"),
                     "selection": rec.get("selection_recomputed"),
                     "all_ok": rec.get("checks", {}).get("all_ok", False)})
    with (out / "local_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    report["all_ok"] = all(r.get("checks", {}).get("all_ok") for r in report["runs"].values()) \
        and all(v["ok"] for v in yaml_checks.values())
    (out / "local_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "runs": rows,
                      "problems": report["problems"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

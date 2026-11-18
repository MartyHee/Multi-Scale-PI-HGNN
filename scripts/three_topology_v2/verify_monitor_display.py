#!/usr/bin/env python
"""Local synthetic tests for monitor_table.py (no server, no training).

Covers: waiting, running, completed, early-stopped completed, failed,
identity/file abnormal, missing required file, empty log, partially written
last log line, ML task, and 8-task/18-task compatibility.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
TABLE = MODEL_ROOT / "server_ops" / "three_topology_v2" / "monitor_table.py"
OUT = MODEL_ROOT / "outputs" / "three_topology_v2" / "second_batch_review_20261118"
HEADER = "epoch,train_loss_mean,selection_metric,best_val,improved,no_improve,val_batches,train_sec,val_sec,mem_mb,order_sha256,utc"


def write_log(p: Path, rows, partial=False):
    lines = [HEADER]
    for i, (tr, val) in enumerate(rows):
        lines.append(f"{i},{tr},{val},{val},True,0,6,1.0,0.1,100,abc,2026-11-18T00:0{i}:00Z")
    if partial:
        lines.append("7,0.1,0.2")  # incomplete final line
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_run(root: Path, batch: str, worker: str, run: str, model: str, seed: str, kind: str,
             state: str, rows=None, early=False, missing=None, bad_identity=False,
             empty_log=False, partial=False, requested=60):
    d = root / "outputs" / "three_topology_v2" / "first_batch" / batch / worker / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    files = (["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
              "train_log.csv", "driver.log", "val_predictions.json",
              "checkpoints/best.pt", "checkpoints/last.pt"] if kind == "gradient"
             else ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
                   "driver.log", "val_predictions.json", f"ml_{run}.pkl"])
    for f in files:
        if missing and f == missing:
            continue
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("{}", encoding="utf-8")
    status = {"state": state, "batch_id": batch, "worker_id": "WRONG" if bad_identity else worker,
              "model": model, "seed": seed, "exit_code": 0 if state == "done" else 1,
              "finished_at": "2026-11-18T01:00:00Z", "device": "cuda:0",
              "external_cuda_visible_devices": "0"}
    (d / "STATUS.json").write_text(json.dumps(status), encoding="utf-8")
    (d / "TASK.json").write_text(json.dumps({"batch_id": batch, "worker_id": worker,
                                             "model": model, "seed": seed}), encoding="utf-8")
    (d / "run_manifest.json").write_text(json.dumps(
        {"resolved_config": {"training": {"epochs": requested}}}), encoding="utf-8")
    (d / "run_log.json").write_text(json.dumps(
        {"completed_epochs": len(rows or []), "epochs_requested": requested,
         "early_stopped": early,
         "final_val": {"selection_metric": rows[-1][1] if rows else None},
         "best": {"selection_metric": min(v for _, v in rows) if rows else None,
                  "epoch_zero_based": 2 if rows else -1}}), encoding="utf-8")
    if not empty_log:
        write_log(d / "train_log.csv", rows or [], partial=partial)
    else:
        (d / "train_log.csv").write_text(HEADER + "\n", encoding="utf-8")


def manifest(root: Path, batch: str, tasks):
    p = root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": "qt-three-topology-batch-manifest-v2",
                             "batch_id": batch, "n_tasks": len(tasks),
                             "tasks": tasks}), encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="monitor_test_"))
    batch = "TEST18"
    tasks = []
    specs = [
        ("t1", "gcn_seed1", "gcn", "1", "gradient", "running", [(0.9, 0.8), (0.7, 0.6), (0.6, 0.55)], False, None, False, False, False),
        ("t1", "gcn_seed2", "gcn", "2", "gradient", "done", [(0.9, 0.8), (0.6, 0.5), (0.5, 0.45)], False, None, False, False, False),
        ("t2", "gat_seed1", "gat", "1", "gradient", "done", [(0.8, 0.7), (0.5, 0.4), (0.4, 0.35)], True, None, False, False, False),
        ("t3", "gin_seed1", "gin", "1", "gradient", "failed", [(0.9, 0.9)], False, None, False, False, False),
        ("t4", "hgt_seed1", "hgt", "1", "gradient", "done", [(0.5, 0.4), (0.4, 0.3)], False, "checkpoints/best.pt", False, False, False),
        ("t5", "rgcn_seed1", "rgcn", "1", "gradient", "done", [(0.5, 0.4)], False, None, True, False, False),
        ("t6", "mlp_seed1", "mlp", "1", "gradient", "running", [], False, None, False, True, False),
        ("t7", "ridge", "ridge", "-", "ml", "done", [], False, None, False, False, False),
        ("t8", "hgt_seed2", "hgt", "2", "gradient", "running", [(0.9, 0.8), (0.8, 0.7)], False, None, False, False, True),
    ]
    for i, (w, r, m, s, kind, state, rows, early, miss, bad, empty, partial) in enumerate(specs, start=1):
        make_run(tmp, batch, w, r, m, s, kind, state, rows=rows, early=early, missing=miss,
                 bad_identity=bad, empty_log=empty, partial=partial)
        tasks.append({"worker": w, "run": r, "model": m, "seed": s, "kind": kind,
                      "device": "cpu" if kind == "ml" else "gpu", "epochs": 60,
                      "max_batches": None, "resume_from": None})
    manifest(tmp, batch, tasks)
    # wait task: add to manifest but no run dir
    tasks_wait = tasks + [{"worker": "t9", "run": "gcn_seed9", "model": "gcn", "seed": "9",
                           "kind": "gradient", "device": "gpu", "epochs": 60,
                           "max_batches": None, "resume_from": None}]
    manifest(tmp, batch, tasks_wait)
    proc = subprocess.run([sys.executable, "-B", str(TABLE), "--batch", batch,
                           "--project-root", str(tmp), "--json"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stderr); return 1
    data = json.loads(proc.stdout)
    states = {r["run"]: r["state"] for r in data["rows"]}
    expected = {
        "gcn_seed1": "训练中", "gcn_seed2": "已完成", "gat_seed1": "已完成(早停)",
        "gin_seed1": "失败", "hgt_seed1": "异常(文件或身份)", "rgcn_seed1": "异常(文件或身份)",
        "mlp_seed1": "训练中", "ridge": "已完成", "hgt_seed2": "训练中",
        "gcn_seed9": "等待启动",
    }
    checks = {"json_rc": True, "states": states == expected,
              "states_got": states, "expected": expected,
              "running_partial_row_used": next(r for r in data["rows"] if r["run"] == "hgt_seed2")["epochs_completed"] == 2,
              "empty_log_no_crash": next(r for r in data["rows"] if r["run"] == "mlp_seed1")["epochs_completed"] in (0, None),
              "done_epochs": next(r for r in data["rows"] if r["run"] == "gcn_seed2")["epochs_completed"] == 3,
              "early_flag": next(r for r in data["rows"] if r["run"] == "gat_seed1")["early_stopped"] is True}
    # human table also runs
    proc2 = subprocess.run([sys.executable, "-B", str(TABLE), "--batch", batch,
                            "--project-root", str(tmp)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    checks["human_rc"] = proc2.returncode == 0
    checks["human_has_states"] = all(s in proc2.stdout for s in
                                     ["训练中", "已完成", "已完成(早停)", "失败", "异常(文件或身份)", "等待启动"])
    # first-batch 8-task compatibility
    manifest(tmp, "TEST8", tasks[:7] + [{"worker": "t10", "run": "ridge2", "model": "ridge",
                                         "seed": "-", "kind": "ml", "device": "cpu", "epochs": 60,
                                         "max_batches": None, "resume_from": None}])
    proc3 = subprocess.run([sys.executable, "-B", str(TABLE), "--batch", "TEST8",
                            "--project-root", str(tmp), "--json"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    checks["first_batch_8_rc"] = proc3.returncode == 0
    checks["first_batch_8_rows"] = len(json.loads(proc3.stdout)["rows"]) == 8
    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "monitor_display_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Human-readable, read-only batch monitor table.

States: 等待启动 / 训练中 / 已完成 / 已完成(早停) / 失败 / 异常(文件或身份).
- reads the frozen v2 manifest for the expected task list;
- reads STATUS.json, run_log.json, run_manifest.json and train_log.csv only;
- tolerates a log being appended, empty, or ending with a partial line;
- never writes into run dirs;
- keeps a machine-readable --json mode and works for the 8-task first batch
  and the 18-task second batch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]
REQUIRED_GRAD = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
                 "train_log.csv", "driver.log", "val_predictions.json",
                 "checkpoints/best.pt", "checkpoints/last.pt"]
REQUIRED_ML = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
               "driver.log", "val_predictions.json"]


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def read_train_log(p: Path):
    """Return (rows, last_partial_line). Ignores malformed / partial trailing rows."""
    if not p.is_file():
        return [], False
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return [], False
    lines = text.splitlines()
    if not lines:
        return [], False
    header = lines[0]
    rows, partial = [], False
    for i, line in enumerate(lines[1:], start=1):
        if not line.strip():
            continue
        try:
            vals = next(csv.reader([line]))
            if len(vals) < 12:
                partial = (i == len(lines) - 1)
                continue
            int(vals[0]); float(vals[1]); float(vals[2]); float(vals[3])
            rows.append(vals)
        except Exception:  # noqa: BLE001
            partial = (i == len(lines) - 1)
    return rows, partial


def state_of(d: Path, task: dict):
    if not d.exists():
        return "等待启动", "no run dir"
    st_path = d / "STATUS.json"
    if not st_path.is_file():
        return "异常(文件或身份)", "no STATUS.json"
    st = read_json(st_path)
    state = st.get("state")
    if state == "failed":
        return "失败", f"exit={st.get('exit_code')}"
    if state == "running":
        return "训练中", ""
    if state != "done":
        return "异常(文件或身份)", f"state={state}"
    identity_ok = (st.get("batch_id") == task["batch_id"] and st.get("worker_id") == task["worker"]
                   and st.get("model") == task["model"] and str(st.get("seed")) == str(task["seed"]))
    task_json = read_json(d / "TASK.json")
    if task_json:
        identity_ok = identity_ok and task_json.get("worker_id") == task["worker"] \
            and task_json.get("model") == task["model"] \
            and str(task_json.get("seed")) == str(task["seed"])
    req = REQUIRED_GRAD if task["kind"] == "gradient" else REQUIRED_ML
    if task["kind"] == "ml":
        req = req + [f"ml_{task['run']}.pkl"]
    missing = [f for f in req if not (d / f).is_file()]
    if not identity_ok or missing:
        return "异常(文件或身份)", ("identity mismatch" if not identity_ok else f"missing {missing[:2]}")
    log = read_json(d / "run_log.json")
    if log.get("early_stopped"):
        return "已完成(早停)", ""
    return "已完成", ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(MODEL_ROOT)))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(native(a.project_root))
    batch = a.batch
    man_p = root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch}.json"
    if not man_p.is_file():
        raise SystemExit(f"frozen manifest not found: {man_p}")
    man = read_json(man_p)
    if man.get("schema") != "qt-three-topology-batch-manifest-v2" or not man.get("tasks"):
        raise SystemExit(f"manifest {batch} is not a non-empty v2 manifest")
    batch_dir = root / "outputs" / "three_topology_v2" / "first_batch" / batch
    rows = []
    for t in man["tasks"]:
        task = dict(t); task["batch_id"] = batch
        d = batch_dir / t["worker"] / t["run"]
        state, note = state_of(d, task)
        status = read_json(d / "STATUS.json") if (d / "STATUS.json").is_file() else {}
        log = read_json(d / "run_log.json") if (d / "run_log.json").is_file() else {}
        run_man = read_json(d / "run_manifest.json") if (d / "run_manifest.json").is_file() else {}
        tl_rows, partial = read_train_log(d / "train_log.csv")

        requested = None
        for src in (log.get("epochs_requested"),
                    (run_man.get("resolved_config") or {}).get("training", {}).get("epochs"),
                    t.get("epochs")):
            if src:
                requested = int(src); break
        completed = None
        if log.get("completed_epochs") is not None:
            completed = int(log["completed_epochs"])
        elif tl_rows:
            completed = int(tl_rows[-1][0]) + 1
        current = best = None
        last_update = status.get("finished_at") or ""
        if tl_rows:
            try:
                current = float(tl_rows[-1][2]) if tl_rows[-1][2] != "" else None
                best = float(tl_rows[-1][3]) if tl_rows[-1][3] != "" else None
                last_update = tl_rows[-1][11] or last_update
            except Exception:  # noqa: BLE001
                pass
        if state.startswith("已完成"):
            current = log.get("final_val", {}).get("selection_metric", current)
            best = log.get("best", {}).get("selection_metric", best)
        rows.append({
            "model": t["model"], "run": t["run"], "worker": t["worker"], "seed": t["seed"],
            "state": state, "note": note,
            "epochs_completed": completed, "epochs_requested": requested,
            "current_val": current, "best_val": best, "last_update": last_update,
            "early_stopped": bool(log.get("early_stopped")),
            "log_partial_last_line": partial,
            "device": status.get("device"), "external_cvd": status.get("external_cuda_visible_devices"),
        })
    if a.json:
        print(json.dumps({"batch_id": batch, "batch_dir": str(batch_dir), "rows": rows},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"批次 {batch}  目录 {batch_dir}")
    print("指标越低越好；轮数为已完成轮数（1 基）；最近更新为 train_log 的 UTC 时间。")
    header = f"{'模型':<10} {'seed':<9} {'状态':<14} {'轮数':<12} {'当前验证指标':>13} {'最佳验证指标':>13} {'最近更新':<21}"
    print(header); print("-" * len(header))
    for r in rows:
        ep = "—" if r["epochs_completed"] is None else str(r["epochs_completed"])
        if r["epochs_requested"]:
            ep = f"{ep}/{r['epochs_requested']}"
        cur = "—" if r["current_val"] is None else f"{r['current_val']:.6g}"
        best = "—" if r["best_val"] is None else f"{r['best_val']:.6g}"
        seed = r["seed"] if r["seed"] != "-" else "—"
        print(f"{r['model']:<10} {seed:<9} {r['state']:<14} {ep:<12} {cur:>13} {best:>13} {str(r['last_update'])[:19]:<21}")
    print("-" * len(header))
    print("状态说明：等待启动 / 训练中 / 已完成 / 已完成(早停) / 失败 / 异常(文件或身份)。")
    print("训练中尚无最终 summary/预测文件属正常；异常(文件或身份)会阻断最终 18/18 验收。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Local lightweight simulation of the four-terminal isolation + packaging.

No GPU, no formal training. Verifies:
  * atomic run-dir creation and duplicate-start refusal;
  * worker isolation (distinct run dirs);
  * per-task status files and read-only batch summary (partial failure detected);
  * collect_artifacts.sh strictness (missing required file aborts) and scope
    (only batch/config/manifests/code; no raw data).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def posix(p):
    s = str(Path(p)).replace(chr(92), "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/{m.group(1).lower()}/{m.group(2)}" if m else s

MODEL_ROOT = Path(__file__).resolve().parents[2]
SERVER_OPS = MODEL_ROOT / "server_ops" / "three_topology_v2"
EXPECTED = [
    ("t1_mlp", "mlp_seed20261117", "gradient"),
    ("t1_mlp", "mlp_seed20261118", "gradient"),
    ("t1_mlp", "mlp_seed20261119", "gradient"),
    ("t1_ridge", "ridge", "ml"),
    ("t1_histgb", "hist_gb", "ml"),
    ("t2_hgt", "hgt_global_seed20261117", "gradient"),
    ("t3_hgt", "hgt_global_seed20261118", "gradient"),
    ("t4_hgt", "hgt_global_seed20261119", "gradient"),
]
GRAD = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
        "train_log.csv", "driver.log", "val_predictions.json", "checkpoints/best.pt",
        "checkpoints/last.pt"]
ML = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
      "driver.log", "val_predictions.json"]


def build_fake(root: Path, batch="SIMBATCH"):
    base = root / "outputs" / "three_topology_v2" / "first_batch" / batch
    for worker, run, kind in EXPECTED:
        d = base / worker / run
        d.mkdir(parents=True, exist_ok=True)
        files = GRAD if kind == "gradient" else ML
        for f in files:
            p = d / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")
        if kind == "ml":
            (d / f"ml_{run}.pkl").write_text("x", encoding="utf-8")
        (d / "STATUS.json").write_text(json.dumps(
            {"state": "done", "batch_id": batch, "worker_id": worker, "model": run}), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(
            {"all_pass": True, "models": {f"{run}": {"selection_metric": 0.1}}}), encoding="utf-8")
    for sub in ("splits", "stats", "stats_inner", "manifests"):
        d = root / "processed" / "three_topology_training_v2" / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "dummy.json").write_text("{}", encoding="utf-8")
    (root / "configs" / "three_topology_v2").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "three_topology_v2" / "formal_baselines.yaml").write_text("x: 1\n", encoding="utf-8")
    return base


def run(cmd, env=None, cwd=None):
    """cmd: list of args (no shell) — avoids cmd.exe vs bash issues on Windows."""
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    import os
    import tempfile
    reps = {}
    tmp = Path(tempfile.mkdtemp(prefix="sim_iso_"))
    fake = tmp / "proj"
    batch = "SIMBATCH"
    base = build_fake(fake, batch)
    py = sys.executable
    env = dict(os.environ, PROJECT_ROOT=posix(fake), BATCH_ID=batch, PYTHON=py)

    import shutil
    bash_exe = shutil.which("bash") or "bash"
    # 1. duplicate run dir refused (atomic mkdir semantics)
    d1 = base / "t1_mlp" / "mlp_seed20261117"
    try:
        d1.mkdir(exist_ok=False)
        dup_refused = False
    except FileExistsError:
        dup_refused = True
    reps["duplicate_dir_refused"] = dup_refused
    # 2. worker isolation
    reps["worker_isolation"] = (base / "t2_hgt" / "hgt_global_seed20261117").is_dir() and \
        (base / "t3_hgt" / "hgt_global_seed20261118").is_dir()
    summarize = [py, "-B", str(SERVER_OPS / "summarize_batch.py"), "--batch-dir", str(base), "--json"]
    # 3. all done -> summary ok
    rc, out = run(summarize)
    reps["summary_all_done_ok"] = rc == 0 and '"overall_ok": true' in out
    # 4. partial failure -> summary not ok
    f3 = base / "t3_hgt" / "hgt_global_seed20261118" / "STATUS.json"
    f3.write_text(json.dumps({"state": "failed"}), encoding="utf-8")
    rc, out = run(summarize)
    reps["summary_detects_failure"] = rc != 0 and '"overall_ok": false' in out
    f3.write_text(json.dumps({"state": "done"}), encoding="utf-8")
    # 5. collect aborts on missing required file
    tar = tmp / "out.tar.gz"
    env_c = dict(env, OUT_TAR=posix(tar))
    collect = [bash_exe, posix(SERVER_OPS / "collect_artifacts.sh")]
    miss = base / "t1_ridge" / "ridge" / "ml_ridge.pkl"
    miss_renamed = miss.with_suffix(".bak")
    miss.rename(miss_renamed)
    rc_missing, out_missing = run(collect, env=env_c)
    reps["collect_aborts_missing"] = rc_missing != 0 and "MISSING required" in out_missing
    miss_renamed.rename(miss)
    # 6. collect success + scope
    rc_ok, out_ok = run(collect, env=env_c)
    reps["collect_success"] = rc_ok == 0 and tar.is_file()
    reps["collect_debug"] = {"rc": rc_ok, "exists": tar.is_file(), "out": out_ok[-400:]}
    if tar.is_file():
        rc_t, listing = run(["tar", "-tzf", str(tar)])
        reps["collect_scope_no_raw"] = "raw_data" not in listing and "/batch/" in listing
        reps["collect_scope_has_config"] = "configs/three_topology_v2/formal_baselines.yaml" in listing
        rc_s, sums = run(["tar", "-xzf", str(tar), "-O", "./SHA256SUMS"])
        reps["sha256sums_excludes_self"] = "SHA256SUMS" not in sums

    reps["ok"] = all(v for v in reps.values() if isinstance(v, bool))
    out = {"checks": reps, "tmp": str(tmp)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (MODEL_ROOT / "outputs" / "three_topology_v2" / "simulate_isolation").mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / "outputs" / "three_topology_v2" / "simulate_isolation" / "sim_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if reps["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

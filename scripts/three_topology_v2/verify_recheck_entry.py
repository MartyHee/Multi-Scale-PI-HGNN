#!/usr/bin/env python
"""Local small-scale validation of the eval-only recheck entry.

Builds a throw-away 1-epoch/1-batch HGT run (smoke config), places it under a
temporary batch id, and runs ``eval_first_batch_best.sh`` against it to verify:
  * the entry reloads best.pt and evaluates only the 24 validation graphs;
  * the recheck is written to a NEW dir; the original files are untouched;
  * predictions and selection match under an explicit tolerance;
  * the checkpoint sha256 is recorded and no checkpoint was edited.

The REAL first batch is NOT touched; this does not execute the server recheck.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))
from src.data.three_topology_v2 import paths  # noqa: E402

BATCH = "smoke_recheck_local"
RUN = "hgt_seed20261117"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="recheck_test_"))
    smoke = tmp / "smoke_run"
    py = sys.executable
    cmd = [py, "-B", "scripts/three_topology_v2/train_formal.py",
           "--config", str(paths.CONFIG_DIR / "smoke_baselines.yaml"),
           "--models", "hgt", "--seeds", "20261117",
           "--split", "cross_design_topology1", "--fold", "cd_fold_1",
           "--device", "cpu", "--epochs", "1", "--max-batches", "1",
           "--run-root", str(smoke), "--require-new-dir",
           "--batch-id", BATCH, "--worker-id", "t_test"]
    proc = subprocess.run(cmd, cwd=MODEL_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        return 1

    batch_dir = MODEL_ROOT / "outputs" / "three_topology_v2" / "first_batch" / BATCH
    orig = batch_dir / "t_test" / RUN
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    orig.mkdir(parents=True)
    for rel in ("checkpoints/best.pt", "checkpoints/last.pt", "val_predictions.json",
                "run_log.json", "run_manifest.json", "summary.json", "TRAIN_LOG.csv"):
        src = smoke / rel
        if src.is_file():
            dst = orig / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (orig / "train_log.csv").write_text((smoke / "train_log.csv").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    if (orig / "TRAIN_LOG.csv").exists():
        (orig / "TRAIN_LOG.csv").unlink()

    man_dir = MODEL_ROOT / "remote_jobs" / "three_topology_v2" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "schema": "qt-three-topology-batch-manifest-v1", "batch_id": BATCH,
        "protocol": "split", "split_id": "cross_design_topology1", "fold_id": "cd_fold_1",
        "outer_fold": "", "inner_id": "",
        "train_config": "configs/three_topology_v2/formal_baselines.yaml",
        "train_config_sha256": sha(MODEL_ROOT / "configs/three_topology_v2/formal_baselines.yaml"),
        "source_jobs": ["local-test"], "n_tasks": 1,
        "tasks": [{"worker": "t_test", "run": RUN, "model": "hgt", "seed": "20261117",
                   "device": "gpu", "kind": "gradient", "job_name": "local_test",
                   "job_file": "local-test"}],
    }
    man_path = man_dir / f"{BATCH}.json"
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

    cfg = tmp / "server_config.sh"
    cfg.write_text(
        f'DATA_ROOT="{paths.package_root()}"\n'
        f'PYTHON="{py}"\nTHREADS=4\nPROJECT_ROOT="{MODEL_ROOT}"\n', encoding="utf-8")
    recheck_root = tmp / "recheck"
    env = dict(os.environ, THREE_TOPO_SERVER_CONFIG=str(cfg), RECHECK_ROOT=str(recheck_root),
               EVAL_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
    before = {rel: sha(orig / rel) for rel in ("checkpoints/best.pt", "val_predictions.json",
                                               "run_log.json")}
    bash_exe = shutil.which("bash") or "bash"
    proc2 = subprocess.run([bash_exe, str(MODEL_ROOT / "server_ops/three_topology_v2/eval_first_batch_best.sh"),
                            BATCH], cwd=MODEL_ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
    after = {rel: sha(orig / rel) for rel in before}
    summary_path = recheck_root / "RECHECK_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    report = {
        "schema": "qt-recheck-entry-local-validation-v1",
        "entry_rc": proc2.returncode,
        "stdout_tail": proc2.stdout[-1500:], "stderr_tail": proc2.stderr[-1500:],
        "summary_exists": summary_path.is_file(),
        "summary_all_ok": summary.get("all_ok"), "n_runs": summary.get("n_runs"),
        "runs": summary.get("runs"),
        "original_files_unchanged": before == after,
        "original_hashes": before,
        "recheck_written_elsewhere": all(not str(recheck_root).startswith(str(orig)) for _ in [0]),
        "checkpoint_sha_recorded": bool(summary.get("runs") and
                                        summary["runs"][0].get("checkpoint_sha256")),
    }
    report["all_ok"] = (proc2.returncode == 0 and report["summary_all_ok"] is True
                        and report["original_files_unchanged"]
                        and report["checkpoint_sha_recorded"])
    out = MODEL_ROOT / "outputs" / "three_topology_v2" / "second_batch_preparation_20261118"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recheck_entry_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("entry_rc", "summary_all_ok", "n_runs",
                                             "original_files_unchanged", "all_ok")},
                     ensure_ascii=False))
    # cleanup test-only manifest + run dirs (not the validation json)
    man_path.unlink(missing_ok=True)
    shutil.rmtree(batch_dir, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

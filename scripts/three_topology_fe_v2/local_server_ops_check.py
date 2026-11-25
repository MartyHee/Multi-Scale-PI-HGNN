#!/usr/bin/env python
"""Local acceptance for the FE v2 server interface.

Uses tiny capped runs (1 train batch/epoch, 1 val batch/slice) in a temporary
output tree.  No full training, no test responses, no server connection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(MODEL_ROOT / "scripts" / "three_topology_fe_v2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

SERVER = MODEL_ROOT / "server_ops" / "three_topology_fe_v2"
JOBS_REAL = MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2"
TOPO = "topology1"
BASH = os.environ.get("SHELL") or shutil.which("bash") or "bash"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, env, timeout=900):
    return subprocess.run(cmd, cwd=str(MODEL_ROOT), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def base_env(root: Path) -> dict:
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(MODEL_ROOT),
        "PYTHON": PY,
        "THREE_TOPO_PACKAGE_ROOT": str(paths.PACKAGE_ROOT),
        "THREE_TOPO_PROCESSED_ROOT": str(paths.PROCESSED_ROOT),
        "THREE_TOPO_MANIFEST_ROOT": str(root / "manifests"),
        "THREE_TOPO_PREFLIGHT_ROOT": str(root / "preflight"),
        "THREE_TOPO_FORMAL_OUTPUT_ROOT": str(root / "formal"),
        "THREE_TOPO_JOBS_ROOT": str(root / "jobs"),
        "THREE_TOPO_OUTPUT_ROOT": str(root / "outputs"),
        "THREE_TOPO_SERVER_CONFIG": str(root / "server_config.sh"),
        "THREADS": "2",
    })
    return env


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_config(lr=0.5, bad=False) -> str:
    hgt = ("    hidden_dim: 8\n    num_layers: 1\n    heads: 3\n    dropout: 0.0\n"
           "    activation: relu\n    use_layer_norm: true\n    decoder_hidden_dims: [4]\n"
           "    global_mode: mesh_global\n    global_fusion: gated_residual\n")
    if not bad:
        hgt = hgt.replace("heads: 3", "heads: 2")
    return f"""data:
  package_root: null
  processed_root: null
  dtype: float32
  verify_hash: true
training:
  seeds: [20261117, 20261118]
  batch_size: 1
  epochs: 4
  lr: {lr}
  weight_decay: 0.0
  early_stopping_patience: 10
  min_delta: 0.0
  grad_clip: 5.0
  selection_metric: three_slice_graph_equal_task_equal_component_equal_normalized_mse
  physics_loss_weight: 0.0
  max_train_batches_per_epoch: 1
  val_batch_size: 1
  max_val_batches: 1
  diagnostic: true
  device: cpu
models:
  mlp:
    hidden_dims: [8]
    dropout: 0.0
    activation: relu
    use_batch_norm: false
  hgt_global:
{hgt}ml:
  ridge:
    alpha: 1.0
  hist_gb:
    max_iter: 2
    learning_rate: 0.1
    max_depth: 2
"""


def job_yaml(job, batch, cfg_rel, tasks) -> str:
    out = [f"job_name: {job}", f"batch_id: {batch}", "mode: formal",
           f"train_config: {cfg_rel}", "tasks:"]
    for t in tasks:
        out.append(f"  - topology: {t['topology']}")
        out.append(f"    model: {t['model']}")
        if "seeds" in t:
            out.append(f"    seeds: {t['seeds']}")
        if "seed" in t:
            out.append(f"    seed: {t['seed']}")
        out.append(f"    device: {t['device']}")
        out.append(f"    worker: {t['worker']}")
    return "\n".join(out) + "\n"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = paths.OUTPUT_ROOT / "_local_server_ops_check" / ts
    root.mkdir(parents=True, exist_ok=False)
    env = base_env(root)
    cfg_rel = f"outputs/three_topology_fe_v2_9216/_local_server_ops_check/{ts}/test_config.yaml"
    cfg_bad_rel = f"outputs/three_topology_fe_v2_9216/_local_server_ops_check/{ts}/test_config_bad.yaml"
    write(root / "test_config.yaml", test_config())
    write(root / "test_config_bad.yaml", test_config(bad=True))
    write(root / "server_config.sh", f"""PACKAGE_ROOT="{paths.PACKAGE_ROOT.as_posix()}"
PROCESSED_ROOT="{paths.PROCESSED_ROOT.as_posix()}"
PYTHON="{Path(PY).as_posix()}"
THREADS=2
PROJECT_ROOT="{MODEL_ROOT.as_posix()}"
""")
    write(root / "jobs/good.yaml", job_yaml(
        "local_good", "local_good_batch", cfg_rel,
        [{"topology": TOPO, "model": "mlp", "seeds": "[20261117, 20261118]",
          "device": "gpu", "worker": "local_mlp"}]))
    write(root / "jobs/job.yaml", job_yaml(
        "local_job", "local_job_batch", cfg_rel,
        [{"topology": TOPO, "model": "mlp", "seeds": "[20261117, 20261118]",
          "device": "gpu", "worker": "local_job_mlp"}]))
    write(root / "jobs/fail.yaml", job_yaml(
        "local_fail", "local_fail_batch", cfg_bad_rel,
        [{"topology": TOPO, "model": "mlp", "seeds": "[20261117]",
          "device": "gpu", "worker": "local_fail_mlp"},
         {"topology": TOPO, "model": "hgt_global", "seeds": "[20261117]",
          "device": "gpu", "worker": "local_fail_hgt"}]))

    results = {}

    def record(name, ok, detail=None):
        results[name] = {"ok": bool(ok), "detail": detail}
        print(("PASS" if ok else "FAIL"), name, detail if detail else "", flush=True)

    # ---- invalid YAML / empty tasks ----
    write(root / "bad_unknown.yaml", "job_name: x\nbatch_id: b\nbad: 1\ntrain_config: x\ntasks: []\n")
    p = run([PY, "-B", str(SERVER / "_job_parse.py"), str(root / "bad_unknown.yaml")], env)
    record("invalid_yaml_nonzero", p.returncode != 0, p.returncode)
    write(root / "empty.yaml", f"job_name: x\nbatch_id: b\ntrain_config: {cfg_rel}\ntasks: []\n")
    p = run([PY, "-B", str(SERVER / "_job_parse.py"), str(root / "empty.yaml")], env)
    record("empty_tasks_nonzero", p.returncode != 0, p.returncode)

    # ---- manifests ----
    for name, expect in (("good", 2), ("job", 2), ("fail", 2)):
        p = run([PY, "-B", str(SERVER / "_batch_manifest.py"),
                 "--job", str(root / f"jobs/{name}.yaml"), "--write", "--expect", str(expect)], env)
        record(f"manifest_{name}", p.returncode == 0, p.stdout.strip()[-120:])

    # ---- PASS records ----
    for batch, checked in (("local_good_batch", ["topology1/mlp"]),
                           ("local_job_batch", ["topology1/mlp"]),
                           ("local_fail_batch", ["topology1/mlp", "topology1/hgt_global"])):
        cmd = [PY, "-B", str(SERVER / "_preflight_gate.py"), "--write",
               "--batch-id", batch, "--project-root", str(MODEL_ROOT), "--device", "0",
               "--visible", "1"]
        for c in checked:
            cmd += ["--checked", c]
        p = run(cmd, env)
        record(f"pass_{batch}", p.returncode == 0, p.stdout.strip()[-100:])

    # ---- formal run: good seed 17 attempt 1 ----
    cmd = [PY, "-B", str(MODEL_ROOT / "scripts/three_topology_fe_v2/train_fe_v2.py"),
           "--mode", "formal", "--config", str(root / "test_config.yaml"),
           "--batch-id", "local_good_batch", "--worker", "local_mlp", "--topology", TOPO,
           "--model", "mlp", "--seed", "20261117", "--device", "cuda:0", "--attempt", "1"]
    p = run(cmd, env)
    run_dir = root / "formal" / "local_good_batch" / TOPO / "local_mlp" / "mlp_seed20261117" / "attempt_01"
    record("formal_run_ok", p.returncode == 0 and (run_dir / "summary.json").is_file(),
           p.stdout.strip()[-200:] if p.returncode else str(run_dir))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
        if (run_dir / "summary.json").is_file() else {}
    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8")) \
        if (run_dir / "run_log.json").is_file() else {}

    # duplicate refusal
    p = run(cmd, env)
    record("duplicate_refused", p.returncode == 3, p.stderr.strip()[-160:])

    # missing preflight
    pf = root / "preflight" / "local_good_batch" / "PREFLIGHT_PASS.json"
    saved = pf.read_bytes()
    pf.unlink()
    cmd18 = [c if c != "20261117" else "20261118" for c in cmd]
    p = run(cmd18, env)
    record("missing_preflight_refused", p.returncode == 4, p.stderr.strip()[-160:])
    pf.write_bytes(saved)

    # hash change invalidates PASS
    cfg = root / "test_config.yaml"
    original_cfg = cfg.read_text(encoding="utf-8")
    cfg.write_text(original_cfg.replace("lr: 0.5", "lr: 0.6"), encoding="utf-8")
    p = run([PY, "-B", str(SERVER / "_preflight_gate.py"), "--check",
             "--batch-id", "local_good_batch", "--project-root", str(MODEL_ROOT)], env)
    record("config_change_invalidates_pass", p.returncode == 4, p.stderr.strip()[-160:])
    cfg.write_text(original_cfg, encoding="utf-8")

    # best vs last and export provenance
    best_epoch = summary.get("best", {}).get("epoch_zero_based")
    completed = summary.get("completed_epochs")
    record("best_differs_from_last", best_epoch is not None and completed is not None
           and best_epoch < completed - 1, {"best_epoch": best_epoch, "completed": completed})
    record("final_eval_from_best",
           summary.get("final_eval_checkpoint_sha256")
           == summary.get("best", {}).get("checkpoint_sha256"),
           summary.get("final_eval_source"))

    # re-evaluate the best checkpoint; compare exported predictions + batch invariance
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{TOPO}.json")
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    val_ids = {
        "design_val": protocol.ids_for(idx, prot, TOPO, "A", "val")[:4],
        "load_val": protocol.ids_for(idx, prot, TOPO, "B", "val")[:4],
        "joint_val": protocol.ids_for(idx, prot, TOPO, "C", "val")[:4],
    }
    val_ds = {k: FeV2Dataset(idx, v, st, dtype=torch.float32, normalize=True)
              for k, v in val_ids.items()}
    import yaml
    local = json.loads(json.dumps(yaml.safe_load((root / "test_config.yaml").read_text(encoding="utf-8"))))
    local["training"]["seed"] = 20261117
    local["models"]["mlp"] = schema.model_cfg_with_dims("mlp", local["models"]["mlp"])
    model = registry.build_gradient_model("mlp", local["models"]["mlp"])
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(torch.device("cuda:0"))
    ev1 = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=None)
    r1 = ev1(model, 1, torch.device("cuda:0"), collect_predictions=True)
    ev2 = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=None)
    r2 = ev2(model, 2, torch.device("cuda:0"), collect_predictions=True)
    record("batch_size_invariant",
           abs(r1["selection_metric"] - r2["selection_metric"]) < 1e-12,
           {"bs1": r1["selection_metric"], "bs2": r2["selection_metric"]})
    pred_file = run_dir / "val_predictions_design_val.json"
    saved_pred = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.is_file() else []
    exported = r1["exported"][:1]
    ok_pred = bool(saved_pred and exported) and np.allclose(
        np.asarray(saved_pred[0]["pred_disp"]), np.asarray(exported[0]["pred_disp"]),
        rtol=0, atol=1e-6) and np.allclose(
        np.asarray(saved_pred[0]["pred_force"]), np.asarray(exported[0]["pred_force"]),
        rtol=0, atol=1e-4)
    record("predictions_from_best", ok_pred, saved_pred[0]["sample_id"] if saved_pred else None)
    agg = json.loads((run_dir / "slice_metrics.json").read_text(encoding="utf-8"))
    mean = float(np.mean([v["selection_metric"] for v in agg["slices"].values()]))
    record("slice_equal_mean", abs(mean - summary["selection_metric"]) < 1e-12,
           {"mean": mean, "summary": summary["selection_metric"]})

    # ---- resume: valid and invalid ----
    best_sha = sha(run_dir / "checkpoints" / "best.pt")
    last_sha = sha(run_dir / "checkpoints" / "last.pt")
    resume_cmd = cmd + ["--attempt", "2", "--resume-from",
                        str(run_dir / "checkpoints" / "last.pt")]
    p = run(resume_cmd, env)
    resume_dir = root / "formal" / "local_good_batch" / TOPO / "local_mlp" / "mlp_seed20261117" / "attempt_02"
    rlog = json.loads((resume_dir / "run_log.json").read_text(encoding="utf-8")) \
        if (resume_dir / "run_log.json").is_file() else {}
    record("resume_valid", p.returncode == 0 and bool(rlog.get("resumed")),
           {"rc": p.returncode, "resumed": rlog.get("resumed")})
    record("resume_inherits_best",
           rlog.get("best", {}).get("source") == "inherited"
           and sha(run_dir / "checkpoints" / "best.pt") == best_sha
           and sha(run_dir / "checkpoints" / "last.pt") == last_sha,
           rlog.get("best", {}).get("source"))
    bad_cmd = cmd18 + ["--attempt", "2", "--resume-from",
                       str(run_dir / "checkpoints" / "last.pt")]
    p = run(bad_cmd, env)
    record("resume_identity_refused", p.returncode != 0, p.stderr.strip()[-200:])

    # ---- ML unified three-slice evaluation (smoke) ----
    ml_out = root / "smoke_ml"
    p = run([PY, "-B", str(MODEL_ROOT / "scripts/three_topology_fe_v2/train_fe_v2.py"),
             "--mode", "smoke", "--config", str(root / "test_config.yaml"),
             "--model", "ridge", "--topology", TOPO, "--seed", "0", "--device", "cpu",
             "--out", str(ml_out)], env)
    ml_dir = ml_out / TOPO / "ridge" / "seed_0"
    ml_sum = json.loads((ml_dir / "summary.json").read_text(encoding="utf-8")) \
        if (ml_dir / "summary.json").is_file() else {}
    ml_agg = json.loads((ml_dir / "slice_metrics.json").read_text(encoding="utf-8")) \
        if (ml_dir / "slice_metrics.json").is_file() else {}
    ml_mean = float(np.mean([v["selection_metric"] for v in ml_agg.get("slices", {}).values()]))
    record("ml_three_slice_equal_mean",
           p.returncode == 0 and abs(ml_mean - ml_sum.get("selection_metric", 0)) < 1e-12
           and all((ml_dir / f"val_predictions_{s}.json").is_file()
                   for s in ("design_val", "load_val", "joint_val")),
           {"mean": ml_mean, "summary": ml_sum.get("selection_metric")})
    record("ml_no_fake_epoch",
           ml_sum.get("best") is None and ml_sum.get("completed_epochs") is None
           and ml_sum.get("stop_reason") == "single_fit_no_epochs",
           ml_sum.get("stop_reason"))

    # ---- job-level success: run_job, summarize, collect ----
    registered = JOBS_REAL / f"_local_job_{ts}.yaml"
    shutil.copy(root / "jobs/job.yaml", registered)
    try:
        p = run([BASH, str(SERVER / "run_job.sh"), str(registered)], env)
        job_rc = p.returncode
    finally:
        registered.unlink(missing_ok=True)
    p_sum = run([PY, "-B", str(SERVER / "summarize_batch.py"), "--batch", "local_job_batch",
                 "--project-root", str(MODEL_ROOT), "--output-root", str(root / "formal"), "--json"], env)
    job_sum = json.loads(p_sum.stdout) if p_sum.stdout.strip().startswith("{") else {}
    record("job_success_and_summary",
           job_rc == 0 and p_sum.returncode == 0 and job_sum.get("overall_ok") is True,
           {"job_rc": job_rc, "done": job_sum.get("done"),
            "job_out": (p.stdout or "")[-300:], "job_err": (p.stderr or "")[-300:],
            "sum_err": (p_sum.stderr or "")[-200:]})
    tar = root / "artifacts" / "local_job.tar.gz"
    p_col = run([BASH, str(SERVER / "collect_artifacts.sh"), "local_job_batch"], env | {"OUT_TAR": tar.as_posix()})
    record("collect_complete", p_col.returncode == 0 and tar.is_file(),
           {"rc": p_col.returncode, "tar": str(tar), "exists": tar.is_file(),
            "out": (p_col.stdout or "")[-200:], "err": (p_col.stderr or "")[-300:]})

    # missing task is detected as incomplete (temporarily move one run dir)
    missing_dir = (root / "formal" / "local_job_batch" / TOPO / "local_job_mlp"
                   / "mlp_seed20261118" / "attempt_01")
    moved = missing_dir.with_name("attempt_01__moved_for_missing_test")
    missing_dir.rename(moved)
    try:
        p_missing = run([PY, "-B", str(SERVER / "summarize_batch.py"), "--batch", "local_job_batch",
                         "--project-root", str(MODEL_ROOT), "--output-root", str(root / "formal"), "--json"], env)
        miss_sum = json.loads(p_missing.stdout) if p_missing.stdout.strip().startswith("{") else {}
        record("missing_task_detected",
               p_missing.returncode != 0 and miss_sum.get("overall_ok") is False
               and miss_sum.get("done") == 1,
               {"rc": p_missing.returncode, "done": miss_sum.get("done")})
    finally:
        moved.rename(missing_dir)

    # ---- subtask failure propagation + incomplete collection ----
    registered = JOBS_REAL / f"_local_fail_{ts}.yaml"
    shutil.copy(root / "jobs/fail.yaml", registered)
    try:
        p = run([BASH, str(SERVER / "run_job.sh"), str(registered)], env)
        fail_rc = p.returncode
    finally:
        registered.unlink(missing_ok=True)
    p_sum = run([PY, "-B", str(SERVER / "summarize_batch.py"), "--batch", "local_fail_batch",
                 "--project-root", str(MODEL_ROOT), "--output-root", str(root / "formal"), "--json"], env)
    fail_sum = json.loads(p_sum.stdout) if p_sum.stdout.strip().startswith("{") else {}
    record("subtask_failure_propagates",
           fail_rc != 0 and p_sum.returncode != 0 and fail_sum.get("overall_ok") is False
           and fail_sum.get("done") == 1,
           {"job_rc": fail_rc, "done": fail_sum.get("done")})
    tar2 = root / "artifacts" / "local_fail.tar.gz"
    p_col = run([BASH, str(SERVER / "collect_artifacts.sh"), "local_fail_batch"], env | {"OUT_TAR": tar2.as_posix()})
    record("collect_incomplete_nonzero", p_col.returncode != 0 and tar2.is_file(),
           {"rc": p_col.returncode, "tar": str(tar2), "exists": tar2.is_file(),
            "out": (p_col.stdout or "")[-200:], "err": (p_col.stderr or "")[-300:]})

    # ---- 18 + 6 task identity uniqueness ----
    man18 = json.loads((JOBS_REAL / "manifests/fv2_first_20261125_01.json").read_text(encoding="utf-8"))
    man6 = json.loads((JOBS_REAL / "manifests/fv2_ml_20261125_01.json").read_text(encoding="utf-8"))
    keys18 = {(t["worker"], t["topology"], t["model"], str(t["seed"])) for t in man18["tasks"]}
    keys6 = {(t["worker"], t["topology"], t["model"], str(t["seed"])) for t in man6["tasks"]}
    record("batch18_unique", man18["n_tasks"] == 18 and len(keys18) == 18, man18["n_tasks"])
    record("batch6_unique", man6["n_tasks"] == 6 and len(keys6) == 6, man6["n_tasks"])
    record("posix_paths", all("\\" not in t["job_file"] for t in man18["tasks"]),
           man18["source_jobs"])

    # ---- CPU isolation / threads mechanism ----
    p = run([PY, "-c", "import torch; print(torch.cuda.is_available())"], env | {"CUDA_VISIBLE_DEVICES": "-1"})
    record("cpu_child_hides_cuda", p.stdout.strip() == "False", p.stdout.strip())
    p = run([PY, "-c", "import os; print(os.environ.get('OMP_NUM_THREADS'))"],
            env | {"OMP_NUM_THREADS": "2"})
    record("thread_limit_env", p.stdout.strip() == "2", p.stdout.strip())

    # ---- manifest LF/CRLF stability ----
    a = root / "lf.txt"; b = root / "crlf.txt"
    a.write_bytes(b"a\nb\n"); b.write_bytes(b"a\r\nb\r\n")
    import sys as _sys
    _sys.path.insert(0, str(SERVER))
    import _batch_manifest as bm  # noqa: PLC0415
    record("lf_crlf_same_hash", bm.sha_lf(a) == bm.sha_lf(b), None)

    payload = {"schema": "qt-fe-v2-local-server-ops-check-v1",
               "created_at": ts, "root": str(root),
               "passed": sum(1 for v in results.values() if v["ok"]),
               "failed": [k for k, v in results.items() if not v["ok"]],
               "results": results}
    (root / "local_check_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "failed": payload["failed"],
                      "root": str(root)}, ensure_ascii=False, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

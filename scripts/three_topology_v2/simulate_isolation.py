#!/usr/bin/env python
"""Local simulation of batch isolation, manifest-driven ops and packaging.

No GPU, no training, no real results. It builds a throw-away project tree with
job YAMLs + a frozen batch manifest and fake run dirs, then verifies that:

  * any job YAML locates the full frozen task list; duplicate identities are
    refused; editing a YAML after freezing is detected (stale manifest);
  * summarize reports missing / failed / identity-mismatched tasks as incomplete
    and never counts extra run dirs as this batch's results;
  * collect_artifacts stages only manifest tasks, includes the manifest and task
    list, flags extra dirs, refuses to overwrite, and exits non-zero for an
    incomplete batch;
  * the preflight gate rejects a stale PASS after any identity change;
  * the first-batch (8-task) manifest/JAML shape still works;
  * atomic run-dir refusal and worker isolation.

Writes outputs/three_topology_v2/simulate_isolation/sim_report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

MODEL_ROOT = Path(__file__).resolve().parents[2]
SERVER_OPS = MODEL_ROOT / "server_ops" / "three_topology_v2"
MANIFEST_TOOL = SERVER_OPS / "_batch_manifest.py"
GATE = SERVER_OPS / "_preflight_gate.py"
RESUME_CHECK = SERVER_OPS / "_resume_check.py"

GRAD_FILES = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
              "train_log.csv", "driver.log", "val_predictions.json",
              "checkpoints/best.pt", "checkpoints/last.pt"]
ML_FILES = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
            "driver.log", "val_predictions.json"]


def posix(p):
    s = str(Path(p)).replace(chr(92), "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/{m.group(1).lower()}/{m.group(2)}" if m else s


def run(cmd, env=None, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def write_job(fake: Path, name: str, batch: str, spec: dict):
    p = fake / "remote_jobs" / "three_topology_v2" / f"{name}.yaml"
    lines = [f"job_name: {name}", f"batch_id: {batch}", "protocol: split",
             "split_id: cross_design_topology1", "fold_id: cd_fold_1",
             "train_config: configs/three_topology_v2/formal_baselines.yaml", "tasks:"]
    for t in spec["tasks"]:
        lines.append(f"  - model: {t['model']}")
        if "seeds" in t:
            lines.append(f"    seeds: [{', '.join(str(s) for s in t['seeds'])}]")
        lines.append(f"    device: {t.get('device', 'gpu')}")
        lines.append(f"    worker: {t['worker']}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_project(fake: Path, batch: str, jobs: dict, expected: int):
    (fake / "scripts" / "three_topology_v2").mkdir(parents=True, exist_ok=True)
    (fake / "scripts" / "three_topology_v2" / "train_formal.py").write_text("", encoding="utf-8")
    (fake / "configs" / "three_topology_v2").mkdir(parents=True, exist_ok=True)
    cfg = {
        "training": {"epochs": 60, "max_train_batches_per_epoch": None,
                     "batch_size": 4, "lr": 0.001},
        "models": {m: {"hidden_dim": 128} for m in
                   ["gcn", "graphsage", "gat", "gin", "rgcn", "hgt", "mlp", "hgt_global"]},
        "ml": {"ridge": {"alpha": 1.0}, "hist_gb": {"max_iter": 10}},
    }
    import yaml
    (fake / "configs" / "three_topology_v2" / "formal_baselines.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    root = fake / "processed" / "three_topology_training_v2"
    for sub in ("splits", "stats", "stats_inner", "manifests"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    fold = {"fold_id": "cd_fold_1",
            "train_sample_ids": [f"tr{i}" for i in range(84)],
            "validation_sample_ids": [f"va{i}" for i in range(24)],
            "test_sample_ids": [f"te{i}" for i in range(36)]}
    (root / "splits" / "split_cross_design_topology1.json").write_text(
        json.dumps({"folds": [fold]}), encoding="utf-8")
    (root / "stats" / "cross_design_topology1").mkdir(parents=True, exist_ok=True)
    (root / "stats" / "cross_design_topology1" / "cd_fold_1.json").write_text(
        json.dumps({"columns": {}}), encoding="utf-8")
    data = fake / "data" / "processed" / "three_topology_graph_v1"
    data.mkdir(parents=True, exist_ok=True)
    (data / "samples.jsonl").write_text("{}", encoding="utf-8")
    (data / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    for name, spec in jobs.items():
        write_job(fake, name, batch, spec)
    env = dict(os.environ, PROJECT_ROOT=posix(fake))
    first_job = next(iter(jobs))
    rc, out = run([sys.executable, "-B", str(MANIFEST_TOOL), "--job",
                   str(fake / "remote_jobs" / "three_topology_v2" / f"{first_job}.yaml"),
                   "--write", "--expect", str(expected)], env=env)
    if rc != 0:
        raise RuntimeError(f"manifest --write failed: {out}")
    return fake / "outputs" / "three_topology_v2" / "first_batch" / batch


def make_runs(base: Path, batch: str, tasks, done=True):
    for worker, run_name, kind in tasks:
        d = base / worker / run_name
        d.mkdir(parents=True, exist_ok=True)
        files = GRAD_FILES if kind == "gradient" else ML_FILES
        for f in files:
            p = d / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")
        if kind == "ml":
            (d / f"ml_{run_name}.pkl").write_text("x", encoding="utf-8")
        model = run_name.split("_seed")[0]
        seed = run_name.split("_seed")[1] if "_seed" in run_name else "-"
        (d / "STATUS.json").write_text(json.dumps(
            {"state": "done" if done else "failed", "batch_id": batch, "worker_id": worker,
             "model": model, "seed": seed, "exit_code": 0 if done else 1}), encoding="utf-8")
        (d / "TASK.json").write_text(json.dumps(
            {"batch_id": batch, "worker_id": worker, "model": model, "seed": seed}), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(
            {"all_pass": done, "models": {f"{model}/seed_{seed}": {"selection_metric": 0.1}}}),
            encoding="utf-8")


def manifest_tasks(fake: Path, job: str):
    env = dict(os.environ, PROJECT_ROOT=posix(fake))
    rc, out = run([sys.executable, "-B", str(MANIFEST_TOOL), "--job",
                   str(fake / "remote_jobs" / "three_topology_v2" / job), "--paths"], env=env)
    rows = [line.split("\t") for line in out.strip().splitlines() if line.strip()]
    return rc, out, rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    reps = {}
    tmp = Path(tempfile.mkdtemp(prefix="sim_iso_"))
    py = sys.executable
    bash_exe = shutil.which("bash") or "bash"
    models = ["gcn", "graphsage", "gat", "gin", "rgcn", "hgt"]
    batch2 = "SIMB2"
    jobs2 = {m: {"tasks": [{"model": m, "seeds": [20261117, 20261118, 20261119],
                            "worker": f"t{i+1}_{m}"}]} for i, m in enumerate(models)}
    fake = tmp / "proj2"
    base = build_project(fake, batch2, jobs2, 18)
    env = dict(os.environ, PROJECT_ROOT=posix(fake), BATCH_ID=batch2, PYTHON=py)
    all_tasks = [(r[0], r[1], r[5]) for r in manifest_tasks(fake, "gcn.yaml")[2]]
    reps["manifest_18_tasks"] = len(all_tasks) == 18
    # any job locates the same full list
    reps["manifest_any_job_same"] = (manifest_tasks(fake, "gat.yaml")[2] == manifest_tasks(fake, "gcn.yaml")[2])

    # ---- cross-platform path representation + effective config freeze ----
    man2 = json.loads((fake / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch2}.json").read_text(encoding="utf-8"))
    job_files = {t["job_file"] for t in man2["tasks"]}
    bs = chr(92)
    reps["manifest_job_files_posix"] = all(bs not in jf and PurePosixPath(jf).as_posix() == jf
                                           for jf in job_files)
    reps["windows_posix_equivalent"] = (
        PureWindowsPath("remote_jobs" + bs + "three_topology_v2" + bs + "gcn.yaml").as_posix()
        == PurePosixPath("remote_jobs/three_topology_v2/gcn.yaml").as_posix())
    reps["effective_defaults"] = all(t["epochs"] == 60 and t["max_batches"] is None
                                     for t in man2["tasks"])

    gcn_yaml = fake / "remote_jobs" / "three_topology_v2" / "gcn.yaml"
    gcn_orig = gcn_yaml.read_text(encoding="utf-8")
    cfg_path = fake / "configs" / "three_topology_v2" / "formal_baselines.yaml"
    cfg_orig = cfg_path.read_text(encoding="utf-8")
    NL = chr(10)

    def verify_rc():
        return run([sys.executable, "-B", str(MANIFEST_TOOL), "--job", str(gcn_yaml), "--verify"], env=env)[0]

    def mutate_and_verify(label, text):
        gcn_yaml.write_text(text, encoding="utf-8")
        rc = verify_rc()
        gcn_yaml.write_text(gcn_orig, encoding="utf-8")
        reps[label] = rc != 0

    mutate_and_verify("verify_detects_toplevel_epochs",
                      gcn_orig.replace("fold_id: cd_fold_1", "fold_id: cd_fold_1" + NL + "epochs: 7"))
    mutate_and_verify("verify_detects_toplevel_max_batches",
                      gcn_orig.replace("fold_id: cd_fold_1",
                                       "fold_id: cd_fold_1" + NL + "max_batches: 1"))
    mutate_and_verify("verify_detects_protocol_change",
                      gcn_orig.replace("protocol: split",
                                       "protocol: inner" + NL + "outer_fold: cd_fold_1" + NL + "inner_id: inner_1"))
    mutate_and_verify("verify_detects_task_epochs",
                      gcn_orig.replace("    worker: t1_gcn", "    worker: t1_gcn" + NL + "    epochs: 7", 1))
    mutate_and_verify("verify_detects_task_max_batches",
                      gcn_orig.replace("    worker: t1_gcn", "    worker: t1_gcn" + NL + "    max_batches: 1", 1))
    mutate_and_verify("verify_detects_resume_from",
                      gcn_orig.replace("    worker: t1_gcn",
                                       "    worker: t1_gcn" + NL + "    resume_from: /tmp/other.pt", 1))
    mutate_and_verify("verify_detects_fold_change",
                      gcn_orig.replace("fold_id: cd_fold_1", "fold_id: cd_fold_2"))
    mutate_and_verify("verify_detects_split_change",
                      gcn_orig.replace("split_id: cross_design_topology1",
                                       "split_id: cross_load_field_topology1"))
    cfg_path.write_text(cfg_orig + "# changed" + NL, encoding="utf-8")
    rc_cfg = verify_rc()
    cfg_path.write_text(cfg_orig, encoding="utf-8")
    reps["verify_detects_config_bytes"] = rc_cfg != 0
    cfg_path.write_text(cfg_orig.replace("epochs: 60", "epochs: 61"), encoding="utf-8")
    rc_cfg2 = verify_rc()
    cfg_path.write_text(cfg_orig, encoding="utf-8")
    reps["verify_detects_config_default_change"] = rc_cfg2 != 0

    # v1 manifest is not silently accepted
    man_path2 = fake / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch2}.json"
    man_text = man_path2.read_text(encoding="utf-8")
    man_path2.write_text(man_text.replace("qt-three-topology-batch-manifest-v2",
                                          "qt-three-topology-batch-manifest-v1"), encoding="utf-8")
    rc_v1 = verify_rc()
    man_path2.write_text(man_text, encoding="utf-8")
    reps["v1_manifest_rejected"] = rc_v1 != 0

    # duplicate identity refused at freeze time
    dup_job = fake / "remote_jobs" / "three_topology_v2" / "dup.yaml"
    dup_job.write_text("job_name: dup\nbatch_id: SIMB2\nprotocol: split\n"
                       "split_id: cross_design_topology1\nfold_id: cd_fold_1\n"
                       "train_config: configs/three_topology_v2/formal_baselines.yaml\ntasks:\n"
                       "  - model: gcn\n    seeds: [20261117]\n    device: gpu\n    worker: t1_gcn\n",
                       encoding="utf-8")
    rc_dup, out_dup = run([py, "-B", str(MANIFEST_TOOL), "--job", str(dup_job), "--write"], env=env)
    reps["duplicate_identity_refused"] = rc_dup != 0 and "duplicate" in out_dup.lower()
    dup_job.unlink()

    # stale manifest detected after a YAML edit
    gcn_yaml = fake / "remote_jobs" / "three_topology_v2" / "gcn.yaml"
    original = gcn_yaml.read_text(encoding="utf-8")
    gcn_yaml.write_text(original.replace("20261119", "20261120"), encoding="utf-8")
    rc_stale, out_stale = run([py, "-B", str(MANIFEST_TOOL), "--job", str(gcn_yaml), "--verify"], env=env)
    gcn_yaml.write_text(original, encoding="utf-8")
    reps["stale_manifest_detected"] = rc_stale != 0 and "changed" in out_stale.lower()

    # make all 18 runs done
    make_runs(base, batch2, all_tasks, done=True)

    # ---- external resume cannot bypass the frozen batch ----
    import torch
    manr = json.loads((fake / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch2}.json").read_text(encoding="utf-8"))
    t0 = next(t for t in manr["tasks"] if t["model"] == "gcn" and t["seed"] == "20261117")
    orig_ck = base / "t1_gcn" / "gcn_seed20261117" / "checkpoints"
    orig_ck.mkdir(parents=True, exist_ok=True)
    data_dir = fake / "data" / "processed" / "three_topology_graph_v1"
    ident = {
        "model_name": "gcn", "seed": 20261117,
        "split_id": manr["split_id"], "fold_id": manr["fold_id"],
        "outer_fold": None, "inner_fold": None,
        "stats_sha256": manr["input_identity"]["stats_sha256"],
        "train_sample_list_sha256": manr["input_identity"]["train_sample_list_sha256"],
        "val_sample_list_sha256": manr["input_identity"]["val_sample_list_sha256"],
        "model_config_sha256": t0["model_config_sha256"],
        "samples_jsonl_sha256": hashlib.sha256((data_dir / "samples.jsonl").read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256((data_dir / "dataset_manifest.json").read_bytes()).hexdigest(),
    }
    ck_path = orig_ck / "last.pt"
    torch.save({"identity": ident, "model_state": {}}, ck_path)

    def resume_yaml(path, seed="20261117", model="gcn", worker="t1_gcn_resume", ckpt=None):
        ck = ckpt or ck_path
        path.write_text(
            "job_name: gcn_resume" + chr(10) + f"batch_id: {batch2}" + chr(10) +
            "protocol: split" + chr(10) + "split_id: cross_design_topology1" + chr(10) +
            "fold_id: cd_fold_1" + chr(10) +
            "train_config: configs/three_topology_v2/formal_baselines.yaml" + chr(10) +
            "tasks:" + chr(10) + f"  - model: {model}" + chr(10) + f"    seeds: [{seed}]" + chr(10) +
            "    device: gpu" + chr(10) + f"    worker: {worker}" + chr(10) +
            f"    resume_from: {ck}" + chr(10), encoding="utf-8")

    def resume_check(job_path):
        return run([sys.executable, "-B", str(RESUME_CHECK), "--job", str(job_path),
                    "--batch-id", batch2, "--project-root", posix(fake),
                    "--data-root", posix(fake)], env=env)

    ext = tmp / "resume_ok.yaml"; resume_yaml(ext)
    rc_ok, out_ok = resume_check(ext)
    reps["resume_valid_external_ok"] = rc_ok == 0 and '"ok": true' in out_ok

    ext_bad = tmp / "resume_wrong_seed.yaml"; resume_yaml(ext_bad, seed="20261118")
    reps["resume_wrong_seed_rejected"] = resume_check(ext_bad)[0] != 0

    ext_model = tmp / "resume_wrong_model.yaml"; resume_yaml(ext_model, model="gat")
    reps["resume_wrong_model_rejected"] = resume_check(ext_model)[0] != 0

    ext_worker = tmp / "resume_same_worker.yaml"; resume_yaml(ext_worker, worker="t1_gcn")
    reps["resume_original_worker_rejected"] = resume_check(ext_worker)[0] != 0

    outside = tmp / "outside.pt"; shutil.copy2(ck_path, outside)
    ext_out = tmp / "resume_outside.yaml"; resume_yaml(ext_out, ckpt=outside)
    reps["resume_outside_batch_rejected"] = resume_check(ext_out)[0] != 0

    norm = tmp / "resume_no_resume.yaml"
    norm.write_text("job_name: ext_normal" + chr(10) + f"batch_id: {batch2}" + chr(10) +
                    "protocol: split" + chr(10) + "split_id: cross_design_topology1" + chr(10) +
                    "fold_id: cd_fold_1" + chr(10) +
                    "train_config: configs/three_topology_v2/formal_baselines.yaml" + chr(10) +
                    "tasks:" + chr(10) + "  - model: gcn" + chr(10) + "    seeds: [20261117]" + chr(10) +
                    "    device: gpu" + chr(10) + "    worker: t_ext" + chr(10), encoding="utf-8")
    reps["external_normal_job_rejected"] = resume_check(norm)[0] != 0

    bad_ident = dict(ident); bad_ident["model_name"] = "mlp"
    bad_ck = tmp / "bad_identity.pt"; torch.save({"identity": bad_ident}, bad_ck)
    ext_bad_id = tmp / "resume_bad_identity.yaml"; resume_yaml(ext_bad_id, ckpt=bad_ck)
    reps["resume_checkpoint_identity_rejected"] = resume_check(ext_bad_id)[0] != 0

    # run_job must reject external normal / malformed / empty jobs before training
    server_cfg = fake / "server_config_test.sh"
    server_cfg.write_text("DATA_ROOT=" + chr(34) + posix(fake) + chr(34) + chr(10) +
                          "PYTHON=" + chr(34) + sys.executable + chr(34) + chr(10) +
                          "THREADS=4" + chr(10) + "PROJECT_ROOT=" + chr(34) + posix(fake) + chr(34) + chr(10),
                          encoding="utf-8")
    rj_env = dict(os.environ, PROJECT_ROOT=posix(fake),
                  PYTHON=sys.executable, THREE_TOPO_SERVER_CONFIG=posix(server_cfg),
                  THREE_TOPO_SKIP_PREFLIGHT="1", CUDA_VISIBLE_DEVICES="0")
    run_job = [bash_exe, posix(SERVER_OPS / "run_job.sh")]
    rc_ext, _ = run(run_job + [str(norm)], env=rj_env)
    reps["run_job_external_normal_rejected"] = rc_ext != 0 and not (base / "t_ext").exists()
    bad_yaml = tmp / "bad.yaml"; bad_yaml.write_text("job_name: [broken", encoding="utf-8")
    rc_bad, _ = run(run_job + [str(bad_yaml)], env=rj_env)
    reps["run_job_invalid_yaml_rejected"] = rc_bad != 0
    empty_yaml = tmp / "empty.yaml"
    empty_yaml.write_text("job_name: empty" + chr(10) + f"batch_id: {batch2}" + chr(10) +
                          "protocol: split" + chr(10) + "split_id: cross_design_topology1" + chr(10) +
                          "fold_id: cd_fold_1" + chr(10) +
                          "train_config: configs/three_topology_v2/formal_baselines.yaml" + chr(10) +
                          "tasks: []" + chr(10), encoding="utf-8")
    rc_empty, _ = run(run_job + [str(empty_yaml)], env=rj_env)
    reps["run_job_empty_tasks_rejected"] = rc_empty != 0 and not (base / "empty").exists()

    # monitor still reads the frozen task list
    rc_mon, out_mon = run([bash_exe, posix(SERVER_OPS / "monitor.sh"), batch2],
                          env=dict(os.environ, PROJECT_ROOT=posix(fake), PYTHON=sys.executable))
    reps["monitor_reads_manifest"] = rc_mon == 0 and "gcn" in out_mon and "20261117" in out_mon

    summarize = [py, "-B", str(SERVER_OPS / "summarize_batch.py"), "--batch-dir", str(base),
                 "--project-root", posix(fake), "--json"]
    rc, out = run(summarize, env=env)
    reps["summary_all_done_ok"] = rc == 0 and '"overall_ok": true' in out and '"done": 18' in out

    # failed task
    f3 = base / "t6_hgt" / "hgt_seed20261119" / "STATUS.json"
    f3.write_text(json.dumps({"state": "failed", "batch_id": batch2, "worker_id": "t6_hgt",
                              "model": "hgt", "seed": "20261119"}), encoding="utf-8")
    rc, out = run(summarize, env=env)
    reps["summary_detects_failure"] = rc != 0 and '"overall_ok": false' in out
    f3.write_text(json.dumps({"state": "done", "batch_id": batch2, "worker_id": "t6_hgt",
                              "model": "hgt", "seed": "20261119", "exit_code": 0}), encoding="utf-8")

    # identity mismatch is not "done"
    st = base / "t3_gat" / "gat_seed20261118" / "STATUS.json"
    good = st.read_text(encoding="utf-8")
    st.write_text(json.dumps({"state": "done", "batch_id": batch2, "worker_id": "wrong",
                              "model": "gat", "seed": "20261118"}), encoding="utf-8")
    rc, out = run(summarize, env=env)
    reps["summary_detects_identity_mismatch"] = rc != 0 and '"identity_ok": false' in out
    st.write_text(good, encoding="utf-8")

    # extra run dir reported, not counted
    (base / "t1_gcn" / "resume_extra").mkdir()
    rc, out = run(summarize, env=env)
    reps["summary_flags_extra_run_dir"] = '"unexpected_run_dirs_present": true' in out and rc == 0
    shutil.rmtree(base / "t1_gcn" / "resume_extra")

    # missing task dir -> incomplete
    missing = base / "t4_gin" / "gin_seed20261119"
    shutil.move(str(missing), str(tmp / "gin_hold"))
    rc, out = run(summarize, env=env)
    reps["summary_detects_missing"] = rc != 0 and "MISSING" in out
    shutil.move(str(tmp / "gin_hold"), str(missing))

    # collect success: manifest tasks only, manifest + tasks included
    tar = tmp / "out2.tar.gz"
    collect = [bash_exe, posix(SERVER_OPS / "collect_artifacts.sh"), batch2]
    env_c = dict(env, OUT_TAR=posix(tar))
    rc_c, out_c = run(collect, env=env_c)
    reps["collect_success"] = rc_c == 0 and tar.is_file()
    reps["collect_debug"] = {"rc": rc_c, "out": out_c[-500:]}
    if tar.is_file():
        rc_t, listing = run(["tar", "-tzf", str(tar)])
        reps["collect_scope_no_raw"] = "raw_data" not in listing
        reps["collect_has_manifest"] = "manifests/batch_manifest.json" in listing
        reps["collect_has_tasks"] = "manifests/TASKS.tsv" in listing
        reps["collect_18_runs"] = listing.count("STATUS.json") == 18
        rc_s, sums = run(["tar", "-xzf", str(tar), "-O", "./SHA256SUMS"])
        reps["sha256sums_excludes_self"] = "SHA256SUMS" not in sums

    # collect missing required file -> non-zero, archive still built for analysis
    miss = base / "t2_graphsage" / "graphsage_seed20261117" / "checkpoints" / "best.pt"
    miss.rename(miss.with_suffix(".bak"))
    tar_miss = tmp / "out2_missing.tar.gz"
    rc_m, out_m = run(collect, env=dict(env, OUT_TAR=posix(tar_miss)))
    reps["collect_aborts_missing"] = rc_m != 0 and "MISSING required" in out_m and tar_miss.is_file()
    miss.with_suffix(".bak").rename(miss)

    # collect extra run dir listing
    (base / "t5_rgcn" / "resume_extra").mkdir()
    (base / "t5_rgcn" / "resume_extra" / "STATUS.json").write_text(
        json.dumps({"state": "done"}), encoding="utf-8")
    tar_x = tmp / "out2_extra.tar.gz"
    rc_x, _ = run(collect, env=dict(env, OUT_TAR=posix(tar_x)))
    rc_l, listing_x = run(["tar", "-tzf", str(tar_x)])
    rc_e, extra_txt = run(["tar", "-xzf", str(tar_x), "-O", "./batch/EXTRA_RUN_DIRS.txt"])
    reps["collect_extra_listed_not_staged"] = ("t5_rgcn/resume_extra" in extra_txt
                                               and "/t5_rgcn/resume_extra/" not in listing_x)
    shutil.rmtree(base / "t5_rgcn" / "resume_extra")

    # default archive dir + overwrite refusal
    rc_def, out_def = run(collect, env=dict(env))
    default_tar = fake / "remote_artifacts" / f"batch_{batch2}_artifacts.tar.gz"
    reps["collect_default_path"] = rc_def == 0 and default_tar.is_file()
    rc_dup2, out_dup2 = run(collect, env=dict(env))
    reps["collect_refuses_overwrite"] = rc_dup2 != 0 and "REFUSE" in out_dup2

    # preflight gate: matching PASS ok, stale PASS rejected
    gate_env = dict(env)
    rc_h, hashes = run([py, "-B", str(GATE), "--hashes", "--batch-id", batch2,
                        "--project-root", posix(fake), "--data-root", posix(fake)], env=gate_env)
    reps["gate_hashes_ok"] = rc_h == 0
    pf_dir = fake / "outputs" / "three_topology_v2" / "preflight" / batch2
    pf_dir.mkdir(parents=True, exist_ok=True)
    rec = json.loads(hashes); rec["result"] = "PASS"
    (pf_dir / "PREFLIGHT_PASS.json").write_text(json.dumps(rec), encoding="utf-8")
    rc_g, _ = run([py, "-B", str(GATE), "--check", "--batch-id", batch2,
                   "--project-root", posix(fake), "--data-root", posix(fake)], env=gate_env)
    reps["gate_accepts_matching"] = rc_g == 0
    rec["stats_sha256"] = "0" * 64
    (pf_dir / "PREFLIGHT_PASS.json").write_text(json.dumps(rec), encoding="utf-8")
    rc_g2, out_g2 = run([py, "-B", str(GATE), "--check", "--batch-id", batch2,
                         "--project-root", posix(fake), "--data-root", posix(fake)], env=gate_env)
    reps["gate_rejects_stale"] = rc_g2 == 4 and "changed since preflight" in out_g2

    # first-batch compatibility: 8-task shape still works
    fake_fb = tmp / "proj_fb"
    batch_fb = "SIMFB"
    jobs_fb = {
        "mlp": {"tasks": [{"model": "mlp", "seeds": [20261117, 20261118, 20261119], "worker": "t1_mlp"},
                          {"model": "ridge", "device": "cpu", "worker": "t1_ridge"},
                          {"model": "hist_gb", "device": "cpu", "worker": "t1_histgb"}]},
        "hgt17": {"tasks": [{"model": "hgt_global", "seeds": [20261117], "worker": "t2_hgt"}]},
        "hgt18": {"tasks": [{"model": "hgt_global", "seeds": [20261118], "worker": "t3_hgt"}]},
        "hgt19": {"tasks": [{"model": "hgt_global", "seeds": [20261119], "worker": "t4_hgt"}]},
    }
    base_fb = build_project(fake_fb, batch_fb, jobs_fb, 8)
    fb_tasks = [(r[0], r[1], r[5]) for r in manifest_tasks(fake_fb, "mlp.yaml")[2]]
    make_runs(base_fb, batch_fb, fb_tasks, done=True)
    env_fb = dict(os.environ, PROJECT_ROOT=posix(fake_fb), BATCH_ID=batch_fb, PYTHON=py)
    rc_fb, out_fb = run([py, "-B", str(SERVER_OPS / "summarize_batch.py"), "--batch-dir", str(base_fb),
                         "--project-root", posix(fake_fb), "--json"], env=env_fb)
    reps["first_batch_8_ok"] = rc_fb == 0 and '"done": 8' in out_fb and len(fb_tasks) == 8
    tar_fb = tmp / "out_fb.tar.gz"
    rc_cfb, out_cfb = run([bash_exe, posix(SERVER_OPS / "collect_artifacts.sh"), batch_fb],
                          env=dict(env_fb, OUT_TAR=posix(tar_fb)))
    reps["first_batch_collect_ok"] = rc_cfb == 0 and tar_fb.is_file()
    reps["first_batch_debug"] = {"rc": rc_cfb, "out": out_cfb[-300:]}

    # atomic dir refusal + worker isolation (unchanged behaviour)
    d1 = base / "t1_gcn" / "gcn_seed20261117"
    try:
        d1.mkdir(exist_ok=False)
        reps["duplicate_dir_refused"] = False
    except FileExistsError:
        reps["duplicate_dir_refused"] = True
    reps["worker_isolation"] = (base / "t2_graphsage" / "graphsage_seed20261117").is_dir() and \
        (base / "t6_hgt" / "hgt_seed20261119").is_dir()

    reps["ok"] = all(v for v in reps.values() if isinstance(v, bool))
    out = {"checks": reps, "tmp": str(tmp)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dest = MODEL_ROOT / "outputs" / "three_topology_v2" / "simulate_isolation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sim_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if reps["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

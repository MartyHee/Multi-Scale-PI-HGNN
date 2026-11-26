#!/usr/bin/env python
"""Finalize the first-batch audit from the already-recomputed CSVs.

The review script recomputes slices from exported predictions, then aborts in
the plotting step on this Windows OpenMP environment.  This helper rebuilds the
audit record from metrics_by_run.csv / sample_metrics.csv and the run metadata;
it does not read predictions or labels again, does no inference, and never
touches test/blocked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol  # noqa: E402

BATCH = "fv2_first_20261125_01"
SLICES = {"design_val": ("A", "val"), "load_val": ("B", "val"), "joint_val": ("C", "val")}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ext, ana = Path(a.extracted), Path(a.analysis)
    out = Path(a.out) if a.out else (ana / "audit.json")

    am = jload(ext / "manifests" / "batch_manifest.json")
    lm = jload(MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2" / "manifests" / f"{BATCH}.json")
    batch = jload(ext / "summary" / "batch_summary.json")
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = jload(paths.PROCESSED_ROOT / "splits" / "protocol_v1.json")
    expected = {t: {n: protocol.ids_for(idx, prot, t, p, r) for n, (p, r) in SLICES.items()}
                for t in ("topology1", "topology2", "topology3")}
    with (ana / "metrics_by_run.csv").open("r", newline="", encoding="utf-8") as fh:
        mrows = list(csv.DictReader(fh))
    with (ana / "sample_metrics.csv").open("r", newline="", encoding="utf-8") as fh:
        srows = list(csv.DictReader(fh))
    by_key = {(r["topology"], r["model"], str(r["seed"])): r for r in mrows}
    # Group sample IDs preserving CSV order (which is prediction order).
    id_groups = {}
    for r in srows:
        id_groups.setdefault((r["topology"], r["model"], str(r["seed"]), r["slice"]), []).append(r["sample_id"])

    problems = []
    frozen = {
        "artifact_manifest_sha256": sha(ext / "manifests" / "batch_manifest.json"),
        "local_manifest_sha256": sha(MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2" / "manifests" / f"{BATCH}.json"),
        "manifest_same": sha(ext / "manifests" / "batch_manifest.json") ==
        sha(MODEL_ROOT / "remote_jobs" / "three_topology_fe_v2" / "manifests" / f"{BATCH}.json"),
        "code_sha256": am["code_sha256"], "local_code_sha256": lm["code_sha256"],
        "code_same": am["code_sha256"] == lm["code_sha256"],
        "train_config_same": am["train_config_sha256"] == lm["train_config_sha256"],
        "input_identity_same": am["input_identity"] == lm["input_identity"],
        "tasks_same": am["tasks"] == lm["tasks"],
        "server_git_commit": am.get("git_commit"), "local_manifest_git_commit": lm.get("git_commit"),
        "n_tasks": am["n_tasks"],
    }
    for k in ("manifest_same", "code_same", "train_config_same", "input_identity_same", "tasks_same"):
        if not frozen[k]:
            problems.append(f"frozen identity mismatch: {k}")
    art_files = {
        "configs/three_topology_fe_v2/feature_schema.json": MODEL_ROOT / "configs/three_topology_fe_v2/feature_schema.json",
        "configs/three_topology_fe_v2/formal_pipeline.yaml": MODEL_ROOT / "configs/three_topology_fe_v2/formal_pipeline.yaml",
        "processed/splits/protocol_v1.json": paths.PROCESSED_ROOT / "splits/protocol_v1.json",
        "processed/splits/role_ledger_v1.csv": paths.PROCESSED_ROOT / "splits/role_ledger_v1.csv",
        "processed/splits/split_manifest.json": paths.PROCESSED_ROOT / "splits/split_manifest.json",
    }
    for t in ("topology1", "topology2", "topology3"):
        art_files[f"processed/stats/{t}.json"] = paths.PROCESSED_ROOT / "stats" / f"{t}.json"
    # Git-tracked text may be LF on the server and CRLF in the local checkout;
    # use the same LF-normalised rule as the frozen manifest for config files.
    text_prefix = ("configs/",)
    def file_hash(p: Path, key: str) -> str:
        data = p.read_bytes()
        if key.startswith(text_prefix):
            data = data.replace(bytes([13, 10]), bytes([10]))
        return hashlib.sha256(data).hexdigest()
    frozen["artifact_files_match_local"] = {
        k: file_hash(ext / k, k) == file_hash(v, k) for k, v in art_files.items()}
    if not all(frozen["artifact_files_match_local"].values()):
        problems.append("artifact config/stats/split hash mismatch vs local")

    runs = []
    if batch.get("overall_ok") is not True or len(batch.get("rows", [])) != 18:
        problems.append("batch summary is not 18/18 overall_ok")
    for row in batch["rows"]:
        key = (row["topology"], row["model"], str(row["seed"]))
        d = ext / "batch" / row["topology"] / row["worker"] / row["run"]
        sm, st, task, log = jload(d / "summary.json"), jload(d / "STATUS.json"), jload(d / "TASK.json"), jload(d / "run_log.json")
        with (d / "train_log.csv").open("r", newline="", encoding="utf-8") as fh:
            tl = list(csv.DictReader(fh))
        mr = by_key.get(key)
        checks = {}
        checks["status_done"] = st.get("state") == "done" and st.get("exit_code") == 0
        checks["task_identity"] = (task.get("batch_id") == BATCH and task.get("topology") == key[0]
                                   and task.get("worker") == row["worker"] and task.get("model") == key[1]
                                   and str(task.get("seed")) == key[2])
        checks["task_manifest_sha"] = task.get("manifest_sha256") == frozen["artifact_manifest_sha256"]
        checks["task_config_sha"] = task.get("train_config_sha256") == am["train_config_sha256"]
        checks["train_graphs"] = sm.get("train_graphs") == 1360
        checks["val_graphs"] = sm.get("val_graphs") == {"design_val": 280, "load_val": 340, "joint_val": 70}
        checks["no_test"] = sm.get("test_constructed") is False and sm.get("protocol") == "protocol_v1"
        checks["physics_zero"] = sm.get("physics_loss_weight") == 0.0
        bsha = sha(d / "checkpoints" / "best.pt")
        checks["best_sha"] = bsha == sm["best"]["checkpoint_sha256"] == sm["final_eval_checkpoint_sha256"]
        checks["final_from_best"] = sm.get("final_eval_source") == "best"
        checks["selection_is_best"] = abs(sm["selection_metric"] - sm["best"]["selection_metric"]) < 1e-8
        checks["log_pass"] = log.get("pass", {}).get("all") is True
        checks["identity_stats"] = log["identity"]["stats_sha256"] == am["input_identity"]["stats_sha256"][key[0]]
        checks["identity_data"] = (log["identity"]["samples_jsonl_sha256"] == am["input_identity"]["samples_jsonl_sha256"]
                                   and log["identity"]["dataset_manifest_sha256"] == am["input_identity"]["dataset_manifest_sha256"])
        checks["identity_read_path"] = log["identity"].get("read_path") == "shared_v1"
        checks["identity_cache"] = log["identity"].get("design_cache_sha256") == am["input_identity"]["design_cache_sha256"][key[0]]
        checks["train_log_rows"] = len(tl) == sm["completed_epochs"]
        checks["epoch_budget"] = sm["completed_epochs"] <= am["training_effective"]["epochs"]
        if sm.get("stop_reason") == "early_stop":
            checks["early_stop_condition"] = int(tl[-1]["no_improve"]) >= int(am["training_effective"]["early_stopping_patience"])
        else:
            checks["early_stop_condition"] = sm.get("stop_reason") == "epoch_budget_reached"
        checks["best_epoch_in_log"] = any(int(x["epoch"]) == sm["best"]["epoch_zero_based"] for x in tl)
        for name in SLICES:
            ids = id_groups.get((key[0], key[1], key[2], name), [])
            checks[f"ids_{name}"] = ids == expected[key[0]][name]
            if any("test" in prot["roles"][sid].values() for sid in ids):
                checks[f"ids_{name}"] = False
                problems.append(f"{key}/{name}: test-role sample")
            checks[f"selection_{name}"] = abs(float(mr[name]) - float(mr[f"recomputed_selection_{name}"])) < 1e-8
        checks["total_selection"] = abs(float(mr["selection"]) - float(mr["recomputed_total"])) < 1e-8
        if not all(checks.values()):
            problems.append({"run": list(key), "failed_checks": [k for k, v in checks.items() if not v]})
        runs.append({"run": list(key), "checks": checks, "best_sha256": bsha,
                     "selection": sm["selection_metric"],
                     "recomputed_total": float(mr["recomputed_total"]),
                     "max_slice_abs_diff": max(abs(float(mr[n]) - float(mr[f"recomputed_selection_{n}"])) for n in SLICES)})

    audit = {
        "schema": "qt-fe-v2-first-batch-audit-v1", "batch_id": BATCH,
        "source": "metrics_by_run.csv recomputed from exported predictions; sample_metrics.csv for IDs",
        "n_runs": len(runs), "frozen": frozen, "runs": runs,
        "max_slice_abs_diff": max(r["max_slice_abs_diff"] for r in runs),
        "max_total_abs_diff": max(abs(r["selection"] - r["recomputed_total"]) for r in runs),
        "problems": problems, "all_checks_ok": not problems,
        "test_role_predictions": sum(1 for r in srows if "test" in prot["roles"][r["sample_id"]].values()),
        "n_prediction_records": len(srows),
    }
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_checks_ok": audit["all_checks_ok"], "problems": problems,
                      "n_runs": len(runs), "max_slice_abs_diff": audit["max_slice_abs_diff"],
                      "max_total_abs_diff": audit["max_total_abs_diff"], "out": str(out)},
                     ensure_ascii=False, indent=2))
    return 0 if audit["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

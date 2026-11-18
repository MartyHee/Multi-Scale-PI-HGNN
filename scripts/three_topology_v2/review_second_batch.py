#!/usr/bin/env python
"""Audit the second-batch server archive (18 runs) and recompute validation metrics.

Separates three evidence levels:
  A. archive self-consistency (SHA256SUMS inside the extracted archive);
  B. consistency with the frozen inputs/manifest (data/split/stats/config/code);
  C. transport/outer SHA256: only the locally computed value is reported unless a
     server-side outer_sha256 is supplied; no assumption of confirmation.

No inference, no training, no outer-test labels (val only; test identity only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402

from scripts.three_topology_v2.analysis_common import (BATCHES, GRADIENT, MODEL_LABEL,  # noqa: E402
                                                       SPLIT_ID, FOLD_ID, load_context,
                                                       model_params, read_json, sha)

BATCH = "sb_20261118_01"
EXPECTED_COMMIT = "836d86445c1ba6462e13986c113221a5975ef698"
REQUIRED = ["STATUS.json", "TASK.json", "run_manifest.json", "run_log.json", "summary.json",
            "train_log.csv", "driver.log", "val_predictions.json",
            "checkpoints/best.pt", "checkpoints/last.pt"]


def sha_text(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def verify_sums(root: Path):
    listed, bad = {}, []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        rel = rel.lstrip("./")
        listed[rel] = h
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    for rel, h in listed.items():
        p = root / rel
        if not p.is_file():
            bad.append({"file": rel, "issue": "missing"})
        elif sha(p) != h:
            bad.append({"file": rel, "issue": "hash_mismatch"})
    return {"listed": len(listed), "actual": len(actual), "bad": bad,
            "extra": sorted(actual - set(listed)), "ok": not bad and actual == set(listed)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(MODEL_ROOT / "remote_artifacts" / f"batch_{BATCH}_review"))
    ap.add_argument("--out", default=str(MODEL_ROOT / "outputs" / "three_topology_v2" / "second_batch_review_20261118"))
    ap.add_argument("--server-outer-sha256", default=None)
    a = ap.parse_args()
    arch = Path(a.archive_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    local_index_hashes = ctx["index"]["hashes"]
    local_manifest = MODEL_ROOT / "remote_jobs" / "three_topology_v2" / "manifests" / f"{BATCH}.json"

    report = {"schema": "qt-second-batch-review-v1", "date": "2026-11-18",
              "batch_id": BATCH, "archive_dir": str(arch),
              "archive_sha256_local": sha(BATCHES[BATCH]["archive"]),
              "server_outer_sha256_supplied": a.server_outer_sha256,
              "transport": ("server outer_sha256 supplied; compare below"
                            if a.server_outer_sha256 else
                            "server outer_sha256 NOT supplied; local hash only, pending comparison"),
              "archive_self_check": verify_sums(arch),
              "frozen_input": {}, "runs": {}, "problems": []}

    man = read_json(arch / "manifests" / "batch_manifest.json")
    config_arch = arch / "configs" / "three_topology_v2" / "formal_baselines.yaml"
    split_arch = arch / "manifests" / "splits" / f"split_{SPLIT_ID}.json"
    stats_arch = arch / "manifests" / "stats" / SPLIT_ID / f"{FOLD_ID}.json"
    local_config = MODEL_ROOT / "configs" / "three_topology_v2" / "formal_baselines.yaml"
    local_split = MODEL_ROOT / "processed" / "three_topology_training_v2" / "splits" / f"split_{SPLIT_ID}.json"
    local_stats = MODEL_ROOT / "processed" / "three_topology_training_v2" / "stats" / SPLIT_ID / f"{FOLD_ID}.json"
    code_version = (arch / "code" / "CODE_VERSION.txt").read_text(encoding="utf-8").splitlines()
    env_txt = (arch / "code" / "ENV.txt").read_text(encoding="utf-8")
    report["frozen_input"] = {
        "manifest_schema": man.get("schema"), "manifest_tasks": man.get("n_tasks"),
        "manifest_sha256": sha_text(arch / "manifests" / "batch_manifest.json"),
        "local_manifest_sha256": sha_text(local_manifest),
        "manifest_match": sha_text(arch / "manifests" / "batch_manifest.json") == sha_text(local_manifest),
        "protocol": man.get("protocol"), "split_id": man.get("split_id"), "fold_id": man.get("fold_id"),
        "train_config_sha256": man.get("train_config_sha256"),
        "archive_config_sha256": sha_text(config_arch),
        "local_config_sha256": sha_text(local_config),
        "config_match": sha_text(config_arch) == sha_text(local_config) == man.get("train_config_sha256"),
        "archive_split_sha256": sha(split_arch), "local_split_sha256": sha(local_split),
        "manifest_split_sha256": man["input_identity"]["split_sha256"],
        "split_match": sha(split_arch) == sha(local_split) == man["input_identity"]["split_sha256"],
        "archive_stats_sha256": sha(stats_arch), "local_stats_sha256": sha(local_stats),
        "manifest_stats_sha256": man["input_identity"]["stats_sha256"],
        "stats_match": sha(stats_arch) == sha(local_stats) == man["input_identity"]["stats_sha256"],
        "samples_jsonl_sha256": local_index_hashes.get("samples_jsonl"),
        "dataset_manifest_sha256": local_index_hashes.get("dataset_manifest"),
        "train_sample_list_sha256": man["input_identity"]["train_sample_list_sha256"],
        "val_sample_list_sha256": man["input_identity"]["val_sample_list_sha256"],
        "code_version_head": code_version[0] if code_version else None,
        "code_version_expected": EXPECTED_COMMIT,
        "code_version_match": bool(code_version) and code_version[0] == EXPECTED_COMMIT,
        "code_dirty_lines": code_version[1:],
        "env": env_txt.strip().splitlines(),
        "reported_validation": {"train": man["input_identity"]["n_train"],
                                "val": man["input_identity"]["n_val"],
                                "test_sealed": len(ctx["test_ids"])},
    }
    if not report["frozen_input"]["config_match"]:
        report["problems"].append("config hash mismatch")
    if not report["frozen_input"]["split_match"]:
        report["problems"].append("split hash mismatch")
    if not report["frozen_input"]["stats_match"]:
        report["problems"].append("stats hash mismatch")
    if not report["frozen_input"]["manifest_match"]:
        report["problems"].append("manifest mismatch (fresh regeneration?)")
    if not report["frozen_input"]["code_version_match"]:
        report["problems"].append("code version != expected commit")

    params = model_params(BATCH)
    run_dirs = sorted((arch / "batch" / BATCH).glob("*/*"))
    tasks_tsv = [l.split("\t") for l in (arch / "TASKS.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    report["frozen_input"].update({"run_dirs": len(run_dirs), "tasks_tsv": len(tasks_tsv),
                                   "manifest_tasks": len(man["tasks"])})
    if len(run_dirs) != 18 or len(tasks_tsv) != 18 or len(man["tasks"]) != 18:
        report["problems"].append("run/task count != 18")

    val = set(ctx["val_ids"]); test = set(ctx["test_ids"])
    comp_rows, graph_rows, worst_rows, group_rows = [], [], [], []
    for d in run_dirs:
        worker, run = d.parent.name, d.name
        model = run.split("_seed")[0]
        seed = run.split("_seed")[1] if "_seed" in run else "-"
        rec = {"worker": worker, "run": run, "model": model, "seed": seed,
               "checks": {}, "problems": []}
        for f in REQUIRED:
            rec["checks"][f"file:{f}"] = (d / f).is_file()
        try:
            status = read_json(d / "STATUS.json"); task = read_json(d / "TASK.json")
            manifest = read_json(d / "run_manifest.json"); log = read_json(d / "run_log.json")
            preds = read_json(d / "val_predictions.json")
        except Exception as exc:  # noqa: BLE001
            rec["problems"].append(f"cannot read run files: {exc!r}")
            report["runs"][f"{worker}/{run}"] = rec
            continue
        rec["checks"]["status_done"] = status.get("state") == "done" and status.get("exit_code") == 0
        rec["checks"]["identity"] = (status.get("batch_id") == BATCH and status.get("worker_id") == worker
                                     and status.get("model") == model and str(status.get("seed")) == str(seed)
                                     and task.get("batch_id") == BATCH and task.get("worker_id") == worker
                                     and task.get("model") == model and str(task.get("seed")) == str(seed))
        rec["checks"]["device_cuda0"] = status.get("device") == "cuda:0"
        rec["checks"]["protocol"] = (manifest.get("split_id") == SPLIT_ID and manifest.get("fold_id") == FOLD_ID
                                     and task.get("protocol") == "split")
        rec["checks"]["train_ids"] = set(manifest.get("train_sample_ids", [])) == set(ctx["fold"]["train_sample_ids"])
        rec["checks"]["val_ids"] = set(manifest.get("val_sample_ids", [])) == val
        rec["checks"]["counts_84_24"] = (len(manifest.get("train_sample_ids", [])) == 84
                                         and len(manifest.get("val_sample_ids", [])) == 24)
        rec["checks"]["test_disjoint"] = not (val & test) and not (test & set(manifest.get("train_sample_ids", [])))
        rec["checks"]["data_hashes"] = (manifest.get("data_hashes", {}).get("samples_jsonl") == local_index_hashes.get("samples_jsonl")
                                        and manifest.get("data_hashes", {}).get("dataset_manifest") == local_index_hashes.get("dataset_manifest"))
        rec["checks"]["stats_hash"] = manifest.get("stats_sha256") == man["input_identity"]["stats_sha256"]
        ident = log.get("identity", {})
        rec["checks"]["no_smoke"] = ident.get("max_train_batches_per_epoch") is None
        rec["checks"]["physics_zero"] = log.get("physics_loss_weight") == 0.0
        rec["checks"]["not_resumed"] = log.get("resumed") is None
        rec["checks"]["epochs_requested_60"] = log.get("epochs_requested") == 60
        rec["checks"]["completed_epochs"] = int(log.get("completed_epochs", -1)) > 0
        rec["checks"]["epoch_log_rows"] = len(log.get("epochs_log", [])) == log.get("completed_epochs")
        rec["checks"]["finite"] = all(math.isfinite(float(e.get("train_loss_mean", float("nan"))))
                                      and math.isfinite(float(e.get("selection_metric", float("nan"))))
                                      for e in log.get("epochs_log", []))
        rec["checks"]["pass_all"] = log.get("pass", {}).get("all") is True
        rec["checks"]["best_epoch_valid"] = 0 <= int(log.get("best", {}).get("epoch_zero_based", -1)) < log.get("completed_epochs", 0)
        rec["checks"]["best_matches_final"] = abs(float(log.get("best", {}).get("selection_metric", 1e9))
                                                  - float(log.get("final_val", {}).get("selection_metric", -1e9))) < 1e-6
        rec["checks"]["best_sha"] = sha(d / "checkpoints/best.pt") == log.get("best", {}).get("checkpoint_sha256")
        rec["checks"]["last_sha"] = sha(d / "checkpoints/last.pt") == log.get("last", {}).get("checkpoint_sha256")
        best_ep = int(log.get("best", {}).get("epoch_zero_based", -1))
        same_end = best_ep == int(log.get("completed_epochs", 0)) - 1
        rec["checks"]["best_differs_from_last_if_not_final_epoch"] = (
            True if same_end else sha(d / "checkpoints/best.pt") != sha(d / "checkpoints/last.pt"))
        rows = (d / "train_log.csv").read_text(encoding="utf-8").splitlines()
        rec["checks"]["train_log_rows"] = len(rows) - 1 == log.get("completed_epochs")
        rec["checks"]["train_log_epochs_contiguous"] = [int(x.split(",")[0]) for x in rows[1:] if x.strip()] == list(range(log.get("completed_epochs", 0)))
        rec["checks"]["pred_covers_val"] = {r["sample_id"] for r in preds} == val
        rec["checks"]["pred_no_test"] = not ({r["sample_id"] for r in preds} & test)
        shape_ok = True
        for r in preds:
            sid = r["sample_id"]
            td, tf = ctx["labels"][sid]
            if (len(r.get("pred_disp", [])) != td.shape[0] or len(r.get("pred_force", [])) != tf.shape[0]
                    or not np.isfinite(np.asarray(r["pred_disp"])).all()
                    or not np.isfinite(np.asarray(r["pred_force"])).all()):
                shape_ok = False
        rec["checks"]["pred_shapes_finite"] = shape_ok
        # recompute
        try:
            from scripts.three_topology_v2.analysis_common import recompute
            metrics = recompute(preds, ctx)
            rec["metrics"] = {k: metrics[k] for k in ("selection", "selection_disp", "selection_force")}
            rec["checks"]["recomputed_matches_log"] = abs(metrics["selection"] - float(log["final_val"]["selection_metric"])) < 1e-6
            for c in metrics["components"]:
                comp_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed, **c})
            for g in metrics["per_graph"]:
                graph_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed, **g})
            worst_rows.extend([{"run": f"{worker}/{run}", "model": model, "seed": seed, **g}
                               for g in sorted(metrics["per_graph"], key=lambda x: -x["selection"])[:5]])
            groups = defaultdict(list)
            for g in metrics["per_graph"]:
                for key in ("design_id", "template_id"):
                    groups[(key, g[key])].append(g["selection"])
            for (key, valg), sels in sorted(groups.items()):
                group_rows.append({"run": f"{worker}/{run}", "model": model, "seed": seed,
                                   "group_by": key, "group": valg, "n_graphs": len(sels),
                                   "selection_graph_equal_mean": float(np.mean(sels))})
        except Exception as exc:  # noqa: BLE001
            rec["checks"]["recomputed_matches_log"] = False
            rec["problems"].append(f"recompute failed: {exc!r}")
        rec.update({"completed_epochs": log.get("completed_epochs"),
                    "early_stopped": log.get("early_stopped"),
                    "best_epoch_zero_based": best_ep,
                    "best_selection": log.get("best", {}).get("selection_metric"),
                    "final_selection": log.get("final_val", {}).get("selection_metric"),
                    "n_params": log.get("n_params", params.get(model)),
                    "device": status.get("device"),
                    "external_cvd": status.get("external_cuda_visible_devices"),
                    "started_at": status.get("started_at") or task.get("started_at"),
                    "finished_at": status.get("finished_at"),
                    "timing": log.get("timing", {})})
        rec["checks"]["all_ok"] = all(bool(v) for v in rec["checks"].values())
        if not rec["checks"]["all_ok"]:
            rec["problems"].extend([k for k, v in rec["checks"].items() if not v])
            report["problems"].append(f"{worker}/{run}: {rec['problems']}")
        report["runs"][f"{worker}/{run}"] = rec

    # CSVs
    def write_csv(name, rows):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    write_csv("per_component_metrics.csv", comp_rows)
    write_csv("per_graph_metrics.csv", graph_rows)
    write_csv("worst_val_samples.csv", worst_rows)
    write_csv("design_field_group.csv", group_rows)
    run_rows = []
    for key, r in report["runs"].items():
        run_rows.append({"run": key, "model": r["model"], "seed": r["seed"],
                         "selection": r.get("metrics", {}).get("selection"),
                         "selection_disp": r.get("metrics", {}).get("selection_disp"),
                         "selection_force": r.get("metrics", {}).get("selection_force"),
                         "best_epoch_zero_based": r.get("best_epoch_zero_based"),
                         "completed_epochs": r.get("completed_epochs"),
                         "early_stopped": r.get("early_stopped"),
                         "n_params": r.get("n_params"), "device": r.get("device"),
                         "external_cvd": r.get("external_cvd"),
                         "all_ok": r["checks"].get("all_ok")})
    write_csv("per_run_metrics.csv", run_rows)
    report["all_ok"] = report["archive_self_check"]["ok"] and not report["problems"]
    (out / "review_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "problems": report["problems"],
                      "self_check": report["archive_self_check"]["ok"],
                      "runs": len(report["runs"])}, ensure_ascii=False, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

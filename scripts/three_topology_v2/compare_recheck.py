#!/usr/bin/env python
"""Compare an eval-only recheck against the original exported predictions.

For ONE historical run this checks, in order:
  1. the checkpoint file actually evaluated (raw sha256);
  2. the checkpoint identity vs the frozen v2 manifest (model/seed/split/fold,
     stats/train-val list hashes, model-config hash, samples/dataset hashes);
  3. per-sample entity counts (node/beam) so array order is not silently assumed;
  4. per-component max absolute/relative prediction differences;
  5. selection-metric difference (original run_log final_val vs recheck).

Tolerance rationale: same checkpoint + same 24 validation graphs + same
environment; only floating-point summation order can differ at the last bits.
Default atol=1e-8 + rtol=1e-5 of the component scale. A model-state change or
wrong graph order is far larger. --strict requires exact prediction equality.

The checkpoint is never edited (no device-string rewriting). Read-only originals;
the caller writes to an independent directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DISP = ["dx", "dy", "dz", "rx", "ry", "rz"]
FORCE = ["I_Fx", "I_Fy", "I_Fz", "I_Mx", "I_My", "I_Mz",
         "J_Fx", "J_Fy", "J_Fz", "J_Mx", "J_My", "J_Mz"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity_check(checkpoint: Path, manifest_path: Path, worker: str, run: str,
                   data_root: Path) -> dict:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    if man.get("schema") != "qt-three-topology-batch-manifest-v2":
        return {"checked": False, "diffs": {"manifest_schema": man.get("schema")}}
    task = next((t for t in man["tasks"] if t["worker"] == worker and t["run"] == run), None)
    if task is None:
        return {"checked": False, "diffs": {"task": f"{worker}/{run} not in manifest"}}
    try:
        import torch
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "diffs": {"checkpoint_load": repr(exc)}}
    ident = ck.get("identity") or {}
    expected = {
        "model_name": task["model"], "seed": int(task["seed"]),
        "split_id": man["split_id"], "fold_id": man["fold_id"],
        "outer_fold": man["outer_fold"] or None,
        "inner_fold": man["inner_id"] or None,
        "stats_sha256": man["input_identity"]["stats_sha256"],
        "train_sample_list_sha256": man["input_identity"]["train_sample_list_sha256"],
        "val_sample_list_sha256": man["input_identity"]["val_sample_list_sha256"],
        "model_config_sha256": task["model_config_sha256"],
    }
    idx = data_root / "data" / "processed" / "three_topology_graph_v1"
    expected["samples_jsonl_sha256"] = sha(idx / "samples.jsonl")
    expected["dataset_manifest_sha256"] = sha(idx / "dataset_manifest.json")
    diffs = {k: {"checkpoint": ident.get(k), "expected": v}
             for k, v in expected.items() if ident.get(k) != v}
    return {"checked": True, "identity": ident, "expected": expected, "diffs": diffs,
            "ok": not diffs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="original run dir")
    ap.add_argument("--recheck", required=True, help="recheck run dir (contains eval/)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--worker", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    orig_dir, rc_dir = Path(a.original), Path(a.recheck)
    orig = {r["sample_id"]: r for r in json.loads(
        (orig_dir / "val_predictions.json").read_text(encoding="utf-8"))}
    rc = {r["sample_id"]: r for r in json.loads(
        (rc_dir / "eval" / "val_predictions.json").read_text(encoding="utf-8"))}
    run_log = json.loads((orig_dir / "run_log.json").read_text(encoding="utf-8"))
    eval_log = json.loads((rc_dir / "eval" / "eval_log.json").read_text(encoding="utf-8"))

    res = {"schema": "qt-historical-best-recheck-v1",
           "original_dir": str(orig_dir), "recheck_dir": str(rc_dir),
           "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha(Path(a.checkpoint)),
           "tolerance": {"atol": a.atol, "rtol": a.rtol, "strict": a.strict,
                         "rationale": "same checkpoint/data/environment; only last-bit "
                                      "floating-point summation order may differ"},
           "ids_match": set(orig) == set(rc), "n_samples": len(orig),
           "torch_note": "eval-only reload; checkpoint not edited",
           "components": [], "problems": []}
    if set(orig) != set(rc):
        res["problems"].append("sample id sets differ")
        res["ok"] = False
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    # entity order: sample ids identical and per-sample node/beam counts identical
    order_ok = True
    for sid in sorted(orig):
        for key in ("node_count", "beam_count"):
            if orig[sid].get(key) != rc[sid].get(key):
                order_ok = False
                res["problems"].append(f"{sid}: {key} differs {orig[sid].get(key)} vs {rc[sid].get(key)}")
    res["entity_order_ok"] = order_ok

    if a.manifest and a.data_root and a.worker and a.run:
        ident = identity_check(Path(a.checkpoint), Path(a.manifest), a.worker, a.run, Path(a.data_root))
        res["identity_check"] = ident
        if not ident.get("ok"):
            res["problems"].append("checkpoint identity mismatch")
    else:
        res["identity_check"] = {"checked": False,
                                 "note": "manifest/data-root/worker/run not supplied"}

    worst = 0.0
    for task, names in (("disp", DISP), ("force", FORCE)):
        od, rd = [], []
        for sid in sorted(orig):
            od.append(np.asarray(orig[sid][f"pred_{task}"], dtype=np.float64))
            rd.append(np.asarray(rc[sid][f"pred_{task}"], dtype=np.float64))
        od, rd = np.concatenate(od), np.concatenate(rd)
        if od.shape != rd.shape:
            res["problems"].append(f"{task}: shape {od.shape} != {rd.shape}")
            continue
        diff = np.abs(od - rd)
        scale = np.maximum(np.abs(od).max(axis=0), 1e-12)
        for c, name in enumerate(names[:od.shape[1]]):
            tol = 0.0 if a.strict else (a.atol + a.rtol * scale[c])
            rel = float(diff[:, c].max() / scale[c])
            ok = bool(diff[:, c].max() <= tol)
            worst = max(worst, rel)
            res["components"].append({"task": task, "component": name,
                                      "max_abs_diff": float(diff[:, c].max()),
                                      "max_rel_diff": rel, "tolerance": tol,
                                      "within_tolerance": ok})
            if not ok:
                res["problems"].append(f"{task}.{name}: max_abs_diff {diff[:,c].max():.3e} > tol {tol:.3e}")

    sel_o = run_log.get("final_val", {}).get("selection_metric")
    sel_r = eval_log.get("selection_metric")
    res["selection_original"] = sel_o
    res["selection_recheck"] = sel_r
    res["selection_abs_diff"] = None if (sel_o is None or sel_r is None) else abs(sel_o - sel_r)
    if sel_o is None or sel_r is None or abs(sel_o - sel_r) > (0.0 if a.strict else 1e-9):
        res["problems"].append(f"selection differs: {sel_o} vs {sel_r}")
    res["worst_rel_diff"] = worst
    res["ok"] = not res["problems"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": res["ok"], "selection_original": sel_o,
                      "selection_recheck": sel_r, "worst_rel_diff": worst,
                      "identity_ok": res["identity_check"].get("ok"),
                      "problems": res["problems"]}, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

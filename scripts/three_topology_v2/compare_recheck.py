#!/usr/bin/env python
"""Compare an eval-only recheck against the original exported predictions.

Reads the original ``val_predictions.json`` + ``run_log.json`` and the recheck
``eval/val_predictions.json`` + ``eval/eval_log.json`` for ONE run.  Reports:
  * checkpoint sha256 of the file actually evaluated;
  * per-component max absolute difference and max difference relative to the
    component's max |prediction|;
  * selection-metric difference (original run_log final_val vs recheck);
  * a pass/fail verdict under an explicit tolerance.

Tolerance rationale: the recheck reloads the SAME checkpoint and evaluates the
SAME 24 validation graphs on the SAME environment.  Floating-point summation
order in PyTorch/cuBLAS can still differ at the last bits, so the default
tolerance is atol=1e-8 + rtol=1e-5 of the component scale; a model-state change
or a wrong graph order is far larger than that.  Pass --strict for exact equality.
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
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="original run dir")
    ap.add_argument("--recheck", required=True, help="recheck run dir (contains eval/)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
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

    res = {"original_dir": str(orig_dir), "recheck_dir": str(rc_dir),
           "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha(Path(a.checkpoint)),
           "ids_match": set(orig) == set(rc), "n_samples": len(orig),
           "torch_note": "same-environment inference recheck; no checkpoint edit",
           "components": [], "problems": []}
    if set(orig) != set(rc):
        res["problems"].append("sample id sets differ")
        res["ok"] = False
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

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
                      "problems": res["problems"]}, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

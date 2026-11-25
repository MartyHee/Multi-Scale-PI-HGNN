#!/usr/bin/env python
"""Verify resume that actually adds new epochs (not just re-loading a finished run).

Continuous run: 6/8 epochs.  Interrupted run: same config/seed, stopped after a
complete checkpoint boundary via the evaluator; then resumed with the same total
epoch budget.  CPU is compared strictly; CUDA with a stated tolerance.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(MODEL_ROOT / "scripts" / "three_topology_fe_v2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator, SliceBundle  # noqa: E402
from src.data.three_topology_fe_v2.shared_cache import SharedCache  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402
from train_fe_v2 import _identity_extra  # noqa: E402


class Interrupted(RuntimeError):
    pass


class StopAfterEvaluator(MultiSliceEvaluator):
    def __init__(self, *a, stop_after: int, **kw):
        super().__init__(*a, **kw)
        self.stop_after = int(stop_after)
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        if self.calls > self.stop_after:
            raise Interrupted(f"simulated interruption after {self.stop_after} epochs")
        return super().__call__(*a, **kw)


def base_cfg(lr: float, epochs: int) -> dict:
    return {
        "data": {"dtype": "float32"},
        "training": {
            "seeds": [20261117], "batch_size": 2, "epochs": epochs, "lr": lr,
            "weight_decay": 0.0, "early_stopping_patience": 20, "min_delta": 0.0,
            "grad_clip": 5.0,
            "selection_metric": "three_slice_graph_equal_task_equal_component_equal_normalized_mse",
            "physics_loss_weight": 0.0, "max_train_batches_per_epoch": 4,
            "val_batch_size": 1, "max_val_batches": 2, "diagnostic": False,
            "device": "cpu",
        },
        "models": {"mlp": {"hidden_dims": [16], "dropout": 0.1,
                           "activation": "relu", "use_batch_norm": False}},
        "ml": {"ridge": {"alpha": 1.0}, "hist_gb": {"max_iter": 2}},
    }


def build_data(idx, prot, st, topology):
    train_ids = protocol.ids_for(idx, prot, topology, "A", "train")[:8]
    val = {k: protocol.ids_for(idx, prot, topology, p, "val")[:2]
           for k, p in (("design_val", "A"), ("load_val", "B"), ("joint_val", "C"))}
    shared = SharedCache(idx, topology)
    tr = FeV2Dataset(idx, train_ids, st, dtype=torch.float32, normalize=True,
                     use_shared=True, shared=shared, topology=topology)
    val_ds = {k: FeV2Dataset(idx, v, st, dtype=torch.float32, normalize=True,
                             use_shared=True, shared=shared, topology=topology)
              for k, v in val.items()}
    return train_ids, val, tr, val_ds, shared


def run_once(name, cfg, out_dir, tr, val_ds, st, idx, train_ids, val, seed,
             resume_from=None, stop_after=None):
    local = json.loads(json.dumps(cfg))
    local["training"]["seed"] = seed
    local["models"]["mlp"] = schema.model_cfg_with_dims("mlp", local["models"]["mlp"])
    engine.seed_everything(seed)
    model = registry.build_gradient_model("mlp", local["models"]["mlp"])
    ev = (StopAfterEvaluator(list(val_ds.items()), st, max_batches=2, stop_after=stop_after)
          if stop_after is not None else MultiSliceEvaluator(list(val_ds.items()), st, max_batches=2))
    identity = engine.build_identity(
        "mlp", seed, st, local, train_ids, [s for v in val.values() for s in v],
        {"samples_jsonl": idx.samples_sha256,
         "dataset_manifest": index_mod.sha256_file(idx.root / "dataset_manifest.json")},
        "protocol_v1", "topology1", extra=_identity_extra("topology1"))
    try:
        log = engine.train_fold("mlp", model, tr, SliceBundle(val_ds.values()), st, local,
                                out_dir, identity, resume_from=resume_from, val_evaluator=ev)
        ev.write_slice_artifacts(out_dir, run_label=f"best:{log['best']['checkpoint_sha256']}")
        return {"ok": True, "log": log, "eval": ev.last}
    except Interrupted as exc:
        return {"ok": False, "interrupted": str(exc), "eval": ev.last}


def _pred(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_pair(cont_dir: Path, res_dir: Path, dev: torch.device):
    out = {}
    ck_c = torch.load(cont_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    ck_r = torch.load(res_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "optimizer_state"):
        a, b = ck_c[key], ck_r[key]
        if key == "model_state":
            maxima = max(float((a[k].float() - b[k].float()).abs().max().item()) for k in a)
            out["model_state_max_abs_diff"] = maxima
            out["model_state_equal"] = maxima == 0.0 if dev.type == "cpu" else maxima < 1e-5
        else:
            # Compare optimizer tensor state keys shared by both.
            maxima = 0.0
            sa, sb = a.get("state", {}), b.get("state", {})
            for k in sa:
                for sk, sv in sa[k].items():
                    if torch.is_tensor(sv) and k in sb and sk in sb[k]:
                        maxima = max(maxima, float((sv.float() - sb[k][sk].float()).abs().max().item()))
            out["optimizer_state_max_abs_diff"] = maxima
            out["optimizer_state_equal"] = maxima == 0.0 if dev.type == "cpu" else maxima < 1e-5
    for slice_name in ("design_val", "load_val", "joint_val"):
        pc = _pred(cont_dir / f"val_predictions_{slice_name}.json")
        pr = _pred(res_dir / f"val_predictions_{slice_name}.json")
        if pc is None or pr is None:
            out[f"{slice_name}_pred_equal"] = False
            continue
        md = 0.0
        for a, b in zip(pc, pr):
            if a["sample_id"] != b["sample_id"]:
                out[f"{slice_name}_pred_equal"] = False
                break
            md = max(md, float(np.max(np.abs(np.asarray(a["pred_disp"]) - np.asarray(b["pred_disp"])))),
                     float(np.max(np.abs(np.asarray(a["pred_force"]) - np.asarray(b["pred_force"])))))
        else:
            out[f"{slice_name}_pred_max_abs_diff"] = md
            out[f"{slice_name}_pred_equal"] = md == 0.0 if dev.type == "cpu" else md < 1e-4
    ck_bc = torch.load(cont_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    ck_br = torch.load(res_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    bmax = max(float((ck_bc["model_state"][k].float() - ck_br["model_state"][k].float()).abs().max())
               for k in ck_bc["model_state"])
    out["best_model_state_max_abs_diff"] = bmax
    out["best_model_state_equal"] = bmax == 0.0 if dev.type == "cpu" else bmax < 1e-5
    out["best_epoch_equal"] = ck_bc.get("best_epoch") == ck_br.get("best_epoch")
    out["best_val_equal"] = float(ck_bc.get("best_val", 0)) == float(ck_br.get("best_val", 0))
    return out


def scenario(root: Path, name: str, lr: float, epochs: int, stop_after: int,
             device: str, expect_new_best: bool):
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / "topology1.json")
    train_ids, val, tr, val_ds, shared = build_data(idx, prot, st, "topology1")
    dev = engine.resolve_device(device)
    cfg = base_cfg(lr, epochs); cfg["training"]["device"] = device
    d = root / name
    # Continuous reference.
    cont = run_once("continuous", cfg, d / "continuous", tr, val_ds, st, idx,
                    train_ids, val, 20261117)
    # Interrupted at a checkpoint boundary.
    intd = run_once("interrupted", cfg, d / "interrupted", tr, val_ds, st, idx,
                    train_ids, val, 20261117, stop_after=stop_after)
    ck = torch.load(d / "interrupted" / "checkpoints" / "last.pt", map_location="cpu",
                    weights_only=False)
    start_epoch = int(ck.get("epoch", 0))
    # Resume with the SAME total budget; actual new epochs required.
    res = run_once("resumed", cfg, d / "resumed", tr, val_ds, st, idx,
                   train_ids, val, 20261117,
                   resume_from=d / "interrupted" / "checkpoints" / "last.pt")
    cmp = compare_pair(d / "continuous", d / "resumed", dev) if res["ok"] and cont["ok"] else {}
    resumed = res["log"].get("resumed") if res["ok"] else None
    best_source = res["log"].get("best", {}).get("source") if res["ok"] else None
    rec = {
        "scenario": name, "lr": lr, "epochs": epochs, "stop_after": stop_after,
        "device": device, "continuous_ok": cont["ok"], "interrupted_ok": intd["ok"],
        "resumed_ok": res["ok"], "interrupted_epoch": start_epoch,
        "resume_start_epoch": (resumed or {}).get("start_epoch"),
        "resume_completed_epochs": (res["log"].get("completed_epochs") if res["ok"] else None),
        "resume_actual_updates": (res["log"].get("train_actual_updates") if res["ok"] else None),
        "new_epochs": (res["log"].get("completed_epochs", 0) - (resumed or {}).get("start_epoch", 0)
                       if res["ok"] else None),
        "best_source": best_source, "expected_new_best": expect_new_best,
        "best_new_in_resume": best_source == "new",
        "best_inherited": best_source == "inherited",
        "compare": cmp, "cache": shared.stats(),
    }
    interrupted_ok = (not intd["ok"]) and bool(intd.get("interrupted"))
    rec["interrupted_as_expected"] = interrupted_ok
    rec["pass"] = bool(
        cont["ok"] and interrupted_ok and res["ok"]
        and rec["new_epochs"] and rec["new_epochs"] > 0
        and rec["resume_start_epoch"] == start_epoch
        and (rec["best_new_in_resume"] if expect_new_best else rec["best_inherited"])
        and all(v for k, v in cmp.items() if k.endswith("_equal")))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "resume_epoch_check"))
    a = ap.parse_args()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = Path(a.out) / ts
    root.mkdir(parents=True, exist_ok=False)
    rec = {"schema": "qt-fe-v2-resume-epoch-check-v1", "created_at": ts,
           "root": str(root), "scenarios": {}}
    # New best after resume: moderate lr, 8 epochs, stop after 5.
    rec["scenarios"]["new_best_cpu"] = scenario(root, "new_best_cpu", 0.01, 8, 5, "cpu", True)
    # No new best after resume: aggressive lr, best early.
    rec["scenarios"]["no_new_best_cpu"] = scenario(root, "no_new_best_cpu", 0.0, 6, 4, "cpu", False)
    if torch.cuda.is_available():
        rec["scenarios"]["new_best_cuda"] = scenario(root, "new_best_cuda", 0.01, 8, 5, "cuda:0", True)
    rec["all_ok"] = all(v["pass"] for v in rec["scenarios"].values())
    (root / "resume_epoch_results.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"pass": v["pass"], "new_epochs": v["new_epochs"],
                          "best_source": v["best_source"],
                          "compare": {kk: vv for kk, vv in v["compare"].items()
                                      if kk.endswith("_equal")}}
                      for k, v in rec["scenarios"].items()},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"all_ok": rec["all_ok"], "out": str(root / "resume_epoch_results.json")},
                     ensure_ascii=False))
    return 0 if rec["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

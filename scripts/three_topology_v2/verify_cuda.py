#!/usr/bin/env python
"""Local CUDA (RTX 4070 Laptop) targeted verification.

Covers:
  * device/version record;
  * same-seed init independence on CUDA (init happens on CPU then .to(cuda));
  * short CUDA train + interrupt/resume: control (two continuous runs) vs
    resume discrepancy, so CUDA op nondeterminism is separated from resume bugs;
  * CUDA RNG captured in the checkpoint; device + optimizer state on CUDA;
  * all gradient models: CUDA forward / loss / backward / param update /
    validation / inverse / export on a tiny fold;
  * one mixed-size batch (topology 1+2+3 graphs) forward+backward for every model.
No full training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_v2 import dataset as ds_mod, graph_io, index, paths  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

REPORT = {"checks": {}, "ok": True}


def rec(name, passed, detail=""):
    REPORT["checks"][name] = {"pass": bool(passed), "detail": detail}
    if not passed:
        REPORT["ok"] = False


def tiny_ds(idx, st, ids_tr, ids_va, root):
    tr = ds_mod.FoldDataset(idx, ids_tr, st, root=root, dtype=torch.float32, verify_hash=False)
    va = ds_mod.FoldDataset(idx, ids_va, st, root=root, dtype=torch.float32, verify_hash=False)
    return tr, va


def cfg_for(name, seed, epochs, max_batches=1):
    import yaml
    base = yaml.safe_load((paths.CONFIG_DIR / "formal_baselines.yaml").read_text(encoding="utf-8"))
    return {
        "training": {"seed": seed, "batch_size": 4, "val_batch_size": 4, "epochs": epochs,
                     "lr": 1e-3, "weight_decay": 0.0, "early_stopping_patience": None,
                     "min_delta": 0.0, "grad_clip": 5.0,
                     "max_train_batches_per_epoch": max_batches,
                     "diagnostic": False, "device": "cuda:0"},
        "models": base["models"],
    }


def checksum(model):
    return float(sum(p.detach().double().abs().sum().item() for p in model.parameters()))


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA not available"}))
        return 1
    paths.ensure_derived_dirs()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda:0")
    REPORT["device_info"] = engine.device_info(dev)
    root = paths.package_root()
    idx = index.load_index(root=root, strict=True)
    outbase = paths.OUTPUT_ROOT / "verify_cuda"
    outbase.mkdir(parents=True, exist_ok=True)

    split = json.loads((paths.SPLIT_DIR / "split_cross_design_topology1.json").read_text(encoding="utf-8"))
    fold = next(f for f in split["folds"] if f["fold_id"] == "cd_fold_1")
    st = json.loads((paths.STATS_DIR / "cross_design_topology1" / "cd_fold_1.json").read_text(encoding="utf-8"))
    st["_file_sha256"] = "test"
    train_ds, val_ds = tiny_ds(idx, st, fold["train_sample_ids"][:8], fold["validation_sample_ids"][:4], root)
    data_hashes = {"samples_jsonl": idx["hashes"]["samples_jsonl"],
                   "dataset_manifest": idx["hashes"]["dataset_manifest"]}

    def identity_for(name, seed):
        return engine.build_identity(name, seed, st, cfg_for(name, seed, 2), train_ds.sample_ids,
                                     val_ds.sample_ids, data_hashes, "cross_design_topology1", "cd_fold_1")

    def identity(seed):
        return identity_for("mlp", seed)

    # ---- init same-seed independence on CUDA ----
    engine.seed_everything(11)
    a = registry.build_gradient_model("mlp", cfg_for("mlp", 11, 2)["models"]["mlp"])
    engine.seed_everything(999)
    _ = registry.build_gradient_model("hgt_global", cfg_for("mlp", 11, 2)["models"]["hgt_global"])
    engine.seed_everything(11)
    b = registry.build_gradient_model("mlp", cfg_for("mlp", 11, 2)["models"]["mlp"])
    rec("cuda_init_order_independent",
        all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())),
        "same seed+reorder -> equal init")

    # ---- control vs resume on CUDA ----
    engine.seed_everything(11)
    mA = registry.build_gradient_model("mlp", cfg_for("mlp", 11, 2)["models"]["mlp"])
    logA = engine.train_fold("mlp", mA, train_ds, val_ds, st, cfg_for("mlp", 11, 2),
                             outbase / "controlA", identity(11))
    engine.seed_everything(11)
    mC = registry.build_gradient_model("mlp", cfg_for("mlp", 11, 2)["models"]["mlp"])
    logC = engine.train_fold("mlp", mC, train_ds, val_ds, st, cfg_for("mlp", 11, 2),
                             outbase / "controlC", identity(11))
    control_param = abs(checksum(mA) - checksum(mC))
    control_sel = abs(logA["final_val"]["selection_metric"] - logC["final_val"]["selection_metric"])

    engine.seed_everything(11)
    mB = registry.build_gradient_model("mlp", cfg_for("mlp", 11, 2)["models"]["mlp"])
    engine.train_fold("mlp", mB, train_ds, val_ds, st, cfg_for("mlp", 11, 1),
                      outbase / "resumeB", identity(11))
    logB = engine.train_fold("mlp", mB, train_ds, val_ds, st, cfg_for("mlp", 11, 2),
                             outbase / "resumeB", identity(11),
                             resume_from=outbase / "resumeB" / "checkpoints" / "last.pt")
    resume_param = abs(checksum(mA) - checksum(mB))
    resume_sel = abs(logA["final_val"]["selection_metric"] - logB["final_val"]["selection_metric"])
    ck = torch.load(outbase / "resumeB" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    rec("cuda_device_used", logB["device_info"]["device"].startswith("cuda"), logB["device_info"]["device"])
    rec("cuda_checkpoint_has_cuda_rng", "cuda" in ck.get("rng", {}), list(ck.get("rng", {}).keys()))
    rec("cuda_optimizer_state_on_cuda",
        all(d.startswith("cuda") for d in logB.get("optimizer_state_devices", [])),
        logB.get("optimizer_state_devices"))
    rec("cuda_resume_requested", logB.get("resumed") is not None,
        str(logB.get("resumed", {}).get("start_epoch")))
    tol = max(control_param * 5.0, 1e-6)
    rec("cuda_resume_param_consistent", resume_param <= tol,
        f"resume_param_diff={resume_param:.3e}, control_param_diff={control_param:.3e}, tol={tol:.3e}")
    sel_tol = max(control_sel * 5.0, 1e-6)
    rec("cuda_resume_metric_consistent", resume_sel <= sel_tol,
        f"resume_sel_diff={resume_sel:.3e}, control_sel_diff={control_sel:.3e}, tol={sel_tol:.3e}")
    REPORT["cuda_nondeterminism"] = {"control_param_diff": control_param,
                                     "resume_param_diff": resume_param,
                                     "control_sel_diff": control_sel, "resume_sel_diff": resume_sel}

    # ---- all gradient models: CUDA train one batch + validate + export ----
    model_pass = {}
    for name in sorted(registry.GRADIENT_MODELS):
        try:
            engine.seed_everything(11)
            m = registry.build_gradient_model(name, cfg_for(name, 11, 1)["models"][name])
            log = engine.train_fold(name, m, train_ds, val_ds, st, cfg_for(name, 11, 1),
                                   outbase / f"all_{name}", identity_for(name, 11))
            model_pass[name] = {"pass": log["pass"]["all"],
                                "device": log["device_info"]["device"],
                                "selection": log["final_val"]["selection_metric"]}
        except Exception as exc:  # noqa: BLE001
            model_pass[name] = {"pass": False, "error": repr(exc)}
    rec("cuda_all_models", all(v.get("pass") for v in model_pass.values()), model_pass)

    # ---- mixed-size batch (topology 1+2+3) forward+backward ----
    row1 = next(r for r in idx["rows"] if r["topology_token"] == "topology1")
    row2 = next(r for r in idx["rows"] if r["topology_token"] == "topology2")
    row3 = next(r for r in idx["rows"] if r["topology_token"] == "topology3")
    g = [graph_io.load_graph(r, root=root, dtype=torch.float32) for r in (row1, row2, row3)]
    batch = Batch.from_data_list(g).to(dev)
    mixed = {}
    for name in sorted(registry.GRADIENT_MODELS):
        try:
            m = registry.build_gradient_model(name, cfg_for(name, 11, 1)["models"][name]).to(dev)
            m.train(); m.zero_grad()
            pd_, pf_ = m(batch)
            loss = torch.nn.functional.mse_loss(pd_, batch["mesh_node"].y_disp) + \
                torch.nn.functional.mse_loss(pf_, batch["beam_element"].y_force)
            loss.backward()
            finite = bool(torch.isfinite(loss).all() and
                          all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            mixed[name] = {"finite": finite, "n_nodes": int(batch["mesh_node"].num_nodes),
                           "n_beams": int(batch["beam_element"].num_nodes),
                           "shape_disp": list(pd_.shape), "shape_force": list(pf_.shape)}
        except Exception as exc:  # noqa: BLE001
            mixed[name] = {"finite": False, "error": repr(exc)}
    rec("cuda_mixed_size_batch", all(v.get("finite") for v in mixed.values()), mixed)

    (outbase / "verify_cuda_report.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(REPORT, ensure_ascii=False, indent=2))
    return 0 if REPORT["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

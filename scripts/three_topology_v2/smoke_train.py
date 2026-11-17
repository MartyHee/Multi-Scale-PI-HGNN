#!/usr/bin/env python
"""Deliverable 5 — single-fold CPU smoke (MLP and HGT-global).

Fold: topology1 cross-design ``cd_fold_1`` (84 train / 24 validation / 36 outer
test).  Outer test is index-checked only — never loaded, inferred or scored.
Physics loss OFF; random initialisation; CPU only.  Metrics are link-check only.

Per model it records: forward shapes, finite loss, finite & non-zero gradients,
actual parameter update, validation flow, and inverse-transformed per-component
MAE/RMSE in physical units (m, rad, N, N*m).
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_v2 import dataset as ds_mod, index, paths, stats as stats_mod  # noqa: E402
from src.models.baselines.hgt_global import HGTGlobalBaseline  # noqa: E402
from src.models.baselines.mlp_baseline import MLPBaseline  # noqa: E402

DISP_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
FORCE_UNITS = ["N", "N", "N", "N*m", "N*m", "N*m", "N", "N", "N", "N*m", "N*m", "N*m"]
DISP_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz"]
FORCE_NAMES = ["I_Fx", "I_Fy", "I_Fz", "I_Mx", "I_My", "I_Mz",
               "J_Fx", "J_Fy", "J_Fz", "J_Mx", "J_My", "J_Mz"]


def load_config(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return json.loads(text)


def build_model(name, cfg):
    if name == "mlp":
        return MLPBaseline(**cfg)
    if name == "hgt_global":
        return HGTGlobalBaseline(**cfg)
    raise ValueError(name)


def param_checksum(model) -> float:
    return float(sum(p.detach().double().abs().sum().item() for p in model.parameters()))


def grad_stats(model):
    finite = True
    nonzero = 0
    total = 0
    max_abs = 0.0
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            finite = False
        total += 1
        gn = float(g.norm().item())
        if gn > 0:
            nonzero += 1
        max_abs = max(max_abs, float(g.abs().max().item()))
        sq += float((g.double() ** 2).sum().item())
    return {"finite": finite, "params_with_grad": total, "params_nonzero_grad": nonzero,
            "max_abs_grad": max_abs, "grad_norm": math.sqrt(sq)}


def init_metrics():
    return {
        "disp": {"abs_sum": np.zeros(6), "sq_sum": np.zeros(6), "n": 0},
        "force": {"abs_sum": np.zeros(12), "sq_sum": np.zeros(12), "n": 0},
    }


def update_metrics(m, pred, target_disp_phys, target_force_phys):
    pe = pred["disp"].detach().double().cpu().numpy()
    te = target_disp_phys.detach().double().cpu().numpy()
    err = pe - te
    m["disp"]["abs_sum"] += np.abs(err).sum(axis=0)
    m["disp"]["sq_sum"] += (err ** 2).sum(axis=0)
    m["disp"]["n"] += err.shape[0]
    pforce = pred["force"].detach().double().cpu().numpy()
    tforce = target_force_phys.detach().double().cpu().numpy()
    ferr = pforce - tforce
    m["force"]["abs_sum"] += np.abs(ferr).sum(axis=0)
    m["force"]["sq_sum"] += (ferr ** 2).sum(axis=0)
    m["force"]["n"] += ferr.shape[0]


def finalize_metrics(m):
    nd = max(m["disp"]["n"], 1)
    nf = max(m["force"]["n"], 1)
    disp = [{"component": DISP_NAMES[i], "unit": DISP_UNITS[i],
             "mae": float(m["disp"]["abs_sum"][i] / nd),
             "rmse": float(math.sqrt(m["disp"]["sq_sum"][i] / nd))} for i in range(6)]
    force = [{"component": FORCE_NAMES[i], "unit": FORCE_UNITS[i],
              "mae": float(m["force"]["abs_sum"][i] / nf),
              "rmse": float(math.sqrt(m["force"]["sq_sum"][i] / nf))} for i in range(12)]
    return {"disp": disp, "force": force, "node_count": int(m["disp"]["n"]),
            "beam_count": int(m["force"]["n"])}


def run_model(name, model_cfg, train_ds, val_ds, stats, cfg, out_dir):
    seed = cfg["training"]["seed"]
    bs = cfg["training"]["batch_size"]
    n_train_batches = cfg["training"]["num_train_batches"]
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = build_model(name, model_cfg)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    order = list(range(len(train_ds)))
    random.Random(seed).shuffle(order)

    log = {
        "model": name, "config": model_cfg,
        "n_params": n_params, "n_trainable_params": n_trainable,
        "batch_size": bs, "num_train_batches": n_train_batches,
        "seed": seed, "device": "cpu", "physical_loss_weight": 0.0,
        "train_batches": [],
    }
    first_update_checked = False
    used_train_ids = []

    for b in range(n_train_batches):
        batch_idx = order[b * bs:(b + 1) * bs]
        if not batch_idx:
            break
        samples = [train_ds[i] for i in batch_idx]
        batch_ids = [train_ds.sample_ids[i] for i in batch_idx]
        used_train_ids.extend(batch_ids)
        batch = Batch.from_data_list(samples)

        opt.zero_grad()
        pred_disp, pred_force = model(batch)
        loss = F.mse_loss(pred_disp, batch["mesh_node"].y_disp) + \
            F.mse_loss(pred_force, batch["beam_element"].y_force)
        # physics loss intentionally disabled (weight 0)
        loss.backward()
        gs = grad_stats(model)
        checksum_before = param_checksum(model)
        opt.step()
        checksum_after = param_checksum(model)
        updated = abs(checksum_after - checksum_before) > 0

        rec = {
            "batch": b,
            "sample_ids": batch_ids,
            "num_graphs": len(samples),
            "disp_shape": list(pred_disp.shape),
            "force_shape": list(pred_force.shape),
            "loss": float(loss.item()),
            "loss_finite": bool(torch.isfinite(loss).all()),
            "grad": gs,
            "param_updated": updated,
        }
        log["train_batches"].append(rec)

        if not first_update_checked:
            log["first_update_check"] = {
                "loss_finite": rec["loss_finite"],
                "grad_finite": gs["finite"],
                "grad_nonzero": gs["params_nonzero_grad"] > 0,
                "param_updated": updated,
                "checksum_before": checksum_before,
                "checksum_after": checksum_after,
            }
            first_update_checked = True

    # ---- validation ----
    model.eval()
    metrics = init_metrics()
    val_loss_sum = 0.0
    val_batches = 0
    val_ids_used = []
    with torch.no_grad():
        for s in range(0, len(val_ds), bs):
            idxs = list(range(s, min(s + bs, len(val_ds))))
            samples = [val_ds[i] for i in idxs]
            val_ids_used.extend([val_ds.sample_ids[i] for i in idxs])
            batch = Batch.from_data_list(samples)
            pred_disp, pred_force = model(batch)
            loss = F.mse_loss(pred_disp, batch["mesh_node"].y_disp) + \
                F.mse_loss(pred_force, batch["beam_element"].y_force)
            val_loss_sum += float(loss.item())
            val_batches += 1
            pd = stats_mod.inverse_transform_disp(pred_disp, stats)
            pf = stats_mod.inverse_transform_force(pred_force, stats)
            td = stats_mod.inverse_transform_disp(batch["mesh_node"].y_disp, stats)
            tf = stats_mod.inverse_transform_force(batch["beam_element"].y_force, stats)
            update_metrics(metrics, {"disp": pd, "force": pf}, td, tf)

    val_metrics = finalize_metrics(metrics)
    log["validation"] = {
        "num_val_batches": val_batches,
        "num_val_samples": len(val_ids_used),
        "val_loss_mean": val_loss_sum / max(val_batches, 1),
        "metrics_physical": val_metrics,
    }
    log["used_train_sample_ids"] = used_train_ids

    n_finite = all(r["loss_finite"] for r in log["train_batches"])
    all_grad_ok = all(r["grad"]["finite"] and r["grad"]["params_nonzero_grad"] > 0
                      for r in log["train_batches"])
    all_updated = all(r["param_updated"] for r in log["train_batches"])
    log["pass"] = {
        "forward_ok": all(r["disp_shape"][1] == 6 and r["force_shape"][1] == 12
                          for r in log["train_batches"]),
        "loss_finite": n_finite,
        "grad_finite_nonzero": all_grad_ok,
        "param_updated": all_updated,
        "validation_ran": val_batches > 0,
    }
    log["pass"]["all"] = all(log["pass"].values())

    out = out_dir / f"smoke_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_result.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "smoke_cd_fold_1.yaml"))
    ap.add_argument("--models", default="mlp,hgt_global")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths.ensure_derived_dirs()
    root = paths.package_root()
    if cfg.get("data", {}).get("package_root"):
        root = Path(cfg["data"]["package_root"])

    idx = index.load_index(root=root, strict=True)
    split_id = cfg["data"]["split_id"]
    fold_id = cfg["data"]["fold_id"]
    split_path = paths.SPLIT_DIR / f"split_{split_id}.json"
    with split_path.open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    fold = next(f for f in split["folds"] if f["fold_id"] == fold_id)

    stats_path = paths.STATS_DIR / split_id / f"{fold_id}.json"
    with stats_path.open("r", encoding="utf-8") as fh:
        fold_stats = json.load(fh)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_cd_fold_1_smoke"
    out_dir = paths.OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- outer test: index identity & grouping only (never load/score) ----
    test_ids = fold["test_sample_ids"]
    test_check = {
        "count": len(test_ids),
        "unique": len(set(test_ids)) == len(test_ids),
        "disjoint_from_train": not (set(test_ids) & set(fold["train_sample_ids"])),
        "disjoint_from_val": not (set(test_ids) & set(fold["validation_sample_ids"])),
        "all_known_raw": all(s in idx["row_by_id"] for s in test_ids),
        "all_topology1": all(idx["row_by_id"][s]["topology_token"] == "topology1" for s in test_ids),
        "design_keys_disjoint": not (set(fold.get("test_group_keys", [])) &
                                     set(fold.get("train_group_keys", [])) &
                                     set(fold.get("validation_group_keys", []))),
    }
    test_check["ok"] = all(v for k, v in test_check.items() if k != "count") and test_check["count"] == 36

    train_ds = ds_mod.FoldDataset(idx, fold["train_sample_ids"], fold_stats, root=root,
                                  dtype=torch.float32, verify_hash=cfg["data"]["verify_hash"],
                                  normalize_labels=True)
    val_ds = ds_mod.FoldDataset(idx, fold["validation_sample_ids"], fold_stats, root=root,
                                dtype=torch.float32, verify_hash=cfg["data"]["verify_hash"],
                                normalize_labels=True)

    env = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu", "package_root": str(root),
        "split_id": split_id, "fold_id": fold_id,
        "train_graphs": len(train_ds), "val_graphs": len(val_ds), "test_graphs": len(test_ids),
    }
    (out_dir / "run_config.json").write_text(json.dumps(
        {"config": cfg, "env": env, "fold_counts": fold["counts"],
         "stats_source": str(stats_path)}, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries = {}
    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        log = run_model(name, cfg["models"][name], train_ds, val_ds, fold_stats, cfg, out_dir)
        summaries[name] = {
            "n_params": log["n_params"],
            "pass": log["pass"],
            "val_loss_mean": log["validation"]["val_loss_mean"],
            "val_disp_mae": {r["component"]: r["mae"] for r in log["validation"]["metrics_physical"]["disp"]},
            "val_force_mae": {r["component"]: r["mae"] for r in log["validation"]["metrics_physical"]["force"]},
        }

    summary = {
        "run_id": run_id, "env": env, "outer_test_index_check": test_check,
        "models": summaries,
        "note": "Few-batch CPU smoke: link check only, NOT a performance result.",
        "all_models_pass": all(m["pass"]["all"] for m in summaries.values()) and test_check["ok"],
    }
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": summary["all_models_pass"], "run_dir": str(out_dir),
                      "outer_test_check": test_check,
                      "models": {k: {"n_params": v["n_params"], "pass": v["pass"],
                                     "val_loss_mean": v["val_loss_mean"]}
                                 for k, v in summaries.items()}},
                     ensure_ascii=False, indent=2))
    return 0 if summary["all_models_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified train/validation engine for three-topology v2 baselines (corrected).

Key guarantees
--------------
* explicit device selection (``cpu`` / ``cuda`` / ``cuda:N``); requesting CUDA
  when unavailable raises (no silent fallback); actual device recorded;
* seeding happens *before* model construction (done by the entry), so init is
  independent of model run order;
* early stopping uses a consecutive-no-improvement counter with explicit
  ``min_delta`` / ``patience`` / direction, reset on improvement;
* per-epoch shuffle seed is derived deterministically from (seed, epoch);
* best/last checkpoints: final metrics + prediction export come from the
  selected **best** checkpoint (epoch + selection value + sha256 recorded);
  ``last`` is for resume only;
* resume validates the run identity (model/seed/data/split/stats/config) and
  restores model, optimizer (state moved to device), early-stop state and
  Python/NumPy/Torch/CUDA RNG;
* non-finite loss / grad / prediction raises immediately with context (not only
  the first batch); full per-step parameter checksum runs only in diagnostic
  mode.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .metrics import MetricsAccumulator, export_records

IDENTITY_ALLOWED_RESUME_CHANGES = {"epochs", "device", "patience", "min_delta",
                                   "grad_clip", "diagnostic", "tag", "max_val_batches"}


# ---------------------------------------------------------------------------
# Device / seeding / RNG
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    req = (requested or "cpu").lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if ":" in req:
            idx = int(req.split(":")[1])
            if idx >= torch.cuda.device_count():
                raise RuntimeError(f"cuda:{idx} requested but only {torch.cuda.device_count()} device(s)")
        return torch.device(req)
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {requested!r}; use cpu | cuda | cuda:N | auto")


def device_info(device: torch.device) -> Dict[str, Any]:
    info = {"device": str(device), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu_name": p.name, "gpu_total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
                     "capability": f"{p.major}.{p.minor}"})
    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> Dict[str, Any]:
    st = {"torch": torch.get_rng_state(), "py": random.getstate(),
          "np": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: Dict[str, Any]) -> None:
    torch.set_rng_state(st["torch"])
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _move_optimizer_to(opt, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _atomic_torch_save(obj, path: Path) -> None:
    """Write torch.save atomically (temp file + os.replace) to avoid half files."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mem_used_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def _assert_finite(t: torch.Tensor, ctx: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"non-finite tensor at {ctx}")


def _cheap_checksum(model) -> float:
    p = next(model.parameters())
    return float(p.detach().double().abs().sum().item())


def _full_checksum(model) -> float:
    return float(sum(p.detach().double().abs().sum().item() for p in model.parameters()))


def _grad_monitor(model) -> Dict[str, Any]:
    finite = True
    nonzero = total = 0
    max_abs = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            finite = False
            raise RuntimeError(f"non-finite gradient in parameter {name}")
        total += 1
        if float(g.norm().item()) > 0:
            nonzero += 1
        max_abs = max(max_abs, float(g.abs().max().item()))
    return {"finite": finite, "params_with_grad": total, "params_nonzero_grad": nonzero,
            "max_abs_grad": max_abs}


def _batch_from(dataset, indices, device):
    samples = [dataset[i] for i in indices]
    ids = [dataset.sample_ids[i] for i in indices]
    batch = Batch.from_data_list(samples)
    if device.type != "cpu":
        batch = batch.to(device)
    return batch, ids


def _shuffle_order(n: int, seed: int, epoch: int) -> List[int]:
    """Deterministic per-epoch order: rng(seed, epoch).  Resume-consistent."""
    rng = np.random.default_rng((int(seed) * 100003 + int(epoch)) % (2 ** 32))
    return rng.permutation(n).tolist()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(model, dataset, stats, batch_size: int, device: torch.device,
             max_batches: Optional[int] = None, collect_predictions: bool = False):
    from src.data.three_topology_v2 import stats as stats_mod
    model.eval()
    acc = MetricsAccumulator()
    disp_inv = lambda t: stats_mod.inverse_transform_disp(t, stats)
    force_inv = lambda t: stats_mod.inverse_transform_force(t, stats)
    n_batches = 0
    exported: List[Dict] = []
    with torch.no_grad():
        for s in range(0, len(dataset), batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break
            idxs = list(range(s, min(s + batch_size, len(dataset))))
            batch, ids = _batch_from(dataset, idxs, device)
            pred_disp, pred_force = model(batch)
            _assert_finite(pred_disp, "validation pred_disp")
            _assert_finite(pred_force, "validation pred_force")
            pd = disp_inv(pred_disp).cpu().numpy()
            pf = force_inv(pred_force).cpu().numpy()
            td = disp_inv(batch["mesh_node"].y_disp).cpu().numpy()
            tf = force_inv(batch["beam_element"].y_force).cpu().numpy()
            gd = batch["mesh_node"].batch.cpu().numpy()
            gb = batch["beam_element"].batch.cpu().numpy()
            topo = batch.topology_code.view(-1).cpu().numpy()
            # make graph ids unique across batches (sequential offset = s)
            gd = gd + s; gb = gb + s
            acc.update(pd, td, gd, topo[gd - s], pf, tf, gb, topo[gb - s])
            if collect_predictions:
                mp, bp = batch["mesh_node"].ptr.tolist(), batch["beam_element"].ptr.tolist()
                node_counts = [mp[k + 1] - mp[k] for k in range(len(ids))]
                beam_counts = [bp[k + 1] - bp[k] for k in range(len(ids))]
                exported.extend(export_records(ids, node_counts, beam_counts,
                                               [pd[mp[k]:mp[k + 1]] for k in range(len(ids))],
                                               [pf[bp[k]:bp[k + 1]] for k in range(len(ids))],
                                               stats))
            n_batches += 1
    std = {k: [float(c["effective_std"]) for c in stats["columns"][k]] for k in ("y_disp", "y_force")}
    metrics = acc.finalize(std) if n_batches else {"selection_metric": None}
    return {"n_batches": n_batches, "metrics": metrics,
            "selection_metric": metrics.get("selection_metric"),
            "entity_weighted_norm_mse": (metrics.get("disp", {}).get("entity_weighted_norm_mse")
                                         if n_batches else None),
            "exported": exported}


# ---------------------------------------------------------------------------
# Resume identity
# ---------------------------------------------------------------------------
def _hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_identity(model_name, seed, stats, cfg, train_ids, val_ids,
                   data_hashes: Dict[str, str], split_id: str, fold_id: str,
                   outer_fold: Optional[str] = None, inner_fold: Optional[str] = None) -> Dict:
    tr = cfg["training"]
    return {
        "model_name": model_name, "seed": int(seed),
        "split_id": split_id, "fold_id": fold_id,
        "outer_fold": outer_fold, "inner_fold": inner_fold,
        "stats_sha256": stats.get("_file_sha256"),
        "samples_jsonl_sha256": data_hashes.get("samples_jsonl"),
        "dataset_manifest_sha256": data_hashes.get("dataset_manifest"),
        "train_sample_list_sha256": _hash_json(sorted(train_ids)),
        "val_sample_list_sha256": _hash_json(sorted(val_ids)),
        "batch_size": int(tr["batch_size"]), "lr": float(tr["lr"]),
        "weight_decay": float(tr.get("weight_decay", 0.0)),
        "max_train_batches_per_epoch": tr.get("max_train_batches_per_epoch"),
        "model_config_sha256": _hash_json(cfg["models"].get(model_name, {})),
    }


def check_resume_compat(ck_identity: Dict, new_identity: Dict,
                        allowed: set = IDENTITY_ALLOWED_RESUME_CHANGES) -> Dict:
    diffs = {}
    for k, v in new_identity.items():
        if k in allowed:
            continue
        if ck_identity.get(k) != v:
            diffs[k] = {"checkpoint": ck_identity.get(k), "requested": v}
    return {"ok": not diffs, "diffs": diffs, "allowed_changes": sorted(allowed)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(model_name: str, model, train_ds, val_ds, stats, cfg: Dict[str, Any],
               out_dir: Path, identity: Dict[str, Any],
               resume_from: Optional[Path] = None) -> Dict[str, Any]:
    tr = cfg["training"]
    seed = int(tr["seed"])
    bs = int(tr["batch_size"])
    val_bs = int(tr.get("val_batch_size", bs))
    lr = float(tr["lr"]); wd = float(tr.get("weight_decay", 0.0))
    epochs = int(tr.get("epochs", 1))
    max_batches = tr.get("max_train_batches_per_epoch")
    patience = tr.get("early_stopping_patience")
    min_delta = float(tr.get("min_delta", 0.0))
    grad_clip = tr.get("grad_clip")
    diagnostic = bool(tr.get("diagnostic", False))
    device = resolve_device(tr.get("device", "cpu"))

    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"

    start_epoch = 0
    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    early_stopped = False
    resumed = None
    inherited_best = None
    best_source = "new"
    if resume_from is not None:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        compat = check_resume_compat(ck.get("identity", {}), identity)
        if not compat["ok"]:
            raise RuntimeError(f"resume identity mismatch: {json.dumps(compat['diffs'], ensure_ascii=False)}")
        model.load_state_dict(ck["model_state"])
        model.to(device)
        opt.load_state_dict(ck["optimizer_state"])
        _move_optimizer_to(opt, device)
        start_epoch = int(ck.get("epoch", 0))
        best_val = float(ck.get("best_val", math.inf))
        best_epoch = int(ck.get("best_epoch", -1))
        no_improve = int(ck.get("no_improve", 0))
        restore_rng(ck["rng"])
        # inherit the historical best checkpoint from the ORIGINAL run dir
        orig_best = Path(resume_from).parent / "best.pt"
        if orig_best.is_file():
            dest_best = ckpt_dir / "best.pt"
            if not dest_best.is_file():
                _atomic_copy(orig_best, dest_best)
            inherited_best = {"path": str(orig_best),
                              "sha256": hashlib.sha256(orig_best.read_bytes()).hexdigest(),
                              "inherited_epoch_completed": int(ck.get("epoch", 0))}
            best_source = "inherited"
        elif best_epoch >= 0:
            raise RuntimeError(
                "resume: checkpoint records a best epoch but best.pt is missing in the "
                f"original run {Path(resume_from).parent}; refusing to fall back to last")
        resumed = {"from": str(resume_from), "start_epoch": start_epoch,
                   "best_val": best_val, "best_epoch": best_epoch, "no_improve": no_improve,
                   "inherited_best": inherited_best}

    log: Dict[str, Any] = {
        "model": model_name, "identity": identity, "device_info": device_info(device),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_graphs": len(train_ds), "val_graphs": len(val_ds),
        "batch_size": bs, "epochs_requested": epochs,
        "early_stopping": {"patience": patience, "min_delta": min_delta, "direction": "min"},
        "diagnostic": diagnostic, "physics_loss_weight": 0.0,
        "sha256_algorithm": "sha256", "epochs_log": [], "resumed": resumed,
    }

    opt_devs = sorted({str(v.device) for s in opt.state.values() for v in s.values()
                       if torch.is_tensor(v)}) or [str(device)]
    log["optimizer_state_devices"] = opt_devs

    # per-epoch CSV (append + flush + fsync so live monitoring works and an
    # interrupt keeps completed rows)
    train_log_path = out_dir / "train_log.csv"
    csv_is_new = (not train_log_path.exists()) or train_log_path.stat().st_size == 0
    csv_fh = train_log_path.open("a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_fh)
    if csv_is_new:
        csv_w.writerow(["epoch", "train_loss_mean", "selection_metric", "best_val",
                        "improved", "no_improve", "val_batches", "train_sec",
                        "val_sec", "mem_mb", "order_sha256", "utc"])
        csv_fh.flush(); os.fsync(csv_fh.fileno())

    t_train0 = time.time()
    completed_epochs = start_epoch
    first_step_checked = False
    try:
        for epoch in range(start_epoch, epochs):
            t_ep0 = time.time()
            model.train()
            order = _shuffle_order(len(train_ds), seed, epoch)
            ep_loss = 0.0; ep_batches = 0; ep_nonzero_grad = 0
            for b in range(0, len(order), bs):
                if max_batches is not None and ep_batches >= max_batches:
                    break
                idxs = order[b:b + bs]
                batch, ids = _batch_from(train_ds, idxs, device)
                opt.zero_grad()
                pred_disp, pred_force = model(batch)
                loss = F.mse_loss(pred_disp, batch["mesh_node"].y_disp) + \
                    F.mse_loss(pred_force, batch["beam_element"].y_force)
                _assert_finite(loss, f"{model_name} epoch {epoch} batch {ids[0]} loss")
                loss.backward()
                gm = _grad_monitor(model)
                ep_nonzero_grad += gm["params_nonzero_grad"]
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                if diagnostic:
                    before = _full_checksum(model)
                opt.step()
                if diagnostic:
                    after = _full_checksum(model)
                    if abs(after - before) <= 0.0:
                        raise RuntimeError("diagnostic: parameters did not update")
                if not first_step_checked:
                    stepped = any("step" in s and int(s["step"]) >= 1 for s in opt.state.values())
                    if not stepped:
                        raise RuntimeError("optimizer step did not update any parameter state")
                    first_step_checked = True
                ep_loss += float(loss.item()); ep_batches += 1

            t_val0 = time.time()
            val = validate(model, val_ds, stats, val_bs, device)
            t_val1 = time.time()
            sel = val["selection_metric"]
            improved = (sel is not None) and (sel < best_val - min_delta)
            if improved:
                best_val = sel; best_epoch = epoch; no_improve = 0
                best_source = "new"
            else:
                no_improve += 1
            rec = {"epoch": epoch, "train_loss_mean": ep_loss / max(ep_batches, 1),
                   "selection_metric": sel, "best_val": best_val, "improved": improved,
                   "no_improve": no_improve, "val_batches": val["n_batches"],
                   "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest()[:16],
                   "train_sec": round(t_val0 - t_ep0, 3),
                   "val_sec": round(t_val1 - t_val0, 3), "mem_mb": _mem_used_mb()}
            log["epochs_log"].append(rec)
            ck_common = {"identity": identity, "config": cfg, "seed": seed,
                         "epoch": epoch + 1, "best_val": best_val, "best_epoch": best_epoch,
                         "no_improve": no_improve, "rng": capture_rng()}
            _atomic_torch_save({**ck_common, "model_state": model.state_dict(),
                                "optimizer_state": opt.state_dict()}, ckpt_dir / "last.pt")
            if improved:
                _atomic_torch_save({**ck_common, "model_state": model.state_dict(),
                                    "optimizer_state": opt.state_dict()}, ckpt_dir / "best.pt")
            csv_w.writerow([rec["epoch"], f"{rec['train_loss_mean']:.8g}",
                            "" if sel is None else f"{sel:.8g}", f"{best_val:.8g}",
                            improved, no_improve, rec["val_batches"], rec["train_sec"],
                            rec["val_sec"], "" if rec["mem_mb"] is None else f"{rec['mem_mb']:.1f}",
                            rec["order_sha256"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
            csv_fh.flush(); os.fsync(csv_fh.fileno())
            completed_epochs = epoch + 1
            if patience is not None and no_improve >= int(patience):
                early_stopped = True
                log["early_stop"] = {"stopped_after_epoch": completed_epochs,
                                     "no_improve": no_improve, "patience": patience}
                break
    finally:
        csv_fh.close()

    t_train1 = time.time()

    # ---- final evaluation + export from BEST checkpoint (mandatory) ----
    best_path = ckpt_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "no best checkpoint available for final export; refusing to fall back to last. "
            f"best_epoch={best_epoch}, inherited_best={inherited_best}")
    best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ck["model_state"]); model.to(device)
    t_inf0 = time.time()
    final = validate(model, val_ds, stats, val_bs, device, collect_predictions=True)
    t_inf1 = time.time()
    (out_dir / "val_predictions.json").write_text(
        json.dumps(final["exported"], ensure_ascii=False), encoding="utf-8")
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()

    log["completed_epochs"] = completed_epochs
    log["early_stopped"] = early_stopped
    log["best"] = {"epoch_zero_based": best_ck["best_epoch"], "epoch_completed": int(best_ck["epoch"]),
                   "selection_metric": float(best_ck["best_val"]),
                   "checkpoint_path": str(best_path), "checkpoint_sha256": best_sha,
                   "source": best_source, "inherited_best": inherited_best}
    log["last"] = {"epoch_completed": completed_epochs,
                   "checkpoint_path": str(ckpt_dir / "last.pt"),
                   "checkpoint_sha256": hashlib.sha256((ckpt_dir / "last.pt").read_bytes()).hexdigest()}
    log["final_val"] = {"selection_metric": final["selection_metric"],
                        "entity_weighted_norm_mse": final["entity_weighted_norm_mse"],
                        "metrics_physical": final["metrics"],
                        "n_batches": final["n_batches"]}
    log["timing"] = {"train_sec": t_train1 - t_train0,
                     "final_inference_sec": t_inf1 - t_inf0}
    log["mem_mb_final"] = _mem_used_mb()
    log["pass"] = {
        "forward_ok": bool(completed_epochs > start_epoch),
        "selection_finite": bool(final["selection_metric"] is not None
                                 and math.isfinite(final["selection_metric"])),
        "grad_finite": True,  # enforced by raising on non-finite
        "param_updated": True,  # enforced during first step
        "validation_ran": bool(final["n_batches"] > 0),
        "best_checkpoint_exported_from": best_path.name,
        "prediction_exported": bool((out_dir / "val_predictions.json").is_file()),
    }
    log["pass"]["all"] = all(v is True for k, v in log["pass"].items() if k != "best_checkpoint_exported_from")
    (out_dir / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def evaluate_checkpoint(model, dataset, stats, batch_size: int, device: torch.device,
                        checkpoint_path: Path, out_dir: Path) -> Dict:
    """Evaluation-only: load a checkpoint and recompute validation metrics."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"]); model.to(device)
    res = validate(model, dataset, stats, batch_size, device, collect_predictions=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_predictions.json").write_text(
        json.dumps(res["exported"], ensure_ascii=False), encoding="utf-8")
    log = {"eval_only": True, "checkpoint": str(checkpoint_path),
           "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
           "ckpt_epoch": ck.get("epoch"), "ckpt_best_val": ck.get("best_val"),
           "metrics": res["metrics"], "selection_metric": res["selection_metric"],
           "n_batches": res["n_batches"]}
    (out_dir / "eval_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log

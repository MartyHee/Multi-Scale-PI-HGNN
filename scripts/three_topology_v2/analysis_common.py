"""Shared loaders/metrics for the first+second batch review.

Read-only: reads extracted archives, the frozen split/stats and validation graph
labels. No inference, no training, no outer-test labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402
from src.train.three_topology_v2.metrics import DISP_NAMES, DISP_UNITS, FORCE_NAMES, FORCE_UNITS  # noqa: E402

SPLIT_ID = "cross_design_topology1"
FOLD_ID = "cd_fold_1"

BATCHES = {
    "fb_20261117_01": {
        "dir": MODEL_ROOT / "remote_artifacts" / "first_batch_fb_20261117_01_review" / "batch" / "fb_20261117_01",
        "archive": MODEL_ROOT / "remote_artifacts" / "first_batch_fb_20261117_01_artifacts.tar.gz",
        "commit": "06c1bf1a93e72286c1c6f63876826a73dbc83e30",
    },
    "sb_20261118_01": {
        "dir": MODEL_ROOT / "remote_artifacts" / "batch_sb_20261118_01_review" / "batch" / "sb_20261118_01",
        "archive": MODEL_ROOT / "remote_artifacts" / "batch_sb_20261118_01_artifacts.tar.gz",
        "commit": "836d86445c1ba6462e13986c113221a5975ef698",
    },
}

MODEL_ORDER = ["mlp", "gcn", "graphsage", "gat", "gin", "rgcn", "hgt", "hgt_global", "hist_gb", "ridge"]
MODEL_LABEL = {
    "mlp": "MLP", "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT",
    "gin": "GIN", "rgcn": "R-GCN", "hgt": "HGT", "hgt_global": "HGT-global",
    "hist_gb": "HistGB", "ridge": "Ridge",
}
GRADIENT = {"mlp", "gcn", "graphsage", "gat", "gin", "rgcn", "hgt", "hgt_global"}
BATCH_OF = {m: "fb_20261117_01" for m in ["mlp", "hgt_global", "hist_gb", "ridge"]}
BATCH_OF.update({m: "sb_20261118_01" for m in ["gcn", "graphsage", "gat", "gin", "rgcn", "hgt"]})


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_context():
    idx = index.load_index(strict=True)
    split = read_json(paths.SPLIT_DIR / f"split_{SPLIT_ID}.json")
    fold = next(f for f in split["folds"] if f["fold_id"] == FOLD_ID)
    val_ids = sorted(fold["validation_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    stats = read_json(paths.STATS_DIR / SPLIT_ID / f"{FOLD_ID}.json")
    std_d = np.array([c["effective_std"] for c in stats["columns"]["y_disp"]], dtype=np.float64)
    std_f = np.array([c["effective_std"] for c in stats["columns"]["y_force"]], dtype=np.float64)
    labels, meta = {}, {}
    for sid in val_ids:
        g = graph_io.load_graph(idx["row_by_id"][sid], dtype=torch.float64, verify_hash=True)
        labels[sid] = (g["mesh_node"].y_disp.numpy(), g["beam_element"].y_force.numpy())
        meta[sid] = {"n_nodes": int(g["mesh_node"].x.shape[0]),
                     "n_beams": int(g["beam_element"].x.shape[0]),
                     "design_id": idx["row_by_id"][sid]["design_id"],
                     "template_id": idx["row_by_id"][sid]["template_id"]}
    return {"index": idx, "split": split, "fold": fold, "val_ids": val_ids,
            "test_ids": test_ids, "stats": stats, "std_d": std_d, "std_f": std_f,
            "labels": labels, "meta": meta}


def discover_runs(batch_dir: Path):
    runs = []
    for st in sorted(Path(batch_dir).glob("*/*/STATUS.json")):
        d = st.parent
        runs.append({"worker": d.parent.name, "run": d.name, "dir": d,
                     "model": d.name.split("_seed")[0] if "_seed" in d.name else d.name,
                     "seed": d.name.split("_seed")[1] if "_seed" in d.name else "-"})
    return runs


def recompute(recs: List[dict], ctx) -> Dict:
    by_id = {r["sample_id"]: r for r in recs}
    val = set(ctx["val_ids"])
    if set(by_id) != val:
        raise RuntimeError(f"prediction ids != validation ids: missing={sorted(val-set(by_id))[:3]} "
                           f"extra={sorted(set(by_id)-val)[:3]}")
    std_d, std_f = ctx["std_d"], ctx["std_f"]
    comps, per_graph = [], []
    gd, gf = [], []
    D, F = [], []
    for sid in ctx["val_ids"]:
        r = by_id[sid]; td, tf = ctx["labels"][sid]
        pd = np.asarray(r["pred_disp"], dtype=np.float64)
        pf = np.asarray(r["pred_force"], dtype=np.float64)
        if pd.shape != td.shape or pf.shape != tf.shape:
            raise RuntimeError(f"{sid}: shape mismatch {pd.shape}/{td.shape} {pf.shape}/{tf.shape}")
        ed, ef = pd - td, pf - tf
        D.append(ed); F.append(ef)
        dmse = (ed ** 2).mean(axis=0); fmse = (ef ** 2).mean(axis=0)
        gd.append(dmse); gf.append(fmse)
        sel = 0.5 * (float(np.mean(dmse / std_d ** 2)) + float(np.mean(fmse / std_f ** 2)))
        per_graph.append({"sample_id": sid, "design_id": ctx["meta"][sid]["design_id"],
                          "template_id": ctx["meta"][sid]["template_id"],
                          "n_nodes": ctx["meta"][sid]["n_nodes"],
                          "n_beams": ctx["meta"][sid]["n_beams"],
                          "selection": sel,
                          "disp_mae_entity": float(np.abs(ed).mean()),
                          "force_mae_entity": float(np.abs(ef).mean())})
    D = np.concatenate(D); F = np.concatenate(F)
    for kind, E, names, units, std in (("disp", D, DISP_NAMES, DISP_UNITS, std_d),
                                       ("force", F, FORCE_NAMES, FORCE_UNITS, std_f)):
        for c, name in enumerate(names):
            comps.append({"task": kind, "component": name, "unit": units[c],
                          "mae": float(np.abs(E[:, c]).mean()),
                          "rmse": float(np.sqrt(np.mean(E[:, c] ** 2))),
                          "norm_mae_by_train_std": float(np.abs(E[:, c]).mean() / std[c]),
                          "mean_abs_target": float(np.abs(np.concatenate(
                              [ctx["labels"][s][0 if kind == "disp" else 1][:, c]
                               for s in ctx["val_ids"]])).mean())})
    gd = np.stack(gd); gf = np.stack(gf)
    sel_d = float(np.nanmean(gd / std_d ** 2)); sel_f = float(np.nanmean(gf / std_f ** 2))
    return {"selection": 0.5 * (sel_d + sel_f), "selection_disp": sel_d,
            "selection_force": sel_f, "components": comps, "per_graph": per_graph}


def load_batch_runs(batch: str):
    """Return run records with logs/predictions and recomputed metrics."""
    ctx = load_context()
    out = []
    for r in discover_runs(BATCHES[batch]["dir"]):
        d = r["dir"]
        rec = dict(r)
        rec["batch_id"] = batch
        rec["status"] = read_json(d / "STATUS.json") if (d / "STATUS.json").is_file() else {}
        rec["task"] = read_json(d / "TASK.json") if (d / "TASK.json").is_file() else {}
        rec["manifest"] = read_json(d / "run_manifest.json") if (d / "run_manifest.json").is_file() else {}
        rec["log"] = read_json(d / "run_log.json") if (d / "run_log.json").is_file() else {}
        rec["summary"] = read_json(d / "summary.json") if (d / "summary.json").is_file() else {}
        preds = read_json(d / "val_predictions.json") if (d / "val_predictions.json").is_file() else []
        rec["preds"] = preds
        rec["metrics"] = recompute(preds, ctx) if preds else None
        out.append(rec)
    return out, ctx


def model_params(batch: str) -> Dict[str, int]:
    """n_params from run logs (same across seeds)."""
    out = {}
    for b in BATCHES:
        for r in discover_runs(BATCHES[b]["dir"]):
            try:
                p = read_json(r["dir"] / "run_log.json").get("n_params")
                if p:
                    out.setdefault(r["model"], int(p))
            except Exception:  # noqa: BLE001
                pass
    return out

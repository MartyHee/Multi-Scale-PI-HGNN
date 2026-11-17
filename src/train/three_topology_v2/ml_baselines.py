"""Classical ML baselines on fixed-dimension node / beam samples (corrected).

Fair-comparison constraints:
  * node sample: ``mesh_node.x`` 15 (normalised) -> y_disp 6
  * beam sample: ``beam_element.x`` 15 + mean of its two endpoint node features
    15 = 30 -> y_force 12  (same information as the MLP force head)
  * inputs never derived from response labels; no topology/design/sample-id
    shortcuts; per-graph evaluation (graph-equal selection metric).
  * HistGradientBoosting runs with ``early_stopping=False`` (no random row-level
    internal validation split); its iteration budget is fixed a priori.
  * predictions are exported per graph with entity mapping / component order /
    units / stats source; serialization round-trip compares node AND beam
    outputs (not just finiteness).
  * Ridge is deterministic -> the entry runs it once (seed repetition unused).

Timing is split into: data_prep / fit / predict / inverse / export.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .metrics import MetricsAccumulator, export_records


def _graph_arrays(data):
    node_x = data["mesh_node"].x.numpy().astype(np.float64)
    node_y = data["mesh_node"].y_disp.numpy().astype(np.float64)
    beam_x = data["beam_element"].x.numpy().astype(np.float64)
    beam_y = data["beam_element"].y_force.numpy().astype(np.float64)
    ei = data["mesh_node", "belongs_to_beam", "beam_element"].edge_index.numpy()
    src, dst = ei[0], ei[1]
    n_beams = beam_x.shape[0]
    sums = np.zeros((n_beams, node_x.shape[1]), dtype=np.float64)
    cnts = np.zeros(n_beams, dtype=np.float64)
    np.add.at(sums, dst, node_x[src]); np.add.at(cnts, dst, 1.0)
    beam_endpoint_mean = sums / np.maximum(cnts, 1.0)[:, None]
    beam_in = np.concatenate([beam_x, beam_endpoint_mean], axis=1)
    return node_x, node_y, beam_in, beam_y


def _collect(dataset, sample_ids):
    Xn, Yn, Xb, Yb = [], [], [], []
    node_slices, beam_slices = [], []
    topo_per_sample, node_counts, beam_counts = [], [], []
    gn, gb = [], []
    n_off = 0; b_off = 0
    for gi, sid in enumerate(sample_ids):
        data = dataset[sid]
        node_x, node_y, beam_in, beam_y = _graph_arrays(data)
        topo = int(data.topology_code.item())
        Xn.append(node_x); Yn.append(node_y); Xb.append(beam_in); Yb.append(beam_y)
        nnod, nbeam = node_x.shape[0], beam_in.shape[0]
        node_slices.append((n_off, n_off + nnod)); beam_slices.append((b_off, b_off + nbeam))
        n_off += nnod; b_off += nbeam
        node_counts.append(nnod); beam_counts.append(nbeam)
        topo_per_sample.append(topo)
        gn.append(np.full(nnod, gi)); gb.append(np.full(nbeam, gi))
    cat = lambda xs, d: np.concatenate(xs, axis=0) if xs else np.zeros((0, d))
    return {
        "Xn": cat(Xn, 15), "Yn": cat(Yn, 6), "Xb": cat(Xb, 30), "Yb": cat(Yb, 12),
        "gn": np.concatenate(gn) if gn else np.zeros(0),
        "gb": np.concatenate(gb) if gb else np.zeros(0),
        "topo": np.asarray(topo_per_sample),
        "node_slices": node_slices, "beam_slices": beam_slices,
        "node_counts": node_counts, "beam_counts": beam_counts,
    }


def _fit(name: str, cfg: Dict[str, Any], Xtr, Ytr):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=float(cfg.get("alpha", 1.0)))
        model.fit(Xtr, Ytr)
        return model
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        models = []
        for c in range(Ytr.shape[1]):
            m = HistGradientBoostingRegressor(
                max_iter=int(cfg.get("max_iter", 200)),
                learning_rate=float(cfg.get("learning_rate", 0.1)),
                max_depth=cfg.get("max_depth"),
                early_stopping=False,          # no random row-level internal validation
                random_state=int(cfg.get("seed", 0)),
            )
            m.fit(Xtr, Ytr[:, c])
            models.append(m)
        return models
    raise ValueError(name)


def _predict(name, model, X):
    if name == "ridge":
        return model.predict(X)
    return np.stack([m.predict(X) for m in model], axis=1)


def run_ml_baseline(name: str, cfg: Dict[str, Any], train_ds, val_ds, stats,
                    disp_inv, force_inv, seed: int, out_dir: Path) -> Dict:
    t0 = time.time()
    tr = _collect(train_ds, train_ds.sample_ids)
    va = _collect(val_ds, val_ds.sample_ids)
    t1 = time.time()

    node_model = _fit(name, cfg, tr["Xn"], tr["Yn"])
    beam_model = _fit(name, cfg, tr["Xb"], tr["Yb"])
    t2 = time.time()

    node_pred_n = _predict(name, node_model, va["Xn"])
    beam_pred_n = _predict(name, beam_model, va["Xb"])
    t_pred = time.time()

    disp_phys = disp_inv(torch.as_tensor(node_pred_n)).numpy()
    force_phys = force_inv(torch.as_tensor(beam_pred_n)).numpy()
    t3 = time.time()

    # physical targets
    disp_tgt = disp_inv(torch.as_tensor(va["Yn"])).numpy()
    force_tgt = force_inv(torch.as_tensor(va["Yb"])).numpy()

    # unified evaluation (graph-equal selection metric shared with gradient engine)
    acc = MetricsAccumulator()
    acc.update(disp_phys, disp_tgt, va["gn"], va["topo"][va["gn"]],
               force_phys, force_tgt, va["gb"], va["topo"][va["gb"]])
    std = {k: [float(c["effective_std"]) for c in stats["columns"][k]] for k in ("y_disp", "y_force")}
    metrics = acc.finalize(std)

    # per-graph export
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_disp_per = [disp_phys[a:b] for a, b in va["node_slices"]]
    pred_force_per = [force_phys[a:b] for a, b in va["beam_slices"]]
    records = export_records(val_ds.sample_ids, va["node_counts"], va["beam_counts"],
                             pred_disp_per, pred_force_per, stats)
    (out_dir / "val_predictions.json").write_text(
        __import__("json").dumps(records, ensure_ascii=False), encoding="utf-8")

    # serialization round-trip: compare node AND beam predictions before/after
    model_path = out_dir / f"ml_{name}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({"node": node_model, "beam": beam_model, "cfg": cfg}, fh)
    with model_path.open("rb") as fh:
        loaded = pickle.load(fh)
    k = min(5, va["Xn"].shape[0])
    reload_node = _predict(name, loaded["node"], va["Xn"][:k])
    reload_beam = _predict(name, loaded["beam"], va["Xb"][:k])
    node_roundtrip = bool(np.allclose(reload_node, node_pred_n[:k], rtol=1e-12, atol=1e-12))
    beam_roundtrip = bool(np.allclose(reload_beam, beam_pred_n[:k], rtol=1e-12, atol=1e-12))
    t4 = time.time()

    log = {
        "model": name, "config": cfg, "seed": seed,
        "deterministic": name == "ridge",
        "n_train_graphs": len(train_ds), "n_val_graphs": len(val_ds),
        "n_train_nodes": int(tr["Xn"].shape[0]), "n_train_beams": int(tr["Xb"].shape[0]),
        "n_val_nodes": int(va["Xn"].shape[0]), "n_val_beams": int(va["Xb"].shape[0]),
        "hist_gb_early_stopping": False if name == "hist_gb" else None,
        "hist_gb_iteration_budget": cfg.get("max_iter") if name == "hist_gb" else None,
        "timing": {"data_prep_sec": t1 - t0, "fit_sec": t2 - t1,
                   "predict_sec": t_pred - t2, "inverse_sec": t3 - t_pred,
                   "export_sec": t4 - t3, "total_sec": t4 - t0},
        "serialization_reload_ok": node_roundtrip and beam_roundtrip,
        "roundtrip_node_equal": node_roundtrip, "roundtrip_beam_equal": beam_roundtrip,
        "model_path": str(model_path),
        "selection_metric": metrics.get("selection_metric"),
        "metrics": metrics,
    }
    log["pass"] = {
        "fit_ok": bool(np.isfinite(node_pred_n).all() and np.isfinite(beam_pred_n).all()),
        "predict_finite": bool(np.isfinite(disp_phys).all() and np.isfinite(force_phys).all()),
        "reload_node_equal": bool(node_roundtrip), "reload_beam_equal": bool(beam_roundtrip),
        "prediction_exported": bool((out_dir / "val_predictions.json").is_file()),
        "selection_finite": bool(metrics.get("selection_metric") is not None
                                 and np.isfinite(metrics["selection_metric"])),
    }
    log["pass"]["all"] = all(log["pass"].values())
    (out_dir / "run_log.json").write_text(__import__("json").dumps(log, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return log

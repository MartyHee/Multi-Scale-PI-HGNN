"""Physical-unit metrics + selection metric for three-topology v2 baselines.

Definitions (explicit, no mixed units):
  * per-component MAE / RMSE in physical units (m / rad / N / N*m)
  * ``norm_MAE_c = MAE_c / effective_std_train_c``  (zero-variance -> std=1)
  * ``selection_metric`` is **task-equal, component-equal, graph-equal**
    normalised MSE — NOT a flat average over all 18 components:
      for each task t in {disp, force}, component c, graph g:
        mse_tgc = mean_rows((pred_phys-target_phys)^2)
      selection_t = mean_c mean_g ( mse_tgc / effective_std_train_tc^2 )
      selection   = 0.5 * (selection_disp + selection_force)   # two tasks equal
    This is independent of evaluation batch size and of the last partial batch,
    and does not let large graphs dominate.
  * ``entity_weighted_norm_mse`` = mean_c ( overall_mse_c / std_c^2 )  (reported
    separately; entity weighted)
  * near-zero: if mean(|y_c|) <= near_zero_tol -> rel_undefined_near_zero=True
    (absolute MAE/RMSE still reported; R^2 is N/A for constant components).

This module is shared by gradient and classical-ML models so the evaluation
interface is identical.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

DISP_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz"]
DISP_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
FORCE_NAMES = ["I_Fx", "I_Fy", "I_Fz", "I_Mx", "I_My", "I_Mz",
               "J_Fx", "J_Fy", "J_Fz", "J_Mx", "J_My", "J_Mz"]
FORCE_UNITS = ["N", "N", "N", "N*m", "N*m", "N*m", "N", "N", "N", "N*m", "N*m", "N*m"]


class MetricsAccumulator:
    """Accumulate per-row errors grouped by component, graph and topology."""

    def __init__(self):
        self._disp = []
        self._force = []

    def update(self, pred_disp, target_disp, graph_disp, topo_disp,
               pred_force, target_force, graph_force, topo_force):
        self._disp.append({
            "pred": np.asarray(pred_disp, dtype=np.float64),
            "target": np.asarray(target_disp, dtype=np.float64),
            "graph": np.asarray(graph_disp),
            "topo": np.asarray(topo_disp),
        })
        self._force.append({
            "pred": np.asarray(pred_force, dtype=np.float64),
            "target": np.asarray(target_force, dtype=np.float64),
            "graph": np.asarray(graph_force),
            "topo": np.asarray(topo_force),
        })

    def _stack(self, kind: str):
        recs = self._disp if kind == "disp" else self._force
        if not recs:
            return None
        return {
            "pred": np.concatenate([r["pred"] for r in recs], axis=0),
            "target": np.concatenate([r["target"] for r in recs], axis=0),
            "graph": np.concatenate([r["graph"] for r in recs], axis=0),
            "topo": np.concatenate([r["topo"] for r in recs], axis=0),
        }

    def finalize(self, train_std: Dict[str, Sequence[float]],
                 near_zero_tol: float = 1e-9) -> Dict:
        out = {}
        for kind, names, units in (("disp", DISP_NAMES, DISP_UNITS),
                                   ("force", FORCE_NAMES, FORCE_UNITS)):
            st = self._stack(kind)
            if st is None:
                out[kind] = {"components": [], "n": 0, "selection_component": None}
                continue
            pred, tgt = st["pred"], st["target"]
            err = pred - tgt
            n_rows, dim = err.shape
            abs_sum = np.abs(err).sum(axis=0)
            sq_sum = (err ** 2).sum(axis=0)
            mae = abs_sum / max(n_rows, 1)
            rmse = np.sqrt(sq_sum / max(n_rows, 1))
            std_key = "y_disp" if kind == "disp" else "y_force"
            std = np.asarray(train_std.get(kind, train_std.get(std_key)), dtype=np.float64)
            norm_mae = mae / np.where(std > 0, std, np.nan)
            mean_abs_target = np.abs(tgt).mean(axis=0)

            comps = []
            for c in range(dim):
                nz = bool(mean_abs_target[c] <= near_zero_tol)
                comps.append({
                    "component": names[c], "unit": units[c],
                    "mae": float(mae[c]), "rmse": float(rmse[c]),
                    "norm_mae_by_train_std": None if not np.isfinite(norm_mae[c]) else float(norm_mae[c]),
                    "mean_abs_target": float(mean_abs_target[c]),
                    "rel_undefined_near_zero": nz,
                })

            # per-graph MAE and per-graph per-component MSE (for selection metric)
            graphs = np.unique(st["graph"])
            g_mae = {c: [] for c in range(dim)}
            g_mse = np.full((len(graphs), dim), np.nan)
            for gi, g in enumerate(graphs):
                m = st["graph"] == g
                if not m.any():
                    continue
                ge = err[m]
                gm = np.abs(ge).mean(axis=0)
                for c in range(dim):
                    g_mae[c].append(float(gm[c]))
                g_mse[gi] = (ge ** 2).mean(axis=0)
            graph_equal = {names[c]: {
                "mean": float(np.mean(g_mae[c])) if g_mae[c] else None,
                "std": float(np.std(g_mae[c])) if g_mae[c] else None,
                "n_graphs": len(g_mae[c]),
            } for c in range(dim)}

            # selection metric: graph-equal, component-equal normalised MSE
            std2 = np.where(std > 0, std ** 2, 1.0)
            per_graph_norm_mse = g_mse / std2[None, :]           # (G, dim)
            selection_component = float(np.nanmean(per_graph_norm_mse))
            entity_norm_mse = float(np.mean((sq_sum / max(n_rows, 1)) / std2))

            # per-topology overall
            per_topo = {}
            for t in np.unique(st["topo"]):
                m = st["topo"] == t
                if not m.any():
                    continue
                et = err[m]
                per_topo[str(int(t))] = {
                    "n": int(m.sum()),
                    "mae": [float(x) for x in np.abs(et).mean(axis=0)],
                    "rmse": [float(x) for x in np.sqrt((et ** 2).mean(axis=0))],
                }

            out[kind] = {
                "components": comps,
                "n": int(n_rows),
                "n_graphs": int(len(graphs)),
                "entity_weighted_mae": [float(x) for x in mae],
                "entity_weighted_rmse": [float(x) for x in rmse],
                "entity_weighted_norm_mse": entity_norm_mse,
                "graph_equal_mae": graph_equal,
                "selection_component": selection_component,
                "per_topology": per_topo,
            }

        # combined selection metric = mean of disp/force component selections
        if out.get("disp", {}).get("selection_component") is not None and \
                out.get("force", {}).get("selection_component") is not None:
            out["selection_metric"] = 0.5 * (out["disp"]["selection_component"]
                                             + out["force"]["selection_component"])
        else:
            out["selection_metric"] = None
        return out


def export_records(sample_ids, node_counts, beam_counts, pred_disp, pred_force,
                   stats: Dict) -> List[Dict]:
    """Per-sample prediction records with entity mapping + units + stats source."""
    recs = []
    for i, sid in enumerate(sample_ids):
        recs.append({
            "sample_id": sid,
            "node_count": int(node_counts[i]),
            "beam_count": int(beam_counts[i]),
            "disp_columns": DISP_NAMES, "disp_units": DISP_UNITS,
            "force_columns": FORCE_NAMES, "force_units": FORCE_UNITS,
            "stats_source": stats.get("source"),
            "stats_fold_id": stats.get("fold_id"),
            "stats_topology": stats.get("topology_token"),
            "stats_file_sha256": stats.get("_file_sha256"),
            "pred_disp": np.asarray(pred_disp[i], dtype=np.float64).tolist(),
            "pred_force": np.asarray(pred_force[i], dtype=np.float64).tolist(),
        })
    return recs

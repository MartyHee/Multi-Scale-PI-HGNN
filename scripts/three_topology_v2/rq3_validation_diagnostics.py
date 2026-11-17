#!/usr/bin/env python
"""RQ3 descriptive diagnostics on the FIRST-BATCH validation predictions.

Scope: topology1 / cross_design_topology1 / cd_fold_1, the same 24 validation
graphs used by the first batch.  Only existing predictions and train/val graphs
are read; no inference, no training, no outer test (never constructed).

Analyses
  1. hop distance to supports (beam-endpoint adjacency; sensitivity: + structural
     links) vs node displacement error (translation and rotation kept separate)
     and vs beam force error (beam hop = min endpoint hop);
  2. local displacement variation (neighbour difference / geometric distance,
     translation m/m and rotation rad/m separately) vs error, with per-graph
     Spearman and within-graph variation quintiles;
  3. stratification by load field (template_id), design (design_id) and response
     magnitude, with entity/graph counts.

Method notes (also written to the JSON/CSV):
  * node adjacency is undirected; beam membership provides the two endpoints;
    structural_link is a directed-as-stored interaction edge, made undirected for
    the sensitivity graph. Zero-length links are counted; they are used for
    connectivity but skipped when dividing by distance.
  * supports are nodes with ANY of Dx/Dy/Dz/Rx/Ry/Rz fix flag == 1.
  * entity alignment is by sample_id and the graph's own raw node/beam IDs; the
    prediction arrays are in the graph's stored order and are verified by counts.
  * per-graph statistics are averaged graph-equal; entities of one mesh are not
    treated as independent samples across graphs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.data.three_topology_v2 import graph_io, index, paths  # noqa: E402
from src.train.three_topology_v2.metrics import DISP_NAMES, FORCE_NAMES  # noqa: E402

BATCH = "fb_20261117_01"
SPLIT = "cross_design_topology1"
FOLD = "cd_fold_1"
ARCH = MODEL_ROOT / "remote_artifacts" / f"first_batch_{BATCH}_review"
RUNS = [
    ("t1_mlp", "mlp_seed20261117", "mlp", 20261117),
    ("t1_mlp", "mlp_seed20261118", "mlp", 20261118),
    ("t1_mlp", "mlp_seed20261119", "mlp", 20261119),
    ("t2_hgt", "hgt_global_seed20261117", "hgt_global", 20261117),
    ("t3_hgt", "hgt_global_seed20261118", "hgt_global", 20261118),
    ("t4_hgt", "hgt_global_seed20261119", "hgt_global", 20261119),
    ("t1_histgb", "hist_gb", "hist_gb", None),
    ("t1_ridge", "ridge", "ridge", None),
]
PRIMARY = {"mlp", "hgt_global"}
HOPS = [0, 1, 2, 3, 4, 5]  # 5 means >=5


def bfs(adj, sources, n):
    dist = np.full(n, -1, dtype=np.int64)
    from collections import deque
    q = deque()
    for s in sources:
        if dist[s] == -1:
            dist[s] = 0; q.append(int(s))
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1; q.append(int(v))
    return dist


def hop_bin(d):
    return np.where(d >= 5, 5, d)


def graph_adjacency(beam_edges, n):
    adj = [set() for _ in range(n)]
    for a, b in beam_edges:
        if a != b:
            adj[a].add(b); adj[b].add(a)
    return adj


def local_variation(coords, adj, y, comps):
    n = coords.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        vals = []
        for j in adj[i]:
            d = float(np.linalg.norm(coords[i, :3] - coords[j, :3]))
            if d <= 1e-12:
                continue
            vals.append(float(np.mean(np.abs(y[i, comps] - y[j, comps])) / d))
        if vals:
            out[i] = float(np.mean(vals))
    return out


def decile_bins(values):
    n = values.size
    bins = np.full(n, 0, dtype=np.int64)
    idx = np.where(np.isfinite(values))[0]
    if idx.size == 0:
        return bins
    order = idx[np.argsort(values[idx], kind="stable")]
    for b, part in enumerate(np.array_split(order, 10), start=1):
        bins[part] = b
    return bins


def quintile_bins(values):
    """Equal-count bins (1..5) among finite values; ties keep a deterministic order."""
    n = values.size
    bins = np.full(n, 0, dtype=np.int64)
    idx = np.where(np.isfinite(values))[0]
    if idx.size == 0:
        return bins
    order = idx[np.argsort(values[idx], kind="stable")]
    for b, part in enumerate(np.array_split(order, 5), start=1):
        bins[part] = b
    return bins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(ARCH))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "second_batch_preparation_20261118"))
    a = ap.parse_args()
    arch = Path(a.archive_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    idx = index.load_index(strict=True)
    split = json.loads((paths.SPLIT_DIR / f"split_{SPLIT}.json").read_text(encoding="utf-8"))
    fold = next(f for f in split["folds"] if f["fold_id"] == FOLD)
    val_ids = sorted(fold["validation_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    assert not (set(val_ids) & test_ids)
    stats = json.loads((paths.STATS_DIR / SPLIT / f"{FOLD}.json").read_text(encoding="utf-8"))
    std_d = np.array([c["effective_std"] for c in stats["columns"]["y_disp"]], dtype=np.float64)
    std_f = np.array([c["effective_std"] for c in stats["columns"]["y_force"]], dtype=np.float64)

    meta = {}
    for sid in val_ids:
        row = idx["row_by_id"][sid]
        g = graph_io.load_graph(row, dtype=torch.float64, verify_hash=True)
        x = g["mesh_node"].x.numpy()
        be = g["mesh_node", "belongs_to_beam", "beam_element"].edge_index.numpy()
        sl = g["mesh_node", "structural_link", "mesh_node"].edge_index.numpy()
        n = x.shape[0]
        beam_edges = []
        for b in range(g["beam_element"].x.shape[0]):
            e = be[0][be[1] == b]
            beam_edges.append((int(e[0]), int(e[1])))
        adj_beam = graph_adjacency(beam_edges, n)
        adj_full = [set(s) for s in adj_beam]
        sl_len = []
        for u, v in zip(sl[0], sl[1]):
            u, v = int(u), int(v)
            if u != v:
                adj_full[u].add(v); adj_full[v].add(u)
            sl_len.append(float(np.linalg.norm(x[u, :3] - x[v, :3])))
        sl_len = np.asarray(sl_len, dtype=np.float64)
        sup_mask = (x[:, 9:15] > 0.5).any(axis=1)
        sup = np.where(sup_mask)[0]
        flags = defaultdict(int)
        for s in sup:
            flags["".join("1" if v > 0.5 else "0" for v in x[s, 9:15])] += 1
        d_beam = bfs(adj_beam, sup, n)
        d_full = bfs(adj_full, sup, n)
        meta[sid] = {
            "n_nodes": n, "n_beams": len(beam_edges), "n_links": int(sl.shape[1]),
            "n_supports": int(sup.size), "support_flags": dict(flags),
            "support_nodes": [int(v) for v in sup],
            "beam_edges": beam_edges, "adj_beam": adj_beam, "adj_full": adj_full,
            "d_beam": d_beam, "d_full": d_full,
            "sl_zero_len": int((sl_len <= 1e-12).sum()) if sl_len.size else 0,
            "sl_min_len": float(sl_len.min()) if sl_len.size else None,
            "unreachable_beam": int((d_beam < 0).sum()), "unreachable_full": int((d_full < 0).sum()),
            "template_id": row["template_id"], "design_id": row["design_id"],
            "x": x, "td": g["mesh_node"].y_disp.numpy(),
            "tf": g["beam_element"].y_force.numpy(),
        }
        # component check + whether every beam component has a support
        parent = list(range(n))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for u, v in beam_edges:
            ru, rv = find(u), find(v)
            if ru != rv: parent[rv] = ru
        comps = defaultdict(list)
        for u in range(n): comps[find(u)].append(u)
        meta[sid]["beam_components"] = len(comps)
        meta[sid]["components_with_support"] = len({find(int(s)) for s in sup})

    def graph_stats(preds_by_id, sid):
        m = meta[sid]
        pd = np.asarray(preds_by_id[sid]["pred_disp"], dtype=np.float64)
        pf = np.asarray(preds_by_id[sid]["pred_force"], dtype=np.float64)
        n = m["n_nodes"]; b = len(m["beam_edges"])
        if pd.shape != (n, 6) or pf.shape != (b, 12):
            raise RuntimeError(f"{sid}: prediction shape {pd.shape}/{pf.shape} != {(n,6)}/{(b,12)}")
        td = m["td"]; tf = m["tf"]
        ed = np.abs(pd - td) / std_d[None, :]
        ef = np.abs(pf - tf) / std_f[None, :]
        node_trans = ed[:, :3].mean(axis=1)
        node_rot = ed[:, 3:].mean(axis=1)
        beam_err = ef.mean(axis=1)
        beam_hop = np.minimum(m["d_beam"][[e[0] for e in m["beam_edges"]]],
                              m["d_beam"][[e[1] for e in m["beam_edges"]]])
        # selection sanity (same definition as the engine)
        gd = ((pd - td) ** 2).mean(axis=0); gf = ((pf - tf) ** 2).mean(axis=0)
        sel = 0.5 * (float(np.mean(gd / std_d ** 2)) + float(np.mean(gf / std_f ** 2)))
        var_t_beam = local_variation(m["x"], m["adj_beam"], td, [0, 1, 2])
        var_r_beam = local_variation(m["x"], m["adj_beam"], td, [3, 4, 5])
        var_t_full = local_variation(m["x"], m["adj_full"], td, [0, 1, 2])
        var_r_full = local_variation(m["x"], m["adj_full"], td, [3, 4, 5])
        return {"node_trans": node_trans, "node_rot": node_rot, "beam_err": beam_err,
                "hop": m["d_beam"], "mag": np.mean(np.abs(td) / std_d[None, :], axis=1),
                "beam_hop": beam_hop, "sel": sel, "var_t_beam": var_t_beam,
                "var_r_beam": var_r_beam, "var_t_full": var_t_full, "var_r_full": var_r_full,
                "td": td, "tf": tf}

    hop_rows, var_rows, corr_rows, group_rows, mag_rows = [], [], [], [], []
    hop_mag_rows, hop_mag_corr = [], []
    summary = {
        "schema": "qt-rq3-validation-diagnostics-v1", "date": "2026-11-18",
        "batch_id": BATCH, "split_id": SPLIT, "fold_id": FOLD, "val_n": len(val_ids),
        "test_sealed": True, "method": {}, "graph_meta": {}, "runs": {},
        "caveats": [],
    }
    summary["method"] = {
        "node_adjacency": "undirected beam-endpoint adjacency (each beam connects its I/J mesh nodes)",
        "sensitivity_adjacency": "beam adjacency + structural_link made undirected",
        "support_definition": "mesh node with any Dx/Dy/Dz/Rx/Ry/Rz fix flag == 1",
        "hop_bins": "0,1,2,3,4,>=5; BFS multi-source from all supports",
        "beam_hop": "min hop of the beam's two end nodes",
        "node_error": "mean over components of |pred-target| / train effective_std; translation (0:3) and rotation (3:6) separate",
        "beam_error": "mean over 12 force components of |pred-target| / train effective_std",
        "local_variation_translation": "mean over beam neighbours of mean_{dx,dy,dz}|u_i-u_j| / geometric distance (m/m)",
        "local_variation_rotation": "mean over beam neighbours of mean_{rx,ry,rz}|u_i-u_j| / geometric distance (rad/m)",
        "within_graph_bins": "equal-count quintiles of the local-variation value inside each graph",
        "aggregation": "within-graph mean, then equal weight per graph; entity counts reported",
        "alignment": "sample_id + graph raw IDs; arrays verified by node/beam counts",
    }
    summary["graph_meta"] = {
        sid: {k: v for k, v in meta[sid].items()
              if k not in ("adj_beam", "adj_full", "x", "beam_edges", "support_nodes", "td", "tf", "d_beam", "d_full")}
        for sid in val_ids}
    summary["totals"] = {
        "graphs": len(val_ids),
        "nodes": int(sum(meta[s]["n_nodes"] for s in val_ids)),
        "beams": int(sum(meta[s]["n_beams"] for s in val_ids)),
        "links": int(sum(meta[s]["n_links"] for s in val_ids)),
        "zero_length_links": int(sum(meta[s]["sl_zero_len"] for s in val_ids)),
        "unreachable_nodes_beam": int(sum(meta[s]["unreachable_beam"] for s in val_ids)),
        "unreachable_nodes_full": int(sum(meta[s]["unreachable_full"] for s in val_ids)),
        "graphs_with_all_components_supported": int(sum(
            1 for s in val_ids if meta[s]["components_with_support"] == meta[s]["beam_components"])),
        "supports": int(sum(meta[s]["n_supports"] for s in val_ids)),
    }

    for worker, run, model, seed in RUNS:
        d = arch / "batch" / BATCH / worker / run
        if not (d / "val_predictions.json").is_file():
            continue
        recs = json.loads((d / "val_predictions.json").read_text(encoding="utf-8"))
        by_id = {r["sample_id"]: r for r in recs}
        if set(by_id) != set(val_ids):
            summary["runs"][f"{worker}/{run}"] = {"error": "prediction ids != val ids"}
            continue
        run_tag = f"{worker}/{run}"
        per_graph = {sid: graph_stats(by_id, sid) for sid in val_ids}
        run_log = json.loads((d / "run_log.json").read_text(encoding="utf-8"))
        run_sel = (run_log.get("final_val", {}).get("selection_metric")
                   if "final_val" in run_log else run_log.get("selection_metric"))
        sel_recomputed = float(np.mean([per_graph[s]["sel"] for s in val_ids]))
        is_primary = model in PRIMARY
        summary["runs"][run_tag] = {
            "model": model, "seed": seed, "primary": is_primary,
            "selection_run_log": run_sel, "selection_recomputed": sel_recomputed,
            "selection_match": (run_sel is not None and abs(sel_recomputed - run_sel) < 1e-6),
        }

        # ---- hop bins ----
        for sid in val_ids:
            m = meta[sid]; pg = per_graph[sid]
            for hop in HOPS:
                for graph_name, dkey in (("beam", "d_beam"), ("full", "d_full")):
                    dm = hop_bin(m[dkey])
                    mask = dm == hop
                    if not mask.any():
                        continue
                    nh = int(mask.sum())
                    bt = mask[[e[0] for e in m["beam_edges"]]] | mask[[e[1] for e in m["beam_edges"]]]
                    hop_rows.append({
                        "run": run_tag, "model": model, "seed": seed, "graph_adj": graph_name,
                        "sample_id": sid, "hop": hop, "n_nodes": nh,
                        "n_beams": int(bt.sum()),
                        "node_trans_norm_mae": float(pg["node_trans"][mask].mean()),
                        "node_rot_norm_mae": float(pg["node_rot"][mask].mean()),
                        "beam_force_norm_mae": float(pg["beam_err"][bt].mean()) if bt.any() else None,
                    })

        # ---- local variation quintiles ----
        for sid in val_ids:
            m = meta[sid]; pg = per_graph[sid]
            for adj_name in ("beam", "full"):
                for task, err, var in (("translation", pg["node_trans"], pg[f"var_t_{adj_name}"]),
                                       ("rotation", pg["node_rot"], pg[f"var_r_{adj_name}"])):
                    bins = quintile_bins(var)
                    for b in range(1, 6):
                        mask = bins == b
                        if not mask.any():
                            continue
                        finite = np.isfinite(var[mask])
                        var_rows.append({
                            "run": run_tag, "model": model, "seed": seed, "graph_adj": adj_name,
                            "sample_id": sid, "task": task, "quintile": b,
                            "n_nodes": int(mask.sum()),
                            "var_mean": float(var[mask][finite].mean()) if finite.any() else None,
                            "var_min": float(var[mask][finite].min()) if finite.any() else None,
                            "var_max": float(var[mask][finite].max()) if finite.any() else None,
                            "err_norm_mae": float(err[mask].mean()),
                        })
                    # per-graph Spearman (descriptive)
                    ok = np.isfinite(var) & np.isfinite(err)
                    rho = p = None
                    if ok.sum() >= 10 and np.std(var[ok]) > 0 and np.std(err[ok]) > 0:
                        rho, p = spearmanr(var[ok], err[ok])
                    corr_rows.append({
                        "run": run_tag, "model": model, "seed": seed, "sample_id": sid,
                        "graph_adj": adj_name, "task": task, "n_nodes": int(ok.sum()),
                        "spearman_rho": None if rho is None else float(rho),
                        "p_value": None if p is None else float(p),
                    })

        # ---- hop vs error at matched response magnitude (within-graph deciles) ----
        for sid in val_ids:
            pg = per_graph[sid]
            bins = decile_bins(pg["mag"])
            for b in range(1, 11):
                mask = bins == b
                if not mask.any():
                    continue
                for task, err in (("translation", pg["node_trans"]), ("rotation", pg["node_rot"])):
                    hop_mag_rows.append({
                        "run": run_tag, "model": model, "seed": seed, "sample_id": sid,
                        "task": task, "magnitude_decile": b, "n_nodes": int(mask.sum()),
                        "mean_hop": float(pg["hop"][mask].mean()),
                        "err_norm_mae": float(err[mask].mean()),
                        "mean_abs_target_norm": float(pg["mag"][mask].mean()),
                    })
                    ok = mask & np.isfinite(err)
                    rho = p = None
                    if ok.sum() >= 10 and np.std(pg["hop"][ok]) > 0 and np.std(err[ok]) > 0:
                        rho, p = spearmanr(pg["hop"][ok], err[ok])
                    hop_mag_corr.append({
                        "run": run_tag, "model": model, "seed": seed, "sample_id": sid,
                        "task": task, "magnitude_decile": b, "n_nodes": int(ok.sum()),
                        "spearman_rho": None if rho is None else float(rho),
                        "p_value": None if p is None else float(p),
                    })

        # ---- load field / design grouping ----
        for group_by in ("template_id", "design_id"):
            groups = defaultdict(list)
            for sid in val_ids:
                groups[meta[sid][group_by]].append(sid)
            for g, sids in sorted(groups.items()):
                group_rows.append({
                    "run": run_tag, "model": model, "seed": seed, "group_by": group_by,
                    "group": g, "n_graphs": len(sids),
                    "selection_graph_equal_mean": float(np.mean([per_graph[s]["sel"] for s in sids])),
                    "node_trans_norm_mae": float(np.mean([per_graph[s]["node_trans"].mean() for s in sids])),
                    "node_rot_norm_mae": float(np.mean([per_graph[s]["node_rot"].mean() for s in sids])),
                    "beam_force_norm_mae": float(np.mean([per_graph[s]["beam_err"].mean() for s in sids])),
                })

        # ---- response magnitude deciles (key + all components) ----
        for task, names, std in (("disp", DISP_NAMES, std_d), ("force", FORCE_NAMES, std_f)):
            truths = np.concatenate([per_graph[s]["td" if task == "disp" else "tf"] for s in val_ids])
            preds = np.concatenate([np.asarray(by_id[s]["pred_disp" if task == "disp" else "pred_force"],
                                               dtype=np.float64) for s in val_ids])
            err = np.abs(preds - truths)
            for c, name in enumerate(names):
                y = np.abs(truths[:, c]); e = err[:, c]
                qs = np.quantile(y, np.linspace(0, 1, 11))
                for b in range(10):
                    lo, hi = qs[b], qs[b + 1]
                    mask = (y >= lo) & (y <= hi) if b == 9 else (y >= lo) & (y < hi)
                    if not mask.any():
                        continue
                    mag_rows.append({
                        "run": run_tag, "model": model, "seed": seed, "task": task,
                        "component": name, "decile": b + 1, "n": int(mask.sum()),
                        "mean_abs_true": float(y[mask].mean()), "mae": float(e[mask].mean()),
                        "norm_mae": float(e[mask].mean() / std[c]),
                    })

    # ---- aggregate hop rows graph-equal ----
    def agg_hop(rows):
        groups = defaultdict(list)
        for r in rows:
            groups[(r["run"], r["model"], r["seed"], r["graph_adj"], r["hop"])].append(r)
        out = []
        for (run, model, seed, adj, hop), rs in sorted(groups.items()):
            out.append({
                "run": run, "model": model, "seed": seed, "graph_adj": adj, "hop": hop,
                "n_graphs": len(rs), "n_nodes": sum(r["n_nodes"] for r in rs),
                "n_beams": sum(r["n_beams"] for r in rs),
                "node_trans_norm_mae": float(np.mean([r["node_trans_norm_mae"] for r in rs])),
                "node_rot_norm_mae": float(np.mean([r["node_rot_norm_mae"] for r in rs])),
                "beam_force_norm_mae": float(np.mean([r["beam_force_norm_mae"] for r in rs
                                                      if r["beam_force_norm_mae"] is not None])),
            })
        return out

    def agg_var(rows):
        groups = defaultdict(list)
        for r in rows:
            groups[(r["run"], r["model"], r["seed"], r["graph_adj"], r["task"], r["quintile"])].append(r)
        out = []
        for (run, model, seed, adj, task, q), rs in sorted(groups.items()):
            out.append({
                "run": run, "model": model, "seed": seed, "graph_adj": adj, "task": task,
                "quintile": q, "n_graphs": len(rs), "n_nodes": sum(r["n_nodes"] for r in rs),
                "err_norm_mae": float(np.mean([r["err_norm_mae"] for r in rs])),
                "var_mean": float(np.mean([r["var_mean"] for r in rs if r["var_mean"] is not None])),
            })
        return out

    hop_agg = agg_hop(hop_rows)
    var_agg = agg_var(var_rows)

    def agg_hop_mag(rows):
        groups = defaultdict(list)
        for r in rows:
            groups[(r["run"], r["model"], r["seed"], r["task"], r["magnitude_decile"])].append(r)
        out = []
        for (run, model, seed, task, dec), rs in sorted(groups.items()):
            out.append({"run": run, "model": model, "seed": seed, "task": task,
                        "magnitude_decile": dec, "n_graphs": len(rs),
                        "n_nodes": sum(r["n_nodes"] for r in rs),
                        "mean_hop": float(np.mean([r["mean_hop"] for r in rs])),
                        "err_norm_mae": float(np.mean([r["err_norm_mae"] for r in rs]))})
        return out

    def agg_hop_mag_corr(rows):
        groups = defaultdict(list)
        for r in rows:
            if r["spearman_rho"] is not None:
                groups[(r["run"], r["model"], r["seed"], r["task"], r["magnitude_decile"])].append(r["spearman_rho"])
        out = []
        for (run, model, seed, task, dec), vals in sorted(groups.items()):
            out.append({"run": run, "model": model, "seed": seed, "task": task,
                        "magnitude_decile": dec, "n_graphs": len(vals),
                        "rho_median": float(np.median(vals)), "rho_min": float(np.min(vals)),
                        "rho_max": float(np.max(vals)),
                        "n_positive": int(sum(v > 0 for v in vals))})
        return out

    def write_csv(name, rows):
        if not rows:
            return
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    write_csv("rq3_hop_bins.csv", hop_agg)
    write_csv("rq3_hop_bins_per_graph.csv", hop_rows)
    write_csv("rq3_local_variation_bins.csv", var_agg)
    write_csv("rq3_local_variation_per_graph.csv", var_rows)
    write_csv("rq3_spearman.csv", corr_rows)
    write_csv("rq3_load_field_design.csv", group_rows)
    write_csv("rq3_magnitude_deciles.csv", mag_rows)
    write_csv("rq3_hop_magnitude.csv", agg_hop_mag(hop_mag_rows))
    write_csv("rq3_hop_magnitude_spearman.csv", agg_hop_mag_corr(hop_mag_corr))

    # Spearman summarised per run/task/adjacency (median across graphs)
    corr_agg = []
    groups = defaultdict(list)
    for r in corr_rows:
        groups[(r["run"], r["model"], r["seed"], r["graph_adj"], r["task"])].append(r["spearman_rho"])
    for (run, model, seed, adj, task), vals in sorted(groups.items()):
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        corr_agg.append({"run": run, "model": model, "seed": seed, "graph_adj": adj,
                         "task": task, "n_graphs": len(vals),
                         "rho_median": float(np.median(vals)),
                         "rho_min": float(np.min(vals)), "rho_max": float(np.max(vals)),
                         "n_positive": int(sum(v > 0 for v in vals))})
    write_csv("rq3_spearman_summary.csv", corr_agg)

    summary["tables"] = {
        "hop_bins": "rq3_hop_bins.csv", "local_variation": "rq3_local_variation_bins.csv",
        "spearman_summary": "rq3_spearman_summary.csv",
        "load_field_design": "rq3_load_field_design.csv",
        "magnitude_deciles": "rq3_magnitude_deciles.csv",
        "hop_magnitude": "rq3_hop_magnitude.csv",
        "hop_magnitude_spearman": "rq3_hop_magnitude_spearman.csv",
    }
    summary["caveats"] = [
        "Single topology (topology1) single fold (cd_fold_1), validation side only.",
        "Descriptive associations; hop distance / local variation are not causal mechanisms.",
        "Competing explanations not excluded: response magnitude and load-field structure co-vary with hop; "
        "support proximity may proxy for boundary-condition encoding; mesh discretisation is fixed per design; "
        "model capacity/feature encoding differ between MLP and HGT-global.",
        "Beam local axes are unconfirmed, so raw force directions are not compared across beams.",
        "Three seeds are repeated runs on the same split, not independent datasets.",
        "No graph-size/topology stratification is possible with one topology.",
    ]
    summary["competing_explanations_and_tests"] = [
        {"explanation": "error is driven by response magnitude, not distance",
         "distinguishing_test": "magnitude-stratified hop analysis / reweighted loss; compare bins at matched |y|"},
        {"explanation": "support proximity is a proxy for boundary-condition information already in node features",
         "distinguishing_test": "ablate support flags or replace with distance-to-support feature; controlled comparison"},
        {"explanation": "local variation reflects high-frequency content the model under-fits",
         "distinguishing_test": "train with spectral/high-pass loss or synthetic inputs with controlled gradient at fixed magnitude"},
        {"explanation": "HGT-global gains come from the global module rather than distance handling",
         "distinguishing_test": "plain HGT (second batch) under the same protocol over hop bins"},
    ]
    (out / "rq3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for model, color in (("mlp", "C0"), ("hgt_global", "C1")):
            for adj, ls in (("beam", "-"), ("full", "--")):
                rows = [r for r in hop_agg if r["model"] == model and r["graph_adj"] == adj]
                if not rows:
                    continue
                hops = sorted({r["hop"] for r in rows})
                for k, (key, lbl) in enumerate((("node_trans_norm_mae", "translation"),
                                                ("node_rot_norm_mae", "rotation"))):
                    ys = [np.mean([r[key] for r in rows if r["hop"] == h]) for h in hops]
                    ax[k].plot(hops, ys, ls, color=color, marker="o", ms=3,
                               label=f"{model} {adj}")
            ax[0].set_title("node translation error vs hop"); ax[1].set_title("node rotation error vs hop")
        for k in range(2):
            ax[k].set_xlabel("hop distance to support"); ax[k].set_ylabel("norm MAE")
            ax[k].set_yscale("log"); ax[k].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "rq3_hop_error.png", dpi=140); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for model, color in (("mlp", "C0"), ("hgt_global", "C1")):
            for adj, ls in (("beam", "-"), ("full", "--")):
                for k, task in enumerate(("translation", "rotation")):
                    rows = [r for r in var_agg if r["model"] == model and r["graph_adj"] == adj
                            and r["task"] == task]
                    if not rows:
                        continue
                    xs = sorted({r["quintile"] for r in rows})
                    ys = [np.mean([r["err_norm_mae"] for r in rows if r["quintile"] == q]) for q in xs]
                    ax[k].plot(xs, ys, ls, color=color, marker="o", ms=3, label=f"{model} {adj}")
                    ax[k].set_title(f"node {task} error vs local variation")
                    ax[k].set_xlabel("within-graph variation quintile")
                    ax[k].set_ylabel("norm MAE"); ax[k].set_yscale("log"); ax[k].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "rq3_local_variation_error.png", dpi=140); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        data = []
        labels = []
        for model in ("mlp", "hgt_global"):
            for task in ("translation", "rotation"):
                rows = [r for r in corr_agg if r["model"] == model and r["task"] == task
                        and r["graph_adj"] == "beam"]
                if rows:
                    data.append([r["rho_median"] for r in rows])
                    labels.append(f"{model}\n{task}")
        ax.boxplot(data, labels=labels)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("per-graph Spearman rho (median per seed)")
        ax.set_title("error vs local variation (beam adjacency)")
        fig.tight_layout(); fig.savefig(out / "rq3_spearman.png", dpi=140); plt.close(fig)
        summary["figures"] = ["rq3_hop_error.png", "rq3_local_variation_error.png", "rq3_spearman.png"]
    except Exception as exc:  # noqa: BLE001
        summary["plot_error"] = repr(exc)

    (out / "rq3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"totals": summary["totals"],
                      "runs": {k: {kk: v[kk] for kk in ("model", "seed", "selection_run_log",
                                                        "selection_recomputed", "selection_match")}
                               for k, v in summary["runs"].items() if "error" not in v}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

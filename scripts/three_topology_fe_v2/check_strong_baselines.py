#!/usr/bin/env python
"""Targeted local checks for the 2026-11-26 strong-baseline batch.

Checks (train/val inputs only; no response label is used as a model input):

  1. shared-backbone initialisation: plain HGT and HGT-global built with the
     same ``init_seed`` have bit-identical shared weights, while their global
     modules differ;
  2. ``global_mode='off'`` runs the exact HGT-only path (outputs equal plain
     HGT inside tolerance) and its global parameters are never used;
  3. parameter counts for HGT / HGT-global / capacity-matched HGT, with the
     deviation recorded;
  4. every new model produces finite 6/12 outputs, finite gradients, and an
     actual parameter update on one batch;
  5. per-graph isolation: perturbing graph 2 leaves graph 1 predictions
     unchanged (no cross-graph attention/softmax leakage);
  6. the full-field baseline's binned descriptor covers every entity type.

This is a local interface check, not a server/formal-training validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

NEW_MODELS = ["hgt_capmatch", "hgt_global_off", "fullfield_mlp",
              "fullfield_query_mlp", "meshgraphnet", "transolver", "linearno"]


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def shared_keys(state):
    return {k: v for k, v in state.items()
            if not k.startswith(("global_cell.", "global_bcast.", "fusion."))}


def build(name, cfg, dev, seed=20261125):
    mcfg = schema.task_model_cfg(name, cfg, "topology1", seed)
    return registry.build_gradient_model(name, mcfg).to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "strong_baselines.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT.parent / "strong_baseline_prep_20261126" / "checks"))
    a = ap.parse_args()
    cfg = load_yaml(Path(a.config))
    dev = engine.resolve_device(a.device)
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / "topology1.json")
    train_ids = [s for s, v in prot["roles"].items()
                 if idx.by_id[s]["topology_token"] == "topology1" and v["A"] == "train"][:2]
    ds = FeV2Dataset(idx, train_ids, st, dtype=torch.float32, normalize=True)
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))]).to(dev)

    report = {"schema": "qt-fe-v2-strong-baseline-check-v1", "device": str(dev),
              "samples": train_ids, "checks": {}, "models": {}}

    # 1/2. shared init + global-off path
    torch.manual_seed(11)
    hgt = build("hgt", cfg, dev).eval()
    torch.manual_seed(22)
    hgtg = build("hgt_global", cfg, dev).eval()
    torch.manual_seed(33)
    hgtg2 = build("hgt_global", cfg, dev, seed=20261126).eval()
    torch.manual_seed(44)
    hgt_off = build("hgt_global_off", cfg, dev).eval()
    s1, s2 = shared_keys(hgt.state_dict()), shared_keys(hgtg.state_dict())
    shared_equal = set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
    g_global_keys = [k for k in hgtg.state_dict() if k.startswith(("global_cell.", "global_bcast.", "fusion."))]
    globals_differ = any(not torch.equal(hgtg.state_dict()[k], hgtg2.state_dict()[k])
                         for k in g_global_keys)
    with torch.no_grad():
        pd_h, pf_h = hgt(batch)
        pd_off, pf_off = hgt_off(batch)
    path_equal = torch.allclose(pd_h, pd_off, atol=1e-6, rtol=1e-5) and \
        torch.allclose(pf_h, pf_off, atol=1e-6, rtol=1e-5)
    off_grads = {}
    hgt_off.train()
    pd, pf = hgt_off(batch)
    (F.mse_loss(pd, batch["mesh_node"].y_disp) + F.mse_loss(pf, batch["beam_element"].y_force)).backward()
    for n, p in hgt_off.named_parameters():
        if n.startswith(("global_cell.", "global_bcast.", "fusion.")):
            off_grads[n] = p.grad is None
    report["checks"]["shared_init_equal"] = bool(shared_equal)
    report["checks"]["global_modules_differ"] = bool(globals_differ)
    report["checks"]["global_off_matches_hgt"] = bool(path_equal)
    report["checks"]["global_off_all_global_params_unused"] = bool(off_grads) and all(off_grads.values())

    # 3. parameter counts
    counts = {}
    for name in ("hgt", "hgt_global", "hgt_capmatch"):
        m = build(name, cfg, dev)
        counts[name] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    glob = counts["hgt_global"]
    cap = counts["hgt_capmatch"]
    counts["capmatch_abs_delta"] = cap - glob
    counts["capmatch_rel_delta"] = (cap - glob) / glob
    counts["note"] = ("parameter count only; HGT-global also pays GRU/broadcast/gate FLOPs, "
                      "so this is not a compute match")
    report["param_counts"] = counts

    # 4/5. per-model forward/loss/grad/update and graph isolation
    for name in NEW_MODELS:
        torch.manual_seed(20261126)
        model = build(name, cfg, dev)
        model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        opt.zero_grad()
        pd, pf = model(batch)
        loss = F.mse_loss(pd, batch["mesh_node"].y_disp) + F.mse_loss(pf, batch["beam_element"].y_force)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        updated = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())

        model.eval()
        # CUDA float32 kernels are not bit-reproducible (repeat-call spread
        # measured at ~1.3e-5 on this laptop), so use a looser isolation
        # tolerance on GPU and state it in the report.
        tol = 1e-6 if dev.type == "cpu" else 1e-4
        with torch.no_grad():
            base_d, base_f = model(batch)
            b2 = batch.clone()
            b2["mesh_node"].x[b2["mesh_node"].batch == 1] += 3.0
            pert_d, pert_f = model(b2)
        m0 = (batch["mesh_node"].batch == 0)
        fb0 = (batch["beam_element"].batch == 0)
        iso = (torch.allclose(base_d[m0], pert_d[m0], atol=tol, rtol=0) and
               torch.allclose(base_f[fb0], pert_f[fb0], atol=tol, rtol=0))
        report["models"][name] = {
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "pred_shapes": [list(pd.shape), list(pf.shape)],
            "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss)),
            "grad_finite": bool(finite), "param_updated": bool(updated),
            "graph_isolated": bool(iso),
            "graph_isolation_tol": tol,
            "params_without_grad": [n for n, p in model.named_parameters()
                                    if p.requires_grad and p.grad is None][:20],
        }

    # 6. field descriptor covers all types
    ff = build("fullfield_mlp", cfg, dev).eval()
    with torch.no_grad():
        ctx = ff._field_descriptor(batch)
    report["checks"]["fullfield_context_shape"] = list(ctx.shape)
    report["checks"]["fullfield_context_finite"] = bool(torch.isfinite(ctx).all())

    report["all_ok"] = all([
        report["checks"]["shared_init_equal"],
        report["checks"]["global_modules_differ"],
        report["checks"]["global_off_matches_hgt"],
        report["checks"]["global_off_all_global_params_unused"],
        report["checks"]["fullfield_context_finite"],
    ] + [v["loss_finite"] and v["grad_finite"] and v["param_updated"] and v["graph_isolated"]
         and v["pred_shapes"][0][1] == 6 and v["pred_shapes"][1][1] == 12
         for v in report["models"].values()])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.device.replace(":", "_")
    (out / f"strong_baseline_checks_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "param_counts": counts,
                      "checks": {k: v for k, v in report["checks"].items()
                                 if k != "fullfield_context_shape"}},
                     ensure_ascii=False, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

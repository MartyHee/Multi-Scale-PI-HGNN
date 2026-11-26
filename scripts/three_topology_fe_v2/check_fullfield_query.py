#!/usr/bin/env python
"""Targeted checks for the full-field query MLP (per template, train/val only).

Checks:
  * field descriptor length equals the frozen template size and uses every
    node/beam/plate channel (one linear weight per flattened channel);
  * a remote load change reaches the graph context and the entity prediction;
  * two different designs produce different descriptors/contexts;
  * graph isolation in a mixed batch;
  * forward/loss/backward/update and inverse-transformed validation are finite.
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

TOPOS = ("topology1", "topology2", "topology3")
EXPECTED = {"topology1": 52956, "topology2": 87267, "topology3": 150264}


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "strong_baselines.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT.parent / "strong_baseline_prep_20261126" / "checks"))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    dev = engine.resolve_device(a.device)
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    rep = {"schema": "qt-fe-v2-fullfield-query-check-v1", "device": str(dev), "topologies": {}}
    for topo in TOPOS:
        st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{topo}.json")
        train_ids = [s for s in protocol.ids_for(idx, prot, topo, "A", "train")]
        # two different designs and one validation sample
        ds = FeV2Dataset(idx, train_ids[:2], st, normalize=True, use_shared=True, topology=topo)
        batch = Batch.from_data_list([ds[0], ds[1]]).to(dev)
        mcfg = schema.task_model_cfg("fullfield_query_mlp", cfg, topo, 20261117)
        torch.manual_seed(20261117)
        model = registry.build_gradient_model("fullfield_query_mlp", mcfg).to(dev)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model.train(); opt.zero_grad()
        pd, pf = model(batch)
        loss = F.mse_loss(pd, batch["mesh_node"].y_disp) + F.mse_loss(pf, batch["beam_element"].y_force)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        updated = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
        model.eval()
        # remote perturbation on graph 0: last node's Fx
        with torch.no_grad():
            base_d, base_f = model(batch)
            b2 = batch.clone()
            g0 = b2["mesh_node"].batch == 0
            last0 = torch.where(g0)[0][-1].item()
            b2["mesh_node"].x[last0, 3] += 2.0
            pert_d, pert_f = model(b2)
            ctx_base = model._fields(batch)
            ctx_pert = model._fields(b2)
            # graph isolation: perturb graph 1 only
            b3 = batch.clone()
            g1 = b3["mesh_node"].batch == 1
            b3["mesh_node"].x[torch.where(g1)[0][-1].item(), 3] += 2.0
            iso_d, iso_f = model(b3)
        m0, fb0 = (batch["mesh_node"].batch == 0), (batch["beam_element"].batch == 0)
        rep["topologies"][topo] = {
            "field_dim": int(mcfg["field_dim"]), "expected_field_dim": EXPECTED[topo],
            "field_dim_ok": int(mcfg["field_dim"]) == EXPECTED[topo],
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "bottleneck_dim": int(model.bottleneck_dim),
            "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss)),
            "grad_finite": bool(finite), "param_updated": bool(updated),
            "pred_shapes": [list(pd.shape), list(pf.shape)],
            "remote_context_changed": bool(not torch.allclose(ctx_base[0], ctx_pert[0])),
            "remote_context_max_diff": float((ctx_base[0] - ctx_pert[0]).abs().max().item()),
            "remote_prediction_changed": bool(not torch.allclose(base_d[m0], pert_d[m0], atol=1e-6)
                                               or not torch.allclose(base_f[fb0], pert_f[fb0], atol=1e-6)),
            "different_designs_distinct": bool(not torch.allclose(ctx_base[0], ctx_base[1])),
            "graph_isolated": bool(torch.allclose(base_d[m0], iso_d[m0], atol=1e-6)
                                   and torch.allclose(base_f[fb0], iso_f[fb0], atol=1e-6)),
            "field_proj_shape": list(model.field_proj.weight.shape),
        }
        # inverse-transformed one-slice validation
        val_ids = protocol.ids_for(idx, prot, topo, "A", "val")[:1]
        vds = FeV2Dataset(idx, val_ids, st, normalize=True, use_shared=True, topology=topo)
        vr = engine.validate(model, vds, st, 1, dev, max_batches=1)
        rep["topologies"][topo]["val_selection"] = vr["selection_metric"]
        rep["topologies"][topo]["val_finite"] = bool(vr["selection_metric"] is not None)
    rep["all_ok"] = all(
        v["field_dim_ok"] and v["loss_finite"] and v["grad_finite"] and v["param_updated"]
        and v["remote_context_changed"] and v["remote_prediction_changed"]
        and v["different_designs_distinct"] and v["graph_isolated"] and v["val_finite"]
        and v["pred_shapes"][0][1] == 6 and v["pred_shapes"][1][1] == 12
        for v in rep["topologies"].values())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.device.replace(":", "_")
    (out / f"fullfield_query_checks_{tag}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": rep["all_ok"],
                      "topologies": {k: {kk: v[kk] for kk in
                                         ("field_dim", "n_params", "remote_context_changed",
                                          "remote_prediction_changed", "different_designs_distinct",
                                          "graph_isolated", "val_selection")}
                                     for k, v in rep["topologies"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

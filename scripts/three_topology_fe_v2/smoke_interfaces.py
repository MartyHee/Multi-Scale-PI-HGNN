#!/usr/bin/env python
"""One-batch interface/gradient checks for all requested gradient models."""
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

from src.data.three_topology_fe_v2 import paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2 import index as index_mod  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

MODELS = ["mlp", "gcn", "graphsage", "gat", "gin", "rgcn", "hgt", "hgt_global"]


def load_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "smoke_pipeline.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "smoke_interfaces"))
    a = ap.parse_args()
    cfg = load_yaml(Path(a.config))
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / "topology1.json")
    train_ids = [s for s, v in prot["roles"].items()
                 if idx.by_id[s]["topology_token"] == "topology1" and v["A"] == "train"][:2]
    ds = FeV2Dataset(idx, train_ids, st, dtype=torch.float32, normalize=True)
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))])
    dev = engine.resolve_device(a.device)
    batch = batch.to(dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "qt-fe-v2-model-interface-v1", "device": str(dev),
              "samples": train_ids, "models": {}}
    for name in MODELS:
        torch.manual_seed(20261125)
        mcfg = schema.model_cfg_with_dims(name, cfg["models"][name])
        model = registry.build_gradient_model(name, mcfg).to(dev)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model.train(); opt.zero_grad()
        pd, pf = model(batch)
        loss = F.mse_loss(pd, batch["mesh_node"].y_disp) + F.mse_loss(pf, batch["beam_element"].y_force)
        loss.backward()
        grad_entries = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_grad = [n for n, p in grad_entries if p.grad is None]
        finite_grad = all(p.grad is None or torch.isfinite(p.grad).all() for _, p in grad_entries)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        changed = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
        report["models"][name] = {
            "n_params": sum(p.numel() for p in model.parameters()),
            "pred_disp_shape": list(pd.shape), "pred_force_shape": list(pf.shape),
            "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss)),
            "grad_finite": bool(finite_grad), "params_without_grad": no_grad, "param_updated": bool(changed),
            "input_dims": {"node": ds[0]["mesh_node"].x.shape[1],
                           "beam": ds[0]["beam_element"].x.shape[1],
                           "plate": ds[0]["plate_element"].x.shape[1]},
        }
        print(name, report["models"][name], flush=True)
    report["all_ok"] = all(v["loss_finite"] and v["grad_finite"] and v["param_updated"]
                           and v["pred_disp_shape"][1] == 6 and v["pred_force_shape"][1] == 12
                           for v in report["models"].values())
    (out / f"interfaces_{a.device.replace(':','_')}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"]}, ensure_ascii=False))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

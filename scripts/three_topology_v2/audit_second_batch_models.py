#!/usr/bin/env python
"""Audit the six second-batch baselines by *behaviour*, not by registry name.

For each model (gcn, graphsage, gat, gin, rgcn, hgt) plus hgt_global (first-batch
anchor) this script:
  * builds the model from the formal config and counts parameters;
  * runs a real forward on two training graphs of cd_fold_1 and checks output
    shapes / finiteness (readout: mesh->6, beam->12);
  * measures input dependency by perturbing one input at a time (permute mesh /
    beam / plate features; empty one relation's edge_index; permute
    structural_link edge_attr) and recording the max output change;
  * checks that the plain HGT trunk shares implementation/hyperparameters with
    HGT-global and reports the exact parameter delta.

No training, no validation scoring, no outer test (never constructed).
Outputs a JSON report; used for the second-batch input matrix.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_v2 import dataset as ds_mod, index, paths, stats as stats_mod  # noqa: E402
from src.train.three_topology_v2 import registry  # noqa: E402

MODELS = ["gcn", "graphsage", "gat", "gin", "rgcn", "hgt", "hgt_global"]
RELATIONS = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]


def load_cfg(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def forward_out(model, batch):
    model.eval()
    with torch.no_grad():
        pd, pf = model(batch)
    return pd, pf


def max_delta(a, b):
    d1 = (a[0] - b[0]).abs().max().item() if a[0].numel() else 0.0
    d2 = (a[1] - b[1]).abs().max().item() if a[1].numel() else 0.0
    return max(d1, d2)


def clone_batch(batch):
    # deep copy via from_data_list of cloned Data objects
    return Batch.from_data_list([d.clone() for d in batch.to_data_list()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "formal_baselines.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "second_batch_preparation_20261118" / "model_audit.json"))
    a = ap.parse_args()
    cfg = load_cfg(Path(a.config))
    idx = index.load_index(strict=True)

    split = json.loads((paths.SPLIT_DIR / "split_cross_design_topology1.json").read_text(encoding="utf-8"))
    fold = next(f for f in split["folds"] if f["fold_id"] == "cd_fold_1")
    stats_path = paths.STATS_DIR / "cross_design_topology1" / "cd_fold_1.json"
    st = json.loads(stats_path.read_text(encoding="utf-8"))
    st["_file_sha256"] = __import__("hashlib").sha256(stats_path.read_bytes()).hexdigest()

    train_ids = list(fold["train_sample_ids"])[:2]
    ds = ds_mod.FoldDataset(index=idx, sample_ids=train_ids, fold_stats=st, root=paths.package_root(),
                            dtype=torch.float32, verify_hash=True, normalize_labels=True)
    base_batch = Batch.from_data_list([ds[i] for i in range(len(ds))])

    report = {
        "schema": "qt-second-batch-model-audit-v1",
        "date": "2026-11-18",
        "split_id": "cross_design_topology1", "fold_id": "cd_fold_1",
        "sample_ids": train_ids,
        "graphs": {"trees": len(ds),
                   "mesh_nodes": int(base_batch["mesh_node"].num_nodes),
                   "beam_elements": int(base_batch["beam_element"].num_nodes),
                   "plate_elements": int(base_batch["plate_element"].num_nodes),
                   "unique_feature_rows": {
                       k: int(len(torch.unique(base_batch[k].x, dim=0)))
                       for k in ("mesh_node", "beam_element", "plate_element")}},
        "models": {},
    }

    for name in MODELS:
        torch.manual_seed(0)
        model = registry.build_gradient_model(name, copy.deepcopy(cfg["models"][name]))
        pd, pf = forward_out(model, base_batch)
        m = {
            "model": name,
            "n_params": n_params(model),
            "param_tensors": len(list(model.parameters())),
            "pred_disp_shape": list(pd.shape),
            "pred_force_shape": list(pf.shape),
            "finite": bool(torch.isfinite(pd).all() and torch.isfinite(pf).all()),
            "readout_ok": (pd.shape[0] == base_batch["mesh_node"].num_nodes
                           and pf.shape[0] == base_batch["beam_element"].num_nodes
                           and pd.shape[1] == 6 and pf.shape[1] == 12),
            "uses": {}, "n_params_by_module": {},
        }
        for mod_name, mod in model.named_children():
            m["n_params_by_module"][mod_name] = sum(p.numel() for p in mod.parameters())
        base = (pd.clone(), pf.clone())

        # feature dependency: permutation (only meaningful for varying rows) and
        # additive shift (also detects a constant input column).
        for key in ("mesh_node", "beam_element", "plate_element"):
            b = clone_batch(base_batch)
            x = b[key].x
            if x.shape[0] > 1:
                b[key].x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(1))]
            out = forward_out(model, b)
            m["uses"][f"{key}.x_permuted"] = max_delta(base, out)

            b = clone_batch(base_batch)
            b[key].x = b[key].x + 1.0
            out = forward_out(model, b)
            m["uses"][f"{key}.x_shifted_plus1"] = max_delta(base, out)

        # relation dependency: drop the relation's edges
        for et in RELATIONS:
            b = clone_batch(base_batch)
            ei = b[et].edge_index
            b[et].edge_index = torch.zeros((2, 0), dtype=ei.dtype)
            out = forward_out(model, b)
            m["uses"][f"edge:{et[1]}_removed"] = max_delta(base, out)

        # structural_link edge_attr dependency (only edge-aware models should react)
        b = clone_batch(base_batch)
        ea = b["mesh_node", "structural_link", "mesh_node"].edge_attr
        if ea.shape[0] > 1:
            b["mesh_node", "structural_link", "mesh_node"].edge_attr = ea[
                torch.randperm(ea.shape[0], generator=torch.Generator().manual_seed(2))]
        out = forward_out(model, b)
        m["uses"]["structural_link.edge_attr_permuted"] = max_delta(base, out)

        def used(key):
            return (m["uses"][f"{key}.x_permuted"] > 0 or m["uses"][f"{key}.x_shifted_plus1"] > 0)

        m["input_notes"] = {
            "mesh_x": "used" if used("mesh_node") else "not used",
            "beam_x": "used" if used("beam_element") else "not used",
            "plate_x": "used" if used("plate_element") else "not used",
            "plate_x_rows_constant_within_graph": report["graphs"]["unique_feature_rows"]["plate_element"] == 1,
            "belongs_to_beam": "used" if m["uses"]["edge:belongs_to_beam_removed"] > 0 else "not used",
            "rev_belongs_to_beam": "used" if m["uses"]["edge:rev_belongs_to_beam_removed"] > 0 else "not used",
            "belongs_to_plate": "used" if m["uses"]["edge:belongs_to_plate_removed"] > 0 else "not used",
            "rev_belongs_to_plate": "used" if m["uses"]["edge:rev_belongs_to_plate_removed"] > 0 else "not used",
            "structural_link.edges": "used" if m["uses"]["edge:structural_link_removed"] > 0 else "not used",
            "structural_link.edge_attr": "used" if m["uses"]["structural_link.edge_attr_permuted"] > 0 else "not used",
        }
        report["models"][name] = m

    # plain HGT vs HGT-global: shared trunk / parameter delta
    torch.manual_seed(0)
    hgt = registry.build_gradient_model("hgt", copy.deepcopy(cfg["models"]["hgt"]))
    torch.manual_seed(0)
    gp = registry.build_gradient_model("hgt_global", copy.deepcopy(cfg["models"]["hgt_global"]))
    sd_h, sd_g = hgt.state_dict(), gp.state_dict()
    shared = sorted(set(sd_h) & set(sd_g))
    only_global = sorted(set(sd_g) - set(sd_h))
    only_hgt = sorted(set(sd_h) - set(sd_g))
    identical_shared = all(torch.equal(sd_h[k], sd_g[k]) for k in shared)
    same_shape_shared = all(sd_h[k].shape == sd_g[k].shape for k in shared)
    cfg_h = cfg["models"]["hgt"]; cfg_g = cfg["models"]["hgt_global"]
    shared_cfg_keys = {k: cfg_h.get(k) for k in cfg_h if k in cfg_g and cfg_h.get(k) == cfg_g.get(k)}
    diff_cfg_keys = {k: {"hgt": cfg_h.get(k), "hgt_global": cfg_g.get(k)}
                     for k in set(cfg_h) | set(cfg_g) if cfg_h.get(k) != cfg_g.get(k)}
    report["hgt_vs_hgt_global"] = {
        "shared_param_tensors": len(shared),
        "shared_shapes_equal": same_shape_shared,
        "shared_values_identical_same_seed": identical_shared,
        "global_only_params": only_global,
        "hgt_only_params": only_hgt,
        "n_params_hgt": n_params(hgt),
        "n_params_hgt_global": n_params(gp),
        "param_delta": n_params(gp) - n_params(hgt),
        "shared_config": shared_cfg_keys,
        "differing_config": diff_cfg_keys,
        "note": ("same trunk code + hyperparameters; HGT-global adds a GRUCell "
                 "global memory, a broadcast Linear and a gated-residual fusion. "
                 "Because the extra modules are constructed before the decoders, the "
                 "same seed does not give an identical decoder initialisation; the "
                 "comparison is not a controlled same-init ablation."),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

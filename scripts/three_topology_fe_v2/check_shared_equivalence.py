#!/usr/bin/env python
"""Verify the shared-cache read path is equivalent to the legacy extraction path.

Compares raw arrays, encoded graph tensors, normalised tensors, dtypes, and a
same-weight MLP forward/loss on CPU (strict) and CUDA (tolerance).  Uses only
training / validation-role samples; test/blocked are never selected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.data.three_topology_fe_v2 import assemble, graph, index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402
from src.data.three_topology_fe_v2.shared_cache import SharedCache  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

TEMPLATES = ("topology1", "topology2", "topology3")


def _cmp(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return {"shape_equal": False, "a": list(a.shape), "b": list(b.shape)}
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        eq = bool(np.array_equal(a, b, equal_nan=True))
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        diff = float(np.nanmax(d)) if d.size and not np.all(np.isnan(d)) else 0.0
        return {"shape_equal": True, "exact": eq, "max_abs_diff": float(diff),
                "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    return {"shape_equal": True, "exact": bool(np.array_equal(a, b)),
            "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}


def _tensor_cmp(a, b):
    if a is None and b is None:
        return {"present": True, "both_none": True, "equal": True}
    if a is None or b is None:
        return {"present": False, "equal": False}
    ac, bc = a.cpu(), b.cpu()
    if ac.dtype.is_floating_point:
        eq = bool(torch.allclose(ac, bc, rtol=0, atol=0, equal_nan=True))
        d = (ac.float() - bc.float()).abs()
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        eq = bool(torch.equal(ac, bc)); d = torch.zeros(1)
    return {"shape": list(a.shape), "equal": eq,
            "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
            "max_abs_diff": float(d.max().item()) if d.numel() else 0.0}


def compare_raw(a, b):
    keys = ["node_ids", "node_xyz", "node_load", "node_support", "node_x", "node_y",
            "beam_ids", "beam_node_ids", "beam_node_index", "beam_raw", "beam_y",
            "beam_end_mask", "beam_sec_id", "beam_mat_id", "plate_ids",
            "plate_node_ids", "plate_node_index", "plate_raw", "support_ids",
            "support_node_index", "support_flags", "link_ids", "link_edge_index",
            "link_attr", "release_ids", "release_beam_index", "release_node_index",
            "release_info", "release_edge_beam", "release_edge_node",
            "release_edge_role", "release_edge_info", "master_ids",
            "master_node_index", "master_info"]
    out = {k: _cmp(a[k], b[k]) for k in keys}
    out["n_equations"] = {"a": int(a["n_equations"]), "b": int(b["n_equations"]),
                          "equal": int(a["n_equations"]) == int(b["n_equations"])}
    out["beam_end_prop_used"] = _cmp(a["beam_end_prop"][:, :, [0, 3, 4, 5]],
                                     b["beam_end_prop"][:, :, [0, 3, 4, 5]])
    return out


def compare_graph(a, b):
    out = {}
    for nt in ("mesh_node", "beam_element", "plate_element"):
        for attr in sorted(set(a[nt].keys()) | set(b[nt].keys())):
            if nt == "beam_element" and attr == "end_section_29":
                continue  # full 29-col sidecar remains in source extraction
            out[f"{nt}.{attr}"] = _tensor_cmp(a[nt].get(attr), b[nt].get(attr))
    out["beam_element.end_section_29_used"] = _tensor_cmp(
        a["beam_element"].end_section_29[..., [0, 3, 4, 5]],
        b["beam_element"].end_section_29[..., [0, 3, 4, 5]])
    for et in graph.EDGE_TYPES:
        key = ".".join(et)
        out[f"edge.{key}.edge_index"] = _tensor_cmp(a[et].edge_index, b[et].edge_index)
        out[f"edge.{key}.edge_attr"] = _tensor_cmp(a[et].get("edge_attr"), b[et].get("edge_attr"))
    for attr in ("support_id", "support_node_index", "support_boundary_info",
                 "beam_release_id", "beam_release_element_index", "beam_release_node_index",
                 "beam_release_info", "beam_release_edge_element_index",
                 "beam_release_edge_node_index", "beam_release_edge_role",
                 "beam_release_edge_info", "master_slave_id", "master_slave_node_index",
                 "master_slave_boundary_info", "constraint_equation_count",
                 "topology_code", "coordinate_system_code", "sample_index"):
        out[f"data.{attr}"] = _tensor_cmp(getattr(a, attr, None), getattr(b, attr, None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "benchmark" / "shared_equivalence.json"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    reader = FeV2Reader(idx, max_runs=2, verify_hash=True)
    rec = {"schema": "qt-fe-v2-shared-equivalence-v1", "device": a.device, "templates": {}}
    ok_all = True
    import yaml
    cfg = yaml.safe_load((paths.CONFIG_ROOT / "formal_pipeline.yaml").read_text(encoding="utf-8"))
    for t in TEMPLATES:
        st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{t}.json")
        shared = SharedCache(idx, t, reader=FeV2Reader(idx, max_runs=8, verify_hash=True))
        ids = []
        for p, role in (("A", "train"), ("A", "val"), ("B", "val"), ("C", "val")):
            ids.extend(protocol.ids_for(idx, prot, t, p, role)[:2])
        ids = list(dict.fromkeys(ids))
        for sid in ids:
            if "test" in prot["roles"][sid].values():
                raise SystemExit(f"{sid}: selected a test-role sample")
        raw_cmp, graph_cmp, norm_cmp = {}, {}, {}
        old_data, new_data = [], []
        for sid in ids:
            old_raw = assemble.assemble_raw(reader.read(sid))
            new_raw = shared.assemble(sid)
            raw_cmp[sid] = compare_raw(old_raw, new_raw)
            go = graph.build_graph(old_raw, dtype=torch.float32)
            gn = graph.build_graph(new_raw, dtype=torch.float32)
            graph_cmp[sid] = compare_graph(go, gn)
            graph.apply_normalization_(go, st); graph.apply_normalization_(gn, st)
            norm_cmp[sid] = compare_graph(go, gn)
            old_data.append(go); new_data.append(gn)
        raw_ok = all(all(v.get("exact", v.get("equal", False)) for v in x.values())
                     for x in raw_cmp.values())
        graph_ok = all(all(v.get("equal", v.get("present", False)) for v in x.values()
                           if "equal" in v or "present" in v) for x in graph_cmp.values())
        norm_ok = all(all(v.get("equal", v.get("present", False)) for v in x.values()
                          if "equal" in v or "present" in v) for x in norm_cmp.values())
        # Same-weight MLP forward/loss.
        engine.seed_everything(20261117)
        mcfg = schema.model_cfg_with_dims("mlp", cfg["models"]["mlp"])
        model = registry.build_gradient_model("mlp", mcfg)
        model.eval()
        dev = engine.resolve_device(a.device)
        model.to(dev)
        bo = Batch.from_data_list(old_data).to(dev)
        bn = Batch.from_data_list(new_data).to(dev)
        with torch.no_grad():
            po = model(bo); pn = model(bn)
        if dev.type == "cpu":
            pred_ok = bool(torch.equal(po[0], pn[0]) and torch.equal(po[1], pn[1]))
            loss_ok = bool(torch.equal(
                torch.nn.functional.mse_loss(po[0], bo["mesh_node"].y_disp)
                + torch.nn.functional.mse_loss(po[1], bo["beam_element"].y_force),
                torch.nn.functional.mse_loss(pn[0], bn["mesh_node"].y_disp)
                + torch.nn.functional.mse_loss(pn[1], bn["beam_element"].y_force)))
            max_diff = 0.0
        else:
            max_diff = max(float((po[i] - pn[i]).abs().max().item()) for i in (0, 1))
            pred_ok = max_diff < 1e-6
            loss_ok = abs(float(torch.nn.functional.mse_loss(po[0], bo["mesh_node"].y_disp)
                                - torch.nn.functional.mse_loss(pn[0], bn["mesh_node"].y_disp))) < 1e-6
        ok = raw_ok and graph_ok and norm_ok and pred_ok and loss_ok
        ok_all &= ok
        rec["templates"][t] = {
            "samples": ids, "raw_ok": raw_ok, "graph_ok": graph_ok,
            "normalized_ok": norm_ok, "forward_ok": pred_ok, "loss_ok": loss_ok,
            "max_forward_abs_diff": max_diff, "cache": shared.stats(),
            "raw": raw_cmp, "graph": graph_cmp, "normalized": norm_cmp,
        }
        print(t, "ok" if ok else "FAIL", {"raw": raw_ok, "graph": graph_ok,
                                          "norm": norm_ok, "fwd": pred_ok, "loss": loss_ok})
    rec["all_ok"] = ok_all
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": ok_all, "out": str(out)}, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

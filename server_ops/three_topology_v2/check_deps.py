#!/usr/bin/env python
"""Read-only dependency / operator check for the server pi_hgnn environment.

Prints versions and import status for the packages the first batch actually
imports, checks the PyG operators used by the baselines (with a tiny CUDA
forward), and reports GPU model/VRAM/device mapping. Does NOT install anything.
Exit 0 iff required imports + operators work.

Required for first batch: torch, torch_geometric, torch_scatter, numpy,
scikit-learn, PyYAML. Optional: psutil (memory logging only).
"""
from __future__ import annotations

import importlib
import json
import sys

REQUIRED = ["torch", "torch_geometric", "torch_scatter", "numpy", "sklearn", "yaml"]
OPTIONAL = ["psutil", "torch_sparse", "scipy"]
PYG_OPS = ["GCNConv", "SAGEConv", "GATConv", "GINConv", "RGCNConv", "GATv2Conv", "GINEConv"]


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return {"import": "ok", "version": getattr(m, "__version__", "?")}
    except Exception as exc:  # noqa: BLE001
        return {"import": "MISSING", "error": repr(exc)}


def main() -> int:
    report = {"python": sys.version.split()[0], "required": {}, "optional": {}, "pyg_ops": {},
              "gpu": {}}
    ok = True
    for m in REQUIRED:
        r = ver(m); report["required"][m] = r
        if r["import"] != "ok":
            ok = False
    for m in OPTIONAL:
        report["optional"][m] = ver(m)

    import torch
    report["gpu"] = {"cuda_available": torch.cuda.is_available(),
                     "cuda_version": torch.version.cuda, "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        devs = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs.append({"index": i, "name": p.name,
                         "total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
                         "capability": f"{p.major}.{p.minor}"})
        report["gpu"]["devices"] = devs

    try:
        import torch_geometric.nn as gnn
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # tiny homogeneous graph: 8 nodes
        x = torch.randn(8, 16, device=device)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], device=device)
        for name in PYG_OPS:
            try:
                cls = getattr(gnn, name)
                if name in ("RGCNConv",):
                    conv = cls(16, 16, num_relations=2).to(device)
                    out = conv(x, ei, torch.zeros(ei.size(1), dtype=torch.long, device=device))
                elif name == "GINConv":
                    conv = cls(torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                                   torch.nn.Linear(16, 16))).to(device)
                    out = conv(x, ei)
                elif name == "GINEConv":
                    conv = cls(torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                                   torch.nn.Linear(16, 16)), edge_dim=16).to(device)
                    ea = torch.randn(ei.size(1), 16, device=device)
                    out = conv(x, ei, ea)
                else:
                    conv = cls(16, 16).to(device)
                    out = conv(x, ei)
                report["pyg_ops"][name] = {"run": "ok", "out_shape": list(out.shape)}
            except Exception as exc:  # noqa: BLE001
                report["pyg_ops"][name] = {"run": "FAIL", "error": repr(exc)}
                ok = False
        # HGTConv needs metadata
        try:
            from torch_geometric.nn import HGTConv
            node_types = ["a", "b"]; edge_types = [("a", "to", "b"), ("b", "rev", "a")]
            conv = HGTConv(16, 16, metadata=(node_types, edge_types), heads=2).to(device)
            xd = {"a": torch.randn(4, 16, device=device), "b": torch.randn(4, 16, device=device)}
            ed = {("a", "to", "b"): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device),
                  ("b", "rev", "a"): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device)}
            out = conv(xd, ed)
            report["pyg_ops"]["HGTConv"] = {"run": "ok", "out": {k: list(v.shape) for k, v in out.items()}}
        except Exception as exc:  # noqa: BLE001
            report["pyg_ops"]["HGTConv"] = {"run": "FAIL", "error": repr(exc)}; ok = False
    except Exception as exc:  # noqa: BLE001
        report["pyg_ops"]["_import"] = {"run": "FAIL", "error": repr(exc)}; ok = False

    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

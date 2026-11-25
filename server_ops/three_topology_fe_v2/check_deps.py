#!/usr/bin/env python
"""Dependency + operator check for the FE v2 preflight (no installs)."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-cuda", action="store_true")
    a = ap.parse_args()
    rec = {"python": sys.version.split()[0], "ok": True, "problems": []}
    try:
        import numpy
        import sklearn
        import torch
        import torch_geometric
        import yaml
        rec.update({"numpy": numpy.__version__, "sklearn": sklearn.__version__,
                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "torch_geometric": torch_geometric.__version__,
                    "pyyaml": yaml.__version__})
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["problems"].append(f"import failed: {exc!r}")
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 1

    cuda = bool(torch.cuda.is_available())
    rec["cuda_available"] = cuda
    rec["cuda_device_count"] = torch.cuda.device_count() if cuda else 0
    if cuda:
        p = torch.cuda.get_device_properties(0)
        rec["gpu_name"] = p.name
        rec["gpu_total_memory_gb"] = round(p.total_memory / 1024 ** 3, 2)
        try:
            x = torch.randn(64, 64, device="cuda:0")
            y = (x @ x).sum().item()
            rec["cuda_matmul_ok"] = bool(y == y)
        except Exception as exc:  # noqa: BLE001
            rec["cuda_matmul_ok"] = False
            rec["problems"].append(f"cuda matmul failed: {exc!r}")
            rec["ok"] = False
    elif a.require_cuda:
        rec["problems"].append("CUDA required but not visible")
        rec["ok"] = False

    try:
        from torch_geometric.utils import scatter
        src = torch.tensor([0, 0, 1], dtype=torch.long)
        out = scatter(torch.ones(3, 1), src, dim=0, reduce="sum")
        rec["pyg_scatter_ok"] = out.shape == (2, 1)
    except Exception as exc:  # noqa: BLE001
        rec["pyg_scatter_ok"] = False
        rec["problems"].append(f"PyG scatter failed: {exc!r}")
        rec["ok"] = False

    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

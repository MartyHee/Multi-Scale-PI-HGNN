#!/usr/bin/env python
"""Small-scale stage timing for the FE v2 training path.

Separates: source read/decompress, raw assembly, graph build+normalisation,
CPU batch, CPU->GPU copy, forward/backward, and three-slice validation/export.
Uses a handful of train/validation samples; never reads test responses.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

MODEL_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import assemble, graph, index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topology", default="topology1", choices=["topology1", "topology2", "topology3"])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = engine.resolve_device(a.device)
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{a.topology}.json")
    train_ids = protocol.ids_for(idx, prot, a.topology, "A", "train")[: a.samples]
    val_ids = {
        "design_val": protocol.ids_for(idx, prot, a.topology, "A", "val")[:2],
        "load_val": protocol.ids_for(idx, prot, a.topology, "B", "val")[:2],
        "joint_val": protocol.ids_for(idx, prot, a.topology, "C", "val")[:2],
    }
    reader = FeV2Reader(idx, max_runs=2, verify_hash=True)
    stage = {"read": [], "assemble": [], "graph_norm": []}
    datas = []
    for sid in train_ids:
        t0 = time.perf_counter(); rec = reader.read(sid); t1 = time.perf_counter()
        raw = assemble.assemble_raw(rec); t2 = time.perf_counter()
        data = graph.build_graph(raw, dtype=torch.float32); graph.apply_normalization_(data, st)
        t3 = time.perf_counter()
        datas.append(data)
        stage["read"].append(t1 - t0); stage["assemble"].append(t2 - t1)
        stage["graph_norm"].append(t3 - t2)
    t0 = time.perf_counter(); batch = Batch.from_data_list(datas); t1 = time.perf_counter()
    cpu_batch = t1 - t0
    if dev.type == "cuda":
        torch.cuda.synchronize(); t0 = time.perf_counter()
        batch = batch.to(dev); torch.cuda.synchronize(); h2d = time.perf_counter() - t0
    else:
        h2d = None

    import yaml
    cfg = yaml.safe_load((paths.CONFIG_ROOT / "formal_pipeline.yaml").read_text(encoding="utf-8"))
    mcfg = schema.model_cfg_with_dims("mlp", cfg["models"]["mlp"])
    model = registry.build_gradient_model("mlp", mcfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fwd_bwd = []
    for _ in range(3):
        opt.zero_grad()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pd, pf = model(batch)
        loss = torch.nn.functional.mse_loss(pd, batch["mesh_node"].y_disp) + \
            torch.nn.functional.mse_loss(pf, batch["beam_element"].y_force)
        loss.backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        fwd_bwd.append(time.perf_counter() - t0)

    val_ds = {k: FeV2Dataset(idx, v, st, dtype=torch.float32, normalize=True) for k, v in val_ids.items()}
    ev = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ev(model, 1, dev, collect_predictions=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    val_sec = time.perf_counter() - t0
    out_dir = Path(a.out) if a.out else (paths.OUTPUT_ROOT / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": "qt-fe-v2-stage-timing-v1", "device": str(dev), "topology": a.topology,
        "samples": len(train_ids), "cache": {"hits": reader.hits, "misses": reader.misses,
                                              "max_runs": reader.max_runs, "disk_cache": False},
        "seconds": {
            "read_decompress": stage["read"],
            "assemble": stage["assemble"],
            "graph_build_norm": stage["graph_norm"],
            "cpu_batch": cpu_batch, "cpu_to_gpu": h2d,
            "forward_backward": fwd_bwd, "three_slice_valid_export": val_sec,
        },
        "means": {
            "read_decompress": float(np.mean(stage["read"])),
            "assemble": float(np.mean(stage["assemble"])),
            "graph_build_norm": float(np.mean(stage["graph_norm"])),
            "forward_backward": float(np.mean(fwd_bwd)),
            "three_slice_valid_export": val_sec,
        },
        "val_slices": {k: len(v) for k, v in val_ids.items()},
        "selection_metric": out["selection_metric"],
        "peak_rss_mb": peak_rss_mb(),
        "six_process_estimate": {
            "peak_rss_mb_each": peak_rss_mb(),
            "peak_rss_mb_total_6": (peak_rss_mb() or 0) * 6,
            "reader_cache_runs_each": reader.max_runs,
            "reader_cache_runs_total_6": reader.max_runs * 6,
            "note": "estimate only; actual server RAM reported by preflight/monitor",
        },
        "test_constructed": False,
    }
    name = f"train_stages_{str(dev).replace(':', '_')}_{a.topology}.json"
    (out_dir / name).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

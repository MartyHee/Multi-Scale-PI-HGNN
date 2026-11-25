#!/usr/bin/env python
"""Bounded benchmark of the legacy extraction read vs the shared-cache read.

Caps are explicit and independent of the formal run identity:
  --samples <= 32, --val-per-slice <= 8, --batches <= 8.
It only touches A/B/C training and validation roles (never test/blocked) and
does not train a model beyond a few forward/backward steps.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

MODEL_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator  # noqa: E402
from src.data.three_topology_fe_v2.shared_cache import SharedCache  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

TEMPLATES = ("topology1", "topology2", "topology3")


def _rss():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def _spread(items, n):
    if len(items) <= n:
        return list(items)
    step = max(1, len(items) // n)
    return list(items[::step])[:n]


def _pick(idx, prot, topology, n):
    ids = _spread(protocol.ids_for(idx, prot, topology, "A", "train"), n)
    val = {k: _spread(protocol.ids_for(idx, prot, topology, p, "val"), n)
           for k, p in (("design_val", "A"), ("load_val", "B"), ("joint_val", "C"))}
    return ids, val


def measure(path_name, idx, prot, topology, st, ids, val, args):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    shared = None
    if path_name == "shared":
        shared = SharedCache(idx, topology)
        ds = FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                         use_shared=True, shared=shared, topology=topology)
    else:
        ds = FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                         use_shared=False)
    setup = time.perf_counter() - t0
    cold, hot, datas = [], [], []
    for i in range(len(ids)):
        t = time.perf_counter(); d = ds[i]; cold.append(time.perf_counter() - t); datas.append(d)
    for i in range(len(ids)):
        t = time.perf_counter(); ds[i]; hot.append(time.perf_counter() - t)
    # Random order with a FRESH cache, so the number is not a warm-cache replay.
    t = time.perf_counter()
    if path_name == "shared":
        rshared = SharedCache(idx, topology)
        rds = FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                          use_shared=True, shared=rshared, topology=topology)
    else:
        rds = FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True,
                          use_shared=False)
    random_setup = time.perf_counter() - t
    order = np.random.default_rng(20261125).permutation(len(ids)).tolist()
    rnd = []
    for i in order:
        t = time.perf_counter(); rds[i]; rnd.append(time.perf_counter() - t)
    t = time.perf_counter(); batch = Batch.from_data_list(datas); batch_sec = time.perf_counter() - t
    val_ds = {k: FeV2Dataset(idx, v, st, dtype=torch.float32, normalize=True,
                             use_shared=shared is not None, shared=shared,
                             topology=topology) for k, v in val.items()}
    dev = engine.resolve_device(args.device)
    if dev.type == "cuda":
        batch = batch.to(dev)
    import yaml
    cfg = yaml.safe_load((paths.CONFIG_ROOT / "formal_pipeline.yaml").read_text(encoding="utf-8"))
    model = registry.build_gradient_model("mlp", schema.model_cfg_with_dims("mlp", cfg["models"]["mlp"])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = []
    for _ in range(args.batches):
        opt.zero_grad()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        pd, pf = model(batch)
        loss = torch.nn.functional.mse_loss(pd, batch["mesh_node"].y_disp) + \
            torch.nn.functional.mse_loss(pf, batch["beam_element"].y_force)
        loss.backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        steps.append(time.perf_counter() - t)
    ev = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter(); out = ev(model, 1, dev, collect_predictions=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    val_sec = time.perf_counter() - t
    rec = {
        "path": path_name, "setup_sec": setup,
        "gpu_max_memory_mb": (round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
                              if dev.type == "cuda" else None),
        "cold_sec": cold, "hot_sec": hot, "random_sec": rnd,
        "random_setup_sec": random_setup,
        "cold_mean": float(np.mean(cold)), "hot_mean": float(np.mean(hot)),
        "random_mean": float(np.mean(rnd)), "batch_sec": batch_sec,
        "forward_backward_sec": steps, "forward_backward_mean": float(np.mean(steps)),
        "three_slice_val_sec": val_sec, "selection_metric": out["selection_metric"],
        "peak_rss_mb": _rss(),
    }
    if shared is not None:
        rec["cache"] = shared.stats()
    else:
        rec["cache"] = {"reader_hits": ds.reader.hits, "reader_misses": ds.reader.misses,
                        "max_runs": ds.reader.max_runs, "disk_cache": False}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topology", default="topology1", choices=list(TEMPLATES))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--val-per-slice", type=int, default=2)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--paths", default="both", choices=["shared", "legacy", "both"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.samples = max(1, min(int(a.samples), 32))
    a.val_per_slice = max(1, min(int(a.val_per_slice), 8))
    a.batches = max(1, min(int(a.batches), 8))
    idx = index_mod.load_index(paths.PACKAGE_ROOT)
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{a.topology}.json")
    ids, val = _pick(idx, prot, a.topology, a.samples)
    val = {k: v[:a.val_per_slice] for k, v in val.items()}
    for sid in ids + [x for v in val.values() for x in v]:
        if "test" in prot["roles"][sid].values():
            raise SystemExit(f"{sid}: selected a test-role sample")
    names = ["shared", "legacy"] if a.paths == "both" else [a.paths]
    rec = {
        "schema": "qt-fe-v2-read-benchmark-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "device_requested": a.device, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": (torch.cuda.get_device_name(0)
                                            if torch.cuda.is_available() else None),
        "topology": a.topology, "samples": ids,
        "val_samples": {k: v for k, v in val.items()},
        "caps": {"samples": a.samples, "val_per_slice": a.val_per_slice,
                 "batches": a.batches, "epochs": 0},
        "paths": {}, "notes": [
            "Only training core and validation roles; test/blocked not loaded.",
            "No formal run identity; output is not part of the 18-task batch.",
            "Cold = first access with fresh reader; hot = same sequence second pass; "
            "random = seeded permutation of the same sample set.",
        ],
    }
    for name in names:
        print("measuring", name, flush=True)
        rec["paths"][name] = measure(name, idx, prot, a.topology, st, ids, val, a)
    rec["processed_bytes"] = sum(p.stat().st_size for p in paths.PROCESSED_ROOT.rglob("*") if p.is_file())
    rec["design_cache_bytes"] = sum(
        (paths.PROCESSED_ROOT / "shared" / f"{t}_design_members.npz").stat().st_size
        for t in TEMPLATES)
    out = Path(a.out) if a.out else (paths.OUTPUT_ROOT / "benchmark" / f"read_benchmark_{a.topology}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "paths": {k: {"cold_mean": v["cold_mean"],
                                                      "hot_mean": v["hot_mean"],
                                                      "random_mean": v["random_mean"],
                                                      "val_sec": v["three_slice_val_sec"]}
                                                  for k, v in rec["paths"].items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

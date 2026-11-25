#!/usr/bin/env python
"""Representative read/cache/batch measurements; no training."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from src.data.three_topology_fe_v2 import assemble, graph, index as index_mod, paths  # noqa: E402
from src.data.three_topology_fe_v2.reader import FeV2Reader  # noqa: E402


def dir_bytes(p: Path) -> int:
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "benchmark"))
    a = ap.parse_args()
    idx = index_mod.load_index()
    rng = random.Random(20261125)
    import json
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
    sample_ids = [r["sample_id"] for r in idx.rows
                  if prot["roles"][r["sample_id"]]["A"] in ("train", "val")]
    picks = rng.sample(sample_ids, 12)
    reader = FeV2Reader(idx, max_runs=2, verify_hash=True)
    try:
        import psutil
        proc = psutil.Process()
        peak_rss = proc.memory_info().rss
    except Exception:  # noqa: BLE001
        proc = None; peak_rss = 0
    def rss():
        nonlocal peak_rss
        if proc is not None:
            peak_rss = max(peak_rss, proc.memory_info().rss)
    times, counts = [], []
    for sid in picks:
        t0 = time.perf_counter()
        raw = assemble.assemble_raw(reader.read(sid))
        d = graph.build_graph(raw)
        times.append((time.perf_counter() - t0) * 1000)
        counts.append((len(raw["node_ids"]), len(raw["beam_ids"]), len(raw["plate_ids"])))
        rss()
    # repeat three fields from the same run to exercise the 2-run LRU.
    base = idx.by_id[picks[0]]
    repeats = [r["sample_id"] for r in idx.rows
               if r["extraction_path"] == base["extraction_path"]][:3]
    rt = []
    for sid in repeats:
        t0 = time.perf_counter(); assemble.assemble_raw(reader.read(sid)); rt.append((time.perf_counter()-t0)*1000); rss()
    # batch construction of four samples, one template
    batch_ids = [r["sample_id"] for r in idx.rows
                 if r["topology_token"] == "topology1" and prot["roles"][r["sample_id"]]["A"] in ("train", "val")][:4]
    t0 = time.perf_counter()
    graphs = [graph.build_graph(assemble.assemble_raw(reader.read(s)) ) for s in batch_ids]
    from torch_geometric.data import Batch
    b = Batch.from_data_list(graphs)
    batch_ms = (time.perf_counter() - t0) * 1000
    rss()
    report = {
        "schema": "qt-fe-v2-io-benchmark-v1",
        "random_samples": picks, "random_ms": times,
        "random_median_ms": statistics.median(times),
        "random_range_ms": [min(times), max(times)],
        "counts": counts,
        "repeat_same_run_ms": rt,
        "reader_cache": {"hits": reader.hits, "misses": reader.misses, "max_runs": reader.max_runs,
                           "disk_cache": False, "dataloader_workers": 0},
        "batch4_ms": batch_ms,
        "batch4_nodes": int(b["mesh_node"].num_nodes), "batch4_beams": int(b["beam_element"].num_nodes),
        "peak_rss_mb": peak_rss / 1024 / 1024,
        "processed_bytes": dir_bytes(paths.PROCESSED_ROOT),
        "source_payload_bytes": int(idx.manifest.get("payload_bytes", 0)),
        "source_budget_bytes": int(idx.manifest.get("budget_bytes", 0)),
        "notes": ["random reads use only A train/val roles; no test/blocked response is loaded",
                  "random reads include extraction/shard JSON/NPZ access and graph construction",
                  "cache is a 2-source-run LRU, not a full response cache",
                  "no response arrays are duplicated into processed"],
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "io_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("random_median_ms", "random_range_ms", "reader_cache",
                                             "batch4_ms", "processed_bytes", "source_payload_bytes")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

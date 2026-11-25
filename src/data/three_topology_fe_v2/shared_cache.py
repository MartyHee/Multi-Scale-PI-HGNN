"""Design/field-shared read path for formal training.

Reuses ``processed/shared/<t>_geometry.npz``, ``shared/<t>_design_members.npz``
and ``fields/<t>_loads.npz``.  Per sample it reads only the response shard row;
no extraction JSON is decompressed or parsed in the hot path.  Global shuffle,
seed, sample order and batch composition are unchanged.

Bounds:
  * geometry: template-sized (small);
  * design members: all 48 designs for one template (npz, typically < 1 MB
    compressed; loaded once per process);
  * field loads: all 64 fields for one template (small);
  * response shards: LRU ``max_shard_runs`` (default 32), configurable via
    ``THREE_TOPO_SHARD_CACHE``; no disk cache and no full source-run cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import assemble, paths
from .reader import FeV2Reader


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SharedCacheError(RuntimeError):
    pass


class SharedCache:
    def __init__(self, index, topology: str, reader: FeV2Reader | None = None,
                 verify: bool = True):
        self.index = index
        self.topology = topology
        base = paths.PROCESSED_ROOT
        self.geometry_path = base / "shared" / f"{topology}_geometry.npz"
        self.design_path = base / "shared" / f"{topology}_design_members.npz"
        self.design_meta_path = base / "shared" / f"{topology}_design_cache.json"
        self.loads_path = base / "fields" / f"{topology}_loads.npz"
        self.fields_csv = base / "fields" / f"{topology}.csv"
        for p in (self.geometry_path, self.design_path, self.design_meta_path,
                  self.loads_path, self.fields_csv):
            if not p.is_file():
                raise SharedCacheError(
                    f"shared cache missing: {p}; run scripts/three_topology_fe_v2/"
                    f"build_design_cache.py or set THREE_TOPO_USE_SHARED=0")
        if verify:
            self._verify()
        with np.load(self.geometry_path, allow_pickle=False) as z:
            self.geometry = {k: z[k] for k in z.files}
        node_pos = {int(n): i for i, n in enumerate(self.geometry["node_ids"])}
        self.geometry["beam_node_index"] = np.asarray(
            [[node_pos[int(n)] for n in row] for row in self.geometry["beam_node_ids"]],
            dtype=np.int64)
        self.geometry["plate_node_index"] = np.asarray(
            [[node_pos[int(n)] for n in row] for row in self.geometry["plate_node_ids"]],
            dtype=np.int64)
        with np.load(self.design_path, allow_pickle=False) as z:
            self.design_arrays = {k: z[k] for k in z.files}
        self.design_ids = [str(x) for x in self.design_arrays["design_ids"]]
        self.design_index = {d: i for i, d in enumerate(self.design_ids)}
        with np.load(self.loads_path, allow_pickle=False) as z:
            field_ids = [str(x) for x in z["field_ids"]]
            loads = z["loads"]
        self.field_index = {f: i for i, f in enumerate(field_ids)}
        self.field_loads = loads
        self.field_meta = {}
        with self.fields_csv.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.field_meta[row["field_id"]] = row
        max_shard = int(os.environ.get("THREE_TOPO_SHARD_CACHE", "32"))
        self.reader = reader or FeV2Reader(index, max_runs=max_shard, verify_hash=True)
        self.reader.max_runs = max(1, max_shard)
        self.n_assemble = 0

    def _verify(self) -> None:
        meta = json.loads(self.design_meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != "qt-fe-v2-design-cache-v1":
            raise SharedCacheError(f"unexpected design cache schema: {meta.get('schema')}")
        if _sha(self.design_path) != meta["sha256"]:
            raise SharedCacheError(f"design cache hash mismatch: {self.design_path}")

    def design(self, design_id: str) -> Dict[str, Any]:
        try:
            i = self.design_index[design_id]
        except KeyError as exc:
            raise SharedCacheError(f"design {design_id!r} missing from {self.design_path}") from exc
        return {
            "beam_raw": self.design_arrays["beam_raw"][i],
            "plate_raw": self.design_arrays["plate_raw"][i],
            "beam_end_prop_used": self.design_arrays["beam_end_prop_used"][i],
            "beam_end_mask": self.design_arrays["beam_end_mask"][i],
            "beam_sec_id": self.design_arrays["beam_sec_id"][i],
            "beam_mat_id": self.design_arrays["beam_mat_id"][i],
            "n_equations": int(self.design_arrays["n_equations"][i]),
        }

    def field(self, field_id: str) -> Dict[str, Any]:
        try:
            i = self.field_index[field_id]
        except KeyError as exc:
            raise SharedCacheError(f"field {field_id!r} missing from {self.loads_path}") from exc
        return {"loads": self.field_loads[i], "meta": self.field_meta.get(field_id, {})}

    def assemble(self, sample_id: str) -> Dict[str, Any]:
        row = self.index.row(sample_id)
        if row["topology_token"] != self.topology:
            raise SharedCacheError(
                f"{sample_id}: topology {row['topology_token']} != cache {self.topology}")
        design = self.design(row["design_id"])
        field = self.field(row["field_id"])
        resp = self.reader.read_response(sample_id)
        meta = field["meta"] or resp["meta"]
        self.n_assemble += 1
        return assemble.assemble_raw_shared({
            "row": row, "meta": meta, "geometry": self.geometry,
            "design": design, "field": field, "arrays": resp["arrays"],
            "field_index": resp["field_index"],
        })

    def stats(self) -> Dict[str, Any]:
        design_bytes = sum(int(a.nbytes) for a in self.design_arrays.values())
        geometry_bytes = sum(int(a.nbytes) for a in self.geometry.values())
        loads_bytes = int(self.field_loads.nbytes)
        return {
            "topology": self.topology,
            "design_cache_bytes": design_bytes,
            "geometry_bytes": geometry_bytes,
            "field_loads_bytes": loads_bytes,
            "total_bytes_excluding_shards": design_bytes + geometry_bytes + loads_bytes,
            "shard_cache_max_runs": self.reader.max_runs,
            "shard_hits": self.reader.shard_hits,
            "shard_misses": self.reader.shard_misses,
            "n_assemble": self.n_assemble,
        }

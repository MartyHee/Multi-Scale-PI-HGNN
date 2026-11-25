"""Read-only sample reader for the compact FE v2 package.

The package stores a 16-case response shard per source run.  This reader caches
one or two *source runs* (extraction + shard + loads), never the full response
set.  It verifies the extraction/shard hashes and maps field_id to the shard
row explicitly.
"""
from __future__ import annotations

import gzip
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import index as index_mod


class FeV2Reader:
    def __init__(self, idx: index_mod.FeV2Index, max_runs: int = 2, verify_hash: bool = True):
        self.idx = idx
        self.root = idx.root
        self.max_runs = max(1, int(max_runs))
        self.verify_hash = verify_hash
        self._cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self._shard_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.shard_hits = 0
        self.shard_misses = 0

    def _path(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            raise ValueError(f"operational path must be package-relative: {rel}")
        out = (self.root / p).resolve()
        if not out.is_relative_to(self.root):
            raise ValueError(f"path escapes package root: {rel}")
        return out

    def _load_run(self, row: Dict[str, Any]) -> Dict[str, Any]:
        key = (row["extraction_path"], row["response_shard"])
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        ex_path = self._path(row["extraction_path"])
        raw = gzip.decompress(ex_path.read_bytes()) if ex_path.suffix == ".gz" else ex_path.read_bytes()
        if self.verify_hash:
            if "extraction_sha256" in row and index_mod.sha256_bytes(raw) != row["extraction_sha256"]:
                raise ValueError(f"{row['sample_id']}: extraction content hash mismatch")
            if "extraction_gzip_sha256" in row and index_mod.sha256_file(ex_path) != row["extraction_gzip_sha256"]:
                raise ValueError(f"{row['sample_id']}: extraction gzip hash mismatch")
        ex = json.loads(raw.decode("utf-8"))
        shard_path = self._path(row["response_shard"])
        compressed = shard_path.read_bytes()
        if self.verify_hash and index_mod.sha256_bytes(compressed) != row["response_shard_sha256"]:
            raise ValueError(f"{row['sample_id']}: response shard hash mismatch")
        with np.load(shard_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for arr in arrays.values():
            arr.setflags(write=False)
        loads = json.loads(self._path(row["loads_path"]).read_text(encoding="utf-8"))["rows"]
        rec = {"row": row, "extraction": ex, "arrays": arrays, "loads": loads,
               "field_ids": arrays["field_ids"].tolist(), "node_ids": arrays["node_id"],
               "beam_ids": arrays["beam_id"]}
        self._cache[key] = rec
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_runs:
            self._cache.popitem(last=False)
        return rec

    def _load_shard(self, row: Dict[str, Any]) -> Dict[str, Any]:
        key = row["response_shard"]
        if key in self._shard_cache:
            self.shard_hits += 1
            self._shard_cache.move_to_end(key)
            return self._shard_cache[key]
        self.shard_misses += 1
        shard_path = self._path(row["response_shard"])
        compressed = shard_path.read_bytes()
        if self.verify_hash and index_mod.sha256_bytes(compressed) != row["response_shard_sha256"]:
            raise ValueError(f"{row['sample_id']}: response shard hash mismatch")
        with np.load(shard_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for arr in arrays.values():
            arr.setflags(write=False)
        rec = {"arrays": arrays, "field_ids": arrays["field_ids"].tolist()}
        self._shard_cache[key] = rec
        self._shard_cache.move_to_end(key)
        while len(self._shard_cache) > self.max_runs:
            self._shard_cache.popitem(last=False)
        return rec

    def read_response(self, sample_id: str) -> Dict[str, Any]:
        """Read only the response shard row plus index metadata (fast path)."""
        row = self.idx.row(sample_id)
        shard = self._load_shard(row)
        fid = row["field_id"]
        try:
            i = shard["field_ids"].index(fid)
        except ValueError as exc:
            raise ValueError(f"{sample_id}: field_id {fid} not in shard {shard['field_ids']}") from exc
        if int(i) != int(row["shard_case_index"]):
            raise ValueError(f"{sample_id}: field/shard index mismatch "
                             f"{i} != {row['shard_case_index']}")
        return {"row": row, "meta": self.idx.field_meta(row["topology_token"], fid),
                "arrays": shard["arrays"], "field_index": int(i)}

    def read(self, sample_id: str) -> Dict[str, Any]:
        row = self.idx.row(sample_id)
        run = self._load_run(row)
        fid = row["field_id"]
        try:
            i = run["field_ids"].index(fid)
        except ValueError as exc:
            raise ValueError(f"{sample_id}: field_id {fid} not in shard {run['field_ids']}") from exc
        if int(i) != int(row["shard_case_index"]):
            raise ValueError(f"{sample_id}: field/shard index mismatch {i} != {row['shard_case_index']}")
        meta = self.idx.field_meta(row["topology_token"], fid)
        return {
            "row": row, "meta": meta, "extraction": run["extraction"],
            "arrays": run["arrays"], "loads": run["loads"], "field_index": int(i),
            "node_ids": run["node_ids"], "beam_ids": run["beam_ids"],
        }

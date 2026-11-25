"""Index/manifest loading for the 9,216 FE v2 package.

Only the source index and small registries are read here.  No response shard is
opened by index validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


class FeV2Index:
    def __init__(self, root: Optional[Path] = None, verify_index_hash: bool = True):
        self.root = Path(root).resolve() if root else paths.PACKAGE_ROOT
        self.manifest = json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "ready":
            raise RuntimeError("FE v2 package status is not ready")
        raw = (self.root / "samples.jsonl").read_bytes()
        self.samples_sha256 = sha256_bytes(raw)
        if verify_index_hash and self.samples_sha256 != self.manifest["samples_sha256"]:
            raise ValueError("samples.jsonl hash != manifest samples_sha256")
        self.rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        self.by_id = {r["sample_id"]: r for r in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("duplicate sample_id in samples.jsonl")
        self.fields = self._load_jsonl(self.manifest["fields_registry"])
        self.designs = self._load_jsonl(self.manifest["designs_registry"])
        self.field_by_key = {(r["topology_token"], r["field_id"]): r for r in self.fields}
        self.design_by_key = {(r["topology_token"], r["design_id"]): r for r in self.designs}
        self.holdout = json.loads((self.root / self.manifest["provisional_holdout"]).read_text(encoding="utf-8"))

    def _load_jsonl(self, rel: str) -> List[Dict[str, Any]]:
        p = self.root / rel
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def row(self, sample_id: str) -> Dict[str, Any]:
        return self.by_id[sample_id]

    def field_meta(self, topology_token: str, field_id: str) -> Dict[str, Any]:
        return self.field_by_key[(topology_token, field_id)]

    def validate_identity(self) -> Dict[str, Any]:
        ids = [r["sample_id"] for r in self.rows]
        cells = {(r["topology_token"], r["design_id"], r["field_id"]) for r in self.rows}
        if len(ids) != 9216 or len(set(ids)) != 9216 or len(cells) != 9216:
            raise ValueError(f"index counts: rows={len(ids)} unique_id={len(set(ids))} unique_cell={len(cells)}")
        by_topo = {}
        for t in ("topology1", "topology2", "topology3"):
            rs = [r for r in self.rows if r["topology_token"] == t]
            by_topo[t] = {
                "samples": len(rs),
                "designs": len({r["design_id"] for r in rs}),
                "fields": len({r["field_id"] for r in rs}),
                "c0": sorted({r["c0_hash"] for r in rs}),
            }
            if (len(rs), by_topo[t]["designs"], by_topo[t]["fields"], len(by_topo[t]["c0"])) != (3072, 48, 64, 1):
                raise ValueError(f"{t} count/C0 mismatch: {by_topo[t]}")
        return {"samples": len(self.rows), "unique_cells": len(cells), "by_topology": by_topo,
                "samples_sha256": self.samples_sha256,
                "manifest_sha256": sha256_file(self.root / "dataset_manifest.json")}


def load_index(root: Optional[Path] = None, verify_index_hash: bool = True) -> FeV2Index:
    return FeV2Index(root=root, verify_index_hash=verify_index_hash)

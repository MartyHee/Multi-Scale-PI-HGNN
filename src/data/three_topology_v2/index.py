"""Streaming reader + integrity checks for the three-topology sample index.

Only the unified index ``samples.jsonl`` is a valid sample source.  No scanning
of raw responses and no use of the old ``qtdata`` addresses.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import paths

EXPECTED_SAMPLES = 432
EXPECTED_TOPOLOGIES = 3
EXPECTED_DESIGNS_PER_TOPO = 24
EXPECTED_FIELDS_PER_TOPO = 6

NODE_FEATURE_DIM = 15
NODE_LABEL_DIM = 6
BEAM_FEATURE_DIM = 15
BEAM_LABEL_DIM = 12
PLATE_FEATURE_DIM = 6
LINK_ATTR_DIM = 10

TOPOLOGY_TOKENS = ("topology1", "topology2", "topology3")


class IntegrityError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise IntegrityError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_index(
    root: Optional[Path] = None,
    index_file: Optional[Path] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Read the unified index and run identity / coverage / path checks.

    Returns a dict with ``rows`` (list), ``row_by_id``, ``coverage`` and
    ``hashes``.  Raises ``IntegrityError`` on any violation when ``strict``.
    """
    root = root or paths.package_root()
    index_file = index_file or paths.index_path(root)
    manifest_file = paths.manifest_path(root)

    if not index_file.is_file():
        _fail(f"index not found: {index_file}")
    rows = read_jsonl(index_file)

    errors: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(len(rows) == EXPECTED_SAMPLES, f"sample count {len(rows)} != {EXPECTED_SAMPLES}")

    id_seen: Counter = Counter()
    key_seen: Counter = Counter()
    triple_seen: Counter = Counter()
    per_topo_designs: Dict[str, set] = defaultdict(set)
    per_topo_fields: Dict[str, set] = defaultdict(set)
    per_topo_rows: Counter = Counter()
    topo_labels: Dict[str, str] = {}

    for i, row in enumerate(rows):
        sid = row.get("sample_id")
        if not sid:
            errors.append(f"row {i}: missing sample_id")
            continue
        id_seen[sid] += 1
        key_seen[row.get("sample_key")] += 1
        topo = row.get("topology_token")
        check(topo in TOPOLOGY_TOKENS, f"{sid}: bad topology_token {topo!r}")
        design = row.get("design_id")
        lfh = row.get("load_field_hash")
        triple_seen[(row.get("topology_id"), design, lfh)] += 1
        per_topo_designs[topo].add(design)
        per_topo_fields[topo].add(lfh)
        per_topo_rows[topo] += 1
        topo_labels[topo] = row.get("topology_label", "")

        # path resolution relative to package root
        for field in ("graph_path", "extraction_path", "run_manifest_path"):
            rel = row.get(field)
            check(bool(rel), f"{sid}: missing {field}")
            if rel:
                p = paths.resolve_under_root(rel, root)
                check(p.is_file(), f"{sid}: {field} not found under root: {p}")

        # sha256 fields present and well-formed
        for field in ("graph_sha256", "extraction_sha256", "run_manifest_sha256"):
            v = row.get(field)
            check(isinstance(v, str) and len(v) == 64, f"{sid}: bad {field}")

        cnt = row.get("counts") or {}
        check(int(cnt.get("node_features", -1)) == NODE_FEATURE_DIM, f"{sid}: node_features != 15")
        check(int(cnt.get("node_labels", -1)) == NODE_LABEL_DIM, f"{sid}: node_labels != 6")
        check(int(cnt.get("beam_features", -1)) == BEAM_FEATURE_DIM, f"{sid}: beam_features != 15")
        check(int(cnt.get("beam_labels", -1)) == BEAM_LABEL_DIM, f"{sid}: beam_labels != 12")
        check(int(cnt.get("plate_features", -1)) == PLATE_FEATURE_DIM, f"{sid}: plate_features != 6")
        check(int(cnt.get("link_features", -1)) == LINK_ATTR_DIM, f"{sid}: link_features != 10")

        c = row.get("coordinate_system")
        expected_cs = {"topology1": "Global", "topology2": "Nodal", "topology3": "Nodal"}.get(topo)
        check(c == expected_cs, f"{sid}: coordinate_system {c!r} != {expected_cs!r}")

    dups = [s for s, n in id_seen.items() if n > 1]
    check(not dups, f"duplicate sample_id(s): {dups[:5]}")
    dups_k = [s for s, n in key_seen.items() if n > 1]
    check(not dups_k, f"duplicate sample_key(s): {dups_k[:5]}")
    dups_t = [k for k, n in triple_seen.items() if n > 1]
    check(not dups_t, f"duplicate (topology_id, design_id, load_field_hash): {dups_t[:5]}")

    for topo in TOPOLOGY_TOKENS:
        check(per_topo_rows[topo] == EXPECTED_DESIGNS_PER_TOPO * EXPECTED_FIELDS_PER_TOPO,
              f"{topo}: {per_topo_rows[topo]} rows != {EXPECTED_DESIGNS_PER_TOPO*EXPECTED_FIELDS_PER_TOPO}")
        check(len(per_topo_designs[topo]) == EXPECTED_DESIGNS_PER_TOPO,
              f"{topo}: {len(per_topo_designs[topo])} designs != {EXPECTED_DESIGNS_PER_TOPO}")
        check(len(per_topo_fields[topo]) == EXPECTED_FIELDS_PER_TOPO,
              f"{topo}: {len(per_topo_fields[topo])} fields != {EXPECTED_FIELDS_PER_TOPO}")

    # every (topology, design) cell must have exactly 6 distinct load fields
    cell_counts: Counter = Counter()
    for row in rows:
        cell_counts[(row.get("topology_token"), row.get("design_id"))] += 1
    bad_cells = [k for k, n in cell_counts.items() if n != EXPECTED_FIELDS_PER_TOPO]
    check(not bad_cells, f"design cells without exactly 6 fields: {bad_cells[:5]}")

    result = {
        "rows": rows,
        "row_by_id": {r["sample_id"]: r for r in rows if "sample_id" in r},
        "coverage": {
            "samples": len(rows),
            "topologies": {t: {"rows": per_topo_rows[t],
                               "designs": len(per_topo_designs[t]),
                               "fields": len(per_topo_fields[t]),
                               "label": topo_labels[t]}
                           for t in TOPOLOGY_TOKENS},
            "cells": len(cell_counts),
        },
        "hashes": {
            "samples_jsonl": sha256_file(index_file),
            "dataset_manifest": sha256_file(manifest_file),
        },
        "index_path": str(index_file),
        "root": str(root),
    }
    if strict and errors:
        raise IntegrityError("index integrity failed:\n  " + "\n  ".join(errors[:50]))
    result["errors"] = errors
    return result


def verify_graph_hashes(
    index: Dict[str, Any],
    sample_ids: Optional[Iterable[str]] = None,
    chunk_reader: bool = True,
) -> Dict[str, Any]:
    """Verify on-disk ``graph_sha256`` for the selected (or all) samples."""
    root = Path(index["root"])
    row_by_id = index["row_by_id"]
    ids = list(sample_ids) if sample_ids is not None else list(row_by_id)
    mismatches = []
    missing = []
    for sid in ids:
        row = row_by_id[sid]
        p = paths.resolve_under_root(row["graph_path"], root)
        if not p.is_file():
            missing.append(sid)
            continue
        digest = sha256_file(p)
        if digest != row["graph_sha256"]:
            mismatches.append({"sample_id": sid, "expected": row["graph_sha256"], "actual": digest})
    return {"checked": len(ids), "mismatches": mismatches, "missing": missing,
            "ok": not mismatches and not missing}

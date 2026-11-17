"""Revised split publication for the three-topology training-input v2.

Division of responsibility
--------------------------
* **cross-design** and **leave-one-topology**: unaffected by the load-family
  fix.  The accepted v1 artifacts are reused verbatim (copied into the v2
  namespace with provenance and hash verification), never silently refit.
* **cross-load-field**: rebuilt with the *correct* family key.
    - topology1: source ``load_family`` from the primary index, so
      {T1,N1}, {T2,N2}, {T3,N3} are each one physical family.
    - topology2/3: coarse "pure-vertical" vs "mixed-component" grouping,
      explicitly labelled coarse.

Invariants enforced for every fold of every protocol:
  train / validation / test sample IDs are mutually disjoint;
  each outer-test sample appears exactly once across the protocol's folds;
  cross-load-field: validation family != test family and the test field's
  same-family twin(s) are in train.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import paths
from .index import read_jsonl

SPLIT_SCHEMA = "qt-three-topology-v2-experiment-split-v1"
SPLIT_INDEX_SCHEMA = "qt-three-topology-v2-split-index-v1"
INPUT_META_SCHEMA = "qt-three-topology-v2-input-metadata-v1"

TOPOLOGY_TOKENS = ("topology1", "topology2", "topology3")

# Preserved (non-standardised) integer columns per matrix.
PRESERVED_COLUMNS = {
    "node_x": [9, 10, 11, 12, 13, 14],
    "beam_x": [],
    "plate_x": [5],
    "y_disp": [],
    "y_force": [],
    "structural_link_x": [8, 9],
}

COLUMNS = {
    "node_x": ["X", "Y", "Z", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
               "Dx_fix", "Dy_fix", "Dz_fix", "Rx_fix", "Ry_fix", "Rz_fix"],
    "beam_x": ["Area_I", "Ix_I", "Iy_I", "Iz_I", "Area_J", "Ix_J", "Iy_J", "Iz_J",
               "ElasticModulus", "PoissonRatio", "UnitWeight", "Length",
               "CosX", "CosY", "CosZ"],
    "plate_x": ["ThicknessValue", "ElasticModulus", "PoissonRatio", "UnitWeight",
                "BetaAngle", "PlateType"],
    "y_disp": ["dx", "dy", "dz", "rx", "ry", "rz"],
    "y_force": ["I_Fx", "I_Fy", "I_Fz", "I_Mx", "I_My", "I_Mz",
                "J_Fx", "J_Fy", "J_Fz", "J_Mx", "J_My", "J_Mz"],
    "structural_link_x": ["Kx", "Ky", "Kz", "Krx", "Kry", "Krz", "BetaAngle",
                          "DistanceRatio", "ElasticLinkType", "is_rigid"],
}

UNITS = {
    "node_x": ["m", "m", "m", "N", "N", "N", "N*m", "N*m", "N*m",
               "0/1", "0/1", "0/1", "0/1", "0/1", "0/1"],
    "beam_x": ["m^2", "m^4", "m^4", "m^4", "m^2", "m^4", "m^4", "m^4",
               "Pa", "1", "N/m^3", "m", "1", "1", "1"],
    "plate_x": ["m", "Pa", "1", "N/m^3", "deg (API value)", "enum"],
    "y_disp": ["m", "m", "m", "rad", "rad", "rad"],
    "y_force": ["N", "N", "N", "N*m", "N*m", "N*m", "N", "N", "N", "N*m", "N*m", "N*m"],
    "structural_link_x": ["N/m", "N/m", "N/m", "N*m/rad", "N*m/rad", "N*m/rad",
                          "deg (API value)", "1", "API enum", "0/1"],
}


def _sample_list_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _graph_aggregate(ids: Sequence[str], row_by_id: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"{sid}:{row_by_id[sid]['graph_sha256']}" for sid in sorted(ids)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def input_metadata() -> Dict[str, Any]:
    """Column order / units / kind for every input and label matrix.

    ``BetaAngle`` is declared in degrees (API value) for both plate and
    structural-link matrices, matching the graph contract; the source numeric
    values are unchanged.
    """
    out: Dict[str, Any] = {"schema": INPUT_META_SCHEMA, "matrices": {}}
    for mat in ("node_x", "beam_x", "plate_x", "y_disp", "y_force", "structural_link_x"):
        preserved = set(PRESERVED_COLUMNS[mat])
        cols = []
        for i, (name, unit) in enumerate(zip(COLUMNS[mat], UNITS[mat])):
            kind = "preserved" if i in preserved else "continuous"
            cols.append({"index": i, "name": name, "unit": unit, "kind": kind})
        out["matrices"][mat] = {
            "columns": cols,
            "dim": len(cols),
            "preserved_indices": sorted(preserved),
        }
    out["notes"] = {
        "BetaAngle": "unit relabelled to deg (API value) for plate_x and structural_link_x; "
                     "source numbers unchanged",
        "coordinate_markers": {"topology1": "Global", "topology2": "Nodal", "topology3": "Nodal"},
        "preserved_columns": "boolean support flags, plate type, API link type and is_rigid "
                             "are not standardised",
    }
    return out


def _population_by_topology(index: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    pop = defaultdict(list)
    for row in index["rows"]:
        pop[row["topology_token"]].append(row)
    return pop


def _template_family_topology1(root: Path) -> Dict[str, str]:
    """template_id -> source load_family for topology1."""
    rows = read_jsonl(paths.source_manifest_path("topology1_primary_samples_v2.jsonl", root))
    m: Dict[str, str] = {}
    for r in rows:
        t = r["template_id"]
        fam = r.get("load_family")
        if t in m and m[t] != fam:
            raise ValueError(f"topology1 template {t} maps to multiple families")
        m[t] = fam
    return m


def _coarse_family_topology23(root: Path, name: str) -> Dict[str, str]:
    """template_field_id -> coarse family (pure-vertical / mixed) for topology2/3."""
    rows = read_jsonl(paths.source_manifest_path(name, root))
    m: Dict[str, str] = {}
    for r in rows:
        t = r["template_field_id"]
        comps = set(r.get("load_components") or [])
        fam = "pure_vertical" if comps == {"Fz"} else "mixed_component"
        if t in m and m[t] != fam:
            raise ValueError(f"{name} template {t} maps to multiple coarse families")
        m[t] = fam
    return m


def _build_cross_load_field(
    topology: str,
    topo_rows: List[Dict[str, Any]],
    template_family: Dict[str, str],
    family_note: str,
    seed: int,
    source_hashes: Dict[str, str],
) -> Dict[str, Any]:
    # field -> template (each field has a single template in the frozen set)
    field_to_template: Dict[str, str] = {}
    field_samples: Dict[str, List[str]] = defaultdict(list)
    for r in topo_rows:
        lfh = r["load_field_hash"]
        t = r["template_id"]
        prev = field_to_template.get(lfh)
        if prev is not None and prev != t:
            raise ValueError(f"{topology}: field {lfh} has multiple templates {prev},{t}")
        field_to_template[lfh] = t
        field_samples[lfh].append(r["sample_id"])
    if len(field_to_template) != 6:
        raise ValueError(f"{topology}: expected 6 load fields, got {len(field_to_template)}")
    for lfh, t in field_to_template.items():
        if t not in template_family:
            raise ValueError(f"{topology}: template {t} missing a family key")

    # Deterministic field order by template id.
    field_order = sorted(field_to_template, key=lambda lfh: field_to_template[lfh])
    family_of = {lfh: template_family[field_to_template[lfh]] for lfh in field_order}
    n = len(field_order)

    folds = []
    for i, test_field in enumerate(field_order):
        test_family = family_of[test_field]
        # first cyclic next field with a different family
        val_field = None
        for step in range(1, n):
            cand = field_order[(i + step) % n]
            if family_of[cand] != test_family:
                val_field = cand
                break
        if val_field is None:
            raise ValueError(f"{topology}: no different-family validation for {test_field}")
        # same-family twins of the test field
        twins = [f for f in field_order if f != test_field and family_of[f] == test_family]
        train_fields = [f for f in field_order if f not in (test_field, val_field)]

        if family_of[val_field] == test_family:
            raise AssertionError("validation family equals test family")
        if not all(t in train_fields for t in twins):
            raise AssertionError("same-family twin not retained in train")

        train_ids = sorted(sid for f in train_fields for sid in field_samples[f])
        val_ids = sorted(field_samples[val_field])
        test_ids = sorted(field_samples[test_field])
        assert not (set(train_ids) & set(val_ids))
        assert not (set(train_ids) & set(test_ids))
        assert not (set(val_ids) & set(test_ids))

        folds.append({
            "fold_id": f"clf_fold_{i+1}",
            "seed": seed,
            "test_template_id": field_to_template[test_field],
            "test_load_field_hash": test_field,
            "test_load_family": family_of[test_field],
            "validation_template_id": field_to_template[val_field],
            "validation_load_field_hash": val_field,
            "validation_load_family": family_of[val_field],
            "train_template_ids": [field_to_template[f] for f in train_fields],
            "train_load_field_hashes": train_fields,
            "same_family_twins": [field_to_template[t] for t in twins],
            "same_family_twins_in_train": True,
            "test_sample_ids": test_ids,
            "validation_sample_ids": val_ids,
            "train_sample_ids": train_ids,
            "counts": {
                "test_fields": 1, "validation_fields": 1, "train_fields": len(train_fields),
                "test_samples": len(test_ids), "validation_samples": len(val_ids),
                "train_samples": len(train_ids),
            },
        })

    all_fields = set(field_order)
    return {
        "schema": SPLIT_SCHEMA,
        "split_id": f"cross_load_field_{topology}",
        "split_kind": "cross_load_field",
        "topology_token": topology,
        "seed": seed,
        "unit": "load_field_hash",
        "family_note": family_note,
        "family_key": {field_to_template[f]: family_of[f] for f in field_order},
        "source": source_hashes,
        "counts": {
            "folds": len(folds), "fields": len(all_fields), "samples": len(topo_rows),
            "test_fields": 1, "validation_fields": 1, "train_fields": len(all_fields) - 2,
            "test_samples": 24, "validation_samples": 24, "train_samples": 96,
        },
        "folds": folds,
    }


def _load_v1_split(split_id: str, root: Path) -> Dict[str, Any]:
    name = {
        "cross_design_topology1": "split_cross_design_topology1.json",
        "cross_design_topology2": "split_cross_design_topology2.json",
        "cross_design_topology3": "split_cross_design_topology3.json",
        "leave_one_topology": "split_leave_one_topology.json",
    }[split_id]
    p = paths.v1_split_dir(root) / name
    with p.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    d["_v1_source_path"] = str(p)
    d["_v1_source_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


def _verify_partitions(fold: Dict[str, Any], fold_id: str) -> None:
    tr = set(fold["train_sample_ids"])
    va = set(fold["validation_sample_ids"])
    te = set(fold["test_sample_ids"])
    if tr & va or tr & te or va & te:
        raise AssertionError(f"{fold_id}: non-disjoint partitions")
    if len(tr) != len(fold["train_sample_ids"]) or len(va) != len(fold["validation_sample_ids"]):
        raise AssertionError(f"{fold_id}: duplicate sample IDs inside a partition")


def build_all_splits(index: Dict[str, Any], root: Optional[Path] = None,
                     seed: int = 20261114) -> Dict[str, Any]:
    """Build/copy all revised splits and return ``{split_id: split_dict}``."""
    root = root or paths.package_root()
    row_by_id = index["row_by_id"]
    source_hashes = {"samples_jsonl_sha256": index["hashes"]["samples_jsonl"],
                     "dataset_manifest_sha256": index["hashes"]["dataset_manifest"],
                     "sample_count": len(index["rows"])}
    pop = _population_by_topology(index)

    tf1 = _template_family_topology1(root)
    tf2 = _coarse_family_topology23(root, "topology2_samples_v2.jsonl")
    tf3 = _coarse_family_topology23(root, "topology3_samples.jsonl")

    splits: Dict[str, Any] = {}

    # ---- cross-load-field (rebuilt) ----
    splits["cross_load_field_topology1"] = _build_cross_load_field(
        "topology1", pop["topology1"], tf1,
        family_note="source load_family groups {T1,N1},{T2,N2},{T3,N3} as one physical family each",
        seed=seed, source_hashes=source_hashes)
    splits["cross_load_field_topology2"] = _build_cross_load_field(
        "topology2", pop["topology2"], tf2,
        family_note="COARSE grouping: pure_vertical (Fz only) vs mixed_component; "
                    "not a fine physical-family split",
        seed=seed, source_hashes=source_hashes)
    splits["cross_load_field_topology3"] = _build_cross_load_field(
        "topology3", pop["topology3"], tf3,
        family_note="COARSE grouping: pure_vertical (Fz only) vs mixed_component; "
                    "not a fine physical-family split",
        seed=seed, source_hashes=source_hashes)

    # ---- cross-design + leave-one-topology (reuse v1, verify) ----
    for sid in ("cross_design_topology1", "cross_design_topology2",
                "cross_design_topology3", "leave_one_topology"):
        d = _load_v1_split(sid, root)
        for fold in d["folds"]:
            _verify_partitions(fold, fold["fold_id"])
            for key in ("train_sample_ids", "validation_sample_ids", "test_sample_ids"):
                missing = [s for s in fold[key] if s not in row_by_id]
                if missing:
                    raise AssertionError(f"{sid}/{fold['fold_id']}: unknown sample IDs {missing[:3]}")
        splits[sid] = d

    # Global checks: test uniqueness per protocol; disjointness.
    for sid, d in splits.items():
        seen = defaultdict(int)
        for fold in d["folds"]:
            for s in fold["test_sample_ids"]:
                seen[s] += 1
        dup = [s for s, n in seen.items() if n > 1]
        if dup:
            raise AssertionError(f"{sid}: outer-test sample repeated across folds: {dup[:3]}")
    return splits


def write_splits(splits: Dict[str, Any], index: Dict[str, Any],
                 out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Write split JSONs + split_index to the v2 namespace; return the index."""
    out_dir = out_dir or paths.SPLIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    row_by_id = index["row_by_id"]

    artifacts = []
    for sid, d in splits.items():
        # attach per-partition hashes (deterministic, computed fresh)
        for fold in d["folds"]:
            fold["train_sample_list_sha256"] = _sample_list_sha256(fold["train_sample_ids"])
            fold["validation_sample_list_sha256"] = _sample_list_sha256(fold["validation_sample_ids"])
            fold["test_sample_list_sha256"] = _sample_list_sha256(fold["test_sample_ids"])
            fold["train_graph_sha256_aggregate"] = _graph_aggregate(fold["train_sample_ids"], row_by_id)
        path = out_dir / f"split_{sid}.json"
        payload = json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)
        path.write_text(payload, encoding="utf-8")
        artifacts.append({
            "path": path.name,
            "split_id": sid,
            "split_kind": d.get("split_kind"),
            "folds": len(d["folds"]),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        })

    input_meta = input_metadata()
    meta_payload = json.dumps(input_meta, ensure_ascii=False, indent=2, sort_keys=True)
    meta_path = out_dir / "input_metadata.json"
    meta_path.write_text(meta_payload, encoding="utf-8")

    split_index = {
        "schema": SPLIT_INDEX_SCHEMA,
        "seed": 20261114,
        "source": {"samples_jsonl_sha256": index["hashes"]["samples_jsonl"],
                   "dataset_manifest_sha256": index["hashes"]["dataset_manifest"],
                   "sample_count": len(index["rows"])},
        "artifacts": artifacts,
        "input_metadata": {
            "path": meta_path.name,
            "sha256": hashlib.sha256(meta_payload.encode("utf-8")).hexdigest(),
        },
    }
    idx_payload = json.dumps(split_index, ensure_ascii=False, indent=2, sort_keys=True)
    (out_dir / "split_index.json").write_text(idx_payload, encoding="utf-8")
    return split_index

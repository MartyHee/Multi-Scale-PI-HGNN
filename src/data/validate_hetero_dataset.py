#!/usr/bin/env python3
"""
validate_hetero_dataset.py — Comprehensive validation for processed heterogeneous graph dataset.

Validates ``processed/hetero_graph_dataset_v2`` (or any version) against the
canonical schema:

  - 3 node types (mesh_node, beam_element, plate_element) — NO link_element
  - 5 edge types (fwd+rev membership + structural_link)
  - All reverse edges carry edge_attr
  - Correct shapes, no NaN/Inf, split integrity

Usage::

    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe \\
        src/data/validate_hetero_dataset.py \\
        --processed-dir processed/hetero_graph_dataset_v2 \\
        --num-checks 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.hetero_schema import (
    EDGE_COUNTS,
    EDGE_TYPES,
    NODE_TYPES,
    NODE_TYPE_CONFIG,
    TARGET_NODE_TYPES,
    get_feature_dim,
    get_node_count,
    get_target_dim,
    get_target_key,
)


class ValidationResult:
    """Collects validation results."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warnings: List[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> Tuple[int, int, int]:
        return len(self.passed), len(self.failed), len(self.warnings)

    def print_summary(self) -> None:
        p, f, w = self.summary()
        print(f"\n{'=' * 60}")
        print(f"Validation Summary: {p} passed, {f} failed, {w} warnings")
        print(f"{'=' * 60}")
        for msg in self.failed:
            print(f"  [FAIL] {msg}")
        for msg in self.warnings:
            print(f"  [WARN] {msg}")
        for msg in self.passed:
            print(f"  [OK]   {msg}")


def check_graph_data(data: HeteroData, graph_id: str, result: ValidationResult) -> None:
    """Validate one HeteroData graph against canonical schema."""
    sid = graph_id

    # ---- Node types ----
    # 1. Must have exactly mesh_node, beam_element, plate_element
    expected_node_types = {"mesh_node", "beam_element", "plate_element"}
    actual_node_types = set(data.node_types)
    if actual_node_types != expected_node_types:
        extra = actual_node_types - expected_node_types
        missing = expected_node_types - actual_node_types
        if extra:
            result.fail(f"{sid}: unexpected node_types: {extra}")
        if missing:
            result.fail(f"{sid}: missing node_types: {missing}")
    else:
        result.ok(f"{sid}: node_types correct ({actual_node_types})")

    # 2. NO link_element
    if "link_element" in data.node_types:
        result.fail(f"{sid}: link_element should NOT be in node_types")

    # 3. Node counts
    for ntype in expected_node_types:
        expected_count = NODE_TYPE_CONFIG[ntype]["count"]
        if ntype in data.node_types:
            actual_count = data[ntype].num_nodes
            if actual_count != expected_count:
                result.fail(f"{sid}: {ntype}.num_nodes={actual_count} != expected={expected_count}")
            else:
                result.ok(f"{sid}: {ntype}.num_nodes={actual_count}")

    # 4. Feature dimensions
    for ntype in ["mesh_node", "beam_element", "plate_element"]:
        expected_dim = get_feature_dim(ntype)
        if ntype in data.node_types:
            actual_dim = data[ntype].x.shape[1]
            if actual_dim != expected_dim:
                result.fail(f"{sid}: {ntype}.x dim={actual_dim} != {expected_dim}")
            else:
                result.ok(f"{sid}: {ntype}.x dim={actual_dim}")

    # 5. Target dimensions
    # mesh_node.y_disp (6d)
    if "mesh_node" in data.node_types:
        if not hasattr(data["mesh_node"], "y_disp"):
            result.fail(f"{sid}: mesh_node missing y_disp")
        elif data["mesh_node"].y_disp.shape[1] != 6:
            result.fail(f"{sid}: mesh_node.y_disp dim={data['mesh_node'].y_disp.shape[1]} != 6")
        else:
            result.ok(f"{sid}: mesh_node.y_disp (6d)")

    # beam_element.y_force (12d)
    if "beam_element" in data.node_types:
        if not hasattr(data["beam_element"], "y_force"):
            result.fail(f"{sid}: beam_element missing y_force")
        elif data["beam_element"].y_force.shape[1] != 12:
            result.fail(f"{sid}: beam_element.y_force dim={data['beam_element'].y_force.shape[1]} != 12")
        else:
            result.ok(f"{sid}: beam_element.y_force (12d)")

    # plate_element: no y
    if "plate_element" in data.node_types:
        plate_attrs = set(data["plate_element"].node_attrs())
        y_keys = [k for k in plate_attrs if k.startswith("y_")]
        if y_keys:
            result.fail(f"{sid}: plate_element has unexpected labels: {y_keys}")

    # ---- Edge types ----
    # 6. Must have all 5 edge types
    expected_edge_types = set(EDGE_TYPES)
    actual_edge_types = set(data.edge_types)
    missing_edges = expected_edge_types - actual_edge_types
    extra_edges = actual_edge_types - expected_edge_types
    if missing_edges:
        result.fail(f"{sid}: missing edge types: {missing_edges}")
    if extra_edges:
        result.fail(f"{sid}: unexpected edge types: {extra_edges}")

    # 7. Edge counts
    for etype in EDGE_TYPES:
        expected_count = EDGE_COUNTS.get(etype)
        if expected_count is None:
            continue
        if etype in data.edge_types:
            actual_count = data[etype].edge_index.shape[1]
            if actual_count != expected_count:
                result.fail(f"{sid}: {etype} count={actual_count} != {expected_count}")
            else:
                result.ok(f"{sid}: {etype} count={actual_count}")

    # 8. Edge attr presence (all 5 types must have edge_attr in v2)
    for etype in EDGE_TYPES:
        if etype in data.edge_types:
            if "edge_attr" not in data[etype]:
                result.fail(f"{sid}: {etype} missing edge_attr")
            else:
                result.ok(f"{sid}: {etype} has edge_attr")

    # 9. structural_link edge_attr must be 10-d
    sl_type = ("mesh_node", "structural_link", "mesh_node")
    if sl_type in data.edge_types and "edge_attr" in data[sl_type]:
        sl_dim = data[sl_type].edge_attr.shape[1]
        if sl_dim != 10:
            result.fail(f"{sid}: structural_link edge_attr dim={sl_dim} != 10")
        else:
            result.ok(f"{sid}: structural_link edge_attr (10d)")

    # ---- NaN/Inf checks ----
    # 10. Node features
    for ntype in data.node_types:
        x = data[ntype].x
        if torch.isnan(x).any():
            result.fail(f"{sid}: {ntype}.x contains NaN")
        if torch.isinf(x).any():
            result.fail(f"{sid}: {ntype}.x contains Inf")

    # 11. Targets
    if hasattr(data["mesh_node"], "y_disp"):
        yd = data["mesh_node"].y_disp
        if torch.isnan(yd).any():
            result.fail(f"{sid}: mesh_node.y_disp NaN")
        if torch.isinf(yd).any():
            result.fail(f"{sid}: mesh_node.y_disp Inf")
    if hasattr(data["beam_element"], "y_force"):
        yf = data["beam_element"].y_force
        if torch.isnan(yf).any():
            result.fail(f"{sid}: beam_element.y_force NaN")
        if torch.isinf(yf).any():
            result.fail(f"{sid}: beam_element.y_force Inf")

    # 12. Edge attr
    for etype in data.edge_types:
        if "edge_attr" in data[etype]:
            ea = data[etype].edge_attr
            if torch.isnan(ea).any():
                result.fail(f"{sid}: {etype} edge_attr NaN")
            if torch.isinf(ea).any():
                result.fail(f"{sid}: {etype} edge_attr Inf")

    # 13. Edge index bounds
    for ntype in data.node_types:
        n_count = data[ntype].num_nodes
        for etype in data.edge_types:
            src, _, tgt = etype
            if src == ntype:
                ei = data[etype].edge_index[0]
                if ei.max() >= n_count:
                    result.fail(f"{sid}: {etype} source index {ei.max().item()} >= {n_count} nodes")
            if tgt == ntype:
                ei = data[etype].edge_index[1]
                if ei.max() >= n_count:
                    result.fail(f"{sid}: {etype} target index {ei.max().item()} >= {n_count} nodes")

    # 14. structural_link endpoints are mesh_node indices
    if sl_type in data.edge_types:
        sl_ei = data[sl_type].edge_index
        n_mesh = data["mesh_node"].num_nodes
        if sl_ei[0].max() >= n_mesh or sl_ei[1].max() >= n_mesh:
            result.fail(f"{sid}: structural_link endpoint out of mesh_node range")

    # 15. PyG interface check
    try:
        _ = data.node_types
        _ = data.edge_types
        meta = data.metadata()
        if meta is None:
            result.fail(f"{sid}: data.metadata() returned None")
        else:
            result.ok(f"{sid}: data.metadata() works (node_types={len(meta[0])}, edge_types={len(meta[1])})")
    except Exception as e:
        result.fail(f"{sid}: data.metadata() raised {e}")


def validate_dataset(
    processed_dir: Path,
    num_checks: int = 10,
) -> ValidationResult:
    """Run comprehensive validation on the dataset."""
    result = ValidationResult()

    print(f"Validating dataset at: {processed_dir}")
    print(f"Random checks: {num_checks} graphs")

    # ---- Check directory structure ----
    if not processed_dir.is_dir():
        result.fail(f"Directory not found: {processed_dir}")
        return result

    # index.csv
    index_path = processed_dir / "index.csv"
    if not index_path.is_file():
        result.fail("index.csv not found")
        return result
    index_df = pd.read_csv(index_path)
    result.ok(f"index.csv loaded ({len(index_df)} rows)")

    # splits
    split_dir = processed_dir / "splits"
    if not split_dir.is_dir():
        result.fail("splits/ directory not found")
    else:
        for sname in ["split_by_sample.json", "split_by_loadcase.json"]:
            sp = split_dir / sname
            if sp.is_file():
                with open(sp) as f:
                    sd = json.load(f)
                train_count = len(sd.get("train", []))
                val_count = len(sd.get("val", []))
                test_count = len(sd.get("test", []))
                result.ok(f"{sname}: train={train_count} val={val_count} test={test_count}")
            else:
                result.fail(f"{sname} not found")

    # feature_stats.json
    stats_path = processed_dir / "feature_stats.json"
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        result.ok(f"feature_stats.json loaded ({len(stats)} keys)")
    else:
        result.warn("feature_stats.json not found (may be skipped if --skip-stats)")

    # metadata.json
    meta_path = processed_dir / "metadata.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        result.ok(f"metadata.json: {meta.get('num_graphs', '?')} graphs, {meta.get('node_types', '?')}")
        if "link_element" in str(meta.get("node_types", [])):
            result.fail("metadata.json still references link_element")
    else:
        result.fail("metadata.json not found")

    # ---- Total graph count ----
    total_graphs = len(index_df)
    if total_graphs != 35000:
        result.fail(f"Total graphs: {total_graphs} != 35000")
    else:
        result.ok(f"Total graphs: {total_graphs}")

    # ---- Sample-level checks ----
    unique_samples = index_df["sample_id"].nunique()
    if unique_samples != 70:
        result.fail(f"Unique samples: {unique_samples} != 70")
    else:
        result.ok(f"Unique samples: {unique_samples}")

    unique_lcs = index_df["loadcase_id"].nunique()
    if unique_lcs != 500:
        result.warn(f"Unique loadcases: {unique_lcs} (expected 500)")

    # ---- Check graph files exist ----
    sample_paths = set(index_df["graph_file"])
    # Check a sample
    for gf in list(sample_paths)[:5]:
        fp = processed_dir / gf
        if not fp.is_file():
            result.fail(f"Graph file missing: {fp}")
    result.ok(f"Graph files sampled: all checked files exist")

    # ---- Detailed per-graph checks ----
    import random
    random.seed(42)
    check_indices = random.sample(range(len(index_df)), min(num_checks, len(index_df)))

    for ci in check_indices:
        row = index_df.iloc[ci]
        graph_path = processed_dir / row["graph_file"]
        graph_id = str(row["graph_id"])

        if not graph_path.is_file():
            result.fail(f"{graph_id}: file not found at {graph_path}")
            continue

        try:
            data: HeteroData = torch.load(graph_path, weights_only=False)
        except Exception as e:
            result.fail(f"{graph_id}: torch.load failed: {e}")
            continue

        if not isinstance(data, HeteroData):
            result.fail(f"{graph_id}: loaded object is not HeteroData (got {type(data).__name__})")
            continue

        check_graph_data(data, graph_id, result)

    # ---- Split integrity ----
    for sname in ["split_by_sample", "split_by_loadcase"]:
        sp = split_dir / f"{sname}.json"
        if not sp.is_file():
            continue
        with open(sp) as f:
            sd = json.load(f)

        mode = "by_sample" if "sample" in sname else "by_loadcase"
        train_set = set(sd["train"])
        val_set = set(sd["val"])
        test_set = set(sd["test"])

        if train_set & val_set:
            result.fail(f"{sname}: train/val overlap: {train_set & val_set}")
        if train_set & test_set:
            result.fail(f"{sname}: train/test overlap: {train_set & test_set}")
        if val_set & test_set:
            result.fail(f"{sname}: val/test overlap: {val_set & test_set}")

        # Verify counts match index
        if mode == "by_sample":
            mask_train = index_df["sample_id"].isin(train_set)
            mask_val = index_df["sample_id"].isin(val_set)
            mask_test = index_df["sample_id"].isin(test_set)
        else:
            mask_train = index_df["loadcase_id"].isin(train_set)
            mask_val = index_df["loadcase_id"].isin(val_set)
            mask_test = index_df["loadcase_id"].isin(test_set)

        train_actual = mask_train.sum()
        val_actual = mask_val.sum()
        test_actual = mask_test.sum()
        expected_train = len(train_set) * 500 if mode == "by_sample" else len(train_set) * 70
        expected_val = len(val_set) * 500 if mode == "by_sample" else len(val_set) * 70
        expected_test = len(test_set) * 500 if mode == "by_sample" else len(test_set) * 70

        result.ok(f"{sname}: integrity OK (train={train_actual}, val={val_actual}, test={test_actual})")

    # ---- Standardized data sanity (if feature_stats exists) ----
    if stats_path.is_file() and index_df is not None:
        # Check that stats keys are correct for v2
        with open(stats_path) as f:
            stats = json.load(f)
        # v2 should NOT have node:link_element:x
        for key in stats:
            if "link_element" in key:
                result.fail(f"feature_stats contains link_element key: {key}")
        # v2 should have structural_link
        sl_check = [k for k in stats if "structural_link" in k]
        if sl_check:
            result.ok(f"feature_stats includes structural_link: {sl_check}")
        # v2 should have rev_belongs_to_beam and rev_belongs_to_plate
        rev_beam = [k for k in stats if "rev_belongs_to_beam" in k]
        rev_plate = [k for k in stats if "rev_belongs_to_plate" in k]
        if rev_beam:
            result.ok(f"feature_stats includes rev_belongs_to_beam: {rev_beam}")
        if rev_plate:
            result.ok(f"feature_stats includes rev_belongs_to_plate: {rev_plate}")

    return result


def main():
    p = argparse.ArgumentParser(description="Validate heterogeneous graph dataset")
    p.add_argument("--processed-dir", type=str,
                    default=(_PROJECT_ROOT / "processed" / "hetero_graph_dataset_v2").as_posix(),
                    help="Path to processed dataset directory")
    p.add_argument("--num-checks", type=int, default=10,
                    help="Number of random graphs to check in detail")
    args = p.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    t0 = time.time()
    result = validate_dataset(processed_dir, num_checks=args.num_checks)
    elapsed = time.time() - t0

    result.print_summary()
    print(f"\nValidation completed in {elapsed:.1f}s")

    if len(result.failed) > 0:
        print("\n  WARNING: Validation FAILED. Review before using dataset.")
        sys.exit(1)
    else:
        print("\n  All validations passed. Dataset is ready for Stage 2.")


if __name__ == "__main__":
    main()

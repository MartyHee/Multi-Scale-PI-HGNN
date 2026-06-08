#!/usr/bin/env python3
"""
build_graph_dataset.py — Baseline graph dataset builder.

Reads external raw CSV data and produces a standardised, per-graph .pt dataset
inside the model project's processed directory.

Usage::

    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe src/data/build_graph_dataset.py

Phases:
    0. Setup, validation & global mapping construction
    1. Graph construction — one .pt file per (SampleID, LoadCaseId)
    2. Index & metadata serialisation
    3. Split generation (by_sample + by_loadcase)
    4. Training statistics via Welford's online algorithm → feature_stats.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Ensure project root is on sys.path so we can import sibling modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.data.split import generate_all_splits, print_split_summary
from src.data.transforms import NodeEdgeStandardScaler


# ============================================================
# Helpers
# ============================================================

def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths
    cfg["_project_root"] = _PROJECT_ROOT
    cfg["raw_data_dir"] = Path(cfg["raw_data_dir"]).resolve()
    cfg["processed_data_dir"] = (_PROJECT_ROOT / cfg["processed_data_dir"]).resolve()
    return cfg


def get_sample_dirs(raw_dir: Path) -> List[str]:
    """Return sorted sample ID strings from the raw data directory.

    Only includes directories that have the required CSV files.
    """
    samples = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        # quick check: must have at least nodes.csv
        if (entry / "nodes.csv").exists():
            samples.append(entry.name)
    return samples


def verify_topology_across_samples(raw_dir: Path, sample_ids: List[str]) -> Dict[int, Tuple[int, int]]:
    """Verify that all samples share the same element topology.

    Returns the canonical (ElementId -> (INodeId, JNodeId)) mapping.
    """
    ref_df = pd.read_csv(raw_dir / sample_ids[0] / "beam_elements.csv")
    ref_df = ref_df.sort_values("ElementId")
    ref_topo = dict(zip(ref_df["ElementId"], zip(ref_df["INodeId"], ref_df["JNodeId"])))

    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "beam_elements.csv").sort_values("ElementId")
        topo = dict(zip(df["ElementId"], zip(df["INodeId"], df["JNodeId"])))
        if topo != ref_topo:
            mismatches = []
            for eid in ref_topo:
                if ref_topo[eid] != topo.get(eid):
                    mismatches.append((eid, ref_topo[eid], topo.get(eid)))
            raise ValueError(
                f"Topology mismatch in sample {sid}! "
                f"First {min(5, len(mismatches))} mismatches: {mismatches[:5]}"
            )
    print(f"  [ok] Topology verified — all {len(sample_ids)} samples share same element topology.")
    return ref_topo


def verify_nodes_match(raw_dir: Path, sample_ids: List[str]):
    """Verify that nodes.csv is identical across all samples."""
    ref = pd.read_csv(raw_dir / sample_ids[0] / "nodes.csv")
    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "nodes.csv")
        if not df.equals(ref):
            raise ValueError(f"nodes.csv mismatch in sample {sid}!")
    print(f"  [ok] Node coordinates verified — identical across all {len(sample_ids)} samples.")


# ============================================================
# Phase 0: Setup
# ============================================================

def phase0_setup(cfg: dict) -> Tuple[dict, List[str], List[int], torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Validate config, raw data, build global mappings and edge_index.

    Returns:
        cfg, sample_ids, loadcase_ids, edge_index, node_coords_tensor, nodes_df
    """
    print("=" * 60)
    print("Phase 0: Setup & Validation")
    print("=" * 60)

    raw_dir = cfg["raw_data_dir"]
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {raw_dir}")

    sample_ids = get_sample_dirs(raw_dir)
    if not sample_ids:
        raise FileNotFoundError(f"No sample directories found in {raw_dir}")
    print(f"  Found {len(sample_ids)} sample directories: {sample_ids[:3]}...{sample_ids[-3:]}")

    # Verify nodes.csv across samples
    verify_nodes_match(raw_dir, sample_ids)

    # Verify topology across samples
    verify_topology_across_samples(raw_dir, sample_ids)

    # Read canonical nodes from first sample
    nodes_df = pd.read_csv(raw_dir / sample_ids[0] / "nodes.csv")
    if len(nodes_df) != cfg["expected_num_nodes"]:
        raise ValueError(
            f"Expected {cfg['expected_num_nodes']} nodes, got {len(nodes_df)}"
        )
    print(f"  Nodes: {len(nodes_df)} (OK)")

    # Build NodeId -> node_idx mapping (sorted by NodeId)
    node_ids = sorted(nodes_df["NodeId"].tolist())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    assert len(node_id_to_idx) == cfg["expected_num_nodes"]

    # Build node coordinate tensor (same for all graphs)
    node_coords_tensor = torch.tensor(
        nodes_df.set_index("NodeId").loc[node_ids][["X", "Y", "Z"]].values,
        dtype=torch.float32,
    )
    print(f"  Node coords tensor: {node_coords_tensor.shape}")

    # Build edge_index from first sample's beam_elements
    elems_df = pd.read_csv(raw_dir / sample_ids[0] / "beam_elements.csv").sort_values("ElementId")
    if len(elems_df) != cfg["expected_num_edges"]:
        raise ValueError(
            f"Expected {cfg['expected_num_edges']} edges, got {len(elems_df)}"
        )
    print(f"  Edges: {len(elems_df)} (OK)")

    elem_ids = elems_df["ElementId"].tolist()
    elem_id_to_idx = {eid: i for i, eid in enumerate(elem_ids)}
    assert len(elem_id_to_idx) == cfg["expected_num_edges"]

    edge_index = torch.tensor(
        [
            [node_id_to_idx[row["INodeId"]] for _, row in elems_df.iterrows()],
            [node_id_to_idx[row["JNodeId"]] for _, row in elems_df.iterrows()],
        ],
        dtype=torch.long,
    )
    print(f"  edge_index: {edge_index.shape}")
    print(f"  ElementId order identical across samples (verified above).")

    # Determine all LoadCaseIds from first sample's nodal_loads
    loads_sample0 = pd.read_csv(
        raw_dir / sample_ids[0] / "nodal_loads.csv",
        usecols=["LoadCaseId"],
    )
    loadcase_ids = sorted(loads_sample0["LoadCaseId"].unique())
    expected_lc = len(loadcase_ids)
    print(f"  LoadCaseIds per sample: {expected_lc}")

    # Verify all samples have the same number of LCs
    for sid in sample_ids:
        df = pd.read_csv(
            raw_dir / sid / "nodal_loads.csv", usecols=["LoadCaseId"]
        )
        u = df["LoadCaseId"].nunique()
        if u != expected_lc:
            raise ValueError(
                f"Sample {sid} has {u} LoadCaseIds, expected {expected_lc}"
            )
    print(f"  All samples have {expected_lc} LoadCaseIds (OK)")

    # Pack into return
    info = {
        "node_id_to_idx": node_id_to_idx,
        "elem_id_to_idx": elem_id_to_idx,
        "node_ids": node_ids,
        "elem_ids": elem_ids,
        "expected_lc": expected_lc,
    }
    return cfg, sample_ids, loadcase_ids, edge_index, node_coords_tensor, nodes_df, info


# ============================================================
# Phase 1: Graph construction
# ============================================================

def build_one_sample_graphs(
    sample_id: str,
    loadcase_ids: List[int],
    edge_index: torch.Tensor,
    node_coords_tensor: torch.Tensor,
    node_ids: List[int],
    elem_ids: List[int],
    raw_sample_dir: Path,
    processed_graph_dir: Path,
    cfg: dict,
) -> List[dict]:
    """Build all graphs for one sample.

    Returns list of index records (one per graph).
    """
    f_cfg = cfg["files"]

    # ---- Load sample-specific files ----
    # beam_elements: get SectionId per element
    elems_df = pd.read_csv(raw_sample_dir / f_cfg["beam_elements"]).sort_values("ElementId")
    section_ids = elems_df["SectionId"].values  # aligned with elem_ids

    # beam_sections: get section properties
    sections_df = pd.read_csv(raw_sample_dir / f_cfg["beam_sections"])
    section_props = sections_df.set_index("SectionId")[cfg["edge_feature_fields"]].to_dict("index")

    # Build edge_attr template (same for all LCs in this sample)
    edge_attr_template = torch.tensor(
        [
            [section_props[sid][f] for f in cfg["edge_feature_fields"]]
            for sid in section_ids
        ],
        dtype=torch.float32,
    )
    assert edge_attr_template.shape == (cfg["expected_num_edges"], cfg["edge_feature_dim"]), \
        f"edge_attr shape {edge_attr_template.shape} != ({cfg['expected_num_edges']}, {cfg['edge_feature_dim']})"

    # ---- Pre-load nodal loads (all LCs at once, grouped) ----
    loads_df = pd.read_csv(
        raw_sample_dir / f_cfg["nodal_loads"],
        usecols=["LoadCaseId", "NodeId"] + cfg["load_fields"],
    )
    loads_grouped = dict(tuple(loads_df.groupby("LoadCaseId")))
    # Validate: each LC has exactly num_nodes records
    for lc_id in loadcase_ids:
        grp = loads_grouped.get(lc_id)
        if grp is None or len(grp) != cfg["expected_num_nodes"]:
            raise ValueError(
                f"[Sample {sample_id}, LC {lc_id}] Nodal loads count: "
                f"got {len(grp) if grp is not None else 0}, "
                f"expected {cfg['expected_num_nodes']}"
            )

    # ---- Pre-load beam element results (all LCs at once, grouped) ----
    results_df = pd.read_csv(
        raw_sample_dir / f_cfg["beam_element_results"],
        usecols=["LoadCaseId", "ElementId", "EndMark", "Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        dtype={"EndMark": str},
    )
    results_grouped = dict(tuple(results_df.groupby("LoadCaseId")))
    # Validate: each LC has exactly 2 * num_edges records
    expected_result_rows = cfg["expected_num_edges"] * 2
    for lc_id in loadcase_ids:
        grp = results_grouped.get(lc_id)
        if grp is None or len(grp) != expected_result_rows:
            raise ValueError(
                f"[Sample {sample_id}, LC {lc_id}] Result records count: "
                f"got {len(grp) if grp is not None else 0}, "
                f"expected {expected_result_rows}"
            )

    # ---- Build graphs for each LC ----
    index_records = []
    sample_graph_dir = processed_graph_dir / sample_id
    sample_graph_dir.mkdir(parents=True, exist_ok=True)

    for lc_id in loadcase_ids:
        # Node features: [X, Y, Z] + [Fx, Fy, Fz, Mx, My, Mz]
        lc_loads = loads_grouped[lc_id].set_index("NodeId").loc[node_ids][cfg["load_fields"]]
        load_tensor = torch.tensor(lc_loads.values, dtype=torch.float32)
        x = torch.cat([node_coords_tensor, load_tensor], dim=1)
        assert x.shape == (cfg["expected_num_nodes"], cfg["node_feature_dim"]), \
            f"x shape {x.shape} != ({cfg['expected_num_nodes']}, {cfg['node_feature_dim']})"

        # Edge labels from results
        lc_results = results_grouped[lc_id].copy()
        lc_results = lc_results.sort_values(["ElementId", "EndMark"])
        # Verify EndMark alternates I, J
        endmarks = lc_results["EndMark"].values
        if len(endmarks) % 2 != 0:
            raise ValueError(
                f"[Sample {sample_id}, LC {lc_id}] Odd number of result rows: {len(endmarks)}"
            )
        i_marks = endmarks[0::2]
        j_marks = endmarks[1::2]
        if not (np.all(i_marks == "I") and np.all(j_marks == "J")):
            # Try to find where it fails
            bad_positions = np.where((i_marks != "I") | (j_marks != "J"))[0]
            raise ValueError(
                f"[Sample {sample_id}, LC {lc_id}] EndMark ordering violation at "
                f"{len(bad_positions)} positions. Expected alternating I/J. "
                f"First bad: {bad_positions[:5]}"
            )

        i_forces = lc_results.iloc[0::2][cfg["load_fields"]].values  # (1646, 6)
        j_forces = lc_results.iloc[1::2][cfg["load_fields"]].values  # (1646, 6)
        y_edge = torch.tensor(np.concatenate([i_forces, j_forces], axis=1), dtype=torch.float32)
        assert y_edge.shape == (cfg["expected_num_edges"], cfg["target_dim"]), \
            f"y_edge shape {y_edge.shape} != ({cfg['expected_num_edges']}, {cfg['target_dim']})"

        # Build Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_template,
            y_edge=y_edge,
        )

        # Save
        filename = f"{lc_id:04d}.pt"
        save_path = sample_graph_dir / filename
        torch.save(data, save_path)

        # Record in index
        graph_id = f"{sample_id}_{lc_id:04d}"
        index_records.append({
            "graph_id": graph_id,
            "sample_id": sample_id,
            "loadcase_id": lc_id,
            "graph_file": str(Path("graphs") / sample_id / filename),
        })

    return index_records


# ============================================================
# Phase 2: Compute training statistics (Welford)
# ============================================================

def compute_training_stats(
    processed_dir: Path,
    index_df: pd.DataFrame,
    split: Dict[str, list],
    split_mode: str,
) -> NodeEdgeStandardScaler:
    """Compute mean/std on training set using Welford's algorithm."""
    print("\n" + "=" * 60)
    print("Phase 2: Compute training statistics (Welford)")
    print("=" * 60)

    # Determine which entries are in the training set
    train_ids = set(split["train"])
    if split_mode == "by_sample":
        is_train = index_df["sample_id"].isin(train_ids)
    elif split_mode == "by_loadcase":
        is_train = index_df["loadcase_id"].isin(train_ids)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    train_df = index_df[is_train].reset_index(drop=True)
    print(f"  Training samples: {len(train_df)}")

    scaler = NodeEdgeStandardScaler()
    count = 0
    t0 = time.time()
    for _, row in train_df.iterrows():
        data_path = processed_dir / row["graph_file"]
        data = torch.load(data_path, weights_only=False)
        scaler.update(data)
        count += 1
        if count % 5000 == 0:
            print(f"    ... processed {count}/{len(train_df)} training graphs")

    stats = scaler.finalize()
    scaler.save(processed_dir / "feature_stats.json")
    elapsed = time.time() - t0
    print(f"  Done. {count} graphs in {elapsed:.1f}s.")
    print(f"  node_mean: {stats['node_mean'].tolist()[:4]}...")
    print(f"  node_std:  {stats['node_std'].tolist()[:4]}...")
    return scaler


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()

    # ---- Phase 0 ----
    cfg, sample_ids, loadcase_ids, edge_index, node_coords_tensor, nodes_df, info = phase0_setup(
        load_config()
    )

    processed_dir = cfg["processed_data_dir"]
    graph_dir = processed_dir / cfg["graph_subdir"]
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Build graphs ----
    print("\n" + "=" * 60)
    print("Phase 1: Graph construction")
    print("=" * 60)
    all_index_records = []
    total_expected = len(sample_ids) * len(loadcase_ids)
    graph_count = 0

    for sid in sample_ids:
        raw_sample_dir = cfg["raw_data_dir"] / sid
        t_s = time.time()
        records = build_one_sample_graphs(
            sample_id=sid,
            loadcase_ids=loadcase_ids,
            edge_index=edge_index,
            node_coords_tensor=node_coords_tensor,
            node_ids=info["node_ids"],
            elem_ids=info["elem_ids"],
            raw_sample_dir=raw_sample_dir,
            processed_graph_dir=graph_dir,
            cfg=cfg,
        )
        graph_count += len(records)
        all_index_records.extend(records)
        print(f"  Sample {sid}: {len(records)} graphs ({time.time()-t_s:.1f}s)")

    if graph_count != total_expected:
        raise RuntimeError(
            f"Graph count mismatch: expected {total_expected}, got {graph_count}"
        )
    print(f"\n  Total graphs built: {graph_count}/{total_expected}")

    # ---- Save index.csv ----
    print("\n" + "-" * 40)
    print("Saving index.csv ...")
    index_df = pd.DataFrame(all_index_records)
    index_csv = processed_dir / "index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8")
    print(f"  Saved: {index_csv} (rows={len(index_df)})")

    # ---- Save metadata.json ----
    metadata = {
        "dataset_name": "graph_dataset_baseline",
        "description": "Baseline homogeneous graph dataset for steel truss bridge edge-level regression",
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_graphs": graph_count,
        "num_samples": len(sample_ids),
        "num_loadcases": len(loadcase_ids),
        "expected_num_nodes": cfg["expected_num_nodes"],
        "expected_num_edges": cfg["expected_num_edges"],
        "node_feature_dim": cfg["node_feature_dim"],
        "edge_feature_dim": cfg["edge_feature_dim"],
        "target_dim": cfg["target_dim"],
        "task_type": "edge-level regression",
        "graph_type": "homogeneous",
        "node_feature_fields": cfg["node_feature_fields"],
        "edge_feature_fields": cfg["edge_feature_fields"],
        "target_fields": cfg["target_fields"],
        "samples": [str(s) for s in sample_ids],
        "loadcase_ids": [int(lc) for lc in loadcase_ids],
        "raw_data_dir": str(cfg["raw_data_dir"]),
        "processed_data_dir": str(processed_dir),
    }
    meta_path = processed_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path}")

    # ---- Phase 3: Splits ----
    print("\n" + "=" * 60)
    print("Phase 3: Split generation")
    print("=" * 60)
    splits = generate_all_splits(
        sample_ids=sample_ids,
        loadcase_ids=loadcase_ids,
        split_dir=processed_dir / cfg["split_dir"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        random_seed=cfg["random_seed"],
    )
    print_split_summary(splits, "split_by_sample")
    print_split_summary(splits, "split_by_loadcase")

    # ---- Phase 4: Training statistics ----
    compute_training_stats(
        processed_dir=processed_dir,
        index_df=index_df,
        split=splits["split_by_sample"],
        split_mode="by_sample",
    )

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE — {elapsed:.1f}s total")
    print("=" * 60)
    print(f"  Output: {processed_dir}")
    print(f"  Graphs: {graph_count}")


if __name__ == "__main__":
    main()

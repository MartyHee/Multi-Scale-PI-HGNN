#!/usr/bin/env python3
"""
build_hetero_graph_dataset.py — Heterogeneous graph dataset builder v2.

Reads GraphTrainingData2 raw CSV data and produces a per-graph ``HeteroData``
dataset inside the model project's processed directory.

Schema (v2, per proposal fix)
------------------------------
- 3 node types:  ``mesh_node, beam_element, plate_element``
- 5 edge types:  ``mesh_node→beam_element`` (fwd+rev),
                  ``mesh_node→plate_element`` (fwd+rev),
                  ``mesh_node→mesh_node`` (structural_link)
- 2 supervision tasks:  mesh_node displacement regression (6-d),
                         beam_element force regression (12-d)

v2 vs v1:
  - Removed ``link_element`` node type.
  - Renamed ``rigid_link`` → ``structural_link``.
  - Rigid_elastic_links.csv modeled as mesh_node ↔ mesh_node interaction edge.
  - Reverse membership edges (rev_belongs_to_beam, rev_belongs_to_plate) now
    carry ``edge_attr`` (copied from forward edges).
  - ``structural_link`` edge_attr is 10-d (includes ``is_rigid``).

Usage::

    cd D:\\CREC\\BiShe\\S1\\Multi-Scale-PI-HGNN
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe \\
        src/data/build_hetero_graph_dataset.py

    # Debug: limit to N graphs
    D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe \\
        src/data/build_hetero_graph_dataset.py --max-graphs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.data.hetero_schema import (
    EDGE_ATTR_DIMS,
    EDGE_COUNTS,
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    NODE_TYPES,
    NODE_TYPE_CONFIG,
    TARGET_NODE_TYPES,
    build_schema_metadata,
    get_feature_dim,
    get_node_count,
    get_target_dim,
    get_target_key,
)
from src.data.hetero_split import generate_split_files, load_split_file
from src.data.hetero_transforms import HeteroFeatureScaler

# ============================================================
# Config
# ============================================================

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load hetero_dataset.yaml and resolve paths."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "hetero_dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_project_root"] = _PROJECT_ROOT
    cfg["raw_data_dir"] = Path(cfg["raw_data_dir"]).resolve()
    cfg["processed_data_dir"] = (_PROJECT_ROOT / cfg["processed_data_dir"]).resolve()
    return cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build heterogeneous graph dataset v2")
    p.add_argument("--max-graphs", type=int, default=None,
                   help="Debug: limit total graphs to build")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Debug: limit number of samples to process")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config file (default: configs/hetero_dataset.yaml)")
    p.add_argument("--skip-stats", action="store_true",
                   help="Skip Phase 4 (training stats computation)")
    return p.parse_args(argv)


# ============================================================
# Helpers
# ============================================================

def get_sample_dirs(raw_dir: Path) -> List[str]:
    """Return sorted sample ID strings from the raw data directory."""
    samples = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        if (entry / "nodes.csv").exists():
            samples.append(entry.name)
    return samples


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    elif hasattr(obj, "dtype"):  # numpy scalar
        return obj.item() if hasattr(obj, "item") else obj.tolist()
    return obj


# ============================================================
# Phase 0: Setup, validation, global mappings, templates
# ============================================================

def phase0_setup(cfg: dict) -> dict:
    """Validate raw data, build global mappings, edge_index, static features.

    Returns a dictionary of global data shared by all graphs.
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
    print(f"  Found {len(sample_ids)} sample directories")

    # ---- 1. Cross-sample consistency checks ----
    # Nodes
    ref_nodes = pd.read_csv(raw_dir / sample_ids[0] / "nodes.csv").sort_values("NodeId")
    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "nodes.csv").sort_values("NodeId")
        if not df[["NodeId", "X", "Y", "Z"]].equals(ref_nodes[["NodeId", "X", "Y", "Z"]]):
            raise ValueError(f"nodes.csv mismatch in sample {sid}!")
    print(f"  [ok] nodes.csv consistent across {len(sample_ids)} samples")

    # Beam topology
    ref_beam = pd.read_csv(raw_dir / sample_ids[0] / "beam_elements.csv").sort_values("ElementId")
    beam_topo_cols = ["ElementId", "INodeId", "JNodeId"]
    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "beam_elements.csv").sort_values("ElementId")
        if not df[beam_topo_cols].equals(ref_beam[beam_topo_cols]):
            raise ValueError(f"beam_elements topology mismatch in sample {sid}!")
    print(f"  [ok] beam_elements topology consistent across {len(sample_ids)} samples")

    # Plate topology
    ref_plate = pd.read_csv(raw_dir / sample_ids[0] / "plate_elements.csv").sort_values("ElementId")
    plate_topo_cols = ["ElementId", "INodeId", "JNodeId", "KNodeId", "LNodeId"]
    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "plate_elements.csv").sort_values("ElementId")
        if not df[plate_topo_cols].equals(ref_plate[plate_topo_cols]):
            raise ValueError(f"plate_elements topology mismatch in sample {sid}!")
    print(f"  [ok] plate_elements topology consistent across {len(sample_ids)} samples")

    # Structural link topology (rigid_elastic_links.csv)
    ref_link = pd.read_csv(raw_dir / sample_ids[0] / "rigid_elastic_links.csv").sort_values("ElasticLinkId")
    link_topo_cols = ["ElasticLinkId", "INodeId", "JNodeId"]
    for sid in sample_ids[1:]:
        df = pd.read_csv(raw_dir / sid / "rigid_elastic_links.csv").sort_values("ElasticLinkId")
        if not df[link_topo_cols].equals(ref_link[link_topo_cols]):
            raise ValueError(f"rigid_elastic_links topology mismatch in sample {sid}!")
    print(f"  [ok] rigid_elastic_links topology consistent across {len(sample_ids)} samples")

    # ---- 2. Global mappings ----
    num_nodes = get_node_count("mesh_node")
    assert len(ref_nodes) == num_nodes, f"Expected {num_nodes} mesh nodes, got {len(ref_nodes)}"
    node_ids = sorted(ref_nodes["NodeId"].tolist())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Beam element mapping
    assert len(ref_beam) == get_node_count("beam_element")
    elem_ids = ref_beam["ElementId"].tolist()
    elem_id_to_idx = {eid: i for i, eid in enumerate(elem_ids)}

    # Plate element mapping
    assert len(ref_plate) == get_node_count("plate_element")
    plate_ids = ref_plate["ElementId"].tolist()
    plate_id_to_idx = {pid: i for i, pid in enumerate(plate_ids)}

    # ---- 3. Node coordinate tensor ----
    node_coords = torch.tensor(
        ref_nodes.set_index("NodeId").loc[node_ids][["X", "Y", "Z"]].values,
        dtype=torch.float32,
    )
    print(f"  Node coords: {node_coords.shape}")

    # ---- 4. Support constraint tensor ----
    supports_df = pd.read_csv(raw_dir / sample_ids[0] / cfg["files"]["general_supports"])
    support_tensor = torch.zeros((num_nodes, 6), dtype=torch.float32)
    for _, row in supports_df.iterrows():
        idx = node_id_to_idx[row["NodeId"]]
        support_tensor[idx] = torch.tensor(
            [row["Dx"], row["Dy"], row["Dz"], row["Rx"], row["Ry"], row["Rz"]],
            dtype=torch.float32,
        )
    assert support_tensor.shape == (num_nodes, 6)
    n_supp = (support_tensor.abs().sum(dim=1) > 0).sum().item()
    print(f"  Support constraint tensor: {support_tensor.shape}, {n_supp} constrained nodes")

    # ---- 5. Edge index construction ----

    # 5a. mesh_node -> belongs_to_beam -> beam_element
    mesh_to_beam_src = []
    mesh_to_beam_tgt = []
    beam_endpoint_type = []

    for be_idx, (_, row) in enumerate(ref_beam.iterrows()):
        i_idx = node_id_to_idx[row["INodeId"]]
        j_idx = node_id_to_idx[row["JNodeId"]]
        mesh_to_beam_src.append(i_idx)
        mesh_to_beam_tgt.append(be_idx)
        beam_endpoint_type.append(0.0)  # I-end
        mesh_to_beam_src.append(j_idx)
        mesh_to_beam_tgt.append(be_idx)
        beam_endpoint_type.append(1.0)  # J-end

    edge_index_mb = torch.tensor([mesh_to_beam_src, mesh_to_beam_tgt], dtype=torch.long)
    edge_attr_mb = torch.tensor(beam_endpoint_type, dtype=torch.float32).unsqueeze(1)
    assert edge_index_mb.shape == (2, 3292)
    assert edge_attr_mb.shape == (3292, 1)

    # 5b. beam_element -> rev_belongs_to_beam -> mesh_node (reverse WITH edge_attr)
    edge_index_bm = torch.tensor([mesh_to_beam_tgt, mesh_to_beam_src], dtype=torch.long)
    edge_attr_bm = edge_attr_mb.clone()  # same endpoint_type as forward edge
    assert edge_index_bm.shape == (2, 3292)
    assert edge_attr_bm.shape == (3292, 1)

    # 5c. mesh_node -> belongs_to_plate -> plate_element
    mesh_to_plate_src = []
    mesh_to_plate_tgt = []
    plate_corner_type = []
    corner_map = {"INodeId": 0, "JNodeId": 1, "KNodeId": 2, "LNodeId": 3}

    for pl_idx, (_, row) in enumerate(ref_plate.iterrows()):
        for col, ctype in corner_map.items():
            nidx = node_id_to_idx[row[col]]
            mesh_to_plate_src.append(nidx)
            mesh_to_plate_tgt.append(pl_idx)
            plate_corner_type.append(float(ctype))

    edge_index_mp = torch.tensor([mesh_to_plate_src, mesh_to_plate_tgt], dtype=torch.long)
    edge_attr_mp = torch.tensor(plate_corner_type, dtype=torch.float32).unsqueeze(1)
    assert edge_index_mp.shape == (2, 3328)
    assert edge_attr_mp.shape == (3328, 1)

    # 5d. plate_element -> rev_belongs_to_plate -> mesh_node (reverse WITH edge_attr)
    edge_index_pm = torch.tensor([mesh_to_plate_tgt, mesh_to_plate_src], dtype=torch.long)
    edge_attr_pm = edge_attr_mp.clone()  # same corner_type as forward edge
    assert edge_index_pm.shape == (2, 3328)
    assert edge_attr_pm.shape == (3328, 1)

    # 5e. mesh_node -> structural_link -> mesh_node (was rigid_link in v1)
    link_src = []
    link_tgt = []
    link_edge_attr_list = []

    for _, row in ref_link.iterrows():
        i_idx = node_id_to_idx[row["INodeId"]]
        j_idx = node_id_to_idx[row["JNodeId"]]
        link_src.append(i_idx)
        link_tgt.append(j_idx)
        # structural_link edge_attr: 10-d
        # [Kx, Ky, Kz, Krx, Kry, Krz, BetaAngle, DistanceRatio, ElasticLinkType, is_rigid]
        link_edge_attr_list.append([
            row["Kx"], row["Ky"], row["Kz"],
            row["Krx"], row["Kry"], row["Krz"],
            row["BetaAngle"], row["DistanceRatio"],
            float(row["ElasticLinkType"]),
            1.0,  # is_rigid — all current links are RIGID
        ])

    edge_index_sl = torch.tensor([link_src, link_tgt], dtype=torch.long)
    edge_attr_sl = torch.tensor(link_edge_attr_list, dtype=torch.float32)
    assert edge_index_sl.shape == (2, 132)
    assert edge_attr_sl.shape == (132, 10)

    print(f"  Edge templates built:")
    print(f"    mesh->beam:           {edge_index_mb.shape}")
    print(f"    beam->mesh (rev):     {edge_index_bm.shape}")
    print(f"    mesh->plate:          {edge_index_mp.shape}")
    print(f"    plate->mesh (rev):    {edge_index_pm.shape}")
    print(f"    structural_link:      {edge_index_sl.shape}")

    # ---- 6. Global static node features ----
    # 6a. plate_element features (cross-sample consistent)
    materials_df_global = pd.read_csv(
        raw_dir / sample_ids[0] / cfg["files"]["materials"]
    ).set_index("MaterialId")

    thick_df_global = pd.read_csv(
        raw_dir / sample_ids[0] / cfg["files"]["thicknesses"]
    ).set_index("ThicknessId")

    plate_feat_list = []
    for _, row in ref_plate.iterrows():
        mat = materials_df_global.loc[row["MaterialId"]]
        thick = thick_df_global.loc[row["ThicknessId"]]
        plate_feat_list.append([
            thick["ThicknessValue"],
            mat["ElasticModulus"],
            mat["PoissonRatio"],
            mat["UnitWeight"],
            row["BetaAngle"],
            float(row["PlateType"]),
        ])
    plate_x_static = torch.tensor(plate_feat_list, dtype=torch.float32)
    assert plate_x_static.shape == (832, 6)
    print(f"  plate_element static features: {plate_x_static.shape}")

    # Note: No link_element features — rigid_elastic_links.csv is an edge, not a node.

    # 6b. beam_sections catalog (for per-sample beam_feature computation)
    beam_sections_global = pd.read_csv(
        raw_dir / sample_ids[0] / cfg["files"]["beam_sections"]
    ).set_index("SectionId")
    print(f"  beam_sections catalog: {len(beam_sections_global)} sections")

    # ---- 7. LoadCase IDs ----
    loads_sample0 = pd.read_csv(
        raw_dir / sample_ids[0] / cfg["files"]["nodal_loads"],
        usecols=["LoadCaseId"],
    )
    loadcase_ids = sorted(loads_sample0["LoadCaseId"].unique())
    expected_lc = len(loadcase_ids)
    print(f"  LoadCaseIds per sample: {expected_lc}")

    # ---- 8. Pack global data ----
    global_data = {
        "cfg": cfg,
        "sample_ids": sample_ids,
        "loadcase_ids": loadcase_ids,
        "raw_dir": raw_dir,
        "node_id_to_idx": node_id_to_idx,
        "node_ids": node_ids,
        "node_coords": node_coords,
        "support_tensor": support_tensor,
        "elem_id_to_idx": elem_id_to_idx,
        "elem_ids": elem_ids,
        "plate_id_to_idx": plate_id_to_idx,
        "plate_ids": plate_ids,
        # Edge templates
        "edge_index_mb": edge_index_mb,
        "edge_attr_mb": edge_attr_mb,
        "edge_index_bm": edge_index_bm,
        "edge_attr_bm": edge_attr_bm,
        "edge_index_mp": edge_index_mp,
        "edge_attr_mp": edge_attr_mp,
        "edge_index_pm": edge_index_pm,
        "edge_attr_pm": edge_attr_pm,
        "edge_index_sl": edge_index_sl,
        "edge_attr_sl": edge_attr_sl,
        # Static features
        "plate_x_static": plate_x_static,
        "beam_sections_catalog": beam_sections_global,
        "materials": materials_df_global,
    }

    return global_data


# ============================================================
# Phase 1: Graph construction (per sample)
# ============================================================

def _make_hetero_data(
    mesh_x: torch.Tensor,
    mesh_y: Optional[torch.Tensor],
    beam_x: torch.Tensor,
    beam_y: Optional[torch.Tensor],
    plate_x: torch.Tensor,
    # Edge templates (5 types, all with edge_attr for reverse edges too)
    edge_index_mb: torch.Tensor,
    edge_attr_mb: torch.Tensor,
    edge_index_bm: torch.Tensor,
    edge_attr_bm: torch.Tensor,
    edge_index_mp: torch.Tensor,
    edge_attr_mp: torch.Tensor,
    edge_index_pm: torch.Tensor,
    edge_attr_pm: torch.Tensor,
    edge_index_sl: torch.Tensor,
    edge_attr_sl: torch.Tensor,
) -> HeteroData:
    """Assemble one HeteroData graph from pre-computed tensors."""
    data = HeteroData()

    # Node types (3 only — no link_element)
    data["mesh_node"].x = mesh_x
    if mesh_y is not None:
        data["mesh_node"].y_disp = mesh_y

    data["beam_element"].x = beam_x
    if beam_y is not None:
        data["beam_element"].y_force = beam_y

    data["plate_element"].x = plate_x

    # Edge types (5)
    # Forward membership
    data["mesh_node", "belongs_to_beam", "beam_element"].edge_index = edge_index_mb
    data["mesh_node", "belongs_to_beam", "beam_element"].edge_attr = edge_attr_mb

    # Reverse membership WITH edge_attr (v2 fix)
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_index = edge_index_bm
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_attr = edge_attr_bm

    # Forward plate membership
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_index = edge_index_mp
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_attr = edge_attr_mp

    # Reverse plate membership WITH edge_attr (v2 fix)
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_index = edge_index_pm
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_attr = edge_attr_pm

    # structural_link interaction edge (was rigid_link in v1)
    data["mesh_node", "structural_link", "mesh_node"].edge_index = edge_index_sl
    data["mesh_node", "structural_link", "mesh_node"].edge_attr = edge_attr_sl

    return data


def _compute_beam_element_features(
    sample_id: str,
    raw_sample_dir: Path,
    cfg: dict,
    global_data: dict,
) -> torch.Tensor:
    """Compute beam_element features for a specific sample.

    Features (11-d):
        [Area, Ix, Iy, Iz] — from beam_sections via SectionId
        [ElasticModulus, PoissonRatio, UnitWeight] — from materials via MaterialId
        [Length, CosX, CosY, CosZ] — derived from node coords

    These are sample-specific (section/material assignments vary) but
    LC-invariant, so computed once per sample.
    """
    node_coords = global_data["node_coords"]
    node_id_to_idx = global_data["node_id_to_idx"]
    node_ids = global_data["node_ids"]
    coords_np = node_coords.numpy()  # (1056, 3), index=mesh_node_idx
    node_id_list = np.array(node_ids)

    # Read sample-specific files
    beam_elems = pd.read_csv(raw_sample_dir / cfg["files"]["beam_elements"]).sort_values("ElementId")
    beam_sections = pd.read_csv(raw_sample_dir / cfg["files"]["beam_sections"]).set_index("SectionId")
    materials = pd.read_csv(raw_sample_dir / cfg["files"]["materials"]).set_index("MaterialId")

    feat_list = []
    for _, row in beam_elems.iterrows():
        # Section properties
        sec = beam_sections.loc[row["SectionId"]]
        area, ix, iy, iz = sec["Area"], sec["Ix"], sec["Iy"], sec["Iz"]

        # Material properties
        mat = materials.loc[row["MaterialId"]]
        E, nu, gamma = mat["ElasticModulus"], mat["PoissonRatio"], mat["UnitWeight"]

        # Geometry: length + direction cosines
        i_idx = node_id_to_idx[row["INodeId"]]
        j_idx = node_id_to_idx[row["JNodeId"]]
        pi = coords_np[i_idx]  # (3,)
        pj = coords_np[j_idx]  # (3,)
        vec = pj - pi
        length = float(np.linalg.norm(vec))
        if length > 1e-12:
            cosx, cosy, cosz = vec[0] / length, vec[1] / length, vec[2] / length
        else:
            cosx, cosy, cosz = 0.0, 0.0, 0.0

        feat_list.append([area, ix, iy, iz, E, nu, gamma, length, cosx, cosy, cosz])

    beam_x = torch.tensor(feat_list, dtype=torch.float32)
    expected_dim = get_feature_dim("beam_element")
    assert beam_x.shape == (1646, expected_dim), \
        f"beam_element features shape {beam_x.shape} != (1646, {expected_dim})"
    return beam_x


def build_one_sample_graphs(
    sample_id: str,
    global_data: dict,
    processed_graph_dir: Path,
    max_graphs: Optional[int],
    running_count: int,
) -> Tuple[List[dict], int]:
    """Build all graphs (one per LC) for one sample.

    Returns (index_records, new_running_count).
    """
    cfg = global_data["cfg"]
    raw_dir = global_data["raw_dir"]
    raw_sample_dir = raw_dir / sample_id
    f_cfg = cfg["files"]
    num_nodes = get_node_count("mesh_node")
    num_beams = get_node_count("beam_element")
    loadcase_ids = global_data["loadcase_ids"]

    # ---- Build beam_element features (sample-specific, LC-invariant) ----
    beam_x = _compute_beam_element_features(sample_id, raw_sample_dir, cfg, global_data)

    # ---- Load & group LC-varying data ----
    # Nodal loads
    loads_df = pd.read_csv(
        raw_sample_dir / f_cfg["nodal_loads"],
        usecols=["LoadCaseId", "NodeId", "Fx", "Fy", "Fz", "Mx", "My", "Mz"],
    )
    loads_grouped = dict(tuple(loads_df.groupby("LoadCaseId")))

    # Beam element results
    results_df = pd.read_csv(
        raw_sample_dir / f_cfg["beam_element_results"],
        usecols=["LoadCaseId", "ElementId", "EndMark", "Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        dtype={"EndMark": str},
    )
    results_grouped = dict(tuple(results_df.groupby("LoadCaseId")))

    # Node displacement results
    disp_df = pd.read_csv(
        raw_sample_dir / f_cfg["node_displacement_results"],
        usecols=["LoadCaseId", "NodeId", "Dx", "Dy", "Dz", "Rx", "Ry", "Rz"],
    )
    disp_grouped = dict(tuple(disp_df.groupby("LoadCaseId")))

    # ---- Validate pre-grouped data ----
    for lc_id in loadcase_ids:
        ld = loads_grouped.get(lc_id)
        if ld is None or len(ld) != num_nodes:
            raise ValueError(
                f"[{sample_id}, LC {lc_id}] nodal_loads count {len(ld) if ld is not None else 0} != {num_nodes}"
            )
        rd = results_grouped.get(lc_id)
        if rd is None or len(rd) != num_beams * 2:
            raise ValueError(
                f"[{sample_id}, LC {lc_id}] beam result records {len(rd) if rd is not None else 0} != {num_beams * 2}"
            )
        dd = disp_grouped.get(lc_id)
        if dd is None or len(dd) != num_nodes:
            raise ValueError(
                f"[{sample_id}, LC {lc_id}] displacement records {len(dd) if dd is not None else 0} != {num_nodes}"
            )

    # ---- Pre-sort beam results for fast per-LC access ----
    results_sorted = results_df.sort_values(["LoadCaseId", "ElementId", "EndMark"])
    results_sorted_grouped = dict(tuple(results_sorted.groupby("LoadCaseId")))

    # Also build per-LC reindexed lookup for loads & displacements
    node_ids = global_data["node_ids"]
    load_fields = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    disp_fields = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]

    # ---- Build graphs for each LC ----
    sample_graph_dir = processed_graph_dir / sample_id
    sample_graph_dir.mkdir(parents=True, exist_ok=True)
    index_records = []

    for ci, lc_id in enumerate(loadcase_ids):
        if max_graphs is not None and running_count >= max_graphs:
            break

        # 1. mesh_node features: coords + loads + support
        lc_loads = loads_grouped[lc_id].set_index("NodeId").reindex(node_ids)
        load_tensor = torch.from_numpy(lc_loads[load_fields].values).float()
        mesh_x = torch.cat([global_data["node_coords"], load_tensor, global_data["support_tensor"]], dim=1)
        assert mesh_x.shape == (num_nodes, 15), f"mesh_x shape {mesh_x.shape}"

        # 2. mesh_node targets: node_displacement
        lc_disp = disp_grouped[lc_id].set_index("NodeId").reindex(node_ids)
        mesh_y = torch.from_numpy(lc_disp[disp_fields].values).float()

        # 3. beam_element targets: I/J-end forces (pre-sorted, no sort per LC)
        lc_res = results_sorted_grouped[lc_id]
        endmarks = lc_res["EndMark"].values
        if len(endmarks) % 2 != 0:
            raise ValueError(f"[{sample_id}, LC {lc_id}] Odd number of result rows: {len(endmarks)}")
        i_marks = endmarks[0::2]
        j_marks = endmarks[1::2]
        if not (np.all(i_marks == "I") and np.all(j_marks == "J")):
            bad = np.where((i_marks != "I") | (j_marks != "J"))[0]
            raise ValueError(f"[{sample_id}, LC {lc_id}] EndMark ordering fails at {bad[:5]}")
        force_fields = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        i_forces = lc_res.iloc[0::2][force_fields].values  # (1646, 6)
        j_forces = lc_res.iloc[1::2][force_fields].values  # (1646, 6)
        beam_y = torch.tensor(np.concatenate([i_forces, j_forces], axis=1), dtype=torch.float32)
        assert beam_y.shape == (1646, 12), f"beam_y shape {beam_y.shape}"

        # 4. Assemble HeteroData
        data = _make_hetero_data(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            beam_x=beam_x,
            beam_y=beam_y,
            plate_x=global_data["plate_x_static"],
            edge_index_mb=global_data["edge_index_mb"],
            edge_attr_mb=global_data["edge_attr_mb"],
            edge_index_bm=global_data["edge_index_bm"],
            edge_attr_bm=global_data["edge_attr_bm"],
            edge_index_mp=global_data["edge_index_mp"],
            edge_attr_mp=global_data["edge_attr_mp"],
            edge_index_pm=global_data["edge_index_pm"],
            edge_attr_pm=global_data["edge_attr_pm"],
            edge_index_sl=global_data["edge_index_sl"],
            edge_attr_sl=global_data["edge_attr_sl"],
        )

        # 5. Save
        filename = f"{lc_id:04d}.pt"
        save_path = sample_graph_dir / filename
        torch.save(data, save_path)

        # 6. Index record
        graph_id = f"{sample_id}_{lc_id:04d}"
        index_records.append({
            "graph_id": graph_id,
            "sample_id": sample_id,
            "loadcase_id": lc_id,
            "graph_file": str(Path("graphs") / sample_id / filename),
        })

        running_count += 1

    return index_records, running_count


# ============================================================
# Phase 4: Compute training statistics
# ============================================================

def compute_training_stats(
    processed_dir: Path,
    index_df: pd.DataFrame,
    split: Dict[str, list],
    split_mode: str,
) -> HeteroFeatureScaler:
    """Compute mean/std on training set using Welford online algorithm."""
    print("\n" + "=" * 60)
    print("Phase 4: Compute training statistics (Welford)")
    print("=" * 60)

    train_ids = set(split["train"])
    if split_mode == "by_sample":
        is_train = index_df["sample_id"].isin(train_ids)
    elif split_mode == "by_loadcase":
        is_train = index_df["loadcase_id"].isin(train_ids)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    train_df = index_df[is_train].reset_index(drop=True)
    total_train = len(train_df)
    print(f"  Training graphs: {total_train}")

    scaler = HeteroFeatureScaler()
    count = 0
    t0 = time.time()
    for _, row in train_df.iterrows():
        data_path = processed_dir / row["graph_file"]
        data: HeteroData = torch.load(data_path, weights_only=False)
        scaler.update(data)
        count += 1
        if count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    ... {count}/{total_train} ({elapsed:.1f}s)")

    scaler.finalize()
    scaler.save(processed_dir / "feature_stats.json")
    elapsed = time.time() - t0
    print(f"  Done. {count} graphs in {elapsed:.1f}s.")

    stats = scaler.get_stats_dict()
    for key in list(stats.keys())[:8]:
        m = stats[key]["mean"][:3]
        s = stats[key]["std"][:3]
        print(f"    {key}: mean={m}, std={s}")
    return scaler


# ============================================================
# Quality checks (v2 — no link_element, structural_link instead of rigid_link)
# ============================================================

def run_quality_checks(processed_dir: Path, index_df: pd.DataFrame, global_data: dict) -> int:
    """Run data consistency checks. Returns number of failures."""
    print("\n" + "=" * 60)
    print("Quality Checks")
    print("=" * 60)
    failures = 0

    import random
    random.seed(42)
    check_ids = random.sample(range(len(index_df)), min(10, len(index_df)))

    for idx in check_ids:
        row = index_df.iloc[idx]
        data_path = processed_dir / row["graph_file"]
        if not data_path.is_file():
            print(f"  [FAIL] Graph file missing: {data_path}")
            failures += 1
            continue
        data: HeteroData = torch.load(data_path, weights_only=False)
        sid = str(row["sample_id"])
        lc = int(row["loadcase_id"])

        # Checks 1-3: Node counts (3 types, no link_element)
        expected_counts = {
            "mesh_node": 1056,
            "beam_element": 1646,
            "plate_element": 832,
        }
        for ntype, expected in expected_counts.items():
            if ntype in data.node_types:
                actual = data[ntype].num_nodes
                if actual != expected:
                    print(f"  [FAIL] {sid}/{lc}: {ntype}.num_nodes={actual}, expected={expected}")
                    failures += 1

        # Check: NO link_element allowed
        if "link_element" in data.node_types:
            print(f"  [FAIL] {sid}/{lc}: link_element should NOT be in node_types")
            failures += 1

        # Check 4: mesh_node has correct feature dim (15)
        if data["mesh_node"].x.shape[1] != 15:
            print(f"  [FAIL] {sid}/{lc}: mesh_node.x dim {data['mesh_node'].x.shape[1]} != 15")
            failures += 1

        # Check 5: beam_element has correct feature dim (11)
        if data["beam_element"].x.shape[1] != 11:
            print(f"  [FAIL] {sid}/{lc}: beam_element.x dim {data['beam_element'].x.shape[1]} != 11")
            failures += 1

        # Check 6: mesh_node has y_disp
        if not hasattr(data["mesh_node"], "y_disp"):
            print(f"  [FAIL] {sid}/{lc}: mesh_node missing y_disp")
            failures += 1
        elif data["mesh_node"].y_disp.shape[1] != 6:
            print(f"  [FAIL] {sid}/{lc}: mesh_node.y_disp dim {data['mesh_node'].y_disp.shape[1]} != 6")
            failures += 1

        # Check 7: beam_element has y_force
        if not hasattr(data["beam_element"], "y_force"):
            print(f"  [FAIL] {sid}/{lc}: beam_element missing y_force")
            failures += 1
        elif data["beam_element"].y_force.shape[1] != 12:
            print(f"  [FAIL] {sid}/{lc}: beam_element.y_force dim {data['beam_element'].y_force.shape[1]} != 12")
            failures += 1

        # Check 8-12: Edge index counts and edge_attr presence
        edge_checks = {
            ("mesh_node", "belongs_to_beam", "beam_element"): {"count": 3292, "has_attr": True},
            ("beam_element", "rev_belongs_to_beam", "mesh_node"): {"count": 3292, "has_attr": True},
            ("mesh_node", "belongs_to_plate", "plate_element"): {"count": 3328, "has_attr": True},
            ("plate_element", "rev_belongs_to_plate", "mesh_node"): {"count": 3328, "has_attr": True},
            ("mesh_node", "structural_link", "mesh_node"): {"count": 132, "has_attr": True},
        }
        for etype, info in edge_checks.items():
            if etype in data.edge_types:
                actual = data[etype].edge_index.shape[1]
                if actual != info["count"]:
                    print(f"  [FAIL] {sid}/{lc}: edge {etype} count={actual}, expected={info['count']}")
                    failures += 1
                if info["has_attr"] and "edge_attr" not in data[etype]:
                    print(f"  [FAIL] {sid}/{lc}: edge {etype} missing edge_attr")
                    failures += 1

        # Check 13: No NaN or inf in node features
        for ntype in data.node_types:
            x = data[ntype].x
            if torch.isnan(x).any():
                print(f"  [FAIL] {sid}/{lc}: {ntype}.x contains NaN")
                failures += 1
            if torch.isinf(x).any():
                print(f"  [FAIL] {sid}/{lc}: {ntype}.x contains Inf")
                failures += 1

        # Check 14: No NaN/Inf in targets
        y = data["mesh_node"].y_disp
        if torch.isnan(y).any():
            print(f"  [FAIL] {sid}/{lc}: mesh_node.y_disp contains NaN")
            failures += 1
        if torch.isinf(y).any():
            print(f"  [FAIL] {sid}/{lc}: mesh_node.y_disp contains Inf")
            failures += 1

        if failures > 5:
            print(f"  Too many failures ({failures}), stopping checks early.")
            break

    if failures == 0:
        print(f"  All {len(check_ids)} sampled graphs passed consistency checks.")
    else:
        print(f"  {failures} failures found in {len(check_ids)} sampled graphs.")
    return failures


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)

    if args.max_graphs is not None:
        cfg.setdefault("build_options", {})["max_graphs"] = args.max_graphs

    max_graphs = cfg.get("build_options", {}).get("max_graphs", None)
    max_samples = args.max_samples

    processed_dir = cfg["processed_data_dir"]
    graph_dir = processed_dir / cfg["graph_subdir"]
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 0
    # ============================================================
    global_data = phase0_setup(cfg)
    sample_ids = global_data["sample_ids"]
    loadcase_ids = global_data["loadcase_ids"]

    if max_samples:
        sample_ids = sample_ids[:max_samples]
        print(f"  [debug] Limited to {max_samples} samples")

    # ============================================================
    # Phase 1: Build graphs
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 1: Graph construction")
    print("=" * 60)
    all_index_records = []
    total_expected = len(sample_ids) * len(loadcase_ids)
    running_count = 0

    for sid in sample_ids:
        if max_graphs is not None and running_count >= max_graphs:
            print(f"  [debug] Reached max_graphs={max_graphs}, stopping.")
            break
        t_s = time.time()
        records, running_count = build_one_sample_graphs(
            sample_id=sid,
            global_data=global_data,
            processed_graph_dir=graph_dir,
            max_graphs=max_graphs,
            running_count=running_count,
        )
        all_index_records.extend(records)
        print(f"  Sample {sid}: {len(records)} graphs ({time.time()-t_s:.1f}s, total={running_count})")

    actual_count = len(all_index_records)
    print(f"\n  Total graphs built: {actual_count}")

    # ============================================================
    # Phase 2: Index & metadata
    # ============================================================
    print("\n" + "-" * 40)
    print("Phase 2: Save index & metadata")
    print("-" * 40)

    index_df = pd.DataFrame(all_index_records)
    index_csv = processed_dir / "index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8")
    print(f"  Saved: {index_csv} (rows={len(index_df)})")

    metadata = {
        "dataset_name": "hetero_graph_dataset_v2",
        "description": "Heterogeneous graph dataset v2 — corrected schema for Multi-Scale PI-HGNN",
        "version": "2.0",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_graphs": actual_count,
        "num_samples": len(sample_ids),
        "num_loadcases": len(loadcase_ids),
        "topology_count": 1,
        "node_types": NODE_TYPES,
        "edge_types": [list(et) for et in EDGE_TYPES],
        "raw_data_dir": str(cfg["raw_data_dir"]),
        "processed_data_dir": str(processed_dir),
        "dataset_type": "heterogeneous",
        "tasks": [
            {"node_type": "mesh_node", "target_key": "y_disp", "target_dim": 6},
            {"node_type": "beam_element", "target_key": "y_force", "target_dim": 12},
        ],
        "link_element_status": (
            "link_element is NOT a node type. rigid_elastic_links.csv is "
            "modeled as mesh_node ↔ structural_link ↔ mesh_node edge."
        ),
        "note": (
            "All 70 samples share identical topology (topology_count=1). "
            "3 node types only (mesh_node, beam_element, plate_element). "
            "rigid_elastic_links.csv → structural_link edge. "
            "Reverse membership edges carry edge_attr. "
            "structural_link edge_attr is 10-d including is_rigid."
        ),
    }
    meta_path = processed_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(metadata), f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path}")

    # schema.json
    schema = build_schema_metadata()
    schema_path = processed_dir / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {schema_path}")

    # ============================================================
    # Phase 3: Splits
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 3: Split generation")
    print("=" * 60)
    split_dir = processed_dir / cfg["split_dir"]
    splits = generate_split_files(
        sample_ids=sample_ids,
        loadcase_ids=loadcase_ids,
        split_dir=split_dir,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        random_seed=cfg["random_seed"],
    )
    for name in ["split_by_sample", "split_by_loadcase"]:
        s = splits[name]
        total = len(s["train"]) + len(s["val"]) + len(s["test"])
        print(f"  {name}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])} total={total}")

    # Verify no overlap
    for split_name, split_data in splits.items():
        mode = "by_sample" if "sample" in split_name else "by_loadcase"
        train_ids_set = set(split_data["train"])
        val_ids_set = set(split_data["val"])
        test_ids_set = set(split_data["test"])

        if mode == "by_sample":
            mask_train = index_df["sample_id"].isin(train_ids_set)
            mask_val = index_df["sample_id"].isin(val_ids_set)
            mask_test = index_df["sample_id"].isin(test_ids_set)
        else:
            mask_train = index_df["loadcase_id"].isin(train_ids_set)
            mask_val = index_df["loadcase_id"].isin(val_ids_set)
            mask_test = index_df["loadcase_id"].isin(test_ids_set)

        assert not (mask_train & mask_val).any(), f"Overlap between train and val in {split_name}"
        assert not (mask_train & mask_test).any(), f"Overlap between train and test in {split_name}"
        assert not (mask_val & mask_test).any(), f"Overlap between val and test in {split_name}"
        print(f"  [ok] {split_name}: indices verified, no overlap")

    # ============================================================
    # Phase 4: Training statistics (Welford)
    # ============================================================
    if not args.skip_stats:
        compute_training_stats(
            processed_dir=processed_dir,
            index_df=index_df,
            split=splits["split_by_sample"],
            split_mode="by_sample",
        )
    else:
        print("\n" + "=" * 60)
        print("Phase 4: SKIPPED (--skip-stats)")
        print("=" * 60)

    # ============================================================
    # Quality checks
    # ============================================================
    n_fail = run_quality_checks(processed_dir, index_df, global_data)

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE — {elapsed:.1f}s total")
    print("=" * 60)
    print(f"  Output:        {processed_dir}")
    print(f"  Graphs built:  {actual_count} / {total_expected}")
    print(f"  Quality fails: {n_fail}")
    print(f"  Node types:    {len(NODE_TYPES)}  {NODE_TYPES}")
    print(f"  Edge types:    {len(EDGE_TYPES)}")
    print(f"  Tasks:         mesh_node.y_disp (6d), beam_element.y_force (12d)")
    print(f"  link_element:  REMOVED (→ structural_link edge)")

    if n_fail > 0:
        print("\n  WARNING: Some quality checks failed. Review before use.")
        sys.exit(1)
    else:
        print("\n  All checks passed. Dataset ready for Stage 2.")


if __name__ == "__main__":
    main()

"""Encode raw arrays and build the canonical PyG heterogeneous graph."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data.three_topology_v2.graph_io import ThreeTopoData

from . import schema

NODE_TYPES = ["mesh_node", "beam_element", "plate_element"]
EDGE_TYPES = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]
TOPOLOGY_CODES = {"topology1": 1, "topology2": 2, "topology3": 3}
COORDINATE_CODES = {"Global": 0, "Nodal": 1}


def encode_raw(raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    beta = np.deg2rad(raw["beam_raw"][:, 15])
    beam = np.concatenate([
        raw["beam_raw"][:, :15],
        np.sin(beta)[:, None], np.cos(beta)[:, None],
        raw["beam_raw"][:, 16:17],
    ], axis=1)
    p = raw["plate_raw"]
    ptype = np.rint(p[:, 5]).astype(np.int64)
    plate = np.concatenate([
        p[:, :4],
        np.sin(np.deg2rad(p[:, 4]))[:, None],
        np.cos(np.deg2rad(p[:, 4]))[:, None],
        (ptype == 0)[:, None].astype(np.float64),
        (ptype == 1)[:, None].astype(np.float64),
        (~np.isin(ptype, [0, 1]))[:, None].astype(np.float64),
    ], axis=1)
    return {"node_x": raw["node_x"].copy(), "beam_x": beam, "plate_x": plate}


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.array(a, copy=True), dtype=dtype)


def build_graph(raw: Dict[str, Any], dtype=torch.float32) -> ThreeTopoData:
    enc = encode_raw(raw)
    data = ThreeTopoData()
    data["mesh_node"].x = _t(enc["node_x"], dtype)
    data["mesh_node"].raw_x = _t(raw["node_x"], torch.float64)
    data["mesh_node"].y_disp = _t(raw["node_y"], dtype)
    data["mesh_node"].y_disp_raw = _t(raw["node_y"], torch.float64)
    data["mesh_node"].raw_id = _t(raw["node_ids"], torch.long)
    data["mesh_node"].support_flag = _t(raw["node_support"], dtype)

    data["beam_element"].x = _t(enc["beam_x"], dtype)
    data["beam_element"].raw_x = _t(raw["beam_raw"], torch.float64)
    data["beam_element"].y_force = _t(raw["beam_y"], dtype)
    data["beam_element"].y_force_raw = _t(raw["beam_y"], torch.float64)
    data["beam_element"].raw_id = _t(raw["beam_ids"], torch.long)
    data["beam_element"].node_index = _t(raw["beam_node_index"], torch.long)
    data["beam_element"].section_id = _t(raw["beam_sec_id"], torch.long)
    data["beam_element"].material_id = _t(raw["beam_mat_id"], torch.long)
    data["beam_element"].end_section_29 = _t(raw["beam_end_prop"], torch.float64)
    data["beam_element"].end_section_mask = _t(raw["beam_end_mask"], torch.bool)
    data["beam_element"].beta_angle_deg = _t(raw["beam_raw"][:, 15], torch.float64)
    data["beam_element"].is_tapered = _t(raw["beam_raw"][:, 16], torch.float64)

    data["plate_element"].x = _t(enc["plate_x"], dtype)
    data["plate_element"].raw_x = _t(raw["plate_raw"], torch.float64)
    data["plate_element"].raw_id = _t(raw["plate_ids"], torch.long)
    data["plate_element"].node_index = _t(raw["plate_node_index"], torch.long)

    # Membership relations (mesh -> beam/plate and reverse).
    mb_src = raw["beam_node_index"].reshape(-1)
    mb_dst = np.repeat(np.arange(len(raw["beam_ids"]), dtype=np.int64), 2)
    mb_role = np.tile(np.array([0, 1], dtype=np.float64), len(raw["beam_ids"]))
    data["mesh_node", "belongs_to_beam", "beam_element"].edge_index = _t(
        np.stack([mb_src, mb_dst]), torch.long)
    data["mesh_node", "belongs_to_beam", "beam_element"].edge_attr = _t(mb_role[:, None], dtype)
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_index = _t(
        np.stack([mb_dst, mb_src]), torch.long)
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_attr = _t(mb_role[:, None], dtype)

    mp_src = raw["plate_node_index"].reshape(-1)
    mp_dst = np.repeat(np.arange(len(raw["plate_ids"]), dtype=np.int64), 4)
    mp_role = np.tile(np.arange(4, dtype=np.float64), len(raw["plate_ids"]))
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_index = _t(
        np.stack([mp_src, mp_dst]), torch.long)
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_attr = _t(mp_role[:, None], dtype)
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_index = _t(
        np.stack([mp_dst, mp_src]), torch.long)
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_attr = _t(mp_role[:, None], dtype)

    sl = data["mesh_node", "structural_link", "mesh_node"]
    sl.edge_index = _t(raw["link_edge_index"], torch.long)
    sl.edge_attr = _t(raw["link_attr"], dtype)
    sl.link_id = _t(raw["link_ids"], torch.long)
    sl.link_type = _t(raw["link_attr"][:, 8], torch.long)
    sl.is_rigid = _t(raw["link_attr"][:, 9], torch.long)

    # Constraint sidecars + reference offsets used by ThreeTopoData.__inc__.
    data.support_id = _t(raw["support_ids"], torch.long)
    data.support_node_index = _t(raw["support_node_index"], torch.long)
    data.support_boundary_info = _t(raw["support_flags"], dtype)
    data.beam_release_id = _t(raw["release_ids"], torch.long)
    data.beam_release_element_index = _t(raw["release_beam_index"], torch.long)
    data.beam_release_node_index = _t(raw["release_node_index"], torch.long)
    data.beam_release_info = _t(raw["release_info"], dtype)
    data.beam_release_edge_element_index = _t(raw["release_edge_beam"], torch.long)
    data.beam_release_edge_node_index = _t(raw["release_edge_node"], torch.long)
    data.beam_release_edge_role = _t(raw["release_edge_role"], torch.long)
    data.beam_release_edge_info = _t(raw["release_edge_info"], dtype)
    data.master_slave_id = _t(raw["master_ids"], torch.long)
    data.master_slave_node_index = _t(raw["master_node_index"], torch.long)
    data.master_slave_boundary_info = _t(raw["master_info"], dtype)
    data.constraint_equation_count = torch.tensor([raw["n_equations"]], dtype=torch.long)
    data.topology_code = torch.tensor([TOPOLOGY_CODES[raw["topology_token"]]], dtype=torch.long)
    data.coordinate_system_code = torch.tensor(
        [COORDINATE_CODES.get(raw["coordinate_system"], -1)], dtype=torch.long)
    data.sample_index = torch.tensor([-1], dtype=torch.long)
    return data


def apply_normalization_(data, stats: Dict[str, Any]) -> None:
    """Apply frozen training-side transforms to encoded inputs/labels."""
    import numpy as _np
    for key, store in (("node_x", "mesh_node"), ("beam_x", "beam_element"), ("plate_x", "plate_element")):
        cols = stats["arrays"][key]
        mean = _t(cols["mean"], torch.float32)
        scale = _t(cols["scale"], torch.float32)
        kinds = cols["kind"]
        x = data[store].x
        out = x.clone()
        for j, kind in enumerate(kinds):
            if kind == "zscore":
                out[:, j] = (x[:, j] - mean[j]) / scale[j]
            elif kind == "zero_preserve_rms":
                out[:, j] = x[:, j] / scale[j]
            # preserve: leave unchanged
        data[store].x = out
    for key, store, attr in (("y_disp", "mesh_node", "y_disp"), ("y_force", "beam_element", "y_force")):
        cols = stats["arrays"][key]
        y = data[store][attr]
        m = _t(cols["mean"], y.dtype); s = _t(cols["scale"], y.dtype)
        data[store][attr] = (y - m) / s

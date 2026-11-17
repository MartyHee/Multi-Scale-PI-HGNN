"""Graph JSON -> ``ThreeTopoData`` adapter for the v1 graph contract.

This mirrors the accepted reference reader vendored in the data package
(``scripts/three_topology_pyg_dataset.py``) but:

* resolves every path against an explicit *package root* (never the index dir);
* never writes cache/results into the source package;
* returns a ``HeteroData`` subclass whose ``__inc__`` makes the top-level
  constraint reference tensors (support / beam-release / master-slave) get the
  correct per-store offset during mixed batching.

Contract preserved: node 15/6, beam 15/12, plate 6, five relations, API link
type keyed by link ID, I/J direction verified against endpoint geometry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from . import paths

NODE_TYPES: List[str] = ["mesh_node", "beam_element", "plate_element"]
EDGE_TYPES = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]

TOPOLOGY_CODES = {"topology1": 1, "topology2": 2, "topology3": 3}
COORDINATE_CODES = {"Global": 0, "Nodal": 1}

NODE_FEATURE_DIM = 15
NODE_LABEL_DIM = 6
BEAM_FEATURE_DIM = 15
BEAM_LABEL_DIM = 12
PLATE_FEATURE_DIM = 6
LINK_ATTR_DIM = 10

# Expected unit/order declarations from the dataset contract (manifest) versus
# the graph ``contract.features`` / ``contract.units``.
EXPECTED_UNITS = {
    "node_x": ["m", "m", "m", "N", "N", "N", "N*m", "N*m", "N*m",
               "0/1", "0/1", "0/1", "0/1", "0/1", "0/1"],
    "node_y": ["m", "m", "m", "rad", "rad", "rad"],
    "beam_x": ["m^2", "m^4", "m^4", "m^4", "m^2", "m^4", "m^4", "m^4",
               "Pa", "1", "N/m^3", "m", "1", "1", "1"],
    "beam_y": ["N", "N", "N", "N*m", "N*m", "N*m", "N", "N", "N", "N*m", "N*m", "N*m"],
    "plate_x": ["m", "Pa", "1", "N/m^3", "deg (API value)", "enum"],
    "link_x": ["N/m", "N/m", "N/m", "N*m/rad", "N*m/rad", "N*m/rad",
               "deg (API value)", "1", "API enum", "0/1"],
}

EXPECTED_FEATURES = {
    "node_x": ["X", "Y", "Z", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
               "Dx_fix", "Dy_fix", "Dz_fix", "Rx_fix", "Ry_fix", "Rz_fix"],
    "node_y": ["dx", "dy", "dz", "rx", "ry", "rz"],
    "beam_x": ["Area_I", "Ix_I", "Iy_I", "Iz_I", "Area_J", "Ix_J", "Iy_J",
               "Iz_J", "ElasticModulus", "PoissonRatio", "UnitWeight",
               "Length", "CosX", "CosY", "CosZ"],
    "beam_y": ["I_Fx", "I_Fy", "I_Fz", "I_Mx", "I_My", "I_Mz",
               "J_Fx", "J_Fy", "J_Fz", "J_Mx", "J_My", "J_Mz"],
    "plate_x": ["ThicknessValue", "ElasticModulus", "PoissonRatio",
                "UnitWeight", "BetaAngle", "PlateType"],
    "link_x": ["Kx", "Ky", "Kz", "Krx", "Kry", "Krz", "BetaAngle",
               "DistanceRatio", "ElasticLinkType", "is_rigid"],
}


class GraphContractError(RuntimeError):
    pass


def _fail(sample_id: str, field: str, message: str) -> None:
    raise GraphContractError(f"[{sample_id}] {field}: {message}")


def _as_f64(value: Any, sample_id: str, field: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _fail(sample_id, field, f"not numeric: {exc}")
    if not np.all(np.isfinite(arr)):
        _fail(sample_id, field, "contains non-finite values")
    return arr


def _as_i64(value: Any, sample_id: str, field: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        _fail(sample_id, field, f"not integer: {exc}")
    return arr


def _tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype)


class ThreeTopoData(HeteroData):
    """HeteroData whose constraint references are offset per store on batching.

    ``support_node_index`` / ``beam_release_node_index`` / ``master_slave_node_index``
    / ``beam_release_edge_node_index`` are *mesh-node* local indices;
    ``beam_release_element_index`` / ``beam_release_edge_element_index`` are
    *beam-element* local indices.  Raw ``*_id`` fields are source IDs and must
    never be offset, so they fall through to the default (0).
    """

    _NODE_INC_KEYS = {
        "support_node_index",
        "beam_release_node_index",
        "beam_release_edge_node_index",
        "master_slave_node_index",
    }
    _BEAM_INC_KEYS = {
        "beam_release_element_index",
        "beam_release_edge_element_index",
    }

    def __inc__(self, key, value, *args, **kwargs):
        if key in self._NODE_INC_KEYS:
            return self["mesh_node"].num_nodes
        if key in self._BEAM_INC_KEYS:
            return self["beam_element"].num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def _merge_membership(edge_obj, member_count, roles, sample_id, relation) -> np.ndarray:
    edge_index = _as_i64(edge_obj["edge_index"], sample_id, f"{relation}.edge_index")
    edge_attr = _as_i64(edge_obj["edge_attr"], sample_id, f"{relation}.edge_attr").reshape(-1)
    if edge_index.shape != (2, member_count * roles):
        _fail(sample_id, f"{relation}.edge_index",
              f"shape {edge_index.shape} != (2, {member_count * roles})")
    if edge_attr.size != member_count * roles:
        _fail(sample_id, f"{relation}.edge_attr", f"size {edge_attr.size} != {member_count*roles}")
    out = np.full((member_count, roles), -1, dtype=np.int64)
    for node_idx, member_idx, role in zip(edge_index[0], edge_index[1], edge_attr):
        if member_idx < 0 or member_idx >= member_count:
            _fail(sample_id, f"{relation}.edge_index", f"member index {member_idx} out of range")
        if role < 0 or role >= roles:
            _fail(sample_id, f"{relation}.edge_attr", f"role {role} out of range")
        if out[member_idx, role] != -1:
            _fail(sample_id, relation, f"duplicate role {role} for member {member_idx}")
        out[member_idx, role] = node_idx
    if np.any(out < 0):
        _fail(sample_id, relation, "missing member/role entries")
    return out


def _check_reverse(forward, reverse, sample_id, fwd_name, rev_name) -> None:
    f_idx = _as_i64(forward["edge_index"], sample_id, f"{fwd_name}.edge_index")
    r_idx = _as_i64(reverse["edge_index"], sample_id, f"{rev_name}.edge_index")
    f_attr = _as_i64(forward["edge_attr"], sample_id, f"{fwd_name}.edge_attr").reshape(-1)
    r_attr = _as_i64(reverse["edge_attr"], sample_id, f"{rev_name}.edge_attr").reshape(-1)
    if f_idx.shape != r_idx.shape or f_attr.size != r_attr.size:
        _fail(sample_id, rev_name, "reverse edge array size differs")
    if not np.array_equal(f_idx[0], r_idx[1]) or not np.array_equal(f_idx[1], r_idx[0]):
        _fail(sample_id, rev_name, "reverse edge_index is not the flipped forward relation")
    if not np.array_equal(f_attr, r_attr):
        _fail(sample_id, rev_name, "reverse edge_attr differs from forward edge_attr")


def verify_contract_units(graph: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    """Check graph contract feature names/units against the frozen declarations."""
    contract = graph.get("contract") or {}
    feats = contract.get("features") or {}
    units = contract.get("units") or {}
    problems = []
    for mat, expected in EXPECTED_FEATURES.items():
        got = feats.get(mat)
        if got is not None and list(got) != expected:
            problems.append(f"{mat}.features mismatch")
    for mat, expected in EXPECTED_UNITS.items():
        got = units.get(mat)
        if got is not None and list(got) != expected:
            problems.append(f"{mat}.units mismatch: {got}")
    if problems:
        _fail(sample_id, "contract", "; ".join(problems))
    return {"sample_id": sample_id,
            "coordinate_system": contract.get("coordinate_system"),
            "node_local_axes": contract.get("node_local_axes")}


def build_hetero_data(graph: Dict[str, Any], row: Dict[str, Any],
                      dtype: torch.dtype = torch.float32) -> ThreeTopoData:
    """Convert one parsed graph JSON into a validated ``ThreeTopoData``."""
    sample_id = row["sample_id"]
    if graph.get("schema") != "qt-three-topology-graph-v1":
        _fail(sample_id, "schema", f"unexpected {graph.get('schema')!r}")
    contract = graph.get("contract") or {}
    for f, expect in (("sample_id", sample_id), ("topology_token", row["topology_token"]),
                      ("design_id", row["design_id"]),
                      ("load_field_hash", row["load_field_hash"]),
                      ("sample_key", row["sample_key"])):
        if contract.get(f) != expect:
            _fail(sample_id, f"contract.{f}", f"{contract.get(f)!r} != index {expect!r}")
    verify_contract_units(graph, sample_id)

    coordinate_system = contract.get("coordinate_system")
    if coordinate_system not in COORDINATE_CODES:
        _fail(sample_id, "contract.coordinate_system", f"unknown {coordinate_system!r}")
    topology_code = TOPOLOGY_CODES.get(row["topology_token"])
    if topology_code is None:
        _fail(sample_id, "topology_token", f"unknown {row['topology_token']!r}")

    node, beam, plate = graph["node"], graph["beam"], graph["plate"]
    node_ids = _as_i64(node["ids"], sample_id, "node.ids")
    beam_ids = _as_i64(beam["ids"], sample_id, "beam.ids")
    plate_ids = _as_i64(plate["ids"], sample_id, "plate.ids")
    node_count, beam_count, plate_count = len(node_ids), len(beam_ids), len(plate_ids)
    for name, ids in (("node.ids", node_ids), ("beam.ids", beam_ids), ("plate.ids", plate_ids)):
        if len(np.unique(ids)) != len(ids):
            _fail(sample_id, name, "duplicate raw IDs")
    id_to_idx = {int(v): i for i, v in enumerate(node_ids)}

    node_x = _as_f64(node["x"], sample_id, "node.x")
    node_y = _as_f64(node["y"], sample_id, "node.y")
    beam_x = _as_f64(beam["x"], sample_id, "beam.x")
    beam_y = _as_f64(beam["y"], sample_id, "beam.y")
    plate_x = _as_f64(plate["x"], sample_id, "plate.x")
    for name, arr, shape in (
        ("node.x", node_x, (node_count, NODE_FEATURE_DIM)),
        ("node.y", node_y, (node_count, NODE_LABEL_DIM)),
        ("beam.x", beam_x, (beam_count, BEAM_FEATURE_DIM)),
        ("beam.y", beam_y, (beam_count, BEAM_LABEL_DIM)),
        ("plate.x", plate_x, (plate_count, PLATE_FEATURE_DIM)),
    ):
        if arr.shape != shape:
            _fail(sample_id, name, f"shape {arr.shape} != {shape}")

    counts = graph.get("counts") or {}
    for key, value in (("nodes", node_count), ("beams", beam_count), ("plates", plate_count)):
        if int(counts.get(key, -1)) != value:
            _fail(sample_id, f"counts.{key}", f"{counts.get(key)} != {value}")

    edges = graph["edges"]
    node_to_beam = _merge_membership(edges["node_to_beam"], beam_count, 2, sample_id, "node_to_beam")
    node_to_plate = _merge_membership(edges["node_to_plate"], plate_count, 4, sample_id, "node_to_plate")
    _check_reverse(edges["node_to_beam"], edges["beam_to_node"], sample_id, "node_to_beam", "beam_to_node")
    _check_reverse(edges["node_to_plate"], edges["plate_to_node"], sample_id, "node_to_plate", "plate_to_node")
    if np.any(node_to_beam < 0) or np.any(node_to_beam >= node_count):
        _fail(sample_id, "node_to_beam", "node index out of range")
    if np.any(node_to_plate < 0) or np.any(node_to_plate >= node_count):
        _fail(sample_id, "node_to_plate", "node index out of range")

    # Exact I/J direction.
    length = beam_x[:, 11]
    if np.any(length <= 0):
        _fail(sample_id, "beam.x.Length", f"non-positive length {float(length.min())}")
    delta = node_x[node_to_beam[:, 1], :3] - node_x[node_to_beam[:, 0], :3]
    geo_length = np.linalg.norm(delta, axis=1)
    rel_err = np.abs(geo_length - length) / np.maximum(length, 1e-30)
    if float(rel_err.max()) > 1e-9:
        _fail(sample_id, "beam.x.Length", "endpoint distance != feature Length")
    cosines = delta / geo_length[:, None]
    if float(np.abs(cosines - beam_x[:, 12:15]).max()) > 1e-9:
        _fail(sample_id, "beam.x.Cos", "direction cosines disagree with I/J order")

    # Structural links.
    link = graph["link"]
    structural = edges["structural_link"]
    link_ids = _as_i64(link["ids"], sample_id, "link.ids")
    sl_ids = _as_i64(structural["ids"], sample_id, "structural_link.ids")
    if not np.array_equal(link_ids, sl_ids):
        _fail(sample_id, "structural_link.ids", "IDs differ from link.ids")
    link_count = len(link_ids)
    if int(counts.get("links", -1)) != link_count:
        _fail(sample_id, "counts.links", f"{counts.get('links')} != {link_count}")
    sl_index = _as_i64(structural["edge_index"], sample_id, "structural_link.edge_index")
    if sl_index.shape != (2, link_count):
        _fail(sample_id, "structural_link.edge_index", f"shape {sl_index.shape}")
    start_local = np.array([id_to_idx.get(int(v), -1) for v in link["start_node_id"]], dtype=np.int64)
    end_local = np.array([id_to_idx.get(int(v), -1) for v in link["end_node_id"]], dtype=np.int64)
    if np.any(start_local < 0) or np.any(end_local < 0):
        _fail(sample_id, "link.start/end_node_id", "node ID missing")
    if not np.array_equal(sl_index[0], start_local) or not np.array_equal(sl_index[1], end_local):
        _fail(sample_id, "structural_link.edge_index", "endpoints differ from raw link start/end")

    link_type = _as_i64(link["link_type"], sample_id, "link.link_type")
    boundary = _as_f64(link["boundary_info"], sample_id, "link.boundary_info")
    if boundary.shape != (link_count, 6):
        _fail(sample_id, "link.boundary_info", f"shape {boundary.shape}")
    beta = _as_f64(link["beta_angle"], sample_id, "link.beta_angle").reshape(-1)
    dis_ratio = _as_f64(link["dis_ratio"], sample_id, "link.dis_ratio").reshape(-1)
    kx = _as_f64(link["kx"], sample_id, "link.kx").reshape(-1)
    gap = _as_f64(link["gap"], sample_id, "link.gap").reshape(-1)
    friction = _as_f64(link["friction"], sample_id, "link.friction").reshape(-1)
    for name, arr in (("beta_angle", beta), ("dis_ratio", dis_ratio), ("kx", kx),
                      ("gap", gap), ("friction", friction)):
        if arr.size != link_count:
            _fail(sample_id, f"link.{name}", f"size {arr.size} != {link_count}")
    canonical = link["canonical_kind"]
    if len(canonical) != link_count:
        _fail(sample_id, "link.canonical_kind", f"size {len(canonical)} != {link_count}")
    is_rigid = np.array([1 if v == "rigid" else 0 for v in canonical], dtype=np.int64)
    if np.any((link_type == 2) != (is_rigid == 1)):
        _fail(sample_id, "link.canonical_kind", "rigid marker inconsistent with API type 2")

    original_attr = _as_f64(structural["edge_attr"], sample_id, "structural_link.edge_attr")
    if original_attr.shape != (link_count, LINK_ATTR_DIM):
        _fail(sample_id, "structural_link.edge_attr", f"shape {original_attr.shape}")
    expected_prefix = np.concatenate([boundary, beta[:, None], dis_ratio[:, None]], axis=1)
    if not np.allclose(original_attr[:, :8], expected_prefix, rtol=0.0, atol=0.0):
        _fail(sample_id, "structural_link.edge_attr[:,0:8]", "stiffness/beta/distance differ from raw link")
    type_mismatch = original_attr[:, 8].astype(np.int64) != link_type
    if np.any(original_attr[:, 9].astype(np.int64) != is_rigid):
        _fail(sample_id, "structural_link.edge_attr[:,9]", "is_rigid differs from raw marker")
    if row["topology_token"] != "topology1" and np.any(type_mismatch):
        _fail(sample_id, "structural_link.edge_attr[:,8]", "API type differs (outside topology-1 legacy)")
    corrected_attr = original_attr.copy()
    corrected_attr[:, 8] = link_type.astype(np.float64)
    corrected_attr[:, 9] = is_rigid.astype(np.float64)

    # Supports / releases / master-slave.
    support = graph.get("support") or {}
    support_ids = _as_i64(support.get("ids", []), sample_id, "support.ids")
    support_nodes = np.array([id_to_idx.get(int(v), -1) for v in support.get("node_id", [])],
                             dtype=np.int64)
    if np.any(support_nodes < 0):
        _fail(sample_id, "support.node_id", "node ID missing")
    support_info = _as_f64(support.get("boundary_info", []), sample_id, "support.boundary_info")
    if support_info.size == 0:
        support_info = support_info.reshape(0, 6)
    if support_info.shape != (len(support_nodes), 6):
        _fail(sample_id, "support.boundary_info", f"shape {support_info.shape}")
    if len(support_nodes) and not np.array_equal(node_x[support_nodes, 9:15], support_info):
        _fail(sample_id, "support.boundary_info", "support flags differ from node.x columns 9:15")

    constraints = graph.get("constraints") or {}
    releases = constraints.get("beam_releases") or []
    beam_id_to_index = {int(v): i for i, v in enumerate(beam_ids)}
    release_element = np.array([int(r["element_id"]) for r in releases], dtype=np.int64)
    if any(int(e) not in beam_id_to_index for e in release_element):
        _fail(sample_id, "constraints.beam_releases.element_id", "element ID missing")
    release_element_index = np.array([beam_id_to_index[int(e)] for e in release_element], dtype=np.int64)
    release_node_i = np.array([id_to_idx.get(int(r["node_i"]), -1) for r in releases], dtype=np.int64)
    release_node_j = np.array([id_to_idx.get(int(r["node_j"]), -1) for r in releases], dtype=np.int64)
    if np.any(release_node_i < 0) or np.any(release_node_j < 0):
        _fail(sample_id, "constraints.beam_releases.node_i/node_j", "node ID missing")
    release_info_i = np.array([r["info_i"] for r in releases], dtype=np.float64).reshape(-1, 6)
    release_info_j = np.array([r["info_j"] for r in releases], dtype=np.float64).reshape(-1, 6)
    release_ids = np.array([int(r["id"]) for r in releases], dtype=np.int64)
    for k in range(len(releases)):
        b = release_element_index[k]
        if int(node_to_beam[b, 0]) != int(release_node_i[k]) or int(node_to_beam[b, 1]) != int(release_node_j[k]):
            _fail(sample_id, f"constraints.beam_releases[{k}]", "node_i/node_j differ from beam I/J")

    release_edges = constraints.get("beam_release_edge") or []
    if len(release_edges) != 2 * len(releases):
        _fail(sample_id, "constraints.beam_release_edge",
              f"{len(release_edges)} records for {len(releases)} releases")
    release_edge_element = np.array([beam_id_to_index[int(r["element_id"])] for r in release_edges], dtype=np.int64)
    release_edge_node = np.array([id_to_idx.get(int(r["node_id"]), -1) for r in release_edges], dtype=np.int64)
    release_edge_role = np.array([0 if r["end"] == "I" else 1 for r in release_edges], dtype=np.int64)
    release_edge_info = np.array([r["info"] for r in release_edges], dtype=np.float64).reshape(-1, 6)
    for k, rec in enumerate(release_edges):
        if rec["end"] not in ("I", "J"):
            _fail(sample_id, f"constraints.beam_release_edge[{k}].end", "must be I or J")
        if int(node_to_beam[release_edge_element[k], release_edge_role[k]]) != int(release_edge_node[k]):
            _fail(sample_id, f"constraints.beam_release_edge[{k}]", "node does not match beam end")

    master_slave = constraints.get("master_slave") or []
    master_nodes = np.array([id_to_idx.get(int(r["master_id"]), -1) for r in master_slave], dtype=np.int64)
    slave_nodes = np.array([id_to_idx.get(int(r["slave_id"]), -1) for r in master_slave], dtype=np.int64)
    if np.any(master_nodes < 0) or np.any(slave_nodes < 0):
        _fail(sample_id, "constraints.master_slave", "node ID missing")
    ms_info = np.array([r["boundary_info"] for r in master_slave], dtype=np.float64).reshape(-1, 6)
    ms_ids = np.array([int(r["id"]) for r in master_slave], dtype=np.int64)
    ms_edges = constraints.get("master_slave_edge") or []
    if len(ms_edges) != len(master_slave):
        _fail(sample_id, "constraints.master_slave_edge", "record count differs from master_slave")
    for k, rec in enumerate(ms_edges):
        if rec.get("direction") != "master_to_slave":
            _fail(sample_id, f"constraints.master_slave_edge[{k}].direction", "not master_to_slave")
        if int(rec["master_id"]) != int(master_slave[k]["master_id"]) or int(rec["slave_id"]) != int(master_slave[k]["slave_id"]):
            _fail(sample_id, f"constraints.master_slave_edge[{k}]", "direction/endpoints differ")
    equations = constraints.get("constraint_equations") or []
    if len(equations) != int(counts.get("constraint_equations", 0)):
        _fail(sample_id, "constraints.constraint_equations", "count differs from graph counts")

    # ---- Build ThreeTopoData ----
    data = ThreeTopoData()
    data["mesh_node"].x = _tensor(node_x.copy(), dtype)
    data["mesh_node"].y_disp = _tensor(node_y.copy(), dtype)
    data["mesh_node"].raw_id = torch.as_tensor(node_ids, dtype=torch.long)
    data["mesh_node"].support_flag = _tensor(node_x[:, 9:15].copy(), dtype)
    data["beam_element"].x = _tensor(beam_x.copy(), dtype)
    data["beam_element"].y_force = _tensor(beam_y.copy(), dtype)
    data["beam_element"].raw_id = torch.as_tensor(beam_ids, dtype=torch.long)
    data["plate_element"].x = _tensor(plate_x.copy(), dtype)
    data["plate_element"].raw_id = torch.as_tensor(plate_ids, dtype=torch.long)

    data["mesh_node", "belongs_to_beam", "beam_element"].edge_index = _tensor(
        _as_i64(edges["node_to_beam"]["edge_index"], sample_id, "node_to_beam.edge_index").copy(), torch.long)
    data["mesh_node", "belongs_to_beam", "beam_element"].edge_attr = _tensor(
        _as_f64(edges["node_to_beam"]["edge_attr"], sample_id, "node_to_beam.edge_attr").reshape(-1, 1), dtype)
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_index = _tensor(
        _as_i64(edges["beam_to_node"]["edge_index"], sample_id, "beam_to_node.edge_index").copy(), torch.long)
    data["beam_element", "rev_belongs_to_beam", "mesh_node"].edge_attr = _tensor(
        _as_f64(edges["beam_to_node"]["edge_attr"], sample_id, "beam_to_node.edge_attr").reshape(-1, 1), dtype)
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_index = _tensor(
        _as_i64(edges["node_to_plate"]["edge_index"], sample_id, "node_to_plate.edge_index").copy(), torch.long)
    data["mesh_node", "belongs_to_plate", "plate_element"].edge_attr = _tensor(
        _as_f64(edges["node_to_plate"]["edge_attr"], sample_id, "node_to_plate.edge_attr").reshape(-1, 1), dtype)
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_index = _tensor(
        _as_i64(edges["plate_to_node"]["edge_index"], sample_id, "plate_to_node.edge_index").copy(), torch.long)
    data["plate_element", "rev_belongs_to_plate", "mesh_node"].edge_attr = _tensor(
        _as_f64(edges["plate_to_node"]["edge_attr"], sample_id, "plate_to_node.edge_attr").reshape(-1, 1), dtype)

    sl_store = data["mesh_node", "structural_link", "mesh_node"]
    sl_store.edge_index = _tensor(sl_index.copy(), torch.long)
    sl_store.edge_attr = _tensor(corrected_attr, dtype)
    sl_store.edge_attr_original = _tensor(original_attr, dtype)
    sl_store.link_id = torch.as_tensor(link_ids, dtype=torch.long)
    sl_store.link_type = torch.as_tensor(link_type, dtype=torch.long)
    sl_store.is_rigid = torch.as_tensor(is_rigid, dtype=torch.long)
    sl_store.legacy_type_column_mismatch = torch.as_tensor(type_mismatch, dtype=torch.bool)

    data.support_node_index = torch.as_tensor(support_nodes, dtype=torch.long)
    data.support_boundary_info = _tensor(support_info.copy(), dtype)
    data.support_id = torch.as_tensor(support_ids, dtype=torch.long)
    data.beam_release_id = torch.as_tensor(release_ids, dtype=torch.long)
    data.beam_release_element_index = torch.as_tensor(release_element_index, dtype=torch.long)
    data.beam_release_node_index = (torch.stack([torch.as_tensor(release_node_i, dtype=torch.long),
                                                 torch.as_tensor(release_node_j, dtype=torch.long)], dim=1)
                                    if len(releases) else torch.zeros((0, 2), dtype=torch.long))
    data.beam_release_info = (torch.stack([torch.as_tensor(release_info_i, dtype=dtype),
                                           torch.as_tensor(release_info_j, dtype=dtype)], dim=1)
                              if len(releases) else torch.zeros((0, 2, 6), dtype=dtype))
    data.beam_release_edge_element_index = torch.as_tensor(release_edge_element, dtype=torch.long)
    data.beam_release_edge_node_index = torch.as_tensor(release_edge_node, dtype=torch.long)
    data.beam_release_edge_role = torch.as_tensor(release_edge_role, dtype=torch.long)
    data.beam_release_edge_info = _tensor(release_edge_info.copy(), dtype)
    data.master_slave_id = torch.as_tensor(ms_ids, dtype=torch.long)
    data.master_slave_node_index = (torch.stack([torch.as_tensor(master_nodes, dtype=torch.long),
                                                 torch.as_tensor(slave_nodes, dtype=torch.long)], dim=1)
                                    if len(master_slave) else torch.zeros((0, 2), dtype=torch.long))
    data.master_slave_boundary_info = _tensor(ms_info.copy(), dtype)
    data.constraint_equation_count = torch.tensor([len(equations)], dtype=torch.long)
    data.sample_index = torch.tensor([-1], dtype=torch.long)
    data.topology_code = torch.tensor([topology_code], dtype=torch.long)
    data.coordinate_system_code = torch.tensor([COORDINATE_CODES[coordinate_system]], dtype=torch.long)
    data.legacy_link_type_column = torch.tensor([int(type_mismatch.any())], dtype=torch.long)
    return data


def load_graph(row: Dict[str, Any], root: Optional[Path] = None,
               dtype: torch.dtype = torch.float32, verify_hash: bool = False) -> ThreeTopoData:
    """Parse and convert one sample graph (path resolved under package root)."""
    import json

    from .index import sha256_bytes

    root = root or paths.package_root()
    path = paths.resolve_under_root(row["graph_path"], root)
    raw = path.read_bytes()
    if verify_hash:
        digest = sha256_bytes(raw)
        if digest != row["graph_sha256"]:
            _fail(row["sample_id"], "graph_sha256", f"{digest} != index {row['graph_sha256']}")
    graph = json.loads(raw.decode("utf-8"))
    data = build_hetero_data(graph, row, dtype=dtype)
    data.graph_contract = graph.get("contract") or {}
    return data

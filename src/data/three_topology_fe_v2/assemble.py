"""Assemble raw arrays from one FE v2 extraction/shard sample.

This is the single field-resolution layer shared by stats fitting, graph
construction and targeted checks.  It keeps raw float64 values; encoding and
normalization happen later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _as_float(v, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite value")
    return a


def _mat_map(ex: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(m["index"]): m for m in ex.get("GET-MATERIAL-DATA", [])}


def _thick_map(ex: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(t["thick_id"]): t for t in ex.get("GET-THICKNESS-DATA", [])}


def _section_ai(sec: Dict[str, Any], end: str, endpoint_prop: np.ndarray | None):
    """Return (A, Ix, Iy, Iz, is_tapered, prop29_or_none)."""
    if sec is None:
        raise ValueError("section object missing")
    if "Ax" in sec:
        return (float(sec["Ax"]), float(sec["Ix"]), float(sec["Iy"]), float(sec["Iz"]), False, None)
    begin, finish = sec.get("sec_begin"), sec.get("sec_end")
    if begin is None or finish is None:
        raise ValueError("section has neither Ax nor sec_begin/sec_end")
    tapered = True
    props = endpoint_prop
    chosen = begin if end == "I" else finish
    if props is not None:
        # API endpoint sec_property: 0/3/4/5 = Area/Ix/Iy/Iz.
        if len(props) != 29:
            raise ValueError(f"endpoint sec_property length {len(props)} != 29")
        return (float(props[0]), float(props[3]), float(props[4]), float(props[5]), tapered, props)
    return (float(chosen["Ax"]), float(chosen["Ix"]), float(chosen["Iy"]), float(chosen["Iz"]),
            tapered, None)


def assemble_raw(ctx: Dict[str, Any]) -> Dict[str, Any]:
    row, meta, ex, z, loads = ctx["row"], ctx["meta"], ctx["extraction"], ctx["arrays"], ctx["loads"]
    fid = row["field_id"]
    i = int(ctx["field_index"])
    node_ids = np.asarray(ctx["node_ids"], dtype=np.int64)
    beam_ids = np.asarray(ctx["beam_ids"], dtype=np.int64)
    n_nodes, n_beams = len(node_ids), len(beam_ids)
    node_by_id = {int(n): j for j, n in enumerate(node_ids)}
    beam_by_id = {int(b): j for j, b in enumerate(beam_ids)}

    node_raw_all = {int(n["Item1"]): n for n in ex["all"]["GET-NODE-DATA"]}
    if any(int(n) not in node_raw_all for n in node_ids):
        raise ValueError("node_id from response shard missing in GET-NODE-DATA")
    node_xyz = np.array([[node_raw_all[int(n)][k] for k in ("Item2", "Item3", "Item4")]
                         for n in node_ids], dtype=np.float64)
    if not np.isfinite(node_xyz).all():
        raise ValueError("non-finite node coordinates")

    # Verified sparse-load contract: absent node row is a zero background.
    if fid not in loads:
        raise ValueError(f"field {fid} missing from loads_before_solve rows")
    node_load = np.zeros((n_nodes, 6), dtype=np.float64)
    for r in loads[fid]:
        n = int(r["node_id"])
        if n not in node_by_id:
            raise ValueError(f"load node {n} not in response shard")
        vals = _as_float(r["load_info"], "node load")
        if vals.shape != (6,):
            raise ValueError("node load vector is not 6 components")
        node_load[node_by_id[n]] = vals

    node_support = np.zeros((n_nodes, 6), dtype=np.float64)
    support_ids, support_nodes, support_flags = [], [], []
    for s in ex["all"].get("GET-GENERAL-SUPPORT-DATA", []):
        n = int(s["node_id"])
        if n not in node_by_id:
            raise ValueError(f"support node {n} not in response shard")
        flags = np.asarray(s["boundary_info"], dtype=np.float64)
        if flags.shape != (6,):
            raise ValueError("support boundary_info is not 6 components")
        node_support[node_by_id[n]] = flags
        support_ids.append(int(s["index"])); support_nodes.append(node_by_id[n])
        support_flags.append(flags)
    support_ids = np.asarray(support_ids, dtype=np.int64)
    support_nodes = np.asarray(support_nodes, dtype=np.int64)
    support_flags = np.asarray(support_flags, dtype=np.float64).reshape(-1, 6)

    node_x = np.concatenate([node_xyz, node_load, node_support], axis=1)
    node_y = _as_float(z["node_disp"][i], "node_disp")
    if node_y.shape != (n_nodes, 6):
        raise ValueError(f"node_disp shape {node_y.shape} != {(n_nodes, 6)}")

    elements = {int(e["index"]): e for e in ex["all"].get("GET-ELEMENT-DATA", [])}
    materials, thicknesses = _mat_map(ex["all"]), _thick_map(ex["all"])
    sections = ex.get("sections", {}) or {}
    endpoints: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for ep in ex.get("endpoints", []) or []:
        endpoints[(int(ep["element_id"]), str(ep["end"]))] = ep

    beam_raw = np.zeros((n_beams, 17), dtype=np.float64)
    beam_node_ids = np.zeros((n_beams, 2), dtype=np.int64)
    beam_node_index = np.zeros((n_beams, 2), dtype=np.int64)
    beam_end_prop = np.full((n_beams, 2, 29), np.nan, dtype=np.float64)
    beam_end_mask = np.zeros((n_beams, 2), dtype=bool)
    beam_sec_id = np.zeros(n_beams, dtype=np.int64)
    beam_mat_id = np.zeros(n_beams, dtype=np.int64)
    for j, bid in enumerate(beam_ids):
        e = elements.get(int(bid))
        if e is None or int(e.get("ele_type", -1)) != 1:
            raise ValueError(f"beam {bid} not found as ele_type=1")
        ij = [int(x) for x in e["node_ids"]]
        if len(ij) != 2:
            raise ValueError(f"beam {bid}: node_ids length != 2")
        if any(n not in node_by_id for n in ij):
            raise ValueError(f"beam {bid}: endpoint node missing")
        ni, nj = node_by_id[ij[0]], node_by_id[ij[1]]
        p_i, p_j = node_xyz[ni], node_xyz[nj]
        delta = p_j - p_i
        length = float(np.linalg.norm(delta))
        if length <= 0:
            raise ValueError(f"beam {bid}: zero length")
        cos = delta / length
        sec_id = int(e["sec_id"]); mat_id = int(e["mat_id"])
        if mat_id not in materials:
            raise ValueError(f"beam {bid}: material {mat_id} missing")
        dinfo = materials[mat_id]["data_info"]
        if len(dinfo) < 3:
            raise ValueError(f"material {mat_id}: data_info < 3")
        sec = sections.get(str(sec_id), sections.get(sec_id))
        a_i, ix_i, iy_i, iz_i, tapered_i, prop_i = _section_ai(
            sec, "I", None)
        ep_i = endpoints.get((int(bid), "I"))
        ep_j = endpoints.get((int(bid), "J"))
        if ep_i is not None:
            prop_i = _as_float(ep_i["definition"]["sec_property"], "endpoint I sec_property")
            beam_end_prop[j, 0] = prop_i; beam_end_mask[j, 0] = True
            a_i, ix_i, iy_i, iz_i = [float(prop_i[k]) for k in (0, 3, 4, 5)]
        if ep_j is not None:
            prop_j = _as_float(ep_j["definition"]["sec_property"], "endpoint J sec_property")
            beam_end_prop[j, 1] = prop_j; beam_end_mask[j, 1] = True
            a_j, ix_j, iy_j, iz_j = [float(prop_j[k]) for k in (0, 3, 4, 5)]
        else:
            a_j, ix_j, iy_j, iz_j, tapered_j, _ = _section_ai(sec, "J", None)
        if not (tapered_i or tapered_j or ep_i is not None or ep_j is not None):
            # Normal section: both ends share the same parent properties.
            a_j, ix_j, iy_j, iz_j = a_i, ix_i, iy_i, iz_i
        tapered = bool(tapered_i or tapered_j or ep_i is not None or ep_j is not None)
        beam_raw[j] = [a_i, ix_i, iy_i, iz_i, a_j, ix_j, iy_j, iz_j,
                       float(dinfo[0]), float(dinfo[2]), float(dinfo[1]),
                       length, float(cos[0]), float(cos[1]), float(cos[2]),
                       float(e.get("beta_angle", 0.0)), float(tapered)]
        beam_node_ids[j] = ij
        beam_node_index[j] = [ni, nj]
        beam_sec_id[j] = sec_id; beam_mat_id[j] = mat_id
        if not np.isfinite(beam_raw[j]).all():
            raise ValueError(f"beam {bid}: non-finite raw feature")
    beam_y = _as_float(z["beam_force"][i], "beam_force")
    if beam_y.shape != (n_beams, 12):
        raise ValueError(f"beam_force shape {beam_y.shape} != {(n_beams, 12)}")

    plate_elems = [e for e in ex["all"].get("GET-ELEMENT-DATA", []) if int(e.get("ele_type", -1)) == 4]
    plate_raw = np.zeros((len(plate_elems), 6), dtype=np.float64)
    plate_ids = np.zeros(len(plate_elems), dtype=np.int64)
    plate_node_ids = np.zeros((len(plate_elems), 4), dtype=np.int64)
    plate_node_index = np.zeros((len(plate_elems), 4), dtype=np.int64)
    for j, e in enumerate(plate_elems):
        nids = [int(x) for x in e["node_ids"]]
        if len(nids) != 4 or any(n not in node_by_id for n in nids):
            raise ValueError(f"plate {e.get('index')}: invalid corner nodes")
        mat_id = int(e["mat_id"]); thick_id = int(e["sec_id"])
        if mat_id not in materials or thick_id not in thicknesses:
            raise ValueError(f"plate {e.get('index')}: missing material/thickness")
        dinfo = materials[mat_id]["data_info"]; th = thicknesses[thick_id]
        plate_raw[j] = [float(th["t"]), float(dinfo[0]), float(dinfo[2]), float(dinfo[1]),
                        float(e.get("beta_angle", 0.0)), float(e.get("plate_type", 0))]
        plate_ids[j] = int(e["index"])
        plate_node_ids[j] = nids
        plate_node_index[j] = [node_by_id[n] for n in nids]
    if not np.isfinite(plate_raw).all():
        raise ValueError("non-finite plate raw feature")

    # Structural links: preserve every same-endpoint connection.
    links = ex["all"].get("GET-ELASTIC-LINK-DATA", []) or []
    link_ids = np.zeros(len(links), dtype=np.int64)
    link_edge_index = np.zeros((2, len(links)), dtype=np.int64)
    link_attr = np.zeros((len(links), 10), dtype=np.float64)
    for j, l in enumerate(links):
        s, t = int(l["start_id"]), int(l["end_id"])
        if s not in node_by_id or t not in node_by_id:
            raise ValueError(f"link {l.get('index')}: endpoint node missing")
        info = _as_float(l["boundary_info"], "link boundary_info")
        if info.shape != (6,):
            raise ValueError("link boundary_info is not 6")
        typ = int(l["link_type"]); rigid = 1.0 if typ == 2 else 0.0
        link_ids[j] = int(l["index"]); link_edge_index[:, j] = [node_by_id[s], node_by_id[t]]
        link_attr[j] = np.concatenate([info, [float(l["beta_angle"]), float(l["dis_ratio"]),
                                             typ, rigid]])

    # Releases: raw true means constrained in the API record; do not translate
    # to the UI "release" checkbox here.
    releases = ex["all"].get("GET-BEAM-CONSTRAINT-DATA", []) or []
    release_ids = np.zeros(len(releases), dtype=np.int64)
    release_beam_index = np.zeros(len(releases), dtype=np.int64)
    release_node_index = np.zeros((len(releases), 2), dtype=np.int64)
    release_info = np.zeros((len(releases), 2, 6), dtype=np.float64)
    for j, r in enumerate(releases):
        bid = int(r["beam_id"])
        if bid not in beam_by_id:
            raise ValueError(f"release {r.get('index')}: beam {bid} not in sample")
        b = beam_by_id[bid]
        release_ids[j] = int(r["index"]); release_beam_index[j] = b
        release_node_index[j] = beam_node_index[b]
        release_info[j, 0] = _as_float(r["info_i"], "release info_i")
        release_info[j, 1] = _as_float(r["info_j"], "release info_j")
    release_edge_beam = np.repeat(release_beam_index, 2)
    release_edge_node = release_node_index.reshape(-1)
    release_edge_role = np.tile(np.array([0, 1], dtype=np.int64), len(releases))
    release_edge_info = release_info.reshape(-1, 6)

    masters = ex["all"].get("GET-MASTER-SLAVE-LINK-DATA", []) or []
    master_ids = np.zeros(len(masters), dtype=np.int64)
    master_node_index = np.zeros((len(masters), 2), dtype=np.int64)
    master_info = np.zeros((len(masters), 6), dtype=np.float64)
    for j, m in enumerate(masters):
        a, b = int(m["master_id"]), int(m["slave_id"])
        if a not in node_by_id or b not in node_by_id:
            raise ValueError(f"master-slave {m.get('index')}: node missing")
        master_ids[j] = int(m["index"]); master_node_index[j] = [node_by_id[a], node_by_id[b]]
        master_info[j] = _as_float(m["boundary_info"], "master boundary_info")

    equations = ex["all"].get("GET-CONSTRAINT-EQUATION-DATA", []) or []
    return {
        "sample_id": row["sample_id"], "topology_token": row["topology_token"],
        "design_id": row["design_id"], "field_id": fid, "family": meta.get("family"),
        "coordinate_system": meta.get("coordinate_system"),
        "node_ids": node_ids, "node_xyz": node_xyz, "node_load": node_load,
        "node_support": node_support, "node_x": node_x, "node_y": node_y,
        "beam_ids": beam_ids, "beam_node_ids": beam_node_ids,
        "beam_node_index": beam_node_index, "beam_raw": beam_raw, "beam_y": beam_y,
        "beam_end_prop": beam_end_prop, "beam_end_mask": beam_end_mask,
        "beam_sec_id": beam_sec_id, "beam_mat_id": beam_mat_id,
        "plate_ids": plate_ids, "plate_node_ids": plate_node_ids,
        "plate_node_index": plate_node_index, "plate_raw": plate_raw,
        "support_ids": support_ids, "support_node_index": support_nodes,
        "support_flags": support_flags,
        "link_ids": link_ids, "link_edge_index": link_edge_index, "link_attr": link_attr,
        "release_ids": release_ids, "release_beam_index": release_beam_index,
        "release_node_index": release_node_index, "release_info": release_info,
        "release_edge_beam": release_edge_beam, "release_edge_node": release_edge_node,
        "release_edge_role": release_edge_role, "release_edge_info": release_edge_info,
        "master_ids": master_ids, "master_node_index": master_node_index,
        "master_info": master_info, "n_equations": len(equations),
    }


def assemble_raw_shared(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble one sample from processed shared geometry/design/field caches.

    The response shard is the only per-sample raw file.  Geometry, supports,
    links/releases/master data are template-shared; beam/plate section state is
    design-shared; node loads are field-shared.  This function is numerically
    equivalent to :func:`assemble_raw` for the public/encoded views and the
    used end-section columns; the full 29-column endpoint sidecar remains
    traceable in the source extraction (see `assemble_raw`).
    """
    row, meta = ctx["row"], ctx["meta"]
    geom, design, field, z = ctx["geometry"], ctx["design"], ctx["field"], ctx["arrays"]
    i = int(ctx["field_index"])
    node_ids = np.asarray(geom["node_ids"], dtype=np.int64)
    node_xyz = np.asarray(geom["node_xyz"], dtype=np.float64)
    n_nodes = len(node_ids)
    node_load = np.asarray(field["loads"], dtype=np.float64)
    if node_load.shape != (n_nodes, 6):
        raise ValueError(f"field load shape {node_load.shape} != {(n_nodes, 6)}")
    node_support = np.zeros((n_nodes, 6), dtype=np.float64)
    for node, flags in zip(np.asarray(geom["support_node_index"], dtype=np.int64),
                           np.asarray(geom["support_flags"], dtype=np.float64)):
        node_support[int(node)] = flags
    node_x = np.concatenate([node_xyz, node_load, node_support], axis=1)
    node_y = np.asarray(z["node_disp"][i], dtype=np.float64)
    if node_y.shape != (n_nodes, 6):
        raise ValueError(f"node_disp shape {node_y.shape} != {(n_nodes, 6)}")

    beam_ids = np.asarray(geom["beam_ids"], dtype=np.int64)
    beam_node_ids = np.asarray(geom["beam_node_ids"], dtype=np.int64)
    beam_node_index = np.asarray(geom["beam_node_index"], dtype=np.int64)
    n_beams = len(beam_ids)
    beam_raw = np.array(design["beam_raw"], dtype=np.float64, copy=True)
    beam_end_prop = np.full((n_beams, 2, 29), np.nan, dtype=np.float64)
    end_used = np.asarray(design["beam_end_prop_used"], dtype=np.float64)  # (n,2,4)
    beam_end_mask = np.asarray(design["beam_end_mask"], dtype=bool).copy()
    beam_end_prop[..., 0] = end_used[..., 0]
    beam_end_prop[..., 3] = end_used[..., 1]
    beam_end_prop[..., 4] = end_used[..., 2]
    beam_end_prop[..., 5] = end_used[..., 3]
    beam_y = np.asarray(z["beam_force"][i], dtype=np.float64)
    if beam_y.shape != (n_beams, 12):
        raise ValueError(f"beam_force shape {beam_y.shape} != {(n_beams, 12)}")

    plate_ids = np.asarray(geom["plate_ids"], dtype=np.int64)
    plate_node_ids = np.asarray(geom["plate_node_ids"], dtype=np.int64)
    plate_node_index = np.asarray(geom["plate_node_index"], dtype=np.int64)
    plate_raw = np.array(design["plate_raw"], dtype=np.float64, copy=True)

    link_ids = np.asarray(geom["link_ids"], dtype=np.int64)
    link_edge_index = np.asarray(geom["link_edge_index"], dtype=np.int64)
    link_attr = np.asarray(geom["link_attr"], dtype=np.float64)
    release_ids = np.asarray(geom["release_ids"], dtype=np.int64)
    release_beam_index = np.asarray(geom["release_beam_index"], dtype=np.int64)
    release_node_index = np.asarray(geom["release_node_index"], dtype=np.int64)
    release_info = np.asarray(geom["release_info"], dtype=np.float64)
    master_ids = np.asarray(geom["master_ids"], dtype=np.int64)
    master_node_index = np.asarray(geom["master_node_index"], dtype=np.int64)
    master_info = np.asarray(geom["master_info"], dtype=np.float64)
    # Edge-expanded release arrays are reconstructed exactly as in assemble_raw.
    release_edge_beam = np.repeat(release_beam_index, 2)
    release_edge_node = release_node_index.reshape(-1)
    release_edge_role = np.tile(np.array([0, 1], dtype=np.int64), len(release_ids))
    release_edge_info = release_info.reshape(-1, 6)

    return {
        "sample_id": row["sample_id"], "topology_token": row["topology_token"],
        "design_id": row["design_id"], "field_id": row["field_id"],
        "family": meta.get("family"), "coordinate_system": meta.get("coordinate_system"),
        "node_ids": node_ids, "node_xyz": node_xyz, "node_load": node_load,
        "node_support": node_support, "node_x": node_x, "node_y": node_y,
        "beam_ids": beam_ids, "beam_node_ids": beam_node_ids,
        "beam_node_index": beam_node_index, "beam_raw": beam_raw, "beam_y": beam_y,
        "beam_end_prop": beam_end_prop, "beam_end_mask": beam_end_mask,
        "beam_sec_id": np.asarray(design["beam_sec_id"], dtype=np.int64),
        "beam_mat_id": np.asarray(design["beam_mat_id"], dtype=np.int64),
        "plate_ids": plate_ids, "plate_node_ids": plate_node_ids,
        "plate_node_index": plate_node_index, "plate_raw": plate_raw,
        "support_ids": np.asarray(geom["support_ids"], dtype=np.int64),
        "support_node_index": np.asarray(geom["support_node_index"], dtype=np.int64),
        "support_flags": np.asarray(geom["support_flags"], dtype=np.float64),
        "link_ids": link_ids, "link_edge_index": link_edge_index, "link_attr": link_attr,
        "release_ids": release_ids, "release_beam_index": release_beam_index,
        "release_node_index": release_node_index, "release_info": release_info,
        "release_edge_beam": release_edge_beam, "release_edge_node": release_edge_node,
        "release_edge_role": release_edge_role, "release_edge_info": release_edge_info,
        "master_ids": master_ids, "master_node_index": master_node_index,
        "master_info": master_info, "n_equations": int(design["n_equations"]),
    }

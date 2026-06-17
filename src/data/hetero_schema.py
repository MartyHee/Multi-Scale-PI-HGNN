"""
hetero_schema.py — Schema definition for the Multi-Scale PI-HGNN heterogeneous graph.

Defines node types, edge types, feature fields, and target fields for the
HeteroData representation of GraphTrainingData2.

This module serves as the single source of truth for the graph schema,
used by the builder, dataset loader, and transforms.

v2 (2026-06-16):
  - Removed ``link_element`` node type (rigid_elastic_links.csv modeled as
    ``mesh_node ↔ structural_link ↔ mesh_node`` interaction edge).
  - Renamed ``rigid_link`` → ``structural_link``.
  - Reverse membership edges (``rev_belongs_to_beam``, ``rev_belongs_to_plate``)
    now carry ``edge_attr``.
  - ``structural_link`` edge_attr is 10-d (includes ``is_rigid``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ============================================================
# Node type definitions (3 types only)
# ============================================================

NODE_TYPE_CONFIG: Dict[str, dict] = {
    "mesh_node": {
        "description": "Finite element mesh node (joint)",
        "source_tables": ["nodes.csv", "nodal_loads.csv", "general_supports.csv"],
        "count": 1056,
        "feature_fields": [
            "X", "Y", "Z",
            "Fx", "Fy", "Fz", "Mx", "My", "Mz",
            "Dx_fix", "Dy_fix", "Dz_fix", "Rx_fix", "Ry_fix", "Rz_fix",
        ],
        "feature_dim": 15,
        "target_fields": ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"],
        "target_dim": 6,
        "target_key": "y_disp",
        "has_labels": True,
    },
    "beam_element": {
        "description": "Beam element connecting two mesh nodes",
        "source_tables": ["beam_elements.csv", "beam_sections.csv", "materials.csv"],
        "count": 1646,
        "feature_fields": [
            "Area", "Ix", "Iy", "Iz",
            "ElasticModulus", "PoissonRatio", "UnitWeight",
            "Length", "CosX", "CosY", "CosZ",
        ],
        "feature_dim": 11,
        "target_fields": [
            "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
            "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
        ],
        "target_dim": 12,
        "target_key": "y_force",
        "has_labels": True,
    },
    "plate_element": {
        "description": "Quadrilateral plate element (thick plate, 4 nodes)",
        "source_tables": ["plate_elements.csv", "thicknesses.csv", "materials.csv"],
        "count": 832,
        "feature_fields": [
            "ThicknessValue",
            "ElasticModulus", "PoissonRatio", "UnitWeight",
            "BetaAngle", "PlateType",
        ],
        "feature_dim": 6,
        "has_labels": False,
        "note": "No plate force/stress labels available in current data.",
    },
    # link_element intentionally removed — rigid_elastic_links.csv is
    # modeled as mesh_node ↔ structural_link ↔ mesh_node edge.
}

#: Node type names in fixed order (3 types, no link_element)
NODE_TYPES: List[str] = ["mesh_node", "beam_element", "plate_element"]

#: Types that have trainable targets (supervision)
TARGET_NODE_TYPES: Dict[str, str] = {
    "mesh_node": "y_disp",
    "beam_element": "y_force",
}


# ============================================================
# Edge type definitions (5 types)
# ============================================================

#: Each entry: (source, relation, target)
EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),
    ("mesh_node", "belongs_to_plate", "plate_element"),
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),
    ("mesh_node", "structural_link", "mesh_node"),
]

#: Edge types that carry edge_attr (now includes reverse membership edges)
EDGE_TYPES_WITH_ATTR: List[Tuple[str, str, str]] = [
    ("mesh_node", "belongs_to_beam", "beam_element"),        # endpoint_type (I=0, J=1)
    ("beam_element", "rev_belongs_to_beam", "mesh_node"),    # endpoint_type (same as fwd)
    ("mesh_node", "belongs_to_plate", "plate_element"),       # corner_type (I/J/K/L)
    ("plate_element", "rev_belongs_to_plate", "mesh_node"),   # corner_type (same as fwd)
    ("mesh_node", "structural_link", "mesh_node"),            # stiffness + metadata + is_rigid
]

#: Edge attr dimension per edge type
EDGE_ATTR_DIMS: Dict[Tuple[str, str, str], int] = {
    ("mesh_node", "belongs_to_beam", "beam_element"): 1,
    ("beam_element", "rev_belongs_to_beam", "mesh_node"): 1,
    ("mesh_node", "belongs_to_plate", "plate_element"): 1,
    ("plate_element", "rev_belongs_to_plate", "mesh_node"): 1,
    ("mesh_node", "structural_link", "mesh_node"): 10,
}

#: For display/lookup: edge type canonical names
EDGE_TYPE_NAMES: Dict[Tuple[str, str, str], str] = {
    ("mesh_node", "belongs_to_beam", "beam_element"): "mesh_to_beam_membership",
    ("beam_element", "rev_belongs_to_beam", "mesh_node"): "beam_to_mesh_membership",
    ("mesh_node", "belongs_to_plate", "plate_element"): "mesh_to_plate_membership",
    ("plate_element", "rev_belongs_to_plate", "mesh_node"): "plate_to_mesh_membership",
    ("mesh_node", "structural_link", "mesh_node"): "mesh_to_mesh_structural_link",
}

#: Expected edge counts per type
EDGE_COUNTS: Dict[Tuple[str, str, str], int] = {
    ("mesh_node", "belongs_to_beam", "beam_element"): 3292,     # 1646 × 2
    ("beam_element", "rev_belongs_to_beam", "mesh_node"): 3292,
    ("mesh_node", "belongs_to_plate", "plate_element"): 3328,   # 832 × 4
    ("plate_element", "rev_belongs_to_plate", "mesh_node"): 3328,
    ("mesh_node", "structural_link", "mesh_node"): 132,
}


# ============================================================
# Helper functions
# ============================================================

def get_feature_dim(node_type: str) -> int:
    """Return feature dimension for a node type."""
    return NODE_TYPE_CONFIG[node_type]["feature_dim"]


def get_target_dim(node_type: str) -> Optional[int]:
    """Return target dimension if the node type has labels, else None."""
    cfg = NODE_TYPE_CONFIG[node_type]
    return cfg.get("target_dim") if cfg.get("has_labels") else None


def get_target_key(node_type: str) -> Optional[str]:
    """Return target attribute key (e.g. 'y_disp') if the node type has labels."""
    cfg = NODE_TYPE_CONFIG[node_type]
    return cfg.get("target_key") if cfg.get("has_labels") else None


def get_node_count(node_type: str) -> int:
    """Return the expected number of nodes of this type."""
    return NODE_TYPE_CONFIG[node_type]["count"]


def build_schema_metadata() -> dict:
    """Build a complete schema metadata dict for serialisation."""
    return {
        "dataset_name": "hetero_graph_dataset_v2",
        "graph_type": "heterogeneous",
        "node_types": [
            {
                "name": nt,
                "count": NODE_TYPE_CONFIG[nt]["count"],
                "feature_dim": NODE_TYPE_CONFIG[nt]["feature_dim"],
                "feature_fields": NODE_TYPE_CONFIG[nt]["feature_fields"],
                "has_labels": NODE_TYPE_CONFIG[nt].get("has_labels", False),
                "target_dim": NODE_TYPE_CONFIG[nt].get("target_dim"),
                "target_fields": NODE_TYPE_CONFIG[nt].get("target_fields"),
                "target_key": NODE_TYPE_CONFIG[nt].get("target_key"),
                "note": NODE_TYPE_CONFIG[nt].get("note", ""),
            }
            for nt in NODE_TYPES
        ],
        "edge_types": [
            {
                "source": src,
                "relation": rel,
                "target": tgt,
                "count": EDGE_COUNTS.get((src, rel, tgt), 0),
                "has_edge_attr": (src, rel, tgt) in EDGE_TYPES_WITH_ATTR,
                "edge_attr_dim": EDGE_ATTR_DIMS.get((src, rel, tgt)),
            }
            for src, rel, tgt in EDGE_TYPES
        ],
        "tasks": [
            {
                "node_type": nt,
                "target_key": get_target_key(nt),
                "target_dim": get_target_dim(nt),
                "description": f"Predict {get_target_key(nt)} for {nt}",
            }
            for nt in TARGET_NODE_TYPES
        ],
        "topology_count": 1,
        "note": (
            "All 70 samples share identical topology. "
            "3 node types: mesh_node, beam_element, plate_element. "
            "rigid_elastic_links.csv modeled as mesh_node ↔ structural_link ↔ mesh_node edge. "
            "link_element is NOT a node type — removed in v2 schema fix per proposal."
        ),
    }

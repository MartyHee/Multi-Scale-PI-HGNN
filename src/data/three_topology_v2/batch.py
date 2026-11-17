"""Mixed-batch handling and verification for constraint references.

``ThreeTopoData.__inc__`` makes PyG offset the top-level constraint reference
tensors during batching:
  - mesh-node referenced: ``support_node_index``, ``beam_release_node_index``,
    ``beam_release_edge_node_index``, ``master_slave_node_index``
  - beam-element referenced: ``beam_release_element_index``,
    ``beam_release_edge_element_index``
Raw ``*_id`` fields are kept un-offset.

``verify_batched_constraints`` independently reconstructs the expected global
indices from the per-graph local tensors and confirms the batched tensors, and
checks the five main relations for cross-graph leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch_geometric.loader import DataLoader

from .graph_io import EDGE_TYPES, ThreeTopoData

NODE_REF_KEYS = ["support_node_index", "beam_release_node_index",
                 "beam_release_edge_node_index", "master_slave_node_index"]
BEAM_REF_KEYS = ["beam_release_element_index", "beam_release_edge_element_index"]
RAW_ID_KEYS = ["support_id", "beam_release_id", "master_slave_id"]


def make_loader(dataset, batch_size: int, shuffle: bool = False, seed: int = 0) -> DataLoader:
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=gen)


def _ptr_bounds(ptr: torch.Tensor) -> List[tuple]:
    return [(int(ptr[i]), int(ptr[i + 1])) for i in range(len(ptr) - 1)]


def _split_by_counts(batched: torch.Tensor, counts: Sequence[int]) -> List[torch.Tensor]:
    out = []
    off = 0
    for c in counts:
        out.append(batched[off:off + c])
        off += c
    assert off == batched.shape[0] or batched.shape[0] == 0, (off, batched.shape)
    return out


def verify_batched_constraints(raw_graphs: Sequence[ThreeTopoData], batch) -> Dict[str, Any]:
    """Verify mixed-batch constraint references against per-graph locals."""
    report: Dict[str, Any] = {"graphs": len(raw_graphs), "checks": [], "problems": []}

    def ok(name: str, cond: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        if not cond:
            report["problems"].append(f"{name}: {detail}")

    mesh_ptr = batch["mesh_node"].ptr
    beam_ptr = batch["beam_element"].ptr
    mesh_bounds = _ptr_bounds(mesh_ptr)
    beam_bounds = _ptr_bounds(beam_ptr)

    # per-graph counts for each reference key
    for key in NODE_REF_KEYS + BEAM_REF_KEYS:
        counts = [getattr(g, key).shape[0] for g in raw_graphs]
        batched = getattr(batch, key)
        # reconstruct expected global tensor
        expected = []
        for gi, g in enumerate(raw_graphs):
            local = getattr(g, key)
            cfg = mesh_bounds[gi][0] if key in NODE_REF_KEYS else beam_bounds[gi][0]
            expected.append(local + cfg)
        expected_cat = torch.cat(expected, dim=0) if expected else batched.new_zeros(batched.shape)
        same = (batched.shape == expected_cat.shape) and bool(torch.equal(batched, expected_cat))
        ok(f"{key}.offset_matches", same,
           f"batched != local+offset for {len(raw_graphs)} graphs" if not same else "ok")

        # every global node/beam index lands inside the right graph slice
        if key in NODE_REF_KEYS and batched.numel():
            parts = _split_by_counts(batched, counts)
            for gi, part in enumerate(parts):
                if part.numel() == 0:
                    continue
                lo, hi = mesh_bounds[gi]
                ok(f"{key}.graph{gi}_in_range", bool(((part >= lo) & (part < hi)).all()),
                   f"values not within [{lo},{hi})")
        if key in BEAM_REF_KEYS and batched.numel():
            parts = _split_by_counts(batched, counts)
            for gi, part in enumerate(parts):
                if part.numel() == 0:
                    continue
                lo, hi = beam_bounds[gi]
                ok(f"{key}.graph{gi}_in_range", bool(((part >= lo) & (part < hi)).all()),
                   f"values not within [{lo},{hi})")

    # value equality: a batched support global index must point to the same node row
    counts_support = [g.support_node_index.shape[0] for g in raw_graphs]
    parts = _split_by_counts(batch.support_node_index, counts_support)
    for gi, g in enumerate(raw_graphs):
        part = parts[gi]
        if part.numel() == 0:
            continue
        same_feat = bool(torch.equal(batch["mesh_node"].x[part],
                                     g["mesh_node"].x[g.support_node_index]))
        ok_ = same_feat
        ok(f"support.graph{gi}_feature_match", ok_, "global support node features differ from local")

    # raw id fields must NOT be offset
    for key in RAW_ID_KEYS:
        counts = [getattr(g, key).shape[0] for g in raw_graphs]
        cat = torch.cat([getattr(g, key) for g in raw_graphs], dim=0) if sum(counts) else getattr(batch, key)
        if cat.numel() or getattr(batch, key).numel():
            same = bool(torch.equal(getattr(batch, key), cat))
            ok(f"{key}.raw_not_offset", same, "raw id field was unexpectedly offset")

    # five relations: no cross-graph leakage
    for et in EDGE_TYPES:
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src_edge, dst_edge = ei[0], ei[1]
        src_type, _, dst_type = et
        src_ptr = _store_ptr(batch, src_type)
        dst_ptr = _store_ptr(batch, dst_type)
        src_g = _graph_ids(src_ptr, src_edge)
        dst_g = _graph_ids(dst_ptr, dst_edge)
        same = bool(torch.equal(src_g, dst_g))
        ok(f"relation.{et[1]}.no_cross_graph", same, "source/target graph ids differ")

    report["all_ok"] = not report["problems"]
    return report


def _graph_ids(ptr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Map global node/edge indices to their graph id via the store ``ptr``."""
    return torch.searchsorted(ptr, indices, right=True) - 1


def _store_ptr(batch, node_type: str) -> torch.Tensor:
    if node_type == "mesh_node":
        return batch["mesh_node"].ptr
    if node_type == "beam_element":
        return batch["beam_element"].ptr
    if node_type == "plate_element":
        return batch["plate_element"].ptr
    raise ValueError(node_type)


def iter_graph_constraints(raw_graphs: Sequence[ThreeTopoData]) -> List[Dict[str, Any]]:
    """Explicit per-graph interface: yield local constraint tensors per graph."""
    out = []
    for g in raw_graphs:
        out.append({
            "support_node_index": g.support_node_index,
            "support_boundary_info": g.support_boundary_info,
            "support_id": g.support_id,
            "beam_release_element_index": g.beam_release_element_index,
            "beam_release_node_index": g.beam_release_node_index,
            "beam_release_info": g.beam_release_info,
            "beam_release_edge_element_index": g.beam_release_edge_element_index,
            "beam_release_edge_node_index": g.beam_release_edge_node_index,
            "beam_release_edge_role": g.beam_release_edge_role,
            "master_slave_node_index": g.master_slave_node_index,
            "master_slave_boundary_info": g.master_slave_boundary_info,
        })
    return out

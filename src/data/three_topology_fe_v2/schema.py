"""Machine-readable feature schema helpers.

Raw columns are 15/17/6.  Models consume the encoded view; dimensions therefore
come from the schema rather than from hard-coded 15/15/6 assumptions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from . import paths

_SCHEMA: Dict[str, Any] | None = None


def load_schema(path: Path | None = None) -> Dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None or path is not None:
        _SCHEMA = json.loads((path or paths.FEATURE_SCHEMA).read_text(encoding="utf-8"))
    return _SCHEMA


def encoded_dim(node_type: str) -> int:
    return int(load_schema()["encoded_dims"][node_type])


def raw_columns(node_type: str):
    return list(load_schema()["raw_view"][node_type]["columns"])


def encoded_columns(node_type: str):
    return list(load_schema()["raw_view"][node_type]["encoded_columns"])


def model_cfg_with_dims(model_name: str, model_cfg: Dict[str, Any],
                        topology: str | None = None) -> Dict[str, Any]:
    """Inject schema-derived feature dimensions into a model config.

    Architecture parameters stay in the config; input widths come from the
    schema.  MLP has no plate encoder, but the extra keys are harmless for the
    other model factories.
    """
    out = dict(model_cfg)
    out.setdefault("mesh_feat_dim", encoded_dim("mesh_node"))
    out.setdefault("beam_feat_dim", encoded_dim("beam_element"))
    out.setdefault("plate_feat_dim", encoded_dim("plate_element"))
    if model_name == "mlp":
        out.pop("plate_feat_dim", None)
    if model_name == "fullfield_query_mlp":
        dims = model_cfg.get("field_dims") or {}
        if topology is None or topology not in dims:
            raise ValueError(
                f"fullfield_query_mlp needs field_dims[{topology!r}]; "
                f"available={sorted(dims)}")
        out["field_dim"] = int(dims[topology])
        out.pop("field_dims", None)
    return out


# Models whose shared backbone is initialised from the run seed.  HGT and
# HGT-global additionally share the same tensors for a given seed; the global
# module uses a separately derived stream (see hgt_global.py).
INIT_FROM_RUN_SEED = {"hgt", "hgt_global", "hgt_capmatch", "hgt_global_off"}


def task_model_cfg(model_name: str, cfg: Dict[str, Any], topology: str,
                   seed: int) -> Dict[str, Any]:
    """Effective per-task model config (schema dims + init policy)."""
    model_cfg = (cfg.get("models") or {})[model_name]
    out = model_cfg_with_dims(model_name, model_cfg, topology=topology)
    if model_name in INIT_FROM_RUN_SEED:
        out["init_seed"] = int(seed)
    return out


def schema_hash() -> str:
    import hashlib
    return hashlib.sha256(paths.FEATURE_SCHEMA.read_bytes()).hexdigest()

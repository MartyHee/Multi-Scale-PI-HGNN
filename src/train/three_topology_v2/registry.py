"""Model factory for the three-topology v2 baselines.

Gradient models reuse the verified implementations in ``src.models.baselines``.
All are constructed with ``beam_feat_dim=15`` (the v2 encoding).  Classical
ML models are handled separately in ``ml_baselines``.
"""

from __future__ import annotations

from typing import Any, Dict

GRADIENT_MODELS = {
    "mlp", "hgt_global", "hgt", "rgcn", "gcn", "gat", "graphsage", "gin",
    "edge_aware_mpnn", "spectral_lift"
}
NON_GRADIENT_MODELS = {"ridge", "hist_gb"}


def build_gradient_model(name: str, cfg: Dict[str, Any]):
    name = name.lower()
    if name == "mlp":
        from src.models.baselines.mlp_baseline import MLPBaseline
        return MLPBaseline(**cfg)
    if name == "hgt_global":
        from src.models.baselines.hgt_global import HGTGlobalBaseline
        return HGTGlobalBaseline(**cfg)
    if name == "hgt":
        from src.models.baselines.hgt_baseline import HGTBaseline
        return HGTBaseline(**cfg)
    if name == "rgcn":
        from src.models.baselines.hetero_rgcn import HeteroRGCNBaseline
        return HeteroRGCNBaseline(**cfg)
    if name == "gcn":
        from src.models.baselines.homogeneous_gcn import HomogeneousGCN
        return HomogeneousGCN(**cfg)
    if name == "gat":
        from src.models.baselines.homogeneous_gat import HomogeneousGAT
        return HomogeneousGAT(**cfg)
    if name == "graphsage":
        from src.models.baselines.homogeneous_extra import GraphSAGEBaseline
        return GraphSAGEBaseline(**cfg)
    if name == "gin":
        from src.models.baselines.homogeneous_extra import GINBaseline
        return GINBaseline(**cfg)
    if name == "edge_aware_mpnn":
        from src.models.baselines.edge_aware_mpnn import EdgeAwareMPNNBaseline
        return EdgeAwareMPNNBaseline(**cfg)
    if name == "spectral_lift":
        from src.models.baselines.spectral_lift import SpectralLiftBaseline
        return SpectralLiftBaseline(**cfg)
    raise ValueError(f"unknown gradient model {name!r}; options {sorted(GRADIENT_MODELS)}")


def list_models() -> Dict[str, list]:
    return {"gradient": sorted(GRADIENT_MODELS), "non_gradient": sorted(NON_GRADIENT_MODELS)}

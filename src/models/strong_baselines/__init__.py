"""Strong-baseline adaptations prepared on 2026-11-26.

Only task adaptations live here; none of these modules claims to reproduce the
original papers' benchmark numbers.  See ``docs/strong_baseline_preparation_20261126``
for the source/version/licence notes and the disclosed differences.
"""
from src.models.strong_baselines.fullfield import FullFieldMLPBaseline
from src.models.strong_baselines.fullfield_query import FullFieldQueryMLPBaseline
from src.models.strong_baselines.meshgraphnet import MeshGraphNetBaseline
from src.models.strong_baselines.transolver import TransolverBaseline
from src.models.strong_baselines.linearno import LinearNOBaseline

__all__ = [
    "FullFieldMLPBaseline",
    "FullFieldQueryMLPBaseline",
    "MeshGraphNetBaseline",
    "TransolverBaseline",
    "LinearNOBaseline",
]

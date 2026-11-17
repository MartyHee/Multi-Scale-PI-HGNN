"""Unified training/validation entry for the three-topology v2 baselines.

Modules
-------
``metrics``    Physical-unit per-component / per-graph / per-topology metrics.
``registry``   Model factory (gradient GNN/MLP + classical ML baselines).
``ml_baselines`` Fixed-dim node/beam featurisation + Ridge / tree boosting.
``engine``     Train/val loop with checkpoint resume, prediction export, timing.
"""

from __future__ import annotations

__all__ = ["metrics", "registry", "ml_baselines", "engine"]
__version__ = "1.0.0"

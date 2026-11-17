"""Three-topology training-input v2 package.

Independent namespace for the revised training inputs of the accepted
``three_topology_graph_v1`` graph dataset.  It never writes the source data
package; all derived artifacts go under the model project's own
``processed/three_topology_training_v2`` and ``outputs/three_topology_v2``
directories.

Submodules
----------
``paths``      Fixed path resolution (source package root, index, output dirs).
``index``      Streaming read + identity/coverage integrity checks of samples.jsonl.
``graph_io``   Graph JSON -> ``ThreeTopoData`` (HeteroData subclass), v1 contract.
``splits``     Revised splits (cross-design, cross-load-field, leave-one-topology).
``stats``      Per-fold normalization fit/apply + label inverse transform.
``dataset``    Lazy normalized dataset for a single fold.
``batch``      Mixed-batch constraint reference offsets + verification helpers.

Nothing here loads a checkpoint, rms-of-optimizer or legacy statistics.
"""

from __future__ import annotations

__all__ = [
    "paths",
    "index",
    "graph_io",
    "splits",
    "stats",
    "dataset",
    "batch",
]

__version__ = "2.0.0"

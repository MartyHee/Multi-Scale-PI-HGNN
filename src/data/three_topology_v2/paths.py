"""Path resolution for the three-topology training-input v2 package.

The accepted graph dataset lives in the S1 data package::

    D:/.../S1/raw_data/three_topology_graph_v1

All index paths (``graph_path``, ``extraction_path``, ``run_manifest_path``)
are resolved **relative to that package root**, never relative to the index
file's own directory.  Derived training inputs and smoke outputs are written
under the model project only::

    <model_root>/processed/three_topology_training_v2/
    <model_root>/outputs/three_topology_v2/

The package root can be overridden with ``THREE_TOPO_PACKAGE_ROOT`` but the
default is the frozen S1 copy.
"""

from __future__ import annotations

import os
from pathlib import Path

# Model project root = .../Multi-Scale-PI-HGNN (this file is src/data/three_topology_v2/)
MODEL_ROOT = Path(__file__).resolve().parents[3]

# S1 project root = parent of the model project
S1_ROOT = MODEL_ROOT.parent

DEFAULT_PACKAGE_ROOT = S1_ROOT / "raw_data" / "three_topology_graph_v1"


def package_root() -> Path:
    """Return the read-only graph data package root."""
    env = os.environ.get("THREE_TOPO_PACKAGE_ROOT")
    root = Path(env).resolve() if env else DEFAULT_PACKAGE_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"three-topology package root not found: {root}")
    return root


def index_path(root: Path | None = None) -> Path:
    root = root or package_root()
    return root / "data" / "processed" / "three_topology_graph_v1" / "samples.jsonl"


def manifest_path(root: Path | None = None) -> Path:
    root = root or package_root()
    return root / "data" / "processed" / "three_topology_graph_v1" / "dataset_manifest.json"


def source_manifest_path(name: str, root: Path | None = None) -> Path:
    """``name`` in {topology1_primary_samples_v2.jsonl, topology2_samples_v2.jsonl,
    topology3_samples.jsonl}."""
    root = root or package_root()
    return root / "manifests" / name


def v1_split_dir(root: Path | None = None) -> Path:
    root = root or package_root()
    return root / "manifests" / "three_topology_experiment_v1"


# ---------------------------------------------------------------------------
# Derived (writable) destinations — model project only
# ---------------------------------------------------------------------------
TRAIN_ROOT = MODEL_ROOT / "processed" / "three_topology_training_v2"
SPLIT_DIR = TRAIN_ROOT / "splits"
STATS_DIR = TRAIN_ROOT / "stats"
MANIFEST_DIR = TRAIN_ROOT / "manifests"
OUTPUT_ROOT = MODEL_ROOT / "outputs" / "three_topology_v2"
DOC_DIR = MODEL_ROOT / "docs" / "three_topology_v2"
CONFIG_DIR = MODEL_ROOT / "configs" / "three_topology_v2"

DERIVED_DIRS = [TRAIN_ROOT, SPLIT_DIR, STATS_DIR, MANIFEST_DIR, OUTPUT_ROOT, DOC_DIR]


def ensure_derived_dirs() -> None:
    for d in DERIVED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def resolve_under_root(rel_or_abs: str, root: Path | None = None) -> Path:
    """Resolve an index path against the package root (absolute paths kept)."""
    root = root or package_root()
    p = Path(rel_or_abs)
    return p if p.is_absolute() else root / p

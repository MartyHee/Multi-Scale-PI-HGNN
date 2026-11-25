"""Path constants for the 9,216-case FE v2 pipeline.

The package/processed/output roots can be overridden by the server launcher
(``THREE_TOPO_PACKAGE_ROOT`` / ``THREE_TOPO_PROCESSED_ROOT`` /
``THREE_TOPO_OUTPUT_ROOT``) so the configured paths are the ones the reader
actually uses; defaults keep local runs unchanged.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[3]
S1_ROOT = MODEL_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    # Accept MSYS/Git-Bash paths (/d/...) on Windows as well as native paths.
    if os.name == "nt":
        m = re.match(r"^/([A-Za-z])/(.*)$", value)
        if m:
            value = f"{m.group(1).upper()}:/{m.group(2)}"
    return Path(value).expanduser().resolve()


PACKAGE_ROOT = _env_path("THREE_TOPO_PACKAGE_ROOT",
                         S1_ROOT / "raw_data" / "three_topology_fe_v2_9216")
PROCESSED_ROOT = _env_path("THREE_TOPO_PROCESSED_ROOT",
                           MODEL_ROOT / "processed" / "three_topology_fe_v2_9216")
CONFIG_ROOT = _env_path("THREE_TOPO_CONFIG_ROOT",
                        MODEL_ROOT / "configs" / "three_topology_fe_v2")
FEATURE_SCHEMA = CONFIG_ROOT / "feature_schema.json"
OUTPUT_ROOT = _env_path("THREE_TOPO_OUTPUT_ROOT",
                        MODEL_ROOT / "outputs" / "three_topology_fe_v2_9216")

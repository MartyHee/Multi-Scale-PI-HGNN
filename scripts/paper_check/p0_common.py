"""
p0_common.py — Shared I/O + metric helpers for ICLR-2027 Stage-1 P0 diagnostics.

Reads the exported full-prediction NPZ files (mesh / beam) plus the test graph
index, rebuilds the GLOBAL graph index, maps every row to a sample_id (design),
and exposes the exact metric definitions used across the project.

Key fact (verified on 2026-08-09):
    The NPZ ``graph_id`` is BATCH-LOCAL (max == 7 for 3500 test graphs), NOT the
    global graph index. Rows are grouped by graph (each block of N_PER_GRAPH rows
    is one graph, in the same order as ``test_graph_index.csv``), so the global
    graph index is rebuilt as ``row_index // N_PER_GRAPH``.

Metric definitions (from ``src/utils/metrics.py`` and ``scripts/compute_region_labels.py``):
    - ``Disp R^2`` / ``Force R^2`` (paper main table) = macro-average R^2, i.e.
      the mean of per-component R^2.
    - ``RelMAE`` (paper combined) = (disp_overall_rel_mae + force_overall_rel_mae) / 2,
      where each overall rel_mae = MAE / mean(|target|) computed over ALL elements.
    - ``Dy R^2`` = per-component R^2 of component index 1 (Dy).

This module is READ-ONLY: it never modifies prediction files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MESH_PER_GRAPH = 1056
N_BEAM_PER_GRAPH = 1646
N_TEST_GRAPHS = 3500
N_TEST_DESIGNS = 7
GRAPHS_PER_DESIGN = 500

DISP_COMPONENTS = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_COMPONENTS = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]

# Model -> export dir (relative to project root). Order used for plots: HGT,
# MS-HGT gated, MS-HGT additive.
MODEL_EXPORTS = {
    "hgt": "outputs/predictions/stage2b/hgt/20260626081841",
    "ms_hgt_gated": "outputs/predictions/stage2b/ms_hgt/20260626082828",
    "ms_hgt_additive": "outputs/predictions/stage2b/ms_hgt_additive/20260626084307",
}
MODEL_ORDER = ["hgt", "ms_hgt_gated", "ms_hgt_additive"]
MODEL_DISPLAY = {
    "hgt": "HGT",
    "ms_hgt_gated": "MS-HGT gated",
    "ms_hgt_additive": "MS-HGT additive",
}

# Default output dir timestamp (all P0 scripts share one timestamped dir).
P0_TIMESTAMP = "20260809164706"
DEFAULT_OUTPUT_DIR = f"outputs/diagnostics/iclr2027_p0/{P0_TIMESTAMP}"


# ---------------------------------------------------------------------------
# Metric helpers (identical definitions to the project)
# ---------------------------------------------------------------------------


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot (handles zero-variance targets)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rel_mae_score(y_true: np.ndarray, y_pred: np.ndarray,
                  eps: float = 1e-8) -> float:
    """RelMAE = MAE / mean(|target|) — normalised by target magnitude (overall, flat)."""
    denom = float(np.mean(np.abs(y_true))) + eps
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def macro_avg_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of per-component R² (paper main-table Disp/Force R² definition)."""
    n_comp = y_true.shape[1]
    return float(np.mean([r2_score(y_true[:, c], y_pred[:, c]) for c in range(n_comp)]))


def per_component_r2(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    return [r2_score(y_true[:, c], y_pred[:, c]) for c in range(y_true.shape[1])]


def combined_rel_mae(disp_true, disp_pred, force_true, force_pred) -> float:
    """(disp_rel_mae + force_rel_mae) / 2 — paper RelMAE definition."""
    return (rel_mae_score(disp_true, disp_pred) + rel_mae_score(force_true, force_pred)) / 2.0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_mesh(npz_path: str | Path) -> dict:
    """Load mesh NPZ; returns arrays + rebuilt global graph index."""
    m = np.load(str(npz_path), allow_pickle=True)
    rows = len(m["y_true_disp"])
    n_graphs = rows // N_MESH_PER_GRAPH
    assert rows == n_graphs * N_MESH_PER_GRAPH, f"mesh rows {rows} not divisible by {N_MESH_PER_GRAPH}"
    return {
        "y_true_disp": m["y_true_disp"],      # (N, 6) float32
        "y_pred_disp": m["y_pred_disp"],      # (N, 6) float32
        "node_id": m["node_id"],              # (N,) within-graph local id
        "node_xyz": m["node_xyz"],            # (N, 3)
        "support_flags": m["support_flags"],  # (N, 6)
        "graph_local": m["graph_id"],         # batch-local id (max == 7)
        "graph_global": np.arange(rows) // N_MESH_PER_GRAPH,
        "n_rows": rows,
        "n_graphs": n_graphs,
    }


def load_beam(npz_path: str | Path) -> dict:
    """Load beam NPZ; returns arrays + rebuilt global graph index."""
    b = np.load(str(npz_path), allow_pickle=True)
    rows = len(b["y_true_force"])
    n_graphs = rows // N_BEAM_PER_GRAPH
    assert rows == n_graphs * N_BEAM_PER_GRAPH, f"beam rows {rows} not divisible by {N_BEAM_PER_GRAPH}"
    return {
        "y_true_force": b["y_true_force"],   # (N, 12) float32
        "y_pred_force": b["y_pred_force"],   # (N, 12) float32
        "element_id": b["element_id"],
        "graph_local": b["graph_id"],
        "graph_global": np.arange(rows) // N_BEAM_PER_GRAPH,
        "n_rows": rows,
        "n_graphs": n_graphs,
    }


def load_graph_index(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path))
    assert len(df) == N_TEST_GRAPHS, f"index rows {len(df)} != {N_TEST_GRAPHS}"
    return df


def map_sample_ids(graph_global: np.ndarray, index_df: pd.DataFrame) -> np.ndarray:
    """Map per-row global graph index -> sample_id (design)."""
    sample_map = index_df["sample_id"].to_numpy()
    return sample_map[graph_global]


def load_model_mesh(model_name: str, root: str | Path = ".") -> dict:
    """Load a model's full mesh prediction arrays + per-row sample_id."""
    export_dir = Path(root) / MODEL_EXPORTS[model_name]
    mesh = load_mesh(export_dir / "mesh_node_predictions.npz")
    idx = load_graph_index(export_dir / "test_graph_index.csv")
    mesh["sample_id"] = map_sample_ids(mesh["graph_global"], idx)
    mesh["index_df"] = idx
    return mesh


def load_model_beam(model_name: str, root: str | Path = ".") -> dict:
    """Load a model's full beam prediction arrays + per-row sample_id."""
    export_dir = Path(root) / MODEL_EXPORTS[model_name]
    beam = load_beam(export_dir / "beam_element_predictions.npz")
    idx = load_graph_index(export_dir / "test_graph_index.csv")
    beam["sample_id"] = map_sample_ids(beam["graph_global"], idx)
    beam["index_df"] = idx
    return beam


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

EXPECTED_MESH_ROWS = N_TEST_GRAPHS * N_MESH_PER_GRAPH       # 3,696,000
EXPECTED_BEAM_ROWS = N_TEST_GRAPHS * N_BEAM_PER_GRAPH       # 5,761,000


def run_integrity_check(root: str | Path = ".") -> dict:
    """Run the integrity check over the 3 core models and report results."""
    results = {}
    ok_all = True
    for model in MODEL_ORDER:
        export_dir = Path(root) / MODEL_EXPORTS[model]
        entry = {"model": model, "export_dir": str(export_dir)}
        try:
            idx = load_graph_index(export_dir / "test_graph_index.csv")
            mesh = load_mesh(export_dir / "mesh_node_predictions.npz")
            beam = load_beam(export_dir / "beam_element_predictions.npz")

            n_designs = int(idx["sample_id"].nunique())
            design_counts = idx.groupby("sample_id").size().to_dict()

            mesh_ok = mesh["n_rows"] == EXPECTED_MESH_ROWS
            beam_ok = beam["n_rows"] == EXPECTED_BEAM_ROWS
            shape_ok = (
                mesh["y_true_disp"].shape == (EXPECTED_MESH_ROWS, 6)
                and mesh["y_pred_disp"].shape == (EXPECTED_MESH_ROWS, 6)
                and beam["y_true_force"].shape == (EXPECTED_BEAM_ROWS, 12)
                and beam["y_pred_force"].shape == (EXPECTED_BEAM_ROWS, 12)
            )
            has_xyz = "node_xyz" in mesh and mesh["node_xyz"].shape[1] == 3
            finite_ok = (
                bool(np.isfinite(mesh["y_pred_disp"]).all())
                and bool(np.isfinite(beam["y_pred_force"]).all())
            )
            local_max = int(mesh["graph_local"].max())
            graph_id_mapping = (
                "batch-local" if local_max < N_TEST_GRAPHS - 1 else "global"
            )

            entry.update({
                "n_graphs": mesh["n_graphs"],
                "n_mesh_rows": mesh["n_rows"],
                "n_beam_rows": beam["n_rows"],
                "mesh_rows_ok": mesh_ok,
                "beam_rows_ok": beam_ok,
                "shape_ok": shape_ok,
                "node_xyz_present": has_xyz,
                "finite_ok": finite_ok,
                "graph_id_mode": graph_id_mapping,
                "graph_id_max": local_max,
                "n_designs": n_designs,
                "designs_have_500": bool(design_counts) and all(
                    c == GRAPHS_PER_DESIGN for c in design_counts.values()
                ),
                "design_graph_counts": design_counts,
            })
            entry_ok = all(entry[k] for k in (
                "mesh_rows_ok", "beam_rows_ok", "shape_ok",
                "node_xyz_present", "finite_ok", "designs_have_500",
            ))
            entry_ok = entry_ok and entry["n_designs"] == N_TEST_DESIGNS
            ok_all = ok_all and entry_ok
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            ok_all = False
        results[model] = entry

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expected": {
            "n_test_graphs": N_TEST_GRAPHS,
            "n_test_designs": N_TEST_DESIGNS,
            "n_mesh_rows": EXPECTED_MESH_ROWS,
            "n_beam_rows": EXPECTED_BEAM_ROWS,
            "graphs_per_design": GRAPHS_PER_DESIGN,
        },
        "all_ok": bool(ok_all),
        "runs": results,
    }


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P0 integrity check")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    result = run_integrity_check(args.root)
    out = ensure_dir(args.output_dir)
    save_json(result, out / "integrity_checks.json")

    # run_manifest.json: one row per model with artifact file inventory
    manifest = {"timestamp": result["timestamp"], "root": args.root,
                "models": []}
    for model in MODEL_ORDER:
        export_dir = Path(args.root) / MODEL_EXPORTS[model]
        files = {}
        for f in ("mesh_node_predictions.npz", "beam_element_predictions.npz",
                  "test_graph_index.csv", "prediction_summary.json",
                  "export_metrics_check.json"):
            p = export_dir / f
            files[f] = {"exists": p.exists(),
                        "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None}
        manifest["models"].append({
            "model": model, "display": MODEL_DISPLAY[model],
            "export_dir": str(export_dir), "files": files,
        })
    save_json(manifest, out / "run_manifest.json")

    print("Integrity check result:")
    for model, r in result["runs"].items():
        if "error" in r:
            print(f"  {model}: ERROR {r['error']}")
        else:
            print(f"  {model}: ok={all(r[k] for k in ('mesh_rows_ok','beam_rows_ok','shape_ok','finite_ok','designs_have_500'))}"
                  f" | n_designs={r['n_designs']} graph_id_mode={r['graph_id_mode']} "
                  f"mesh={r['n_mesh_rows']} beam={r['n_beam_rows']}")
    print(f"  all_ok = {result['all_ok']}")
    print(f"  outputs -> {out}")

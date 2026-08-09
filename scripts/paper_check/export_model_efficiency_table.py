"""
export_model_efficiency_table.py — A9: model / parameter / training-time table.

Aggregates, for the main-table models, test metrics + parameter count + best
epoch + training time from the AUTHORITATIVE server artifacts:
  - extracted runs:  read metrics_summary.json directly;
  - un-extracted runs: read metrics_summary.json straight out of the tarball
                       via tarfile (no extraction, read-only).

All numbers come from local files; if a field is unavailable it is left empty
with a source note — nothing is fabricated.

Outputs (into <output-dir>/):
  model_efficiency_table.csv
  model_efficiency_source_notes.md
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.paper_check.p0_common import DEFAULT_OUTPUT_DIR, ensure_dir  # noqa: E402

ROOT = Path(".").resolve()

# model -> metrics_summary location (dir path or tarball path)
MODEL_SOURCES = {
    "MLP": ("remote_artifacts/server_mlp_full_20260620060955.tar.gz", None),
    "GCN": ("remote_artifacts/server_gcn_full_20260620143146.tar.gz", None),
    "GAT": ("remote_artifacts/server_gat_full_20260620182256.tar.gz", None),
    "RGCN/HeteroConv": ("remote_artifacts/server_rgcn_full_20260621063600.tar.gz", None),
    "HGT": (None, "remote_artifacts/extracted/server_hgt_200epoch_20260624135223/metrics_summary.json"),
    "MS-HGT gated": (None, "remote_artifacts/extracted/server_ms_hgt_gated_20260625013512/metrics_summary.json"),
    "MS-HGT additive": (None, "remote_artifacts/extracted/server_ms_hgt_additive_20260625213023/metrics_summary.json"),
    "MS-PI-HGT full": (None, "remote_artifacts/extracted_stage5_validation/full/server_ms_pi_hgt_full_20260627185645/metrics_summary.json"),
}


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_from_tar(tarball: str | Path) -> dict:
    with tarfile.open(str(tarball), "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("metrics_summary.json"):
                raw = tf.extractfile(m).read().decode("utf-8")
                return json.loads(raw)
    raise FileNotFoundError(f"metrics_summary.json not found in {tarball}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A9 model efficiency table")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)

    rows = []
    notes = ["# A9 Model Efficiency Table — source notes", ""]
    for model, (tarball, direct) in MODEL_SOURCES.items():
        src = None
        try:
            if direct is not None:
                d = read_json(ROOT / direct)
                src = f"metrics_summary.json ({direct})"
            else:
                d = read_json_from_tar(ROOT / tarball)
                src = f"metrics_summary.json inside {tarball} (read via tarfile, no extraction)"
        except Exception as exc:  # noqa: BLE001
            notes.append(f"- **{model}**: FAILED to read metrics_summary -> {exc}")
            rows.append({"model": model, "params": None, "best_epoch": None,
                         "train_time_min": None, "disp_r2": None, "dy_r2": None,
                         "force_r2": None, "rel_mae": None,
                         "source": f"READ FAILED ({exc})"})
            continue

        tr = d.get("training", {})
        disp = d.get("test", {}).get("disp", {})
        force = d.get("test", {}).get("force", {})
        t_sec = tr.get("total_time_seconds")
        rows.append({
            "model": model,
            "params": d.get("num_params"),
            "best_epoch": tr.get("best_epoch"),
            "stopped_epoch": tr.get("stopped_epoch"),
            "train_time_sec": round(t_sec, 1) if t_sec else None,
            "train_time_min": round(t_sec / 60, 1) if t_sec else None,
            "disp_r2": disp.get("macro_avg_r2"),
            "dy_r2": (disp.get("per_component_r2") or [None] * 6)[1],
            "force_r2": force.get("macro_avg_r2"),
            "rel_mae": d.get("test", {}).get("combined_rel_mae"),
            "source": src,
        })
        notes.append(f"- **{model}**: {src}")

    df = pd.DataFrame(rows)
    # Readable columns for the CSV (drop the verbose source column from main table)
    main_cols = ["model", "params", "train_time_min", "best_epoch",
                 "disp_r2", "dy_r2", "force_r2", "rel_mae", "source"]
    df[main_cols].to_csv(out / "model_efficiency_table.csv", index=False)

    notes += ["", "Training time source: `training.total_time_seconds` in "
                   "`metrics_summary.json` (server wall-clock)."]
    (out / "model_efficiency_source_notes.md").write_text("\n".join(notes), encoding="utf-8")

    print("\n  Model efficiency table:")
    print(f"    {'Model':<16s} {'Params':>10s} {'Train(min)':>11s} {'BestEp':>7s} "
          f"{'DispR2':>9s} {'DyR2':>9s} {'ForceR2':>9s} {'RelMAE':>8s}")
    for _, r in df.iterrows():
        print(f"    {r['model']:<16s} {r['params'] or '':>10,} {r['train_time_min'] or '':>11.1f} "
              f"{r['best_epoch'] or '':>7} {r['disp_r2'] or '':>9.4f} "
              f"{r['dy_r2'] or '':>9.4f} {r['force_r2'] or '':>9.4f} {r['rel_mae'] or '':>8.4f}")
    print(f"\n  outputs -> {out}")


if __name__ == "__main__":
    main()

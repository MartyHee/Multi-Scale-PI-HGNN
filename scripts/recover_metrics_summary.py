#!/usr/bin/env python3
"""
recover_metrics_summary.py — Recover or re-generate metrics_summary.json from existing run directory.

This script reads train_log.csv, model_summary.json, config_resolved.yaml, and
server_output.log in a run directory and reconstructs metrics_summary.json in the
canonical schema used by train_baseline.py and generate_stage2a_summary().

Use cases:
  1. A training run completed but metrics_summary.json was not written (e.g., crash
     during save_sample_predictions after test evaluation).
  2. Manual consistency check / repair of a run's complete metrics.
  3. Re-packing an artifact that was packaged with the wrong run directory (parallel-
     execution LATEST_RUN bug).

Usage:
  python scripts/recover_metrics_summary.py --run-dir <path>
  python scripts/recover_metrics_summary.py --run-dir <path> --overwrite
  python scripts/recover_metrics_summary.py --run-dir <path> --dry-run

Schema output:
  Matches the canonical schema written by train_baseline.py lines 672-712:
    - config: model_name, model_class, split_mode, batch_size, seed
    - num_params
    - dataset_sizes: train, val, test
    - training: best_epoch, stopped_epoch, early_stopped, best_val_metric,
                best_val_disp_mae, best_val_force_mae, best_val_disp_r2,
                best_val_force_r2, total_time_seconds
    - test.loss_standardised, test.combined_rel_mae
    - test.disp / test.force:
        overall_mse, overall_mae, overall_rel_mae, overall_r2
        macro_avg_mae, macro_avg_rel_mae, macro_avg_r2
        per_component_mae[D], per_component_r2[D]
    - standardisation: source, train_only
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Regex patterns for server_output.log
# ---------------------------------------------------------------------------

PER_COMPONENT_RE = re.compile(
    r"^\s{4}(?P<name>\w+)\s+: MAE=(?P<mae>[-+]?[\d.]+(?:e[+-]?\d+)?)\s+"
    r"R2=(?P<r2>[-+]?[\d.]+(?:e[+-]?\d+)?)"
)

TEST_DISP_R2_RE = re.compile(r"Test Disp R2:\s+([\d.]+)")
TEST_FORCE_R2_RE = re.compile(r"Test Force R2:\s+([\d.]+)")
TEST_DISP_MAE_RE = re.compile(r"Test Disp MAE:\s+([\d.]+)")
TEST_FORCE_MAE_RE = re.compile(r"Test Force MAE:\s+([\d.]+)")
REL_MAE_RE = re.compile(r"Combined RelMAE:\s+([\d.]+)")
TOTAL_TIME_RE = re.compile(r"Total training time:\s+([\d.]+)")
BEST_EPOCH_RE = re.compile(r"Best epoch:\s+(\d+)")
EARLY_STOPPED_RE = re.compile(r"\[early stopping\]")
STOPPED_EPOCH_RE = re.compile(r"\[early stopping\].*Best epoch:\s+(\d+)")

DISP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_NAMES = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _read_lines(log_path: Path) -> List[str]:
    """Read a log file, splitting on both \\r and \\n."""
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r", "\n").split("\n")


def parse_server_log(log_path: Path) -> dict:
    """Extract test metrics and training info from server_output.log."""
    lines = _read_lines(log_path)
    if not lines:
        return {"_warning": "server_output.log not found"}

    result: Dict = {}

    # --- test-level metrics ---
    for line in lines:
        for attr, pattern in [
            ("test_disp_r2", TEST_DISP_R2_RE),
            ("test_force_r2", TEST_FORCE_R2_RE),
            ("test_disp_mae", TEST_DISP_MAE_RE),
            ("test_force_mae", TEST_FORCE_MAE_RE),
            ("test_combined_rel_mae", REL_MAE_RE),
        ]:
            m = pattern.search(line)
            if m:
                result[attr] = float(m.group(1))

        m = TOTAL_TIME_RE.search(line)
        if m:
            result["training_time_seconds"] = float(m.group(1))

        m = STOPPED_EPOCH_RE.search(line)
        if m:
            result["best_epoch"] = int(m.group(1))

        if EARLY_STOPPED_RE.search(line):
            result["early_stopped"] = True

    # Fallback: "Best epoch:" from summary section (no early stopping)
    if "best_epoch" not in result:
        for line in lines:
            m = BEST_EPOCH_RE.search(line)
            if m:
                result["best_epoch"] = int(m.group(1))

    # --- per-component metrics ---
    disp_mae: Dict[str, float] = {}
    disp_r2: Dict[str, float] = {}
    force_mae: Dict[str, float] = {}
    force_r2: Dict[str, float] = {}

    for line in lines:
        m = PER_COMPONENT_RE.search(line)
        if m:
            name, mae, r2 = m.group("name"), float(m.group("mae")), float(m.group("r2"))
            if name in DISP_NAMES:
                disp_mae[name] = mae
                disp_r2[name] = r2
            elif name in FORCE_NAMES:
                force_mae[name] = mae
                force_r2[name] = r2

    if disp_mae:
        result["per_component_disp_mae"] = [disp_mae.get(n) for n in DISP_NAMES]
        result["per_component_disp_r2"] = [disp_r2.get(n) for n in DISP_NAMES]
    if force_mae:
        result["per_component_force_mae"] = [force_mae.get(n) for n in FORCE_NAMES]
        result["per_component_force_r2"] = [force_r2.get(n) for n in FORCE_NAMES]

    return result


def parse_train_log(log_path: Path) -> dict:
    """Extract best-epoch info from train_log.csv."""
    if not log_path.exists():
        return {"_warning": "train_log.csv not found"}

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    best_epoch = None
    best_val_loss = float("inf")
    best_row = {}
    for row in rows:
        epoch = int(row.get("epoch", 0))
        val_loss = float(row.get("val_loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_row = row

    result = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "total_epochs": len(rows),
        "stopped_epoch": int(rows[-1]["epoch"]) if rows else None,
    }

    # Lambda values (from any row — they should be constant)
    row0 = rows[0] if rows else {}
    for key in ["lambda_bc", "lambda_link"]:
        val = row0.get(key)
        if val:
            result[key] = float(val)

    # Val metrics at best epoch
    for key in ["val_disp_mae", "val_force_mae", "val_disp_r2", "val_force_r2"]:
        val = best_row.get(key)
        if val:
            result[f"best_val_{key}"] = float(val)

    return result


def parse_model_summary(path: Path) -> dict:
    """Extract metadata from model_summary.json."""
    if not path.exists():
        return {"_warning": "model_summary.json not found"}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "model_name": data.get("model_name"),
        "model_class": data.get("model_class"),
        "total_params": data.get("total_params"),
        "trainable_params": data.get("trainable_params"),
    }

    phys = data.get("physics_loss", {})
    if phys:
        result["physics_enabled"] = phys.get("enabled", False)
        result["lambda_bc"] = phys.get("lambda_bc", 0.0)
        result["lambda_link"] = phys.get("lambda_link", 0.0)

    return result


def parse_config_yaml(path: Path) -> dict:
    """Extract training config from config_resolved.yaml."""
    if not path.exists():
        return {"_warning": "config_resolved.yaml not found"}

    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    result: Dict = {}
    if not isinstance(data, dict):
        return result

    result["dataset"] = data.get("dataset")
    result["split_mode"] = data.get("split_mode")
    result["batch_size"] = data.get("batch_size") or data.get("train", {}).get("batch_size")
    result["epochs"] = data.get("epochs") or data.get("train", {}).get("epochs")
    result["seed"] = data.get("seed") or data.get("train", {}).get("seed")
    result["device"] = data.get("device")

    # Physics loss (top-level takes precedence)
    lbc = data.get("lambda_bc")
    llk = data.get("lambda_link")
    if lbc is None:
        lbc = data.get("train", {}).get("lambda_bc")
    if llk is None:
        llk = data.get("train", {}).get("lambda_link")
    result["lambda_bc"] = lbc or 0.0
    result["lambda_link"] = llk or 0.0

    # Standardisation source
    std = data.get("standardisation", data.get("data", {}))
    if isinstance(std, dict):
        result["standardisation_source"] = std.get("stats_path") or std.get("feature_stats_path")
        result["standardisation_source"] = str(result["standardisation_source"]) if result.get("standardisation_source") else None

    # dataset_sizes (may be in data or eval section)
    for section in ["data", "eval"]:
        sec = data.get(section, {})
        if isinstance(sec, dict):
            for k in ["train", "val", "test"]:
                v = sec.get(k)
                if v is not None and k not in result:
                    result[f"dataset_{k}"] = v

    return result


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _build_component_metrics(per_mae: Optional[List[float]],
                              per_r2: Optional[List[float]],
                              log_macro_mae: Optional[float],
                              log_macro_r2: Optional[float],
                              overall_r2_approx: Optional[float]) -> dict:
    """Build the disp/force sub-dict matching the canonical schema."""
    n = len(per_mae) if per_mae else 0
    macro_mae = sum(per_mae) / n if per_mae and n > 0 else log_macro_mae
    macro_r2 = sum(per_r2) / n if per_r2 and n > 0 else log_macro_r2
    overall_r2 = overall_r2_approx or macro_r2

    return {
        "overall_mse": None,
        "overall_mae": log_macro_mae,
        "overall_rel_mae": None,
        "overall_r2": overall_r2,
        "macro_avg_mae": macro_mae,
        "macro_avg_rel_mae": None,
        "macro_avg_r2": macro_r2,
        "per_component_mae": per_mae if per_mae else None,
        "per_component_r2": per_r2 if per_r2 else None,
    }


def build_metrics_summary(run_dir: Path, overwrite: bool = False) -> dict:
    """Recover metrics_summary.json from files in run_dir."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    existing = run_dir / "metrics_summary.json"
    if existing.exists() and not overwrite:
        print(f"  [skip] metrics_summary.json already exists (use --overwrite to replace)")
        with open(existing, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"  Reading files in: {run_dir}")
    log_data = parse_server_log(run_dir / "server_output.log")
    csv_data = parse_train_log(run_dir / "train_log.csv")
    model_data = parse_model_summary(run_dir / "model_summary.json")
    config_data = parse_config_yaml(run_dir / "config_resolved.yaml")

    warnings = []
    for src_name, src in [
        ("server_output.log", log_data),
        ("train_log.csv", csv_data),
        ("model_summary.json", model_data),
        ("config_resolved.yaml", config_data),
    ]:
        w = src.pop("_warning", None)
        if w:
            warnings.append(f"{src_name}: {w}")

    # --- Resolve best_epoch ---
    best_epoch = csv_data.get("best_epoch") or log_data.get("best_epoch")
    if best_epoch is None:
        warnings.append("Could not determine best_epoch")
        best_epoch = None

    # --- Lambda values: csv > model_summary > config ---
    lambda_bc = csv_data.get("lambda_bc") or model_data.get("lambda_bc") or config_data.get("lambda_bc") or 0.0
    lambda_link = csv_data.get("lambda_link") or model_data.get("lambda_link") or config_data.get("lambda_link") or 0.0
    physics_enabled = model_data.get("physics_enabled", lambda_bc > 0.0 or lambda_link > 0.0)

    # --- Dataset sizes ---
    dataset_sizes = {
        "train": config_data.get("dataset_train"),
        "val": config_data.get("dataset_val"),
        "test": config_data.get("dataset_test"),
    }
    if not any(dataset_sizes.values()):
        dataset_sizes = {"train": None, "val": None, "test": None}

    # --- Test sub-dict ---
    test_disp = _build_component_metrics(
        per_mae=log_data.get("per_component_disp_mae"),
        per_r2=log_data.get("per_component_disp_r2"),
        log_macro_mae=log_data.get("test_disp_mae"),
        log_macro_r2=log_data.get("test_disp_r2"),
        overall_r2_approx=log_data.get("test_disp_r2"),
    )
    test_force = _build_component_metrics(
        per_mae=log_data.get("per_component_force_mae"),
        per_r2=log_data.get("per_component_force_r2"),
        log_macro_mae=log_data.get("test_force_mae"),
        log_macro_r2=log_data.get("test_force_r2"),
        overall_r2_approx=log_data.get("test_force_r2"),
    )

    summary = {
        "config": {
            "model_name": model_data.get("model_name"),
            "model_class": model_data.get("model_class"),
            "split_mode": config_data.get("split_mode"),
            "batch_size": config_data.get("batch_size"),
            "seed": config_data.get("seed"),
        },
        "num_params": model_data.get("total_params"),
        "dataset_sizes": dataset_sizes,
        "training": {
            "best_epoch": best_epoch,
            "stopped_epoch": csv_data.get("stopped_epoch"),
            "total_epochs": csv_data.get("total_epochs"),
            "early_stopped": log_data.get("early_stopped", False),
            "best_val_metric": csv_data.get("best_val_loss"),
            "best_val_disp_mae": csv_data.get("best_val_val_disp_mae"),
            "best_val_force_mae": csv_data.get("best_val_val_force_mae"),
            "best_val_disp_r2": csv_data.get("best_val_val_disp_r2"),
            "best_val_force_r2": csv_data.get("best_val_val_force_r2"),
            "total_time_seconds": log_data.get("training_time_seconds"),
        },
        "test": {
            "loss_standardised": csv_data.get("best_val_loss"),
            "combined_rel_mae": log_data.get("test_combined_rel_mae"),
            "disp": test_disp,
            "force": test_force,
        },
        "standardisation": {
            "source": config_data.get("standardisation_source") or
                      "processed/hetero_graph_dataset_v2/feature_stats.json",
            "train_only": True,
        },
        "recovered_from_existing_artifact": True,
        "recovery_warnings": warnings if warnings else None,
    }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recover metrics_summary.json from existing run directory."
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to experiment run directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing metrics_summary.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary = build_metrics_summary(run_dir, overwrite=args.overwrite)

    output_path = run_dir / "metrics_summary.json"
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n  (dry-run — would write to {output_path})")
        return 0

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [ok] metrics_summary.json written")
    print(f"       {output_path}")

    if summary.get("recovery_warnings"):
        print(f"  [warn] Recovery warnings:")
        for w in summary["recovery_warnings"]:
            print(f"    - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

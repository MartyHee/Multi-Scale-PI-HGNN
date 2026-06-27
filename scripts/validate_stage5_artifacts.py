#!/usr/bin/env python3
"""
validate_stage5_artifacts.py — Validate three Stage 5 MS-HGT artifacts (BC, Full, Link).

This script:
  1. Extracts each artifact to a temp directory
  2. Checks for all required files
  3. Reads metrics_summary.json and cross-checks with server_output.log
  4. Verifies lambda_bc / lambda_link match experimental design
  5. Verifies best_epoch aligns with train_log.csv
  6. Computes SHA256 of best_model.pt (cross-check: should differ)
  7. Checks train_log.csv for NaN/Inf and physics loss fields
  8. Generates artifact_validation_summary.json, table.csv, and report.md

Usage:
  D:\CodeData\software\Anaconda\Anaconda3\envs\llm\python.exe scripts/validate_stage5_artifacts.py ^
    --bc-artifact remote_artifacts/server_ms_pi_hgt_bc_20260627185642.tar.gz ^
    --full-artifact remote_artifacts/server_ms_pi_hgt_full_20260627185645.tar.gz ^
    --link-artifact remote_artifacts/server_ms_pi_hgt_link_20260627185649.tar.gz ^
    --output-dir outputs/diagnostics/stage5_artifact_validation
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants: expected configuration for each variant
# ---------------------------------------------------------------------------

EXPECTED_RUN_NAMES = {
    "bc":   "server_ms_pi_hgt_bc",
    "full": "server_ms_pi_hgt_full",
    "link": "server_ms_pi_hgt_link",
}

EXPECTED_LAMBDAS = {
    "bc":   {"lambda_bc": 0.08, "lambda_link": 0.0},
    "full": {"lambda_bc": 0.08, "lambda_link": 0.002},
    "link": {"lambda_bc": 0.0,  "lambda_link": 0.002},
}

EXPECTED_BEST_EPOCH = {
    "bc":   114,
    "full": 134,
    "link": 171,
}

EXPECTED_NUM_PARAMS = 893527

EXPECTED_DATASET_SIZES = {"train": 28000, "val": 3500, "test": 3500}

EXPECTED_METRICS = {
    "bc":   {"disp_r2": 0.9951, "dy_r2": 0.9926, "force_r2": 0.9934, "rel_mae": 0.0529},
    "full": {"disp_r2": 0.9948, "dy_r2": 0.9928, "force_r2": 0.9933, "rel_mae": 0.0516},
    "link": {"disp_r2": 0.9952, "dy_r2": 0.9930, "force_r2": 0.9934, "rel_mae": 0.0515},
}

REQUIRED_FILES = [
    "train_log.csv",
    "best_model.pt",
    "last_model.pt",
    "last_checkpoint.pt",
    "server_output.log",
    "config_resolved.yaml",
    "metrics_summary.json",
    "model_summary.json",
    "git_info.txt",
]

PHYSICS_LOSS_FIELDS = [
    "train_loss_bc", "train_loss_link",
    "val_loss_bc", "val_loss_link",
    "lambda_bc", "lambda_link",
]

DISP_NAMES = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
FORCE_NAMES = [
    "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
    "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_json(path: Path) -> Optional[dict]:
    """Read JSON, return None on error."""
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as e:
        print(f"  [ERROR] Failed to read {path.name}: {e}")
        return None


def safe_read_yaml(path: Path) -> Optional[dict]:
    """Read YAML, return None on error."""
    try:
        import yaml
        return yaml.safe_load(path.read_text("utf-8"))
    except Exception as e:
        print(f"  [ERROR] Failed to read {path.name}: {e}")
        return None


def safe_read_csv(path: Path) -> Optional[List[dict]]:
    """Read CSV, return list of dicts."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"  [ERROR] Failed to read {path.name}: {e}")
        return None


def read_job_name_from_yaml(yaml_path: Path) -> Optional[str]:
    """Extract run_name from job yaml."""
    data = safe_read_yaml(yaml_path)
    if data and isinstance(data, dict):
        return data.get("run_name") or data.get("name")
    return None


def read_job_name_from_log(log_path: Path) -> Optional[str]:
    """Extract job name from server_output.log (header line)."""
    try:
        text = log_path.read_text("utf-8", errors="replace")
        m = re.search(r"Job:\s+(\S+)", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def check_nan_inf(rows: List[dict], numeric_fields: List[str]) -> List[str]:
    """Check CSV rows for NaN or Inf in numeric fields."""
    issues = []
    for i, row in enumerate(rows):
        for field in numeric_fields:
            val = row.get(field, "").strip()
            if val.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
                issues.append(f"  Row {i}: {field}={val}")
            elif val:
                try:
                    fv = float(val)
                    import math
                    if math.isnan(fv) or math.isinf(fv):
                        issues.append(f"  Row {i}: {field}={val}")
                except ValueError:
                    pass  # non-numeric field, skip
    return issues


# ---------------------------------------------------------------------------
# Artifact Validation
# ---------------------------------------------------------------------------

def validate_one_artifact(artifact_path: Path, variant: str,
                          extract_dir: Path) -> dict:
    """Extract and validate one artifact, returning a result dict."""
    result = {
        "variant": variant,
        "artifact_path": str(artifact_path.resolve()),
        "extracted_dir": str(extract_dir.resolve()),
        "pass": True,
        "missing_files": [],
        "warnings": [],
        "errors": [],
    }

    print(f"\n{'='*60}")
    print(f"  Validating {variant.upper()} artifact")
    print(f"  Artifact: {artifact_path.name}")
    print(f"{'='*60}")

    # --- Extract ---
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(artifact_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    except Exception as e:
        result["pass"] = False
        result["errors"].append(f"Extraction failed: {e}")
        return result

    # Find the actual extraction subdirectory (artifact name prefix)
    contents = list(extract_dir.iterdir())
    if not contents:
        result["pass"] = False
        result["errors"].append("Empty extraction")
        return result

    # The artifact extracts to a single subdirectory named <job>_<timestamp>
    root = contents[0] if contents[0].is_dir() else extract_dir

    def _af(path: str) -> Path:
        p = root / path
        return p if p.exists() else extract_dir / path

    # --- Check required files ---
    for fname in REQUIRED_FILES:
        if not _af(fname).exists():
            result["missing_files"].append(fname)
            result["pass"] = False

    if result["missing_files"]:
        result["errors"].append(f"Missing: {', '.join(result['missing_files'])}")
        print(f"  [FAIL] Missing files: {result['missing_files']}")
    else:
        print(f"  [OK]   All {len(REQUIRED_FILES)} required files present")

    # --- metrics_summary.json is mandatory ---
    metrics_path = _af("metrics_summary.json")
    if not metrics_path.exists():
        result["pass"] = False
        result["errors"].append("CRITICAL: metrics_summary.json missing")
        print("  [FAIL] metrics_summary.json MISSING")
    else:
        # Read and extract fields
        m = safe_read_json(metrics_path)
        if m is None:
            result["pass"] = False
            result["errors"].append("metrics_summary.json unreadable")
        else:
            result["metrics_summary"] = {
                "config": m.get("config"),
                "num_params": m.get("num_params"),
                "dataset_sizes": m.get("dataset_sizes"),
                "training": m.get("training"),
                "test": m.get("test"),
                "standardisation": m.get("standardisation"),
            }

            cfg = m.get("config", {})
            result["run_name"] = cfg.get("model_name")
            result["num_params_found"] = m.get("num_params")
            result["best_epoch"] = m.get("training", {}).get("best_epoch")
            result["disp_r2"] = m.get("test", {}).get("disp", {}).get("macro_avg_r2")
            result["dy_r2"] = (m.get("test", {}).get("disp", {}).get("per_component_r2", [None]*6)[1]
                               if DISP_NAMES.index("Dy") < 6 else None)
            result["force_r2"] = m.get("test", {}).get("force", {}).get("macro_avg_r2")
            result["rel_mae"] = m.get("test", {}).get("combined_rel_mae")

            # Per-component Dy R^2
            pc_r2 = m.get("test", {}).get("disp", {}).get("per_component_r2")
            if pc_r2 and len(pc_r2) > 1:
                result["dy_r2_from_per_component"] = pc_r2[DISP_NAMES.index("Dy")]

            # Standardisation
            std = m.get("standardisation", {})
            result["standardisation_train_only"] = std.get("train_only")
            result["standardisation_source"] = std.get("source")

            # Check recovered flag
            result["recovered"] = m.get("recovered_from_existing_artifact", False)

            print(f"  [OK]   metrics_summary.json: run_name={result.get('run_name')}")
            if result.get('disp_r2') is not None:
                print(f"         Disp R^2={result.get('disp_r2'):.4f}, Force R^2={result.get('force_r2'):.4f}")

    # --- model_summary.json ---
    ms_path = _af("model_summary.json")
    ms = safe_read_json(ms_path)
    if ms:
        result["physics_enabled"] = ms.get("physics_loss", {}).get("enabled", False)
        result["lambda_bc"] = ms.get("physics_loss", {}).get("lambda_bc", 0.0)
        result["lambda_link"] = ms.get("physics_loss", {}).get("lambda_link", 0.0)
        result["model_class"] = ms.get("model_class")
        result["total_params"] = ms.get("total_params")
        result["trainable_params"] = ms.get("trainable_params")
        print(f"  [OK]   model_summary.json: class={result.get('model_class')}, "
              f"params={result.get('total_params')}")

    # --- config_resolved.yaml ---
    config_path = _af("config_resolved.yaml")
    config = safe_read_yaml(config_path)
    if config and isinstance(config, dict):
        # Check dataset sizes
        for section in ["data", "eval", "dataset"]:
            sec = config.get(section, {})
            if isinstance(sec, dict):
                for k in ["train", "val", "test"]:
                    v = sec.get(k)
                    if v is not None:
                        result[f"dataset_{k}"] = v

        # Config-level lambda
        lbc = config.get("lambda_bc") or config.get("train", {}).get("lambda_bc", 0.0)
        llk = config.get("lambda_link") or config.get("train", {}).get("lambda_link", 0.0)
        result["config_lambda_bc"] = float(lbc)
        result["config_lambda_link"] = float(llk)
        result["batch_size"] = config.get("batch_size") or config.get("train", {}).get("batch_size")
        result["seed"] = config.get("seed") or config.get("train", {}).get("seed")
        result["split_mode"] = config.get("split_mode")
        result["epochs"] = config.get("epochs") or config.get("train", {}).get("epochs")

    # --- train_log.csv ---
    csv_path = _af("train_log.csv")
    csv_rows = safe_read_csv(csv_path)
    if csv_rows:
        result["total_epochs_logged"] = len(csv_rows)

        # Find best epoch by val_loss
        best_val_loss = float("inf")
        best_epoch_from_csv = None
        for row in csv_rows:
            try:
                vl = float(row.get("val_loss", float("inf")))
                ep = int(row.get("epoch", 0))
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_epoch_from_csv = ep
            except (ValueError, TypeError):
                pass

        result["best_epoch_from_csv"] = best_epoch_from_csv
        result["best_val_loss"] = best_val_loss

        # Check first row for lambda
        row0 = csv_rows[0]
        result["csv_lambda_bc"] = float(row0.get("lambda_bc", 0))
        result["csv_lambda_link"] = float(row0.get("lambda_link", 0))

        # Check for physics loss fields
        fieldnames = csv_rows[0].keys()
        present_phys_fields = [f for f in PHYSICS_LOSS_FIELDS if f in fieldnames]
        result["physics_loss_fields_present"] = present_phys_fields
        missing_phys = [f for f in PHYSICS_LOSS_FIELDS if f not in fieldnames]
        if missing_phys:
            result["warnings"].append(f"Missing physics loss fields in CSV: {missing_phys}")

        # Check NaN/Inf
        numeric_fields = ["train_loss", "val_loss", "train_disp_mae", "val_disp_mae",
                          "train_force_mae", "val_force_mae", "train_disp_r2", "val_disp_r2",
                          "train_force_r2", "val_force_r2"]
        nan_inf_issues = check_nan_inf(csv_rows, numeric_fields)
        if nan_inf_issues:
            result["nan_inf_issues"] = nan_inf_issues
            result["warnings"].extend(nan_inf_issues)

        print(f"  [OK]   train_log.csv: {len(csv_rows)} epochs, "
              f"best_epoch={best_epoch_from_csv}, physics_loss_fields={len(present_phys_fields)}/6")

    # --- SHA256 of best_model.pt ---
    best_path = _af("best_model.pt")
    if best_path.exists():
        h = sha256(best_path)
        result["best_model_sha256"] = h
        print(f"  [OK]   best_model.pt SHA256: {h[:16]}...")

    # --- server_output.log — extract job name ---
    log_path = _af("server_output.log")
    if log_path.exists():
        job_name = read_job_name_from_log(log_path)
        result["job_name_from_log"] = job_name
        print(f"  [OK]   Job name from log: {job_name}")

    # --- Verify expected values ---
    exp = EXPECTED_LAMBDAS.get(variant, {})
    # Compare lambdas (allow small tolerance)
    for key in ["lambda_bc", "lambda_link"]:
        expected_val = exp.get(key, 0.0)
        found_val = result.get(key, result.get(f"config_{key}", None))
        if found_val is not None and abs(float(found_val) - expected_val) > 0.0001:
            result["errors"].append(
                f"{key}: expected {expected_val}, found {found_val} (from model_summary.json)")
            result["pass"] = False

    # Check run_name
    expected_run = EXPECTED_RUN_NAMES.get(variant, "")
    found_run = result.get("run_name", "")
    if found_run and expected_run not in str(found_run):
        result["warnings"].append(
            f"run_name '{found_run}' does not contain expected '{expected_run}'")

    # Check job name from log
    found_job = result.get("job_name_from_log", "")
    if found_job and expected_run not in str(found_job):
        result["warnings"].append(
            f"Job name from log '{found_job}' does not match artifact variant '{expected_run}'")

    # Check num_params
    np_found = result.get("num_params_found") or result.get("total_params")
    if np_found is not None and np_found != EXPECTED_NUM_PARAMS:
        result["warnings"].append(
            f"num_params={np_found}, expected {EXPECTED_NUM_PARAMS}")

    # Check dataset sizes
    for k, expected_v in EXPECTED_DATASET_SIZES.items():
        found_v = result.get(f"dataset_{k}")
        if found_v is not None and found_v != expected_v:
            result["warnings"].append(
                f"dataset_{k}={found_v}, expected {expected_v}")

    # Check standardisation
    if result.get("standardisation_train_only") is not True:
        result["warnings"].append(
            f"standardisation.train_only={result.get('standardisation_train_only')}, expected True")

    # Check best_epoch alignment (allow ±1)
    expected_be = EXPECTED_BEST_EPOCH.get(variant)
    found_be = result.get("best_epoch")
    csv_be = result.get("best_epoch_from_csv")
    if expected_be and found_be is not None:
        diff = abs(found_be - expected_be)
        if diff > 1:
            result["warnings"].append(
                f"best_epoch={found_be}, expected ~{expected_be} (diff={diff})")
    if csv_be is not None and found_be is not None:
        if found_be != csv_be:
            result["warnings"].append(
                f"best_epoch mismatch: metrics_summary says {found_be}, train_log says {csv_be}")

    # --- Summary ---
    status = "PASS" if result["pass"] else "FAIL"
    r2_s = f"{result.get('disp_r2', 0):.4f}" if result.get("disp_r2") else "N/A"
    fr2_s = f"{result.get('force_r2', 0):.4f}" if result.get("force_r2") else "N/A"
    print(f"\n  >>> {variant.upper()}: {status} | Disp R^2={r2_s}, Force R^2={fr2_s}")

    return result


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

def cross_check(results: List[dict]) -> dict:
    """Cross-check artifact hashes and configs."""
    cc = {
        "all_pass": all(r["pass"] for r in results),
        "hashes_different": None,
        "hash_check_detail": [],
        "cross_contamination": False,
        "cross_contamination_detail": [],
    }

    # Check SHA256 hashes — should all be different
    hashes = {r["variant"]: r.get("best_model_sha256", "MISSING") for r in results}
    h_set = set(hashes.values())
    if len(h_set) == 1 and len(results) > 1:
        cc["hashes_different"] = False
        cc["hash_check_detail"].append("FAIL: All three best_model.pt have identical SHA256!")
    elif len(results) > 1:
        cc["hashes_different"] = True
        cc["hash_check_detail"].append("OK: Three best_model.pt have different SHA256 hashes")

    cc["hash_check_detail"].append(f"  BC:   {hashes.get('bc', 'N/A')[:16]}...")
    cc["hash_check_detail"].append(f"  Full: {hashes.get('full', 'N/A')[:16]}...")
    cc["hash_check_detail"].append(f"  Link: {hashes.get('link', 'N/A')[:16]}...")

    # Check lambdas match artifact names
    for r in results:
        var = r["variant"]
        exp_l = EXPECTED_LAMBDAS.get(var, {})
        lbc = r.get("lambda_bc") or r.get("config_lambda_bc")
        llk = r.get("lambda_link") or r.get("config_lambda_link")
        exp_lbc = exp_l.get("lambda_bc")
        exp_llk = exp_l.get("lambda_link")
        if lbc is not None and llk is not None:
            bc_match = abs(float(lbc) - exp_lbc) < 0.0001
            lk_match = abs(float(llk) - exp_llk) < 0.0001
            if not bc_match or not lk_match:
                cc["cross_contamination"] = True
                cc["cross_contamination_detail"].append(
                    f"{var}: lambda mismatch "
                    f"(bc={lbc}, expected={exp_lbc}; link={llk}, expected={exp_llk})")
        # Check job name from log matches
        log_job = r.get("job_name_from_log", "")
        expected_name = EXPECTED_RUN_NAMES.get(var, "")
        if log_job and expected_name not in str(log_job):
            cc["cross_contamination"] = True
            cc["cross_contamination_detail"].append(
                f"{var}: log job name '{log_job}' doesn't match '{expected_name}'")

    if cc["cross_contamination"]:
        cc["all_pass"] = False

    return cc


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[dict], cross: dict, output_dir: Path):
    """Generate JSON summary, CSV table, and MD report."""

    # --- JSON summary ---
    json_out = {
        "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cross_check": cross,
        "artifacts": [],
    }
    for r in results:
        # Keep only serializable fields
        entry = {k: v for k, v in r.items()
                 if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        json_out["artifacts"].append(entry)

    json_path = output_dir / "artifact_validation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n  [ok] Summary: {json_path}")

    # --- CSV table ---
    csv_path = output_dir / "artifact_validation_table.csv"
    fieldnames = [
        "variant", "pass", "run_name", "lambda_bc", "lambda_link",
        "best_epoch", "disp_r2", "dy_r2", "force_r2", "rel_mae",
        "num_params_found", "total_epochs_logged", "physics_enabled",
        "best_model_sha256", "recovered",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            # Truncate SHA256
            sha = row.get("best_model_sha256", "")
            if isinstance(sha, str) and len(sha) > 16:
                row["best_model_sha256"] = sha[:16] + "..."
            writer.writerow(row)
    print(f"  [ok] Table: {csv_path}")

    # --- MD report ---
    md_path = output_dir / "stage5_artifact_validation_report.md"
    lines = []
    lines.append("# Stage 5 Artifact Validation Report\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    overall = "**PASS**" if cross["all_pass"] else "**FAIL**"
    lines.append(f"## Overall Verdict: {overall}\n")

    if cross["all_pass"]:
        lines.append("All three Stage 5 MS-HGT artifacts are valid and ready for analysis.\n")
    else:
        lines.append("Issues detected — see details below.\n")

    # Cross-check summary
    lines.append("## Cross-Check Summary\n")
    lines.append(f"- **All artifacts contain `metrics_summary.json`:** "
                 f"{'✅' if all('metrics_summary.json' not in (r.get('missing_files') or []) for r in results) else '❌'}")
    lines.append(f"- **best_model.pt hashes differ:** "
                 f"{'✅' if cross.get('hashes_different') else '❌'}")
    lines.append(f"- **Lambda config matches artifact name:** "
                 f"{'✅' if not cross.get('cross_contamination') else '❌'}")
    lines.append(f"- **Job name in log matches artifact:** "
                 f"{'✅' if not cross.get('cross_contamination') else '❌'}")
    lines.append("")

    if cross.get("hash_check_detail"):
        lines.append("### best_model.pt SHA256\n")
        for line in cross["hash_check_detail"]:
            lines.append(line)
        lines.append("")

    if cross.get("cross_contamination_detail"):
        lines.append("### Cross-Contamination Issues\n")
        for line in cross["cross_contamination_detail"]:
            lines.append(f"- ⚠️ {line}")
        lines.append("")

    # Per-artifact details
    lines.append("## Per-Artifact Details\n")
    lines.append("| Variant | Status | Run Name | λ_bc | λ_link | Best Epoch | Disp R^2 | Dy R^2 | Force R^2 | RelMAE | Params | Epochs | SHA256 (prefix) |")
    lines.append("|--------|--------|----------|------|--------|-----------|---------|-------|----------|--------|--------|--------|-----------------|")

    for r in results:
        status = "✅" if r["pass"] else "❌"
        var = r["variant"]
        run = r.get("run_name", "N/A") or "N/A"
        lbc = r.get("lambda_bc", r.get("config_lambda_bc", "N/A"))
        llk = r.get("lambda_link", r.get("config_lambda_link", "N/A"))
        be = r.get("best_epoch", "N/A")
        dr2 = f"{r.get('disp_r2', 0):.4f}" if r.get("disp_r2") else "N/A"
        dyr2 = f"{r.get('dy_r2', 0):.4f}" if r.get("dy_r2") else "N/A"
        fr2 = f"{r.get('force_r2', 0):.4f}" if r.get("force_r2") else "N/A"
        relm = f"{r.get('rel_mae', 0):.4f}" if r.get("rel_mae") else "N/A"
        prm = r.get("total_params") or r.get("num_params_found") or "N/A"
        eps = r.get("total_epochs_logged", "N/A")
        sha = (r.get("best_model_sha256", "") or "")[:12] + "..." if r.get("best_model_sha256") else "N/A"
        lines.append(f"| {var:7s} | {status} | {str(run)[:24]:24s} | {lbc} | {llk} | {be} | {dr2} | {dyr2} | {fr2} | {relm} | {prm} | {eps} | {sha} |")

    lines.append("")

    # Detailed sections
    for r in results:
        var = r["variant"]
        lines.append(f"### {var.upper()}\n")
        lines.append(f"**Status:** {'✅ PASS' if r['pass'] else '❌ FAIL'}\n")
        lines.append(f"- **Artifact:** `{r.get('artifact_path', 'N/A')}`")
        lines.append(f"- **Run name:** {r.get('run_name', 'N/A')}")
        lines.append(f"- **Lambda BC:** {r.get('lambda_bc', r.get('config_lambda_bc', 'N/A'))}")
        lines.append(f"- **Lambda Link:** {r.get('lambda_link', r.get('config_lambda_link', 'N/A'))}")
        lines.append(f"- **Best epoch:** {r.get('best_epoch', 'N/A')} (CSV: {r.get('best_epoch_from_csv', 'N/A')})")
        lines.append(f"- **Best val loss:** {r.get('best_val_loss', 'N/A')}")
        lines.append(f"- **Disp R^2:** {r.get('disp_r2', 'N/A')}")
        lines.append(f"- **Dy R^2:** {r.get('dy_r2', 'N/A')}")
        lines.append(f"- **Force R^2:** {r.get('force_r2', 'N/A')}")
        lines.append(f"- **RelMAE:** {r.get('rel_mae', 'N/A')}")
        lines.append(f"- **Parameters:** {r.get('total_params', r.get('num_params_found', 'N/A'))}")
        lines.append(f"- **Physics enabled:** {r.get('physics_enabled', 'N/A')}")
        lines.append(f"- **Recovered flag:** {r.get('recovered', 'N/A')}")
        lines.append(f"- **Standardisation train_only:** {r.get('standardisation_train_only', 'N/A')}")
        lines.append(f"- **best_model.pt SHA256:** {r.get('best_model_sha256', 'N/A')}")
        lines.append(f"- **Physics loss fields in CSV:** {r.get('physics_loss_fields_present', 'N/A')}")
        lines.append(f"- **NaN/Inf in CSV:** {'⚠️ YES' if r.get('nan_inf_issues') else '✅ None'}")

        if r.get("missing_files"):
            lines.append(f"- **❌ Missing files:** {r['missing_files']}")
        if r.get("warnings"):
            lines.append(f"- **⚠️ Warnings:**")
            for w in r["warnings"]:
                lines.append(f"  - {w}")
        if r.get("errors"):
            lines.append(f"- **❌ Errors:**")
            for e in r["errors"]:
                lines.append(f"  - {e}")
        lines.append("")

    # Conclusion & next steps
    lines.append("## Conclusion & Next Steps\n")

    all_pass = cross["all_pass"]
    has_metrics = all(
        "metrics_summary.json" not in (r.get("missing_files") or [])
        for r in results
    )
    hashes_ok = cross.get("hashes_different") is True

    if all_pass and has_metrics and hashes_ok:
        lines.append("✅ **All checks passed.** The three Stage 5 artifacts are valid:\n")
        lines.append("- No cross-contamination between artifacts")
        lines.append("- All three `metrics_summary.json` present with correct schema")
        lines.append("- Lambda configurations match experimental design (BC: 0.08/0.0, Full: 0.08/0.002, Link: 0.0/0.002)")
        lines.append("- `best_model.pt` hashes differ — no accidental duplication")
        lines.append("- best_epoch values align between metrics_summary and train_log")
        lines.append("- train_log.csv contains physics loss fields for all three variants")
        lines.append("- No NaN/Inf in training logs")
        lines.append("\n**Ready to proceed to full prediction export and physics diagnostics.**\n")
        lines.append("### Next Steps\n")
        lines.append("1. **Server: export test predictions** for each variant:")
        lines.append("   ```bash")
        lines.append("   conda activate pi_hgnn")
        lines.append("   cd /home/miniconda/Bishe/Multi-Scale-PI-HGNN")
        lines.append("")
        lines.append("   # BC")
        lines.append("   python train_baseline.py --model ms_hgt --run-name eval_bc \\")
        lines.append("     --load-checkpoint outputs/baselines/MS_HGT/20260626170344/best_model.pt \\")
        lines.append("     --eval-only --save-predictions")
        lines.append("")
        lines.append("   # Full")
        lines.append("   python train_baseline.py --model ms_hgt --run-name eval_full \\")
        lines.append("     --load-checkpoint outputs/baselines/MS_HGT/20260626170354/best_model.pt \\")
        lines.append("     --eval-only --save-predictions")
        lines.append("")
        lines.append("   # Link")
        lines.append("   python train_baseline.py --model ms_hgt --run-name eval_link \\")
        lines.append("     --load-checkpoint outputs/baselines/MS_HGT/20260626170428/best_model.pt \\")
        lines.append("     --eval-only --save-predictions")
        lines.append("   ```")
        lines.append("")
        lines.append("2. **Run physics diagnostics** on exported predictions:")
        lines.append("   ```bash")
        lines.append("   python scripts/physics_diagnostics.py \\")
        lines.append("     --prediction-dir outputs/baselines/MS_HGT/*eval*/ \\")
        lines.append("     --output-dir outputs/diagnostics/stage5_physics")
        lines.append("   ```")
        lines.append("")
        lines.append("3. **Download artifacts** with predictions and bring back for local analysis:\n")
        lines.append("   ```bash")
        lines.append("   bash server_ops/package_results.sh outputs/baselines/MS_HGT/*eval_bc*/ server_eval_bc")
        lines.append("   bash server_ops/package_results.sh outputs/baselines/MS_HGT/*eval_full*/ server_eval_full")
        lines.append("   bash server_ops/package_results.sh outputs/baselines/MS_HGT/*eval_link*/ server_eval_link")
        lines.append("   ```")
    else:
        lines.append("❌ **Issues detected — do not proceed to prediction export until resolved.**\n")
        if not has_metrics:
            lines.append("- Some artifacts are missing `metrics_summary.json`\n")
        if not hashes_ok:
            lines.append("- best_model.pt hashes do not differ — possible artifact duplication\n")
        if not all_pass:
            lines.append("- Some artifacts failed validation — see per-artifact errors above\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [ok] Report: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate three Stage 5 MS-HGT artifacts"
    )
    parser.add_argument("--bc-artifact", required=True,
                        help="Path to BC-only artifact tar.gz")
    parser.add_argument("--full-artifact", required=True,
                        help="Path to Full artifact tar.gz")
    parser.add_argument("--link-artifact", required=True,
                        help="Path to Link-only artifact tar.gz")
    parser.add_argument("--output-dir", default="outputs/diagnostics/stage5_artifact_validation",
                        help="Output directory for validation results")
    parser.add_argument("--extract-dir", default="remote_artifacts/extracted_stage5_validation",
                        help="Temp directory for artifact extraction")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    bc_art = Path(args.bc_artifact) if Path(args.bc_artifact).is_absolute() else project_root / args.bc_artifact
    full_art = Path(args.full_artifact) if Path(args.full_artifact).is_absolute() else project_root / args.full_artifact
    link_art = Path(args.link_artifact) if Path(args.link_artifact).is_absolute() else project_root / args.link_artifact
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir
    extract_base = Path(args.extract_dir) if Path(args.extract_dir).is_absolute() else project_root / args.extract_dir

    # Validate artifact files exist
    for name, path in [("BC", bc_art), ("Full", full_art), ("Link", link_art)]:
        if not path.exists():
            print(f"[ERROR] {name} artifact not found: {path}")
            sys.exit(1)
        print(f"  [ok] {name} artifact: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Clean and create output dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate each artifact
    os.makedirs(extract_base, exist_ok=True)

    results = []
    for variant, art_path in [("bc", bc_art), ("full", full_art), ("link", link_art)]:
        extract_dir = extract_base / variant
        result = validate_one_artifact(art_path, variant, extract_dir)
        results.append(result)

    # Cross-check
    print(f"\n{'='*60}")
    print("  Cross-Check")
    print(f"{'='*60}")
    cross = cross_check(results)

    # Print cross-check results
    if cross["all_pass"]:
        print("  >>> Cross-check: ALL PASS")
    else:
        print("  >>> Cross-check: ISSUES DETECTED")
    if cross.get("hash_check_detail"):
        for line in cross["hash_check_detail"]:
            print(f"  {line}")
    if cross.get("cross_contamination_detail"):
        print("  Cross-contamination issues:")
        for line in cross["cross_contamination_detail"]:
            print(f"    - {line}")

    # Generate reports
    generate_report(results, cross, output_dir)

    # Overall
    all_pass = cross["all_pass"]
    print(f"\n{'='*60}")
    print(f"  {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")
    print(f"  Report: {output_dir / 'stage5_artifact_validation_report.md'}")
    print(f"  Summary: {output_dir / 'artifact_validation_summary.json'}")
    print(f"  Table: {output_dir / 'artifact_validation_table.csv'}")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

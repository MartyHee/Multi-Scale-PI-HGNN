# -*- coding: utf-8 -*-
"""
check_paper_upgrade_configs.py — fairness check for paper-upgrade model configs.

For every model that appears in the paper-upgrade experiment matrix, report:

  1. model-level hyper-params (hidden_dim, dropout, activation, use_layer_norm,
     decoder_hidden_dims, num_layers);
  2. train-level hyper-params from configs/train_baseline.yaml (optimizer, lr,
     epochs, split_mode) — identical across all models by construction;
  3. parameter count (built on CPU via train_baseline.build_model);
  4. batch_size (from the seed42 job yaml in remote_jobs/paper_upgrade/);
  5. explicit confounder flags.

The main confounder to surface is **batch_size**: HGT-6/HGT-9 use batch_size=4
(for memory), while HGT-3/MS-HGT use batch_size=8. This is a potential training
confounder and must be reported transparently, NOT hidden.

Output:
  docs/paper_upgrade_config_fairness_check.md

Usage (from the project root, using the local llm interpreter):
  D:/CodeData/software/Anaconda/Anaconda3/envs/llm/python.exe \\
      scripts/paper_check/check_paper_upgrade_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths (assume cwd = project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs"
PAPER_CONFIG_DIR = CONFIG_DIR / "paper_upgrade"
JOBS_DIR = PROJECT_ROOT / "remote_jobs" / "paper_upgrade"
OUTPUT_DOC = PROJECT_ROOT / "docs" / "paper_upgrade_config_fairness_check.md"

# Models in the paper-upgrade matrix (references + variants)
MODEL_KEYS = [
    # (display_name, model_key, config_source, seed42_job_name or None)
    ("HGT-3 (reference)", "hgt", "configs/models_baseline.yaml -> hgt", None),
    ("MS-HGT gated (reference)", "ms_hgt", "configs/models_baseline.yaml -> ms_hgt", None),
    ("HGT-6", "hgt6", "paper_upgrade/hgt_6layer.yaml", "paper_hgt_6_seed42.yaml"),
    ("HGT-9", "hgt9", "paper_upgrade/hgt_9layer.yaml", "paper_hgt_9_seed42.yaml"),
    ("MS-HGT random-anchor", "ms_hgt_random_anchor", "paper_upgrade/ms_hgt_random_anchor.yaml", "paper_ms_hgt_random_anchor_seed42.yaml"),
    ("HGT+global (mesh_global)", "hgt_global", "paper_upgrade/hgt_global_mesh.yaml", "paper_hgt_global_mesh_seed42.yaml"),
    ("Edge-aware MPNN", "edge_aware_mpnn", "paper_upgrade/edge_aware_mpnn.yaml", "paper_edge_aware_mpnn_seed42.yaml"),
]

TRAIN_KEYS = ["optimizer", "lr", "weight_decay", "scheduler", "scheduler_patience",
              "scheduler_factor", "early_stopping_patience", "early_stopping_metric",
              "loss_name", "lambda_disp", "lambda_force"]
DATA_KEYS = ["split_mode"]

REFERENCE_BATCH = 8  # HGT-3 / MS-HGT used batch_size=8


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_params(model_key: str, model_cfg: dict) -> int:
    """Build the model on CPU and count parameters (defensive)."""
    try:
        import torch
        import train_baseline as tb
        model = tb.build_model(model_key, model_cfg, torch.device("cpu"))
        return sum(p.numel() for p in model.parameters())
    except Exception as exc:  # noqa: BLE001 — report and continue
        print(f"  [warn] param build failed for {model_key}: {exc}")
        return -1


def _job_batch_size(job_name: str | None) -> int | None:
    """Extract batch_size from a seed42 job yaml (mirrors run_job.sh parser)."""
    if not job_name:
        return None
    job_path = JOBS_DIR / job_name
    if not job_path.is_file():
        return None
    text = job_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("batch_size:"):
            return int(line.split(":", 1)[1].strip())
    return None


def main() -> None:
    print("=" * 60)
    print("Paper-upgrade config fairness check")
    print("=" * 60)

    train_cfg = _load_yaml(CONFIG_DIR / "train_baseline.yaml")
    models_cfg = _load_yaml(CONFIG_DIR / "models_baseline.yaml")

    # Resolve model config for each entry (models_baseline.yaml first,
    # then paper_upgrade fallback — same rule as load_baseline_configs:
    # the fallback file is named by the CONFIG key, e.g. hgt6 -> hgt_6layer).
    import train_baseline as tb
    rows = []
    for display, key, source, job in MODEL_KEYS:
        cfg = models_cfg.get(key)
        cfg_key = tb.MODEL_CONFIG_KEYS[key]
        if not cfg:
            cfg = _load_yaml(PAPER_CONFIG_DIR / f"{cfg_key}.yaml")
        if not cfg:
            print(f"  [warn] no config resolved for {key}")
            continue

        params = _build_params(key, cfg)
        batch = _job_batch_size(job)

        rows.append({
            "display": display,
            "model_key": key,
            "source": source,
            "hidden_dim": cfg.get("hidden_dim"),
            "num_layers": cfg.get("num_layers"),
            "heads": cfg.get("heads"),
            "dropout": cfg.get("dropout"),
            "activation": cfg.get("activation"),
            "use_layer_norm": cfg.get("use_layer_norm"),
            "decoder_hidden_dims": cfg.get("decoder_hidden_dims"),
            "params": params,
            "batch_size": batch,
        })
        print(f"  {display:26s} hidden={cfg.get('hidden_dim'):>4} layers={cfg.get('num_layers'):>3} "
              f"dropout={cfg.get('dropout')} params={params:>9,} batch={batch}")

    # ---- Train-level fairness (should be identical by construction) ----
    print("\n  Train-level keys (configs/train_baseline.yaml):")
    for k in TRAIN_KEYS:
        print(f"    {k:28s} = {train_cfg.get('train', {}).get(k)}")
    for k in DATA_KEYS:
        print(f"    {k:28s} = {train_cfg.get('data', {}).get(k)}")
    train_seed = train_cfg.get("train", {}).get("seed")
    print(f"    {'seed (default)':28s} = {train_seed}")

    # ---- Parameter spread ----
    param_values = [r["params"] for r in rows if r["params"] > 0]
    if param_values:
        lo, hi = min(param_values), max(param_values)
        print(f"\n  Param range: {lo:,} .. {hi:,}  (ratio {hi / max(lo, 1):.2f}x)")

    # ---- Write markdown ----
    lines = [
        "# Paper-Upgrade Config Fairness Check",
        "",
        f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')} by "
        f"`scripts/paper_check/check_paper_upgrade_configs.py`",
        "",
        "## 1. Train-level keys (shared, from `configs/train_baseline.yaml`)",
        "",
        "| Key | Value |",
        "|-----|-------|",
    ]
    for k in TRAIN_KEYS:
        lines.append(f"| {k} | `{train_cfg.get('train', {}).get(k)}` |")
    for k in DATA_KEYS:
        lines.append(f"| {k} | `{train_cfg.get('data', {}).get(k)}` |")
    lines += [
        f"| seed (default) | `{train_seed}` |",
        "",
        "These are identical across all models **by construction** (the same "
        "`train_baseline.yaml` is merged for every model).",
        "",
        "## 2. Model-level comparison",
        "",
        "| Model | source | hidden_dim | num_layers | heads | dropout | activation | "
        "use_layer_norm | decoder | params | seed42 batch_size |",
        "|-------|--------|-----------:|-----------:|------:|--------:|-----------|"
        "----------------|---------|--------:|------------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['display']} | `{r['source']}` | {r['hidden_dim']} | {r['num_layers']} | "
            f"{r['heads']} | {r['dropout']} | {r['activation']} | {r['use_layer_norm']} | "
            f"{r['decoder_hidden_dims']} | {r['params']:,} | {r['batch_size'] or '-'} |"
        )

    lines += [
        "",
        "## 3. Confounder flags",
        "",
    ]
    flags = []
    for r in rows:
        if r["batch_size"] is not None and r["batch_size"] != REFERENCE_BATCH:
            flags.append(
                f"- **POTENTIAL CONFOUNDER**: `{r['model_key']}` uses "
                f"batch_size={r['batch_size']} vs reference batch_size={REFERENCE_BATCH} "
                f"(HGT-3/MS-HGT). Reason: HGT-6/HGT-9 memory (2x/3x HGTConv "
                f"attention). MUST be reported in the paper or mitigated "
                f"(e.g., verify the result at batch_size=8 on one seed if memory permits)."
            )
    if not flags:
        flags.append("- No batch_size mismatch vs the reference (batch_size=8).")
    lines += flags

    lines += [
        "",
        "## 4. Notes / interpretation",
        "",
        "- HGT-6/HGT-9 intentionally use `num_layers` 6/9 (the depth variable); "
        "all other model hyper-params are matched to HGT-3.",
        "- MS-HGT random-anchor matches MS-HGT gated exactly except "
        "`anchor_mode/anchor_seed` (parameters identical: 893,527).",
        "- Edge-aware MPNN uses shared-across-layers message/update MLPs, so "
        "`num_layers=6` adds compute, not parameters (~450k total, **below** "
        "MS-HGT's 893k). Lower params must be reported honestly.",
        "- Parameter counts are measured on CPU build; server training may report "
        "the same number (verify against `model_summary.json`).",
        "- Training time is not estimable locally (no full training run); record "
        "wall-clock from `train_log.csv` on the server and add here when available.",
        "",
    ]
    OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  [ok] Fairness doc: {OUTPUT_DOC}")


if __name__ == "__main__":
    main()

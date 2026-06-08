"""
experiment.py — Experiment directory management, CSV logging, and visualisation.

Responsibilities:
  - Create timestamped output directories per (model_name, run).
  - Save resolved config as YAML.
  - CSVLogger for per-epoch metrics.
  - Loss / metric curve plotting (matplotlib, no Chinese fonts needed).
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt


# ============================================================
# Experiment directory
# ============================================================

def create_experiment_dir(
    output_root: Union[str, Path],
    model_name: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Create ``output_root / ModelName / YYYYMMDDHHMMSS / `` and return the path.

    Args:
        output_root: Base output directory (e.g. ``outputs``).
        model_name: Model name (e.g. ``MLP``).
        timestamp: Optional ISO-like string. Generated from now if None.

    Returns:
        Absolute Path to the created experiment directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_dir = Path(output_root).resolve() / model_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def save_resolved_config(config: Dict, save_path: Path) -> None:
    """Save resolved config dict as a YAML file."""
    import yaml
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  [ok] Config saved: {save_path}")


# ============================================================
# CSV Logger
# ============================================================

class CSVLogger:
    """Per-epoch metrics logger that writes to a CSV file.

    Usage::
        logger = CSVLogger(exp_dir / "train_log.csv", fieldnames=["epoch", "train_loss", "val_loss"])
        logger.append({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})
        logger.close()
    """

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file = open(self.path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def append(self, row: Dict) -> None:
        """Write one row (dict of fieldname → value)."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()


# ============================================================
# Plotting helpers
# ============================================================

def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    train_label: str = "Train Loss",
    val_label: str = "Val Loss",
) -> None:
    """Plot and save train / val loss curve."""
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label=train_label, alpha=0.8)
    plt.plot(epochs, val_losses[: len(epochs)], label=val_label, alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [ok] Loss curve saved: {save_path}")


def plot_metric_curve(
    values: List[float],
    save_path: Path,
    ylabel: str = "Metric",
    label: str = "Val",
) -> None:
    """Plot and save a single metric curve (typically validation metric)."""
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(values) + 1))
    plt.plot(epochs, values, label=label, alpha=0.8, marker=".", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [ok] Metric curve saved: {save_path}")

"""
trainer.py — Unified Trainer for edge-level regression on graph datasets.

Supports:
  - Per-epoch train / validation loops
  - Final test evaluation
  - Configurable loss, optimiser, scheduler
  - Early stopping with best-model checkpoint
  - Metric computation in *original* (un-standardised) space
  - CSV logging and console progress reporting

Usage::
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        experiment_dir=exp_dir,
        target_mean=target_mean,
        target_std=target_std,
    )
    history = trainer.fit()
    test_metrics = trainer.test(test_loader)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainers.early_stopping import EarlyStopping
from src.utils.experiment import CSVLogger, plot_loss_curve, plot_metric_curve
from src.utils.metrics import compute_all_metrics

# ---------- loss registry ----------

LOSS_FACTORY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}

# ---------- optimiser registry ----------

OPTIM_FACTORY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class Trainer:
    """Unified trainer for graph edge-level regression.

    Parameters:
        model: PyTorch model (e.g. ``MLPEdgeRegressor``).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Flat dict with keys::

            - optimizer (str): adam | adamw | sgd
            - lr (float)
            - weight_decay (float)
            - scheduler (str | None): reduce_on_plateau | cosine | null
            - scheduler_patience (int)
            - scheduler_factor (float)
            - loss (str): mse | l1 | smooth_l1
            - epochs (int)
            - early_stopping (bool)
            - early_stopping_patience (int)
            - early_stopping_metric (str): val_loss | val_mae
            - early_stopping_mode (str): min | max
            - log_interval (int): batches between console logs

        device: torch device.
        experiment_dir: Path for saving checkpoints and logs.
        target_mean: (12,) tensor — mean for inverse transform to original scale.
        target_std: (12,) tensor — std for inverse transform to original scale.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        experiment_dir: Union[str, Path],
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.target_mean = target_mean
        self.target_std = target_std

        # ---- Loss ----
        loss_name = config.get("loss", "mse")
        loss_cls = LOSS_FACTORY.get(loss_name)
        if loss_cls is None:
            raise ValueError(f"Unknown loss '{loss_name}'. Options: {list(LOSS_FACTORY.keys())}")
        self.loss_fn = loss_cls()

        # ---- Optimiser ----
        opt_name = config.get("optimizer", "adamw")
        opt_cls = OPTIM_FACTORY.get(opt_name)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer '{opt_name}'. Options: {list(OPTIM_FACTORY.keys())}")
        self.optimizer = opt_cls(
            model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # ---- Scheduler ----
        sched_name = config.get("scheduler")
        if sched_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=config.get("scheduler_patience", 10),
                factor=config.get("scheduler_factor", 0.5),
            )
        elif sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.get("epochs", 100)
            )
        else:
            self.scheduler = None

        # ---- Early stopping ----
        if config.get("early_stopping", True):
            self.early_stopping = EarlyStopping(
                patience=config.get("early_stopping_patience", 30),
                mode=config.get("early_stopping_mode", "min"),
            )
        else:
            self.early_stopping = None

        # ---- Training state ----
        self.current_epoch: int = 0
        self.best_epoch: int = 0
        self.best_val_metric: float = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_r2": [],
            "lr": [],
            "epoch_time": [],
        }

    # ============================================================
    # Core loops
    # ============================================================

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
            loss = self.loss_fn(pred, batch.y_edge)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.config.get("log_interval", 50) == 0:
                print(
                    f"    Epoch {self.current_epoch} | Batch {batch_idx + 1}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        """Run validation. Returns dict with loss and metrics."""
        self.model.eval()
        all_preds: list = []
        all_targets: list = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
            loss = self.loss_fn(pred, batch.y_edge)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(batch.y_edge.cpu())

        # Concatenate over all batches
        pred_all = torch.cat(all_preds, dim=0)      # (total_val_edges, 12)
        target_all = torch.cat(all_targets, dim=0)  # (total_val_edges, 12)

        # Convert to original scale for interpretable metrics
        if self.target_std is not None and self.target_mean is not None:
            target_std = self.target_std.to(pred_all.device)
            target_mean = self.target_mean.to(pred_all.device)
            pred_orig = pred_all * target_std + target_mean
            target_orig = target_all * target_std + target_mean
        else:
            pred_orig, target_orig = pred_all, target_all

        metrics = compute_all_metrics(pred_orig, target_orig)

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_mae": metrics["macro_avg"]["mae"],
            "val_r2": metrics["macro_avg"]["r2"],
            "val_rel_mae": metrics["macro_avg"]["rel_mae"],
            "metrics_detail": metrics,
        }

    # ============================================================
    # Public API
    # ============================================================

    def fit(self) -> Dict:
        """Run the full training loop with early stopping and checkpointing.

        Returns:
            Summary dict with best epoch, stopped epoch, best metrics.
        """
        epochs = self.config.get("epochs", 100)
        print(f"\n{'='*60}")
        print(f"Training — {epochs} max epochs")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.config.get('optimizer')}, lr={self.config.get('lr')}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples:   {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        # CSV logger
        csv_logger = CSVLogger(
            self.experiment_dir / "train_log.csv",
            fieldnames=[
                "epoch", "train_loss", "val_loss", "val_mae",
                "val_rel_mae", "val_r2", "lr", "epoch_time",
            ],
        )

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()

            # Train
            train_loss = self._train_epoch()

            # Validate
            val_results = self._val_epoch()
            val_loss = val_results["val_loss"]
            val_mae = val_results["val_mae"]
            val_r2 = val_results["val_r2"]
            val_rel_mae = val_results["val_rel_mae"]

            # LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - t_epoch

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_r2"].append(val_r2)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # CSV row
            csv_logger.append({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_mae": f"{val_mae:.6f}",
                "val_rel_mae": f"{val_rel_mae:.6f}",
                "val_r2": f"{val_r2:.6f}",
                "lr": f"{current_lr:.8f}",
                "epoch_time": f"{epoch_time:.2f}",
            })

            # Console log
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Val R2: {val_r2:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping
            if self.early_stopping is not None:
                # Determine which metric to monitor
                es_metric = self.config.get("early_stopping_metric", "val_loss")
                monitor_value = val_loss if es_metric == "val_loss" else val_mae

                self.early_stopping(monitor_value, self.model, current_epoch=epoch)
                if self.early_stopping.should_stop:
                    print(
                        f"\n  [early stopping] No improvement for "
                        f"{self.early_stopping.patience} epochs. "
                        f"Best epoch: {self.early_stopping.best_epoch} "
                        f"(best {es_metric}: {self.early_stopping.best_value:.6f})"
                    )
                    break

        # End of training
        total_time = time.time() - t_start
        csv_logger.close()

        # Determine best epoch and value
        if self.early_stopping is not None:
            best_epoch = self.early_stopping.best_epoch
            best_val_metric = self.early_stopping.best_value
            early_stopped = self.early_stopping.should_stop
            # Restore best weights
            self.early_stopping.load_best(self.model)
        else:
            # Without early stopping, pick best by validation loss
            best_idx = int(torch.tensor(self.history["val_loss"]).argmin())
            best_epoch = best_idx + 1
            best_val_metric = self.history["val_loss"][best_idx]
            early_stopped = False

        # Save best model
        best_model_path = self.experiment_dir / "best_model.pt"
        torch.save(self.model.state_dict(), best_model_path)
        print(f"\n  [ok] Best model (epoch {best_epoch}) saved: {best_model_path}")

        # Save last model
        last_model_path = self.experiment_dir / "last_model.pt"
        torch.save(self.model.state_dict(), last_model_path)
        print(f"  [ok] Last model saved: {last_model_path}")

        # Plot curves
        plot_loss_curve(
            self.history["train_loss"],
            self.history["val_loss"],
            self.experiment_dir / "loss_curve.png",
        )
        plot_metric_curve(
            self.history["val_mae"],
            self.experiment_dir / "metric_curve.png",
            ylabel="Val MAE",
        )

        print(f"\n  Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Best epoch: {best_epoch} | Best val metric: {best_val_metric:.6f}")

        summary = {
            "best_epoch": best_epoch,
            "stopped_epoch": self.current_epoch,
            "early_stopped": early_stopped,
            "best_val_metric": best_val_metric,
            "total_time_seconds": total_time,
        }
        return summary

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict:
        """Evaluate on the test set (one pass, no grad).

        Args:
            test_loader: DataLoader for the test split.

        Returns:
            Dict with test metrics (loss, macro_avg, per_component, overall).
        """
        self.model.eval()
        all_preds: list = []
        all_targets: list = []
        total_loss = 0.0
        num_batches = 0

        for batch in test_loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
            loss = self.loss_fn(pred, batch.y_edge)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(batch.y_edge.cpu())

        pred_all = torch.cat(all_preds, dim=0)
        target_all = torch.cat(all_targets, dim=0)

        # Convert to original scale for metrics
        if self.target_std is not None and self.target_mean is not None:
            target_std = self.target_std.to(pred_all.device)
            target_mean = self.target_mean.to(pred_all.device)
            pred_orig = pred_all * target_std + target_mean
            target_orig = target_all * target_std + target_mean
        else:
            pred_orig, target_orig = pred_all, target_all

        metrics = compute_all_metrics(pred_orig, target_orig)

        test_loss = total_loss / max(num_batches, 1)
        print(f"\n{'='*60}")
        print("Test Evaluation")
        print(f"{'='*60}")
        print(f"  Test Loss (standardised): {test_loss:.6f}")
        print(f"  Test MAE  (original, macro avg): {metrics['macro_avg']['mae']:.4f}")
        print(f"  Test R2   (original, macro avg): {metrics['macro_avg']['r2']:.4f}")
        print(f"  Test RelMAE (original, macro avg): {metrics['macro_avg']['rel_mae']:.4f}")
        print(f"\n  Per-component MAE:")
        comp_names = [
            "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
            "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
        ]
        per_mae = metrics["per_component"]["mae"]
        per_r2 = metrics["per_component"]["r2"]
        for i, name in enumerate(comp_names):
            print(f"    {name:6s}: MAE={per_mae[i]:10.4f}  R2={per_r2[i]:.4f}")
        print(f"{'='*60}\n")

        result = {
            "test_loss": test_loss,
            "test_metrics": metrics,
        }
        return result

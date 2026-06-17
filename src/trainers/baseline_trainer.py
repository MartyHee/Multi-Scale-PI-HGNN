"""
baseline_trainer.py — Unified trainer for Stage 2-A baseline models.

Supports:
  - Dual-output models (displacement + force) on ``HeteroData`` / ``HeteroDataBatch``
  - Combined displacement + force loss with configurable weights
  - Per-epoch train / validation / test loops
  - Inverse-transform to physical-space metrics
  - Early stopping with best-model checkpoint
  - CSV logging and console progress

Usage::

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=exp_dir,
        disp_mean=disp_mean,
        disp_std=disp_std,
        force_mean=force_mean,
        force_std=force_std,
    )
    history = trainer.fit()
    test_metrics = trainer.test(test_loader)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainers.early_stopping import EarlyStopping
from src.trainers.losses import CombinedLoss
from src.utils.experiment import CSVLogger, plot_loss_curve, plot_metric_curve
from src.utils.metrics import compute_all_metrics

# ---------- optimiser registry ----------

OPTIM_FACTORY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class BaselineTrainer:
    """Unified trainer for dual-output heterogeneous-graph baseline models.

    Parameters:
        model: PyTorch model whose ``forward(batch)`` returns
            ``(pred_disp, pred_force)``.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Config dict (see ``configs/train_baseline.yaml``).
        device: torch device.
        experiment_dir: Path for saving checkpoints and logs.
        disp_mean: (6,) tensor — mean for disp inverse transform.
        disp_std: (6,) tensor — std for disp inverse transform.
        force_mean: (12,) tensor — mean for force inverse transform.
        force_std: (12,) tensor — std for force inverse transform.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        experiment_dir: Union[str, Path],
        disp_mean: Optional[torch.Tensor] = None,
        disp_std: Optional[torch.Tensor] = None,
        force_mean: Optional[torch.Tensor] = None,
        force_std: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.disp_mean = disp_mean
        self.disp_std = disp_std
        self.force_mean = force_mean
        self.force_std = force_std

        # ---- Combined loss ----
        self.loss_fn = CombinedLoss(
            loss_name=config.get("loss_name", "mse"),
            lambda_disp=config.get("lambda_disp", 1.0),
            lambda_force=config.get("lambda_force", 1.0),
        )

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
        es_patience = config.get("early_stopping_patience", 30)
        if es_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=es_patience,
                mode="min",
            )
        else:
            self.early_stopping = None

        # ---- Training state ----
        self.current_epoch: int = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_disp_mae": [],
            "val_force_mae": [],
            "val_disp_r2": [],
            "val_force_r2": [],
            "lr": [],
            "epoch_time": [],
        }

    # ============================================================
    # Core loops
    # ============================================================

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            ``(mean_total_loss, mean_disp_loss, mean_force_loss)``.
        """
        self.model.train()
        total_loss = 0.0
        total_disp_loss = 0.0
        total_force_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            pred_disp, pred_force = self.model(batch)

            loss_total, loss_disp, loss_force = self.loss_fn(
                pred_disp, pred_force,
                batch["mesh_node"].y_disp, batch["beam_element"].y_force,
            )
            loss_total.backward()
            self.optimizer.step()

            total_loss += loss_total.item()
            total_disp_loss += loss_disp.item()
            total_force_loss += loss_force.item()
            num_batches += 1

            if (batch_idx + 1) % max(1, self.config.get("log_interval", 500)) == 0:
                print(
                    f"    Epoch {self.current_epoch} | Batch {batch_idx + 1}/{len(self.train_loader)} "
                    f"| Loss: {loss_total.item():.4f} (D:{loss_disp.item():.4f} F:{loss_force.item():.4f})"
                )

        return (
            total_loss / max(num_batches, 1),
            total_disp_loss / max(num_batches, 1),
            total_force_loss / max(num_batches, 1),
        )

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        """Run validation.

        Returns:
            Dict with loss and metrics (both standardised and original-scale).
        """
        self.model.eval()
        all_disp_pred: List[torch.Tensor] = []
        all_disp_target: List[torch.Tensor] = []
        all_force_pred: List[torch.Tensor] = []
        all_force_target: List[torch.Tensor] = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            pred_disp, pred_force = self.model(batch)

            loss_total, _, _ = self.loss_fn(
                pred_disp, pred_force,
                batch["mesh_node"].y_disp, batch["beam_element"].y_force,
            )
            total_loss += loss_total.item()
            num_batches += 1

            all_disp_pred.append(pred_disp.cpu())
            all_disp_target.append(batch["mesh_node"].y_disp.cpu())
            all_force_pred.append(pred_force.cpu())
            all_force_target.append(batch["beam_element"].y_force.cpu())

        # Concatenate over all batches
        disp_pred = torch.cat(all_disp_pred, dim=0)
        disp_target = torch.cat(all_disp_target, dim=0)
        force_pred = torch.cat(all_force_pred, dim=0)
        force_target = torch.cat(all_force_target, dim=0)

        # Inverse transform to original (physical) scale
        disp_pred_orig, disp_target_orig = self._inverse_transform_disp(disp_pred, disp_target)
        force_pred_orig, force_target_orig = self._inverse_transform_force(force_pred, force_target)

        # Compute metrics in original space
        disp_metrics = compute_all_metrics(disp_pred_orig, disp_target_orig)
        force_metrics = compute_all_metrics(force_pred_orig, force_target_orig)

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_disp_mae": disp_metrics["macro_avg"]["mae"],
            "val_disp_r2": disp_metrics["macro_avg"]["r2"],
            "val_force_mae": force_metrics["macro_avg"]["mae"],
            "val_force_r2": force_metrics["macro_avg"]["r2"],
            "disp_metrics": disp_metrics,
            "force_metrics": force_metrics,
        }

    # ============================================================
    # Inverse transform helpers
    # ============================================================

    def _inverse_transform_disp(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.disp_std is not None and self.disp_mean is not None:
            std = self.disp_std.to(pred.device)
            mean = self.disp_mean.to(pred.device)
            return pred * std + mean, target * std + mean
        return pred, target

    def _inverse_transform_force(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.force_std is not None and self.force_mean is not None:
            std = self.force_std.to(pred.device)
            mean = self.force_mean.to(pred.device)
            return pred * std + mean, target * std + mean
        return pred, target

    # ============================================================
    # Public API
    # ============================================================

    def fit(self) -> Dict:
        """Run the full training loop with early stopping and checkpointing.

        Returns:
            Summary dict with best epoch, best metrics, total time.
        """
        epochs = self.config.get("epochs", 100)
        print(f"\n{'=' * 60}")
        print(f"Training — {epochs} max epochs")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.config.get('optimizer')}, lr={self.config.get('lr')}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples:   {len(self.val_loader.dataset)}")
        print(f"Batch size:    {self.train_loader.batch_size}")
        print(f"{'=' * 60}\n")

        # CSV logger
        csv_logger = CSVLogger(
            self.experiment_dir / "train_log.csv",
            fieldnames=[
                "epoch", "train_loss", "val_loss",
                "val_disp_mae", "val_disp_r2",
                "val_force_mae", "val_force_r2",
                "lr", "epoch_time",
            ],
        )

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()

            # Train
            train_loss, train_disp_loss, train_force_loss = self._train_epoch()

            # Validate
            val_results = self._val_epoch()
            val_loss = val_results["val_loss"]

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
            self.history["val_disp_mae"].append(val_results["val_disp_mae"])
            self.history["val_force_mae"].append(val_results["val_force_mae"])
            self.history["val_disp_r2"].append(val_results["val_disp_r2"])
            self.history["val_force_r2"].append(val_results["val_force_r2"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # CSV row
            csv_logger.append({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_disp_mae": f"{val_results['val_disp_mae']:.6f}",
                "val_disp_r2": f"{val_results['val_disp_r2']:.6f}",
                "val_force_mae": f"{val_results['val_force_mae']:.6f}",
                "val_force_r2": f"{val_results['val_force_r2']:.6f}",
                "lr": f"{current_lr:.8f}",
                "epoch_time": f"{epoch_time:.2f}",
            })

            # Console log
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"D-MAE: {val_results['val_disp_mae']:.4f} | "
                f"F-MAE: {val_results['val_force_mae']:.4f} | "
                f"D-R2: {val_results['val_disp_r2']:.4f} | "
                f"F-R2: {val_results['val_force_r2']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping (monitor val_loss by default)
            if self.early_stopping is not None:
                es_metric = self.config.get("early_stopping_metric", "val_loss")
                monitor_value = val_loss if es_metric == "val_loss" else val_results.get(es_metric, val_loss)

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

        # Determine best epoch and restore weights
        if self.early_stopping is not None and self.early_stopping.best_state_dict is not None:
            best_epoch = self.early_stopping.best_epoch
            best_val_metric = self.early_stopping.best_value
            early_stopped = self.early_stopping.should_stop
            self.early_stopping.load_best(self.model)
        else:
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
            self.history["val_disp_mae"],
            self.experiment_dir / "metric_curve_disp.png",
            ylabel="Val Disp MAE",
        )
        plot_metric_curve(
            self.history["val_force_mae"],
            self.experiment_dir / "metric_curve_force.png",
            ylabel="Val Force MAE",
        )

        print(f"\n  Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Best epoch: {best_epoch} | Best val metric: {best_val_metric:.6f}")

        summary = {
            "best_epoch": best_epoch,
            "stopped_epoch": self.current_epoch,
            "early_stopped": early_stopped,
            "best_val_metric": best_val_metric,
            "total_time_seconds": total_time,
            "best_val_disp_mae": self.history["val_disp_mae"][best_epoch - 1] if best_epoch - 1 < len(self.history["val_disp_mae"]) else None,
            "best_val_force_mae": self.history["val_force_mae"][best_epoch - 1] if best_epoch - 1 < len(self.history["val_force_mae"]) else None,
            "best_val_disp_r2": self.history["val_disp_r2"][best_epoch - 1] if best_epoch - 1 < len(self.history["val_disp_r2"]) else None,
            "best_val_force_r2": self.history["val_force_r2"][best_epoch - 1] if best_epoch - 1 < len(self.history["val_force_r2"]) else None,
        }
        return summary

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict:
        """Evaluate on the test set.

        Args:
            test_loader: DataLoader for the test split.

        Returns:
            Dict with ``"test_loss"``, ``"disp_metrics"``, ``"force_metrics"``.
        """
        self.model.eval()
        all_disp_pred: List[torch.Tensor] = []
        all_disp_target: List[torch.Tensor] = []
        all_force_pred: List[torch.Tensor] = []
        all_force_target: List[torch.Tensor] = []
        total_loss = 0.0
        num_batches = 0

        for batch in test_loader:
            batch = batch.to(self.device)
            pred_disp, pred_force = self.model(batch)

            loss_total, _, _ = self.loss_fn(
                pred_disp, pred_force,
                batch["mesh_node"].y_disp, batch["beam_element"].y_force,
            )
            total_loss += loss_total.item()
            num_batches += 1

            all_disp_pred.append(pred_disp.cpu())
            all_disp_target.append(batch["mesh_node"].y_disp.cpu())
            all_force_pred.append(pred_force.cpu())
            all_force_target.append(batch["beam_element"].y_force.cpu())

        disp_pred = torch.cat(all_disp_pred, dim=0)
        disp_target = torch.cat(all_disp_target, dim=0)
        force_pred = torch.cat(all_force_pred, dim=0)
        force_target = torch.cat(all_force_target, dim=0)

        disp_pred_orig, disp_target_orig = self._inverse_transform_disp(disp_pred, disp_target)
        force_pred_orig, force_target_orig = self._inverse_transform_force(force_pred, force_target)

        disp_metrics = compute_all_metrics(disp_pred_orig, disp_target_orig)
        force_metrics = compute_all_metrics(force_pred_orig, force_target_orig)

        test_loss = total_loss / max(num_batches, 1)

        # ---- Print results ----
        print(f"\n{'=' * 60}")
        print("Test Evaluation")
        print(f"{'=' * 60}")
        print(f"  Test Loss (standardised): {test_loss:.6f}")
        print(f"\n  --- Displacement ---")
        print(f"    MAE (macro avg): {disp_metrics['macro_avg']['mae']:.6f}")
        print(f"    R2  (macro avg): {disp_metrics['macro_avg']['r2']:.6f}")
        print(f"    RelMAE (overall): {disp_metrics['overall']['rel_mae']:.6f}")
        disp_comp_names = ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"]
        for i, name in enumerate(disp_comp_names):
            print(f"    {name:4s}: MAE={disp_metrics['per_component']['mae'][i]:10.6f}  R2={disp_metrics['per_component']['r2'][i]:.4f}")

        print(f"\n  --- Beam Force ---")
        print(f"    MAE (macro avg): {force_metrics['macro_avg']['mae']:.6f}")
        print(f"    R2  (macro avg): {force_metrics['macro_avg']['r2']:.6f}")
        print(f"    RelMAE (overall): {force_metrics['overall']['rel_mae']:.6f}")
        force_comp_names = [
            "Fx_I", "Fy_I", "Fz_I", "Mx_I", "My_I", "Mz_I",
            "Fx_J", "Fy_J", "Fz_J", "Mx_J", "My_J", "Mz_J",
        ]
        for i, name in enumerate(force_comp_names):
            print(f"    {name:6s}: MAE={force_metrics['per_component']['mae'][i]:10.6f}  R2={force_metrics['per_component']['r2'][i]:.4f}")
        print(f"{'=' * 60}\n")

        return {
            "test_loss": test_loss,
            "disp_metrics": disp_metrics,
            "force_metrics": force_metrics,
        }

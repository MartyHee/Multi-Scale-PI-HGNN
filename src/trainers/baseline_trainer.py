"""
baseline_trainer.py — Unified trainer for Stage 2-A baseline models.

Supports:
  - Dual-output models (displacement + force) on ``HeteroData`` / ``HeteroDataBatch``
  - Combined displacement + force loss with configurable weights
  - Per-epoch train / validation / test loops
  - Inverse-transform to physical-space metrics
  - Early stopping with best-model checkpoint (saved immediately on improvement)
  - Atomic checkpoint writes for crash safety
  - CSV logging with best-epoch tracking and tqdm progress bars
  - Configurable logging interval and progress display

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

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.physics_loss import compute_physics_losses
from src.trainers.early_stopping import EarlyStopping
from src.trainers.losses import CombinedLoss
from src.utils.experiment import CSVLogger, plot_loss_curve, plot_metric_curve
from src.utils.metrics import compute_all_metrics

# ---------- optional tqdm ----------
_HAVE_TQDM = False
try:
    from tqdm.auto import tqdm as _tqdm

    _HAVE_TQDM = True
except ImportError:
    pass

# ---------- optimiser registry ----------

OPTIM_FACTORY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def _atomic_save(state: dict, path: Path) -> None:
    """Write ``state`` via temp file then atomic replace (safe against crash)."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    # Use os.replace for cross-platform atomic overwrite (rename on
    # POSIX, MoveFileEx on Windows).
    os.replace(str(tmp), str(path))


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
        scaler: ``HeteroFeatureScaler`` (optional) — used by physics loss
            for support-flag restoration via ``_stats``.
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
        scaler=None,
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
        self.scaler = scaler

        # ---- Physics losses (Stage 5) ----
        self.lambda_bc = config.get("lambda_bc", 0.0)
        self.lambda_link = config.get("lambda_link", 0.0)
        self.physics_enabled = (self.lambda_bc > 0 or self.lambda_link > 0)
        self._physics_dofs = config.get("physics_dofs", "translation")

        # ---- Progress / logging config ----
        prog_cfg = config.get("progress", {})
        self.use_tqdm = prog_cfg.get("use_tqdm", False) and _HAVE_TQDM
        self.log_interval = prog_cfg.get("log_interval", 20)
        self.leave_epoch_bar = prog_cfg.get("leave_epoch_bar", False)

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
        # Track best inside fit() for immediate checkpoint saving
        self._best_val_loss: float = float("inf")
        self._best_epoch: int = 0

    # ============================================================
    # Core loops
    # ============================================================

    def _train_epoch(self) -> Dict:
        """Run one training epoch.

        Returns:
            Dict with loss values and physics-loss diagnostics:

            - ``"loss"`` — total loss (supervised + physics)
            - ``"disp_loss"`` — displacement MSE
            - ``"force_loss"`` — force MSE
            - ``"bc_loss"`` — BC translation loss (unweighted)
            - ``"link_loss"`` — link consistency loss (unweighted)
            - ``"n_bc_dofs"`` — number of constrained BC DOFs
            - ``"n_link_edges"`` — number of structural link edges used
        """
        self.model.train()
        total_loss = 0.0
        total_disp_loss = 0.0
        total_force_loss = 0.0
        total_bc_loss = 0.0
        total_link_loss = 0.0
        total_n_bc_dofs = 0
        total_n_link_edges = 0
        num_batches = 0

        iterator = self.train_loader
        if self.use_tqdm:
            iterator = _tqdm(
                self.train_loader,
                desc=f"  Epoch {self.current_epoch}",
                leave=self.leave_epoch_bar,
                unit="batch",
                ncols=100,
            )

        for batch_idx, batch in enumerate(iterator):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            pred_disp, pred_force = self.model(batch)

            # Supervised loss (always)
            loss_sup, loss_disp, loss_force = self.loss_fn(
                pred_disp, pred_force,
                batch["mesh_node"].y_disp, batch["beam_element"].y_force,
            )

            # Physics losses (optional)
            bc_loss_val = 0.0
            link_loss_val = 0.0
            bc_dofs = 0
            link_edges = 0

            if self.physics_enabled:
                phy_total, phy_bc, phy_link, bc_dofs, link_edges = compute_physics_losses(
                    pred_disp, batch["mesh_node"].y_disp, batch,
                    lambda_bc=self.lambda_bc, lambda_link=self.lambda_link,
                    scaler=self.scaler,
                )
                bc_loss_val = phy_bc.item()
                link_loss_val = phy_link.item()
                loss_total = loss_sup + phy_total
            else:
                loss_total = loss_sup

            loss_total.backward()
            self.optimizer.step()

            total_loss += loss_total.item()
            total_disp_loss += loss_disp.item()
            total_force_loss += loss_force.item()
            total_bc_loss += bc_loss_val
            total_link_loss += link_loss_val
            total_n_bc_dofs += bc_dofs
            total_n_link_edges += link_edges
            num_batches += 1

            # Update tqdm postfix or log periodically
            if self.use_tqdm and hasattr(iterator, "set_postfix"):
                postfix = {
                    "loss": f"{loss_total.item():.4f}",
                    "D": f"{loss_disp.item():.4f}",
                    "F": f"{loss_force.item():.4f}",
                }
                if self.physics_enabled:
                    postfix["BC"] = f"{bc_loss_val:.4f}"
                    postfix["LK"] = f"{link_loss_val:.4f}"
                iterator.set_postfix(postfix)
            elif not self.use_tqdm and (batch_idx + 1) % max(1, self.log_interval) == 0:
                msg = (
                    f"    Epoch {self.current_epoch} | Batch {batch_idx + 1}/{len(self.train_loader)} "
                    f"| Loss: {loss_total.item():.4f} (D:{loss_disp.item():.4f} F:{loss_force.item():.4f})"
                )
                if self.physics_enabled:
                    msg += f" BC:{bc_loss_val:.4f} LK:{link_loss_val:.4f}"
                print(msg)

        nb = max(num_batches, 1)
        return {
            "loss": total_loss / nb,
            "disp_loss": total_disp_loss / nb,
            "force_loss": total_force_loss / nb,
            "bc_loss": total_bc_loss / nb,
            "link_loss": total_link_loss / nb,
            "n_bc_dofs": total_n_bc_dofs,
            "n_link_edges": total_n_link_edges,
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        """Run validation.

        Returns:
            Dict with loss and metrics (both standardised and original-scale),
            plus physics-loss diagnostics when enabled.
        """
        self.model.eval()
        all_disp_pred: List[torch.Tensor] = []
        all_disp_target: List[torch.Tensor] = []
        all_force_pred: List[torch.Tensor] = []
        all_force_target: List[torch.Tensor] = []
        total_loss = 0.0
        total_bc_loss = 0.0
        total_link_loss = 0.0
        total_n_bc_dofs = 0
        total_n_link_edges = 0
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

            # Physics losses (logging only — NOT for validation metric)
            if self.physics_enabled:
                _, phy_bc, phy_link, bc_dofs, link_edges = compute_physics_losses(
                    pred_disp, batch["mesh_node"].y_disp, batch,
                    lambda_bc=self.lambda_bc, lambda_link=self.lambda_link,
                    scaler=self.scaler,
                )
                total_bc_loss += phy_bc.item()
                total_link_loss += phy_link.item()
                total_n_bc_dofs += bc_dofs
                total_n_link_edges += link_edges

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

        nb = max(num_batches, 1)
        return {
            "val_loss": total_loss / nb,
            "val_loss_bc": total_bc_loss / nb,
            "val_loss_link": total_link_loss / nb,
            "val_n_bc_dofs": total_n_bc_dofs,
            "val_n_link_edges": total_n_link_edges,
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

        Best-model checkpoint is saved **immediately** when val loss improves
        (atomic write via temp-file + rename).  Full last-checkpoint (model +
        optimizer + config) is saved each epoch for resume support.

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
        print(f"Progress bar:  {'tqdm' if self.use_tqdm else 'console log'}")
        if self.physics_enabled:
            print(f"Physics loss:  BC λ={self.lambda_bc} | Link λ={self.lambda_link}")
        print(f"{'=' * 60}\n")

        # CSV logger with best-epoch tracking columns
        csv_fieldnames = [
            "epoch", "train_loss", "val_loss",
            "val_disp_mae", "val_disp_r2",
            "val_force_mae", "val_force_r2",
            "lr", "epoch_time_sec",
            "best_epoch", "best_val_loss",
        ]
        if self.physics_enabled:
            csv_fieldnames += [
                "train_loss_bc", "train_loss_link",
                "val_loss_bc", "val_loss_link",
                "n_bc_dofs", "n_link_edges",
                "lambda_bc", "lambda_link",
            ]
        csv_logger = CSVLogger(
            self.experiment_dir / "train_log.csv",
            fieldnames=csv_fieldnames,
        )

        # Build last-checkpoint skeleton (filled each epoch)
        _ckpt_base = {
            "model_name": self.config.get("model_name", "unknown"),
            "split_mode": self.config.get("data", {}).get("split_mode", "by_sample"),
            "config": self.config,
        }

        t_start = time.time()
        early_stopped = False

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()

            # ---- Train ----
            train_result = self._train_epoch()
            train_loss = train_result["loss"]
            train_disp_loss = train_result["disp_loss"]
            train_force_loss = train_result["force_loss"]
            train_bc_loss = train_result["bc_loss"]
            train_link_loss = train_result["link_loss"]
            train_n_bc_dofs = train_result["n_bc_dofs"]
            train_n_link_edges = train_result["n_link_edges"]

            # ---- Validate ----
            val_results = self._val_epoch()
            val_loss = val_results["val_loss"]
            val_bc_loss = val_results.get("val_loss_bc", 0.0)
            val_link_loss = val_results.get("val_loss_link", 0.0)
            val_n_bc_dofs = val_results.get("val_n_bc_dofs", 0)
            val_n_link_edges = val_results.get("val_n_link_edges", 0)

            # ---- LR ----
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- Scheduler step ----
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - t_epoch

            # ---- Record history ----
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_disp_mae"].append(val_results["val_disp_mae"])
            self.history["val_force_mae"].append(val_results["val_force_mae"])
            self.history["val_disp_r2"].append(val_results["val_disp_r2"])
            self.history["val_force_r2"].append(val_results["val_force_r2"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # ---- Save best_model.pt immediately on improvement ----
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
                self._best_epoch = epoch
                best_path = self.experiment_dir / "best_model.pt"
                _atomic_save(self.model.state_dict(), best_path)

            # ---- Save last_checkpoint.pt each epoch ----
            last_ckpt = {
                **_ckpt_base,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self._best_val_loss,
                "best_epoch": self._best_epoch,
            }
            _atomic_save(last_ckpt, self.experiment_dir / "last_checkpoint.pt")
            # Also keep a lighter last_model.pt (state_dict only)
            _atomic_save(self.model.state_dict(), self.experiment_dir / "last_model.pt")

            # ---- CSV row ----
            csv_row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_disp_mae": f"{val_results['val_disp_mae']:.6f}",
                "val_disp_r2": f"{val_results['val_disp_r2']:.6f}",
                "val_force_mae": f"{val_results['val_force_mae']:.6f}",
                "val_force_r2": f"{val_results['val_force_r2']:.6f}",
                "lr": f"{current_lr:.8f}",
                "epoch_time_sec": f"{epoch_time:.2f}",
                "best_epoch": self._best_epoch,
                "best_val_loss": f"{self._best_val_loss:.6f}",
            }
            if self.physics_enabled:
                csv_row.update({
                    "train_loss_bc": f"{train_bc_loss:.6f}",
                    "train_loss_link": f"{train_link_loss:.6f}",
                    "val_loss_bc": f"{val_bc_loss:.6f}",
                    "val_loss_link": f"{val_link_loss:.6f}",
                    "n_bc_dofs": train_n_bc_dofs + val_n_bc_dofs,
                    "n_link_edges": train_n_link_edges + val_n_link_edges,
                    "lambda_bc": f"{self.lambda_bc:.6f}",
                    "lambda_link": f"{self.lambda_link:.6f}",
                })
            csv_logger.append(csv_row)

            # ---- Console log ----
            elapsed = time.time() - t_start
            avg_epoch_time = elapsed / epoch
            remaining = avg_epoch_time * (epochs - epoch)
            best_marker = " *BEST" if is_best else ""
            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"D-MAE: {val_results['val_disp_mae']:.4f} | "
                f"F-MAE: {val_results['val_force_mae']:.4f} | "
                f"D-R2: {val_results['val_disp_r2']:.4f} | "
                f"F-R2: {val_results['val_force_r2']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s | "
                f"ETA:{remaining:.0f}s{best_marker}"
            )
            if self.physics_enabled:
                log_msg += (
                    f" | BC:{train_bc_loss:.4f}/{val_bc_loss:.4f}"
                    f" LK:{train_link_loss:.4f}/{val_link_loss:.4f}"
                )
            print(log_msg)

            # ---- Early stopping ----
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
                    early_stopped = True
                    break

        # ---- End of training loop ----
        total_time = time.time() - t_start
        csv_logger.close()

        # Restore best weights
        if self.early_stopping is not None and self.early_stopping.best_state_dict is not None:
            best_epoch = self.early_stopping.best_epoch
            best_val_metric = self.early_stopping.best_value
            self.early_stopping.load_best(self.model)
        else:
            # Fallback: use tracked best
            best_epoch = self._best_epoch
            best_val_metric = self._best_val_loss
            # Load from saved best_model.pt if exists
            best_path = self.experiment_dir / "best_model.pt"
            if best_path.exists():
                try:
                    self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
                except Exception:
                    pass

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
        print(f"  Best epoch: {best_epoch} | Best val loss: {best_val_metric:.6f}")

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

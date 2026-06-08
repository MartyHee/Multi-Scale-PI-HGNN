"""
early_stopping.py — Early stopping with best checkpoint tracking.

Tracks a primary metric (e.g. validation loss) and signals when to stop
training after ``patience`` epochs without improvement.

Usage::
    early_stop = EarlyStopping(patience=30, mode="min")
    for epoch in range(epochs):
        val_loss = ...
        early_stop(val_loss, model)
        if early_stop.should_stop:
            break
    print(f"Best epoch: {early_stop.best_epoch}, best value: {early_stop.best_value}")
    early_stop.load_best(model)  # restore best weights
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Monitor a validation metric and stop when it stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        mode: ``"min"`` (lower is better, e.g. loss) or ``"max"``
            (higher is better, e.g. R², accuracy).
        min_delta: Minimum change in the monitored value to qualify as
            improvement.
    """

    def __init__(
        self,
        patience: int = 30,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.counter: int = 0
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.best_state_dict: Optional[dict] = None
        self.should_stop: bool = False

        if mode == "min":
            self._improved = lambda cur, best: cur < best - self.min_delta
            self._best_init = float("inf")
        elif mode == "max":
            self._improved = lambda cur, best: cur > best + self.min_delta
            self._best_init = float("-inf")
        else:
            raise ValueError(f"Expected mode='min' or 'max', got '{mode}'")

    def __call__(self, current_value: float, model: nn.Module,
                 current_epoch: int = 0) -> None:
        """Update state with the current epoch's metric value.

        Args:
            current_value: The monitored metric for this epoch.
            model: The model whose state_dict to snapshot if improved.
            current_epoch: The actual epoch number (1-based) for tracking.
        """
        if self.best_value is None:
            # First call — initialise
            self.best_value = current_value
            self.best_epoch = current_epoch or 1
            self.best_state_dict = deepcopy(model.state_dict())
            return

        if self._improved(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = current_epoch
            self.best_state_dict = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best(self, model: nn.Module) -> None:
        """Restore model parameters from the best epoch."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        else:
            print("  [warn] EarlyStopping: no best state to load.")

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_state_dict": self.best_state_dict,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.patience = state["patience"]
        self.mode = state["mode"]
        self.min_delta = state["min_delta"]
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.best_state_dict = state["best_state_dict"]
        self.should_stop = state["should_stop"]
        # restore _improved function
        if self.mode == "min":
            self._improved = lambda cur, best: cur < best - self.min_delta
        else:
            self._improved = lambda cur, best: cur > best + self.min_delta

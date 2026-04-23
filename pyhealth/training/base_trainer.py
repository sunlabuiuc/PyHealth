"""Base training loop for PyHealth models with support for learning rate schedules.

This module implements a general-purpose training loop that extends PyHealth's
Trainer class with support for custom learning rate schedules, early stopping,
subset-based training/validation, and detailed progress tracking.

The base trainer is designed to be extended for model-specific training procedures
(e.g., wav2sleep with warmup + exponential decay schedule).
"""

import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tqdm.autonotebook import trange

try:
    from pyhealth.utils import create_directory
except ImportError:
    def create_directory(path: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

logger = logging.getLogger(__name__)


def set_logger(log_path: str) -> None:
    """Configure logging to file."""
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaseTrainer:
    """General training loop with support for learning rate schedules and early stopping.

    This trainer extends the PyHealth training framework to support:
    - Custom learning rate schedules (warmup, decay, etc.)
    - Flexible batch sampling (subset-based training and validation)
    - Early stopping with configurable patience
    - Comprehensive logging and metrics tracking

    Args:
        model: PyTorch model to train.
        device: Device for training ('cuda' or 'cpu'). If None, auto-detect.
        enable_logging: Whether to save training logs. Default is True.
        output_path: Directory for saving checkpoints and logs. Default is './output'.
        exp_name: Experiment name. If None, uses current timestamp.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.device = device
        self.model.to(self.device)

        # Setup logging
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, object]] = None,
        scheduler_class: Optional[Type[LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, object]] = None,
        weight_decay: float = 0.01,
        max_grad_norm: Optional[float] = 1.0,
        patience: Optional[int] = None,
        eval_metric: str = "loss",
        eval_mode: str = "min",
    ) -> Dict[str, List[float]]:
        """Train the model with optional learning rate scheduling and early stopping.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).
            epochs: Number of training epochs.
            optimizer_class: Optimizer class (e.g., torch.optim.AdamW).
            optimizer_params: Keyword arguments for optimizer (e.g., {'lr': 1e-3}).
            scheduler_class: Optional learning rate scheduler class.
            scheduler_params: Keyword arguments for scheduler.
            weight_decay: L2 regularization coefficient.
            max_grad_norm: Maximum gradient norm for clipping. If None, no clipping.
            patience: Number of epochs with no improvement before early stopping.
                If None, no early stopping.
            eval_metric: Metric to monitor for early stopping ('loss', 'accuracy', etc.).
            eval_mode: 'min' to minimize metric, 'max' to maximize.

        Returns:
            Dictionary with lists of training and validation metrics.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # Initialize optimizer
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        # Initialize scheduler if provided
        scheduler = None
        if scheduler_class is not None:
            scheduler_params = scheduler_params or {}
            scheduler = scheduler_class(optimizer, **scheduler_params)

        # Tracking metrics
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_score = float("inf") if eval_mode == "min" else float("-inf")
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(
                self.model,
                train_dataloader,
                optimizer,
                max_grad_norm,
            )
            history["train_loss"].append(train_loss)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Validation phase
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["loss"])
                if "accuracy" in val_metrics:
                    history["val_accuracy"].append(val_metrics["accuracy"])

                # Early stopping check
                current_score = val_metrics[eval_metric]
                is_improvement = (
                    current_score < best_score if eval_mode == "min"
                    else current_score > best_score
                )

                if is_improvement:
                    best_score = current_score
                    patience_counter = 0
                    if self.exp_path is not None:
                        self.save_checkpoint(
                            os.path.join(self.exp_path, "best.pt")
                        )
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        break

            # Save checkpoint
            if self.exp_path is not None:
                self.save_checkpoint(os.path.join(self.exp_path, "last.pt"))

        # Load best model if available
        if self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.pt")
        ):
            self.load_checkpoint(os.path.join(self.exp_path, "best.pt"))

        return history

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        max_grad_norm: Optional[float] = None,
    ) -> float:
        """Train for one epoch.

        Args:
            model: Model to train.
            dataloader: Training data loader.
            optimizer: Optimizer instance.
            max_grad_norm: Optional gradient clipping threshold.

        Returns:
            Average training loss for the epoch.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            optimizer.zero_grad()

            # Forward pass
            output = model(**batch)
            loss = output["loss"]

            # Backward pass
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            dataloader: Data loader for evaluation.

        Returns:
            Dictionary of metric values.
        """
        self.model.eval()
        y_true_all = []
        y_prob_all = []
        loss_all = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                output = self.model(**batch)
                loss_all.append(output["loss"].item())
                y_true_all.append(output["y_true"].cpu().numpy())
                y_prob_all.append(output["y_prob"].cpu().numpy())

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)

        loss_mean = np.mean(loss_all)
        metrics = {"loss": loss_mean}

        # Compute accuracy if output probabilities are available
        if y_prob_all.ndim > 1:
            preds = np.argmax(y_prob_all, axis=-1)
            accuracy = np.mean(preds == y_true_all)
            metrics["accuracy"] = float(accuracy)

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

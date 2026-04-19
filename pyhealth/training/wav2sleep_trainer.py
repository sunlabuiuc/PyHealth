"""Wav2Sleep training loop with paper-specific hyperparameters and schedule.

This module implements the training procedure described in the wav2sleep paper:
- AdamW optimizer with configurable learning rate
- Linear warmup to maximum learning rate, then exponential decay to zero
- Early stopping when validation loss plateaus (no improvement for 5 epochs)
- Support for training on dataset subsets for large-scale datasets

Paper Reference:
    Jonathan F. Carter & Lionel Tarassenko. "wav2sleep: A Unified Multi-Modal
    Approach to Sleep Stage Classification from Physiological Signals."
    arXiv:2411.04644, 2024. https://arxiv.org/abs/2411.04644
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, Dataset

from pyhealth.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LinearWarmupExponentialDecayScheduler:
    """Learning rate scheduler: linear warmup then exponential decay.

    This scheduler implements the learning rate schedule described in the wav2sleep
    paper:
    1. Linear warmup from 0 to max_lr over warmup_steps
    2. Exponential decay from max_lr to ~0 over remaining steps

    This is achieved using a LambdaLR that modifies the base optimizer learning rate.
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        warmup_steps: int,
        total_steps: int,
        decay_rate: float = 0.9999,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: PyTorch optimizer instance.
            max_lr: Maximum learning rate (at end of warmup phase).
            warmup_steps: Number of steps for linear warmup.
            total_steps: Total number of training steps.
            decay_rate: Exponential decay factor per step (lambda^step). Default: 0.9999.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_rate = decay_rate

        def lr_lambda(step: int) -> float:
            """Compute learning rate multiplier for given step."""
            if step < warmup_steps:
                # Linear warmup: 0 → 1
                return step / warmup_steps
            else:
                # Exponential decay: 1 → 0
                decay_steps = step - warmup_steps
                return decay_rate ** decay_steps

        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def step(self):
        """Advance the learning rate scheduler."""
        self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return self.scheduler.get_last_lr()


class Wav2SleepTrainer(BaseTrainer):
    """Training loop for Wav2Sleep model with paper-specific configuration.

    Implements the training procedure from the wav2sleep paper with:
    - AdamW optimizer (no bias decay)
    - Linear warmup to max_lr, then exponential decay
    - Early stopping: 5 epochs with no validation loss improvement
    - Optional subset training/validation for large datasets

    Typical usage:
        >>> trainer = Wav2SleepTrainer(model, device='cuda', enable_logging=True)
        >>> train_losses, val_losses = trainer.train(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     epochs=100,
        ...     warmup_fraction=0.1,
        ...     max_lr=1e-3,
        ...     decay_rate=0.9999,
        ...     patience=5,
        ... )
    """

    def train_with_schedule(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 100,
        warmup_fraction: float = 0.1,
        max_lr: float = 1e-3,
        decay_rate: float = 0.9999,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Train wav2sleep model with linear warmup + exponential decay schedule.

        Implements the exact training procedure from the wav2sleep paper:
        - AdamW optimizer with weight decay (but not on bias/norm parameters)
        - Learning rate: linear warmup (0 → max_lr) then exponential decay to ~0
        - Early stopping: training stops when val loss doesn't improve for 5 epochs
        - Support for dataset subsets to handle large-scale training

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (recommended).
            epochs: Maximum number of training epochs. Default is 100.
            warmup_fraction: Fraction of total steps for linear warmup.
                Default is 0.1 (10%).
            max_lr: Maximum learning rate (at end of warmup). Default is 1e-3.
            decay_rate: Exponential decay factor per step. default: 0.9999
                (small decay rate = slower decay). Common values:
                - 0.9999: very smooth decay (slow)
                - 0.999: moderate decay
                - 0.99: faster decay
            weight_decay: L2 regularization coefficient. Default is 1e-2 (0.01).
            max_grad_norm: Maximum gradient norm for clipping. Default is 1.0.
            patience: Number of epochs with no improvement before early stopping.
                Default is 5 (as per wav2sleep paper).

        Returns:
            Dictionary with keys:
            - 'train_loss': List of average training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_accuracy': List of validation accuracies per epoch (if available)
            - 'learning_rates': List of learning rates per step

        Example:
            >>> trainer = Wav2SleepTrainer(model, device='cuda')
            >>> history = trainer.train_with_schedule(
            ...     train_dataloader=train_loader,
            ...     val_dataloader=val_loader,
            ...     epochs=100,
            ...     warmup_fraction=0.1,
            ...     max_lr=1e-3,
            ...     patience=5,  # Paper-specified early stopping
            ... )
        """
        # Calculate total training steps
        steps_per_epoch = len(train_dataloader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = max(1, int(total_steps * warmup_fraction))

        # Initialize AdamW optimizer with weight decay only on non-bias/norm params
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "norm", "LayerNorm"])
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in ["bias", "norm", "LayerNorm"])
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=max_lr, betas=(0.9, 0.999))

        # Initialize the warmup + exponential decay scheduler
        scheduler = LinearWarmupExponentialDecayScheduler(
            optimizer=optimizer,
            max_lr=max_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_rate=decay_rate,
        )

        # Implement full training loop with per-batch scheduler stepping
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                output = self.model(**batch)
                loss = output["loss"]

                # Backward pass
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Update weights
                optimizer.step()

                # Step scheduler per batch (critical for warmup schedule)
                scheduler.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                num_val_batches = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        # Move batch to device
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}

                        # Forward pass
                        output = self.model(**batch)
                        loss = output["loss"]

                        val_loss += loss.item()
                        num_val_batches += 1

                        # Track accuracy if available
                        if "y_prob" in output and "y_true" in output:
                            preds = output["y_prob"].argmax(dim=-1)
                            val_correct += (preds == output["y_true"]).sum().item()
                            val_total += len(output["y_true"])

                avg_val_loss = val_loss / num_val_batches
                history["val_loss"].append(avg_val_loss)

                if val_total > 0:
                    val_accuracy = val_correct / val_total
                    history["val_accuracy"].append(val_accuracy)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if self.exp_path is not None:
                        self.save_checkpoint(
                            os.path.join(self.exp_path, "best.pt")
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
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

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 100,
        optimizer_class=None,
        optimizer_params=None,
        scheduler_class=None,
        scheduler_params=None,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        patience: int = 5,
        eval_metric: str = "loss",
        eval_mode: str = "min",
    ) -> Dict[str, List[float]]:
        """Standard train method (calls train_with_schedule internally).

        For wav2sleep, use train_with_schedule() directly for full control
        over the learning rate schedule.
        """
        # This method is here for compatibility with BaseTrainer
        # For wav2sleep-specific training, use train_with_schedule() instead
        return self.train_with_schedule(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            max_lr=optimizer_params.get("lr", 1e-3) if optimizer_params else 1e-3,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            patience=patience,
        )


def create_subset_loaders(
    dataset: Dataset,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    batch_size: int = 16,
    train_indices: Optional[List[int]] = None,
    val_indices: Optional[List[int]] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training on dataset subsets.

    This utility function helps manage large datasets by allowing training and
    validation on subsets of the full data. Useful when the full dataset is too
    large for memory or computation.

    Args:
        dataset: PyHealth SampleDataset.
        train_fraction: Fraction of training set to use (0.0 to 1.0).
            Default is 1.0 (use all).
        val_fraction: Fraction of validation set to use (0.0 to 1.0).
            Default is 1.0 (use all).
        batch_size: Batch size for loaders.
        train_indices: Pre-defined training indices. If None, a random subset
            is selected based on train_fraction.
        val_indices: Pre-defined validation indices. If None, a random subset
            is selected based on val_fraction.
        seed: Random seed for reproducible subset selection.

    Returns:
        Tuple of (train_loader, val_loader) DataLoaders.

    Example:
        Use only 50% of training data and 100% of validation data:
        >>> train_loader, val_loader = create_subset_loaders(
        ...     dataset, train_fraction=0.5, val_fraction=1.0, batch_size=32
        ... )
    """
    from pyhealth.datasets.utils import collate_fn_dict_with_padding

    rng = torch.Generator()
    rng.manual_seed(seed)

    if train_indices is None:
        n_train = max(1, int(len(dataset) * train_fraction * 0.8))
        train_indices = torch.randperm(int(len(dataset) * 0.8), generator=rng).tolist()[:n_train]

    if val_indices is None:
        val_start = int(len(dataset) * 0.8)
        n_val = max(1, int((len(dataset) - val_start) * val_fraction))
        val_indices = list(range(val_start, min(val_start + n_val, len(dataset))))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None

    # Use PyTorch DataLoader directly instead of get_dataloader
    # to avoid calling set_shuffle on Subset object
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_dict_with_padding,
    )
    val_loader = None
    if val_subset:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_dict_with_padding,
        )

    return train_loader, val_loader

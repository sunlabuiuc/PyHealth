"""Training helpers for the DeepHit synthetic Neural Fine-Gray experiment.

Run this module through ``examples/deephit_competing_risks_neuralfg.py``.
It uses only synthetic data and keeps the test set separate from model selection.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from pyhealth.datasets import get_dataloader
from pyhealth.models.neural_fg import NeuralFG

from .data import create_dataset, prepare_pyhealth_samples


@dataclass
class TrainConfig:
    """Options for one reproducible NeuralFG training experiment."""

    max_samples: int | None = 2_000  # Set to None for all 30,000 rows.
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 42
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    hidden_dim: int = 16
    embedding_dim: int = 16
    embedding_layers: int = 2
    monotonic_layers: int = 2
    dropout: float = 0.0
    device: str = "cpu"  # CPU is the reliable default on macOS.


def stratified_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make nonoverlapping, reproducible splits stratified by event label."""

    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to < 1")
    if not df.index.is_unique:
        raise ValueError("Input dataframe indices must be unique")
    if "label" not in df:
        raise ValueError("Expected a 'label' column")

    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for label in sorted(df["label"].unique()):
        indices = df.index[df["label"] == label].to_numpy()
        if len(indices) < 3:
            raise ValueError(
                f"Event class {label} has fewer than 3 samples; "
                "cannot place it in all three splits"
            )
        indices = rng.permutation(indices)
        n_train = max(1, int(len(indices) * train_fraction))
        n_val = max(1, int(len(indices) * val_fraction))
        if n_train + n_val >= len(indices):
            n_val = len(indices) - n_train - 1
        train_ids.extend(indices[:n_train])
        val_ids.extend(indices[n_train : n_train + n_val])
        test_ids.extend(indices[n_train + n_val :])

    def subset(ids: list[int]) -> pd.DataFrame:
        return df.loc[ids].sample(frac=1, random_state=seed).copy()

    return subset(train_ids), subset(val_ids), subset(test_ids)

def _run_epoch(
    model: NeuralFG,
    loader,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> float:
    """Train one epoch, or evaluate an epoch when optimizer is None.

    Even for evaluation, NeuralFG needs autograd enabled: its event
    density is computed via the time derivative of the CIF.
    """

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_samples = 0

    # Do not wrap validation in torch.no_grad(): the NeuralFG loss
    # explicitly calls torch.autograd.grad(dCIF / dt).
    with torch.enable_grad():
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(**batch)
            loss: Tensor = output["loss"]

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite NeuralFG loss encountered")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
                optimizer.step()

            batch_size = int(output["y_true"].shape[0])
            total_loss += loss.detach().item() * batch_size
            n_samples += batch_size

    if n_samples == 0:
        raise ValueError("DataLoader produced no samples")
    return total_loss / n_samples


def run_training(
    df: pd.DataFrame,
    config: TrainConfig | None = None,
) -> tuple[NeuralFG, pd.DataFrame, float]:
    """Train NeuralFG, select by validation NLL, evaluate test NLL once.

    Args:
        df: DeepHit dataframe containing the 12 features, time, and label.
        config: Training options; defaults to a 2,000-row smoke experiment.

    Returns:
        model: Restored best-validation NeuralFG model.
        history: Per-epoch training and validation losses.
        test_loss: Test negative log-likelihood after model selection.
    """

    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.max_samples is not None:
        if config.max_samples < 3:
            raise ValueError("max_samples must be at least 3")
        df = df.sample(
            n=min(config.max_samples, len(df)),
            random_state=config.seed,
        ).copy()

    event_values = sorted(df["label"].unique().tolist())
    if event_values != list(range(len(event_values))):
        raise ValueError(
            "Event codes must be consecutive integers 0, 1, ..., R"
        )
    if len(event_values) < 2:
        raise ValueError("Expected censoring (0) and at least one event")
    num_risks = len(event_values) - 1

    train_df, val_df, test_df = stratified_split(
        df,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    print(
        f"Split sizes: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )
    print("Training event counts:", train_df["label"].value_counts().to_dict())

    train_ds = create_dataset(prepare_pyhealth_samples(train_df))
    val_ds = create_dataset(prepare_pyhealth_samples(val_df))
    test_ds = create_dataset(prepare_pyhealth_samples(test_df))

    train_loader = get_dataloader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        val_ds, batch_size=config.batch_size, shuffle=False
    )
    test_loader = get_dataloader(
        test_ds, batch_size=config.batch_size, shuffle=False
    )

    # These quantiles use training event times ONLY to avoid test leakage.
    observed_train_times = train_df.loc[train_df["label"] > 0, "time"]
    if observed_train_times.empty:
        raise ValueError("Training split must contain observed events")
    evaluation_times = torch.tensor(
        observed_train_times.quantile([0.25, 0.50, 0.75]).to_numpy(),
        dtype=torch.float32,
    )
    print(
        "Training-set evaluation times:",
        evaluation_times.tolist(),
    )

    model = NeuralFG(
        dataset=train_ds,
        num_risks=num_risks,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        embedding_layers=config.embedding_layers,
        monotonic_layers=config.monotonic_layers,
        dropout=config.dropout,
        time_grid=evaluation_times,
    ).to(torch.device(config.device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, optimizer, config.gradient_clip
        )
        val_loss = _run_epoch(model, val_loader, None, config.gradient_clip)
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(
            f"Epoch {epoch:03d}: train NLL={train_loss:.4f} "
            f"| val NLL={val_loss:.4f}"
        )

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    # These statements must be OUTSIDE the epoch loop.
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")

    model.load_state_dict(best_state)

    test_loss = _run_epoch(
        model,
        test_loader,
        None,
        config.gradient_clip,
    )

    print(
        f"Best epoch: {best_epoch} | "
        f"validation NLL: {best_val_loss:.4f}"
    )
    print(f"Held-out test NLL: {test_loss:.4f}")

    # IMPORTANT: return all three outputs.
    return model, pd.DataFrame(history_rows), test_loss
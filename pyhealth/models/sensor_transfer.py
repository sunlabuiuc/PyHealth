"""Adaptive transfer learning from source sensors to a target sensor.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Pretrains a :class:`~pyhealth.models.SensorLSTM` on the source
    sensors one after another, scaling each source's learning rate by its
    similarity to the target (IPD), then fine-tunes on the target sensor.
    The weighting can be switched to equal or inverse weights, or turned off
    for a no-transfer baseline, to run the weighting ablation.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .ipd import learning_rate_scales, paired_ipd, source_weights
from .sensor_lstm import SensorLSTM

ORDERS = ("ascending_weight", "descending_weight", "given")


def _stream(batch: dict[str, Any], key: str) -> torch.Tensor:
    """Returns the tensor of an input feature from a batch."""
    feature = batch[key]
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    return feature


def compute_source_distances(
    loader: Any,
    target_key: str,
    source_keys: Sequence[str],
    max_pairs: int = 64,
    window: int | None = 10,
) -> dict[str, float]:
    """Computes the IPD between each source stream and the target stream.

    Args:
        loader: A dataloader whose batches hold the target and source keys.
        target_key: Name of the target feature, e.g. ``"signal"``.
        source_keys: Names of the source features, e.g. ``["source_torso"]``.
        max_pairs: Maximum number of paired segments to use (DTW is slow).
        window: Sakoe-Chiba band half-width.

    Returns:
        Maps each source key to its IPD to the target.

    Examples:
        >>> distances = compute_source_distances(
        ...     loader, "signal", ["source_torso"]
        ... )  # doctest: +SKIP
    """
    target: list[np.ndarray] = []
    sources: dict[str, list[np.ndarray]] = {key: [] for key in source_keys}
    count = 0
    for batch in loader:
        target.append(_stream(batch, target_key).float().cpu().numpy())
        for key in source_keys:
            sources[key].append(_stream(batch, key).float().cpu().numpy())
        count += len(target[-1])
        if count >= max_pairs:
            break
    target_all = np.concatenate(target)[:max_pairs]
    return {
        key: paired_ipd(np.concatenate(parts)[:max_pairs], target_all, window)
        for key, parts in sources.items()
    }


def train_epochs(
    model: SensorLSTM, loader: Any, lr: float, epochs: int = 1
) -> list[float]:
    """Trains the model on its current ``input_key`` stream.

    Args:
        model: The model to train.
        loader: Training dataloader.
        lr: Learning rate of the Adam optimizer.
        epochs: Number of passes over the loader.

    Returns:
        The mean training loss of each epoch.

    Examples:
        >>> losses = train_epochs(model, loader, lr=1e-3)  # doctest: +SKIP
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        total, batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append(total / max(batches, 1))
    return history


def pretrain_on_sources(
    model: SensorLSTM,
    loader: Any,
    weights: dict[str, float],
    base_lr: float = 1e-3,
    epochs_per_source: int = 1,
    order: str = "ascending_weight",
) -> dict[str, list[float]]:
    """Trains on the source sensors in turn, scaling the learning rates.

    Each source is trained with ``base_lr`` times its learning-rate scale
    (see :func:`~pyhealth.models.ipd.learning_rate_scales`), so more similar
    sources influence the weights more. The model's ``input_key`` is restored
    afterwards.

    Args:
        model: The model to pretrain.
        loader: Training dataloader holding every source key.
        weights: Maps each source key to its weight.
        base_lr: Learning rate before scaling.
        epochs_per_source: Epochs spent on each source.
        order: ``"ascending_weight"`` (most similar source last),
            ``"descending_weight"`` or ``"given"`` (the order of ``weights``).

    Returns:
        Maps each source key to its per-epoch losses.

    Raises:
        ValueError: If ``order`` is unknown.

    Examples:
        >>> history = pretrain_on_sources(
        ...     model, loader, {"source_torso": 1.0}
        ... )  # doctest: +SKIP
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got '{order}'")
    scales = learning_rate_scales(weights)
    keys = list(weights)
    if order == "ascending_weight":
        keys.sort(key=lambda k: weights[k])
    elif order == "descending_weight":
        keys.sort(key=lambda k: -weights[k])

    original_key = model.input_key
    history: dict[str, list[float]] = {}
    try:
        for key in keys:
            model.input_key = key
            history[key] = train_epochs(
                model, loader, base_lr * scales[key], epochs_per_source
            )
    finally:
        model.input_key = original_key
    return history


def finetune(
    model: SensorLSTM,
    loader: Any,
    target_key: str = "signal",
    lr: float = 1e-3,
    epochs: int = 1,
) -> list[float]:
    """Fine-tunes the model on the target sensor.

    Args:
        model: The (pretrained) model.
        loader: Training dataloader.
        target_key: Name of the target feature.
        lr: Learning rate.
        epochs: Number of epochs.

    Returns:
        The mean training loss of each epoch.

    Examples:
        >>> losses = finetune(model, loader, epochs=2)  # doctest: +SKIP
    """
    model.input_key = target_key
    return train_epochs(model, loader, lr, epochs)


def evaluate_rcc(model: SensorLSTM, loader: Any) -> float:
    """Computes the ratio of correct classification (RCC) on the target.

    Args:
        model: The model, read through its current ``input_key``.
        loader: Evaluation dataloader.

    Returns:
        The fraction of correctly classified samples, between 0 and 1.

    Examples:
        >>> rcc = evaluate_rcc(model, val_loader)  # doctest: +SKIP
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            predicted = out["y_prob"].argmax(dim=-1)
            truth = out["y_true"].view(-1).to(predicted.device)
            correct += int((predicted == truth).sum())
            total += len(truth)
    return correct / max(total, 1)


def transfer_learn(
    model: SensorLSTM,
    train_loader: Any,
    val_loader: Any,
    target_key: str,
    source_keys: Sequence[str],
    weighting: str | None = "ipd",
    distances: dict[str, float] | None = None,
    base_lr: float = 1e-3,
    epochs_per_source: int = 1,
    finetune_epochs: int = 1,
    finetune_lr: float = 1e-3,
    order: str = "ascending_weight",
    max_pairs: int = 64,
) -> dict[str, Any]:
    """Runs the full pipeline: IPD, source pretraining, fine-tuning, RCC.

    Args:
        model: A freshly initialized model.
        train_loader: Training dataloader with target and source keys.
        val_loader: Validation dataloader used for the final RCC.
        target_key: Name of the target feature.
        source_keys: Names of the source features.
        weighting: ``"ipd"``, ``"equal"`` or ``"inverse"`` to pretrain with
            that weighting, or ``None`` to skip pretraining (the no-transfer
            baseline).
        distances: Precomputed IPDs, to avoid recomputing them for each
            weighting of an ablation.
        base_lr: Source learning rate before scaling.
        epochs_per_source: Epochs spent on each source.
        finetune_epochs: Epochs on the target.
        finetune_lr: Target learning rate.
        order: Order of the sources, see :func:`pretrain_on_sources`.
        max_pairs: Pairs used for the IPD.

    Returns:
        A dict with ``rcc``, ``distances``, ``weights``,
        ``pretrain_losses`` and ``finetune_losses``.

    Examples:
        >>> result = transfer_learn(
        ...     model, train_loader, val_loader, "signal", ["source_torso"]
        ... )  # doctest: +SKIP
    """
    weights: dict[str, float] = {}
    pretrain_losses: dict[str, list[float]] = {}
    if weighting is not None and source_keys:
        if distances is None:
            distances = compute_source_distances(
                train_loader, target_key, source_keys, max_pairs
            )
        weights = source_weights(distances, weighting)
        pretrain_losses = pretrain_on_sources(
            model, train_loader, weights, base_lr, epochs_per_source, order
        )
    finetune_losses = finetune(
        model, train_loader, target_key, finetune_lr, finetune_epochs
    )
    return {
        "rcc": evaluate_rcc(model, val_loader),
        "distances": distances or {},
        "weights": weights,
        "pretrain_losses": pretrain_losses,
        "finetune_losses": finetune_losses,
    }

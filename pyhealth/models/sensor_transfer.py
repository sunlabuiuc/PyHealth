"""Adaptive transfer learning from source sensors to a target sensor.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Pretrains a :class:`~pyhealth.models.SensorLSTM` on the source
    sensors one after another, then fine-tunes it on the target sensor. Every
    source starts from the same learning rate, which is multiplied by
    ``1 - alpha`` after each epoch, where ``alpha`` comes from the source's IPD
    (see :mod:`pyhealth.models.ipd`): closer sources keep a larger learning
    rate for longer. The weighting can be switched to equal or inverse, or
    turned off for a no-transfer baseline, to run the weighting ablation.

Differences from the paper: fine-tuning uses a fixed learning rate and a fixed
    number of epochs. The paper's rule scales the target learning rate by one
    minus the validation loss and stops early; as written it would give a
    negative learning rate whenever the loss is above one.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .ipd import decay_rates, paired_ipd
from .sensor_lstm import SensorLSTM

ORDERS = ("descending_distance", "ascending_distance", "given")


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
    seed: int = 0,
) -> dict[str, float]:
    """Computes the IPD between each source stream and the target stream.

    All paired segments of the loader are collected, then ``max_pairs`` of
    them are drawn at random with ``seed`` (DTW is slow). The draw picks
    positions, so pass an unshuffled loader (``shuffle=False``) to get the
    same pairs, and so the same IPD, on every run.

    Args:
        loader: A dataloader whose batches hold the target and source keys.
        target_key: Name of the target feature, e.g. ``"signal"``.
        source_keys: Names of the source features, e.g. ``["source_torso"]``.
        max_pairs: Maximum number of paired segments to use.
        window: Sakoe-Chiba band half-width.
        seed: Seed of the random choice of pairs and of the smooth bootstrap.

    Returns:
        Maps each source key to its IPD to the target.

    Examples:
        >>> distances = compute_source_distances(
        ...     loader, "signal", ["source_torso"]
        ... )  # doctest: +SKIP
    """
    target: list[np.ndarray] = []
    sources: dict[str, list[np.ndarray]] = {key: [] for key in source_keys}
    for batch in loader:
        target.append(_stream(batch, target_key).float().cpu().numpy())
        for key in source_keys:
            sources[key].append(_stream(batch, key).float().cpu().numpy())
    target_all = np.concatenate(target)
    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(len(target_all), size=min(max_pairs, len(target_all)), replace=False)
    )
    return {
        key: paired_ipd(
            np.concatenate(parts)[chosen], target_all[chosen], window, seed=seed
        )
        for key, parts in sources.items()
    }


def train_epochs(
    model: SensorLSTM,
    loader: Any,
    lr: float,
    epochs: int = 1,
    max_grad_norm: float | None = 1.0,
    lr_decay: float = 1.0,
) -> list[float]:
    """Trains the model on its current ``input_key`` stream.

    Args:
        model: The model to train.
        loader: Training dataloader.
        lr: Initial learning rate of the Adam optimizer.
        epochs: Number of passes over the loader.
        max_grad_norm: Gradient-norm clipping threshold, which keeps LSTM
            training stable at higher learning rates; ``None`` turns it off.
        lr_decay: The learning rate is multiplied by this after every epoch
            (``1.0`` keeps it constant).

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
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append(total / max(batches, 1))
        for group in optimizer.param_groups:
            group["lr"] *= lr_decay
    return history


def pretrain_on_sources(
    model: SensorLSTM,
    loader: Any,
    distances: dict[str, float],
    alphas: dict[str, float],
    base_lr: float = 5e-4,
    epochs_per_source: int = 1,
    order: str = "descending_distance",
) -> dict[str, list[float]]:
    """Trains on the source sensors in turn with a per-source learning decay.

    Each source starts at ``base_lr`` and its learning rate is multiplied by
    ``1 - alpha`` after every epoch, so a source with a small ``alpha`` keeps
    teaching the model for longer. The model's ``input_key`` is restored
    afterwards.

    Args:
        model: The model to pretrain.
        loader: Training dataloader holding every source key.
        distances: Maps each source key to its IPD (used only for the order).
        alphas: Maps each source key to its decay rate, from
            :func:`~pyhealth.models.ipd.decay_rates`.
        base_lr: Initial learning rate of every source.
        epochs_per_source: Epochs spent on each source.
        order: ``"descending_distance"`` (the paper: the farthest source first,
            the most similar last), ``"ascending_distance"`` or ``"given"``
            (the order of ``distances``).

    Returns:
        Maps each source key to its per-epoch losses.

    Raises:
        ValueError: If ``order`` is unknown or the keys of ``distances`` and
            ``alphas`` differ.

    Examples:
        >>> history = pretrain_on_sources(
        ...     model, loader, {"source_torso": 1.0}, {"source_torso": 1.0}
        ... )  # doctest: +SKIP
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got '{order}'")
    if set(distances) != set(alphas):
        raise ValueError("distances and alphas must have the same keys")
    keys = list(distances)
    if order == "descending_distance":
        keys.sort(key=lambda k: -distances[k])
    elif order == "ascending_distance":
        keys.sort(key=lambda k: distances[k])

    original_key = model.input_key
    history: dict[str, list[float]] = {}
    try:
        for key in keys:
            model.input_key = key
            history[key] = train_epochs(
                model,
                loader,
                base_lr,
                epochs_per_source,
                lr_decay=1.0 - alphas[key],
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
        lr: Learning rate (kept constant).
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
    base_lr: float = 5e-4,
    epochs_per_source: int = 1,
    finetune_epochs: int = 1,
    finetune_lr: float = 1e-3,
    order: str = "descending_distance",
    max_pairs: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Runs the full pipeline: IPD, source pretraining, fine-tuning, RCC.

    Args:
        model: A freshly initialized model.
        train_loader: Training dataloader with target and source keys. It is
            used to compute the IPD, so pass ``distances`` (from an unshuffled
            loader) for reproducible runs.
        val_loader: Validation dataloader used for the final RCC.
        target_key: Name of the target feature.
        source_keys: Names of the source features.
        weighting: ``"ipd"``, ``"equal"`` or ``"inverse"`` to pretrain with
            that decay rule (see :func:`~pyhealth.models.ipd.decay_rates`), or
            ``None`` to skip pretraining (the no-transfer baseline).
        distances: Precomputed IPDs, to avoid recomputing them for each
            weighting of an ablation.
        base_lr: Source learning rate at the start of each source.
        epochs_per_source: Epochs spent on each source.
        finetune_epochs: Epochs on the target.
        finetune_lr: Target learning rate.
        order: Order of the sources, see :func:`pretrain_on_sources`.
        max_pairs: Pairs used for the IPD.
        seed: Seed of the IPD estimate.

    Returns:
        A dict with ``rcc``, ``distances``, ``alphas``, ``pretrain_losses``
        and ``finetune_losses``.

    Examples:
        >>> result = transfer_learn(
        ...     model, train_loader, val_loader, "signal", ["source_torso"]
        ... )  # doctest: +SKIP
    """
    alphas: dict[str, float] = {}
    pretrain_losses: dict[str, list[float]] = {}
    if weighting is not None and source_keys:
        if distances is None:
            distances = compute_source_distances(
                train_loader, target_key, source_keys, max_pairs, seed=seed
            )
        alphas = decay_rates(distances, weighting)
        pretrain_losses = pretrain_on_sources(
            model,
            train_loader,
            distances,
            alphas,
            base_lr,
            epochs_per_source,
            order,
        )
    finetune_losses = finetune(
        model, train_loader, target_key, finetune_lr, finetune_epochs
    )
    return {
        "rcc": evaluate_rcc(model, val_loader),
        "distances": distances or {},
        "alphas": alphas,
        "pretrain_losses": pretrain_losses,
        "finetune_losses": finetune_losses,
    }

"""Trains one :class:`~pyhealth.autoimprove.PipelineConfig` and scores it."""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import time
import traceback
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.trainer import Trainer, get_metrics_fn

from .space import PipelineConfig, SearchSpace

logger = logging.getLogger(__name__)

LOWER_IS_BETTER = frozenset(
    {"loss", "brier", "mse", "mae", "rmse", "kl_divergence", "ece", "ece_adapt",
     "hamming_loss"}
)
"""Metric names where a smaller value is better."""

MAX_LABEL_SCAN = 50000
"""Most samples whose labels are read to count positives; reading a label runs
the sample's processors, so larger datasets skip the count."""


def greater_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is better.

    Examples:
        >>> greater_is_better("pr_auc"), greater_is_better("brier")
        (True, False)
    """
    return metric not in LOWER_IS_BETTER


def train_pipeline(
    config: PipelineConfig,
    space: SearchSpace,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    metric: str,
    seed: int = 0,
    device: str | None = None,
) -> torch.nn.Module:
    """Builds and trains the model a config describes.

    Training uses only ``train_dataset``. ``val_dataset`` is used for early
    stopping (``torch`` models) or early-stopping rounds (``fit`` models).
    The global torch random state is restored afterwards.

    Args:
        config: A config already checked by ``space.validate``.
        space: The search space the config belongs to.
        train_dataset: Training samples.
        val_dataset: Validation samples.
        metric: Metric monitored for early stopping.
        seed: Seed for weight initialisation and shuffling.
        device: Torch device; defaults to CUDA when available.

    Returns:
        The trained model.

    Examples:
        >>> model = train_pipeline(config, space, train_ds, val_ds, "pr_auc")  # doctest: +SKIP
    """
    spec = space.models[config.model]
    model_cls = spec.load()
    with torch.random.fork_rng(devices=[]), _quiet_trainer():
        torch.manual_seed(seed)
        model = model_cls(train_dataset, **config.model_params)
        if spec.trainer == "fit":
            return model.fit(train_dataset, val_dataset)
        tp = config.train_params
        if tp.get("pos_weight") == "balanced":
            model.set_pos_weight("balanced", train_dataset)
        batch_size = int(tp.get("batch_size", 64))
        trainer = Trainer(model, metrics=[metric], device=device, enable_logging=False)
        trainer.train(
            get_dataloader(train_dataset, batch_size, shuffle=True, seed=seed),
            get_dataloader(val_dataset, batch_size),
            epochs=int(tp.get("epochs", 10)),
            optimizer_params={"lr": float(tp.get("lr", 1e-3))},
            weight_decay=float(tp.get("weight_decay", 0.0)),
            monitor=metric,
            monitor_criterion="max" if greater_is_better(metric) else "min",
            patience=int(tp.get("patience", 3)),
        )
    return model


def predict(
    model: torch.nn.Module, dataset: SampleDataset, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray, list]:
    """Returns labels, predicted probabilities and patient ids for a dataset.

    Examples:
        >>> y_true, y_prob, patient_ids = predict(model, test_ds)  # doctest: +SKIP
    """
    with _quiet_trainer():
        trainer = Trainer(model, device=_device_of(model), enable_logging=False)
    loader = get_dataloader(dataset, batch_size)
    try:
        y_true, y_prob, _, patient_ids = trainer.inference(
            loader, return_patient_ids=True
        )
    except KeyError:  # samples without a patient_id field
        y_true, y_prob, _ = trainer.inference(loader)
        patient_ids = []
    return y_true, y_prob, patient_ids


def score(
    model: torch.nn.Module, dataset: SampleDataset, metrics: Sequence[str]
) -> dict[str, float]:
    """Computes ``metrics`` for a trained model on a dataset.

    Examples:
        >>> score(model, val_ds, ["pr_auc", "roc_auc"])  # doctest: +SKIP
        {'pr_auc': 0.41, 'roc_auc': 0.78}
    """
    y_true, y_prob, _ = predict(model, dataset)
    values = get_metrics_fn(model.mode)(y_true, y_prob, metrics=list(metrics))
    return {k: float(v) for k, v in values.items()}


def run_config(
    config: PipelineConfig,
    space: SearchSpace,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    metric: str,
    report_metrics: Sequence[str] = (),
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Trains a config on the training set and scores it on validation.

    Returns:
        ``{"metrics": {...}, "seconds": float, "n_params": int}``.

    Examples:
        >>> run_config(config, space, train_ds, val_ds, "pr_auc")  # doctest: +SKIP
        {'metrics': {'pr_auc': 0.41}, 'seconds': 12.3, 'n_params': 51521}
    """
    start = time.perf_counter()
    model = train_pipeline(
        config, space, train_dataset, val_dataset, metric, seed, device
    )
    names = list(dict.fromkeys([metric, *report_metrics]))
    return {
        "metrics": score(model, val_dataset, names),
        "seconds": time.perf_counter() - start,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def patient_folds(
    dataset: SampleDataset, n_folds: int, seed: int = 0
) -> list[np.ndarray]:
    """Splits a dataset's sample indices into folds by patient.

    Every sample of a patient lands in the same fold, so a patient with
    several samples never sits on both sides of a fold. Patients are shuffled
    with ``seed``. When the label is 0/1 and the dataset has at most
    ``MAX_LABEL_SCAN`` (50,000) samples, the patients with a positive sample are
    dealt first, each to the fold with the fewest of them, so every fold has
    positives if at least ``n_folds`` patients do. The other patients go to
    the currently smallest fold, which keeps fold sizes close when patients
    have different numbers of samples.

    Raises:
        ValueError: If there are fewer patients than folds.

    Examples:
        >>> folds = patient_folds(train_ds, 5)  # doctest: +SKIP
        >>> sum(len(f) for f in folds) == len(train_ds)  # doctest: +SKIP
        True
    """
    patients = sorted(dataset.patient_to_index)
    if len(patients) < n_folds:
        raise ValueError(f"{len(patients)} patients cannot fill {n_folds} folds")
    rng = np.random.default_rng(seed)
    order = [patients[i] for i in rng.permutation(len(patients))]
    positive = _positive_patients(dataset) or set()
    if 0 < len(positive) < n_folds:
        logger.warning(
            "only %d patients have a positive label, so some of the %d folds "
            "have none and the metric may be undefined there",
            len(positive), n_folds,
        )
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    n_positive = [0] * n_folds
    for patient in [p for p in order if p in positive]:
        k = min(range(n_folds), key=lambda j: (n_positive[j], len(folds[j])))
        folds[k].extend(dataset.patient_to_index[patient])
        n_positive[k] += 1
    for patient in [p for p in order if p not in positive]:
        k = min(range(n_folds), key=lambda j: len(folds[j]))
        folds[k].extend(dataset.patient_to_index[patient])
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def _positive_patients(dataset: SampleDataset) -> set[str] | None:
    """Patients with a positive sample, or ``None`` if the label is not 0/1
    or the dataset is too large to read every label."""
    if len(dataset) > MAX_LABEL_SCAN:
        logger.info(
            "%d samples: folds are not stratified by label (at most %d are read)",
            len(dataset), MAX_LABEL_SCAN,
        )
        return None
    key = next(iter(dataset.output_schema))
    positive = set()
    for patient, indices in dataset.patient_to_index.items():
        for i in indices:
            try:
                value = float(dataset[i][key])
            except (TypeError, ValueError, RuntimeError):  # not a single number
                return None
            if value not in (0.0, 1.0):
                return None
            if value == 1.0:
                positive.add(patient)
    return positive


def run_config_cv(
    config: PipelineConfig,
    space: SearchSpace,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    metric: str,
    folds: Sequence[np.ndarray],
    report_metrics: Sequence[str] = (),
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Scores a config by patient-grouped cross-validation on the training set.

    For each fold, the config is trained on the other folds and scored on
    the held-out fold. ``val_dataset`` is used only for early stopping, so
    no fold is scored by a model that saw it. Each metric is the mean over
    folds; ``<metric>_sd`` holds the spread between folds, and
    ``fold_scores`` the per-fold values of ``metric`` (for paired
    comparisons between configs scored on the same folds).

    Scoring on every training patient gives a much less noisy estimate than
    one small validation split, at the cost of training ``len(folds)``
    models per trial. Processors are not refitted per fold.

    Returns:
        ``{"metrics": {...}, "fold_scores": [...], "seconds": float,
        "n_params": int}``.

    Examples:
        >>> folds = patient_folds(train_ds, 5)  # doctest: +SKIP
        >>> run_config_cv(config, space, train_ds, val_ds, "pr_auc", folds)  # doctest: +SKIP
        {'metrics': {'pr_auc': 0.47, 'pr_auc_sd': 0.03}, 'seconds': 61.0, 'n_params': 51521}
    """
    start = time.perf_counter()
    names = list(dict.fromkeys([metric, *report_metrics]))
    all_indices = np.arange(len(train_dataset))
    per_fold: list[dict[str, float]] = []
    n_params = 0
    for held_out in folds:
        fit_part = train_dataset.subset(np.setdiff1d(all_indices, held_out).tolist())
        model = train_pipeline(
            config, space, fit_part, val_dataset, metric, seed, device
        )
        per_fold.append(score(model, train_dataset.subset(held_out.tolist()), names))
        n_params = sum(p.numel() for p in model.parameters())
    metrics = {}
    for name in names:
        values = np.array([m[name] for m in per_fold], dtype=float)
        metrics[name] = float(values.mean())
        metrics[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "metrics": metrics,
        "fold_scores": [float(m[metric]) for m in per_fold],
        "seconds": time.perf_counter() - start,
        "n_params": n_params,
    }


def check_models(
    space: SearchSpace,
    dataset: SampleDataset,
    n_samples: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, str | None]:
    """Checks that every model in a space runs on a dataset's inputs.

    Each model is built with its default arguments and takes one training
    step on ``n_samples`` samples (taking each label value in turn when the
    label is a single number, so a binary sample has both classes): a
    forward and backward pass for ``torch`` models, a fit and a forward
    pass for ``fit`` models. A model that fails here would fail every
    trial, e.g. a model that cannot read one of the task's input types.

    Args:
        space: The models to check.
        dataset: Samples with the task's inputs, e.g. the training split.
        n_samples: How many samples to try.
        seed: Seed for weight initialisation.
        device: Torch device for ``torch`` models.

    Returns:
        Model name to ``None`` if it ran, else the error
        (:func:`describe_error`).

    Examples:
        >>> check_models(with_xgboost(SearchSpace()), code_seq_ds)  # doctest: +SKIP
        {'CNN': None, ..., 'XGBoost': "ValueError: Field 'conditions' uses ..."}
    """
    probe = dataset.subset(_probe_indices(dataset, n_samples))
    batch = next(iter(get_dataloader(probe, len(probe))))
    out: dict[str, str | None] = {}
    for name in sorted(space.models):
        spec = space.models[name]
        config, _ = space.validate(PipelineConfig(name))
        try:
            with torch.random.fork_rng(devices=[]), _quiet_trainer():
                torch.manual_seed(seed)
                model = spec.load()(probe, **config.model_params)
                if spec.trainer == "fit":
                    model.fit(probe)
                    with torch.no_grad():
                        model(**batch)
                else:
                    if device is not None:
                        model = model.to(device)
                    model(**batch)["loss"].backward()
            out[name] = None
        except Exception as e:  # noqa: BLE001 - every failure is reported
            out[name] = describe_error(e)
    return out


def describe_error(error: BaseException, message: bool = True) -> str:
    """An error as text, with the line that raised it.

    A bare ``assert`` has no message, so the failing line is the only clue a
    proposer gets about what to change. With ``message=False`` the message is
    left out: an exception raised while reading a sample can quote a raw
    value, so this is the form shown to a proposer.

    Examples:
        >>> try:
        ...     assert 197 % 4 == 0
        ... except AssertionError as e:
        ...     print(describe_error(e))  # doctest: +ELLIPSIS
        AssertionError: (no message) at ...: assert 197 % 4 == 0
        >>> def parse(value):
        ...     return int(value)
        >>> try:
        ...     parse("a raw value")
        ... except ValueError as e:
        ...     print(describe_error(e, message=False))  # doctest: +ELLIPSIS
        ValueError at ...: return int(value)
    """
    text = type(error).__name__
    if message:
        text += f": {str(error)[:300] or '(no message)'}"
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        last = frames[-1]
        text += f" at {os.path.basename(last.filename)}:{last.lineno}"
        if last.line:
            text += f": {last.line[:200]}"
    return text


def _probe_indices(dataset: SampleDataset, n: int, scan: int = 1000) -> list[int]:
    """Up to ``n`` sample indices, taking each label value in turn."""
    key = next(iter(dataset.output_schema))
    by_label: dict[float, list[int]] = {}
    for i in range(min(len(dataset), scan)):
        try:
            label = float(dataset[i][key])
        except (TypeError, ValueError, RuntimeError):  # not a single number
            return list(range(min(n, len(dataset))))
        by_label.setdefault(label, []).append(i)
    rounds = itertools.zip_longest(*by_label.values())
    picked = [i for group in rounds for i in group if i is not None]
    return sorted(picked[:n])


@contextlib.contextmanager
def _quiet_trainer() -> Iterator[None]:
    """Hides the Trainer's per-epoch log lines while a trial trains."""
    trainer_logger = logging.getLogger("pyhealth.trainer")
    level = trainer_logger.level
    trainer_logger.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        trainer_logger.setLevel(level)


def _device_of(model: torch.nn.Module) -> str:
    param = next(model.parameters(), None)
    return "cpu" if param is None else str(param.device)

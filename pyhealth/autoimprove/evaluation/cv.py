"""Patient-grouped cross-validation: folds, fold parts, and scoring a config on them."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pyhealth.datasets import SampleDataset

from ..search.space import PipelineConfig, SearchSpace
from ..utils import data
from .scoring import out_of_fold, score_predictions
from .training import fit_predict


logger = logging.getLogger(__name__)


def run_config(
    config: PipelineConfig,
    space: SearchSpace,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    metric: str,
    report_metrics: Sequence[str] = (),
    seed: int = 0,
    device: str | None = None,
    timeout: float | None = None,
    log_dir: str | None = None,
    label: str = "trial",
) -> dict[str, Any]:
    """Trains a config on the training set and scores it on validation.

    Returns:
        ``{"metrics": {...}, "seconds": float, "n_params": int, "oof": ...}``,
        with ``oof`` the validation predictions (:func:`out_of_fold`).

    Examples:
        >>> run_config(config, space, train_ds, val_ds, "pr_auc")  # doctest: +SKIP
        {'metrics': {'pr_auc': 0.41}, 'seconds': 12.3, 'n_params': 51521, ...}
    """
    start = time.perf_counter()
    out = fit_predict(config, space, train_dataset, val_dataset, [val_dataset],
                      metric, seed, device, timeout, log_dir, label)
    names = list(dict.fromkeys([metric, *report_metrics]))
    y_true, y_prob, _ = out["predictions"][0]
    return {
        "metrics": score_predictions(y_true, y_prob, out["mode"], names,
                                     **out["metric_kwargs"]),
        "seconds": time.perf_counter() - start,
        "n_params": out["n_params"],
        "oof": out_of_fold([np.arange(len(val_dataset))], out["predictions"],
                           split="val"),
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
    if len(dataset) > data.MAX_LABEL_SCAN:
        logger.info(
            "%d samples: folds are not stratified by label (at most %d are read)",
            len(dataset), data.MAX_LABEL_SCAN,
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
    stop_if: Callable[[list[float]], bool] | None = None,
    timeout: float | None = None,
    log_dir: str | None = None,
    label: str = "trial",
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
    models per trial. The processors ``set_task`` fitted are reused;
    engineered feature columns are imputed and scaled per fold
    (:func:`fold_parts`).

    ``stop_if(fold_scores)`` is called after each fold; when it returns
    True the remaining folds are skipped (early pruning of a trial that is
    clearly worse) and ``stopped_early`` is set.

    Every held-out fold's predictions are kept, so each training sample gets
    one prediction from a model that did not see it (``oof``,
    :func:`out_of_fold`). Guardrails score these without retraining.

    Returns:
        ``{"metrics": {...}, "fold_scores": [...], "seconds": float,
        "n_params": int, "stopped_early": bool, "oof": ...}``; ``oof`` is
        ``None`` when the run stopped early.

    Examples:
        >>> folds = patient_folds(train_ds, 5)  # doctest: +SKIP
        >>> run_config_cv(config, space, train_ds, val_ds, "pr_auc", folds)  # doctest: +SKIP
        {'metrics': {'pr_auc': 0.47, 'pr_auc_sd': 0.03}, 'seconds': 61.0, 'n_params': 51521}
    """
    start = time.perf_counter()
    names = list(dict.fromkeys([metric, *report_metrics]))
    all_indices = np.arange(len(train_dataset))
    per_fold: list[dict[str, float]] = []
    held: list[np.ndarray] = []
    fold_preds: list[tuple] = []
    n_params = 0
    for k, held_out in enumerate(folds):
        fit_idx = np.setdiff1d(all_indices, held_out).tolist()
        fit_part, held_part, val_part = fold_parts(
            train_dataset, val_dataset, fit_idx, held_out.tolist()
        )
        out = fit_predict(config, space, fit_part, val_part, [held_part], metric,
                          seed, device, timeout, log_dir, f"{label}_fold{k}")
        y_true, y_prob, _ = out["predictions"][0]
        per_fold.append(score_predictions(y_true, y_prob, out["mode"], names,
                                          **out["metric_kwargs"]))
        held.append(np.asarray(held_out))
        fold_preds.append(out["predictions"][0])
        n_params = out["n_params"]
        if stop_if is not None and len(per_fold) < len(folds) and stop_if(
            [float(m[metric]) for m in per_fold]
        ):
            break
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
        "stopped_early": len(per_fold) < len(folds),
        "oof": out_of_fold(held, fold_preds) if len(per_fold) == len(folds) else None,
    }


def fold_parts(
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    fit_idx: Sequence[int],
    held_idx: Sequence[int],
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """The training part, held-out fold and validation set for one CV fold.

    Engineered feature columns
    (:class:`~pyhealth.autoimprove.search.features.EngineeredDataset`) are imputed
    and scaled with statistics from the fold's training part only, so a
    held-out fold never informs its own medians and scales. Other fields
    keep the processors ``set_task`` fitted.

    Examples:
        >>> fit, held, val = fold_parts(train_ds, val_ds, [0, 1, 2], [3])  # doctest: +SKIP
    """
    from ..search.features import EngineeredDataset

    if isinstance(train_dataset, EngineeredDataset):
        stats = train_dataset.fit_stats(fit_idx)
        train_dataset = train_dataset.with_stats(stats)
        if isinstance(val_dataset, EngineeredDataset):
            val_dataset = val_dataset.with_stats(stats)
    return (train_dataset.subset(list(fit_idx)), train_dataset.subset(list(held_idx)),
            val_dataset)

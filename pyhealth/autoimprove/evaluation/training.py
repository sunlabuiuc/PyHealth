"""Trains one :class:`~pyhealth.autoimprove.PipelineConfig` and scores it."""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.trainer import Trainer, get_metrics_fn, model_metrics_kwargs

from ..search.features import EngineeredDataset
from ..search.prompts import ENTRY
from ..search.space import PipelineConfig, SearchSpace
from ..utils import data
from ..utils.data import _label_mode, _probe_indices, _quiet_trainer, describe_error
from ..utils.sandbox import ContractError, run_sandboxed

logger = logging.getLogger(__name__)


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

    Raises:
        ValueError: For a code config (``config.code``): generated code is
            trained in its own process by :func:`fit_predict`, which
            returns its predictions rather than the model.

    Examples:
        >>> model = train_pipeline(config, space, train_ds, val_ds, "pr_auc")  # doctest: +SKIP
    """
    if config.code:
        raise ValueError("a code config is trained in its own process: use fit_predict")
    spec = space.models[config.model]
    model_cls = spec.load()
    with torch.random.fork_rng(devices=[]), _quiet_trainer():
        torch.manual_seed(seed)
        model = model_cls(train_dataset, **config.model_params)
        if spec.trainer == "fit":
            return model.fit(train_dataset, val_dataset)
        tp = config.train_params
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
    metrics_fn = get_metrics_fn(model.mode)
    return score_predictions(y_true, y_prob, model.mode, metrics,
                             **model_metrics_kwargs(model, metrics_fn))


def fit_predict(
    config: PipelineConfig,
    space: SearchSpace,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    predict_on: Sequence[SampleDataset],
    metric: str,
    seed: int = 0,
    device: str | None = None,
    timeout: float | None = None,
    log_dir: str | None = None,
    label: str = "trial",
) -> dict[str, Any]:
    """Trains a config and returns its predictions on other datasets.

    A config from the search space is trained here with
    :func:`train_pipeline`. A code config (``config.code``) is trained in its
    own process
    (:func:`~pyhealth.autoimprove.evaluation.training.fit_predict_code`) and
    only its predictions come back. Either way the caller scores the
    predictions the same way (:func:`score_predictions`).

    Args:
        predict_on: Datasets to predict after training, e.g. a held-out fold.
        timeout: Seconds a code config may take (``None``: no limit for
            configs from the space, ten minutes for code).
        log_dir: Where a code config's output is logged.
        label: Name of the job in logs and tracebacks.

    Returns:
        ``{"predictions": [(y_true, y_prob, patient_ids), ...], "mode": str,
        "n_params": int, "metric_kwargs": dict}``, one prediction tuple per
        dataset in ``predict_on``.

    Examples:
        >>> out = fit_predict(config, space, train_ds, val_ds, [val_ds], "pr_auc")  # doctest: +SKIP
        >>> y_true, y_prob, _ = out["predictions"][0]  # doctest: +SKIP
    """
    if config.code:

        return fit_predict_code(
            config.code, train_dataset, val_dataset, predict_on, metric, seed,
            device, timeout=600.0 if timeout is None else timeout,
            log_path=os.path.join(log_dir, f"{label}.log") if log_dir else None,
            label=label,
        )
    model = train_pipeline(config, space, train_dataset, val_dataset, metric, seed,
                           device)
    metrics_fn = get_metrics_fn(model.mode)
    return {
        "predictions": [predict(model, d) for d in predict_on],
        "mode": model.mode,
        "n_params": sum(p.numel() for p in model.parameters()),
        "metric_kwargs": model_metrics_kwargs(model, metrics_fn),
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


def _device_of(model: torch.nn.Module) -> str:
    param = next(model.parameters(), None)
    return "cpu" if param is None else str(param.device)


# ---- scoring predictions ----------------------------------------------------
LOWER_IS_BETTER = frozenset(
    {"loss", "brier", "mse", "mae", "rmse", "kl_divergence", "ece", "ece_adapt",
     "hamming_loss"}
)
"""Metric names where a smaller value is better."""


def greater_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is better.

    Examples:
        >>> greater_is_better("pr_auc"), greater_is_better("brier")
        (True, False)
    """
    return metric not in LOWER_IS_BETTER


def score_predictions(
    y_true: Any, y_prob: Any, mode: str, metrics: Sequence[str], **kwargs: Any
) -> dict[str, float]:
    """Computes ``metrics`` from predictions, with the label kind's metrics function.

    This is how every attempt is scored: the label processor of ``mode``
    (e.g. ``"binary"``) picks the :mod:`pyhealth.metrics` function, as
    :meth:`~pyhealth.trainer.Trainer.evaluate` does.

    Examples:
        >>> score_predictions(np.array([0, 1]), np.array([0.2, 0.9]), "binary", ["roc_auc"])
        {'roc_auc': 1.0}
    """
    values = get_metrics_fn(mode)(y_true, y_prob, metrics=list(metrics), **kwargs)
    return {k: float(v) for k, v in values.items()}


def out_of_fold(
    indices: Sequence[np.ndarray], predictions: Sequence[tuple], split: str = "train"
) -> dict[str, Any] | None:
    """Joins per-fold predictions into one record per sample.

    Returns:
        ``{"index", "fold", "y_true", "y_prob", "split"}`` as arrays in sample
        order of ``index``, or ``None`` when the predictions are not one row
        per sample (e.g. time-series labels), which guardrails cannot use.

    Examples:
        >>> oof = out_of_fold([np.array([1]), np.array([0])],
        ...                   [(np.array([1]), np.array([0.9]), []),
        ...                    (np.array([0]), np.array([0.2]), [])])
        >>> oof["index"].tolist(), oof["fold"].tolist(), oof["y_prob"].tolist()
        ([0, 1], [1, 0], [0.20000000298023224, 0.8999999761581543])
    """
    try:
        y_true = [np.asarray(p[0]) for p in predictions]
        y_prob = [np.asarray(p[1], dtype=np.float32) for p in predictions]
        if any(t.dtype == object or len(t) != len(i) or len(q) != len(i)
               for t, q, i in zip(y_true, y_prob, indices)):
            return None
        index = np.concatenate([np.asarray(i, dtype=np.int64) for i in indices])
        fold = np.concatenate([np.full(len(i), k, dtype=np.int16)
                               for k, i in enumerate(indices)])
        order = np.argsort(index, kind="stable")
        return {
            "index": index[order], "fold": fold[order],
            "y_true": np.concatenate(y_true)[order],
            "y_prob": np.concatenate(y_prob)[order], "split": split,
        }
    except (ValueError, TypeError):
        return None


# ---- patient-grouped cross-validation ---------------------------------------
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
    if isinstance(train_dataset, EngineeredDataset):
        stats = train_dataset.fit_stats(fit_idx)
        train_dataset = train_dataset.with_stats(stats)
        if isinstance(val_dataset, EngineeredDataset):
            val_dataset = val_dataset.with_stats(stats)
    return (train_dataset.subset(list(fit_idx)), train_dataset.subset(list(held_idx)),
            val_dataset)


# ---- generated-code trials --------------------------------------------------
def fit_predict_code(
    code: str,
    train_dataset: Any,
    val_dataset: Any,
    predict_on: Sequence[Any],
    metric: str,
    seed: int = 0,
    device: str | None = None,
    timeout: float = 600.0,
    log_path: str | None = None,
    label: str = "trial",
) -> dict[str, Any]:
    """Trains generated code in its own process and predicts ``predict_on``.

    Returns:
        The same as :func:`~pyhealth.autoimprove.evaluation.training.fit_predict`.

    Raises:
        SandboxError: If the code failed, broke the contract or timed out.

    Examples:
        >>> fit_predict_code(TEMPLATE, train_ds, val_ds, [val_ds], "pr_auc")  # doctest: +SKIP
    """
    payload = {
        "train": train_dataset, "val": val_dataset, "predict_on": list(predict_on),
        "metric": metric, "seed": seed, "device": device,
        "time_limit": max(timeout * 0.9, 1.0),
    }
    return run_sandboxed(code, f"{__name__}:run_code_trial",
                         payload, timeout=timeout, log_path=log_path, label=label)


def run_code_trial(
    namespace: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """The child side of a code trial: train, check, predict."""
    import torch

    from pyhealth.models import BaseModel

    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = payload["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    train = payload["train"]
    build = namespace.get(ENTRY)
    if not callable(build):
        raise ContractError(f"the module does not define {ENTRY}()")
    start = time.perf_counter()
    with _quiet_trainer():
        model = build(train, payload["val"], metric=payload["metric"], seed=seed,
                      device=device, time_limit=payload["time_limit"])
    if not isinstance(model, BaseModel):
        raise ContractError(f"{ENTRY} must return a pyhealth BaseModel, not "
                            f"{type(model).__name__}")
    mode = _label_mode(train)
    targets = payload["predict_on"]
    check_model(model, targets[0] if targets else train, mode)
    predictions = [predict(model, d) for d in targets]
    for y_true, y_prob, _ in predictions:
        _check_probabilities(np.asarray(y_prob, dtype=float), mode, len(y_true))
    return {
        "predictions": predictions, "mode": mode, "metric_kwargs": {},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": time.perf_counter() - start,
    }


def check_model(model: Any, dataset: Any, mode: str, n: int = 32) -> None:
    """Checks a trained model's outputs on one batch of ``dataset``.

    Raises:
        ContractError: If an output is missing or malformed, the predictions
            are not repeatable, every sample gets the same prediction, or the
            predictions change when the labels in the batch are shuffled.

    Examples:
        >>> check_model(model, held_out_fold, "binary")  # doctest: +SKIP
    """
    import torch

    from pyhealth.datasets import get_dataloader

    probe = dataset.subset(_probe_indices(dataset, n))
    batch = next(iter(get_dataloader(probe, len(probe))))
    label_keys = list(dataset.output_schema)
    model.eval()
    with torch.no_grad():
        first = model(**batch)
        again = model(**batch)
        shuffled = model(**_rolled(batch, label_keys))
    if not isinstance(first, dict):
        raise ContractError(f"model(**batch) must return a dict, not "
                            f"{type(first).__name__}")
    missing = [k for k in ("loss", "y_prob", "y_true") if k not in first]
    if missing:
        raise ContractError(f"model(**batch) is missing {', '.join(missing)}")
    y_prob = _as_array(first["y_prob"])
    if y_prob.shape[0] != len(probe):
        raise ContractError(f"y_prob has {y_prob.shape[0]} rows for a batch of "
                            f"{len(probe)} samples")
    _check_probabilities(y_prob, mode, len(probe))
    if not np.allclose(y_prob, _as_array(again["y_prob"]), atol=1e-6):
        raise ContractError("predictions differ between two identical calls in "
                            "eval mode")
    if len(probe) > 1 and float(np.ptp(y_prob, axis=0).max()) < 1e-9:
        raise ContractError("the model gives every sample the same prediction")
    if len(probe) > 1 and not np.allclose(y_prob, _as_array(shuffled["y_prob"]),
                                          atol=1e-5):
        raise ContractError("predictions change when the labels in the batch are "
                            "shuffled: the model reads the label")


def _rolled(batch: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """The batch with each label moved to the next sample."""
    out = dict(batch)
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        if hasattr(value, "roll"):
            out[key] = value.roll(1, dims=0)
        elif isinstance(value, list):
            out[key] = value[-1:] + value[:-1]
    return out


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def _check_probabilities(y_prob: np.ndarray, mode: str, n: int) -> None:
    if y_prob.dtype == object:
        return
    if not np.isfinite(y_prob).all():
        raise ContractError("y_prob holds NaN or infinite values")
    if mode in ("binary", "multiclass", "multilabel") and (
        y_prob.min() < -1e-6 or y_prob.max() > 1 + 1e-6
    ):
        raise ContractError(f"y_prob must be probabilities in [0, 1] for {mode} "
                            "labels; apply label_processor.activation")

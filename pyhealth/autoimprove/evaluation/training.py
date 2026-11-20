"""Trains one :class:`~pyhealth.autoimprove.PipelineConfig` and scores it."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.trainer import Trainer, get_metrics_fn, model_metrics_kwargs

from ..search.space import PipelineConfig, SearchSpace
from ..utils.data import _probe_indices, _quiet_trainer
from ..utils.errors import describe_error
from .scoring import greater_is_better, score_predictions

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
    (:func:`~pyhealth.autoimprove.evaluation.code_trials.fit_predict_code`) and
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
        from .code_trials import fit_predict_code

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

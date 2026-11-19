"""Code turns: a proposer writes its own model and how to train it.

The proposal is a Python module that defines::

    def build_and_train(train_dataset, val_dataset, *, metric, seed, device,
                        time_limit) -> BaseModel

It may define any model (a :class:`~pyhealth.models.BaseModel` subclass), any
feature transforms (inside the model, so they also apply when it predicts)
and any training loop, e.g. :class:`~pyhealth.trainer.Trainer`. The harness
runs it in its own process (:mod:`~pyhealth.autoimprove.sandbox`) on the
training data only, then predicts the held-out data itself, so the attempt is
scored exactly like a model from the search space.

After training, :func:`check_model` runs on one batch of the held-out data:
the outputs must have the BaseModel layout, the predictions must not change
when the labels in the batch are shuffled (a model that reads the answer
fails), and they must not be the same for every sample.
"""

from __future__ import annotations

import inspect
import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .sandbox import ContractError, run_sandboxed

ENTRY = "build_and_train"

CODE_CONTRACT = """\
Write one Python module that defines

    def build_and_train(train_dataset, val_dataset, *, metric, seed, device,
                        time_limit):
        ...
        return model

- `train_dataset` and `val_dataset` are pyhealth SampleDatasets (iterate them
  with `pyhealth.datasets.get_dataloader(dataset, batch_size, shuffle=...)`).
  Train only on `train_dataset`; use `val_dataset` for early stopping.
- Return a trained `pyhealth.models.BaseModel` subclass. `model(**batch)` on a
  batch from get_dataloader must return a dict with "loss", "y_prob" and
  "y_true", in the layout of the label processor
  (`self.label_processor.output_size()`, `.activation(logits)`,
  `.loss(logits, y_true)`).
- Put every feature transform inside the model's forward: the harness predicts
  held-out data by calling `model(**batch)` on raw batches.
- Predictions must not depend on the label in the batch.
- Finish well within `time_limit` seconds. `metric` is what the harness scores
  (e.g. "pr_auc"); `device` is where to train; `seed` seeds your randomness.
- Allowed imports: torch, numpy, scipy, sklearn, pandas, polars, pyhealth,
  math, statistics, collections, itertools, functools, typing, dataclasses,
  copy, random. No file, network or shell access: no open(), os, sys,
  subprocess, pickle, torch.save/load, exec/eval, or __dunder__ attributes
  other than __init__."""

TEMPLATE = '''\
import torch
import torch.nn as nn
from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.trainer import Trainer


class PooledLinear(BaseModel):
    """Embeds every input, averages over events, then one linear layer."""

    def __init__(self, dataset, embedding_dim=64):
        super().__init__(dataset)
        self.label_key = self.label_keys[0]
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim,
                            self.label_processor.output_size())

    def forward(self, **kwargs):
        inputs = {k: kwargs[k].to(self.device) for k in self.feature_keys}
        embedded = self.embedding_model(inputs)
        pooled = []
        for k in self.feature_keys:
            x = embedded[k]
            while x.dim() > 2:  # average over events (and codes)
                x = x.mean(dim=1)
            pooled.append(x)
        logits = self.fc(torch.cat(pooled, dim=1))
        y_true = kwargs[self.label_key].to(self.device)
        return {"loss": self.label_processor.loss(logits, y_true),
                "y_prob": self.label_processor.activation(logits),
                "y_true": y_true, "logit": logits}


def build_and_train(train_dataset, val_dataset, *, metric, seed, device,
                    time_limit):
    torch.manual_seed(seed)
    model = PooledLinear(train_dataset)
    trainer = Trainer(model, metrics=[metric], device=device,
                      enable_logging=False)
    lower_is_better = metric in ("loss", "mse", "mae", "rmse", "brier", "ece")
    trainer.train(get_dataloader(train_dataset, 64, shuffle=True, seed=seed),
                  get_dataloader(val_dataset, 64), epochs=5, monitor=metric,
                  monitor_criterion="min" if lower_is_better else "max",
                  patience=2)
    return model
'''
"""A minimal module that meets the contract, shown in prompts as a starting point."""


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
        The same as :func:`~pyhealth.autoimprove.evaluate.fit_predict`.

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
    return run_sandboxed(code, "pyhealth.autoimprove.codegen:run_code_trial",
                         payload, timeout=timeout, log_path=log_path, label=label)


def run_code_trial(
    namespace: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """The child side of a code trial: train, check, predict."""
    import torch

    from pyhealth.models import BaseModel

    from .evaluate import _quiet_trainer, predict
    from .improver import _label_mode

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

    from .evaluate import _probe_indices

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


def task_context(task: Any, dataset: Any, max_source: int = 8000) -> str:
    """What a code prompt says about the task: its source, inputs and label.

    Only the task's code and the shapes and types of one small batch are
    described, never sample values.

    Examples:
        >>> print(task_context(MortalityPredictionMIMIC3(), train_ds))  # doctest: +SKIP
    """
    parts = []
    if task is not None:
        try:
            source = inspect.getsource(type(task))
        except (OSError, TypeError):
            source = ""
        if source:
            if len(source) > max_source:
                source = source[:max_source] + "\n# ... (truncated)"
            parts.append(f"Task code ({type(task).__name__}), which built these "
                         f"samples:\n```python\n{source}\n```")
    lines = ["Inputs (one batch from get_dataloader):"]
    batch = _probe_batch(dataset)
    for key, proc in getattr(dataset, "input_processors", {}).items():
        try:
            size = f", size {proc.size()}"
        except Exception:  # noqa: BLE001 - size() is optional
            size = ""
        shape = _describe(batch.get(key)) if batch else "?"
        lines.append(f"- {key}: {type(proc).__name__}{size}; batch value: {shape}")
    for key, proc in getattr(dataset, "output_processors", {}).items():
        mode = getattr(proc, "mode", "?")
        try:
            out = f", output_size() = {proc.output_size()}"
        except Exception:  # noqa: BLE001
            out = ""
        shape = _describe(batch.get(key)) if batch else "?"
        lines.append(f"- label {key}: {type(proc).__name__} (mode {mode}{out}); "
                     f"batch value: {shape}")
    parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _probe_batch(dataset: Any) -> dict[str, Any] | None:
    from pyhealth.datasets import get_dataloader

    try:
        probe = dataset.subset(list(range(min(4, len(dataset)))))
        return next(iter(get_dataloader(probe, len(probe))))
    except Exception:  # noqa: BLE001 - the description is best effort
        return None


def _describe(value: Any) -> str:
    if value is None:
        return "?"
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return f"tensor {str(value.dtype).replace('torch.', '')} {tuple(value.shape)}"
    if isinstance(value, tuple):
        return "tuple(" + ", ".join(_describe(v) for v in value) + ")"
    if isinstance(value, list):
        inner = _describe(value[0]) if value else "?"
        return f"list of {len(value)} x {inner}"
    return type(value).__name__

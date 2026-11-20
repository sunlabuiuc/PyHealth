"""Code turns, the proposer's side: the contract and the task context in prompts.

The harness side (training generated code in its own process and checking it)
is :mod:`pyhealth.autoimprove.evaluation.code_trials`.
"""

from __future__ import annotations

import inspect
from typing import Any


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

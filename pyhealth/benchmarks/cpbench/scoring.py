"""Compute multiclass metrics, returning None for undefined conditional results."""

import numpy as np

from pyhealth.benchmarks import Result
from pyhealth.metrics import multiclass_metrics_fn


def score_multiclass_predictions(predictions, metric_names, vocabulary):
    """Score multiclass prediction sets, regardless of how they were produced."""
    metrics = multiclass_metrics_fn(**predictions, metrics=list(metric_names))
    return Result(
        "success",
        _mark_undefined_metrics(metrics, predictions),
        {"class_labels": sorted(vocabulary, key=vocabulary.get)},
    )


def _mark_undefined_metrics(metrics, predictions):
    """Keep genuine zeros; use None when a conditional metric has no observations."""
    truth = predictions["y_true"]
    singletons = predictions["y_predset"].sum(axis=1) == 1
    normalized = {}
    for name, value in metrics.items():
        if name in {"miscoverage_ps", "error_ps"}:
            observed = truth[singletons] if name == "error_ps" else truth
            present = np.isin(np.arange(len(value)), observed)
            normalized[name] = [
                float(item) if exists else None for item, exists in zip(value, present)
            ]
        elif name == "error_overall_ps" and not singletons.any():
            normalized[name] = None
        else:
            normalized[name] = float(value)
    return normalized

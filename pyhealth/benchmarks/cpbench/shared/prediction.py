"""Collect, copy, join and save arrays without assuming a prediction format."""

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pyhealth.datasets import get_dataloader

# --- Batch collection ---


def collect_predictions(dataset, batch_size, predict_batch):
    """Call an adapter callback for each batch and collect one row per sample.

    The callback receives the full batch and returns a mapping of names to arrays.
    It owns the native call, input selection, gradient context and tensor-to-array
    conversion. Arrays are copied before the next callback can reuse buffers.
    """

    def batches():
        remaining = len(dataset)

        for batch in get_dataloader(dataset, batch_size, shuffle=False):
            predictions = predict_batch(batch)
            count = _row_count(predictions)
            if count != min(batch_size, remaining):
                raise ValueError("batch predictions must have one row per input sample")

            remaining -= count
            yield predictions

        if remaining:
            raise ValueError("prediction count does not match the dataset")

    return concatenate_predictions(batches())


def copy_predictions(predictions):
    """Validate row alignment and snapshot arrays before an online method updates."""
    _row_count(predictions)

    return {name: np.array(values, copy=True) for name, values in predictions.items()}


def concatenate_predictions(batches):
    """Snapshot and join batches in order, requiring consistent output names."""
    collected = {}

    for batch in batches:
        predictions = copy_predictions(batch)
        if collected and predictions.keys() != collected.keys():
            raise ValueError("prediction batches must contain the same output names")

        for name, values in predictions.items():
            collected.setdefault(name, []).append(values)

    if not collected:
        raise ValueError("cannot collect predictions from empty batches")

    return {name: np.concatenate(values) for name, values in collected.items()}


def _row_count(predictions):
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("a prediction batch must be a nonempty mapping of arrays")

    arrays = [np.asarray(values) for values in predictions.values()]
    if any(values.ndim == 0 for values in arrays):
        raise ValueError("prediction arrays must have a sample dimension")

    count = len(arrays[0])
    if any(len(values) != count for values in arrays):
        raise ValueError("prediction arrays must have matching row counts")

    return count


# --- Output files ---


def save_predictions(evaluation, result, predictions, output_dir):
    """Write successful prediction arrays and point Result.artifact to the NPZ."""
    if output_dir is None or result.status != "success":
        return result

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{evaluation.id}.npz"

    with TemporaryDirectory(dir=directory, prefix=f".{evaluation.id}.") as temporary:
        path = Path(temporary) / "predictions.npz"
        np.savez_compressed(path, **predictions)
        os.replace(path, artifact)

    return replace(result, artifact=artifact)

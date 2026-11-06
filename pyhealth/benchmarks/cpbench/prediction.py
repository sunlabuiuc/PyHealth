"""Prediction-set collection, with task validation called separately by adapters."""

import numpy as np
import torch

from pyhealth.datasets import get_dataloader


def validate_multiclass(model, splits):
    """Check the task and shared class encoding; return the class vocabulary."""
    if model.mode != "multiclass" or len(model.label_keys) != 1:
        raise ValueError("multiclass prediction requires one multiclass label")
    return validate_label_encodings(model, splits)


def validate_label_encodings(model, splits):
    """Check feature/label separation and shared class ordering across splits."""
    if len(model.label_keys) != 1:
        raise ValueError("prediction-set collection requires one label key")
    label_key = model.label_keys[0]
    if label_key in model.feature_keys:
        raise ValueError("the label must not be a model feature")
    vocabulary = model.dataset.output_processors[label_key].label_vocab
    for name, dataset in splits.items():
        processor = dataset.output_processors.get(label_key)
        if getattr(processor, "label_vocab", None) != vocabulary:
            raise ValueError(
                f"{name} label encoding differs from model; reuse fitted processors"
            )
    return vocabulary


def predict_test_set(predictor, model, test_dataset, batch_size):
    """Collect predictions in dataset order; the adapter validates task semantics."""
    label_key = model.label_keys[0]
    predictions = concatenate_predictions(
        predict_batch(predictor, batch, model.feature_keys, label_key)
        for batch in get_dataloader(test_dataset, batch_size, shuffle=False)
    )
    if len(predictions["y_true"]) != len(test_dataset):
        raise ValueError("prediction count does not match the test dataset")
    return predictions


def predict_batch(predictor, batch, feature_keys, label_key):
    """Predict from features only; snapshot outputs before any method updates.

    predictor is a callable accepting model features and returning torch tensors
    named y_prob and y_predset. Labels are attached only after prediction.
    """
    features = {key: batch[key] for key in feature_keys}
    with torch.no_grad():
        output = predictor(**features)
    if not {"y_prob", "y_predset"} <= output.keys():
        raise ValueError("predictor must return y_prob and y_predset")
    truth = torch.as_tensor(batch[label_key]).detach().cpu().numpy().copy()
    probabilities = output["y_prob"].detach().cpu().numpy().copy()
    sets = output["y_predset"].detach().cpu().numpy().copy()
    predictions = {"y_true": truth, "y_prob": probabilities, "y_predset": sets}
    if truth.ndim == 0 or any(
        values.ndim == 0 or len(values) != len(truth) for values in predictions.values()
    ):
        raise ValueError("batch predictions must have one row per input sample")
    return predictions


def concatenate_predictions(batches):
    """Join recorded batches in the order supplied by the adapter."""
    batches = list(batches)
    return {
        name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]
    }


def validate_multiclass_predictions(predictions, class_count) -> None:
    """Check alignment, class membership, probabilities and encoded labels."""
    truth, probabilities, sets = (
        predictions["y_true"],
        predictions["y_prob"],
        predictions["y_predset"],
    )
    if truth.ndim != 1 or probabilities.shape != (len(truth), class_count):
        raise ValueError("prediction/label shape does not match the class vocabulary")
    if sets.shape != probabilities.shape or not np.isin(sets, [0, 1]).all():
        raise ValueError("y_predset must be a class membership mask matching y_prob")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("y_prob must contain finite, normalized class probabilities")
    if not np.issubdtype(truth.dtype, np.integer) or np.any(
        (truth < 0) | (truth >= class_count)
    ):
        raise ValueError("test labels are outside the encoded class vocabulary")

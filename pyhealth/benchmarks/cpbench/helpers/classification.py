"""Shared PyTorch classification-set calls and class/label validation for adapters."""

import numpy as np
import torch

from .prediction import collect_predictions


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


def predict_classification_sets(predictor, model, dataset, batch_size):
    """Collect classification sets through the format-independent collector."""
    return collect_predictions(
        dataset,
        batch_size,
        predict_batch=lambda batch: predict_classification_set_batch(
            predictor, batch, model.feature_keys, model.label_keys[0]
        ),
    )


def predict_classification_set_batch(predictor, batch, feature_keys, label_key):
    """Call a PyTorch set predictor with features only, returning NumPy views.

    The collector copies these views. An online adapter recording them before
    feedback must use copy_predictions before calling the method's update.
    """
    features = {key: batch[key] for key in feature_keys}
    with torch.no_grad():
        output = predictor(**features)
    if not {"y_prob", "y_predset"} <= output.keys():
        raise ValueError(
            "classification set predictor must return y_prob and y_predset"
        )
    return {
        "y_true": torch.as_tensor(batch[label_key]).detach().cpu().numpy(),
        "y_prob": output["y_prob"].detach().cpu().numpy(),
        "y_predset": output["y_predset"].detach().cpu().numpy(),
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

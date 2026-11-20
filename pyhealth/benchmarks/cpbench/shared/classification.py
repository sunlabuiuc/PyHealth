"""Classification prediction calls, label/output checks and multiclass scoring."""

import numpy as np
import torch

from pyhealth.benchmarks import Result
from pyhealth.metrics import multiclass_metrics_fn

from .collection import collect_predictions

# --- Label and output validation ---


def validate_label_encodings(model, splits):
    """Check feature/label separation and shared class ordering across splits.

    Examples:
        >>> from types import SimpleNamespace
        >>> processor = SimpleNamespace(label_vocab={"a": 0, "b": 1})
        >>> dataset = SimpleNamespace(output_processors={"label": processor})
        >>> model = SimpleNamespace(
        ...     feature_keys=["x"], label_keys=["label"], dataset=dataset
        ... )
        >>> validate_label_encodings(model, {"calibration": dataset, "test": dataset})
        {'a': 0, 'b': 1}
    """
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


def validate_multiclass_predictions(predictions, class_count) -> None:
    """Check alignment, class membership, probabilities and encoded labels.

    Examples:
        >>> predictions = {
        ...     "y_true": np.array([0, 1]),
        ...     "y_prob": np.array([[0.8, 0.2], [0.1, 0.9]]),
        ...     "y_predset": np.array([[1, 0], [0, 1]]),
        ... }
        >>> validate_multiclass_predictions(predictions, class_count=2)
    """
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


# --- Prediction ---


def predict_classification_sets(predictor, model, dataset, batch_size):
    """Collect classification sets through the format-independent collector.

    Examples:
        >>> from types import SimpleNamespace
        >>> model = SimpleNamespace(feature_keys=["x"], label_keys=["label"])
        >>> samples = [{"x": torch.tensor([0.8, 0.2]), "label": torch.tensor(0)}]
        >>> predictor = lambda x: {"y_prob": x, "y_predset": x > 0.5}
        >>> predictions = predict_classification_sets(predictor, model, samples, 1)
        >>> predictions["y_predset"].tolist()
        [[True, False]]
    """
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

    Examples:
        >>> predictor = lambda x: {"y_prob": x, "y_predset": x > 0.5}
        >>> batch = {"x": torch.tensor([[0.8, 0.2]]), "label": torch.tensor([0])}
        >>> predictions = predict_classification_set_batch(
        ...     predictor, batch, feature_keys=["x"], label_key="label"
        ... )
        >>> predictions["y_true"].tolist()
        [0]
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


# --- Scoring ---


def score_multiclass_predictions(
    predictions, metric_names, vocabulary, *, model_predictions=None
):
    """Score multiclass sets, preserving native metric arrays and NaNs.

    Examples:
        >>> predictions = {
        ...     "y_true": np.array([0, 1]),
        ...     "y_prob": np.array([[0.9, 0.1], [0.9, 0.1]]),
        ...     "y_predset": np.array([[1, 0], [1, 0]]),
        ... }
        >>> result = score_multiclass_predictions(
        ...     predictions, ["miscoverage_overall_ps"], {0: 0, 1: 1}
        ... )
        >>> float(result.metrics["miscoverage_overall_ps"])
        0.5
    """
    metrics = multiclass_metrics_fn(**predictions, metrics=list(metric_names))

    metadata = {"class_labels": sorted(vocabulary, key=vocabulary.get)}
    if model_predictions is not None:
        metadata["model_predictions"] = str(model_predictions)

    return Result("success", metrics, metadata)

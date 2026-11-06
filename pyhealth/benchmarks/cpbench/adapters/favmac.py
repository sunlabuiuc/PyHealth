"""Held-out multilabel cost-control evaluation through FavMac's native API."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.helpers.classification import (
    predict_classification_sets,
    validate_label_encodings,
)
from pyhealth.benchmarks.cpbench.helpers.preparation import (
    prepare_evaluation,
    require_protocol,
)
from pyhealth.metrics import multilabel_metrics_fn


def run(
    evaluation: Evaluation, cache_dir: Path | None, output_dir: Path | None
) -> Result:
    """Use native FavMac params; vector weights may be lists in class order."""
    return execute(
        evaluation, **deepcopy(evaluation.params["method"].get("options", {}))
    )


def execute(evaluation, *, calibration_kwargs=None) -> Result:
    """Calibrate once and score without test feedback; tp/fp remain unweighted."""
    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")
    params, splits, model = prepare_evaluation(
        evaluation, required=("calibration", "test")
    )
    if model.mode != "multilabel":
        raise ValueError("FavMac requires a multilabel model")
    vocabulary = validate_label_encodings(model, splits)
    arguments = dict(params["method"].get("params", {}))
    for name in ("cost_weights", "value_weights"):
        if isinstance(arguments.get(name), (list, tuple)):
            arguments[name] = np.asarray(arguments[name], dtype=float)
    predictor = params["method"]["class"](model, **arguments)
    predictor.calibrate(splits["calibration"], **(calibration_kwargs or {}))
    predictor.eval()
    predictions = predict_classification_sets(
        predictor, model, splits["test"], params["protocol"].get("batch_size", 32)
    )
    validate_multilabel_predictions(predictions, len(vocabulary))
    metrics = multilabel_metrics_fn(**predictions, metrics=list(params["metrics"]))
    return Result(
        "success",
        {name: float(value) for name, value in metrics.items()},
        {"class_labels": sorted(vocabulary, key=vocabulary.get)},
    )


def validate_multilabel_predictions(predictions, class_count) -> None:
    """Require binary label/set matrices and independent probabilities in [0, 1]."""
    truth, probabilities, sets = (
        predictions["y_true"],
        predictions["y_prob"],
        predictions["y_predset"],
    )
    if (
        truth.ndim != 2
        or truth.shape[1] != class_count
        or probabilities.shape != truth.shape
        or sets.shape != truth.shape
    ):
        raise ValueError(
            "multilabel outputs must match the sample and class dimensions"
        )
    if not np.isin(truth, [0, 1]).all() or not np.isin(sets, [0, 1]).all():
        raise ValueError("multilabel labels and prediction sets must be binary masks")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("multilabel probabilities must be finite and in [0, 1]")

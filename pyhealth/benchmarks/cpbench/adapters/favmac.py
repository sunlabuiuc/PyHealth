"""Held-out multilabel cost-control evaluation through FavMac's native API."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.shared.classification import (
    predict_classification_sets,
    validate_label_encodings,
)
from pyhealth.benchmarks.cpbench.shared.evaluation_preparation import (
    prediction_cache_path,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.benchmarks.cpbench.shared.model_outputs import save_predictions
from pyhealth.metrics import multilabel_metrics_fn


def required_model_outputs(evaluation):
    """Declare the base outputs required by this adapter.

    Examples:
        >>> from types import SimpleNamespace
        >>> ev = SimpleNamespace(params={"method": {"class": dict}})
        >>> "calibration" in required_model_outputs(ev)
        True
    """
    return {"calibration": ("logit",), "test": ("logit", "y_prob")}


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Calibrate and score FavMac prediction sets for a multilabel model.

    Requires consistent class encodings. Constructor weights use encoded class
    order; ``method.options`` accepts only ``calibration_kwargs``. Weighted metrics
    receive the native weights and cost limit; ordinary tp/fp stay unweighted.
    Execution uses dataset order without feedback. See :ref:`cpbench-favmac`.

    Complete example (run from the repository root)::

        from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        from pyhealth.benchmarks import Evaluation
        from pyhealth.benchmarks.cpbench import CPBench
        from pyhealth.calib.predictionset import FavMac
        from pyhealth.datasets import PatientSplit
        from pyhealth.models import MLP

        evaluation = Evaluation({
            # Generate multilabel samples, such as [0, 2], rather than class indices.
            "dataset": {
                "class": SyntheticDataset,
                "params": {"seed": 5, "mode": "multilabel"},
            },
            # The task chooses the label processor, without changing the raw labels.
            "task": {"class": SyntheticTask, "params": {"mode": "multilabel"}},
            "model": {
                "class": MLP,
                "params": {"embedding_dim": 8, "hidden_dim": 8},
                "training": {"epochs": 1},
            },
            "method": {
                "class": FavMac,
                "params": {
                    "target_cost": 0.8,
                    "cost_weights": [1, 2, 1],
                    "value_weights": [2, 1, 3],
                },
                "options": {"calibration_kwargs": {}},
            },
            "metrics": ["tp", "fp", "fp_cost", "tp_value", "set_size",
                "miscoverage_overall_ps", "fp_cost_violation_rate"],
            "protocol": {
                "split": {
                    "class": PatientSplit,
                    "params": {"ratios": [0.4, 0.4, 0.2], "seed": 5},
                },
                "seed": 17,
                "batch_size": 16,
                "device": "cpu",
                "order": "dataset",
                "feedback": "none",
            },
        })

        result = CPBench().evaluate([evaluation])[evaluation]
        assert result.status == "success", result.metadata

    Examples:
        >>> result = run(evaluation, cache_dir=None)  # doctest: +SKIP
    """
    options = deepcopy(evaluation.params["method"].get("options", {}))
    unknown = options.keys() - {"calibration_kwargs"}
    if unknown:
        raise TypeError(f"unsupported method options: {sorted(unknown)}")

    calibration_kwargs = options.get("calibration_kwargs")

    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")

    params, splits, model = prepare_evaluation(
        evaluation,
        required=("calibration", "test"),
        outputs=required_model_outputs(evaluation),
        cache_dir=cache_dir,
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
    metrics = multilabel_metrics_fn(
        **predictions,
        metrics=list(params["metrics"]),
        cost_weights=arguments.get("cost_weights", 1.0),
        value_weights=arguments.get("value_weights", 1.0),
        target_cost=predictor.target_cost,
    )
    metadata = {"class_labels": sorted(vocabulary, key=vocabulary.get)}
    if getattr(model, "predictions_path", None) is not None:
        metadata["model_predictions"] = str(model.predictions_path)
    result = Result("success", metrics, metadata)

    if cache_dir is not None:
        path = prediction_cache_path(params, cache_dir)
        save_predictions(predictions, path)
        result = replace(result, artifact=path)

    return result


def validate_multilabel_predictions(predictions, class_count) -> None:
    """Require binary label/set matrices and independent probabilities in [0, 1].

    Examples:
        >>> predictions = {
        ...     "y_true": np.array([[1, 0]]),
        ...     "y_prob": np.array([[0.8, 0.3]]),
        ...     "y_predset": np.array([[1, 1]]),
        ... }
        >>> validate_multilabel_predictions(predictions, class_count=2)
    """
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

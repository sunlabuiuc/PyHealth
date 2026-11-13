"""Fixed calibration for multiclass prediction-set methods."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.shared.classification import (
    predict_classification_sets,
    score_multiclass_predictions,
    validate_label_encodings,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.shared.evaluation_preparation import (
    prediction_cache_path,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.benchmarks.cpbench.shared.model_outputs import save_predictions
from pyhealth.calib.predictionset import NeighborhoodLabel


def required_model_outputs(evaluation):
    """Declare the base outputs required by this adapter.

    Examples:
        >>> from types import SimpleNamespace
        >>> ev = SimpleNamespace(params={"method": {"class": dict}})
        >>> "calibration" in required_model_outputs(ev)
        True
    """
    fields = {"calibration": ("y_prob",), "test": ("y_prob",)}
    if evaluation.params["method"]["class"] is NeighborhoodLabel:
        fields["test"] += ("embed",)
        calibration = (
            evaluation.params["method"].get("options", {}).get("calibration_kwargs", {})
            or {}
        )
        if calibration.get("cal_embeddings") is None:
            fields["calibration"] += ("embed",)

    return fields


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Calibrate and score multiclass prediction sets using native method arguments.

    Requires a multiclass model and consistent label encodings. ``method.options``
    accepts only ``calibration_kwargs``; LabelShift requires ``target_label_dist``
    in encoded class order. Execution uses dataset order without test feedback.
    See :ref:`cpbench-fixed-calibration` for complete evaluation examples.

    Complete example (run from the repository root)::

        from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        from pyhealth.benchmarks import Evaluation
        from pyhealth.benchmarks.cpbench import CPBench
        from pyhealth.calib.predictionset import BaseConformal
        from pyhealth.datasets import PatientSplit
        from pyhealth.models import MLP

        evaluation = Evaluation({
            # seed controls feature noise; mode generates one integer label per sample.
            "dataset": {
                "class": SyntheticDataset,
                "params": {"seed": 5, "mode": "multiclass"},
            },
            # The task chooses the label processor, without changing the raw labels.
            "task": {"class": SyntheticTask, "params": {"mode": "multiclass"}},
            "model": {
                "class": MLP,
                "params": {"embedding_dim": 8, "hidden_dim": 8},
                "training": {"epochs": 1},
            },
            "method": {
                "class": BaseConformal,
                "params": {"alpha": 0.1, "random_state": 17},
                "options": {"calibration_kwargs": {}},
            },
            "metrics": ["accuracy", "miscoverage_overall_ps", "set_size"],
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

    if model.mode != "multiclass":
        raise ValueError("multiclass prediction requires a multiclass model")

    vocabulary = validate_label_encodings(model, splits)

    predictor = params["method"]["class"](model, **params["method"].get("params", {}))
    predictor.calibrate(splits["calibration"], **(calibration_kwargs or {}))

    predictor.eval()
    predictions = predict_classification_sets(
        predictor, model, splits["test"], params["protocol"].get("batch_size", 32)
    )

    validate_multiclass_predictions(predictions, len(vocabulary))
    result = score_multiclass_predictions(
        predictions,
        params["metrics"],
        vocabulary,
        model_predictions=getattr(model, "predictions_path", None),
    )

    if cache_dir is not None:
        path = prediction_cache_path(params, cache_dir)
        save_predictions(predictions, path)
        result = replace(result, artifact=path)

    return result

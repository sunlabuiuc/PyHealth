"""Fixed calibration for multiclass prediction-set methods."""

from copy import deepcopy
from pathlib import Path

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.prediction import (
    predict_test_set,
    validate_multiclass,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.preparation import (
    construct,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.benchmarks.cpbench.scoring import score_multiclass_predictions


def run(
    evaluation: Evaluation, cache_dir: Path | None, output_dir: Path | None
) -> Result:
    """Use native constructor params and optional calibration_kwargs in options."""
    return execute(
        evaluation, **deepcopy(evaluation.params["method"].get("options", {}))
    )


def execute(evaluation, *, calibration_kwargs=None) -> Result:
    """Calibrate once; predict in dataset order without test feedback."""
    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")
    params, splits, model = prepare_evaluation(
        evaluation, required=("calibration", "test")
    )
    vocabulary = validate_multiclass(model, splits)
    predictor = construct(params["method"], model)
    predictor.calibrate(splits["calibration"], **(calibration_kwargs or {}))
    predictor.eval()
    predictions = predict_test_set(
        predictor, model, splits["test"], params["protocol"].get("batch_size", 32)
    )
    validate_multiclass_predictions(predictions, len(vocabulary))
    return score_multiclass_predictions(predictions, params["metrics"], vocabulary)

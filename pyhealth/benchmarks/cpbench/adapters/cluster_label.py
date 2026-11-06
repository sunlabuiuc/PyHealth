"""Cluster-conditional prediction sets using model-derived reference embeddings."""

from copy import deepcopy
from pathlib import Path

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.helpers.classification import (
    predict_classification_sets,
    validate_multiclass,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.helpers.preparation import (
    construct,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.benchmarks.cpbench.helpers.scoring import score_multiclass_predictions
from pyhealth.calib.utils import extract_embeddings


def run(
    evaluation: Evaluation, cache_dir: Path | None, output_dir: Path | None
) -> Result:
    """Use native ClusterLabel params and optional calibration_kwargs in options.

    The train split supplies reference embeddings, including in checkpoint mode.
    The model must support embed=True. Calibration embeddings are extracted by
    ClusterLabel unless supplied through its native calibration arguments.
    """
    return execute(
        evaluation, **deepcopy(evaluation.params["method"].get("options", {}))
    )


def execute(evaluation, *, calibration_kwargs=None) -> Result:
    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")
    arguments = dict(calibration_kwargs or {})
    if "train_embeddings" in arguments:
        raise ValueError("train_embeddings are computed from the declared train split")
    params, splits, model = prepare_evaluation(
        evaluation, required=("train", "calibration", "test")
    )
    vocabulary = validate_multiclass(model, splits)
    batch_size = params["protocol"].get("batch_size", 32)
    predictor = construct(params["method"], model)
    train_embeddings = extract_embeddings(
        model,
        splits["train"],
        batch_size=batch_size,
        device=params["protocol"].get("device", "cpu"),
    )
    arguments.setdefault("batch_size", batch_size)
    predictor.calibrate(
        splits["calibration"], train_embeddings=train_embeddings, **arguments
    )
    predictor.eval()
    predictions = predict_classification_sets(
        predictor, model, splits["test"], batch_size
    )
    validate_multiclass_predictions(predictions, len(vocabulary))
    return score_multiclass_predictions(predictions, params["metrics"], vocabulary)

"""Cluster-conditional prediction sets using model-derived reference embeddings."""

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
from pyhealth.calib.utils import extract_embeddings


def required_model_outputs(evaluation):
    """Declare the base outputs required by this adapter.

    Examples:
        >>> from types import SimpleNamespace
        >>> ev = SimpleNamespace(params={"method": {"class": dict}})
        >>> "calibration" in required_model_outputs(ev)
        True
    """
    calibration = (
        evaluation.params["method"].get("options", {}).get("calibration_kwargs", {})
        or {}
    )
    fields = {
        "train": ("embed",),
        "calibration": ("y_prob",),
        "test": ("y_prob", "embed"),
    }
    if calibration.get("cal_embeddings") is None:
        fields["calibration"] += ("embed",)

    return fields


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Calibrate ClusterLabel using reference embeddings from the train partition.

    Requires a multiclass model supporting ``embed=True``. Calibration arguments
    come from ``method.options.calibration_kwargs``; omit ``train_embeddings``
    because CPBench computes them. Execution uses dataset order without feedback.
    See :ref:`cpbench-cluster-label` for configuration and a complete example.

    Examples:
        >>> result = run(evaluation, cache_dir=None)  # doctest: +SKIP
    """
    options = deepcopy(evaluation.params["method"].get("options", {}))
    unknown = options.keys() - {"calibration_kwargs"}
    if unknown:
        raise TypeError(f"unsupported method options: {sorted(unknown)}")

    calibration_kwargs = options.get("calibration_kwargs")

    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")

    arguments = dict(calibration_kwargs or {})
    if "train_embeddings" in arguments:
        raise ValueError("train_embeddings are computed from the declared train split")

    params, splits, model = prepare_evaluation(
        evaluation,
        required=("train", "calibration", "test"),
        outputs=required_model_outputs(evaluation),
        cache_dir=cache_dir,
    )

    if model.mode != "multiclass":
        raise ValueError("multiclass prediction requires a multiclass model")

    vocabulary = validate_label_encodings(model, splits)
    batch_size = params["protocol"].get("batch_size", 32)

    predictor = params["method"]["class"](model, **params["method"].get("params", {}))

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

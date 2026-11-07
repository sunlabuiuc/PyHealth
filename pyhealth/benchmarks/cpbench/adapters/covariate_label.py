"""Covariate-shift correction using native density estimation or supplied weights."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.shared.classification import (
    predict_classification_sets,
    score_multiclass_predictions,
    validate_label_encodings,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.shared.collection import collect_predictions
from pyhealth.benchmarks.cpbench.shared.evaluation_preparation import (
    prediction_cache_path,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.benchmarks.cpbench.shared.model_outputs import save_predictions


def required_model_outputs(evaluation):
    """Declare the base outputs required by this adapter.

    Examples:
        >>> from types import SimpleNamespace
        >>> ev = SimpleNamespace(params={"method": {"class": dict}})
        >>> "calibration" in required_model_outputs(ev)
        True
    """
    method = evaluation.params["method"]
    calibration = method.get("options", {}).get("calibration_kwargs", {}) or {}
    fields = {"calibration": ("y_prob",), "test": ("y_prob",)}
    if calibration.get("cal_weights") is None:
        fields["calibration"] += ("embed",)
        if method.get("params", {}).get("kde_cal") is None:
            fields["test"] += ("embed",)

    return fields


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Calibrate and score CovariateLabel using native density weighting.

    Requires a multiclass model. Default density fitting uses all test inputs
    before prediction, without their labels, and requires ``embed=True``.
    Supplied ``cal_weights`` bypass embedding extraction. Omit calibration/test
    embeddings from options; CPBench computes them. Execution uses dataset order
    without feedback. See :ref:`cpbench-covariate-label` for options and examples.

    Examples:
        >>> result = run(evaluation, cache_dir=None)  # doctest: +SKIP
    """
    options = deepcopy(evaluation.params["method"].get("options", {}))
    unknown = options.keys() - {"calibration_kwargs"}
    if unknown:
        raise TypeError(f"unsupported method options: {sorted(unknown)}")

    calibration = dict(options.get("calibration_kwargs") or {})
    supplied_embeddings = calibration.keys() & {"cal_embeddings", "test_embeddings"}
    if supplied_embeddings:
        raise ValueError(
            "cal_embeddings/test_embeddings are computed from declared splits"
        )

    require_protocol(evaluation.params["protocol"], order="dataset", feedback="none")

    method_params = evaluation.params["method"].get("params", {})
    has_source_density = method_params.get("kde_cal") is not None
    has_target_density = method_params.get("kde_test") is not None
    if has_source_density != has_target_density:
        raise ValueError("kde_cal and kde_test must be supplied together")

    params, splits, model = prepare_evaluation(
        evaluation,
        required=("calibration", "test"),
        outputs=required_model_outputs(evaluation),
        cache_dir=cache_dir,
    )

    if model.mode != "multiclass":
        raise ValueError("multiclass prediction requires a multiclass model")

    vocabulary = validate_label_encodings(model, splits)
    batch_size = params["protocol"].get("batch_size", 32)

    predictor = params["method"]["class"](model, **params["method"].get("params", {}))

    if calibration.get("cal_weights") is None:
        calibration["cal_embeddings"] = _extract_embeddings(
            model, splits["calibration"], batch_size
        )

        if not has_source_density:
            calibration["test_embeddings"] = _extract_embeddings(
                model, splits["test"], batch_size
            )

    predictor.calibrate(splits["calibration"], **calibration)

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


def _extract_embeddings(model, dataset, batch_size):
    """Collect model embeddings using features only, preserving dataset order."""

    def predict_batch(batch):
        features = {key: batch[key] for key in model.feature_keys}
        output = model(**features, embed=True)
        if "embed" not in output:
            raise ValueError(
                "CovariateLabel density estimation requires embed=True output"
            )

        return {"embed": output["embed"].detach().cpu().numpy()}

    with torch.no_grad():
        predictions = collect_predictions(dataset, batch_size, predict_batch)

    return predictions["embed"]

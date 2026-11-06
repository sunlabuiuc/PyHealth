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
from pyhealth.benchmarks.cpbench.shared.prediction import (
    collect_predictions,
    prediction_cache_path,
    save_predictions,
)
from pyhealth.benchmarks.cpbench.shared.preparation import (
    prepare_evaluation,
    require_protocol,
)


def run(evaluation: Evaluation, cache_dir: Path | None) -> Result:
    """Calibrate CovariateLabel and evaluate multiclass prediction sets.

    Complete example (run from the repository root)::

        from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        from pyhealth.benchmarks import Evaluation
        from pyhealth.benchmarks.cpbench import CPBench
        from pyhealth.calib.predictionset import CovariateLabel
        from pyhealth.datasets import PatientSplit
        from pyhealth.models import MLP

        # params are constructor arguments, not compatibility declarations.
        evaluation = Evaluation({
            "dataset": {
                "class": SyntheticDataset,
                "params": {"mode": "multiclass"},  # Generates integer labels.
            },
            # Selects label processing; does not change the raw labels.
            "task": {"class": SyntheticTask, "params": {"mode": "multiclass"}},
            "model": {
                "class": MLP,  # Supports embed=True for density estimation.
                "params": {"embedding_dim": 8, "hidden_dim": 8},
                "training": {"epochs": 1},
            },
            "method": {
                "class": CovariateLabel,
                "params": {"alpha": 0.1, "random_state": 0},
            },
            "metrics": ["accuracy", "set_size", "miscoverage_overall_ps"],
            "protocol": {
                "split": {
                    "class": PatientSplit,
                    "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0},
                    "roles": ["train", "calibration", "test"],
                },
                "seed": 0,  # Split/method seeds are independent of this seed.
            },
        })

        result = CPBench().evaluate([evaluation])[evaluation]
        assert result.status == "success", result.metadata

    The synthetic example demonstrates execution, without simulating a shift.
    Real dataset/task constructors must produce one multiclass label and share
    the model's fitted class encoding. model selects training or a checkpoint;
    training requires train as the first split role. Both sources require
    calibration/test roles. Metrics name native multiclass metrics. The only
    supported protocol.order/feedback values are their defaults: dataset/none.

    By default this procedure uses ALL test inputs for density estimation before
    prediction, following native CovariateLabel. No test labels enter embedding
    extraction, calibration or prediction calls. The model must support embed=True.
    Embeddings follow dataset order, using protocol.batch_size (default 32).

    method.params are native constructor arguments. Optional kde_cal/kde_test
    must be supplied together as importable callable references; with this pair,
    only calibration embeddings are extracted. Otherwise native calibrate fits
    the density estimators from calibration/test embeddings.

    method.options may contain only calibration_kwargs. To supply known
    importance weights instead of fitting densities, pass cal_weights as a list
    in calibration-dataset row order. These are calibration-example likelihood
    ratios, not class costs. This path needs no embedding-capable model.
    cal_embeddings/test_embeddings must be omitted: the adapter derives them
    from the declared splits when needed. Other arguments go to native calibrate.
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
        evaluation, required=("calibration", "test"), cache_dir=cache_dir
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
    result = score_multiclass_predictions(predictions, params["metrics"], vocabulary)

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

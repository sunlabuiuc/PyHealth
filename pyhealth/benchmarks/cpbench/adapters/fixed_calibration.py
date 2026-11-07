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
    """Calibrate once and evaluate multiclass prediction sets.

    Complete example (run from the repository root)::

        from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        from pyhealth.benchmarks import Evaluation
        from pyhealth.benchmarks.cpbench import CPBench
        from pyhealth.calib.predictionset import BaseConformal
        from pyhealth.datasets import PatientSplit
        from pyhealth.models import MLP

        # params are arguments to each class, not compatibility declarations.
        evaluation = Evaluation({
            "dataset": {
                "class": SyntheticDataset,
                # Generates one integer label per sample.
                "params": {"mode": "multiclass"},
            },
            # Encodes each integer label as a class index.
            "task": {"class": SyntheticTask, "params": {"mode": "multiclass"}},
            "model": {
                "class": MLP,
                "params": {"embedding_dim": 8, "hidden_dim": 8},
                "training": {"epochs": 1},  # Train MLP on the train split only.
            },
            # alpha is target miscoverage: 0.1 requests 90% coverage.
            "method": {
                "class": BaseConformal,
                "params": {"alpha": 0.1, "random_state": 0},
            },
            "metrics": ["accuracy", "set_size", "miscoverage_overall_ps"],
            "protocol": {
                "split": {
                    "class": PatientSplit,
                    # Split patients into 40% train, 40% calibration, 20% test.
                    "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0},  # train, calibration, test
                },
                "seed": 0,  # Global RNGs; split/method seeds are independent.
            },
        })

        result = CPBench().evaluate([evaluation])[evaluation]
        assert result.status == "success", result.metadata

    SyntheticDataset.mode changes the generated labels. SyntheticTask.mode
    chooses their processor through its output schema; it does not change the
    raw labels. Real datasets/tasks take their own constructor arguments.

    dataset/task must produce SampleDataset splits with one multiclass label
    and the same fitted class encoding as the model. model must construct a
    BaseModel and supply training arguments.

    method.class must be explicitly mapped to this adapter: BaseConformal,
    LABEL, RC3P, SCRIB, NeighborhoodLabel or LabelShift. params are passed to
    that method's constructor; alpha and other arguments follow its native API.
    options may contain only calibration_kwargs and defaults to an empty mapping.
    LabelShift requires target_label_dist in calibration_kwargs, for example
    [0.2, 0.3, 0.5] for three encoded classes. source_label_dist is optional;
    both distributions use the dataset's class order.

    metrics must name native multiclass metrics. protocol.order must be
    "dataset" and protocol.feedback must be "none"; both default to those values.
    Other protocol fields follow benchmark.validate_evaluation. The
    example sets protocol.seed explicitly; omitting it uses the default of 0.

    With cache_dir, shared preparation reuses checkpoints and base outputs or
    produces missing artifacts. CP calibration runs for each evaluation. Without
    cache_dir, the model is trained afresh. output_dir is handled by CPBench.

    With the evaluation defined above:

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

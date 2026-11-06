"""Cluster-conditional prediction sets using model-derived reference embeddings."""

from copy import deepcopy
from pathlib import Path

from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench.shared.classification import (
    predict_classification_sets,
    score_multiclass_predictions,
    validate_label_encodings,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.shared.prediction import save_predictions
from pyhealth.benchmarks.cpbench.shared.preparation import (
    prepare_evaluation,
    require_protocol,
)
from pyhealth.calib.utils import extract_embeddings


def run(
    evaluation: Evaluation, cache_dir: Path | None, output_dir: Path | None
) -> Result:
    """Calibrate ClusterLabel using reference embeddings from the train split.

    Complete example (run from the repository root)::

        from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        from pyhealth.benchmarks import Evaluation
        from pyhealth.benchmarks.cpbench import CPBench
        from pyhealth.calib.predictionset import ClusterLabel
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
            "method": {
                "class": ClusterLabel,
                "params": {"alpha": 0.1, "n_clusters": 2, "random_state": 0},
            },
            "metrics": ["accuracy", "set_size", "miscoverage_overall_ps"],
            "protocol": {
                "split": {
                    "class": PatientSplit,
                    "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0},
                    "roles": ["train", "calibration", "test"],  # Same order as ratios.
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
    BaseModel supporting embed=True and select either training arguments or a
    checkpoint directory. The train role supplies reference embeddings even
    when loading a checkpoint; training requires it as the first split role.

    params are native ClusterLabel constructor arguments. options may contain
    only calibration_kwargs and defaults to an empty mapping. Those arguments
    are passed to calibrate, except train_embeddings must be omitted: this
    adapter extracts them from the declared train split. Calibration embeddings
    are extracted by ClusterLabel unless supplied through its native arguments.
    Calibration batch_size defaults to protocol.batch_size (32 if omitted).

    metrics must name native multiclass metrics. protocol.order must be
    "dataset" and protocol.feedback must be "none"; both default to those values.
    Other protocol fields follow benchmark.validate_evaluation. The
    example sets protocol.seed explicitly; omitting it uses the default of 0.
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
        evaluation, required=("train", "calibration", "test"), cache_dir=cache_dir
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
    result = score_multiclass_predictions(predictions, params["metrics"], vocabulary)

    return save_predictions(evaluation, result, predictions, output_dir)

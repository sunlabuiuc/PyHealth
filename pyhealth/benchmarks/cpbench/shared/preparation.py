"""Prepare explicit model sources and provide checkpoint-writing utilities."""

import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from pyhealth.benchmarks import Evaluation
from pyhealth.datasets import SampleBuilder, SampleDataset, Split, get_dataloader
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer
from pyhealth.utils import preserve_rng_state

from .classification import validate_label_encodings
from .model_predictions import (
    _CachedModel,
    collect_model_predictions,
    load_model_predictions,
    require_model_outputs,
)

# --- Evaluation preparation ---


def prepare_evaluation(evaluation: Evaluation, *, required, outputs=None):
    """Load the explicitly requested source without automatic cache lookup.

    model.training trains a fresh model; model.checkpoint loads fitted weights
    and preprocessing; model.predictions loads saved base outputs. outputs maps
    split names to the fields needed by the adapter. No original model or raw
    dataset is constructed when saved outputs are supplied.
    Runners own cache policy.

    Examples:
        >>> from .model_predictions import save_model_predictions
        >>> import numpy as np
        >>> values = {"test": {"y_true": np.array([0]),
        ...                    "y_prob": np.array([[0.8, 0.2]])}}
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "model.npz"
        ...     save_model_predictions(values, path, mode="multiclass",
        ...                            label_key="label", class_labels=[0, 1])
        ...     ev = Evaluation({"model": {"predictions": str(path)},
        ...                      "protocol": {"split": {"roles": ["test"]}}})
        ...     _, splits, model = prepare_evaluation(
        ...         ev, required=("test",), outputs={"test": ("y_prob",)}
        ...     )
        ...     print(model.mode, len(splits["test"]))
        multiclass 1
    """
    params = deepcopy(dict(evaluation.params))
    model_spec, protocol = params["model"], params["protocol"]
    missing = set(required) - set(protocol["split"]["roles"])
    if missing:
        raise ValueError(f"adapter requires split roles: {sorted(missing)}")

    with preserve_rng_state():
        if "predictions" in model_spec:
            predictions, metadata = load_model_predictions(model_spec["predictions"])
            require_model_outputs(predictions, required, outputs)
            model = _CachedModel(
                predictions,
                metadata,
                model_spec["predictions"],
                protocol.get("device", "cpu"),
            )

            return params, model.splits, model

        return _prepare_model_and_splits(params)


def _prepare_model_and_splits(params):
    model_spec, protocol = params["model"], params["protocol"]
    schema = None
    if "checkpoint" in model_spec:
        schema = SampleBuilder.load(str(Path(model_spec["checkpoint"]) / "schema.pkl"))

    splits = load_splits(
        params["dataset"], params["task"], protocol["split"], schema=schema
    )
    model = prepare_model(model_spec, splits, protocol)

    return params, splits, model


def predict_model_outputs(evaluation, *, outputs):
    """Collect the default adapters' requested base outputs without persistence.

    Adapters re-export this producer or supply their own implementation. Returns
    arrays by split and metadata accepted by save_model_predictions.

    Examples:
        >>> from .model_predictions import save_model_predictions
        >>> import numpy as np
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "base.npz"
        ...     save_model_predictions(
        ...         {"test": {"y_true": np.array([0]),
        ...                   "y_prob": np.array([[0.75, 0.25]])}}, path,
        ...         mode="multiclass", label_key="label", class_labels=[0, 1],
        ...     )
        ...     ev = Evaluation({"model": {"predictions": str(path)},
        ...                      "protocol": {"split": {"roles": ["test"]}}})
        ...     arrays, metadata = predict_model_outputs(
        ...         ev, outputs={"test": ("y_prob",)}
        ...     )
        ...     print(arrays["test"]["y_prob"].tolist())
        [[0.75, 0.25]]
    """
    params, splits, model = prepare_evaluation(
        evaluation, required=tuple(outputs), outputs=outputs
    )
    if len(model.label_keys) != 1:
        raise ValueError("base-output collection requires one label key")
    label_key = model.label_keys[0]
    vocabulary = getattr(
        model.dataset.output_processors[label_key], "label_vocab", None
    )
    if vocabulary is not None:
        vocabulary = validate_label_encodings(model, splits)

    predictions = collect_model_predictions(
        model, splits, outputs, params["protocol"].get("batch_size", 32)
    )

    return predictions, {
        "mode": model.mode,
        "label_key": label_key,
        "class_labels": None
        if vocabulary is None
        else sorted(vocabulary, key=vocabulary.get),
    }


def require_protocol(protocol, *, order, feedback):
    """Reject execution rules an adapter does not implement.

    Examples:
        >>> require_protocol({}, order="dataset", feedback="none")
        >>> require_protocol({"feedback": "immediate"}, order="dataset", feedback="none")
        Traceback (most recent call last):
            ...
        ValueError: adapter requires order='dataset', feedback='none'
    """
    if (
        protocol.get("order", "dataset") != order
        or protocol.get("feedback", "none") != feedback
    ):
        raise ValueError(f"adapter requires order={order!r}, feedback={feedback!r}")


# --- Datasets and models ---


def load_splits(dataset_spec, task_spec, split_spec, *, schema=None):
    """Use set_task's split API so training alone fits preprocessing.

    Examples:
        >>> from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        >>> from pyhealth.datasets import PatientSplit
        >>> splits = load_splits(
        ...     {"class": SyntheticDataset}, {"class": SyntheticTask},
        ...     {"class": PatientSplit, "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0},
        ...      "roles": ["train", "calibration", "test"]},
        ... )
        >>> sorted(splits)
        ['calibration', 'test', 'train']
    """
    task = task_spec["class"](**task_spec.get("params", {}))
    processors = {}
    if schema is not None:
        if (
            task.input_schema != schema.input_schema
            or task.output_schema != schema.output_schema
        ):
            raise ValueError("task schema differs from checkpoint preprocessing schema")

        if set(schema.input_processors) != set(task.input_schema) or set(
            schema.output_processors
        ) != set(task.output_schema):
            raise ValueError("checkpoint must supply every fitted processor")

        processors = {
            "input_processors": schema.input_processors,
            "output_processors": schema.output_processors,
        }

    splitter = split_spec["class"](**split_spec.get("params", {}))
    if not isinstance(splitter, Split):
        raise TypeError("protocol.split must construct a PyHealth Split")

    dataset = dataset_spec["class"](**dataset_spec.get("params", {}))
    parts = dataset.set_task(task, split=splitter, **processors)
    roles = split_spec["roles"]

    if not isinstance(parts, (list, tuple)) or len(parts) != len(roles):
        raise ValueError("split parts must match protocol.split.roles")

    for role, part in zip(roles, parts):
        if not isinstance(part, SampleDataset) or len(part) == 0:
            raise ValueError(f"{role} must be a nonempty SampleDataset")

    if len({id(part) for part in parts}) != len(parts):
        raise ValueError("split roles must not refer to the same dataset object")

    return dict(zip(roles, parts))


def prepare_model(model_spec, splits, protocol):
    """Build and train a model, or reconstruct it using checkpoint-fitted metadata.

    Examples:
        >>> from examples.cpbench.synthetic_data import make_splits
        >>> from pyhealth.models import MLP
        >>> splits = make_splits(n_train=6, n_calibration=3, n_test=3)
        >>> model = prepare_model(
        ...     {"class": MLP, "params": {"embedding_dim": 4, "hidden_dim": 4},
        ...      "training": {"epochs": 0}}, splits, {"batch_size": 3},
        ... )
        >>> model.training
        False
    """
    reference = (
        splits["train"] if "training" in model_spec else next(iter(splits.values()))
    )
    model = model_spec["class"](dataset=reference, **model_spec.get("params", {}))
    if not isinstance(model, BaseModel):
        raise TypeError("model.class must construct a BaseModel")

    device = protocol.get("device", "cpu")
    model.to(device)

    if "training" in model_spec:
        train_model(
            model,
            splits["train"],
            splits.get("validation"),
            model_spec["training"],
            batch_size=protocol.get("batch_size", 32),
            seed=protocol.get("seed", 0),
            device=device,
        )

    else:
        Trainer(model, device=device, enable_logging=False).load_ckpt(
            str(Path(model_spec["checkpoint"]) / "model.ckpt")
        )

    model.eval()

    return model


def train_model(
    model, train_dataset, validation_dataset, train_kwargs, *, batch_size, seed, device
) -> None:
    """Train using only the declared training and optional validation data.

    Examples:
        >>> from examples.cpbench.synthetic_data import make_splits
        >>> from pyhealth.models import MLP
        >>> splits = make_splits(n_train=6, n_calibration=3, n_test=3)
        >>> model = MLP(splits["train"], embedding_dim=4, hidden_dim=4)
        >>> train_model(
        ...     model, splits["train"], None, {"epochs": 1},
        ...     batch_size=3, seed=0, device="cpu",
        ... )
    """
    trainer = Trainer(model, device=device, enable_logging=False)
    if train_kwargs.get("monitor") is not None and validation_dataset is None:
        raise ValueError("training monitor requires a separate validation split")

    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True, seed=seed)
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = get_dataloader(validation_dataset, batch_size)

    trainer.train(train_loader, val_dataloader=validation_loader, **train_kwargs)


# --- Explicit checkpoint files ---


def save_checkpoint(model, dataset, destination: Path) -> None:
    """Atomically save weights and fitted preprocessing to a new directory.

    dataset must carry the model's fitted input/output processors. Provided
    processors are saved without refitting. The directory contains model.ckpt
    and schema.pkl, loadable via CPBench's model.checkpoint input.

    The destination must not exist; an existing path raises FileExistsError.
    Cache keys, cache lookup and cache locks are the caller's responsibility.

    Examples:
        >>> from examples.cpbench.synthetic_data import make_splits
        >>> from pyhealth.models import MLP
        >>> train = make_splits(n_train=6, n_calibration=3, n_test=3)["train"]
        >>> model = MLP(train, embedding_dim=4, hidden_dim=4)
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "checkpoint"
        ...     save_checkpoint(model, train, path)
        ...     print(sorted(item.name for item in path.iterdir()))
        ['model.ckpt', 'schema.pkl']
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        path = Path(temporary)
        Trainer(model, device=model.device, enable_logging=False).save_ckpt(
            str(path / "model.ckpt")
        )

        builder = SampleBuilder(
            dataset.input_schema,
            dataset.output_schema,
            input_processors=dataset.input_processors,
            output_processors=dataset.output_processors,
        )
        builder.fit(dataset)  # Provided processors are not refitted.
        builder.save(str(path / "schema.pkl"))

        os.replace(path, destination)

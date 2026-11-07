"""Prepare datasets, models and base outputs, reusing managed caches when enabled."""

import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from pyhealth.benchmarks import Evaluation
from pyhealth.datasets import (
    BaseDataset,
    SampleBuilder,
    SampleDataset,
    Split,
    get_dataloader,
)
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer
from pyhealth.utils import preserve_rng_state

from .classification import validate_label_encodings
from .model_outputs import (
    _CachedModel,
    _merge_model_predictions,
    collect_model_predictions,
    load_model_predictions,
    require_model_outputs,
    save_model_predictions,
)

# --- Evaluation preparation ---


def prepare_evaluation(
    evaluation: Evaluation, *, required, outputs=None, cache_dir=None
):
    """Prepare native inputs, loading or producing managed artifacts as needed.

    Training arguments describe how to produce a missing model. With cache_dir,
    checkpoints and base outputs are reused across CP methods. Without it, a new
    model is trained. The evaluation is never changed by cache resolution.

    Examples:
        >>> from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        >>> from pyhealth.datasets import PatientSplit
        >>> from pyhealth.models import MLP
        >>> ev = Evaluation({
        ...     "dataset": {"class": SyntheticDataset},
        ...     "task": {"class": SyntheticTask},
        ...     "model": {"class": MLP, "params": {"embedding_dim": 4, "hidden_dim": 4},
        ...               "training": {"epochs": 0}},
        ...     "protocol": {"split": {"class": PatientSplit,
        ...                   "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0}}},
        ... })
        >>> with TemporaryDirectory() as directory:
        ...     _, splits, model = prepare_evaluation(
        ...         ev, required=("test",), outputs={"test": ("y_prob",)},
        ...         cache_dir=Path(directory),
        ...     )
        ...     print(model.mode, sorted(splits))
        multiclass ['test']
    """
    params = deepcopy(dict(evaluation.params))
    if outputs is not None and not outputs:
        raise ValueError("outputs must request at least one split")

    required = (*required, *(outputs or {}))

    with preserve_rng_state():
        if cache_dir is None:
            return _prepare_model_and_splits(params, required=required)

        root = Path(cache_dir) / "cpbench"
        key = _model_cache_key(params)
        checkpoint = root / "checkpoints" / key
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        # Serialize creation and extension of one model's artifacts. Readers of
        # saved prediction files only observe complete, atomically written files.
        with FileLock(str(checkpoint) + ".lock"):
            return _prepare_cached(params, required, outputs, root, checkpoint)


def prediction_cache_path(params, cache_dir):
    """Key prediction arrays by their producing inputs, excluding scored metrics.

    Examples:
        >>> params = {"method": "example", "metrics": ["accuracy"]}
        >>> first = prediction_cache_path(params, Path("cache"))
        >>> params["metrics"] = ["set_size"]
        >>> prediction_cache_path(params, Path("cache")) == first
        True
    """
    key = Evaluation(
        {name: value for name, value in params.items() if name != "metrics"}
    ).id

    return Path(cache_dir) / "cpbench" / "predictions" / "conformal" / f"{key}.npz"


def _model_cache_key(params):
    protocol = params["protocol"]

    return Evaluation(
        {
            "dataset": params["dataset"],
            "task": params["task"],
            "model": params["model"],
            "split": protocol["split"],
            "seed": protocol.get("seed", 0),
            "batch_size": protocol.get("batch_size", 32),
            "device": protocol.get("device", "cpu"),
        }
    ).id


def _prepare_cached(params, required, outputs, root, checkpoint):
    requested = (
        None
        if outputs is None
        else {role: tuple(fields) for role, fields in outputs.items()}
    )
    if requested is not None:
        for role in required:
            requested.setdefault(role, ())

    prediction_dir = root / "predictions" / "base" / checkpoint.name
    predictions, metadata = {}, None
    missing = {}

    for role, fields in (requested or {}).items():
        path = prediction_dir / f"{role}.npz"
        if path.exists():
            loaded, current_metadata = load_model_predictions(path)
            if set(loaded) != {role}:
                raise ValueError(f"cached predictions must contain only split {role!r}")
            if metadata is not None and metadata != current_metadata:
                raise ValueError(
                    "cached prediction splits have different label metadata"
                )
            predictions.update(loaded)
            metadata = current_metadata

        absent = set(fields) - predictions.get(role, {}).keys()
        if role not in predictions or absent:
            missing[role] = tuple(sorted(absent))

    if requested is None or missing:
        existing_checkpoint = checkpoint if checkpoint.exists() else None
        if existing_checkpoint is None:
            # A newly trained model may differ from the previous instance even
            # with the same recipe. Invalidate old outputs before saving weights,
            # so an interrupted rebuild cannot leave new weights with old outputs.
            for path in prediction_dir.glob("*.npz"):
                path.unlink()
            predictions, metadata = {}, None
            missing = requested or {}

        _, splits, model = _prepare_model_and_splits(
            params,
            checkpoint=existing_checkpoint,
            dataset_cache=root / "datasets",
            required=required,
        )
        if existing_checkpoint is None:
            save_checkpoint(model, splits["train"], checkpoint)

        if requested is None:
            return params, splits, model

        current_metadata = _prediction_metadata(model, splits)
        if metadata is not None and metadata != current_metadata:
            raise ValueError(
                "cached predictions and checkpoint have different label metadata"
            )
        metadata = current_metadata
        produced = collect_model_predictions(
            model, splits, missing, params["protocol"].get("batch_size", 32)
        )

        for role, values in produced.items():
            combined = _merge_model_predictions(predictions.get(role), values)
            save_model_predictions(
                {role: combined}, prediction_dir / f"{role}.npz", **metadata
            )
            predictions[role] = combined

    require_model_outputs(predictions, required, requested)
    model = _CachedModel(
        predictions, metadata, prediction_dir, params["protocol"].get("device", "cpu")
    )

    return params, model.splits, model


def _prepare_model_and_splits(
    params, *, checkpoint=None, dataset_cache=None, required=()
):
    model_spec, protocol = params["model"], params["protocol"]
    schema = None
    if checkpoint is not None:
        schema = SampleBuilder.load(str(checkpoint / "schema.pkl"))

    splits = load_splits(
        params["dataset"],
        params["task"],
        protocol["split"],
        schema=schema,
        cache_dir=dataset_cache,
    )
    missing = set(required) - splits.keys()
    if missing:
        raise ValueError(f"adapter requires missing splits: {sorted(missing)}")

    model = prepare_model(model_spec, splits, protocol, checkpoint=checkpoint)

    return params, splits, model


def _prediction_metadata(model, splits):
    if len(model.label_keys) != 1:
        raise ValueError("base-output collection requires one label key")

    label_key = model.label_keys[0]
    vocabulary = getattr(
        model.dataset.output_processors[label_key], "label_vocab", None
    )
    if vocabulary is not None:
        vocabulary = validate_label_encodings(model, splits)

    return {
        "mode": model.mode,
        "label_key": label_key,
        "class_labels": None
        if vocabulary is None
        else sorted(vocabulary, key=vocabulary.get),
    }


def predict_model_outputs(evaluation, *, outputs, cache_dir=None):
    """Collect requested base outputs, reusing managed artifacts when enabled.

    Examples:
        >>> from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        >>> from pyhealth.datasets import PatientSplit
        >>> from pyhealth.models import MLP
        >>> ev = Evaluation({
        ...     "dataset": {"class": SyntheticDataset}, "task": {"class": SyntheticTask},
        ...     "model": {"class": MLP, "training": {"epochs": 0}},
        ...     "protocol": {"split": {"class": PatientSplit,
        ...                   "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0}}},
        ... })
        >>> values, metadata = predict_model_outputs(ev, outputs={"test": ("y_prob",)})
        >>> sorted(values["test"])[-1]
        'y_true'
    """
    params, splits, model = prepare_evaluation(
        evaluation, required=tuple(outputs), outputs=outputs, cache_dir=cache_dir
    )
    if isinstance(model, _CachedModel):
        return model.predictions, _prediction_metadata(model, splits)

    predictions = collect_model_predictions(
        model, splits, outputs, params["protocol"].get("batch_size", 32)
    )

    return predictions, _prediction_metadata(model, splits)


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


def load_splits(dataset_spec, task_spec, split_spec, *, schema=None, cache_dir=None):
    """Name three split outputs train/calibration/test, or four with validation second.

    set_task fits preprocessing on the first (training) partition only.

    Examples:
        >>> from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        >>> from pyhealth.datasets import PatientSplit
        >>> splits = load_splits(
        ...     {"class": SyntheticDataset}, {"class": SyntheticTask},
        ...     {"class": PatientSplit, "params": {"ratios": [0.4, 0.4, 0.2], "seed": 0}},
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

    dataset_params = dict(dataset_spec.get("params", {}))
    dataset_class = dataset_spec["class"]
    if (
        cache_dir is not None
        and isinstance(dataset_class, type)
        and issubclass(dataset_class, BaseDataset)
    ):
        dataset_params["cache_dir"] = str(cache_dir)

    dataset = dataset_class(**dataset_params)
    if cache_dir is not None and isinstance(dataset, BaseDataset):
        # Native caches already identify preprocessing by task and training split.
        # Supplying those same processors explicitly would create a second cache.
        processors = {}
    parts = dataset.set_task(task, split=splitter, **processors)

    if not isinstance(parts, (list, tuple)) or len(parts) not in (3, 4):
        raise ValueError("splitter must return three or four partitions")

    names = (
        ("train", "calibration", "test")
        if len(parts) == 3
        else ("train", "validation", "calibration", "test")
    )
    for name, part in zip(names, parts):
        if not isinstance(part, SampleDataset) or len(part) == 0:
            raise ValueError(f"{name} must be a nonempty SampleDataset")

    if len({id(part) for part in parts}) != len(parts):
        raise ValueError("split partitions must not refer to the same dataset object")

    return dict(zip(names, parts))


def prepare_model(model_spec, splits, protocol, *, checkpoint=None):
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
    reference = splits["train"]
    model = model_spec["class"](dataset=reference, **model_spec.get("params", {}))
    if not isinstance(model, BaseModel):
        raise TypeError("model.class must construct a BaseModel")

    device = protocol.get("device", "cpu")
    model.to(device)

    if checkpoint is None:
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
            str(checkpoint / "model.ckpt")
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
    and schema.pkl, used together by managed checkpoint loading.

    The destination must not exist; an existing path raises FileExistsError.
    This writer performs no cache lookup; shared preparation manages reuse and locking.

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

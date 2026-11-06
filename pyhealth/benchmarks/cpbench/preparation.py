"""Shared construction, task preparation, splitting, model loading and training."""

import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import SampleBuilder, SampleDataset, Split, get_dataloader
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer
from pyhealth.utils import preserve_rng_state


@contextmanager
def seeded(seed):
    """Seed an evaluation and restore the caller's random states on exit."""
    with preserve_rng_state():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield


def construct(specification, *args, **kwargs):
    """Construct the class/factory named by an evaluation block."""
    return specification["class"](*args, **kwargs, **specification.get("params", {}))


def prepare_evaluation(evaluation, *, required):
    """Prepare an owned model and named datasets, using the caller's RNG scope.

    Checkpoints are directories containing native Trainer model.ckpt weights and
    a native SampleBuilder schema.pkl with fitted processors. Loading one never
    fits processors or trains a model. Dataset APIs may use their configured caches;
    no benchmark artifacts are written.
    """
    params = deepcopy(dict(evaluation.params))
    model_spec, protocol = params["model"], params["protocol"]
    if "predictions" in model_spec:
        raise NotImplementedError(
            "this adapter requires a model; cached predictions are unsupported"
        )
    roles = protocol["split"]["roles"]
    missing = set(required) - set(roles)
    if missing:
        raise ValueError(f"adapter requires split roles: {sorted(missing)}")
    schema = None
    if "checkpoint" in model_spec:
        schema = SampleBuilder.load(str(Path(model_spec["checkpoint"]) / "schema.pkl"))
    splits = load_splits(
        params["dataset"], params["task"], protocol["split"], schema=schema
    )
    model = prepare_model(model_spec, splits, protocol)
    return params, splits, model


def load_splits(dataset_spec, task_spec, split_spec, *, schema=None):
    """Use set_task's split API so training alone fits preprocessing."""
    task = construct(task_spec)
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
    splitter = construct(split_spec)
    if not isinstance(splitter, Split):
        raise TypeError("protocol.split must construct a PyHealth Split")
    dataset = construct(dataset_spec)
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
    """Build and train a model, or reconstruct it using checkpoint-fitted metadata."""
    reference = (
        splits["train"] if "training" in model_spec else next(iter(splits.values()))
    )
    model = construct(model_spec, dataset=reference)
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
    """Train using only the declared training and optional validation data."""
    trainer = Trainer(model, device=device, enable_logging=False)
    if train_kwargs.get("monitor") is not None and validation_dataset is None:
        raise ValueError("training monitor requires a separate validation split")
    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True, seed=seed)
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = get_dataloader(validation_dataset, batch_size)
    trainer.train(train_loader, val_dataloader=validation_loader, **train_kwargs)


def require_protocol(protocol, *, order, feedback):
    """Reject execution rules an adapter does not implement."""
    if (
        protocol.get("order", "dataset") != order
        or protocol.get("feedback", "none") != feedback
    ):
        raise ValueError(f"adapter requires order={order!r}, feedback={feedback!r}")

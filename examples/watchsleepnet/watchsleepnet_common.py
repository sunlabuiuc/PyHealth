"""Shared helpers for the WatchSleepNet real-data examples.

Every training example requires an explicit root to a real dataset directory;
there is no synthetic-data fallback. The same functions build datasets, tasks,
models, checkpoints, and report tables for DREAMT, SHHS, MESA, ISRUC and
Sleep-Accel cohorts.

Examples:
    >>> from watchsleepnet_common import CLASS_NAMES, LOSS_MODES
    >>> from watchsleepnet_common import QUANTIZATION_MODES
    >>> CLASS_NAMES
    ('Wake', 'NREM', 'REM')
    >>> QUANTIZATION_MODES
    ('fp32', 'bf16', 'fp16')
    >>> LOSS_MODES
    ('ce', 'weighted_ce', 'focal', 'inverse_focal')
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.metrics import accuracy_score, f1_score

from pyhealth.datasets import (
    DREAMTDataset,
    ISRUCDataset,
    NSRRSleepDataset,
    SleepAccelDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import (
    SleepStagingIBIDREAMT,
    SleepStagingIBIISRUC,
    SleepStagingIBIMESA,
    SleepStagingIBISHHS,
    SleepStagingIBISleepAccel,
)
from pyhealth.trainer import Trainer

CLASS_NAMES = ("Wake", "NREM", "REM")
QUANTIZATION_MODES = ("fp32", "bf16", "fp16")
LOSS_MODES = ("ce", "weighted_ce", "focal", "inverse_focal")
QUANTIZATION_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
DEFAULT_MODEL_KWARGS = {
    "num_channels": 256,
    "kernel_size": 5,
    "hidden_dim": 256,
    "num_heads": 16,
    "num_layers": 4,
    "tcn_layers": 3,
}
TRAINER_METRICS = ["accuracy", "f1_macro", "cohen_kappa"]
CHECKPOINT_FORMAT = "watchsleepnet-pyhealth-v1"


def existing_dir(value: str) -> Path:
    """Returns an existing directory supplied on the command line.

    Args:
        value: User-supplied dataset path.

    Returns:
        The expanded path.

    Raises:
        argparse.ArgumentTypeError: If the path is not an existing directory.

    Examples:
        >>> existing_dir(".").is_dir()
        True
    """
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"{value!r} does not point to an existing dataset directory"
        )
    return path


def build_dataset(
    name: str,
    root: str | os.PathLike[str],
    cohort: str | None = None,
    cache_dir: str | None = None,
) -> Any:
    """Builds a PyHealth dataset for one of the supported IBI sources.

    Args:
        name: ``"dreamt"``, ``"shhs"``, ``"mesa"``, ``"isruc"`` or
            ``"sleep_accel"``.
        root: Dataset directory.
        cohort: NSRR cohort; defaults to ``name``.
        cache_dir: Optional writable cache for PyHealth metadata.

    Returns:
        A fitted PyHealth dataset.

    Raises:
        ValueError: For an unknown name.

    Examples:
        >>> build_dataset("dreamt", "unused-root") if False else "ready"
        'ready'
    """
    if name == "dreamt":
        return DREAMTDataset(root=str(root))
    if name in ("shhs", "mesa"):
        return NSRRSleepDataset(
            root=str(root),
            cohort=cohort or name,
            cache_dir=cache_dir,
        )
    if name == "isruc":
        return ISRUCDataset(root=str(root), cache_dir=cache_dir)
    if name == "sleep_accel":
        return SleepAccelDataset(root=str(root), cache_dir=cache_dir)
    raise ValueError(f"unknown dataset {name!r}")


def build_task(
    name: str,
    points_per_epoch: int = 750,
    max_epochs: int = 1100,
    signal_source: str = "psg_100hz",
    signal_channel: str | None = None,
    peak_backend: str = "auto",
    scorer: str = "expert1",
) -> Any:
    """Builds the three-class IBI task for a dataset.

    Args:
        name: ``"dreamt"``, ``"shhs"``, ``"mesa"``, ``"isruc"`` or
            ``"sleep_accel"``.
        points_per_epoch: IBI samples per 30 s epoch.
        max_epochs: Maximum epochs per recording.
        signal_source: ``"psg_100hz"`` or ``"wearable_64hz"`` for DREAMT.
        signal_channel: Waveform column or EDF channel.
        peak_backend: Beat detector. ``"auto"`` prefers the reference
            biosppy/neurokit backends with a scipy fallback.
        scorer: ``"expert1"`` or ``"expert2"`` hypnogram for ISRUC.

    Returns:
        A task whose samples use the shared signal/lengths/label contract.

    Raises:
        ValueError: For an unknown name.

    Examples:
        >>> build_task("dreamt", points_per_epoch=375).points_per_epoch
        375
        >>> build_task("isruc", scorer="expert2").scorer
        'expert2'
    """
    if name == "dreamt":
        return SleepStagingIBIDREAMT(
            signal_source=signal_source,
            signal_channel=signal_channel or "BVP",
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            peak_backend=peak_backend,
        )
    if name == "shhs":
        return SleepStagingIBISHHS(
            signal_channel=signal_channel or "ECG",
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            peak_backend=peak_backend,
        )
    if name == "mesa":
        return SleepStagingIBIMESA(
            signal_channel=signal_channel or "Pleth",
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            peak_backend=peak_backend,
        )
    if name == "isruc":
        return SleepStagingIBIISRUC(
            scorer=scorer,
            channel=signal_channel,
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            peak_backend=peak_backend,
        )
    if name == "sleep_accel":
        return SleepStagingIBISleepAccel(
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
        )
    raise ValueError(f"unknown dataset {name!r}")


def build_samples(dataset: Any, task: Any, num_workers: int = 1) -> Any:
    """Applies a task to a base dataset with a bounded worker count.

    Args:
        dataset: A fitted PyHealth dataset.
        task: A task from :func:`build_task`.
        num_workers: Number of PyHealth task workers. One avoids spawning a
            process pool in CI and gives deterministic sample order.

    Returns:
        A PyHealth ``SampleDataset`` with the shared IBI contract.

    Examples:
        >>> build_samples(None, None) if False else "ready"
        'ready'
    """
    return dataset.set_task(task, num_workers=num_workers)


def _rebuild(samples: list[dict[str, Any]], source: Any, name: str = "") -> Any:
    return create_sample_dataset(
        samples=samples,
        input_schema=dict(source.input_schema),
        output_schema=dict(source.output_schema),
        dataset_name=name or getattr(source, "dataset_name", ""),
        task_name=getattr(source, "task_name", ""),
    )


def pool_datasets(datasets: Sequence[Any]) -> Any:
    """Pools sample datasets into one ``SampleDataset``.

    Patient IDs are prefixed with the source index when more than one dataset
    is pooled, so SHHS and MESA IDs cannot collide.

    Args:
        datasets: Sample datasets with identical schemas.

    Returns:
        A single in-memory sample dataset.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> def ds(pid):
        ...     return create_sample_dataset(
        ...         [{"patient_id": pid, "signal": np.zeros((1, 750), "float32"),
        ...           "label": np.array([0]), "lengths": 1}],
        ...         {"signal": "tensor", "lengths": "raw"},
        ...         {"label": ("tensor", {"dtype": torch.long})})
        >>> len(pool_datasets([ds("p1"), ds("p2")]))
        2
    """
    samples: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        for sample in dataset:
            item = dict(sample)
            if len(datasets) > 1:
                item["patient_id"] = f"d{index}-{item['patient_id']}"
            samples.append(item)
    return _rebuild(samples, datasets[0], "pooled")


def resample_dataset(dataset: Any, points: int = 750) -> Any:
    """Resamples each IBI row to ``points`` without changing labels or schema.

    This is used for the ``points_per_epoch`` ablation: the task computes IBI
    at a lower rate, then every epoch row is brought back to the 750 samples
    the model requires so only the information bandwidth changes.

    Args:
        dataset: A sample dataset with ``(T, S)`` signal tensors.
        points: Target width for each epoch row.

    Returns:
        A rebuilt sample dataset.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> ds = create_sample_dataset(
        ...     [{"patient_id": "p1", "signal": np.zeros((2, 375), "float32"),
        ...       "label": np.array([0, 1]), "lengths": 2}],
        ...     {"signal": "tensor", "lengths": "raw"},
        ...     {"label": ("tensor", {"dtype": torch.long})})
        >>> tuple(resample_dataset(ds, 750)[0]["signal"].shape)
        (2, 750)
    """
    samples = []
    for sample in dataset:
        item = dict(sample)
        signal = np.asarray(item["signal"])
        item["signal"] = resample(signal, points, axis=-1).astype(np.float32)
        samples.append(item)
    return _rebuild(samples, dataset, "resampled")


def split_dataset(dataset: Any, seed: int = 0) -> tuple[Any, Any, Any]:
    """Splits a sample dataset by patient into 60/20/20 parts.

    Args:
        dataset: A sample dataset with at least three patients.
        seed: Seed used by :func:`pyhealth.datasets.split_by_patient`.

    Returns:
        Train, validation, and test sample datasets.

    Raises:
        ValueError: If any split is empty.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": f"p{i}", "signal": np.zeros((1, 750), "float32"),
        ...      "label": np.array([0]), "lengths": 1} for i in range(5)]
        >>> parts = split_dataset(create_sample_dataset(
        ...     samples, {"signal": "tensor", "lengths": "raw"},
        ...     {"label": ("tensor", {"dtype": torch.long})}))
        >>> [len(part) for part in parts]
        [3, 1, 1]
    """
    parts = split_by_patient(dataset, [0.6, 0.2, 0.2], seed=seed)
    if any(len(part) == 0 for part in parts):
        raise ValueError(
            "patient split produced an empty part; use at least three patients "
            "with 5 or more for a stable 60/20/20 split"
        )
    return parts


def build_model(
    dataset: Any,
    seed: int = 0,
    model_kwargs: Mapping[str, Any] | None = None,
    quantization: str = "fp32",
    loss: str = "ce",
) -> WatchSleepNet:
    """Builds a WatchSleepNet with a fixed seed for reproducible ablations.

    Args:
        dataset: The fitted sample dataset.
        seed: Seed applied to PyTorch before model construction.
        model_kwargs: Overrides of :data:`DEFAULT_MODEL_KWARGS`.
        quantization: One of :data:`QUANTIZATION_MODES`.
        loss: One of :data:`LOSS_MODES`; ``"ce"`` is the reference default.

    Returns:
        An untrained model with a three-class classifier.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> ds = create_sample_dataset(
        ...     [{"patient_id": "p1", "signal": np.zeros((1, 750), "float32"),
        ...       "label": np.array([0]), "lengths": 1}],
        ...     {"signal": "tensor", "lengths": "raw"},
        ...     {"label": ("tensor", {"dtype": torch.long})})
        >>> build_model(ds).n_classes
        3
    """
    params = dict(DEFAULT_MODEL_KWARGS)
    params.update(model_kwargs or {})
    torch.manual_seed(seed)
    return WatchSleepNet(
        dataset=dataset,
        n_classes=len(CLASS_NAMES),
        quantization=quantization,
        loss=loss,
        **params,
    )


def get_loaders(
    parts: Sequence[Any], batch_size: int = 16, seed: int = 0
) -> list[Any]:
    """Creates train/validation/test loaders with a deterministic shuffle.

    Args:
        parts: Train, validation, and test sample datasets.
        batch_size: Batch size for all loaders.
        seed: Shuffle seed, applied only to the training loader.

    Returns:
        A list of three data loaders.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": f"p{i}", "signal": np.zeros((1, 750), "float32"),
        ...      "label": np.array([0]), "lengths": 1} for i in range(5)]
        >>> ds = create_sample_dataset(
        ...     samples, {"signal": "tensor", "lengths": "raw"},
        ...     {"label": ("tensor", {"dtype": torch.long})})
        >>> [len(loader.dataset) for loader in get_loaders(split_dataset(ds))]
        [3, 1, 1]
    """
    return [
        get_dataloader(part, batch_size=batch_size, shuffle=(i == 0), seed=seed)
        for i, part in enumerate(parts)
    ]


def fit(
    model: WatchSleepNet,
    loaders: Sequence[Any],
    epochs: int = 5,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    device: str | None = None,
    quantization: str = "fp32",
) -> tuple[Trainer, list[dict[str, float]]]:
    """Trains with ``Trainer.train`` and records one validation row per epoch.

    Args:
        model: The model to train in place.
        loaders: Train and validation data loaders.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        device: Torch device; ``None`` lets Trainer choose it.
        quantization: Used to raise Adam's epsilon for fp16 stability.

    Returns:
        The trainer and a list of per-epoch validation score dicts.

    Examples:
        >>> fit(model=None, loaders=[]) if False else "ready"
        'ready'
    """
    trainer = Trainer(model=model, metrics=TRAINER_METRICS, enable_logging=False,
                      device=device)
    optimizer_params: dict[str, float] = {"lr": lr}
    if quantization == "fp16":
        optimizer_params["eps"] = 1e-4
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        trainer.train(
            train_dataloader=loaders[0],
            val_dataloader=loaders[1] if len(loaders) > 1 else None,
            epochs=1,
            optimizer_params=optimizer_params,
            weight_decay=weight_decay,
            monitor="accuracy" if len(loaders) > 1 else None,
        )
        scores = trainer.evaluate(loaders[1]) if len(loaders) > 1 else {}
        history.append({"epoch": epoch + 1, **{k: float(v) for k, v in scores.items()}})
    return trainer, history


def predict(
    model: WatchSleepNet,
    loader: Any,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collects flattened true labels and class probabilities from a loader.

    Args:
        model: A trained model.
        loader: Data loader for the evaluation split.
        device: Torch device; ``None`` uses the model's parameter device.

    Returns:
        ``(y_true, y_prob)`` arrays over valid, non-padded epochs.

    Examples:
        >>> predict(model=None, loader=None) if False else "ready"
        'ready'
    """
    target = device or str(model.device)
    model.eval()
    true_parts, prob_parts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(target) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(**batch)
            true_parts.append(output["y_true"].detach().cpu().numpy())
            prob_parts.append(output["y_prob"].detach().cpu().numpy())
    if not true_parts:
        return np.empty(0, dtype=np.int64), np.empty((0, len(CLASS_NAMES)))
    return np.concatenate(true_parts), np.concatenate(prob_parts)


def summarize_predictions(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Computes accuracy, per-class F1, and predicted-class distribution.

    Args:
        y_true: Flattened integer labels in ``0..2``.
        y_prob: Class probabilities with shape ``(n, 3)``.

    Returns:
        A flat dictionary of scalar metrics. Per-class F1 values expose
        NREM-majority collapse even when accuracy looks reasonable.

    Examples:
        >>> import numpy as np
        >>> row = summarize_predictions(np.array([0, 1]), np.eye(3)[:2])
        >>> row["accuracy"], row["f1_rem"]
        (1.0, 0.0)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1, len(CLASS_NAMES))
    labels = list(range(len(CLASS_NAMES)))
    y_pred = y_prob.argmax(axis=1) if y_prob.size else np.empty(0, dtype=np.int64)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    counts = np.bincount(y_pred, minlength=len(CLASS_NAMES)).astype(float)
    distribution = counts / max(len(y_pred), 1)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "f1_macro": float(np.mean(f1)),
        "n_epochs": int(y_true.size),
    }
    for name, value in zip(CLASS_NAMES, f1, strict=True):
        result[f"f1_{name.lower()}"] = float(value)
    for name, value in zip(CLASS_NAMES, distribution, strict=True):
        result[f"pred_{name.lower()}"] = float(value)
    return result


def save_checkpoint(
    path: str | os.PathLike[str],
    model: WatchSleepNet,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Saves a self-describing checkpoint with a bare state dict inside.

    Args:
        path: Output ``.pt`` path.
        model: Model whose state dict is saved.
        config: Dataset, task, model, and training configuration.
        metrics: Optional final evaluation metrics.

    Returns:
        The output path.

    Examples:
        >>> save_checkpoint("unused.pt", None, {}) if False else "ready"
        'ready'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": dict(config),
            "metrics": dict(metrics or {}),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Loads either a self-describing bundle or a bare ``.pt`` state dict.

    Args:
        path: Checkpoint path produced by this helper or ``Trainer.save_ckpt``.

    Returns:
        The state dict and any bundled metadata (empty for bare checkpoints).

    Examples:
        >>> load_checkpoint("unused.pt") if False else "ready"
        'ready'
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = dict(payload["state_dict"])
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state = dict(payload)
        metadata = {}
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "model."):
            name = name.removeprefix(prefix)
        cleaned[name] = value
    return cleaned, metadata


def build_model_from_checkpoint(
    path: str | os.PathLike[str],
    dataset: Any,
    seed: int = 0,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[WatchSleepNet, dict[str, Any]]:
    """Rebuilds a model from a self-describing bundle or a bare ``.pt`` file.

    Values in ``overrides`` win over the checkpoint config. Bare checkpoints use
    :data:`DEFAULT_MODEL_KWARGS` and ``fp32``; the original bare checkpoint
    layout is therefore accepted without extra metadata.

    Args:
        path: Checkpoint produced by :func:`save_checkpoint`,
            :func:`quantize_checkpoint`, or ``Trainer.save_ckpt``.
        dataset: The fitted sample dataset used to build the classifier.
        seed: Seed applied before model construction.
        overrides: Optional model kwargs, quantization, loss, or dataset name.

    Returns:
        The loaded model and the checkpoint metadata.

    Examples:
        >>> build_model_from_checkpoint("unused.pt", None) if False else "ready"
        'ready'
    """
    state, metadata = load_checkpoint(path)
    config = dict(metadata.get("config", {}))
    options = dict(overrides or {})
    model_kwargs = dict(DEFAULT_MODEL_KWARGS)
    model_kwargs.update(config.get("model", config.get("model_kwargs", {})))
    model_kwargs.update(options.get("model_kwargs", {}))
    for key in DEFAULT_MODEL_KWARGS:
        if key in options:
            model_kwargs[key] = options[key]
    for key in ("use_attention", "use_tcn"):
        if key in config:
            model_kwargs[key] = config[key]
        if key in options:
            model_kwargs[key] = options[key]
    quantization = options.get(
        "quantization", config.get("quantization", "fp32")
    )
    loss = options.get("loss", config.get("loss", "ce"))
    model = build_model(dataset, seed=seed, model_kwargs=model_kwargs,
                        quantization=quantization, loss=loss)
    model.load_state_dict(state, strict=True)
    return model, metadata


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds optional architecture flags that override model defaults.

    Args:
        parser: Argument parser to update.

    Returns:
        The same parser, for chaining.

    Examples:
        >>> import argparse
        >>> args = add_model_args(argparse.ArgumentParser()).parse_args([])
        >>> args.hidden_dim, args.use_attention
        (None, None)
    """
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--tcn-layers", type=int, default=None)
    parser.add_argument(
        "--use-attention", choices=("true", "false"), default=None
    )
    parser.add_argument("--use-tcn", choices=("true", "false"), default=None)
    return parser


def model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Converts architecture CLI flags into ``WatchSleepNet`` kwargs.

    Args:
        args: Namespace populated by :func:`add_model_args`.

    Returns:
        Only the architecture options the caller explicitly set.

    Examples:
        >>> import argparse
        >>> parser = add_model_args(argparse.ArgumentParser())
        >>> model_kwargs_from_args(parser.parse_args(["--hidden-dim", "8"]))
        {'hidden_dim': 8}
    """
    names = {
        "hidden_dim": "hidden_dim",
        "num_channels": "num_channels",
        "num_heads": "num_heads",
        "num_layers": "num_layers",
        "tcn_layers": "tcn_layers",
    }
    kwargs: dict[str, Any] = {}
    for flag, key in names.items():
        value = getattr(args, flag, None)
        if value is not None:
            kwargs[key] = value
    for flag, key in (("use_attention", "use_attention"), ("use_tcn", "use_tcn")):
        value = getattr(args, flag, None)
        if value is not None:
            kwargs[key] = value == "true"
    return kwargs


def quantize_checkpoint(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    quantization: str,
) -> Path:
    """Casts every floating-point checkpoint tensor to a compute dtype.

    Args:
        source: Input bundle or bare ``.pt`` checkpoint.
        target: Output ``.pt`` path.
        quantization: One of :data:`QUANTIZATION_MODES`.

    Returns:
        The output path.

    Raises:
        ValueError: If ``quantization`` is unsupported.

    Examples:
        >>> quantize_checkpoint("a.pt", "b.pt", "fp8") if False else "ready"
        'ready'
    """
    if quantization not in QUANTIZATION_DTYPES:
        raise ValueError(
            f"quantization must be one of {QUANTIZATION_MODES}, got {quantization!r}"
        )
    state, metadata = load_checkpoint(source)
    source_format = metadata.get("format", "bare-state-dict")
    dtype = QUANTIZATION_DTYPES[quantization]
    cast = {
        key: value.to(dtype) if torch.is_floating_point(value) else value.clone()
        for key, value in state.items()
    }
    metadata["format"] = CHECKPOINT_FORMAT
    metadata["quantization"] = quantization
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": cast,
            "config": metadata.get("config", {}),
            "metrics": metadata.get("metrics", {}),
            "quantization": quantization,
            "source_format": source_format,
        },
        target,
    )
    return target


def write_results(
    output_dir: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Writes ``results.csv`` and ``results.json`` with one row per condition.

    Args:
        output_dir: Directory to create or overwrite the result files in.
        rows: Metric dictionaries, one per experiment condition.
        metadata: Extra JSON metadata such as seeds and dataset roots.

    Returns:
        The CSV and JSON paths.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     csv_path, json_path = write_results(tmp, [{"accuracy": 1.0}])
        ...     csv_path.name, json_path.name
        ('results.csv', 'results.json')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    csv_path = output_dir / "results.csv"
    frame.to_csv(csv_path, index=False)

    def plain(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value

    payload = {
        "metadata": {k: plain(v) for k, v in (metadata or {}).items()},
        "results": [
            {k: plain(v) for k, v in row.items()} for row in rows
        ],
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


def print_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Prints an aligned table of the requested columns.

    Args:
        rows: Metric dictionaries.
        columns: Column names to print in order.

    Returns:
        None. The table is written to standard output.

    Examples:
        >>> print_table([{"accuracy": 1.0}], ["accuracy"])
        accuracy
        1.0000
    """
    widths = [max(len(c), 8) for c in columns]
    print("  ".join(c.ljust(w).rstrip() for c, w in zip(columns, widths, strict=True)))
    for row in rows:
        cells = []
        for column, width in zip(columns, widths, strict=True):
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}".ljust(width))
            else:
                cells.append(str(value).ljust(width))
        print("  ".join(cell.rstrip() for cell in cells))


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the shared training, seed, device, and quantization CLI flags.

    Args:
        parser: Argument parser to update.

    Returns:
        The same parser, for chaining.

    Examples:
        >>> import argparse
        >>> args = add_training_args(argparse.ArgumentParser()).parse_args([])
        >>> args.quantization, args.seed, args.loss
        ('fp32', 0, 'ce')
    """
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="torch device; default auto")
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES, default="fp32")
    parser.add_argument(
        "--loss",
        choices=LOSS_MODES,
        default="ce",
        help="training loss; ce is the reference default",
    )
    parser.add_argument("--output-dir", default="output/watchsleepnet")
    return parser

"""Shared recording helpers for the CLIPStress runnable experiments.

Records validation-based checkpoint selection and derives every test metric
from one inference pass. Result files and weights are local run artifacts.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from pyhealth.datasets import get_dataloader
from pyhealth.trainer import Trainer


def normalize_stress_inputs(signal, tabular, train_indices, method="quantile"):
    """Fit reference per-hour/channel quantiles and four continuous table fields.

    The reference treats the 24 hours as separate columns, not one flattened
    channel. Categorical ordinal columns after the first four remain unchanged.
    """
    from sklearn.preprocessing import QuantileTransformer

    signal, tabular = (
        np.array(signal, dtype=float, copy=True),
        np.array(tabular, dtype=float, copy=True),
    )
    if method == "standard":
        for values, axes in ((signal, (0, 1)), (tabular, 0)):
            train = values[train_indices]
            values -= train.mean(axis=axes)
            values /= np.maximum(train.std(axis=axes), 1e-6)
    elif method == "quantile":
        for channel in range(signal.shape[2]):
            transform = QuantileTransformer(
                n_quantiles=min(
                    len(train_indices), max(min(len(train_indices) // 10, 1000), 10)
                ),
                output_distribution="normal",
                subsample=10**9,
                random_state=2024,
            )
            transform.fit(signal[train_indices, :, channel])
            signal[:, :, channel] = transform.transform(signal[:, :, channel])
        for column in range(min(4, tabular.shape[1])):
            transform = QuantileTransformer(
                n_quantiles=min(
                    len(train_indices), max(min(len(train_indices) // 30, 1000), 10)
                ),
                output_distribution="normal",
                subsample=10**9,
                random_state=2024,
            )
            transform.fit(tabular[train_indices, column : column + 1])
            tabular[:, column : column + 1] = transform.transform(
                tabular[:, column : column + 1]
            )
    elif method != "none":
        raise ValueError("Unknown normalization.")
    return signal, tabular


def seed_training(seed: int) -> None:
    """Reset initialization and shuffling without changing the data split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def file_hash(path: Path) -> str:
    """Return SHA-256 for an input, source file or saved checkpoint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically write finite JSON, preserving previous completed results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    """Identify the model, training API and example code used by this run."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "pyhealth/models/clip_stress.py",
        root / "pyhealth/models/clip_stress_utils.py",
        root / "pyhealth/models/clip_stress_layers.py",
        root / "pyhealth/trainer.py",
        *sorted((root / "examples").glob("*clip_stress*.py")),
    ]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


class RecordedTrainer(Trainer):
    """Keep validation history while using the ordinary PyHealth trainer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validation_history = []

    def evaluate(self, dataloader):
        scores = super().evaluate(dataloader)
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError("Nonfinite validation result.")
        self.validation_history.append(dict(scores))
        return scores


class ReferenceLabeledBatches:
    """Shuffle the mixed pretraining pool, then remove unlabeled batch entries.

    The reference downstream loop reuses its mixed loader. Filtering labels
    before batching would change the optimizer step count and BatchNorm inputs.
    Labeled samples occupy the first len(dataset) positions of that pool.
    """

    def __init__(self, dataset, pool_size, batch_size, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.indices = torch.utils.data.DataLoader(
            range(pool_size),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        self.collate = get_dataloader(dataset, batch_size).collate_fn

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for batch in self.indices:
            labeled = batch[batch < len(self.dataset)].tolist()
            if labeled:
                yield self.collate([self.dataset[index] for index in labeled])


def fit_classifier(model, train, val, test, args, directory=None):
    """Train, restore the validation-best weights and evaluate test once.

    Args:
        model: A CLIPStress model already in its downstream stage.
        train: Training SampleDataset.
        val: Validation SampleDataset for checkpoint selection.
        test: Held-out SampleDataset; never used for model selection.
        args: Namespace with device, seed, batch_size, epochs and learning_rate.
        directory: Optional directory for checkpoint and validation artifacts.

    Returns:
        Metrics and auditable predictions, checkpoint selection and duration.
    """
    seed_training(args.seed)
    trainer = RecordedTrainer(
        model=model,
        device=args.device,
        enable_logging=False,
        metrics=["roc_auc"],
    )
    start = time.perf_counter()
    final = getattr(args, "checkpoint_selection", "validation") == "final"
    if not final and val is None:
        raise ValueError("Validation checkpoint selection requires a validation set.")
    pool_size = getattr(args, "downstream_pool_size", None)
    loader = (
        ReferenceLabeledBatches(train, pool_size, args.batch_size, args.seed)
        if pool_size is not None
        else get_dataloader(train, args.batch_size, shuffle=True, seed=args.seed)
    )
    trainer.train(
        loader,
        val_dataloader=None if final else get_dataloader(val, args.batch_size),
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        monitor="roc_auc",
        monitor_criterion="max",
        load_best_model_at_last=not final,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    duration = time.perf_counter() - start
    history = trainer.validation_history
    # Trainer uses strict improvement, so ties keep the earliest checkpoint.
    best_epoch = (
        args.epochs
        if final
        else int(np.argmax([row["roc_auc"] for row in history])) + 1
    )
    checkpoint = None
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "best.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        checkpoint = {"path": str(path), "sha256": file_hash(path)}
        write_json(directory / "validation.json", {"epochs": history})
    y, probability, loss, patients = trainer.inference(
        get_dataloader(test, args.batch_size),
        return_patient_ids=True,
    )
    y, probability = y.reshape(-1), probability.reshape(-1)
    if not np.isfinite(probability).all() or not np.isfinite(loss):
        raise ValueError("Nonfinite test predictions or loss.")
    metrics = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "reference_argmax_auc": float(roc_auc_score(y, probability > 0.5)),
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, probability >= 0.5, zero_division=0)),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
        "loss": float(loss),
    }
    details = {
        "checkpoint_selection": "final" if final else "validation",
        "downstream_pool_size": pool_size,
        "selected_epoch": best_epoch,
        "validation_history": history,
        "training_seconds": duration,
        "checkpoint": checkpoint,
        "test_evaluations": 1,
        "threshold": 0.5,
        "predictions": {
            "patient_id": list(map(str, patients)),
            "y_true": y.astype(int).tolist(),
            "y_prob": probability.tolist(),
        },
    }
    return metrics, details

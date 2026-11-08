"""Run CLIPStress pretraining, linear probing, fine-tuning and ablations.

Contributor: Siqi Zhu (UIUC NetID: siqizhu4).
Paper: Contrastive Pretraining for Stress Detection with Multimodal Wearable
Sensor Data and Surveys (CHIL 2025).

Paper: https://proceedings.mlr.press/v287/yang25a.html

From the repository root: python examples/clip_stress.py --ablation all
The default uses synthetic data and synthetic text vectors for an offline
integration demonstration, not a reproduction of the paper's accuracy.
Use --data prepared.npz for real windows with cached BERT embeddings.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CLIPStress
from pyhealth.trainer import Trainer


class _FullBatches:
    """Drop partial contrastive batches while retaining PyHealth collation."""

    def __init__(self, dataset, batch_size: int, seed: int):
        self.batch_size = min(batch_size, len(dataset))
        self.dataset = dataset
        self.loader = get_dataloader(dataset, self.batch_size, shuffle=True, seed=seed)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return itertools.islice(self.loader, len(self))


def synthetic_data(seed: int) -> dict[str, np.ndarray]:
    """Create 12 subjects with feature-derived, non-BERT text placeholders."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(48, 12, 3)).astype(np.float32)
    tabular = rng.normal(size=(48, 4)).astype(np.float32)
    features = np.concatenate([signal.mean(1), signal.std(1), tabular], axis=1)
    return {
        "signal": signal,
        "tabular": tabular,
        "text_embedding": (features @ rng.normal(size=(10, 16))).astype(np.float32),
        "label": (signal.mean((1, 2)) + tabular[:, 0] > 0).astype(np.int64),
        "patient_id": np.repeat(np.arange(12), 4).astype(str),
    }


def prepare_datasets(
    data: dict,
    seed: int,
    label_fraction: float,
    require_text: bool = True,
    subject_splits: dict[str, list[str]] | None = None,
) -> tuple:
    """Split subjects before fitting normalization or label processors."""
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1].")
    required = {"signal", "tabular", "label", "patient_id"}
    if require_text or "text_embedding" in data:
        required.add("text_embedding")
    if not required.issubset(data):
        raise ValueError(f"NPZ must contain {sorted(required)}.")
    for key, rank in (("signal", 3), ("tabular", 2), ("text_embedding", 2)):
        if key not in required:
            continue
        values = np.asarray(data[key])
        if (
            values.ndim != rank
            or not all(values.shape)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{key} must be a nonempty finite array of rank {rank}.")
    if np.asarray(data["label"]).ndim != 1:
        raise ValueError("label must be a one-dimensional array.")
    if np.asarray(data["patient_id"]).ndim != 1:
        raise ValueError("patient_id must be a one-dimensional array.")
    count = len(data["label"])
    if any(len(data[k]) != count for k in required):
        raise ValueError("All NPZ arrays must have the same number of samples.")
    if not np.isin(data["label"], [-1, 0, 1]).all():
        raise ValueError("Labels must be 0/1, or -1 for unlabeled windows.")
    identifiers = np.asarray(data["patient_id"])
    if any(
        not isinstance(value, (str, int))
        or isinstance(value, bool)
        or not str(value).strip()
        for value in identifiers.tolist()
    ):
        raise ValueError("patient_id must contain nonempty strings or integers.")
    data = dict(data, patient_id=identifiers.astype(str))
    rng = np.random.default_rng(seed)
    patients = np.unique(data["patient_id"])
    if len(patients) < 5:
        raise ValueError("At least five subjects are needed for disjoint splits.")
    rng.shuffle(patients)
    if subject_splits is None:
        n_test = max(1, len(patients) // 5)
        groups = (
            patients[2 * n_test :],
            patients[:n_test],
            patients[n_test : 2 * n_test],
        )
    else:
        names = ("train", "validation", "test")
        if not isinstance(subject_splits, dict) or set(subject_splits) != set(names):
            raise ValueError("subject_splits requires train, validation and test.")
        groups = tuple(np.asarray(subject_splits[name]) for name in names)
        if any(group.ndim != 1 or not len(group) for group in groups):
            raise ValueError("Each subject split must be a nonempty list.")
        combined = np.concatenate(groups)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("Subject splits must be disjoint without duplicates.")
        if set(combined) != set(patients):
            raise ValueError("Subject splits must cover every subject exactly once.")
    indices = [np.flatnonzero(np.isin(data["patient_id"], group)) for group in groups]
    data = {k: np.array(data[k], copy=True) for k in required}
    for key in ("signal", "tabular"):
        values = data[key][indices[0]]
        axes = (0, 1) if key == "signal" else 0
        mean, std = values.mean(axis=axes), values.std(axis=axes)
        data[key] = ((data[key] - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    labeled_indices = [idx[data["label"][idx] >= 0] for idx in indices]
    # Retain at least one sample from each class when subsampling training labels.
    labeled_train = np.concatenate(
        [
            rng.permutation(
                labeled_indices[0][data["label"][labeled_indices[0]] == label]
            )[
                : max(
                    1,
                    int(
                        np.ceil(
                            label_fraction
                            * np.sum(data["label"][labeled_indices[0]] == label)
                        )
                    ),
                )
            ]
            for label in (0, 1)
        ]
    )
    if any(
        len(np.unique(data["label"][idx])) != 2
        for idx in (labeled_train, *labeled_indices[1:])
    ):
        raise ValueError(
            "Each labeled split needs both classes; try another split seed."
        )

    def samples(idx: np.ndarray, labels: bool) -> list[dict]:
        keys = required if labels else required - {"label"}
        return [
            {k: data[k][i].item() if data[k][i].ndim == 0 else data[k][i] for k in keys}
            for i in idx
        ]

    schema = {k: "tensor" for k in sorted(required - {"label", "patient_id"})}
    train = create_sample_dataset(
        samples(labeled_train, True), schema, {"label": "binary"}
    )
    pretrain = create_sample_dataset(
        samples(indices[0], False), schema, {}, input_processors=train.input_processors
    )
    held_out = [
        create_sample_dataset(
            samples(idx, True),
            schema,
            {"label": "binary"},
            input_processors=train.input_processors,
            output_processors=train.output_processors,
        )
        for idx in labeled_indices[1:]
    ]
    split_info = {
        name: list(map(str, group))
        for name, group in zip(("train", "validation", "test"), groups)
    }
    return train, pretrain, *held_out, split_info


def run(
    args: argparse.Namespace,
    data: dict | None = None,
    subject_splits: dict[str, list[str]] | None = None,
) -> dict:
    """Evaluate fixed recipes on identical subject splits and initialization seeds."""
    torch.set_num_threads(args.threads)
    validate_args(args)
    if data is None:
        if args.data:
            with np.load(args.data, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
        else:
            data = synthetic_data(args.seed)
    train, pretrain, val, test, splits = prepare_datasets(
        data,
        args.seed,
        args.label_fraction,
        require_text=args.ablation != "supervised",
        subject_splits=subject_splits,
    )
    batch_size = args.batch_size
    val_loader = get_dataloader(val, batch_size=batch_size)
    test_loader = get_dataloader(test, batch_size=batch_size)
    variants = (
        ["multimodal", "supervised", "signal", "tabular"]
        if args.ablation == "all"
        else [args.ablation]
    )
    results = {
        "data": str(args.data) if args.data else "SYNTHETIC (not paper results)",
        "seed": args.seed,
        "subjects": splits,
        "label_fraction": args.label_fraction,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("torch", "numpy", "scipy", "transformers", "pyhealth")
            },
        },
        "configuration": {
            key: getattr(args, key)
            for key in (
                "batch_size",
                "learning_rate",
                "hidden_dim",
                "projection_dim",
                "num_layers",
                "num_heads",
                "dropout",
                "temperature",
                "device",
                "threads",
            )
        },
        "sample_counts": {
            "train_labeled": len(train),
            "train_pretrain": len(pretrain),
            "validation": len(val),
            "test": len(test),
        },
        "results": {},
    }
    if "metadata" in data:
        results["preprocessing"] = json.loads(str(data["metadata"].item()))
    if args.data:
        with args.data.open("rb") as stream:
            results["data_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    for variant in variants:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        config = {
            "signal_key": None if variant == "tabular" else "signal",
            "tabular_key": None if variant == "signal" else "tabular",
            **{
                key: getattr(args, key)
                for key in (
                    "hidden_dim",
                    "projection_dim",
                    "num_layers",
                    "num_heads",
                    "dropout",
                    "temperature",
                )
            },
        }
        if variant != "supervised":
            pretrain_loader = _FullBatches(pretrain, batch_size, args.seed)
            pretrained_model = CLIPStress(pretrain, **config)
            trainer = Trainer(
                model=pretrained_model, device=args.device, enable_logging=False
            )
            trainer.train(
                pretrain_loader,
                epochs=args.pretrain_epochs,
                optimizer_params={"lr": args.learning_rate},
                load_best_model_at_last=False,
            )
        # Downstream heads start from the same seed for each recipe.
        torch.manual_seed(args.seed)
        model = CLIPStress(train, **config, stage="finetune")
        if variant != "supervised":
            model.load_pretrained_encoder(pretrained_model)
        pretrained = copy.deepcopy(model.state_dict())
        stages = (
            ["finetune"] if variant == "supervised" else ["linear_probe", "finetune"]
        )
        for stage in stages:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model.load_state_dict(pretrained)
            model.set_stage(stage)
            train_loader = get_dataloader(
                train, batch_size=batch_size, shuffle=True, seed=args.seed
            )
            trainer = Trainer(
                model=model,
                device=args.device,
                enable_logging=False,
                metrics=["roc_auc", "accuracy", "f1"],
            )
            trainer.train(
                train_loader,
                val_dataloader=val_loader,
                epochs=args.epochs,
                optimizer_params={"lr": args.learning_rate},
                monitor="roc_auc",
                monitor_criterion="max",
            )
            results["results"][f"{variant}/{stage}"] = trainer.evaluate(test_loader)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Shared training options for the example and cross-validation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--ablation",
        choices=["all", "multimodal", "supervised", "signal", "tabular"],
        default="multimodal",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training settings before constructing datasets."""
    if (
        not 0 < args.label_fraction <= 1
        or min(args.epochs, args.pretrain_epochs, args.threads) < 1
    ):
        raise ValueError("Use positive epochs/threads and label-fraction in (0, 1].")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least two for contrastive training.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if min(args.hidden_dim, args.projection_dim, args.num_layers, args.num_heads) < 1:
        raise ValueError("Model dimensions and layer/head counts must be positive.")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden-dim must be divisible by num-heads.")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    result = run(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

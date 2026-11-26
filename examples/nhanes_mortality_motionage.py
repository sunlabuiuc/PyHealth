"""Train and evaluate MotionAge on NHANES mortality prediction.

Reference:
    Song et al., MotionAge: A Deep Learning Framework for Biological
    Age Prediction from Wearable Activity, MLHC 2026.
    https://openreview.net/pdf?id=xGw1o9acWa
    https://github.com/jackie-jiaqi-yin/MotionAge

Contributors:
    Inoh Pak (inohpak2), Yejin Kim (yejink6).

Usage:
    # Synthetic smoke test without downloading NHANES.
    python examples/nhanes_mortality_motionage.py --synthetic --epochs 1

    # Compare binary and continuous wear pooling.
    python examples/nhanes_mortality_motionage.py --synthetic --ablation

    # Model ablation: pooling and hidden size, one change at a time.
    python examples/nhanes_mortality_motionage.py --synthetic --model-ablation

    # Real NHANES experiment.
    python examples/nhanes_mortality_motionage.py --root /path/to/nhanes

    # Reproduce the experimental configuration.
    python examples/nhanes_mortality_motionage.py --root /path/to/nhanes \
        --ablation --sample-fraction 0.25 --seed 42

    # Real-data model ablation with the reported configuration.
    python examples/nhanes_mortality_motionage.py --root /path/to/nhanes \
        --model-ablation --sample-fraction 0.25 --seed 42

Experimental results:
    NHANES 2003-2006, 25% stratified participant subset (2,129 participants).
    Train/validation/test: 1,490/319/320 participants.
    Test set: 592 windows, including 21 positive participants.
    Wearable-only GRU, hidden dimension 32, one recurrent layer,
    10 epochs, batch size 64, AdamW, learning rate 0.001,
    weight decay 0.01, and seed 42.

    Window-level results:
        Binary:   AUROC 0.858731, AUPRC 0.318581
        Fraction: AUROC 0.864716, AUPRC 0.337056

    Participant-level results:
        Binary:   AUROC 0.856187, AUPRC 0.333264, F1 0.377778
        Fraction: AUROC 0.866699, AUPRC 0.360472, F1 0.418605

    Paired bootstrap confidence intervals for fraction minus binary:
        AUROC: 0.010511 [-0.000163, 0.023571]
        AUPRC: 0.027209 [-0.009142, 0.077841]

    Continuous wear weighting improved the observed metrics, but the
    confidence intervals included zero.


    Model ablation results (participant level):
        Same participant split and seed. Pooling and hidden-size
        experiments use binary wear masks.

        Configuration      Params  AUROC     AP     F1    Acc
        Binary baseline      4765 0.8562 0.3333 0.3778 0.8250
        Continuous wear      4765 0.8667 0.3605 0.4186 0.8438
        Mean pooling         4765 0.8530 0.3781 0.3855 0.8406
        Last pooling         4765 0.7476 0.2891 0.2171 0.6844
        Hidden dim 16        1725 0.8366 0.3780 0.3256 0.8187
        Hidden dim 64       15453 0.8718 0.3308 0.3864 0.8313

    Mean pooling achieved the highest AP, while hidden dimension
    64 achieved the highest AUROC with more parameters.
    Last pooling performed worse than the masked baseline.
    Continuous wear weighting improved all four observed metrics
    over the binary baseline without additional parameters.

    AUPRC is reported as average precision (AP).
    Model ablation results are exploratory single-seed comparisons
    without separate statistical significance testing.

Limitations:
    This experiment reproduces the mortality prediction component,
    not the biological age calibration or full cross-validation protocol.
    Temporal day indices represent relative monitoring days (1-7), while
    the paper uses day-of-week indices.
    Results are based on one participant split and one random seed.
    Synthetic results are for pipeline validation, not clinical evidence.
"""

import argparse
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pyhealth.datasets import (
    NHANESDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import MotionAge
from pyhealth.tasks import NHANESMortalityPrediction
from pyhealth.trainer import Trainer

# Model ablation: each condition changes one MotionAge hyperparameter from the
# baseline (masked pooling, --hidden-dim). "mean" averages every epoch,
# including non-wear, and "last" keeps only the final hidden state, which is
# how PyHealth's built-in RNN summarizes a sequence. The data split, wear
# representation, seed, and training settings stay fixed.
MODEL_ABLATION_METRICS = ("roc_auc", "pr_auc", "f1", "accuracy")


def model_ablation_conditions(
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, Any]]]:
    """List the model ablation conditions relative to the baseline."""
    return [
        ("baseline", {}),
        ("pooling=mean", {"pooling": "mean"}),
        ("pooling=last", {"pooling": "last"}),
        (f"hidden_dim={args.hidden_dim // 2}", {"hidden_dim": args.hidden_dim // 2}),
        (f"hidden_dim={args.hidden_dim * 2}", {"hidden_dim": args.hidden_dim * 2}),
    ]


def set_seed(seed: int) -> None:
    """Reset random generators for controlled experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_samples(
    num_patients: int = 80,
    windows_per_patient: int = 2,
    seq_len: int = 96,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic activity windows for pipeline validation."""
    rng = np.random.default_rng(seed)
    samples = []

    for p in range(num_patients):
        label = int(p % 5 == 0)
        level = 0.4 if label else 0.7

        static = np.array(
            [
                rng.normal(65 if label else 55, 10),
                float(p % 2),
                rng.normal(28, 4),
                np.nan if p % 7 == 0 else rng.normal(125, 15),
                rng.normal(75, 10),
            ],
            dtype=np.float32,
        )

        for w in range(windows_per_patient):
            wear_minutes = rng.choice(
                [0, 1, 2, 3, 4, 5],
                size=seq_len,
                p=[0.25, 0.01, 0.01, 0.01, 0.02, 0.70],
            )

            wear_fraction = wear_minutes.astype(np.float32) / 5
            intensity = (level + 0.3 * rng.random(seq_len)) * (wear_minutes > 0)

            epochs = np.arange(seq_len) + w * seq_len

            samples.append(
                {
                    "patient_id": f"seqn{p}",
                    "window_id": w + 1,
                    "intensity": intensity.astype(np.float32),
                    "hour": ((epochs // 12) % 24).astype(np.float32),
                    "day": ((epochs // 288) % 7).astype(np.float32),
                    "wear_mask": wear_fraction,
                    "static": static.copy(),
                    "label": label,
                }
            )

    return samples


def build_real_dataset(
    root: str,
    include_static: bool,
) -> list[dict[str, Any]]:
    """Extract MotionAge windows from NHANES participant records."""
    dataset = NHANESDataset(root=root)

    task = NHANESMortalityPrediction(
        root=root,
        feature_format="motionage",
        wear_representation="fraction",
        include_static=include_static,
    )

    samples = []

    for patient_id in sorted(dataset.unique_patient_ids):
        patient = dataset.get_patient(patient_id)
        samples.extend(task(patient))

    return samples


def split_samples(
    samples: list[dict[str, Any]],
    sample_fraction: float,
    seed: int,
) -> tuple[list, list, list]:
    """Create stratified participant-level train/validation/test splits."""
    labels = {}

    for sample in samples:
        pid = str(sample["patient_id"])
        label = int(sample["label"])

        if pid in labels and labels[pid] != label:
            raise ValueError(f"Inconsistent labels for participant {pid}")

        labels[pid] = label

    patient_ids = sorted(labels)

    if sample_fraction < 1.0:
        patient_ids, _ = train_test_split(
            patient_ids,
            train_size=sample_fraction,
            stratify=[labels[p] for p in patient_ids],
            random_state=seed,
        )

    train_ids, remaining_ids = train_test_split(
        patient_ids,
        test_size=0.30,
        stratify=[labels[p] for p in patient_ids],
        random_state=seed,
    )

    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        stratify=[labels[p] for p in remaining_ids],
        random_state=seed,
    )

    splits = []

    for ids in (train_ids, val_ids, test_ids):
        selected = set(ids)
        splits.append(
            [sample for sample in samples if str(sample["patient_id"]) in selected]
        )

    return tuple(splits)


def participant_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: list[str],
) -> dict[str, float]:
    """Evaluate mean window predictions at participant level."""
    risk = MotionAge.aggregate_participant_risk(torch.as_tensor(y_prob), patient_ids)

    labels = {str(pid): int(y) for pid, y in zip(patient_ids, y_true.reshape(-1))}

    pids = sorted(risk)
    truth = np.array([labels[p] for p in pids])
    probs = np.array([risk[p] for p in pids])
    pred = (probs >= 0.5).astype(int)

    return {
        "participants": len(pids),
        "positive_patients": int(truth.sum()),
        "roc_auc": roc_auc_score(truth, probs),
        "pr_auc": average_precision_score(truth, probs),
        "f1": f1_score(truth, pred, zero_division=0),
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "recall": recall_score(truth, pred, zero_division=0),
    }


def prepare_samples(
    samples: list[dict[str, Any]],
    representation: str,
    include_static: bool,
) -> list[dict[str, Any]]:
    """Change only pooling weights between ablation conditions."""
    prepared = []

    for sample in samples:
        item = dict(sample)

        if representation == "binary":
            item["wear_mask"] = (np.asarray(sample["wear_mask"]) > 0).astype(np.float32)

        if not include_static:
            item.pop("static", None)

        prepared.append(item)

    return prepared


def run_experiment(
    splits: tuple[list, list, list],
    representation: str,
    args: argparse.Namespace,
    label: str | None = None,
) -> dict[str, float]:
    """Train one MotionAge configuration and evaluate the test set."""
    set_seed(args.seed)
    include_static = args.static == "full"

    train, val, test = [
        prepare_samples(part, representation, include_static) for part in splits
    ]

    positives = sum(int(s["label"]) for s in train)
    negatives = len(train) - positives

    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both classes.")

    pos_weight = negatives / positives

    input_schema = {
        "intensity": "tensor",
        "hour": "tensor",
        "day": "tensor",
        "wear_mask": "tensor",
    }

    if include_static:
        input_schema["static"] = (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        )

    output_schema = {"label": ("binary", {"pos_weight": pos_weight})}

    # Fit processors using training samples only.
    train_ds = create_sample_dataset(
        samples=train,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="nhanes_train",
    )

    # Reuse fitted processors for validation and test samples.
    def transform(samples: list[dict[str, Any]]):
        return create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            input_processors=train_ds.input_processors,
            output_processors=train_ds.output_processors,
            dataset_name="nhanes_eval",
        )

    val_ds = transform(val)
    test_ds = transform(test)

    train_loader = get_dataloader(
        train_ds,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=0,
    )

    val_loader = get_dataloader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    test_loader = get_dataloader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    model = MotionAge(
        train_ds,
        static_key="static" if include_static else None,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
        pooling=args.pooling,
    )

    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=False,
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nExperiment: {label or representation}")
    print(f"Trainable parameters: {num_parameters}")
    print(f"Positive class weight: {pos_weight:.4f}")

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
        weight_decay=0.01,
        monitor="pr_auc",
        monitor_criterion="max",
        load_best_model_at_last=True,
    )

    print("[window-level test]", trainer.evaluate(test_loader))

    y_true, y_prob, _, pids = trainer.inference(test_loader, return_patient_ids=True)

    result = participant_metrics(np.asarray(y_true), np.asarray(y_prob), pids)
    result["parameters"] = num_parameters

    print("[participant-level test]", result)
    return result


def run_model_ablation(
    splits: tuple[list, list, list],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """Train each model ablation condition and print a comparison table."""
    results = {}

    for name, overrides in model_ablation_conditions(args):
        condition_args = argparse.Namespace(**{**vars(args), **overrides})
        results[name] = run_experiment(
            splits,
            args.wear_representation,
            condition_args,
            label=f"model ablation, {name}",
        )

    baseline = results["baseline"]
    print(
        "\nParticipant-level model ablation "
        f"(wear representation: {args.wear_representation}; "
        "values with difference from baseline):"
    )
    header = f"{'condition':<16}{'params':>8}" + "".join(
        f"{metric:>21}" for metric in MODEL_ABLATION_METRICS
    )
    print(header)

    for name, result in results.items():
        cells = "".join(
            f"{result[metric]:>11.4f} ({result[metric] - baseline[metric]:+.4f})"
            for metric in MODEL_ABLATION_METRICS
        )
        print(f"{name:<16}{result['parameters']:>8}{cells}")

    return results


def main() -> None:
    """Run synthetic or real-data MotionAge experiments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--root", type=str)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--model-ablation", action="store_true")
    parser.add_argument(
        "--wear-representation",
        choices=["binary", "fraction"],
        default="binary",
    )
    parser.add_argument("--static", choices=["none", "full"], default="none")
    parser.add_argument(
        "--pooling",
        choices=["masked", "mean", "last"],
        default="masked",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if not args.synthetic and not args.root:
        parser.error("--root is required for real NHANES data.")

    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be in (0, 1].")

    if args.ablation and args.pooling != "masked":
        parser.error("Wear-weight ablation requires masked pooling.")

    if args.model_ablation and args.ablation:
        parser.error("Run --ablation and --model-ablation separately.")

    if args.model_ablation and args.pooling != "masked":
        parser.error("Model ablation uses masked pooling as its baseline.")

    if args.model_ablation and args.hidden_dim < 2:
        parser.error("Model ablation needs --hidden-dim of at least 2.")

    set_seed(args.seed)

    if args.synthetic:
        samples = synthetic_samples(seed=args.seed)
    else:
        samples = build_real_dataset(args.root, include_static=args.static == "full")

    splits = split_samples(samples, args.sample_fraction, args.seed)

    for name, part in zip(("Train", "Validation", "Test"), splits):
        print(
            name,
            "participants:",
            len({s["patient_id"] for s in part}),
            "windows:",
            len(part),
        )

    if args.model_ablation:
        run_model_ablation(splits, args)
        print("Configuration:", vars(args))
        return

    representations = (
        ["binary", "fraction"] if args.ablation else [args.wear_representation]
    )

    results = {}

    for representation in representations:
        results[representation] = run_experiment(splits, representation, args)

    if args.ablation:
        print("\nParticipant-level ablation (fraction - binary):")

        for metric in ("roc_auc", "pr_auc", "f1", "accuracy"):
            difference = results["fraction"][metric] - results["binary"][metric]
            print(f"{metric}: {difference:+.6f}")

    print("Configuration:", vars(args))


if __name__ == "__main__":
    main()

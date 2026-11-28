"""Synthetic DREAMT sleep-staging ablation using the PyHealth MLP.

This example compares 30-second and 60-second epochs using the same
PyHealth MLP architecture and patient-level train/validation/test split.

The script creates a temporary DREAMT-style dataset with 15 synthetic
patients. Each patient has 180 seconds of precomputed IBI data: 60 seconds
each of Wake, NREM, and REM. This makes the
example runnable without downloading DREAMT or installing NeuroKit2.

Run from the repository root:

    python examples/eeg/sleep_staging/dreamt_sleep_staging_mlp.py

For a shorter smoke test:

    python examples/eeg/sleep_staging/dreamt_sleep_staging_mlp.py --epochs 2

The reported metrics are illustrative results on synthetic data only.
They should not be interpreted as clinical results or as evidence that
one epoch duration is better on real DREAMT recordings.
"""

import argparse
import tempfile
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DREAMTDataset, PatientSplit, SampleDataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.tasks import SleepStagingDREAMT
from pyhealth.trainer import Trainer


METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]


def create_synthetic_dreamt(root: Path, seed: int) -> None:
    """Create a small DREAMT-style dataset with synthetic IBI recordings."""
    rng = np.random.default_rng(seed)
    data_dir = root / "data_64Hz"
    data_dir.mkdir(parents=True)

    metadata = []

    samples_per_second = 64
    recording_seconds = 180
    samples_per_patient = samples_per_second * recording_seconds

    for patient_number in range(15):
        patient_id = f"S{patient_number + 1:03d}"
        patient_offset = rng.normal(0.0, 0.015)

        # Wake and NREM have different synthetic IBI distributions.
        wake_ibi = rng.normal(
            0.72 + patient_offset,
            0.015,
            samples_per_second * 60,
        )
        nrem_ibi = rng.normal(
            0.95 + patient_offset,
            0.015,
            samples_per_second * 60,
        )
        rem_ibi = rng.normal(
            1.10 + patient_offset,
            0.015,
            samples_per_second * 60,
        )
        ibi = np.concatenate([wake_ibi, nrem_ibi, rem_ibi]).astype(
            np.float32
        )
        stages = np.array(
            ["W"] * (samples_per_second * 60)
            + ["N2"] * (samples_per_second * 60)
            + ["R"] * (samples_per_second * 60)
        )

        if len(ibi) != samples_per_patient:
            raise RuntimeError("Unexpected synthetic recording length.")

        patient_file = data_dir / f"{patient_id}_whole_df.csv"
        pd.DataFrame(
            {
                "IBI": ibi,
                "Sleep_Stage": stages,
            }
        ).to_csv(patient_file, index=False)

        metadata.append(
            {
                "patient_id": patient_id,
                "age": int(rng.integers(25, 70)),
                "gender": "F" if patient_number % 2 else "M",
                "bmi": 25.0,
                "oahi": 5.0,
                "ahi": 5.0,
                "mean_sao2": 95.0,
                "arousal_index": 10.0,
                "medical_history": "Synthetic",
                "file_64hz": str(patient_file),
                "file_100hz": "",
            }
        )

    pd.DataFrame(metadata).to_csv(root / "dreamt-metadata.csv", index=False)


def patient_ids(dataset: SampleDataset) -> Set[str]:
    """Return the patient IDs represented in a sample dataset."""
    return set(dataset.patient_to_index.keys())


def check_matching_splits(
    first: Tuple[SampleDataset, ...],
    second: Tuple[SampleDataset, ...],
) -> None:
    """Check that both epoch settings use the same patients per split."""
    for name, first_part, second_part in zip(
        ("train", "validation", "test"), first, second
    ):
        first_ids = patient_ids(first_part)
        second_ids = patient_ids(second_part)

        if first_ids != second_ids:
            raise ValueError(f"Patient assignments differ in the {name} split.")

        print(f"{name.capitalize()} patients: {len(first_ids)}")


def run_experiment(
    duration: int,
    splits: Tuple[SampleDataset, ...],
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    """Train and evaluate an MLP for one epoch duration."""
    train_data, val_data, test_data = splits

    if any(len(part) == 0 for part in splits):
        raise ValueError("A data split is empty; use more synthetic patients.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(
        dataset=train_data,
        embedding_dim=128,
        hidden_dim=128,
        n_layers=2,
    )

    train_loader = get_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = get_dataloader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    # Inspect one training batch before fitting the model.
    debug_batch = next(iter(train_loader))
    signal = debug_batch["signal"][0]
    labels = debug_batch["label"]

    print("\\nTraining batch diagnostics:")
    print("  Signal shape:", tuple(signal.shape))
    print("  Labels:", labels.tolist())
    print("  Unique labels:", torch.unique(labels).tolist())
    print("  Signal min:", signal.min().item())
    print("  Signal max:", signal.max().item())
    print("  Signal mean:", signal.mean().item())
    print("  Signal std:", signal.std().item())
    print("  Contains NaN:", torch.isnan(signal).any().item())

    # Compare signal values across sleep-stage labels.
    class_signal_means = {}
    for sample_index in range(len(train_data)):
        sample = train_data[sample_index]
        sample_signal = sample["signal"]
        if isinstance(sample_signal, (tuple, list)):
            sample_signal = sample_signal[0]
        sample_label = int(torch.as_tensor(sample["label"]).item())
        signal_mean = torch.as_tensor(sample_signal).float().mean().item()
        class_signal_means.setdefault(sample_label, []).append(signal_mean)

    print("\\nAverage signal by label:")
    for label, values in sorted(class_signal_means.items()):
        print(
            f"  Label {label}: "
            f"mean={np.mean(values):.6f}, "
            f"std={np.std(values):.6f}, "
            f"samples={len(values)}"
        )

    trainer = Trainer(
        model=model,
        device=device,
        metrics=METRICS,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="cohen_kappa",
        optimizer_params={"lr": 1e-3},
    )

    train_results = trainer.evaluate(train_loader)
    print(f"\\n{duration}-second epoch training results:")
    for metric in METRICS:
        print(
            f"  {metric}: "
            f"{train_results.get(metric, float('nan')):.4f}"
        )

    results = trainer.evaluate(test_loader)
    print(f"\n{duration}-second epoch test results:")
    for metric in METRICS:
        print(f"  {metric}: {results.get(metric, float('nan')):.4f}")

    return results


def main() -> None:
    """Run the synthetic epoch-duration comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    with tempfile.TemporaryDirectory(prefix="dreamt_synthetic_") as tmp:
        root = Path(tmp)
        create_synthetic_dreamt(root, args.seed)

        dataset = DREAMTDataset(
            root=str(root),
            cache_dir=str(root / "cache"),
        )
        split = PatientSplit(
            ratios=(0.6, 0.2, 0.2),
            seed=args.seed,
        )

        print("Preparing 30-second epoch samples...")
        splits_30s = dataset.set_task(
            SleepStagingDREAMT(signal_type="IBI", chunk_duration=30),
            split=split,
        )

        print("\nPreparing 60-second epoch samples...")
        splits_60s = dataset.set_task(
            SleepStagingDREAMT(signal_type="IBI", chunk_duration=60),
            split=split,
        )

        check_matching_splits(splits_30s, splits_60s)

        print("\nSamples per split:")
        for name, part_30s, part_60s in zip(
            ("train", "validation", "test"), splits_30s, splits_60s
        ):
            print(f"  {name}: {len(part_30s)} (30s), " f"{len(part_60s)} (60s)")

        print("\nTraining the 30-second model...")
        results_30s = run_experiment(
            30,
            splits_30s,
            args.epochs,
            args.batch_size,
            args.seed,
            args.device,
        )

        print("\nTraining the 60-second model...")
        results_60s = run_experiment(
            60,
            splits_60s,
            args.epochs,
            args.batch_size,
            args.seed,
            args.device,
        )

        print("\nSynthetic ablation comparison")
        print(f"{'Metric':<20}{'30-second':>14}{'60-second':>14}")
        for metric in METRICS:
            value_30s = results_30s.get(metric, float("nan"))
            value_60s = results_60s.get(metric, float("nan"))
            print(f"{metric:<20}{value_30s:>14.4f}{value_60s:>14.4f}")


if __name__ == "__main__":
    main()

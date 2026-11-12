"""Weighting ablation for adaptive transfer on the DSA dataset.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Runs the full pipeline (DSADataset -> DSAActivityClassification
    -> SensorLSTM -> transfer learning) and compares four arms with the same
    classifier, subject split, preprocessing and target sensor:

    * ``no_transfer``: train on the target sensor only (baseline).
    * ``ipd``: sources pretrained with IPD-weighted learning rates, so more
      similar sources get more influence (the paper's method).
    * ``equal``: sources pretrained with equal weights.
    * ``inverse``: the weighting direction reversed (less similar sources
      get more influence).

    Comparing ``ipd`` with ``equal`` and ``inverse`` shows whether the gain
    comes from the IPD weighting itself or only from pretraining on sources.

Usage:
    Synthetic demo data (default, no download needed)::

        python examples/dsa_activity_classification_sensor_lstm.py

    Real DSA data (15 repetitions as in the paper; slow on a CPU)::

        python examples/dsa_activity_classification_sensor_lstm.py \\
            --real --root /path/to/dsa --seeds 15 --finetune-epochs 10

Notes:
    * The synthetic data only checks that the pipeline runs end to end; its
      numbers say nothing about the paper's claims.
    * Min-max ranges are computed over all segments (as a global rescaling
      to [-1, 1]), which includes the validation subjects.
    * Each repetition draws a new random subject split: 6 training subjects
      and 2 held-out subjects. ``split_by_patient`` returns train / val / test,
      so the held-out subjects are one "val" and one "test" subject and the
      reported RCC is the average of the two (equal to the pooled RCC when
      subjects have the same number of segments).
    * Every arm starts from the same random initialization.

Results:
    Paste the table printed by the script here after running it, together with
    the settings used (seeds, epochs, target sensor, data).
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DSADataset, get_dataloader, split_by_patient
from pyhealth.datasets.dsa import compute_channel_ranges
from pyhealth.models.sensor_lstm import SensorLSTM
from pyhealth.models.sensor_transfer import (
    compute_source_distances,
    evaluate_rcc,
    transfer_learn,
)
from pyhealth.tasks import DSAActivityClassification

ARMS = ("no_transfer", "ipd", "equal", "inverse")


def make_synthetic_dsa(
    root: str,
    n_subjects: int = 8,
    n_activities: int = 4,
    n_segments: int = 8,
    seed: int = 0,
) -> None:
    """Writes a small fake DSA folder in the real ``aXX/pY/sZZ.txt`` layout.

    Each activity has its own oscillation frequency, shared by the five
    sensors with different gains and noise levels, so sensors are related
    but not identical.

    Args:
        root: Directory to write into.
        n_subjects: Number of subjects.
        n_activities: Number of activities.
        n_segments: Segments per subject and activity.
        seed: Random seed.

    Examples:
        >>> make_synthetic_dsa("/tmp/fake_dsa")  # doctest: +SKIP
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 125)
    gains = np.array([1.0, 0.9, 0.8, 0.6, 0.5])
    noise = np.array([0.2, 0.3, 0.4, 0.8, 1.0])
    for act in range(1, n_activities + 1):
        for sub in range(1, n_subjects + 1):
            folder = os.path.join(root, f"a{act:02d}", f"p{sub}")
            os.makedirs(folder)
            for seg in range(1, n_segments + 1):
                columns = []
                for sensor in range(5):
                    for channel in range(9):
                        wave = np.sin(2 * np.pi * (act + 1) * t + 0.3 * channel)
                        columns.append(
                            gains[sensor] * wave
                            + noise[sensor] * rng.normal(size=125)
                        )
                data = np.stack(columns, axis=1)
                np.savetxt(
                    os.path.join(folder, f"s{seg:02d}.txt"), data, delimiter=","
                )


def summarize(results: dict[str, list[float]]) -> str:
    """Formats mean and standard deviation of RCC per arm as a table.

    Args:
        results: Maps each arm to its RCC values across repetitions.

    Returns:
        A Markdown table, with the change from ``no_transfer``.

    Examples:
        >>> print(summarize({"no_transfer": [0.5, 0.7], "ipd": [0.8, 0.9]}))
        ... # doctest: +SKIP
    """
    base = float(np.mean(results["no_transfer"]))
    lines = [
        "| arm | mean RCC | std | change vs no_transfer |",
        "|---|---|---|---|",
    ]
    for arm, values in results.items():
        mean, std = float(np.mean(values)), float(np.std(values))
        lines.append(f"| {arm} | {mean:.4f} | {std:.4f} | {mean - base:+.4f} |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, list[float]]:
    """Runs every arm for every repetition and returns the RCC values.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Maps each arm to its list of RCC values, one per repetition.

    Examples:
        >>> results = run(parse_args([]))  # doctest: +SKIP
    """
    tmp_dirs = []
    root = args.root
    if not args.real:
        root = tempfile.mkdtemp()
        tmp_dirs.append(root)
        make_synthetic_dsa(root)
    cache_dir = args.cache_dir or tempfile.mkdtemp()
    if args.cache_dir is None:
        tmp_dirs.append(cache_dir)

    try:
        dataset = DSADataset(root=root, cache_dir=cache_dir)
        metadata = pd.read_csv(os.path.join(root, "dsa-pyhealth.csv"))
        ranges = compute_channel_ranges(metadata["signal_file"])
        task = DSAActivityClassification(
            target_sensor=args.target_sensor, channel_ranges=ranges
        )
        samples = dataset.set_task(task)
        source_keys = [f"source_{name}" for name in task.source_sensors]

        results: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for seed in range(args.seeds):
            train_ds, val_ds, test_ds = split_by_patient(
                samples, [0.75, 0.125, 0.125], seed=seed
            )
            print(
                f"seed {seed}: {len(train_ds)} train / {len(val_ds)} val / "
                f"{len(test_ds)} test samples"
            )
            train_loader = get_dataloader(
                train_ds, batch_size=args.batch_size, shuffle=True
            )
            val_loader = get_dataloader(
                val_ds, batch_size=args.batch_size, shuffle=False
            )
            test_loader = get_dataloader(
                test_ds, batch_size=args.batch_size, shuffle=False
            )
            distances = compute_source_distances(
                train_loader, "signal", source_keys, args.max_pairs
            )
            print(f"seed {seed}: IPD {distances}")
            for arm in ARMS:
                torch.manual_seed(seed)
                model = SensorLSTM(samples, hidden_dim=args.hidden_dim)
                out = transfer_learn(
                    model,
                    train_loader,
                    val_loader,
                    "signal",
                    source_keys,
                    weighting=None if arm == "no_transfer" else arm,
                    distances=distances,
                    base_lr=args.lr,
                    epochs_per_source=args.source_epochs,
                    finetune_epochs=args.finetune_epochs,
                    finetune_lr=args.lr,
                )
                # After fine-tuning the model reads the target sensor, so it
                # can be scored on the second held-out subject directly.
                rcc = (out["rcc"] + evaluate_rcc(model, test_loader)) / 2
                results[arm].append(rcc)
                print(f"seed {seed} {arm}: RCC {rcc:.4f}")
        return results
    finally:
        for path in tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv``.

    Returns:
        The parsed arguments.

    Examples:
        >>> parse_args(["--seeds", "2"]).seeds
        2
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true", help="use real DSA data")
    parser.add_argument("--root", help="DSA folder (required with --real)")
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--target-sensor", default="right_arm")
    parser.add_argument("--seeds", type=int, default=3, help="repetitions")
    parser.add_argument("--source-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-pairs", type=int, default=32, help="pairs for IPD")
    parser.add_argument("--out", help="write the results table to this file")
    args = parser.parse_args(argv)
    if args.real and not args.root:
        parser.error("--root is required with --real")
    return args


def main() -> None:
    """Runs the ablation and prints (and optionally saves) the table."""
    args = parse_args()
    table = summarize(run(args))
    print("\n" + table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
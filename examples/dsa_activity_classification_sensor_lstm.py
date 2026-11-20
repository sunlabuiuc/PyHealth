"""Weighting ablation for adaptive transfer on the DSA dataset.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Runs the full pipeline (DSADataset -> DSAActivityClassification
    -> SensorLSTM -> transfer learning) and compares four arms with the same
    classifier, subject split, preprocessing and target sensor:

    * ``no_transfer``: train on the target sensor only (baseline).
    * ``no_transfer_matched``: target sensor only, but with as many epochs
      as the transfer arms use in total (source + target), so any gain of
      transfer is not just from training longer.
    * ``ipd``: sources pretrained as in the paper. Each source starts from
      the same learning rate, multiplied by ``1 - alpha`` after every epoch,
      where ``alpha`` is the source's share of the total IPD. Closer sources
      keep a larger learning rate for longer and the farthest source is
      trained first.
    * ``equal``: every source gets the same ``alpha``.
    * ``inverse``: the direction reversed (``alpha`` proportional to one over
      the IPD), so closer sources decay faster. The source order is unchanged.

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
    Real DSA data, target sensor right arm, 3 repetitions (seeds 0-2), each with
    a new random split of 6 training / 2 held-out subjects. Settings: SensorLSTM
    with hidden size 64, Adam with gradient clipping at 1.0, batch size 32,
    learning rate 3e-3 for sources and target, 8 epochs per source, 40 target
    epochs (72 for ``no_transfer_matched``), min-max scaling to [-1, 1]. RCC is
    the mean over the two held-out subjects::

        | arm | mean RCC | std | vs no_transfer | vs matched |
        |---|---|---|---|---|
        | no_transfer | 0.5183 | 0.0704 | +0.0000 | -0.0863 |
        | no_transfer_matched | 0.6045 | 0.1048 | +0.0863 | +0.0000 |
        | ipd | 0.5039 | 0.0142 | -0.0143 | -0.1006 |
        | equal | 0.5538 | 0.0539 | +0.0355 | -0.0507 |
        | inverse | 0.4515 | 0.0417 | -0.0668 | -0.1531 |

    Findings in this reduced setting (the paper trains far longer and repeats
    15 times):

    * Transfer did not beat the matched baseline: ``ipd`` was better than it in
      1 of 3 seeds, ``equal`` in 1 of 3 and ``inverse`` in 0 of 3.
    * ``ipd`` was not better than ``equal``; it was better than ``inverse`` on
      average (+0.052, 2 of 3 seeds). With 3 seeds and paired standard errors of
      0.03-0.09 none of these differences is reliable.
    * The IPD ordering was stable and sensible: the left arm and torso were
      closest to the right arm, the legs farthest. Pretraining gave a lower loss
      after the first target epoch, but the baselines caught up.
    * Accuracy is far below the paper's (about 0.85 for LSTM without transfer),
      most likely because of the much shorter training; the target loss was still
      falling at the end of training.

    Deviations from the paper: far fewer epochs and repetitions, one shared
    learning rate for sources and target, a fixed fine-tuning rate and number of
    epochs (the paper's validation-loss rule gives a negative rate when the loss
    is above one), Adam with gradient clipping, and a train / val / test split in
    which the two held-out subjects are one "val" and one "test" subject.
"""

import argparse
import json
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

ARMS = ("no_transfer", "no_transfer_matched", "ipd", "equal", "inverse")


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


def load_results(path: str) -> dict[str, dict[str, float]]:
    """Loads saved results: ``{arm: {seed: RCC}}`` (empty if no file).

    Args:
        path: Path of a results JSON file.

    Returns:
        The saved results.

    Examples:
        >>> load_results("does-not-exist.json")
        {}
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_result(path: str, arm: str, seed: int, rcc: float) -> None:
    """Adds one result to a results JSON file, so a long run loses nothing.

    Args:
        path: Path of the results JSON file.
        arm: The arm name.
        seed: The repetition (seed).
        rcc: The RCC of this arm and seed.

    Examples:
        >>> save_result("results.json", "ipd", 0, 0.5)  # doctest: +SKIP
    """
    data = load_results(path)
    data.setdefault(arm, {})[str(seed)] = rcc
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_merged(paths: list[str]) -> dict[str, dict[str, float]]:
    """Merges results files into ``{arm: {seed: RCC}}``.

    Args:
        paths: Results JSON files, e.g. from separate runs of other seeds.

    Returns:
        The merged results, in the order of ARMS.

    Examples:
        >>> load_merged(["run_a.json", "run_b.json"])  # doctest: +SKIP
    """
    merged: dict[str, dict[str, float]] = {}
    for path in paths:
        for arm, by_seed in load_results(path).items():
            merged.setdefault(arm, {}).update(by_seed)
    return {arm: merged[arm] for arm in ARMS if arm in merged}


def merge_results(paths: list[str]) -> dict[str, list[float]]:
    """Merges results files into RCC lists per arm, in the order of ARMS.

    Args:
        paths: Results JSON files.

    Returns:
        Maps each arm to its RCC values.

    Examples:
        >>> merge_results(["run_a.json", "run_b.json"])  # doctest: +SKIP
    """
    return {arm: list(v.values()) for arm, v in load_merged(paths).items()}


def paired_table(merged: dict[str, dict[str, float]], reference: str) -> str:
    """Compares every arm with a reference arm, seed by seed.

    All arms of a seed share the subject split and the initialization, so the
    per-seed difference removes much of the split-to-split noise.

    Args:
        merged: Results as ``{arm: {seed: RCC}}``.
        reference: The arm to compare against.

    Returns:
        A Markdown table with the mean difference, its standard error and the
        number of seeds where the arm beat the reference.

    Examples:
        >>> paired_table({"a": {"0": 0.5}, "b": {"0": 0.6}}, "a")  # doctest: +SKIP
    """
    ref = merged[reference]
    lines = [
        f"Paired difference (arm minus {reference}), per seed:",
        "",
        "| arm | seeds | mean diff | std error | arm better in |",
        "|---|---|---|---|---|",
    ]
    for arm, values in merged.items():
        if arm == reference:
            continue
        seeds = sorted(set(ref) & set(values))
        diffs = np.array([values[sd] - ref[sd] for sd in seeds])
        if len(diffs) == 0:
            continue
        error = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.nan
        lines.append(
            f"| {arm} | {len(diffs)} | {diffs.mean():+.4f} | {error:.4f} "
            f"| {int((diffs > 0).sum())} of {len(diffs)} |"
        )
    return "\n".join(lines)


def summarize(results: dict[str, list[float]]) -> str:
    """Formats mean and standard deviation of RCC per arm as a table.

    Args:
        results: Maps each arm to its RCC values across repetitions.

    Returns:
        A Markdown table. The change from ``no_transfer`` and from
        ``no_transfer_matched`` (the fairer baseline) is shown when those
        arms were run, and ``n/a`` otherwise.

    Examples:
        >>> print(summarize({"no_transfer": [0.5], "ipd": [0.6]}))
        ... # doctest: +SKIP
    """

    def mean_of(arm: str) -> float | None:
        return float(np.mean(results[arm])) if arm in results else None

    def delta(mean: float, base: float | None) -> str:
        return "n/a" if base is None else f"{mean - base:+.4f}"

    base, matched = mean_of("no_transfer"), mean_of("no_transfer_matched")
    lines = [
        "| arm | mean RCC | std | vs no_transfer | vs matched |",
        "|---|---|---|---|---|",
    ]
    for arm, values in results.items():
        mean, std = float(np.mean(values)), float(np.std(values))
        lines.append(
            f"| {arm} | {mean:.4f} | {std:.4f} | {delta(mean, base)} "
            f"| {delta(mean, matched)} |"
        )
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

        results: dict[str, list[float]] = {arm: [] for arm in args.arms}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            train_ds, val_ds, test_ds = split_by_patient(
                samples, [0.75, 0.125, 0.125], seed=seed
            )
            print(
                f"seed {seed}: {len(train_ds)} train / {len(val_ds)} val / "
                f"{len(test_ds)} test samples"
            )
            train_loader = get_dataloader(
                train_ds, batch_size=args.batch_size, shuffle=True, seed=seed
            )
            # Unshuffled, so the same pairs are used for the IPD every run.
            distance_loader = get_dataloader(
                train_ds, batch_size=args.batch_size, shuffle=False
            )
            val_loader = get_dataloader(
                val_ds, batch_size=args.batch_size, shuffle=False
            )
            test_loader = get_dataloader(
                test_ds, batch_size=args.batch_size, shuffle=False
            )
            distances = compute_source_distances(
                distance_loader, "signal", source_keys, args.max_pairs, seed=seed
            )
            print(f"seed {seed}: IPD {distances}")
            for arm in args.arms:
                target_epochs = args.finetune_epochs
                if arm == "no_transfer_matched":
                    target_epochs += len(source_keys) * args.source_epochs
                torch.manual_seed(seed)
                model = SensorLSTM(samples, hidden_dim=args.hidden_dim)
                out = transfer_learn(
                    model,
                    train_loader,
                    val_loader,
                    "signal",
                    source_keys,
                    weighting=None if arm.startswith("no_transfer") else arm,
                    distances=distances,
                    base_lr=args.source_lr,
                    epochs_per_source=args.source_epochs,
                    finetune_epochs=target_epochs,
                    finetune_lr=args.lr,
                )
                # After fine-tuning the model reads the target sensor, so it
                # can be scored on the second held-out subject directly.
                rcc = (out["rcc"] + evaluate_rcc(model, test_loader)) / 2
                results[arm].append(rcc)
                if args.results_json:
                    save_result(args.results_json, arm, seed, rcc)
                print(f"seed {seed} {arm}: RCC {rcc:.4f}")
                if out["alphas"]:
                    alphas = {k: round(v, 3) for k, v in out["alphas"].items()}
                    print(f"seed {seed} {arm}: alpha {alphas}")
                losses = [round(x, 3) for x in out["finetune_losses"]]
                print(f"seed {seed} {arm}: target-epoch losses {losses}")
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
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="which arms to run (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=3, help="repetitions")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument(
        "--results-json", help="save each RCC here as soon as it is computed"
    )
    parser.add_argument(
        "--summarize",
        nargs="+",
        metavar="JSON",
        help="only print the table of saved results files (no training)",
    )
    parser.add_argument("--source-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--source-lr", type=float, default=5e-4, help="initial lr of each source"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="target lr")
    parser.add_argument("--max-pairs", type=int, default=32, help="pairs for IPD")
    parser.add_argument("--out", help="write the results table to this file")
    args = parser.parse_args(argv)
    if args.real and not args.root:
        parser.error("--root is required with --real")
    return args


def main() -> None:
    """Runs the ablation (or summarizes saved results) and prints the table."""
    args = parse_args()
    if args.summarize:
        merged = load_merged(args.summarize)
        results = {arm: list(v.values()) for arm, v in merged.items()}
    else:
        results = run(args)
        merged = None
    table = summarize(results)
    if merged is not None:
        for reference in ("no_transfer_matched", "equal"):
            if reference in merged:
                table += "\n\n" + paired_table(merged, reference)
    print("\n" + table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()

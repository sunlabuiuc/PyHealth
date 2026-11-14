"""Train EEGNet on PhysioNet EEG Motor Movement/Imagery (EEGBCI) data.

Author: Axel Coiffard-Donaty
Paper: V. J. Lawhern et al. "EEGNet: a compact convolutional neural network
    for EEG-based brain-computer interfaces". Journal of Neural Engineering,
    15(5):056013, 2018. https://doi.org/10.1088/1741-2552/aace8c
Description: Cross-subject motor imagery classification (rest, imagined left
    fist, imagined right fist) with EEGNet-8,2 on the imagery runs 4, 8 and 12
    of EEGBCI. Signals are resampled to 128 Hz, the rate the paper's kernel
    sizes were designed for, and cut into 2 s windows. The split is by subject,
    so test subjects are never seen during training.

    This is a usage example, not a reproduction of the paper: the paper
    evaluates EEGNet on other datasets (for example BCI Competition IV 2a).

Data: with ``--download``, MNE fetches only the requested EDF files from
    https://physionet.org/content/eegmmidb/1.0.0/ (open access, Open Data
    Commons Attribution License v1.0), about 2.6 MB per run, so about 80 MB for
    the default 10 subjects.

Usage:
    python examples/eeg/eegnet/eegnet_eegbci_motor_imagery.py \\
        --root /path/to/eegbci --download
"""

import argparse

import numpy as np
import torch

from pyhealth.datasets import EEGBCIDataset, get_dataloader, split_by_patient
from pyhealth.models import EEGNet
from pyhealth.tasks import EEGMotorImageryEEGBCI
from pyhealth.trainer import Trainer

SEED = 42
IMAGERY_RUNS = [4, 8, 12]  # imagined opening/closing of the left or right fist
SAMPLING_RATE = 128  # EEGNet's kernel sizes were designed for 128 Hz
WINDOW_SECONDS = 2.0
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / validation / test, by subject
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="EEGBCI data folder.")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="PhysioNet subject numbers (1-109). Default: 1 to 10.",
    )
    parser.add_argument("--download", action="store_true", help="Fetch EDFs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    """Build the dataset, train EEGNet and report test metrics."""
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = EEGBCIDataset(
        root=args.root,
        subjects=args.subjects,
        runs=IMAGERY_RUNS,
        download=args.download,
    )
    task = EEGMotorImageryEEGBCI(
        window_size=WINDOW_SECONDS,
        resample_rate=SAMPLING_RATE,
        channel_mode="all",
        compute_stft=False,
    )
    samples = dataset.set_task(task)
    train_ds, val_ds, test_ds = split_by_patient(samples, SPLIT_RATIOS, seed=SEED)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError(
            f"{len(args.subjects)} subjects are too few for a train/val/test "
            f"split by subject with ratios {SPLIT_RATIOS}; pass at least 5."
        )
    print(
        f"windows: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}"
    )

    model = EEGNet(dataset=samples, sampling_rate=SAMPLING_RATE)
    trainer = Trainer(model=model, metrics=METRICS, device=args.device)
    trainer.train(
        train_dataloader=get_dataloader(train_ds, args.batch_size, shuffle=True),
        val_dataloader=get_dataloader(val_ds, args.batch_size, shuffle=False),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="balanced_accuracy",
    )
    scores = trainer.evaluate(get_dataloader(test_ds, args.batch_size, shuffle=False))
    print("test:", {name: round(float(value), 4) for name, value in scores.items()})


if __name__ == "__main__":
    main()

"""Alzheimer's disease against controls on the AHEPA EEG dataset with SparcNet.

Author: Axel Coiffard-Donaty

Loads the dataset (OpenNeuro ds004504) with ``AHEPADementiaDataset``, cuts
the artifact-cleaned recordings into 30 second windows with 15 seconds of
overlap with ``DementiaClassificationAHEPA``, splits by subject and trains
``SparcNet``.

The split is by subject: every window of a subject carries that subject's
diagnosis, so a split by window puts the same person in training and test
data and inflates the scores. The scores printed here are per window; the
studies on this dataset usually also report leave-one-subject-out results,
which this short example does not do.

The recordings are resampled from 500 to 125 Hz to keep the windows small.
The artifact-cleaned recordings were band-pass filtered to 0.5-45 Hz, below
half of that rate.

Data: open under CC0. Download https://openneuro.org/datasets/ds004504
(5.8 GB, or about 3 GB for the ``derivatives`` folder and
``participants.tsv`` alone), then run::

    python examples/eeg/ahepa_dementia/ahepa_dementia_classification.py \
        --root /path/to/ds004504

Other problems: ``--problem ftd_vs_cn``, ``ad_vs_ftd``, ``dementia_vs_cn``.
"""

import argparse

import numpy as np
import torch

from pyhealth.datasets import AHEPADementiaDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import DementiaClassificationAHEPA
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / validation / test, by subject
RESAMPLE_RATE = 125  # Hz; the cleaned recordings are band-passed to 0.5-45 Hz
BINARY_PROBLEMS = ("ad_vs_cn", "ftd_vs_cn", "ad_vs_ftd", "dementia_vs_cn")
METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "f1"]


def main() -> None:
    """Train SparcNet on a binary dementia problem and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", required=True, help="BIDS root, the folder with participants.tsv."
    )
    parser.add_argument("--problem", default="ad_vs_cn", choices=BINARY_PROBLEMS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = AHEPADementiaDataset(root=args.root)
    dataset.stats()

    task = DementiaClassificationAHEPA(
        problem=args.problem, resample_rate=RESAMPLE_RATE
    )
    samples = dataset.set_task(task)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, SPLIT_RATIOS, seed=SEED
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise ValueError(
            "Too few recordings for a train/validation/test split by subject "
            f"with ratios {SPLIT_RATIOS}: one of the three parts is empty."
        )
    print(
        "samples train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = SparcNet(dataset=samples)
    trainer = Trainer(model=model, device=args.device, metrics=METRICS)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="roc_auc",
    )
    print(trainer.evaluate(test_loader))


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()

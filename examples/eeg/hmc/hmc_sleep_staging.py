"""Sleep staging on the HMC sleep staging database with SparcNet.

Author: Axel Coiffard-Donaty

Loads the Haaglanden Medisch Centrum (HMC) recordings with ``HMCDataset``,
cuts each recording into 30 second epochs on the four EEG derivations with
``SleepStagingHMC`` (between lights off and lights on), splits by subject
and trains ``SparcNet``. Cohen's kappa and macro F1 are reported next to
accuracy, as is usual for sleep staging, because the stages are unbalanced
(N2 is the most frequent and N1 the hardest).

Data: open under CC BY 4.0. Download it from
https://physionet.org/content/hmc-sleep-staging/1.1/ (15.7 GB), for example
with ``wget -r -N -c -np https://physionet.org/files/hmc-sleep-staging/1.1/``,
then run::

    python examples/eeg/hmc/hmc_sleep_staging.py --root /path/to/hmc/recordings

Each recording is one subject, so a split by patient is a split by subject.
The first run writes ``hmc-pyhealth.csv`` next to the recordings, or in the
PyHealth cache when that folder is read only.
"""

import argparse

import numpy as np
import torch

from pyhealth.datasets import HMCDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import SleepStagingHMC
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # train / validation / test, by subject
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]


def main() -> None:
    """Train SparcNet on HMC sleep staging and print the test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        required=True,
        help="HMC recordings folder (SNxxx.edf and SNxxx_sleepscoring.txt).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = HMCDataset(root=args.root)
    dataset.stats()

    samples = dataset.set_task(SleepStagingHMC())
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
        monitor="cohen_kappa",
    )
    print(trainer.evaluate(test_loader))


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()

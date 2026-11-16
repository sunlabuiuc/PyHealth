"""Neonatal seizure detection on the Helsinki neonatal EEG dataset with SparcNet.

Author: Axel Coiffard-Donaty

Loads the recordings with ``HelsinkiNeonatalDataset``, builds 16 second
windows of the 18-channel bipolar montage with ``SeizureDetectionHelsinki``
(32 Hz, band-pass filtered, labels by consensus of the three experts), splits by
neonate and trains ``SparcNet``.

Seizure windows are a minority (about 10% of the consensus windows), so the
area under the precision-recall curve is monitored and reported next to the
ROC AUC. The scores are computed over all test windows pooled together;
the papers on this dataset usually also report the AUC per neonate, which
this short example does not do.

Data: open under CC BY 4.0. Download the 79 ``eeg<N>.edf`` files, the three
``annotations_2017_*.csv`` files and ``clinical_information.csv`` from
https://zenodo.org/records/4940267 (4.3 GB) into one folder, then run::

    python examples/eeg/helsinki_neonatal/helsinki_seizure_detection.py \\
        --root /path/to/helsinki

Use ``--label-mode A`` (or ``B``, ``C``) to train on one expert,
``majority`` or ``any`` for other ways of combining the three experts.
"""

import argparse

import numpy as np
import torch

from pyhealth.datasets import (
    HelsinkiNeonatalDataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import SparcNet
from pyhealth.tasks import SeizureDetectionHelsinki
from pyhealth.tasks.seizure_detection_helsinki import LABEL_MODES
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # train / validation / test, by neonate
METRICS = ["pr_auc", "roc_auc", "balanced_accuracy", "f1"]


def main() -> None:
    """Train SparcNet on neonatal seizure detection and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        required=True,
        help="Folder with the eeg<N>.edf files and the annotation CSV files.",
    )
    parser.add_argument("--label-mode", default="consensus", choices=LABEL_MODES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = HelsinkiNeonatalDataset(root=args.root)
    dataset.stats()

    samples = dataset.set_task(SeizureDetectionHelsinki(label_mode=args.label_mode))
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, SPLIT_RATIOS, seed=SEED
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise ValueError(
            "Too few recordings for a train/validation/test split by neonate "
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
        monitor="pr_auc",
    )
    print(trainer.evaluate(test_loader))


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()
